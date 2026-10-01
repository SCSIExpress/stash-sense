"""TDD coverage for the "async" fix group (findings 8 and 10):

Finding 10 - face_clusters_router.py endpoints were `async def` but ran sync
sqlite/clustering/httpx work directly, blocking the shared event loop (which
also serves the fingerprint job's /identify/scene self-calls). Fixed by making
every endpoint a plain `def` (FastAPI then dispatches it to its threadpool),
and by running jobs/cluster_faces_job.py's build via asyncio.to_thread.

Finding 8 - face_cluster_service.merge_clusters could merge into a
nonexistent target (swallowed FK error, sources silently deleted) or merge
clusters assigned to conflicting performers. The fix (owned by the service
fix-group, running concurrently with this one) is expected to raise
ClusterNotFound / InvalidClusterOperation, which this router now maps to
404 / 409. A concurrent build guard (BuildInProgress via
face_cluster_service._BUILD_LOCK) is expected from the same group.

Some tests below (marked "pending service group" in the return summary)
depend on symbols that live in face_cluster_service.py, which a different,
concurrently-running fix group owns. They are written against the documented
target interface and are left in place un-skipped; they'll go green once
that group lands.
"""
from __future__ import annotations

import asyncio
import inspect
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import face_clusters_router
from face_clusters_router import router as face_groups_router
from face_cluster_service import FaceClusterService
from recommendations_db import RecommendationsDB

import jobs.cluster_faces_job as cluster_faces_job_module
from jobs.cluster_faces_job import ClusterLibraryFacesJob


# ==================== helpers ====================


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(face_groups_router)

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    return app


def _emb_bytes(seed: int, dim: int = 512) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, dim).astype(np.float32).tobytes()


def _add_face(db: RecommendationsDB, scene_id: int, frame: int = 0) -> int:
    fid = db.add_library_face(
        stash_scene_id=scene_id,
        frame_index=frame,
        timestamp_sec=1.0,
        bbox={"x": 10, "y": 10, "w": 40, "h": 40},
        det_confidence=0.9,
        yaw=0.0,
        facenet_emb=_emb_bytes(scene_id * 100 + frame),
        arcface_emb=_emb_bytes(999_000 + scene_id * 100 + frame),
        best_match_id=None,
        best_match_name=None,
        best_match_confidence=None,
    )
    assert fid is not None
    return fid


class FakeService:
    """Stand-in for FaceClusterService for router-only tests that don't need
    real clustering/db logic — just controllable call recording."""

    def __init__(self):
        self.build_calls: list[dict] = []
        self.update_calls: list[tuple] = []
        self.build_fn = None

    def build_clusters(self, **kwargs):
        self.build_calls.append(kwargs)
        if self.build_fn is not None:
            return self.build_fn(**kwargs)
        return {"clusters_created": 0, "faces_assigned": 0, "faces_total": 0}

    def update_cluster(self, cluster_id, name=None, status=None):
        self.update_calls.append((cluster_id, name, status))
        allowed = {None, "open", "matched", "assigned", "ignored", "banned"}
        if status not in allowed:
            raise face_clusters_router.InvalidClusterOperation(
                f"invalid status: {status!r}"
            )
        return True

    def stats(self):
        return {}


@pytest.fixture(autouse=True)
def _reset_router_service_global(monkeypatch):
    # Every test either sets face_clusters_router._service explicitly, or
    # relies on get_face_cluster_service() lazily building one; make sure
    # tests never see a leftover instance from a previous test.
    monkeypatch.setattr(face_clusters_router, "_service", None)
    yield
    monkeypatch.setattr(face_clusters_router, "_service", None)


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "test.db")


# ==================== Finding 10: endpoints are sync ====================


def test_all_face_group_endpoints_are_sync():
    routes = [r for r in face_groups_router.routes if isinstance(r, APIRoute)]
    assert routes, "expected at least one route on the face-groups router"
    still_async = [r.path for r in routes if inspect.iscoroutinefunction(r.endpoint)]
    assert still_async == [], f"these routes are still async def: {still_async}"


@pytest.mark.asyncio
async def test_build_does_not_block_event_loop(monkeypatch):
    fake = FakeService()

    def slow_build(**kwargs):
        time.sleep(0.6)
        return {"clusters_created": 0, "faces_assigned": 0, "faces_total": 0}

    fake.build_fn = slow_build
    monkeypatch.setattr(face_clusters_router, "_service", fake)

    app = _make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def do_build():
            t0 = time.monotonic()
            r = await client.post("/face-groups/build", json={})
            return time.monotonic() - t0, r

        async def do_ping():
            t0 = time.monotonic()
            r = await client.get("/ping")
            return time.monotonic() - t0, r

        (build_dt, build_resp), (ping_dt, ping_resp) = await asyncio.gather(
            do_build(), do_ping()
        )

    assert build_resp.status_code == 200
    assert ping_resp.status_code == 200
    assert ping_dt < 0.3, f"/ping took {ping_dt:.3f}s while build ran; event loop was blocked"
    assert build_dt >= 0.6
    assert len(fake.build_calls) == 1


# ==================== GET merged-cluster redirect ====================


def test_get_merged_cluster_returns_404_with_merged_into(monkeypatch, db):
    source_id = db.create_face_cluster(status="open")
    target_id = db.create_face_cluster(status="open")
    db.record_cluster_merge([source_id], target_id)
    db.delete_face_cluster(source_id)  # a real merge deletes the source cluster row

    monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: db)
    monkeypatch.setattr(face_clusters_router, "_service", FaceClusterService(db))

    client = TestClient(_make_app())
    resp = client.get(f"/face-groups/{source_id}")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert str(source_id) in detail
    assert str(target_id) in detail
    assert "merged into" in detail


def test_get_missing_cluster_with_no_merge_returns_plain_404(monkeypatch, db):
    monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: db)
    monkeypatch.setattr(face_clusters_router, "_service", FaceClusterService(db))

    client = TestClient(_make_app())
    resp = client.get("/face-groups/999999")
    assert resp.status_code == 404
    assert "merged into" not in resp.json()["detail"]


# ==================== PATCH / eject validation ====================


def test_patch_invalid_status_returns_400(monkeypatch):
    fake = FakeService()
    monkeypatch.setattr(face_clusters_router, "_service", fake)
    monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: None)

    client = TestClient(_make_app())
    resp = client.patch("/face-groups/1", json={"status": "not-a-real-status"})
    assert resp.status_code == 400
    assert fake.update_calls == [(1, None, "not-a-real-status")]


def test_eject_invalid_mode_returns_400():
    client = TestClient(_make_app())
    resp = client.post(
        "/face-groups/1/eject",
        json={"face_ids": [1, 2], "eject_mode": "bogus"},
    )
    assert resp.status_code == 400


# ==================== Finding 8: merge safety (real service + tmp db) ====================


def test_merge_missing_target_returns_404_and_sources_intact(monkeypatch, db):
    source_id = db.create_face_cluster(status="open")
    fid = _add_face(db, scene_id=1)
    db.add_faces_to_cluster(source_id, [fid])
    missing_target = source_id + 12345

    monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: db)
    monkeypatch.setattr(face_clusters_router, "_service", FaceClusterService(db))

    client = TestClient(_make_app())
    resp = client.post(
        "/face-groups/merge",
        json={"source_ids": [source_id], "target_id": missing_target},
    )
    assert resp.status_code == 404

    # Source cluster and its membership must be untouched.
    assert db.get_face_cluster(source_id) is not None
    assert db.get_cluster_face_count(source_id) == 1


def test_merge_conflicting_performers_returns_409(monkeypatch, db):
    target_id = db.create_face_cluster(
        status="assigned", performer_id="local-1", performer_name="Alice"
    )
    source_id = db.create_face_cluster(
        status="assigned", performer_id="local-2", performer_name="Bob"
    )
    fid_t = _add_face(db, scene_id=1)
    fid_s = _add_face(db, scene_id=2)
    db.add_faces_to_cluster(target_id, [fid_t])
    db.add_faces_to_cluster(source_id, [fid_s])

    monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: db)
    monkeypatch.setattr(face_clusters_router, "_service", FaceClusterService(db))

    client = TestClient(_make_app())
    resp = client.post(
        "/face-groups/merge",
        json={"source_ids": [source_id], "target_id": target_id},
    )
    assert resp.status_code == 409, (
        "pending service group: merging two clusters assigned to different "
        "performers must raise InvalidClusterOperation -> 409"
    )


def test_build_in_progress_returns_409(monkeypatch, db):
    import face_cluster_service

    monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: db)
    monkeypatch.setattr(face_clusters_router, "_service", FaceClusterService(db))

    client = TestClient(_make_app())

    lock = face_cluster_service._BUILD_LOCK  # pending service group: no such lock yet
    acquired = lock.acquire(blocking=False)
    assert acquired, "test setup: lock should have been free"
    try:
        resp = client.post("/face-groups/build", json={})
    finally:
        lock.release()

    assert resp.status_code == 409, (
        "pending service group: a build already in progress must raise "
        "BuildInProgress -> 409"
    )


# ==================== Finding 10: job runs off the event loop ====================


class _StubJobContext:
    def __init__(self):
        self.progress: list[tuple] = []

    def is_stop_requested(self) -> bool:
        return False

    async def report_progress(self, items_processed, items_total=None):
        self.progress.append((items_processed, items_total))


class _ThreadRecordingFakeService:
    def __init__(self, db=None):
        self.db = db
        self.call_thread_ident = None
        self.calls: list[dict] = []

    def build_clusters(self, **kwargs):
        self.call_thread_ident = threading.get_ident()
        self.calls.append(kwargs)
        return {"faces_total": 0}


class _BuildInProgressFakeService:
    def __init__(self, db=None):
        self.db = db

    def build_clusters(self, **kwargs):
        raise cluster_faces_job_module.BuildInProgress("build already running")


@pytest.mark.asyncio
async def test_job_runs_build_off_event_loop_thread(monkeypatch):
    main_thread_ident = threading.get_ident()
    fake = _ThreadRecordingFakeService()

    monkeypatch.setattr(cluster_faces_job_module, "FaceClusterService", lambda db: fake)
    monkeypatch.setattr(cluster_faces_job_module, "get_rec_db", lambda: object())

    class _FakeDB:
        def get_library_face_count(self):
            return 0

    monkeypatch.setattr(cluster_faces_job_module, "get_rec_db", lambda: _FakeDB())

    job = ClusterLibraryFacesJob()
    ctx = _StubJobContext()
    result = await job.run(ctx)

    assert result is None
    assert fake.call_thread_ident is not None
    assert fake.call_thread_ident != main_thread_ident, (
        "build_clusters ran on the event loop thread instead of a worker thread"
    )


@pytest.mark.asyncio
async def test_job_handles_build_in_progress(monkeypatch):
    fake = _BuildInProgressFakeService()

    monkeypatch.setattr(cluster_faces_job_module, "FaceClusterService", lambda db: fake)

    class _FakeDB:
        def get_library_face_count(self):
            return 0

    monkeypatch.setattr(cluster_faces_job_module, "get_rec_db", lambda: _FakeDB())

    job = ClusterLibraryFacesJob()
    ctx = _StubJobContext()

    # A skipped run is reported (the queue records it as the job's error)
    # instead of being marked completed as if the build had run.
    with pytest.raises(RuntimeError, match="skipped"):
        await job.run(ctx)
