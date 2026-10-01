"""Repair round 1: regression tests for the skeptic findings on the face-cluster fixes.

Each test fails on the pre-round code and passes with the fix. Grouped by the
original review finding they extend.
"""
from __future__ import annotations

import inspect
import shutil
import sqlite3
import subprocess
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import face_clusters_router
import identification_router
import library_face_persist
import recommendations_db as rdb
import scene_matcher
from face_cluster_service import (
    ClusterNotFound,
    FaceClusterService,
    InvalidClusterOperation,
)
from identification_router import _match_to_response, distance_to_confidence
from recommendations_db import RecommendationsDB
from tests.test_fc_fix_service import (
    FakeStash,
    Person,
    _add_face,
    _members,
    _memberships,
    _max_memberships,
    _live_clusters_with_stash_id,
)

SCENE = 7070


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "r2.db")


@pytest.fixture
def svc(db):
    return FaceClusterService(db)


# ======================================================================
# Findings 1-3: per-face anchoring (hybrid claim, one face per person per
# frame, face-level agreement, per-face confidence)
# ======================================================================

_rng = np.random.default_rng(77)
_BASES: dict[str, np.ndarray] = {}


def _base(person: str) -> np.ndarray:
    if person not in _BASES:
        v = _rng.normal(size=512)
        _BASES[person] = v / np.linalg.norm(v)
    return _BASES[person]


def _emb(person: str, jitter: float = 0.01):
    v = _base(person) + _rng.normal(scale=jitter, size=512)
    v = (v / np.linalg.norm(v)).astype(np.float32)
    return NS(facenet=v, arcface=v)


def _m(sid: str, score: float):
    return NS(stashdb_id=sid, name=f"name-{sid}", combined_score=score,
              facenet_distance=score, arcface_distance=score, country=None,
              image_url=None, universal_id=None, endpoint=None)


def _res(person: str, matches: list):
    return NS(embedding=_emb(person), matches=matches, face=None)


def _det(x: int, y: int = 10):
    return NS(bbox={"x": x, "y": y, "w": 60, "h": 60}, confidence=0.99, yaw=0.0, image=None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    from library_face_store import LibraryFaceStore
    s = LibraryFaceStore(tmp_path)

    def fake_save_crop(scene_id, frame_index, bbox, frame_image, timestamp_sec=None):
        rel = f"{scene_id}/{frame_index}_{bbox['x']}_{bbox['y']}_{timestamp_sec}.jpg"
        p = s.base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpg")
        return rel

    monkeypatch.setattr(s, "save_crop", fake_save_crop)
    return s


def _identify_and_persist(db, store, all_results, persons):
    """Mimic identify_scene: detected faces parallel to all_results, then persist."""
    detected = [(frame, _det(10 + 100 * i)) for i, (frame, _r) in enumerate(all_results)]
    frames = [NS(frame_index=f, timestamp_sec=float(f), image=np.zeros((480, 640, 3), np.uint8))
              for f in sorted({f for f, _ in detected})]
    library_face_persist.persist_from_identify(
        scene_id=SCENE,
        extraction_frames=frames,
        detected_faces=detected,
        embeddings=[r.embedding for _f, r in all_results],
        persons=persons,
        face_matches=[r.matches for _f, r in all_results],
        db_version="v1",
        db=db,
        store=store,
    )
    with db._connection() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM library_faces WHERE stash_scene_id = ? ORDER BY bbox_x", (SCENE,))]
    return {i: rows[i] for i in range(len(rows))}


def test_hybrid_claim_requires_face_top_match():
    # C's own top match is sC (not a final person: single appearance); its
    # second match sA must not hand it to sA.
    all_results = [
        (0, _res("A", [_m("sA", 0.2)])),
        (1, _res("A", [_m("sA", 0.22)])),
        (2, _res("A", [_m("sA", 0.21)])),
        (3, _res("C", [_m("sC", 0.15), _m("sA", 0.48)])),
    ]
    persons = scene_matcher.hybrid_matching(
        all_results, recognizer=None, top_k=5, max_distance=0.6,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    ids = {p.best_match.stashdb_id: scene_matcher.get_face_indices(p) for p in persons}
    assert ids["sA"] == [0, 1, 2]


def test_hybrid_stranger_in_same_frame_not_claimed():
    # frame 0 holds A and a stranger X whose top match is sZ (sA only second)
    all_results = [
        (0, _res("A", [_m("sA", 0.2)])),
        (0, _res("X", [_m("sZ", 0.35), _m("sA", 0.45)])),
        (1, _res("A", [_m("sA", 0.21)])),
        (2, _res("A", [_m("sA", 0.22)])),
    ]
    persons = scene_matcher.hybrid_matching(
        all_results, recognizer=None, top_k=5, max_distance=0.6,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    sa = next(p for p in persons if p.best_match.stashdb_id == "sA")
    assert scene_matcher.get_face_indices(sa) == [0, 2, 3]


def test_claim_by_match_gives_person_at_most_one_face_per_frame():
    # Two unclustered faces in one frame both have top match sA: only the
    # closer one may be claimed.
    all_results = [
        (5, NS(matches=[_m("sA", 0.40)])),
        (5, NS(matches=[_m("sA", 0.30)])),
    ]
    persons = [NS(best_match=NS(stashdb_id="sA"))]
    claims = scene_matcher._claim_faces_by_match(all_results, persons, 0.6, claimed=set())
    assert claims == {1: 0}


def test_hybrid_costar_with_loose_second_match_not_anchored(db, store):
    # Skeptic repro: co-star M's faces have distinct top matches, sA only second.
    all_results = [(f, _res("A", [_m("sA", 0.25)])) for f in range(4)] + [
        (0, _res("M", [_m("sM1", 0.50), _m("sA", 0.57)])),
        (1, _res("M", [_m("sM2", 0.50), _m("sA", 0.58)])),
        (2, _res("M", [_m("sM3", 0.50), _m("sA", 0.59)])),
    ]
    persons = scene_matcher.hybrid_matching(
        all_results, recognizer=None, top_k=5, max_distance=0.6,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    rows = _identify_and_persist(db, store, all_results, persons)
    assert [rows[i]["best_match_id"] for i in range(4)] == ["sA"] * 4
    assert [rows[i]["best_match_id"] for i in range(4, 7)] == [None, None, None]


@pytest.mark.parametrize("mode", ["cluster", "hybrid"])
def test_two_faces_same_frame_same_person_only_closest_anchored(db, store, mode):
    # frame 0 holds A and a different person Y whose top match is also sA.
    all_results = [
        (0, _res("A", [_m("sA", 0.2)])),
        (0, _res("Y", [_m("sA", 0.4)])),
        (1, _res("A", [_m("sA", 0.21)])),
        (2, _res("A", [_m("sA", 0.22)])),
    ]
    if mode == "cluster":
        persons = scene_matcher.cluster_mode_matching(
            all_results, recognizer=None, top_k=3,
            _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    else:
        persons = scene_matcher.hybrid_matching(
            all_results, recognizer=None, top_k=5, max_distance=0.6,
            _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    rows = _identify_and_persist(db, store, all_results, persons)
    assert rows[0]["best_match_id"] == "sA"
    assert rows[1]["best_match_id"] is None
    assert rows[2]["best_match_id"] == "sA" and rows[3]["best_match_id"] == "sA"


def test_face_confidence_is_the_faces_own_score(db, store):
    all_results = [
        (0, _res("A", [_m("sA", 0.10)])),
        (1, _res("A", [_m("sA", 0.30)])),
    ]
    persons = scene_matcher.cluster_mode_matching(
        all_results, recognizer=None, top_k=3,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    rows = _identify_and_persist(db, store, all_results, persons)
    assert rows[0]["best_match_confidence"] == pytest.approx(0.90)
    assert rows[1]["best_match_confidence"] == pytest.approx(0.70)


def test_reranked_person_not_backed_by_face_top_match_leaves_face_unanchored():
    # rerank changed the person's best match to sB; the face itself says sA
    persons = [NS(best_match=NS(stashdb_id="sB", name="B", confidence=0.7), _face_indices=[0])]
    assign = library_face_persist.face_person_assignment(
        persons, n_faces=1, face_frames=[0], face_matches=[[_m("sA", 0.2), _m("sB", 0.3)]])
    assert assign == {}


def test_identify_scene_passes_face_matches_to_persist():
    src = inspect.getsource(identification_router.identify_scene)
    call = src[src.index("_persist_library_faces_off_loop("):]
    call = call[: call.index(")\n") + 1]
    assert "face_matches=" in call


# ======================================================================
# Schema v13: poisoned anchors, UNIQUE key with timestamp, synced flag
# ======================================================================

def _emb_b(seed: int) -> bytes:
    return np.random.default_rng(seed).normal(0, 1, 512).astype(np.float32).tobytes()


def _downgrade_to_v12(path):
    """Turn a fresh (v13) DB into the v12 layout (inline UNIQUE without timestamp)."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = OFF")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(library_faces)")]
    assert conn.execute("SELECT COUNT(*) FROM library_faces").fetchone()[0] == 0
    conn.executescript("""
        DROP TABLE library_faces;
        CREATE TABLE library_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stash_scene_id INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,
            timestamp_sec REAL,
            bbox_x REAL NOT NULL, bbox_y REAL NOT NULL, bbox_w REAL NOT NULL, bbox_h REAL NOT NULL,
            det_confidence REAL NOT NULL, yaw REAL,
            facenet_emb BLOB NOT NULL, arcface_emb BLOB NOT NULL,
            crop_path TEXT, best_match_id TEXT, best_match_name TEXT, best_match_confidence REAL,
            db_version TEXT, created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(stash_scene_id, frame_index, bbox_x, bbox_y)
        );
        CREATE INDEX IF NOT EXISTS idx_lib_faces_scene ON library_faces(stash_scene_id);
        CREATE INDEX IF NOT EXISTS idx_lib_faces_match ON library_faces(best_match_id);
        ALTER TABLE face_clusters DROP COLUMN stash_ids_synced;
        UPDATE schema_version SET version = 12;
    """)
    assert "id" in cols
    conn.commit()
    conn.close()


def _raw_face(conn, scene, frame=0, ts=1.0, x=10, match=None):
    return conn.execute(
        """INSERT INTO library_faces (stash_scene_id, frame_index, timestamp_sec, bbox_x, bbox_y,
           bbox_w, bbox_h, det_confidence, facenet_emb, arcface_emb,
           best_match_id, best_match_name, best_match_confidence)
           VALUES (?, ?, ?, ?, 10, 40, 40, 0.9, ?, ?, ?, ?, ?)""",
        (scene, frame, ts, x, _emb_b(scene * 10 + frame), _emb_b(scene * 10 + frame + 5),
         match, match and f"n-{match}", 0.8 if match else None)).lastrowid


def test_fresh_db_is_v13_with_timestamped_unique_key(db):
    assert rdb.SCHEMA_VERSION == 14
    with db._connection() as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == rdb.SCHEMA_VERSION
        assert "stash_ids_synced" in {r[1] for r in conn.execute("PRAGMA table_info(face_clusters)")}
    a = db.add_library_face(1, 3, 22.5, {"x": 10, "y": 10, "w": 40, "h": 40}, 0.9, 0.0,
                            _emb_b(1), _emb_b(2))
    b = db.add_library_face(1, 3, 30.0, {"x": 10, "y": 10, "w": 40, "h": 40}, 0.9, 0.0,
                            _emb_b(3), _emb_b(4))
    dup = db.add_library_face(1, 3, 30.0, {"x": 10, "y": 10, "w": 40, "h": 40}, 0.9, 0.0,
                              _emb_b(5), _emb_b(6))
    assert a and b and a != b
    assert dup is None


def test_migrate_v12_to_v13(tmp_path):
    path = tmp_path / "m12.db"
    RecommendationsDB(path)
    _downgrade_to_v12(path)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    f = {i: _raw_face(conn, i, match="uuid-bad") for i in range(1, 8)}
    unpinned_m = conn.execute(
        "INSERT INTO face_clusters (status, name) VALUES ('matched', 'Bad')").lastrowid
    conn.execute("INSERT INTO face_cluster_stash_ids VALUES (?, 'uuid-bad')", (unpinned_m,))
    for k in (1, 2):
        conn.execute("INSERT INTO face_cluster_members VALUES (?, ?)", (unpinned_m, f[k]))
    pinned_m = conn.execute(
        "INSERT INTO face_clusters (status, pinned) VALUES ('matched', 1)").lastrowid
    conn.execute("INSERT INTO face_cluster_stash_ids VALUES (?, 'uuid-keep')", (pinned_m,))
    conn.execute("INSERT INTO face_cluster_members VALUES (?, ?)", (pinned_m, f[3]))
    assigned = conn.execute(
        "INSERT INTO face_clusters (status, performer_id, pinned) VALUES ('assigned', '42', 1)"
    ).lastrowid
    conn.execute("INSERT INTO face_cluster_members VALUES (?, ?)", (assigned, f[4]))
    banned = conn.execute(
        "INSERT INTO face_clusters (status, pinned) VALUES ('banned', 1)").lastrowid
    conn.execute("INSERT INTO face_cluster_members VALUES (?, ?)", (banned, f[5]))
    conn.execute("INSERT INTO face_rejections (face_id, kind, ref) VALUES (?, 'cluster', '99')", (f[6],))
    # a deleted face leaves the AUTOINCREMENT sequence above max(id)
    gone = _raw_face(conn, 50)
    conn.execute("DELETE FROM library_faces WHERE id = ?", (gone,))
    conn.commit()
    conn.close()

    db = RecommendationsDB(path)
    with db._connection() as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == rdb.SCHEMA_VERSION
        # anchors written by pre-fix code are cleared
        assert conn.execute("SELECT COUNT(best_match_id) FROM library_faces").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM library_faces").fetchone()[0] == 7
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    # unpinned matched group (seeded from those anchors) dissolved, faces pooled
    assert db.get_face_cluster(unpinned_m) is None
    assert _memberships(db, f[1]) == [] and _memberships(db, f[2]) == []
    # curated state kept (the table rebuild must not cascade)
    assert _memberships(db, f[3]) == [pinned_m]
    assert db.get_cluster_stash_ids(pinned_m) == ["uuid-keep"]
    assert _memberships(db, f[4]) == [assigned]
    assert _memberships(db, f[5]) == [banned]
    assert db.get_face_rejections([f[6]])[f[6]]["clusters"] == {99}
    # face ids are never reused
    new_id = _add_face(db, scene_id=60)
    assert new_id > gone
    # same frame/x/y at another timestamp now fits
    assert db.add_library_face(1, 0, 99.0, {"x": 10, "y": 10, "w": 40, "h": 40}, 0.9, 0.0,
                               _emb_b(1), _emb_b(2)) is not None


# ======================================================================
# Finding 4: assign keeps the StashDB anchor; sync adds performer ids later;
# member-derived anchor for groups with none
# ======================================================================

class TestAnchorSurvivesAssign:
    @pytest.mark.parametrize("perf_ids", [[], ["fansdb-uuid"]])
    @pytest.mark.parametrize("incremental", [True, False])
    def test_assign_to_unlinked_performer_keeps_anchor(self, db, svc, perf_ids, incremental):
        p = Person(31000)
        f1 = p.face(db, 1, match_id="uuid-x", match_name="Jane")
        f2 = p.face(db, 2, match_id="uuid-x", match_name="Jane")
        cid = db.create_face_cluster(status="matched", stash_ids=["uuid-x"], name="Jane")
        db.add_faces_to_cluster(cid, [f1, f2])
        stash = FakeStash(performer_stash_ids={"42": perf_ids})

        svc.assign_performer(cid, "42", "Jane", stash)
        assert "uuid-x" in db.get_cluster_stash_ids(cid)
        assert set(perf_ids) <= set(db.get_cluster_stash_ids(cid))

        f3 = p.face(db, 3, match_id="uuid-x", match_name="Jane")
        svc.build_clusters(incremental=incremental, stash_client=stash)

        assert f3 in _members(db, cid)
        assert db.list_face_clusters("matched") == []

    def test_lookup_failure_at_assign_is_synced_by_a_later_build(self, db, svc):
        f1 = _add_face(db, scene_id=1, match_id="uuid-x")
        cid = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        db.add_faces_to_cluster(cid, [f1])
        svc.assign_performer(cid, "42", "Jane", FakeStash(fail_lookup=True))
        assert db.get_cluster_stash_ids(cid) == ["uuid-x"]

        stash = FakeStash(performer_stash_ids={"42": ["uuid-real"]})
        svc.build_clusters(incremental=True, stash_client=stash)
        # the performer's StashDB link replaces the match anchor (round 3, #4)
        assert db.get_cluster_stash_ids(cid) == ["uuid-real"]
        # synced once; later builds do not ask again
        svc.build_clusters(incremental=True, stash_client=stash)
        assert stash.lookups == ["42"]

    def test_legacy_assigned_group_gets_anchor_from_members(self, db, svc):
        # migrated assigned group: no stash ids, performer not linked in Stash
        p = Person(32000)
        members = [p.face(db, i, match_id="uuid-x", match_name="Jane") for i in (1, 2, 3)]
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane",
                                     pinned=True)
        db.add_faces_to_cluster(cid, members)
        newcomer = _add_face(db, scene_id=9, match_id="uuid-x", match_name="Jane")

        for stash in (None, FakeStash(performer_stash_ids={"42": []})):
            svc.build_clusters(incremental=True, stash_client=stash)

        assert newcomer in _members(db, cid)
        assert _live_clusters_with_stash_id(db, "uuid-x") == [cid]

    def test_member_anchor_needs_a_majority(self, db, svc):
        p = Person(33000)
        members = [p.face(db, 1, match_id="uuid-x"), p.face(db, 2, match_id="uuid-y"),
                   p.face(db, 3, match_id="uuid-z")]
        cid = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        db.add_faces_to_cluster(cid, members)
        svc.build_clusters(incremental=True)
        assert db.get_cluster_stash_ids(cid) == []


# ======================================================================
# Findings 5 / 8 / 10: builds tolerate concurrent mutations; incremental
# builds reconcile duplicates; ban clears ignored membership
# ======================================================================

class TestBuildRobustness:
    @pytest.mark.parametrize("incremental", [True, False])
    def test_face_deleted_mid_build_does_not_abort(self, db, svc, monkeypatch, incremental):
        a, b = Person(41000), Person(42000)
        a_faces = [a.face(db, 1 + i, match_id="uuid-a") for i in range(3)]
        b_faces = [b.face(db, 10 + i) for i in range(4)]
        victim_matched, victim_open = a_faces[0], b_faces[0]

        orig = db.get_face_rejections

        def reidentify_mid_build(face_ids=None):
            # a concurrent re-identify drops two pooled faces after the pool was read
            with db._connection() as conn:
                conn.execute("DELETE FROM library_faces WHERE id IN (?, ?)",
                             (victim_matched, victim_open))
            monkeypatch.setattr(db, "get_face_rejections", orig)
            return orig(face_ids)

        monkeypatch.setattr(db, "get_face_rejections", reidentify_mid_build)
        svc.build_clusters(incremental=incremental)

        matched = db.list_face_clusters("matched")
        assert len(matched) == 1 and _members(db, matched[0]["id"]) == set(a_faces[1:])
        opens = db.list_face_clusters("open")
        assert len(opens) == 1 and _members(db, opens[0]["id"]) == set(b_faces[1:])
        assert all(c["face_count"] > 0 for c in db.list_face_clusters())

    def test_group_deleted_mid_build_does_not_abort(self, db, svc, monkeypatch):
        p = Person(43000)
        target = db.create_face_cluster()
        db.add_faces_to_cluster(target, [p.face(db, i) for i in range(3)])
        newcomers = [p.face(db, 10 + i) for i in range(3)]

        orig = db.get_cluster_centroid

        def centroid_then_user_deletes(cid):
            cent = orig(cid)
            if cid == target:
                db.delete_face_cluster(cid)
            return cent

        monkeypatch.setattr(db, "get_cluster_centroid", centroid_then_user_deletes)
        svc.build_clusters(incremental=True)  # must not raise

        assert db.get_face_cluster(target) is None
        # the newcomers were not lost: still pooled (or regrouped), never in a dead group
        assert all(_memberships(db, f) in ([],) or len(_memberships(db, f)) == 1 for f in newcomers)
        assert all(c["face_count"] > 0 for c in db.list_face_clusters())

    def test_group_merged_mid_build_places_into_merge_target(self, db, svc, monkeypatch):
        p = Person(44000)
        target = db.create_face_cluster()
        db.add_faces_to_cluster(target, [p.face(db, i) for i in range(3)])
        other = db.create_face_cluster(pinned=True)
        db.add_faces_to_cluster(other, [_add_face(db, scene_id=90)])
        newcomer = p.face(db, 20)

        orig = db.get_cluster_centroid

        def centroid_then_user_merges(cid):
            cent = orig(cid)
            if cid == target:
                db.move_faces_to_cluster([target], other)
            return cent

        monkeypatch.setattr(db, "get_cluster_centroid", centroid_then_user_merges)
        svc.build_clusters(incremental=True)

        assert _memberships(db, newcomer) == [other]

    def test_incremental_build_folds_matched_group_into_assigned_group(self, db, svc):
        # Repro B: matched M(x) exists; another group is assigned to the performer with x
        p, q = Person(45000), Person(46000)
        m_faces = [p.face(db, 1 + i, match_id="uuid-x") for i in range(3)]
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        db.add_faces_to_cluster(m, m_faces)
        o_faces = [q.face(db, 10 + i) for i in range(3)]
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, o_faces)
        stash = FakeStash(performer_stash_ids={"42": ["uuid-x"]})
        svc.assign_performer(o, "42", "Jane", stash)

        svc.build_clusters(incremental=True, stash_client=stash)

        assert db.get_face_cluster(m) is None
        assert _members(db, o) == set(m_faces) | set(o_faces)
        assert _live_clusters_with_stash_id(db, "uuid-x") == [o]
        assert db.resolve_cluster_id(m) == o

    def test_backfill_later_folds_matched_duplicate(self, db, svc):
        # Repro A: Stash unreachable -> a matched group for x appears; once the
        # assigned group learns x, the next incremental build folds it in.
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        db.add_faces_to_cluster(a, [_add_face(db, scene_id=1)])
        fx = _add_face(db, scene_id=2, match_id="uuid-x")
        svc.build_clusters(incremental=True, stash_client=FakeStash(fail_lookup=True))
        assert len(db.list_face_clusters("matched")) == 1

        svc.build_clusters(incremental=True,
                           stash_client=FakeStash(performer_stash_ids={"42": ["uuid-x"]}))
        assert db.list_face_clusters("matched") == []
        assert fx in _members(db, a)

    def test_fold_skips_faces_rejected_from_target(self, db, svc):
        p = Person(47000)
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True,
                                   stash_ids=["uuid-x"])
        ejected = p.face(db, 1, match_id="uuid-x")
        db.add_faces_to_cluster(a, [ejected, p.face(db, 2)])
        svc.eject_faces(a, [ejected])
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        db.add_faces_to_cluster(m, [ejected])  # stale duplicate group holding it

        svc.build_clusters(incremental=True)

        assert ejected not in _members(db, a)

    def test_ban_face_in_ignored_group_leaves_one_membership(self, db, svc):
        fid = _add_face(db, scene_id=1)
        ign = db.create_face_cluster(status="ignored", pinned=True)
        db.add_faces_to_cluster(ign, [fid])
        svc.ban_faces([fid])
        ms = db.get_face_cluster_membership(fid)
        assert [m["status"] for m in ms] == ["banned"]

    def test_full_build_replaces_face_whose_match_changed(self, db, svc):
        p = Person(48000)
        faces = [p.face(db, 1 + i, match_id="uuid-x") for i in range(2)]
        svc.build_clusters()
        [mx] = db.list_face_clusters("matched")
        # re-identify re-anchors one face to y
        with db._connection() as conn:
            conn.execute("UPDATE library_faces SET best_match_id = 'uuid-y' WHERE id = ?", (faces[1],))
        svc.build_clusters()
        assert _memberships(db, faces[0]) == [mx["id"]]
        [my] = _live_clusters_with_stash_id(db, "uuid-y")
        assert _memberships(db, faces[1]) == [my]

    def test_build_can_be_stopped(self, db, svc):
        p = Person(49000)
        for i in range(5):
            p.face(db, i)
        res = svc.build_clusters(incremental=True, should_stop=lambda: True)
        assert res["cancelled"] is True
        assert db.list_face_clusters() == []


# ======================================================================
# Finding 6: rejection memory holes (ban mode, split source, empty anchor)
# ======================================================================

def _make_app():
    app = FastAPI()
    app.include_router(face_clusters_router.router)
    return app


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: db)
    monkeypatch.setattr(face_clusters_router, "_service", FaceClusterService(db))
    monkeypatch.setattr(face_clusters_router, "_optional_stash_client", lambda: None)
    return TestClient(_make_app())


class TestRejectionHoles:
    @pytest.mark.parametrize("status", ["open", "matched"])
    def test_ban_then_unban_does_not_rejoin(self, db, client, status):
        p = Person(51000)
        match = "uuid-x" if status == "matched" else None
        faces = [p.face(db, 1 + i, match_id=match) for i in range(4)]
        cid = db.create_face_cluster(status=status, stash_ids=[match] if match else None)
        db.add_faces_to_cluster(cid, faces)

        r = client.post(f"/face-groups/{cid}/eject", json={"face_ids": [faces[0]], "eject_mode": "ban"})
        assert r.status_code == 200 and r.json()["banned"] == 1
        assert client.post("/face-groups/unban", json={"face_ids": [faces[0]]}).json()["unbanned"] == 1
        assert client.post("/face-groups/build", json={"incremental": True}).status_code == 200

        assert faces[0] not in _members(db, cid)

    def test_split_then_eject_then_full_build_keeps_face_apart(self, db, svc):
        p = Person(52000)
        faces = [p.face(db, 1 + i) for i in range(5)]
        svc.build_clusters()
        [o] = db.list_face_clusters("open")
        n = svc.split_cluster(o["id"], [faces[0]])
        assert db.get_face_cluster(o["id"])["pinned"] == 1
        svc.eject_faces(n, [faces[0]])
        svc.build_clusters()

        home = _memberships(db, faces[1])
        assert home and _memberships(db, faces[0]) != home

    def test_eject_from_assigned_without_stash_ids_does_not_reseed(self, db, svc):
        p = Person(53000)
        faces = [p.face(db, 1 + i, match_id="uuid-x", match_name="Jane") for i in range(3)]
        cid = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        db.add_faces_to_cluster(cid, faces)
        svc.eject_faces(cid, [faces[0]])
        svc.build_clusters(incremental=True)
        assert db.list_face_clusters("matched") == []
        assert faces[0] not in _members(db, cid)

    def test_eject_only_anchored_member_from_assigned_goes_to_its_own_identity(self, db, svc):
        # the group has no stash ids and its other members carry no match: the
        # ejected face's own match is evidence against this group, not an
        # identity to reject it from (round 3, #6), so it may seed its own group
        p = Person(54000)
        fx = p.face(db, 1, match_id="uuid-x", match_name="Jane")
        cid = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        db.add_faces_to_cluster(cid, [fx, p.face(db, 2), p.face(db, 3)])
        svc.eject_faces(cid, [fx])
        assert db.get_face_rejections([fx])[fx]["stash_ids"] == set()
        svc.build_clusters(incremental=True)
        assert fx not in _members(db, cid)
        assert [_memberships(db, fx)] == [_live_clusters_with_stash_id(db, "uuid-x")]

    def test_eject_mode_validated_once(self):
        src = inspect.getsource(face_clusters_router.eject_faces)
        assert src.count("req.eject_mode not in") == 1


# ======================================================================
# Finding 7: upsert (sampling change, resolution change, race, conflicts)
# ======================================================================

def _fd(frame, ts, x=10, y=10, w=100, h=100, seed=None, match=None, emb=None):
    s = seed if seed is not None else frame * 1000 + int(x)
    fn, af = emb if emb is not None else (_emb_b(s), _emb_b(s + 7))
    return {"frame_index": frame, "timestamp_sec": ts, "bbox": {"x": x, "y": y, "w": w, "h": h},
            "det_confidence": 0.95, "yaw": 0.0, "facenet_emb": fn, "arcface_emb": af,
            "crop_path": None, "best_match_id": match, "best_match_name": None,
            "best_match_confidence": None, "db_version": None}


class TestUpsert:
    def test_sampling_change_never_drops_new_faces(self, db, svc):
        first = db.upsert_scene_library_faces(1, [_fd(i, 10.0 * i) for i in range(6)])
        old_ids = first["face_ids"]
        svc.ban_faces(old_ids)

        res = db.upsert_scene_library_faces(1, [_fd(i, 7.5 * i, seed=500 + i) for i in range(8)])

        assert None not in res["face_ids"]
        assert res["conflicts"] == 0
        assert res["inserted"] + res["updated"] == 8
        assert len(svc.get_banned_faces()) == 6

    def test_crop_key_includes_timestamp(self, tmp_path):
        from library_face_store import LibraryFaceStore
        s = LibraryFaceStore(tmp_path)
        img = np.zeros((200, 200, 3), np.uint8)
        bbox = {"x": 10, "y": 10, "w": 50, "h": 50}
        assert s.save_crop(1, 3, bbox, img, timestamp_sec=22.5) != s.save_crop(1, 3, bbox, img, timestamp_sec=30.0)

    def test_resolution_change_keeps_curated_ids(self, db, svc):
        embs = [(_emb_b(900 + i), _emb_b(950 + i)) for i in range(3)]
        first = db.upsert_scene_library_faces(1, [_fd(i, 10.0 * i, x=100, y=50, w=80, h=80, emb=embs[i])
                                                  for i in range(3)])
        ids = first["face_ids"]
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        db.add_faces_to_cluster(a, ids)
        # same timestamps, file replaced at 2x resolution: boxes scale, embeddings ~equal
        res = db.upsert_scene_library_faces(1, [_fd(i, 10.0 * i, x=200, y=100, w=160, h=160, emb=embs[i])
                                                for i in range(3)])
        assert res["face_ids"] == ids
        assert res["retained"] == 0 and res["inserted"] == 0
        assert _members(db, a) == set(ids)

    def test_concurrent_ban_during_upsert_is_not_cascaded_away(self, db, monkeypatch):
        r = db.upsert_scene_library_faces(1, [_fd(0, 0.0, x=10), _fd(0, 0.0, x=300)])
        keep, vanish = r["face_ids"]
        outcome = {}
        orig_iou = rdb._bbox_iou

        def iou_with_concurrent_ban(a, b):
            if "done" not in outcome:
                outcome["done"] = True
                other = sqlite3.connect(db.db_path, timeout=0.2)
                other.execute("PRAGMA foreign_keys = ON")
                try:
                    cid = other.execute(
                        "INSERT INTO face_clusters (status, pinned) VALUES ('banned', 1)").lastrowid
                    other.execute("INSERT INTO face_cluster_members VALUES (?, ?)", (cid, vanish))
                    other.commit()
                    outcome["ban_committed"] = True
                except sqlite3.OperationalError:
                    other.rollback()
                    outcome["ban_committed"] = False
                finally:
                    other.close()
            return orig_iou(a, b)

        monkeypatch.setattr(rdb, "_bbox_iou", iou_with_concurrent_ban)
        db.upsert_scene_library_faces(1, [_fd(0, 0.0, x=10)])

        if outcome["ban_committed"]:
            assert db.get_library_face(vanish) is not None
        else:
            assert db.get_library_face(vanish) is None  # the ban simply had to wait

    def test_update_conflict_still_refreshes_anchor(self, db):
        # v14 key includes the box size: only a duplicate detection collides
        r = db.upsert_scene_library_faces(2, [
            _fd(0, 9.0, x=10, y=10, w=100, h=100, seed=1),   # keep
            _fd(0, 9.0, x=12, y=10, w=100, h=100, seed=2),   # mover
        ])
        keep, mover = r["face_ids"]
        db.add_face_rejections([keep], cluster_id=1)
        dup1 = _fd(0, 9.0, x=10, y=10, w=100, h=100, seed=1)
        dup2 = _fd(0, 9.0, x=10, y=10, w=100, h=100, seed=2, match="uuid-new")   # mover's person
        res = db.upsert_scene_library_faces(2, [dup1, dup2])
        assert res["conflicts"] == 1
        assert res["face_ids"] == [keep, mover]
        assert db.get_library_face(mover)["best_match_id"] == "uuid-new"


# ======================================================================
# Finding 8 / 9: merge edge cases
# ======================================================================

class TestMergeEdges:
    def test_move_integrity_error_maps_to_cluster_not_found(self, db, svc, monkeypatch):
        s = db.create_face_cluster()
        t = db.create_face_cluster()

        def boom(sources, target):
            raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")

        monkeypatch.setattr(db, "move_faces_to_cluster", boom)
        with pytest.raises(ClusterNotFound):
            svc.merge_clusters([s], t)

    def test_merge_into_assigned_tags_moved_scenes(self, db, svc):
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        db.add_faces_to_cluster(a, [_add_face(db, scene_id=1)])
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, [_add_face(db, scene_id=5), _add_face(db, scene_id=6)])
        stash = FakeStash()

        res = svc.merge_clusters([o], a, stash_client=stash)

        assert res["scenes_tagged"] == 2
        assert sorted(s for s, _ in stash.writes) == ["5", "6"]
        assert all("42" in ids for _s, ids in stash.writes)

    def test_router_merge_passes_stash_client(self, db, monkeypatch):
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, [_add_face(db, scene_id=5)])
        stash = FakeStash()
        monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: db)
        monkeypatch.setattr(face_clusters_router, "_service", FaceClusterService(db))
        monkeypatch.setattr(face_clusters_router, "_optional_stash_client", lambda: stash)
        r = TestClient(_make_app()).post("/face-groups/merge", json={"source_ids": [o], "target_id": a})
        assert r.status_code == 200
        assert stash.writes == [("5", ["42"])]

    @pytest.mark.parametrize("target_ignored", [True, False])
    def test_merge_mixing_ignored_and_live_rejected(self, db, svc, target_ignored):
        ign = db.create_face_cluster(status="ignored", pinned=True)
        live = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        with pytest.raises(InvalidClusterOperation):
            if target_ignored:
                svc.merge_clusters([live], ign)
            else:
                svc.merge_clusters([ign], live)

    def test_ignored_group_keeps_absorbing_its_identity(self, db, svc):
        p = Person(61000)
        faces = [p.face(db, 1 + i, match_id="uuid-x") for i in range(2)]
        svc.build_clusters(incremental=True)
        [m] = db.list_face_clusters("matched")
        svc.ignore_cluster(m["id"])
        newcomer = _add_face(db, scene_id=30, match_id="uuid-x")

        svc.build_clusters(incremental=True)

        assert db.list_face_clusters("matched") == []
        assert newcomer in _members(db, m["id"])
        assert set(faces) <= _members(db, m["id"])

    def test_merge_matched_into_larger_open_promotes_target(self, db, svc):
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-x"], name="Jane",
                                   performer_name="Jane")
        db.add_faces_to_cluster(m, [_add_face(db, scene_id=1)])
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, [_add_face(db, scene_id=2 + i) for i in range(3)])

        svc.merge_clusters([m], o)

        t = db.get_face_cluster(o)
        assert t["status"] == "matched"
        assert t["stash_ids"] == ["uuid-x"]
        assert t["name"] == "Jane"


# ======================================================================
# Finding 9: plugin merge request / selection
# ======================================================================

PLUGIN_JS = Path(__file__).resolve().parents[2] / "plugin" / "stash-sense-face-groups.js"

_HARNESS = """
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const window = { StashSense: {
  getRoute: () => ({ type: 'other' }), onNavigate() {}, onLeavePlugin() {},
  PLUGIN_NAME: 'Stash Sense', escapeHtml: s => s,
  getSettings: async () => ({}), runPluginOperation: async () => ({}) } };
const sandbox = { window, console, setTimeout, clearTimeout, clearInterval };
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'stash-sense-face-groups.js' });
const FG = window.StashSenseFaceGroups;
const cases = JSON.parse(process.argv[3]);
const out = cases.map(([fn, args]) => {
  if (fn === 'buildMergeRequest') {
    const [selected, list] = args;
    return FG.buildMergeRequest(selected, new Map(list.map(c => [c.id, c])));
  }
  return FG[fn](...args);
});
console.log(JSON.stringify(out));
"""


def _node(tmp_path, cases):
    if shutil.which("node") is None:
        pytest.skip("node is not available")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    proc = subprocess.run(["node", str(harness), str(PLUGIN_JS), json.dumps(cases)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _c(id_, status="open", n=3, pid=None):
    return {"id": id_, "status": status, "face_count": n, "performer_id": pid,
            "performer_name": pid, "name": None}


class TestPluginMerge:
    def test_ignored_mixed_with_live_is_refused(self, tmp_path):
        [r1, r2] = _node(tmp_path, [
            ["chooseMergeTarget", [[_c(1, "open", 3), _c(2, "ignored", 30)]]],
            ["chooseMergeTarget", [[_c(1, "ignored", 3), _c(2, "ignored", 30)]]],
        ])
        assert r1["targetId"] is None and "gnored" in r1["error"]
        assert r2["targetId"] == 2 and r2["error"] is None

    def test_build_merge_request(self, tmp_path):
        clusters = [_c(1, "open", 3), _c(2, "assigned", 1, "p1"), _c(3, "open", 9)]
        [ok, stale, one] = _node(tmp_path, [
            ["buildMergeRequest", [[1, 2, 3], clusters]],
            ["buildMergeRequest", [[1, 7], clusters]],
            ["buildMergeRequest", [[1], clusters]],
        ])
        assert ok == {"targetId": 2, "sourceIds": [1, 3], "error": None, "target": clusters[1]}
        assert stale["targetId"] is None and "stale" in stale["error"]
        assert one["targetId"] is None

    def test_filter_change_clears_selection(self):
        src = PLUGIN_JS.read_text(encoding="utf-8")
        block = src[src.index("data-fg-filter]').forEach"):]
        block = block[: block.index("});\n    });") + 1]
        assert "selectedClusters.clear()" in block

    def test_old_plugin_test_does_not_write_into_tests_dir(self):
        src = (Path(__file__).parent / "test_fc_fix_plugin.py").read_text(encoding="utf-8")
        assert "Path(__file__).resolve().parent / \"_fc_fix_plugin_harness.js\"" not in src


# ======================================================================
# Finding 10: job reports skips; job forwards stop requests
# ======================================================================

@pytest.mark.asyncio
async def test_job_build_in_progress_is_reported(monkeypatch):
    import jobs.cluster_faces_job as mod
    from face_cluster_service import BuildInProgress

    class Svc:
        def __init__(self, db):
            pass

        def build_clusters(self, **kw):
            raise BuildInProgress("busy")

    class DB:
        def get_library_face_count(self):
            return 0

    class Ctx:
        def is_stop_requested(self):
            return False

        async def report_progress(self, *a, **k):
            pass

    monkeypatch.setattr(mod, "FaceClusterService", Svc)
    monkeypatch.setattr(mod, "get_rec_db", lambda: DB())
    with pytest.raises(RuntimeError, match="skipped"):
        await mod.ClusterLibraryFacesJob().run(Ctx())


@pytest.mark.asyncio
async def test_job_forwards_stop_requests(monkeypatch):
    import jobs.cluster_faces_job as mod
    seen = {}

    class Svc:
        def __init__(self, db):
            pass

        def build_clusters(self, **kw):
            seen["should_stop"] = kw.get("should_stop")
            return {"faces_total": 0}

    class DB:
        def get_library_face_count(self):
            return 0

    class Ctx:
        stop = False

        def is_stop_requested(self):
            return self.stop

        async def report_progress(self, *a, **k):
            pass

    ctx = Ctx()
    monkeypatch.setattr(mod, "FaceClusterService", Svc)
    monkeypatch.setattr(mod, "get_rec_db", lambda: DB())
    await mod.ClusterLibraryFacesJob().run(ctx)
    assert seen["should_stop"]() is False
    ctx.stop = True
    assert seen["should_stop"]() is True
