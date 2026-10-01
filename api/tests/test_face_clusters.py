"""Tests for face-cluster DB layer and clustering."""
import sqlite3

import numpy as np
import pytest

from library_clustering import cluster_library_faces
from recommendations_db import RecommendationsDB


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "test.db")


def _emb_bytes(seed: int, dim: int = 512) -> bytes:
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, dim).astype(np.float32)
    return v.tobytes()


def _similar_emb(base: bytes, jitter: float = 0.01, seed: int = 0) -> bytes:
    v = np.frombuffer(base, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    v = v + rng.normal(0, jitter, v.shape).astype(np.float32)
    return v.astype(np.float32).tobytes()


def _add_face(db, scene_id=1, frame=0, match_id=None, match_name=None, facenet=None, arcface=None):
    return db.add_library_face(
        stash_scene_id=scene_id,
        frame_index=frame,
        timestamp_sec=1.0,
        bbox={"x": 10, "y": 10, "w": 40, "h": 40},
        det_confidence=0.9,
        yaw=0.0,
        facenet_emb=facenet or _emb_bytes(scene_id * 100 + frame),
        arcface_emb=arcface or _emb_bytes(999_000 + scene_id * 100 + frame),
        best_match_id=match_id,
        best_match_name=match_name,
        best_match_confidence=0.8 if match_id else None,
    )


class TestLibraryFaces:
    def test_add_and_get(self, db):
        fid = _add_face(db, scene_id=5, frame=2, match_id="abc", match_name="Jane")
        assert fid is not None
        face = db.get_library_face(fid)
        assert face["stash_scene_id"] == 5
        assert face["best_match_id"] == "abc"
        assert len(face["facenet_emb"]) == 512 * 4

    def test_duplicate_rejected(self, db):
        f1 = _add_face(db, scene_id=5, frame=2)
        f2 = _add_face(db, scene_id=5, frame=2)
        assert f1 is not None
        assert f2 is None

    def test_delete_scene_cascades(self, db):
        fid = _add_face(db, scene_id=5)
        assert db.get_library_face(fid) is not None
        db.delete_library_faces_for_scene(5)
        assert db.get_library_face(fid) is None

    def test_iter_batches(self, db):
        for i in range(10):
            _add_face(db, scene_id=i)
        batches = list(db.iter_library_faces(batch_size=3))
        total = sum(len(b) for b in batches)
        assert total == 10
        assert all(len(b) <= 3 for b in batches)


class TestClusters:
    def test_create_list_update(self, db):
        cid = db.create_face_cluster(name="Group A")
        assert db.get_face_cluster(cid)["status"] == "open"
        assert db.update_face_cluster(cid, status="assigned", performer_id="p1", performer_name="Jane")
        row = db.list_face_clusters("assigned")[0]
        assert row["performer_id"] == "p1"
        assert row["performer_name"] == "Jane"

    def test_membership(self, db):
        f1 = _add_face(db, scene_id=1)
        f2 = _add_face(db, scene_id=2)
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, [f1, f2])
        assert db.get_cluster_face_count(cid) == 2
        assert sorted(db.get_cluster_scene_ids(cid)) == [1, 2]
        # idempotent
        db.add_faces_to_cluster(cid, [f1])
        assert db.get_cluster_face_count(cid) == 2

    def test_unassigned_excludes_assigned(self, db):
        f1 = _add_face(db, scene_id=1)
        f2 = _add_face(db, scene_id=2)
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, [f1])
        unassigned = db.get_unassigned_face_ids()
        assert f1 not in unassigned
        assert f2 in unassigned

    def test_ignored_cluster_faces_not_unassigned(self, db):
        """Faces in ignored clusters stay put: they are never re-clustered."""
        f1 = _add_face(db, scene_id=1)
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, [f1])
        db.update_face_cluster(cid, status="ignored")
        assert f1 not in db.get_unassigned_face_ids()

    def test_top_matches(self, db):
        f1 = _add_face(db, scene_id=1, match_id="p1", match_name="Jane")
        f2 = _add_face(db, scene_id=2, match_id="p1", match_name="Jane")
        f3 = _add_face(db, scene_id=3, match_id="p2", match_name="Bob")
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, [f1, f2, f3])
        tops = db.get_cluster_top_matches(cid)
        assert tops[0]["performer_id"] == "p1"
        assert tops[0]["face_count"] == 2

    def test_merge_clusters(self, db):
        f1 = _add_face(db, scene_id=1)
        f2 = _add_face(db, scene_id=2)
        c1 = db.create_face_cluster()
        c2 = db.create_face_cluster()
        db.add_faces_to_cluster(c1, [f1])
        db.add_faces_to_cluster(c2, [f2])
        assert db.delete_face_cluster(c1) is not None
        db.add_faces_to_cluster(c2, [f1])
        assert db.get_cluster_face_count(c2) == 2
        assert db.get_face_cluster(c1) is None

    def test_scene_count_in_listing(self, db):
        f1 = _add_face(db, scene_id=1)
        f2 = _add_face(db, scene_id=2)
        f3 = _add_face(db, scene_id=2, frame=1)
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, [f1, f2, f3])
        rows = db.list_face_clusters()
        assert rows[0]["face_count"] == 3
        assert rows[0]["scene_count"] == 2


class TestClustering:
    def test_distinct_people_separate(self):
        rng = np.random.default_rng(42)
        centers = [rng.normal(0, 1, 1024).astype(np.float32) for _ in range(3)]
        faces = []
        for c in centers:
            for _ in range(8):
                v = (c + rng.normal(0, 0.05, 1024)).astype(np.float32)
                v /= np.linalg.norm(v)
                faces.append({
                    "id": len(faces),
                    "facenet_emb": v[:512].tobytes(),
                    "arcface_emb": v[512:].tobytes(),
                })
        for _ in range(2):
            v = rng.normal(0, 1, 1024).astype(np.float32)
            v /= np.linalg.norm(v)
            faces.append({"id": len(faces), "facenet_emb": v[:512].tobytes(), "arcface_emb": v[512:].tobytes()})
        clusters = cluster_library_faces(faces, distance_threshold=0.5)
        sizes = sorted((len(c) for c in clusters), reverse=True)
        assert sizes == [8, 8, 8, 1, 1]

    def test_identical_faces_cluster(self):
        v = np.random.default_rng(1).normal(0, 1, 1024).astype(np.float32)
        v /= np.linalg.norm(v)
        faces = [
            {"id": i, "facenet_emb": v[:512].tobytes(), "arcface_emb": v[512:].tobytes()}
            for i in range(5)
        ]
        clusters = cluster_library_faces(faces, distance_threshold=0.5)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_empty(self):
        assert cluster_library_faces([], 0.5) == []


class TestBuildClustersService:
    def test_seed_by_match(self, db):
        from face_cluster_service import FaceClusterService
        svc = FaceClusterService(db)
        _add_face(db, scene_id=1, match_id="p1", match_name="Jane")
        _add_face(db, scene_id=2, match_id="p1", match_name="Jane")
        # unmatched similar faces — same seed vector so they cluster together
        base = _emb_bytes(777)
        for i in range(4):
            _add_face(db, scene_id=10 + i, facenet=_similar_emb(base, seed=i), arcface=_similar_emb(_emb_bytes(99_777), seed=i))

        result = svc.build_clusters(distance_threshold=0.4)
        assert result["faces_total"] == 6
        assert result["clusters_created"] >= 2

        matched = db.list_face_clusters("matched")
        assert len(matched) == 1
        # performer_id is a LOCAL Stash id; the StashDB match lives in stash_ids
        assert matched[0]["performer_id"] is None
        assert db.get_cluster_stash_ids(matched[0]["id"]) == ["p1"]
        assert matched[0]["performer_name"] == "Jane"
        assert matched[0]["face_count"] == 2

    def test_assign_performer_tags_scenes(self, db, monkeypatch):
        from face_cluster_service import FaceClusterService

        f1 = _add_face(db, scene_id=1)
        f2 = _add_face(db, scene_id=2)
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, [f1, f2])

        # Stub stash client
        tagged = []

        class FakeStash:
            def get_scene_performer_ids_sync(self, scene_id):
                return {"id": scene_id, "performer_ids": ["existing"] if scene_id == "1" else []}

            def update_scene_performers_sync(self, scene_id, performer_ids):
                tagged.append((scene_id, performer_ids))

        svc = FaceClusterService(db)
        result = svc.assign_performer(cid, "performer-9", "Jane Doe", FakeStash())

        assert result["scenes_tagged"] == 2
        assert result["scenes_already_tagged"] == 0
        # Scene 1 keeps existing performer and gains the new one
        assert ("1", ["existing", "performer-9"]) in tagged
        # Scene 2 just gains the new performer
        assert ("2", ["performer-9"]) in tagged
        cluster = db.get_face_cluster(cid)
        assert cluster["status"] == "assigned"
        assert cluster["performer_id"] == "performer-9"

    def test_assign_skips_already_tagged(self, db):
        from face_cluster_service import FaceClusterService
        f1 = _add_face(db, scene_id=1)
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, [f1])

        class FakeStash:
            def get_scene_performer_ids_sync(self, scene_id):
                return {"id": scene_id, "performer_ids": ["performer-9"]}

            def update_scene_performers_sync(self, scene_id, performer_ids):
                raise AssertionError("should not write")

        result = FaceClusterService(db).assign_performer(cid, "performer-9", "Jane", FakeStash())
        assert result["scenes_tagged"] == 0
        assert result["scenes_already_tagged"] == 1

    def test_merge_service(self, db):
        from face_cluster_service import FaceClusterService
        f1 = _add_face(db, scene_id=1)
        f2 = _add_face(db, scene_id=2)
        c1 = db.create_face_cluster(name="A")
        c2 = db.create_face_cluster(name="B")
        db.add_faces_to_cluster(c1, [f1])
        db.add_faces_to_cluster(c2, [f2])

        svc = FaceClusterService(db)
        result = svc.merge_clusters([c1], c2)
        assert result["faces_moved"] == 1
        assert db.get_face_cluster(c1) is None
        assert db.get_cluster_face_count(c2) == 2


class TestIncrementalClustering:
    def test_new_faces_absorb_into_assigned_group(self, db):
        from face_cluster_service import FaceClusterService
        svc = FaceClusterService(db)

        # Base person: two faces assigned to local performer 42 (StashDB uuid-9)
        base_fn = _emb_bytes(555)
        base_af = _emb_bytes(556)
        f1 = _add_face(db, scene_id=1, match_id="uuid-9", match_name="Jane")
        f2 = _add_face(db, scene_id=2, facenet=_similar_emb(base_fn, seed=1), arcface=_similar_emb(base_af, seed=1))
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        db.set_cluster_stash_ids(cid, ["uuid-9"])
        db.add_faces_to_cluster(cid, [f1, f2])

        # New similar face, no match anchor
        f3 = _add_face(db, scene_id=3, facenet=_similar_emb(base_fn, seed=2), arcface=_similar_emb(base_af, seed=2))

        result = svc.build_clusters(incremental=True)
        assert result["mode"] == "incremental"
        assert result["absorbed"] == 1
        assert f3 in {r["id"] for r in db.get_representative_faces(cid, limit=100)}
        assert db.list_face_clusters("matched") == []

    def test_auto_tag_absorbed_scenes(self, db):
        from face_cluster_service import FaceClusterService
        svc = FaceClusterService(db)

        base_fn = _emb_bytes(444)
        base_af = _emb_bytes(445)
        f1 = _add_face(db, scene_id=1, facenet=base_fn, arcface=base_af)
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        db.set_cluster_stash_ids(cid, ["uuid-9"])
        db.add_faces_to_cluster(cid, [f1])

        f2 = _add_face(db, scene_id=7, match_id="uuid-9", match_name="Jane",
                       facenet=_similar_emb(base_fn, seed=3), arcface=_similar_emb(base_af, seed=3))

        tagged = []

        class FakeStash:
            def get_scene_performer_ids_sync(self, scene_id):
                return {"id": scene_id, "performer_ids": []}
            def update_scene_performers_sync(self, scene_id, performer_ids):
                tagged.append((scene_id, performer_ids))

        result = svc.build_clusters(incremental=True, auto_tag=True, stash_client=FakeStash())
        assert result["tagged_scenes"] == 1
        assert ("7", ["42"]) in tagged

    def test_dissimilar_faces_form_new_group(self, db):
        from face_cluster_service import FaceClusterService
        svc = FaceClusterService(db)

        f1 = _add_face(db, scene_id=1, facenet=_emb_bytes(444), arcface=_emb_bytes(445))
        cid = db.create_face_cluster(status="assigned", performer_id="p9", performer_name="Jane")
        db.add_faces_to_cluster(cid, [f1])

        # similar-to-each-other but dissimilar-from-group new faces
        base = _emb_bytes(8888)
        for i in range(4):
            _add_face(db, scene_id=10 + i, facenet=_similar_emb(base, seed=10 + i), arcface=_similar_emb(_emb_bytes(8889), seed=10 + i))

        result = svc.build_clusters(incremental=True, min_cluster_size=3)
        assert result["absorbed"] == 0
        assert result["clusters_created"] == 1

    def test_merge_is_recorded(self, db):
        from face_cluster_service import FaceClusterService
        svc = FaceClusterService(db)
        f1 = _add_face(db, scene_id=1)
        c1 = db.create_face_cluster()
        c2 = db.create_face_cluster()
        db.add_faces_to_cluster(c1, [f1])
        svc.merge_clusters([c1], c2)
        mmap = db.get_cluster_merge_map()
        assert mmap[c1] == c2
        assert db.get_face_cluster(c1) is None

    def test_merge_chain_resolution(self, db):
        db.record_cluster_merge([1], 2)
        db.record_cluster_merge([2], 3)
        mmap = db.get_cluster_merge_map()
        assert mmap[1] == 3

    def test_full_rebuild_grows_assigned_group(self, db):
        from face_cluster_service import FaceClusterService
        svc = FaceClusterService(db)

        f1 = _add_face(db, scene_id=1, match_id="uuid-9", match_name="Jane")
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        db.set_cluster_stash_ids(cid, ["uuid-9"])
        db.add_faces_to_cluster(cid, [f1])

        # new face, same StashDB match anchor
        f2 = _add_face(db, scene_id=2, match_id="uuid-9", match_name="Jane")

        svc.build_clusters(replace_existing=True)  # full build
        assert db.get_cluster_face_count(cid) == 2
        assert db.get_face_cluster(cid)["status"] == "assigned"
        assert db.list_face_clusters("matched") == []

    def test_schema_v14(self, db):
        with db._connection() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == 14
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "face_cluster_merge_log" in tables
            assert "face_cluster_stash_ids" in tables
            assert "face_rejections" in tables
