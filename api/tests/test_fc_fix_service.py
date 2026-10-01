"""Regression tests for face-cluster service review findings #4, #5, #6, #8.

#4  best_match_id is a StashDB id; assigned performer_id is a local Stash id.
#5  full builds duplicate groups, re-cluster curated faces, undo merges.
#6  ejected faces rejoin the group they were ejected from.
#8  merge into a missing target silently loses faces.
"""
import numpy as np
import pytest

from recommendations_db import RecommendationsDB


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "test.db")


@pytest.fixture
def svc(db):
    from face_cluster_service import FaceClusterService
    return FaceClusterService(db)


# ---------------------------------------------------------------- helpers

def _emb_bytes(seed: int, dim: int = 512) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, dim).astype(np.float32).tobytes()


def _similar_emb(base: bytes, jitter: float = 0.01, seed: int = 0) -> bytes:
    v = np.frombuffer(base, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    v = v + rng.normal(0, jitter, v.shape).astype(np.float32)
    return v.astype(np.float32).tobytes()


def _add_face(db, scene_id=1, frame=0, match_id=None, match_name=None, facenet=None, arcface=None):
    return db.add_library_face(
        stash_scene_id=scene_id,
        frame_index=frame,
        timestamp_sec=1.0 + frame,
        bbox={"x": 10, "y": 10, "w": 40, "h": 40},
        det_confidence=0.9,
        yaw=0.0,
        facenet_emb=facenet or _emb_bytes(scene_id * 100 + frame),
        arcface_emb=arcface or _emb_bytes(999_000 + scene_id * 100 + frame),
        best_match_id=match_id,
        best_match_name=match_name,
        best_match_confidence=0.8 if match_id else None,
    )


class Person:
    """A synthetic identity: faces drawn near one base embedding."""

    def __init__(self, seed: int):
        self.fn = _emb_bytes(seed)
        self.af = _emb_bytes(seed + 1)
        self._n = 0

    def face(self, db, scene_id, frame=0, match_id=None, match_name=None, jitter=0.01):
        self._n += 1
        s = self._n * 7919 + scene_id * 31 + frame
        return _add_face(
            db, scene_id=scene_id, frame=frame, match_id=match_id, match_name=match_name,
            facenet=_similar_emb(self.fn, jitter=jitter, seed=s),
            arcface=_similar_emb(self.af, jitter=jitter, seed=s + 1),
        )


class FakeStash:
    """Stateful stash client fake: remembers scene performers, counts calls."""

    def __init__(self, performer_stash_ids=None, scene_performers=None, fail_lookup=False):
        self.performer_stash_ids = performer_stash_ids or {}
        self.scenes = {str(k): list(v) for k, v in (scene_performers or {}).items()}
        self.fail_lookup = fail_lookup
        self.reads: list[str] = []
        self.writes: list[tuple[str, list[str]]] = []
        self.lookups: list[str] = []

    def get_scene_performer_ids_sync(self, scene_id):
        self.reads.append(scene_id)
        return {"id": scene_id, "performer_ids": list(self.scenes.get(scene_id, []))}

    def update_scene_performers_sync(self, scene_id, performer_ids):
        self.writes.append((scene_id, list(performer_ids)))
        self.scenes[scene_id] = list(performer_ids)

    def get_performer_stash_ids_sync(self, performer_id):
        self.lookups.append(performer_id)
        if self.fail_lookup:
            raise RuntimeError("stash unreachable")
        if performer_id not in self.performer_stash_ids:
            return None
        # ids prefixed "fansdb-" / "tpdb-" are links to other stash-boxes
        return [{"endpoint": ("https://fansdb.cc/graphql" if s.startswith("fansdb-")
                              else "https://theporndb.net/graphql" if s.startswith("tpdb-")
                              else "https://stashdb.org/graphql"), "stash_id": s}
                for s in self.performer_stash_ids[performer_id]]


def _members(db, cluster_id) -> set[int]:
    return {r["id"] for r in db.get_representative_faces(cluster_id, limit=10**9)}


def _memberships(db, face_id) -> list[int]:
    return [m["id"] for m in db.get_face_cluster_membership(face_id)]


def _max_memberships(db) -> int:
    with db._connection() as conn:
        row = conn.execute(
            "SELECT MAX(n) FROM (SELECT COUNT(*) AS n FROM face_cluster_members GROUP BY face_id)"
        ).fetchone()
    return row[0] or 0


def _live_clusters_with_stash_id(db, stash_id) -> list[int]:
    out = []
    for c in db.get_clusters_by_status("assigned", "matched", "open"):
        if stash_id in db.get_cluster_stash_ids(c["id"]):
            out.append(c["id"])
    return out


def _snapshot(db) -> list[tuple]:
    return sorted(
        (c["status"], tuple(sorted(_members(db, c["id"]))))
        for c in db.list_face_clusters()
    )


# ================================================================== #4

class TestIdSpaces:
    def test_assign_stores_performer_stash_ids_from_stash(self, db, svc):
        f1 = _add_face(db, scene_id=1)
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, [f1])
        stash = FakeStash(performer_stash_ids={"42": ["uuid-x"]})

        result = svc.assign_performer(cid, "42", "Jane", stash)

        assert result["stash_ids"] == ["uuid-x"]
        c = db.get_face_cluster(cid)
        assert c["performer_id"] == "42"
        assert c["status"] == "assigned"
        assert c["pinned"] == 1
        assert db.get_cluster_stash_ids(cid) == ["uuid-x"]
        assert stash.lookups == ["42"]

    @pytest.mark.parametrize("mode", ["raise", "none"])
    def test_assign_stash_lookup_failure_keeps_existing_stash_ids(self, db, svc, mode):
        f1 = _add_face(db, scene_id=1)
        cid = db.create_face_cluster(status="matched", stash_ids=["uuid-old"])
        db.add_faces_to_cluster(cid, [f1])
        stash = FakeStash(fail_lookup=(mode == "raise"))  # "none": performer not found

        result = svc.assign_performer(cid, "42", "Jane", stash)

        assert result["scenes_tagged"] == 1
        assert db.get_cluster_stash_ids(cid) == ["uuid-old"]
        assert result["stash_ids"] == ["uuid-old"]
        assert db.get_face_cluster(cid)["status"] == "assigned"

    def test_assign_missing_cluster_raises_cluster_not_found(self, svc):
        from face_cluster_service import ClusterNotFound
        with pytest.raises(ClusterNotFound):
            svc.assign_performer(9999, "42", "Jane", FakeStash())

    def test_incremental_stashdb_match_joins_assigned_group_with_local_id(self, db, svc):
        f1 = _add_face(db, scene_id=1)
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane",
                                     stash_ids=["uuid-x"])
        db.add_faces_to_cluster(cid, [f1])
        # dissimilar embedding: only the StashDB match can place it
        f2 = _add_face(db, scene_id=2, match_id="uuid-x", match_name="Jane")

        result = svc.build_clusters(incremental=True)

        assert f2 in _members(db, cid)
        assert db.list_face_clusters("matched") == []
        assert result["absorbed"] == 1

    def test_full_build_no_matched_duplicate_for_assigned_performer(self, db, svc):
        f1 = _add_face(db, scene_id=1, match_id="uuid-x", match_name="Jane")
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane",
                                     stash_ids=["uuid-x"], pinned=True)
        db.add_faces_to_cluster(cid, [f1])
        f2 = _add_face(db, scene_id=2, match_id="uuid-x", match_name="Jane")

        for _ in range(2):
            svc.build_clusters()

        assert db.list_face_clusters("matched") == []
        assert _members(db, cid) == {f1, f2}
        assert _live_clusters_with_stash_id(db, "uuid-x") == [cid]
        assert _max_memberships(db) == 1

    def test_sync_assigned_stash_ids_backfills(self, db, svc):
        a = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        b = db.create_face_cluster(status="assigned", performer_id="43", performer_name="Bob",
                                   stash_ids=["keep"])
        c = db.create_face_cluster(status="assigned", performer_id="44", performer_name="Gone")
        stash = FakeStash(performer_stash_ids={"42": ["uuid-x"], "43": ["other"]})

        assert svc.sync_assigned_stash_ids(stash) == 2

        assert db.get_cluster_stash_ids(a) == ["uuid-x"]
        # performer linked on StashDB: its links are the identity (round 3, #4)
        assert db.get_cluster_stash_ids(b) == ["other"]
        assert db.get_blocked_stash_ids(b) == {"keep"}
        assert db.get_cluster_stash_ids(c) == []  # not found in Stash
        # synced groups are not looked up again; not-found ones are retried
        stash.lookups.clear()
        assert svc.sync_assigned_stash_ids(stash) == 0
        assert stash.lookups == ["44"]

    def test_sync_assigned_stash_ids_swallows_errors(self, db, svc):
        db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        assert svc.sync_assigned_stash_ids(FakeStash(fail_lookup=True)) == 0

    def test_build_backfills_then_absorbs_by_stash_id(self, db, svc):
        f1 = _add_face(db, scene_id=1)
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        db.add_faces_to_cluster(cid, [f1])
        f2 = _add_face(db, scene_id=2, match_id="uuid-x", match_name="Jane")

        svc.build_clusters(incremental=True, stash_client=FakeStash(performer_stash_ids={"42": ["uuid-x"]}))

        assert f2 in _members(db, cid)
        assert db.list_face_clusters("matched") == []

    def test_get_performer_stash_ids_sync(self, monkeypatch):
        from stash_client_unified import StashClientUnified
        client = StashClientUnified("http://stash.local", "key")
        calls = []

        def fake_exec(query, variables=None):
            calls.append((query, variables))
            if variables["id"] == "42":
                return {"findPerformer": {"id": "42", "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "uuid-x"},
                    {"endpoint": "https://fansdb.cc/graphql", "stash_id": "uuid-y"},
                ]}}
            return {"findPerformer": None}

        monkeypatch.setattr(client, "_execute_sync", fake_exec)

        assert client.get_performer_stash_ids_sync("42") == [
            {"endpoint": "https://stashdb.org/graphql", "stash_id": "uuid-x"},
            {"endpoint": "https://fansdb.cc/graphql", "stash_id": "uuid-y"},
        ]
        assert client.get_performer_stash_ids_sync("404") is None
        assert "findPerformer" in calls[0][0] and "stash_ids" in calls[0][0]
        assert calls[0][1] == {"id": "42"}


# ================================================================== #5

def _library(db):
    """Two StashDB-matched identities, one unmatched identity, one stray face."""
    a, b, p = Person(1000), Person(2000), Person(3000)
    ids = {
        "a": [a.face(db, 1, match_id="uuid-a", match_name="Alice"), a.face(db, 2, match_id="uuid-a", match_name="Alice")],
        "b": [b.face(db, 3, match_id="uuid-b", match_name="Bea"), b.face(db, 4, match_id="uuid-b", match_name="Bea")],
        "p": [p.face(db, 10 + i) for i in range(4)],
        "stray": [_add_face(db, scene_id=50)],
    }
    return ids


class TestFullBuild:
    @pytest.mark.parametrize("replace", [False, True])
    def test_full_build_twice_is_idempotent(self, db, svc, replace):
        _library(db)
        svc.build_clusters(distance_threshold=0.4, replace_existing=replace)
        first = _snapshot(db)
        svc.build_clusters(distance_threshold=0.4, replace_existing=replace)
        second = _snapshot(db)

        assert second == first
        assert len(db.list_face_clusters()) == 3  # matched a, matched b, open p
        assert len(_live_clusters_with_stash_id(db, "uuid-a")) == 1
        assert len(_live_clusters_with_stash_id(db, "uuid-b")) == 1
        assert _max_memberships(db) == 1

    def test_full_build_never_reclusters_assigned_ignored_banned_faces(self, db, svc):
        q, r, s = Person(4000), Person(5000), Person(6000)
        # assigned group (local id 42) whose face matched uuid-q; another loose uuid-q face
        fa = q.face(db, 1, match_id="uuid-q", match_name="Quinn")
        ga = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Quinn", pinned=True)
        db.add_faces_to_cluster(ga, [fa])
        q.face(db, 2, match_id="uuid-q", match_name="Quinn")
        # ignored face that looks like 3 unassigned faces
        fi = r.face(db, 3)
        gi = db.create_face_cluster(status="ignored")
        db.add_faces_to_cluster(gi, [fi])
        for i in range(3):
            r.face(db, 20 + i)
        # banned face that looks like 3 unassigned faces
        fb = s.face(db, 4)
        svc.ban_faces([fb])
        gb = _memberships(db, fb)[0]
        for i in range(3):
            s.face(db, 30 + i)

        for _ in range(2):
            svc.build_clusters(distance_threshold=0.4)

        assert _memberships(db, fa) == [ga]
        assert _memberships(db, fi) == [gi]
        assert _memberships(db, fb) == [gb]
        assert _max_memberships(db) == 1

    def test_full_build_preserves_merged_group(self, db, svc):
        p, q = Person(7000), Person(8000)
        pa = [p.face(db, 1 + i) for i in range(3)]
        qb = [q.face(db, 10 + i) for i in range(3)]
        ga = db.create_face_cluster()
        gb = db.create_face_cluster()
        db.add_faces_to_cluster(ga, pa)
        db.add_faces_to_cluster(gb, qb)
        svc.merge_clusters([gb], ga)

        svc.build_clusters(distance_threshold=0.4)

        assert db.get_face_cluster(ga) is not None
        assert _members(db, ga) == set(pa + qb)
        others = [c["id"] for c in db.list_face_clusters() if c["id"] != ga]
        for cid in others:
            assert not (_members(db, cid) & set(pa + qb))

    def test_full_build_reuses_matched_cluster_id(self, db, svc):
        a = Person(9000)
        a.face(db, 1, match_id="uuid-a", match_name="Alice")
        a.face(db, 2, match_id="uuid-a", match_name="Alice")

        svc.build_clusters()
        [m1] = db.list_face_clusters("matched")
        a.face(db, 3, match_id="uuid-a", match_name="Alice")
        svc.build_clusters()
        matched = db.list_face_clusters("matched")

        assert [m["id"] for m in matched] == [m1["id"]]
        assert matched[0]["face_count"] == 3
        assert matched[0]["performer_id"] is None
        assert db.get_cluster_stash_ids(m1["id"]) == ["uuid-a"]

    def test_full_build_dissolves_unpinned_open_groups(self, db, svc):
        loose = db.create_face_cluster(status="open")
        db.add_faces_to_cluster(loose, [_add_face(db, scene_id=1), _add_face(db, scene_id=2)])
        kept = db.create_face_cluster(status="open", pinned=True)
        kept_faces = [_add_face(db, scene_id=3), _add_face(db, scene_id=4)]
        db.add_faces_to_cluster(kept, kept_faces)

        result = svc.build_clusters(min_cluster_size=3)

        assert result["mode"] == "full"
        assert result["dissolved"] == 1
        assert db.get_face_cluster(loose) is None
        assert _members(db, kept) == set(kept_faces)

    def test_full_build_auto_tags_assigned_scenes(self, db, svc):
        f1 = _add_face(db, scene_id=1)
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane",
                                     stash_ids=["uuid-x"], pinned=True)
        db.add_faces_to_cluster(cid, [f1])
        _add_face(db, scene_id=7, match_id="uuid-x")
        stash = FakeStash()

        result = svc.build_clusters(auto_tag=True, stash_client=stash)

        assert stash.writes == [("7", ["42"])]
        assert result["tagged_scenes"] == 1


# ================================================================== #6

class TestRejectionMemory:
    def test_ejected_face_not_reabsorbed_by_centroid(self, db, svc):
        p = Person(11000)
        faces = [p.face(db, 1 + i) for i in range(4)]
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, faces)

        assert svc.eject_faces(cid, [faces[0]]) == 1
        svc.build_clusters(incremental=True)

        assert faces[0] not in _members(db, cid)
        assert db.get_face_cluster(cid)["pinned"] == 1

    def test_ejected_matched_face_not_rejoined_by_best_match_shortcut(self, db, svc):
        # dissimilar embeddings: only the best-match shortcut groups them
        fs = [_add_face(db, scene_id=1 + i, match_id="uuid-a", match_name="Alice") for i in range(3)]
        svc.build_clusters()
        [m] = db.list_face_clusters("matched")

        svc.eject_faces(m["id"], [fs[0]])
        for _ in range(2):
            svc.build_clusters(incremental=True)

        assert fs[0] not in _members(db, m["id"])
        assert _memberships(db, fs[0]) == []  # and no fresh uuid-a group re-seeded for it

    def test_ejected_from_assigned_not_reseeded_as_matched_group(self, db, svc):
        f1 = _add_face(db, scene_id=1)
        fx = _add_face(db, scene_id=2, match_id="uuid-x")
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane",
                                     stash_ids=["uuid-x"])
        # performer really linked to uuid-x (round 4: an unsynced group's id is
        # only a member guess, and a face is never rejected from its own match)
        db.set_cluster_stash_ids_synced(cid, True)
        db.add_faces_to_cluster(cid, [f1, fx])

        svc.eject_faces(cid, [fx])
        svc.build_clusters()
        svc.build_clusters(incremental=True)

        assert _memberships(db, fx) == []
        assert db.list_face_clusters("matched") == []

    def test_ejected_from_assigned_not_joined_to_other_group_of_same_performer(self, db, svc):
        p = Person(12000)
        fx = p.face(db, 1)
        g1 = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        db.add_faces_to_cluster(g1, [fx, p.face(db, 2), p.face(db, 3)])
        g2 = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        db.add_faces_to_cluster(g2, [p.face(db, 4), p.face(db, 5)])

        svc.eject_faces(g1, [fx])
        svc.build_clusters(incremental=True)

        assert _memberships(db, fx) == []

    def test_rejection_follows_merge_map(self, db, svc):
        p = Person(13000)
        fx = p.face(db, 1)
        s = db.create_face_cluster()
        db.add_faces_to_cluster(s, [fx, p.face(db, 2), p.face(db, 3)])
        t = db.create_face_cluster()
        db.add_faces_to_cluster(t, [p.face(db, 4), p.face(db, 5)])

        svc.eject_faces(s, [fx])
        svc.merge_clusters([s], t)
        svc.build_clusters(incremental=True)

        assert fx not in _members(db, t)

    def test_ejected_face_can_join_a_different_group(self, db, svc):
        p = Person(14000)
        fx = p.face(db, 1)
        near = db.create_face_cluster()
        db.add_faces_to_cluster(near, [fx, p.face(db, 2), p.face(db, 3)])
        # same neighbourhood but farther away: second-best centroid
        far = db.create_face_cluster()
        db.add_faces_to_cluster(far, [p.face(db, 10 + i, jitter=0.3) for i in range(3)])

        svc.eject_faces(near, [fx])
        svc.build_clusters(incremental=True)

        assert _memberships(db, fx) == [far]

    def test_split_records_rejection(self, db, svc):
        p = Person(15000)
        faces = [p.face(db, 1 + i) for i in range(4)]
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, faces)

        new_cid = svc.split_cluster(cid, [faces[0]])

        assert _memberships(db, faces[0]) == [new_cid]
        assert db.get_face_cluster(new_cid)["pinned"] == 1
        assert db.get_face_rejections([faces[0]])[faces[0]]["clusters"] == {cid}
        # ejected again from the split group, it must not drift back into the original
        svc.eject_faces(new_cid, [faces[0]])
        svc.build_clusters(incremental=True)
        assert faces[0] not in _members(db, cid)

    def test_split_missing_cluster_raises(self, svc):
        from face_cluster_service import ClusterNotFound
        with pytest.raises(ClusterNotFound):
            svc.split_cluster(9999, [1])

    def test_eject_missing_cluster_raises(self, svc):
        from face_cluster_service import ClusterNotFound
        with pytest.raises(ClusterNotFound):
            svc.eject_faces(9999, [1])


# ================================================================== #8

class TestMerge:
    def _two(self, db, **target_kw):
        f1 = _add_face(db, scene_id=1)
        f2 = _add_face(db, scene_id=2)
        src = db.create_face_cluster()
        db.add_faces_to_cluster(src, [f1])
        tgt = db.create_face_cluster(**target_kw)
        db.add_faces_to_cluster(tgt, [f2])
        return src, tgt, f1, f2

    def test_merge_missing_target_raises_cluster_not_found_and_sources_intact(self, db, svc):
        from face_cluster_service import ClusterNotFound
        src, _tgt, f1, _ = self._two(db)
        with pytest.raises(ClusterNotFound):
            svc.merge_clusters([src], 9999)
        assert db.get_face_cluster(src) is not None
        assert _members(db, src) == {f1}
        assert db.get_cluster_merge_map() == {}

    def test_merge_missing_source_raises(self, db, svc):
        from face_cluster_service import ClusterNotFound
        src, tgt, f1, f2 = self._two(db)
        with pytest.raises(ClusterNotFound):
            svc.merge_clusters([src, 9999], tgt)
        assert _members(db, src) == {f1}
        assert _members(db, tgt) == {f2}

    def test_merge_only_target_rejected(self, db, svc):
        from face_cluster_service import InvalidClusterOperation
        _src, tgt, _, _ = self._two(db)
        with pytest.raises(InvalidClusterOperation):
            svc.merge_clusters([tgt], tgt)

    def test_merge_assigned_source_into_open_target_rejected(self, db, svc):
        from face_cluster_service import InvalidClusterOperation
        src, tgt, f1, f2 = self._two(db)
        db.update_face_cluster(src, status="assigned", performer_id="42", performer_name="Jane")
        with pytest.raises(InvalidClusterOperation, match="assigned group"):
            svc.merge_clusters([src], tgt)
        assert _members(db, src) == {f1}
        assert db.get_face_cluster(src)["performer_id"] == "42"

    def test_merge_two_assigned_different_performers_rejected(self, db, svc):
        from face_cluster_service import InvalidClusterOperation
        src, tgt, f1, f2 = self._two(db, status="assigned", performer_id="43", performer_name="Bob")
        db.update_face_cluster(src, status="assigned", performer_id="42", performer_name="Jane")
        with pytest.raises(InvalidClusterOperation, match="different performers"):
            svc.merge_clusters([src], tgt)
        assert _members(db, src) == {f1}
        assert _members(db, tgt) == {f2}

    def test_merge_banned_rejected(self, db, svc):
        from face_cluster_service import InvalidClusterOperation
        src, tgt, _, _ = self._two(db, status="banned")
        with pytest.raises(InvalidClusterOperation):
            svc.merge_clusters([src], tgt)

    def test_merge_same_performer_allowed(self, db, svc):
        src, tgt, f1, f2 = self._two(db, status="assigned", performer_id="42", performer_name="Jane")
        db.update_face_cluster(src, status="assigned", performer_id="42", performer_name="Jane")
        result = svc.merge_clusters([src], tgt)
        assert result == {"target_id": tgt, "merged_sources": [src], "faces_moved": 1,
                          "scenes_tagged": 0}
        assert _members(db, tgt) == {f1, f2}
        assert db.get_face_cluster(src) is None

    def test_merge_open_into_assigned_allowed(self, db, svc):
        src, tgt, f1, f2 = self._two(db, status="assigned", performer_id="42", performer_name="Jane")
        svc.merge_clusters([src], tgt)
        assert _members(db, tgt) == {f1, f2}

    def test_merge_moves_stash_ids_and_pins_target(self, db, svc):
        src, tgt, f1, f2 = self._two(db)
        db.set_cluster_stash_ids(src, ["uuid-a"])
        db.update_face_cluster(src, status="matched")
        svc.merge_clusters([src], tgt)
        t = db.get_face_cluster(tgt)
        assert t["stash_ids"] == ["uuid-a"]
        assert t["pinned"] == 1
        assert db.get_cluster_merge_map() == {src: tgt}


# ================================================================== misc

class TestMisc:
    def test_build_in_progress_raises(self, db, svc):
        import face_cluster_service as fcs
        _add_face(db, scene_id=1)
        assert fcs._BUILD_LOCK.acquire(blocking=False)
        try:
            with pytest.raises(fcs.BuildInProgress):
                svc.build_clusters()
            with pytest.raises(fcs.BuildInProgress):
                svc.build_clusters(incremental=True)
        finally:
            fcs._BUILD_LOCK.release()
        svc.build_clusters()  # lock released: works again
        assert fcs._BUILD_LOCK.acquire(blocking=False)
        fcs._BUILD_LOCK.release()

    def test_build_releases_lock_on_error(self, db, svc, monkeypatch):
        import face_cluster_service as fcs
        _add_face(db, scene_id=1)
        monkeypatch.setattr(svc, "_place_pool", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            svc.build_clusters(incremental=True)
        assert fcs._BUILD_LOCK.acquire(blocking=False)
        fcs._BUILD_LOCK.release()

    def test_auto_tag_one_write_per_scene(self, db, svc):
        p = Person(16000)
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        db.add_faces_to_cluster(cid, [p.face(db, 1)])
        new = [p.face(db, 7, frame=i) for i in range(3)] + [p.face(db, 8)]
        stash = FakeStash()

        result = svc.build_clusters(incremental=True, auto_tag=True, stash_client=stash)

        assert result["absorbed"] == len(new)
        assert sorted(stash.reads) == ["7", "8"]
        assert sorted(stash.writes) == [("7", ["42"]), ("8", ["42"])]
        assert result["tagged_scenes"] == 2

    def test_auto_tag_failure_isolated_per_scene(self, db, svc):
        p = Person(17000)
        cid = db.create_face_cluster(status="assigned", performer_id="42", performer_name="Jane")
        db.add_faces_to_cluster(cid, [p.face(db, 1)])
        p.face(db, 7)
        p.face(db, 8)

        class Flaky(FakeStash):
            def get_scene_performer_ids_sync(self, scene_id):
                if scene_id == "7":
                    raise RuntimeError("timeout")
                return super().get_scene_performer_ids_sync(scene_id)

        stash = Flaky()
        result = svc.build_clusters(incremental=True, auto_tag=True, stash_client=stash)
        assert stash.writes == [("8", ["42"])]
        assert result["tagged_scenes"] == 1

    def test_update_cluster_rename_pins(self, db, svc):
        cid = db.create_face_cluster()
        out = svc.update_cluster(cid, name="Background guy")
        assert out["name"] == "Background guy"
        assert out["pinned"] == 1
        out = svc.update_cluster(cid, status="ignored")
        assert out["status"] == "ignored"

    @pytest.mark.parametrize("status", ["assigned", "matched", "banned", "bogus"])
    def test_update_cluster_invalid_status_raises(self, db, svc, status):
        from face_cluster_service import InvalidClusterOperation
        cid = db.create_face_cluster()
        with pytest.raises(InvalidClusterOperation):
            svc.update_cluster(cid, status=status)
        assert db.get_face_cluster(cid)["status"] == "open"
        assert db.get_face_cluster(cid)["pinned"] == 0

    def test_update_cluster_missing_raises(self, svc):
        from face_cluster_service import ClusterNotFound
        with pytest.raises(ClusterNotFound):
            svc.update_cluster(9999, name="x")

    def test_exceptions_are_value_error_subclasses(self):
        from face_cluster_service import BuildInProgress, ClusterNotFound, InvalidClusterOperation
        assert issubclass(ClusterNotFound, ValueError)
        assert issubclass(InvalidClusterOperation, ValueError)
        assert issubclass(BuildInProgress, RuntimeError)

    def test_ignore_pins_and_stats_count_banned(self, db, svc):
        cid = db.create_face_cluster()
        db.add_faces_to_cluster(cid, [_add_face(db, scene_id=1)])
        assert svc.ignore_cluster(cid)
        assert db.get_face_cluster(cid)["pinned"] == 1
        fb = _add_face(db, scene_id=2)
        svc.ban_faces([fb])
        assert db.get_face_cluster(_memberships(db, fb)[0])["pinned"] == 1
        st = svc.stats()
        assert st["banned"] == 1
        assert st["ignored"] == 1
        assert st["clusters"] == 1
