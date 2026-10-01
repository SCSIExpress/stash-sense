"""Repair round 2: regression tests for the second skeptic pass on the
face-cluster fixes. Each test fails on the pre-round code and passes with the
fix. Grouped by the original review finding they extend.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
from fastapi.testclient import TestClient

import face_cluster_service as fcs
import face_clusters_router
import library_face_persist
import scene_matcher
import signal_scoring
from face_cluster_service import BuildInProgress, ClusterNotFound, FaceClusterService
from identification_router import _match_to_response, distance_to_confidence
from recommendations_db import RecommendationsDB
from tests.test_fc_fix_round2 import _m, _make_app, _res
from tests.test_fc_fix_service import (
    FakeStash,
    Person,
    _add_face,
    _live_clusters_with_stash_id,
    _max_memberships,
    _members,
    _memberships,
)


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "r3.db")


@pytest.fixture
def svc(db):
    return FaceClusterService(db)


# ======================================================================
# Findings 1 / 3: a rerank can give two persons the same best match; two
# faces in one frame must never both be anchored to that performer
# ======================================================================

def _rerank(persons, monkeypatch, favour: str | None = None, disfavour: str | None = None):
    """Run the real multi-signal rerank with a scene-level tattoo multiplier."""
    monkeypatch.setattr(signal_scoring, "body_ratio_penalty", lambda *a, **k: 1.0)

    def tattoo(_tr, uid, _scores, _has):
        if favour and uid.endswith(favour):
            return 1.5
        if disfavour and uid.endswith(disfavour):
            return 0.85
        return 1.0

    monkeypatch.setattr(signal_scoring, "tattoo_adjustment", tattoo)
    matcher = NS(_get_candidate_body_ratios=lambda uid: None, performers_with_tattoo_embeddings=set())
    return scene_matcher._rerank_scene_persons(persons, matcher, None, None, None, ["face", "tattoo"], 1)


def _anchors(persons, all_results):
    frames = [f for f, _ in all_results]
    fa = library_face_persist.face_person_assignment(
        persons, len(all_results), face_frames=frames,
        face_matches=[r.matches for _f, r in all_results])
    return {i: persons[k].best_match.stashdb_id for i, k in fa.items()}, frames


def _per_frame_collisions(anchors, frames):
    by: dict = {}
    for i, sid in anchors.items():
        by.setdefault((sid, frames[i]), []).append(i)
    return {k: v for k, v in by.items() if len(v) > 1}


def test_frequency_mode_rerank_to_shared_best_match_anchors_one_face_per_frame(monkeypatch):
    # skeptic #1 repro: B's frame-0 face tops sA, the tattoo signal flips B to sA
    a = [(f, _res("A", [_m("sA", 0.2)])) for f in range(3)]
    b = [(0, _res("B", [_m("sA", 0.35), _m("sB", 0.38)])),
         (1, _res("B", [_m("sB", 0.30)])),
         (2, _res("B", [_m("sB", 0.30)]))]
    all_results = a + b
    persons = scene_matcher.clustered_frequency_matching(
        all_results, None, top_k=5, max_distance=0.6,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    assert [p.best_match.stashdb_id for p in persons] == ["sA", "sB"]
    persons = _rerank(persons, monkeypatch, disfavour="sB")
    assert [p.best_match.stashdb_id for p in persons] == ["sA", "sA"]  # the rerank collided

    anchors, frames = _anchors(persons, all_results)
    assert _per_frame_collisions(anchors, frames) == {}
    # B's faces (3..5) are a different person: none is anchored to A's performer
    assert all(i < 3 for i in anchors)
    assert anchors == {0: "sA", 1: "sA", 2: "sA"}


def test_rerank_boost_does_not_anchor_costar_face_in_shared_frame(monkeypatch):
    # skeptic #3 repro: B's frame-3 face tops sA, a tattoo boost for sA flips B
    all_results = [
        (0, _res("A", [_m("sA", 0.20)])),
        (1, _res("A", [_m("sA", 0.21)])),
        (2, _res("A", [_m("sA", 0.22)])),
        (3, _res("A", [_m("sA", 0.23)])),
        (0, _res("B", [_m("sB", 0.30), _m("sA", 0.35)])),
        (3, _res("B", [_m("sA", 0.33), _m("sB", 0.40)])),
    ]
    persons = scene_matcher.clustered_frequency_matching(
        all_results, None, top_k=5, max_distance=0.6,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    persons = _rerank(persons, monkeypatch, favour="sA")
    assert sorted(p.best_match.stashdb_id for p in persons) == ["sA", "sA"]

    anchors, frames = _anchors(persons, all_results)
    assert _per_frame_collisions(anchors, frames) == {}
    assert 5 not in anchors and 4 not in anchors
    assert anchors == {0: "sA", 1: "sA", 2: "sA", 3: "sA"}


def test_shared_best_match_goes_to_person_with_most_face_support():
    # person order does not decide: the person the faces actually agree with does
    small = NS(best_match=NS(stashdb_id="sA"), all_matches=[])
    big = NS(best_match=NS(stashdb_id="sA"), all_matches=[])
    scene_matcher.set_face_indices(small, [0])
    scene_matcher.set_face_indices(big, [1, 2, 3])
    matches = [[_m("sA", 0.3)], [_m("sA", 0.2)], [_m("sA", 0.2)], [_m("sA", 0.2)]]
    fa = library_face_persist.face_person_assignment(
        [small, big], 4, face_frames=[0, 1, 2, 3], face_matches=matches)
    assert fa == {1: 1, 2: 1, 3: 1}


def test_persist_after_rerank_never_puts_costar_in_performer_group(monkeypatch, db, tmp_path):
    """End to end: identify persons -> rerank -> persist -> build. B's faces
    must not be dropped into A's matched group by the match shortcut."""
    from tests.test_fc_fix_round2 import _identify_and_persist
    from library_face_store import LibraryFaceStore

    s = LibraryFaceStore(tmp_path)

    def fake_save_crop(scene_id, frame_index, bbox, frame_image, timestamp_sec=None):
        rel = f"{scene_id}/{frame_index}_{bbox['x']}_{timestamp_sec}.jpg"
        (s.base / rel).parent.mkdir(parents=True, exist_ok=True)
        (s.base / rel).write_bytes(b"jpg")
        return rel

    monkeypatch.setattr(s, "save_crop", fake_save_crop)
    a = [(f, _res("A", [_m("sA", 0.2)])) for f in range(3)]
    b = [(0, _res("B", [_m("sA", 0.35), _m("sB", 0.38)])),
         (1, _res("B", [_m("sB", 0.30)])),
         (2, _res("B", [_m("sB", 0.30)]))]
    all_results = a + b
    persons = scene_matcher.clustered_frequency_matching(
        all_results, None, top_k=5, max_distance=0.6,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    persons = _rerank(persons, monkeypatch, disfavour="sB")
    rows = _identify_and_persist(db, s, all_results, persons)
    assert [rows[i]["best_match_id"] for i in range(6)] == ["sA", "sA", "sA", None, None, None]


# ======================================================================
# Finding 7 leftovers: box size in the face key and the crop key; paired
# updates never deadlock on each other's keys
# ======================================================================

def _fd(frame, ts, x=10, y=10, w=100, h=100, seed=0, match=None):
    from tests.test_fc_fix_round2 import _emb_b
    return {"frame_index": frame, "timestamp_sec": ts, "bbox": {"x": x, "y": y, "w": w, "h": h},
            "det_confidence": 0.95, "yaw": 0.0, "facenet_emb": _emb_b(seed),
            "arcface_emb": _emb_b(seed + 7), "crop_path": None, "best_match_id": match,
            "best_match_name": None, "best_match_confidence": None, "db_version": None}


class TestUpsertKeys:
    def test_same_corner_different_size_is_stored_next_to_curated_row(self, db):
        (kept,) = db.upsert_scene_library_faces(5, [_fd(0, 9.0, w=100, h=100, seed=1)])["face_ids"]
        db.add_face_rejections([kept], cluster_id=1)          # curated: retained
        new = _fd(0, 9.0, w=20, h=20, seed=500)                 # IoU 0.04, far embedding
        res = db.upsert_scene_library_faces(5, [new])
        assert res["face_ids"][0] not in (None, kept)
        assert res["inserted"] == 1 and res["retained"] == 1

    def test_crop_key_includes_box_size(self, tmp_path):
        from library_face_store import LibraryFaceStore
        s = LibraryFaceStore(tmp_path)
        img = np.full((300, 300, 3), 7, np.uint8)
        a = s.save_crop(1, 0, {"x": 10, "y": 10, "w": 100, "h": 100}, img, timestamp_sec=9.0)
        b = s.save_crop(1, 0, {"x": 10, "y": 10, "w": 20, "h": 20}, img, timestamp_sec=9.0)
        assert a != b   # the kept row's crop file is never overwritten

    def test_timestamp_swap_updates_both_rows(self, db):
        # same box at two moments; the re-detection lists them the other way round
        a, b = db.upsert_scene_library_faces(6, [
            _fd(0, 9.0, seed=1), _fd(0, 9.2, seed=2)])["face_ids"]
        res = db.upsert_scene_library_faces(6, [
            _fd(0, 9.2, seed=12, match="uuid-b"), _fd(0, 9.0, seed=11, match="uuid-a")])
        assert res["conflicts"] == 0 and res["updated"] == 2
        assert res["face_ids"] == [b, a]
        assert db.get_library_face(a)["best_match_id"] == "uuid-a"
        assert db.get_library_face(b)["best_match_id"] == "uuid-b"

    def test_paired_update_retried_after_blocking_row_moves(self, db, monkeypatch):
        # force an ordering where the first update needs the second row's key
        a, b = db.upsert_scene_library_faces(8, [
            _fd(0, 1.0, x=10, seed=1), _fd(0, 1.0, x=300, seed=2)])["face_ids"]
        # new face 0 pairs with a but takes b's key; new face 1 pairs with b and moves it
        import recommendations_db as rdb
        real_iou = rdb._bbox_iou
        monkeypatch.setattr(rdb, "_bbox_iou", lambda n, o: 1.0 if (n["x"], o["x"]) in (
            (300, 10), (600, 300)) else (0.0 if n["x"] in (300, 600) else real_iou(n, o)))
        res = db.upsert_scene_library_faces(8, [
            _fd(0, 1.0, x=300, seed=1, match="m-a"), _fd(0, 1.0, x=600, seed=2, match="m-b")])
        assert res["conflicts"] == 0 and res["updated"] == 2
        assert db.get_library_face(a)["bbox_x"] == 300
        assert db.get_library_face(b)["bbox_x"] == 600


def test_migrate_v13_to_v14_keeps_data_and_widens_key(tmp_path):
    import sqlite3
    import recommendations_db as rdb
    path = tmp_path / "m13.db"
    d = RecommendationsDB(path)
    fid = _add_face(d, scene_id=1)
    cid = d.create_face_cluster(status="assigned", performer_id="42", pinned=True)
    d.add_faces_to_cluster(cid, [fid])
    conn = sqlite3.connect(path)
    conn.executescript("""
        DROP TABLE face_cluster_blocked_stash_ids;
        DROP INDEX uq_lib_faces_key;
        CREATE UNIQUE INDEX uq_lib_faces_key ON library_faces(
            stash_scene_id, frame_index, IFNULL(timestamp_sec, -1.0), bbox_x, bbox_y);
        UPDATE schema_version SET version = 13;
    """)
    conn.close()

    d = RecommendationsDB(path)
    with d._connection() as c:
        assert c.execute("SELECT version FROM schema_version").fetchone()[0] == rdb.SCHEMA_VERSION == 14
        cols = [r[2] for r in c.execute("PRAGMA index_info(uq_lib_faces_key)")]
        assert "bbox_w" in cols and "bbox_h" in cols
        assert c.execute("PRAGMA busy_timeout").fetchone()[0] >= 30000
    assert _memberships(d, fid) == [cid]
    d.block_cluster_stash_ids(cid, ["uuid-x"])
    assert d.get_blocked_stash_ids(cid) == {"uuid-x"}


# ======================================================================
# Finding 4: the performer's StashDB link is the identity; unlinked
# performers keep being re-checked; a re-assign is not undone by members
# ======================================================================

class TestStashIdentity:
    def test_performer_link_replaces_contradicted_anchor(self, db, svc):
        p1, p2 = Person(61000), Person(62000)
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-x"], name="Look-alike")
        db.add_faces_to_cluster(m, [p1.face(db, 100 + i, match_id="uuid-x") for i in range(3)])
        g2 = db.create_face_cluster(status="assigned", performer_id="42", performer_name="P2",
                                    pinned=True, stash_ids=["uuid-x"])
        db.add_faces_to_cluster(g2, [p2.face(db, 200 + i, match_id="uuid-x") for i in range(2)])
        stash = FakeStash(performer_stash_ids={"41": ["uuid-y"], "42": ["uuid-x"]})

        svc.assign_performer(m, "41", "P1", stash)       # user: these faces are 41 (uuid-y)
        assert db.get_cluster_stash_ids(m) == ["uuid-y"]

        new = p2.face(db, 50, match_id="uuid-x")
        for incremental in (True, False):
            svc.build_clusters(incremental=incremental, auto_tag=True, stash_client=stash)
            assert _memberships(db, new) == [g2]
        assert stash.scenes.get("50") == ["42"]          # never tagged with 41

    def test_unlinked_performer_is_resynced_once_linked(self, db, svc):
        p = Person(63000)
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        db.add_faces_to_cluster(m, [p.face(db, 1 + i, match_id="uuid-x") for i in range(2)])
        stash = FakeStash(performer_stash_ids={"42": []})
        svc.assign_performer(m, "42", "Jane", stash)
        assert db.get_cluster_stash_ids(m) == ["uuid-x"]
        assert db.get_face_cluster(m)["stash_ids_synced"] == 0

        stash.performer_stash_ids["42"] = ["uuid-real"]   # linked in Stash later
        new = p.face(db, 9, match_id="uuid-real")
        svc.build_clusters(incremental=True, stash_client=stash)

        assert db.get_cluster_stash_ids(m) == ["uuid-real"]
        assert new in _members(db, m)
        assert db.list_face_clusters("matched") == []
        stash.lookups.clear()
        svc.build_clusters(incremental=True, stash_client=stash)
        assert stash.lookups == []                        # linked: synced for good

    def test_reassign_to_unlinked_performer_is_not_undone_by_member_anchor(self, db, svc):
        p = Person(64000)
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        db.add_faces_to_cluster(m, [p.face(db, 1 + i, match_id="uuid-x") for i in range(3)])
        stash = FakeStash(performer_stash_ids={"41": ["uuid-x"], "43": []})
        svc.assign_performer(m, "41", "Wrong", stash)
        svc.assign_performer(m, "43", "Right", stash)
        assert db.get_cluster_stash_ids(m) == []
        for incremental in (True, False):
            svc.build_clusters(incremental=incremental, stash_client=stash)
            assert db.get_cluster_stash_ids(m) == []

        newcomer = _add_face(db, scene_id=77, match_id="uuid-x")
        svc.build_clusters(incremental=True, auto_tag=True, stash_client=stash)
        assert newcomer not in _members(db, m)
        assert "43" not in stash.scenes.get("77", [])

    def test_user_merge_lifts_the_block(self, db, svc):
        p = Person(64500)
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        db.add_faces_to_cluster(m, [p.face(db, 1 + i, match_id="uuid-x") for i in range(3)])
        stash = FakeStash(performer_stash_ids={"41": ["uuid-x"], "43": []})
        svc.assign_performer(m, "41", "Wrong", stash)
        svc.assign_performer(m, "43", "Right", stash)
        other = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        db.add_faces_to_cluster(other, [p.face(db, 20, match_id="uuid-x")])
        svc.merge_clusters([other], m)                   # explicit: uuid-x is 43 after all
        assert db.get_cluster_stash_ids(m) == ["uuid-x"]
        assert db.get_blocked_stash_ids(m) == set()


# ======================================================================
# Finding 5: fold tags scenes; other-endpoint performers get the member
# anchor; builds never place a face that gained a membership
# ======================================================================

class TestBuildPlacement:
    def test_faces_folded_into_assigned_group_are_tagged(self, db, svc):
        p, q = Person(65000), Person(66000)
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        db.add_faces_to_cluster(m, [p.face(db, 100 + i, match_id="uuid-x") for i in range(3)])
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, [q.face(db, 200 + i) for i in range(3)])
        stash = FakeStash(performer_stash_ids={"42": ["uuid-x"]})
        svc.assign_performer(o, "42", "Jane", stash)
        stash.writes.clear()

        res = svc.build_clusters(incremental=True, auto_tag=True, stash_client=stash)

        assert res["groups_folded"] == 1
        assert sorted(sid for sid, _ in stash.writes) == ["100", "101", "102"]
        assert res["tagged_scenes"] == 3

    @pytest.mark.parametrize("incremental", [True, False])
    def test_performer_linked_only_elsewhere_absorbs_matching_faces(self, db, svc, incremental):
        p = Person(67000)
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, [p.face(db, 1 + i, match_id="uuid-x") for i in range(3)])
        stash = FakeStash(performer_stash_ids={"42": ["tpdb-y"]})
        svc.assign_performer(o, "42", "Jane", stash)
        assert db.get_cluster_stash_ids(o) == ["tpdb-y"]

        new = [p.face(db, 10 + i, match_id="uuid-x") for i in range(2)]
        svc.build_clusters(incremental=incremental, stash_client=stash)

        assert db.list_face_clusters("matched") == []
        assert set(new) <= _members(db, o)
        assert db.get_cluster_stash_ids(o) == ["tpdb-y", "uuid-x"]

    @pytest.mark.parametrize("incremental", [True, False])
    def test_face_banned_after_pool_read_is_not_placed(self, db, svc, monkeypatch, incremental):
        p = Person(68000)
        faces = [p.face(db, 1 + i, match_id="uuid-x") for i in range(3)]
        orig = db.get_face_rejections

        def ban_mid_build(face_ids=None):
            # a writer outside the service bans a pooled face after the pool read
            bid = db.create_face_cluster(status="banned", pinned=True)
            db.add_faces_to_cluster(bid, [faces[0]])
            monkeypatch.setattr(db, "get_face_rejections", orig)
            return orig(face_ids)

        monkeypatch.setattr(db, "get_face_rejections", ban_mid_build)
        svc.build_clusters(incremental=incremental)
        assert [m["status"] for m in db.get_face_cluster_membership(faces[0])] == ["banned"]
        assert _max_memberships(db) == 1

    def test_group_pinned_after_listing_is_not_dissolved(self, db, svc, monkeypatch):
        p, q = Person(69000), Person(69500)
        o = db.create_face_cluster()
        o_faces = [p.face(db, 1 + i) for i in range(3)]
        db.add_faces_to_cluster(o, o_faces)
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-m"])
        m_faces = [q.face(db, 10 + i, match_id="uuid-m") for i in range(3)]
        db.add_faces_to_cluster(m, m_faces)
        orig = db.get_clusters_by_status
        pinned = []

        def list_then_curate(*statuses):
            rows = orig(*statuses)
            if not pinned:          # a curation outside the build pins both
                db.update_face_cluster(o, pinned=True, name="Kept")
                db.update_face_cluster(m, pinned=True)
                pinned.append(True)
            return rows

        monkeypatch.setattr(db, "get_clusters_by_status", list_then_curate)
        res = svc.build_clusters()
        assert db.get_face_cluster(o)["name"] == "Kept"
        assert _members(db, o) == set(o_faces)
        assert _members(db, m) == set(m_faces)
        assert res["dissolved"] == 0

    def test_dissolved_counts_only_groups_the_build_removed(self, db, svc, monkeypatch):
        p = Person(70000)
        o1, o2 = db.create_face_cluster(), db.create_face_cluster()
        db.add_faces_to_cluster(o1, [p.face(db, 1 + i) for i in range(3)])
        db.add_faces_to_cluster(o2, [_add_face(db, scene_id=90)])
        orig = db.get_clusters_by_status
        done = []

        def list_then_user_deletes(*statuses):
            rows = orig(*statuses)
            if not done:
                db.delete_face_cluster(o2)
                done.append(True)
            return rows

        monkeypatch.setattr(db, "get_clusters_by_status", list_then_user_deletes)
        assert svc.build_clusters()["dissolved"] == 1

    def test_fold_skips_duplicate_pinned_after_scan(self, db, svc, monkeypatch):
        p = Person(70500)
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True,
                                   stash_ids=["uuid-x"])
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
        db.add_faces_to_cluster(m, [p.face(db, 1 + i, match_id="uuid-x") for i in range(2)])
        orig = db.get_duplicate_stash_id_groups

        def scan_then_curate():
            out = orig()
            db.update_face_cluster(m, pinned=True, name="Mine")
            return out

        monkeypatch.setattr(db, "get_duplicate_stash_id_groups", scan_then_curate)
        assert svc.fold_duplicate_groups() == 0
        assert db.get_face_cluster(m)["name"] == "Mine"
        assert _members(db, a) == set()


# ======================================================================
# Finding 6: the ejected face never votes on the identity it is rejected from
# ======================================================================

class TestEjectIdentity:
    def test_wrong_face_ejected_from_unlinked_group_joins_its_own_group(self, db, svc):
        bob, jane = Person(71000), Person(72000)
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        f = jane.face(db, 1, match_id="uuid-jane")
        db.add_faces_to_cluster(a, [bob.face(db, 2), bob.face(db, 3), f])
        m = db.create_face_cluster(status="matched", stash_ids=["uuid-jane"])
        db.add_faces_to_cluster(m, [jane.face(db, 10 + i, match_id="uuid-jane") for i in range(2)])

        svc.eject_faces(a, [f])
        assert db.get_face_rejections([f])[f]["stash_ids"] == set()
        svc.build_clusters(incremental=True)
        assert _memberships(db, f) == [m]

    def test_ejected_majority_does_not_ban_its_own_identity(self, db, svc):
        bob, jane = Person(73000), Person(74000)
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        wrong = [jane.face(db, 1 + i, match_id="uuid-jane") for i in range(2)]
        db.add_faces_to_cluster(a, [bob.face(db, 5), *wrong])
        j = db.create_face_cluster(status="assigned", performer_id="77", pinned=True,
                                   stash_ids=["uuid-jane"])
        db.add_faces_to_cluster(j, [jane.face(db, 20 + i, match_id="uuid-jane") for i in range(2)])

        svc.eject_faces(a, wrong)
        svc.build_clusters()
        assert all(_memberships(db, f) == [j] for f in wrong)

    def test_remaining_members_still_define_the_rejected_identity(self, db, svc):
        p = Person(75000)
        # round 4: the ejected face's own match is never rejected (unlinked
        # performer), so it carries a different match here
        faces = [p.face(db, 1, match_id="uuid-y"),
                 *(p.face(db, 2 + i, match_id="uuid-x") for i in range(2))]
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        db.add_faces_to_cluster(a, faces)
        svc.eject_faces(a, [faces[0]])
        assert db.get_face_rejections([faces[0]])[faces[0]]["stash_ids"] == {"uuid-x"}


# ======================================================================
# Findings 5 / 6 / 8 / 10: curation is serialized with builds
# ======================================================================

class _PausedBuild:
    """Run a build on a worker thread and hold it at a hook until released."""

    def __init__(self, svc, db, monkeypatch, hook="get_clusters_by_status", **build_kw):
        self.reached, self.go = threading.Event(), threading.Event()
        self.error = None
        orig = getattr(db, hook)
        owner = []

        def hooked(*a, **k):
            out = orig(*a, **k)
            if not owner:
                owner.append(threading.get_ident())
            if threading.get_ident() == owner[0] and not self.reached.is_set():
                self.reached.set()
                assert self.go.wait(15)
            return out

        monkeypatch.setattr(db, hook, hooked)
        self._t = threading.Thread(target=self._run, args=(svc, build_kw), daemon=True)

    def _run(self, svc, kw):
        try:
            self.result = svc.build_clusters(**kw)
        except Exception as e:  # pragma: no cover - surfaced in __exit__
            self.error = e

    def __enter__(self):
        self._t.start()
        assert self.reached.wait(15), "build never reached the hook"
        return self

    def __exit__(self, *exc):
        self.go.set()
        self._t.join(15)
        if self.error:
            raise self.error
        return False


@pytest.fixture
def fast_lock(monkeypatch):
    monkeypatch.setattr(fcs, "LOCK_WAIT_SEC", 0.5, raising=False)


class TestCurationDuringBuild:
    def test_assign_during_build_is_refused_and_nothing_is_tagged(self, db, svc, monkeypatch, fast_lock):
        p = Person(76000)
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, [p.face(db, 1 + i) for i in range(3)])
        stash = FakeStash(performer_stash_ids={"42": []})
        with _PausedBuild(svc, db, monkeypatch):
            with pytest.raises(BuildInProgress):
                svc.assign_performer(o, "42", "Jane", stash)
        assert stash.writes == [] and stash.reads == []
        # after the build the same assign works and sticks
        [grp] = db.list_face_clusters("open")
        svc.assign_performer(grp["id"], "42", "Jane", stash)
        svc.build_clusters()
        assert db.get_face_cluster(grp["id"])["status"] == "assigned"

    def test_merge_during_full_build_is_refused(self, db, svc, monkeypatch, fast_lock):
        p, q = Person(77000), Person(78000)
        o1, o2 = db.create_face_cluster(), db.create_face_cluster()
        db.add_faces_to_cluster(o1, [p.face(db, 1 + i) for i in range(3)])
        db.add_faces_to_cluster(o2, [q.face(db, 10 + i) for i in range(3)])
        with _PausedBuild(svc, db, monkeypatch):
            with pytest.raises(BuildInProgress):
                svc.merge_clusters([o2], o1)
        a, b = (c["id"] for c in db.list_face_clusters("open"))
        svc.merge_clusters([b], a)
        svc.build_clusters()
        assert db.get_face_cluster(a) is not None and db.get_face_cluster(a)["pinned"] == 1

    @pytest.mark.parametrize("op", ["split", "eject", "ban", "unban", "update", "ignore", "delete"])
    def test_other_curation_during_build_is_refused_without_side_effects(
            self, db, svc, monkeypatch, fast_lock, op):
        p = Person(79000)
        faces = [p.face(db, 1 + i) for i in range(4)]
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, faces)
        q = Person(79500)
        for i in range(3):          # a pool, so the build reads rejections
            q.face(db, 50 + i)
        call = {
            "split": lambda: svc.split_cluster(o, [faces[0]]),
            "eject": lambda: svc.eject_faces(o, [faces[0]]),
            "ban": lambda: svc.ban_faces([faces[0]]),
            "unban": lambda: svc.unban_faces([faces[0]]),
            "update": lambda: svc.update_cluster(o, name="Mine"),
            "ignore": lambda: svc.ignore_cluster(o),
            "delete": lambda: svc.delete_cluster(o),
        }[op]
        with _PausedBuild(svc, db, monkeypatch, hook="get_face_rejections", incremental=True):
            with pytest.raises(BuildInProgress):
                call()
            assert db.get_face_rejections([faces[0]]) == {}
        assert _max_memberships(db) == 1
        assert {len(_memberships(db, f)) for f in faces} == {1}

    def test_curation_waits_for_another_curation_then_runs(self, db, svc, monkeypatch):
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, [_add_face(db, scene_id=1)])
        held, release = threading.Event(), threading.Event()

        def hold():
            with fcs.curation_lock():
                held.set()
                release.wait(5)

        t = threading.Thread(target=hold, daemon=True)
        t.start()
        assert held.wait(5)
        threading.Timer(0.2, release.set).start()
        assert svc.update_cluster(o, name="After")["name"] == "After"   # waited, did not fail
        t.join(5)

    def test_build_waits_for_inflight_curation(self, db, svc):
        _add_face(db, scene_id=1)
        held, release = threading.Event(), threading.Event()

        def hold():
            with fcs.curation_lock():
                held.set()
                release.wait(5)

        t = threading.Thread(target=hold, daemon=True)
        t.start()
        assert held.wait(5)
        threading.Timer(0.2, release.set).start()
        assert svc.build_clusters(incremental=True)["faces_new"] == 1
        t.join(5)

    def test_assign_racing_merge_is_serialized(self, db, svc, monkeypatch):
        """#8 TOCTOU: a source re-assigned between merge's checks and its move."""
        p = Person(80000)
        tgt = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        src = db.create_face_cluster()
        db.add_faces_to_cluster(tgt, [p.face(db, 1)])
        db.add_faces_to_cluster(src, [p.face(db, 2)])
        orig = db.move_faces_to_cluster
        box = {}

        def assign_other():
            try:
                box["res"] = svc.assign_performer(src, "99", "Other", FakeStash())
            except Exception as e:
                box["err"] = e

        def move_with_race(sources, target):
            t = threading.Thread(target=assign_other, daemon=True)
            t.start()
            t.join(0.3)   # without serialization the assign lands right here
            box["t"] = t
            return orig(sources, target)

        monkeypatch.setattr(db, "move_faces_to_cluster", move_with_race)
        svc.merge_clusters([src], tgt)
        box["t"].join(5)
        assert isinstance(box.get("err"), ClusterNotFound)     # it ran after the merge
        assert db.get_face_cluster(tgt)["performer_id"] == "42"


# ---------------------------------------------------------------- router

@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: db)
    monkeypatch.setattr(face_clusters_router, "_service", FaceClusterService(db))
    monkeypatch.setattr(face_clusters_router, "_optional_stash_client", lambda: None)
    return TestClient(_make_app())


class TestRouterDuringBuild:
    def test_every_curation_endpoint_returns_409_while_building(self, db, client, monkeypatch):
        o = db.create_face_cluster()
        f = _add_face(db, scene_id=1)
        db.add_faces_to_cluster(o, [f])
        created = []
        stash = NS(create_performer_sync=lambda *a, **k: created.append(a) or {"id": "9", "name": "N"},
                   get_scene_performer_ids_sync=lambda sid: {"performer_ids": []},
                   update_scene_performers_sync=lambda *a: None)
        monkeypatch.setattr(face_clusters_router, "get_stash_client", lambda: stash)
        calls = [
            ("patch", f"/face-groups/{o}", {"name": "x"}),
            ("post", f"/face-groups/{o}/assign", {"performer_id": "1", "performer_name": "A"}),
            ("post", f"/face-groups/{o}/create-and-assign", {"name": "New"}),
            ("post", f"/face-groups/{o}/ignore", None),
            ("delete", f"/face-groups/{o}", None),
            ("post", "/face-groups/merge", {"source_ids": [o], "target_id": o}),
            ("post", f"/face-groups/{o}/split", {"face_ids": [f]}),
            ("post", f"/face-groups/{o}/eject", {"face_ids": [f], "eject_mode": "ban"}),
            ("post", "/face-groups/unban", {"face_ids": [f]}),
        ]
        assert fcs._BUILD_LOCK.acquire(blocking=False)
        try:
            codes = {}
            for method, url, body in calls:
                kw = {"json": body} if body is not None else {}
                codes[url + method] = client.request(method.upper(), url, **kw).status_code
        finally:
            fcs._BUILD_LOCK.release()
        assert set(codes.values()) == {409}, codes
        assert created == []                   # no orphan performer
        assert _memberships(db, f) == [o]

    def test_patch_missing_group_is_404(self, client):
        assert client.patch("/face-groups/12345", json={"name": "x"}).status_code == 404

    def test_split_without_member_faces_is_400(self, db, client):
        o = db.create_face_cluster()
        db.add_faces_to_cluster(o, [_add_face(db, scene_id=1)])
        other = _add_face(db, scene_id=2)
        assert client.post(f"/face-groups/{o}/split", json={"face_ids": [other]}).status_code == 400

    def test_ban_via_eject_only_bans_members_of_that_group(self, db, client):
        p = Person(81000)
        o, other = db.create_face_cluster(), db.create_face_cluster()
        mine, theirs = p.face(db, 1), p.face(db, 2)
        db.add_faces_to_cluster(o, [mine, p.face(db, 3)])
        db.add_faces_to_cluster(other, [theirs])
        r = client.post(f"/face-groups/{o}/eject", json={"face_ids": [mine, theirs], "eject_mode": "ban"})
        assert r.status_code == 200 and r.json() == {"banned": 1}
        assert _memberships(db, theirs) == [other]


# ======================================================================
# Findings 2 / 3 / 10: identify_scene end to end (wiring + off-loop persist)
# ======================================================================

def test_identify_scene_end_to_end_persists_aligned_faces_off_the_loop(monkeypatch):
    import identification_router as ir

    rng = np.random.default_rng(5)

    def emb(seed):
        v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
        v /= np.linalg.norm(v)
        return NS(facenet=v, arcface=v)

    frames = [NS(frame_index=i, timestamp_sec=float(i), image=np.zeros((10, 10, 3), np.uint8))
              for i in range(3)]
    # frame f: face A (x=10) and, in frame 0 only, face B (x=300)
    def detect(image, min_confidence=None):
        if image.shape == (20, 20, 3):   # screenshot
            return [NS(bbox={"x": 1, "y": 1, "w": 200, "h": 200}, confidence=0.99, yaw=0.0,
                       image="ss")]
        idx = int(image[0, 0, 0])
        out = [NS(bbox={"x": 10, "y": 10, "w": 200, "h": 200}, confidence=0.99, yaw=0.0,
                  image=("A", idx))]
        if idx == 0:
            out.append(NS(bbox={"x": 300, "y": 10, "w": 200, "h": 200}, confidence=0.99,
                          yaw=0.0, image=("B", idx)))
        return out

    for fr in frames:
        fr.image = np.full((10, 10, 3), fr.frame_index, np.uint8)

    def embed_batch(images):
        return [emb(1 if (im[0] if isinstance(im, tuple) else im) == "A" else 2 if (
            isinstance(im, tuple) and im[0] == "B") else 3) for im in images]

    def recognize(face, cfg, embedding=None):
        who = face.image[0] if isinstance(face.image, tuple) else "S"
        return ({"A": [_m("sA", 0.2)], "B": [_m("sB", 0.25)], "S": [_m("sS", 0.1)]}[who],
                None, None)

    recognizer = NS(generator=NS(detect_faces=detect, get_embeddings_batch=embed_batch),
                    recognize_face_v2=recognize)

    class FakeResp:
        status_code = 200
        content = b"img"

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"findScene": {"files": [{"duration": 100.0, "width": 20, "height": 20}],
                                           "paths": {"screenshot": "http://s/ss.jpg"}}}}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return FakeResp()

        async def get(self, *a, **k):
            return FakeResp()

    async def extract(**kw):
        return NS(frames=frames, errors=[])

    async def no_images(_matches):
        return None

    seen = {}

    def fake_persist(**kw):
        seen.update(kw)
        seen["thread"] = threading.get_ident()
        return len(kw["detected_faces"])

    monkeypatch.setattr(ir, "check_ffmpeg_available", lambda: True)
    monkeypatch.setattr(ir, "_stash_url", "http://stash")
    monkeypatch.setattr(ir, "_recognizer", recognizer)
    monkeypatch.setattr(ir, "_multi_signal_matcher", None)
    monkeypatch.setattr(ir.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(ir, "extract_frames_from_stash_scene", extract)
    monkeypatch.setattr(ir, "_fetch_missing_images", no_images)
    monkeypatch.setattr(ir, "load_image", lambda b: np.zeros((20, 20, 3), np.uint8))
    monkeypatch.setattr(ir, "save_scene_fingerprint", lambda **kw: (1, None))
    monkeypatch.setattr(library_face_persist, "persist_from_identify", fake_persist)

    async def run():
        loop_thread = threading.get_ident()
        req = ir.SceneIdentifyRequest(scene_id="55", matching_mode="frequency")
        resp = await ir.identify_scene(req, None)
        return loop_thread, resp

    loop_thread, resp = asyncio.run(run())

    assert seen, "persist was not called"
    assert seen["thread"] != loop_thread                  # #10: off the event loop
    det = seen["detected_faces"]
    assert [(f, face.image) for f, face in det] == [(0, ("A", 0)), (0, ("B", 0)), (1, ("A", 1)),
                                                    (2, ("A", 2))]
    # face_matches[i] belongs to detected_faces[i]; the screenshot face is left out
    assert [[m.stashdb_id for m in ms] for ms in seen["face_matches"]] == [["sA"], ["sB"], ["sA"], ["sA"]]
    anchors = library_face_persist.face_person_assignment(
        seen["persons"], len(det), face_frames=[f for f, _ in det], face_matches=seen["face_matches"])
    got = {i: seen["persons"][k].best_match.stashdb_id for i, k in anchors.items()}
    assert got == {0: "sA", 1: "sB", 2: "sA", 3: "sA"}
    assert resp.persons


# ======================================================================
# Finding 9 leftovers: banned groups, null performer ids, the merge action
# ======================================================================

_PLUGIN_JS = __import__("pathlib").Path(__file__).resolve().parents[2] / "plugin" / "stash-sense-face-groups.js"

_HARNESS3 = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const window = { StashSense: { getRoute: () => ({ type: 'other' }), onNavigate() {}, onLeavePlugin() {},
  PLUGIN_NAME: 'Stash Sense', escapeHtml: s => s, getSettings: async () => ({}),
  runPluginOperation: async () => ({}) } };
const sandbox = { window, console, setTimeout, clearTimeout, clearInterval };
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'stash-sense-face-groups.js' });
const FG = window.StashSenseFaceGroups;
(async () => {
  const cases = JSON.parse(process.argv[3]);
  const out = [];
  for (const [fn, args] of cases) {
    if (fn === 'performMerge') {
      const [selected, list, confirmAnswer, failWith] = args;
      const calls = [], alerts = [], confirms = [];
      const api = { merge: async (s, t) => { calls.push([s, t]); if (failWith) throw new Error(failWith); return {}; } };
      const r = await FG.performMerge(new Set(selected), new Map(list.map(c => [c.id, c])), api,
        m => { confirms.push(m); return confirmAnswer; }, m => alerts.push(m));
      out.push({ r, calls, alerts, confirms: confirms.length });
    } else {
      out.push(FG[fn](...args));
    }
  }
  console.log(JSON.stringify(out));
})();
"""


def _node3(tmp_path, cases):
    import json
    import shutil
    import subprocess
    if shutil.which("node") is None:
        pytest.skip("node is not available")
    h = tmp_path / "h3.js"
    h.write_text(_HARNESS3, encoding="utf-8")
    proc = subprocess.run(["node", str(h), str(_PLUGIN_JS), json.dumps(cases)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _cj(id_, status="open", n=3, pid=None):
    return {"id": id_, "status": status, "face_count": n, "performer_id": pid,
            "performer_name": pid, "name": None}


class TestPluginRound3:
    def test_banned_groups_are_never_merge_targets(self, tmp_path):
        [r] = _node3(tmp_path, [["chooseMergeTarget", [[_cj(1, "banned", 30), _cj(2, "open", 3)]]]])
        assert r["targetId"] is None and "anned" in r["error"]

    def test_assigned_without_performer_id_does_not_conflict(self, tmp_path):
        [r] = _node3(tmp_path, [["chooseMergeTarget", [[_cj(1, "assigned", 2, None),
                                                        _cj(2, "assigned", 5, "p1")]]]])
        assert r["error"] is None and r["targetId"] == 2

    def test_banned_cards_hidden_and_actionless(self, tmp_path):
        vis, a, b, c = _node3(tmp_path, [
            ["visibleClusters", [[_cj(1, "banned"), _cj(2, "open"), _cj(3, "assigned", pid="p")]]],
            ["cardActions", [_cj(1, "banned")]],
            ["cardActions", [_cj(2, "open")]],
            ["cardActions", [_cj(3, "assigned", pid="p")]],
        ])
        assert [x["id"] for x in vis] == [2, 3]
        assert (a, b, c) == ("banned", "curate", "performer")

    def test_merge_action_sends_computed_target_to_api(self, tmp_path):
        lst = [_cj(1, "open", 3), _cj(2, "assigned", 1, "p1"), _cj(3, "open", 9)]
        ok, declined, failed, refused = _node3(tmp_path, [
            ["performMerge", [[1, 2, 3], lst, True, None]],
            ["performMerge", [[1, 3], lst, False, None]],
            ["performMerge", [[1, 3], lst, True, "a face group build is running"]],
            ["performMerge", [[1, 2], [_cj(1, "open"), _cj(2, "banned")], True, None]],
        ])
        assert ok["calls"] == [[[1, 3], 2]] and ok["r"]["merged"] is True
        assert declined["calls"] == [] and declined["r"]["merged"] is False
        assert failed["r"]["merged"] is False and "build is running" in failed["alerts"][0]
        assert refused["calls"] == [] and refused["confirms"] == 0 and refused["alerts"]

    def test_merge_button_uses_perform_merge(self):
        src = _PLUGIN_JS.read_text(encoding="utf-8")
        block = src[src.index("const mergeBtn = container.querySelector('#fg-merge-btn');"):]
        block = block[: block.index("});\n") + 4]
        assert "performMerge(selectedClusters, clustersById, FaceGroupsAPI" in block
