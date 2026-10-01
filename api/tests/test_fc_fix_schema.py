"""Schema v12 + DB-layer fixes for face clusters (review findings #4-#8)."""
import inspect
import sqlite3

import numpy as np
import pytest

import recommendations_db as rdb
from recommendations_db import RecommendationsDB


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "test.db")


def _emb(seed: int, dim: int = 512) -> bytes:
    return np.random.default_rng(seed).normal(0, 1, dim).astype(np.float32).tobytes()


_seq = [0]


def _add_face(db, scene_id=1, frame=0, x=10, y=10, w=40, h=40, ts=1.0,
              match_id=None, match_name=None, crop_path=None):
    _seq[0] += 1
    return db.add_library_face(
        stash_scene_id=scene_id, frame_index=frame, timestamp_sec=ts,
        bbox={"x": x, "y": y, "w": w, "h": h}, det_confidence=0.9, yaw=0.0,
        facenet_emb=_emb(_seq[0]), arcface_emb=_emb(10_000 + _seq[0]),
        crop_path=crop_path, best_match_id=match_id, best_match_name=match_name,
        best_match_confidence=0.8 if match_id else None,
    )


def _face_dict(frame=0, x=10, y=10, w=40, h=40, ts=..., crop_path=None,
               match_id=None, match_name=None, conf=None, db_version=None, seed=None):
    """Face dict for upsert. Default timestamp is frame * 1.0s so frames differ in time.

    The embedding comes from `seed` (default frame * 1000 + x); pass the
    original face's seed for a re-detection of the same person.
    """
    if ts is ...:
        ts = float(frame)
    if seed is None:
        seed = frame * 1000 + int(x)
    return {
        "frame_index": frame, "timestamp_sec": ts,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "det_confidence": 0.95, "yaw": 1.0,
        "facenet_emb": _emb(seed), "arcface_emb": _emb(7 + seed),
        "crop_path": crop_path, "best_match_id": match_id, "best_match_name": match_name,
        "best_match_confidence": conf, "db_version": db_version,
    }


def _version(db):
    with db._connection() as conn:
        return conn.execute("SELECT version FROM schema_version").fetchone()[0]


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _members(db, cid):
    with db._connection() as conn:
        return sorted(r[0] for r in conn.execute(
            "SELECT face_id FROM face_cluster_members WHERE cluster_id = ?", (cid,)))


def _all_memberships(db, fid):
    with db._connection() as conn:
        return sorted(r[0] for r in conn.execute(
            "SELECT cluster_id FROM face_cluster_members WHERE face_id = ?", (fid,)))


# ==================== Schema ====================


def test_fresh_db_v13_has_tables_and_pinned_column(db):
    assert rdb.SCHEMA_VERSION == 14
    assert _version(db) == rdb.SCHEMA_VERSION
    with db._connection() as conn:
        tables = _tables(conn)
        assert {"face_cluster_stash_ids", "face_rejections", "face_cluster_merge_log"} <= tables
        assert "pinned" in _columns(conn, "face_clusters")
        idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_fc_stash_ids_stash" in idx
    cid = db.create_face_cluster()
    assert db.get_face_cluster(cid)["pinned"] == 0


def _downgrade_to_v11(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        ALTER TABLE face_clusters DROP COLUMN pinned;
        DROP TABLE face_cluster_stash_ids;
        DROP TABLE face_rejections;
        UPDATE schema_version SET version = 11;
    """)
    conn.commit()
    conn.close()


def _downgrade_to_v10(path):
    _downgrade_to_v11(path)
    conn = sqlite3.connect(path)
    conn.executescript("""
        DROP TABLE face_cluster_merge_log;
        UPDATE schema_version SET version = 10;
    """)
    conn.commit()
    conn.close()


def _downgrade_to_v9(path):
    _downgrade_to_v10(path)
    conn = sqlite3.connect(path)
    conn.executescript("""
        DROP TABLE face_cluster_members;
        DROP TABLE face_clusters;
        DROP TABLE library_faces;
        UPDATE schema_version SET version = 9;
    """)
    conn.commit()
    conn.close()


def _raw_face(conn, scene, frame=0):
    cur = conn.execute(
        """INSERT INTO library_faces (stash_scene_id, frame_index, timestamp_sec, bbox_x, bbox_y,
           bbox_w, bbox_h, det_confidence, facenet_emb, arcface_emb)
           VALUES (?, ?, 1.0, 10, 10, 40, 40, 0.9, ?, ?)""",
        (scene, frame, _emb(scene), _emb(scene + 1)))
    return cur.lastrowid


def _raw_cluster(conn, status="open", performer_id=None, name=None):
    cur = conn.execute(
        "INSERT INTO face_clusters (name, status, performer_id) VALUES (?, ?, ?)",
        (name, status, performer_id))
    return cur.lastrowid


def _raw_member(conn, cid, fid):
    conn.execute("INSERT INTO face_cluster_members (cluster_id, face_id) VALUES (?, ?)", (cid, fid))


def test_migrate_v11_to_latest(tmp_path):
    path = tmp_path / "m.db"
    RecommendationsDB(path)
    _downgrade_to_v11(path)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    f = {i: _raw_face(conn, i) for i in range(1, 10)}
    m_small = _raw_cluster(conn, "matched", "uuid-x")      # 1 member
    _raw_member(conn, m_small, f[1])
    m_big = _raw_cluster(conn, "matched", "uuid-x")        # 2 members -> kept
    _raw_member(conn, m_big, f[2])
    _raw_member(conn, m_big, f[3])
    # a merge target: v12 pins it, so v13 (which dissolves unpinned matched
    # groups seeded from pre-fix anchors) keeps it
    conn.execute("INSERT INTO face_cluster_merge_log (source_cluster_id, target_cluster_id) VALUES (998, ?)",
                 (m_big,))
    m_other = _raw_cluster(conn, "matched", "uuid-y")
    _raw_member(conn, m_other, f[9])
    assigned = _raw_cluster(conn, "assigned", "42")
    _raw_member(conn, assigned, f[4])
    dup_open = _raw_cluster(conn, "open")
    _raw_member(conn, dup_open, f[4])                      # extra membership -> removed
    _raw_member(conn, dup_open, f[5])
    merge_target = _raw_cluster(conn, "open")
    _raw_member(conn, merge_target, f[6])
    conn.execute("INSERT INTO face_cluster_merge_log (source_cluster_id, target_cluster_id) VALUES (999, ?)",
                 (merge_target,))
    named = _raw_cluster(conn, "open", name="Somebody")
    _raw_member(conn, named, f[7])
    plain_open = _raw_cluster(conn, "open")
    _raw_member(conn, plain_open, f[8])
    ghost_banned = _raw_cluster(conn, "banned")
    ignored = _raw_cluster(conn, "ignored")
    conn.commit()
    conn.close()

    db = RecommendationsDB(path)
    assert _version(db) == rdb.SCHEMA_VERSION

    # v12: stash ids moved, performer_id NULL, one cluster for uuid-x with union
    matched = db.get_clusters_by_status("matched")
    by_sid = {}
    for c in matched:
        assert c["performer_id"] is None
        for s in db.get_cluster_stash_ids(c["id"]):
            by_sid.setdefault(s, []).append(c["id"])
    # v13: the unpinned matched group (uuid-y) is dissolved, its face pooled
    assert by_sid == {"uuid-x": [m_big]}
    assert db.get_face_cluster(m_small) is None
    assert db.get_face_cluster(m_other) is None
    assert _all_memberships(db, f[9]) == []
    assert _members(db, m_big) == sorted([f[1], f[2], f[3]])

    # pinned flags
    pinned = {cid: db.get_face_cluster(cid)["pinned"]
              for cid in (m_big, assigned, dup_open, merge_target, named, plain_open, ignored)}
    assert pinned == {m_big: 1, assigned: 1, dup_open: 0, merge_target: 1,
                      named: 1, plain_open: 0, ignored: 1}

    # duplicate open membership removed; assigned kept
    assert _all_memberships(db, f[4]) == [assigned]
    assert _members(db, dup_open) == [f[5]]
    # ghost banned deleted
    assert db.get_face_cluster(ghost_banned) is None
    # assigned performer id untouched
    assert db.get_face_cluster(assigned)["performer_id"] == "42"


def test_migrate_v10_to_latest(tmp_path):
    path = tmp_path / "m10.db"
    RecommendationsDB(path)
    _downgrade_to_v10(path)
    conn = sqlite3.connect(path)
    fid = _raw_face(conn, 1)
    cid = _raw_cluster(conn, "matched", "uuid-z", name="Zed")
    _raw_member(conn, cid, fid)
    conn.commit()
    conn.close()

    db = RecommendationsDB(path)
    assert _version(db) == rdb.SCHEMA_VERSION
    with db._connection() as conn:
        assert {"face_cluster_merge_log", "face_cluster_stash_ids", "face_rejections"} <= _tables(conn)
        assert {"pinned", "stash_ids_synced"} <= _columns(conn, "face_clusters")
    # an unpinned matched group seeded from pre-v13 anchors is dissolved
    assert db.get_face_cluster(cid) is None
    assert _all_memberships(db, fid) == []


def test_migrate_v9_to_latest(tmp_path):
    path = tmp_path / "m9.db"
    RecommendationsDB(path)
    _downgrade_to_v9(path)
    db = RecommendationsDB(path)
    assert _version(db) == rdb.SCHEMA_VERSION
    with db._connection() as conn:
        assert {"library_faces", "face_clusters", "face_cluster_members", "face_cluster_merge_log",
                "face_cluster_stash_ids", "face_rejections"} <= _tables(conn)
        assert "pinned" in _columns(conn, "face_clusters")
    fid = _add_face(db)
    cid = db.create_face_cluster(pinned=True, stash_ids=["a"])
    assert db.add_faces_to_cluster(cid, [fid]) == 1
    assert db.get_face_cluster(cid)["pinned"] == 1


def test_reopen_latest_is_noop(tmp_path):
    path = tmp_path / "r.db"
    db = RecommendationsDB(path)
    cid = db.create_face_cluster(status="matched", stash_ids=["s1"])
    db2 = RecommendationsDB(path)
    assert db2.get_cluster_stash_ids(cid) == ["s1"]


# ==================== iter / unassigned ====================


def test_iter_unassigned_excludes_faces_in_ignored_banned_assigned_open(db):
    fids = {s: _add_face(db, scene_id=i + 1)
            for i, s in enumerate(["open", "matched", "assigned", "ignored", "banned", "free"])}
    for status in ["open", "matched", "assigned", "ignored", "banned"]:
        cid = db.create_face_cluster(status=status)
        db.add_faces_to_cluster(cid, [fids[status]])
    ids = [f["id"] for batch in db.iter_library_faces(unassigned_only=True) for f in batch]
    assert ids == [fids["free"]]
    assert db.get_unassigned_face_ids() == [fids["free"]]
    all_ids = [f["id"] for batch in db.iter_library_faces() for f in batch]
    assert sorted(all_ids) == sorted(fids.values())


def test_iter_returns_best_match_name(db):
    fid = _add_face(db, match_id="uuid-1", match_name="Jane")
    for unassigned in (False, True):
        rows = [f for batch in db.iter_library_faces(unassigned_only=unassigned) for f in batch]
        assert rows[0]["id"] == fid
        assert rows[0]["best_match_name"] == "Jane"
        assert rows[0]["best_match_id"] == "uuid-1"
        assert set(rows[0]) == {"id", "facenet_emb", "arcface_emb", "best_match_id",
                                "best_match_name", "stash_scene_id"}


def test_iter_batches(db):
    for i in range(5):
        _add_face(db, scene_id=i + 1)
    batches = list(db.iter_library_faces(batch_size=2))
    assert [len(b) for b in batches] == [2, 2, 1]


def test_only_one_iter_library_faces_definition():
    assert inspect.getsource(RecommendationsDB).count("def iter_library_faces") == 1


# ==================== add_faces_to_cluster ====================


def test_add_faces_missing_cluster_raises_value_error(db):
    fid = _add_face(db)
    with pytest.raises(ValueError, match="cluster 12345 not found"):
        db.add_faces_to_cluster(12345, [fid])


def test_add_faces_bad_face_id_raises_integrity_error_and_inserts_nothing(db):
    fid = _add_face(db)
    cid = db.create_face_cluster()
    with pytest.raises(sqlite3.IntegrityError):
        db.add_faces_to_cluster(cid, [fid, 999_999])
    assert _members(db, cid) == []


def test_add_faces_returns_inserted_count_excluding_duplicates(db):
    f1, f2 = _add_face(db, scene_id=1), _add_face(db, scene_id=2)
    cid = db.create_face_cluster()
    assert db.add_faces_to_cluster(cid, []) == 0
    assert db.add_faces_to_cluster(cid, [f1]) == 1
    assert db.add_faces_to_cluster(cid, [f1, f2, f2]) == 1
    assert _members(db, cid) == sorted([f1, f2])


# ==================== move_faces_to_cluster ====================


def test_move_faces_missing_target_raises_and_sources_intact(db):
    f1 = _add_face(db)
    src = db.create_face_cluster(stash_ids=["s"])
    db.add_faces_to_cluster(src, [f1])
    with pytest.raises(ValueError, match="target cluster 777 not found"):
        db.move_faces_to_cluster([src], 777)
    assert db.get_face_cluster(src) is not None
    assert _members(db, src) == [f1]
    assert db.get_cluster_stash_ids(src) == ["s"]
    assert db.get_cluster_merge_map() == {}


def test_move_faces_missing_source_raises_nothing_changed(db):
    f1, f2 = _add_face(db, scene_id=1), _add_face(db, scene_id=2)
    src = db.create_face_cluster()
    tgt = db.create_face_cluster()
    db.add_faces_to_cluster(src, [f1])
    db.add_faces_to_cluster(tgt, [f2])
    with pytest.raises(ValueError, match="source clusters not found"):
        db.move_faces_to_cluster([src, 555], tgt)
    assert _members(db, src) == [f1]
    assert _members(db, tgt) == [f2]
    assert db.get_face_cluster(tgt)["pinned"] == 0
    assert db.get_cluster_merge_map() == {}
    with pytest.raises(ValueError, match="no source clusters"):
        db.move_faces_to_cluster([tgt], tgt)
    with pytest.raises(ValueError, match="no source clusters"):
        db.move_faces_to_cluster([], tgt)


def test_move_faces_moves_members_stash_ids_logs_merge_pins_target_deletes_sources(db):
    f1, f2, f3 = (_add_face(db, scene_id=i) for i in (1, 2, 3))
    s1 = db.create_face_cluster(status="matched", stash_ids=["uuid-a"])
    s2 = db.create_face_cluster(stash_ids=["uuid-b"])
    tgt = db.create_face_cluster(stash_ids=["uuid-a"])
    db.add_faces_to_cluster(s1, [f1, f2])
    db.add_faces_to_cluster(s2, [f2, f3])
    res = db.move_faces_to_cluster([s1, s2], tgt)
    assert res == {"faces_moved": 3, "sources_deleted": [s1, s2]}
    assert _members(db, tgt) == sorted([f1, f2, f3])
    assert db.get_cluster_stash_ids(tgt) == ["uuid-a", "uuid-b"]
    assert db.get_face_cluster(tgt)["pinned"] == 1
    assert db.get_face_cluster(s1) is None and db.get_face_cluster(s2) is None
    assert db.get_cluster_merge_map() == {s1: tgt, s2: tgt}
    with db._connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM face_cluster_members WHERE cluster_id IN (?, ?)",
                            (s1, s2)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM face_cluster_stash_ids WHERE cluster_id IN (?, ?)",
                            (s1, s2)).fetchone()[0] == 0


def test_move_faces_ignores_self_in_sources(db):
    f1, f2 = _add_face(db, scene_id=1), _add_face(db, scene_id=2)
    src = db.create_face_cluster()
    tgt = db.create_face_cluster()
    db.add_faces_to_cluster(src, [f1])
    db.add_faces_to_cluster(tgt, [f2])
    res = db.move_faces_to_cluster([src, tgt, src], tgt)
    assert res["sources_deleted"] == [src]
    assert db.get_face_cluster(tgt) is not None
    assert _members(db, tgt) == sorted([f1, f2])
    assert tgt not in db.get_cluster_merge_map()


def test_record_cluster_merge_skips_self(db):
    a = db.create_face_cluster()
    b = db.create_face_cluster()
    db.record_cluster_merge([a, b], b)
    assert db.get_cluster_merge_map() == {a: b}


def test_resolve_cluster_id_follows_chain_and_none_for_deleted_target(db):
    a, b, c = (db.create_face_cluster() for _ in range(3))
    for cid in (a, b, c):
        db.add_faces_to_cluster(cid, [_add_face(db, scene_id=100 + cid)])
    assert db.resolve_cluster_id(a) == a
    db.move_faces_to_cluster([a], b)
    db.move_faces_to_cluster([b], c)
    assert db.resolve_cluster_id(a) == c
    assert db.resolve_cluster_id(b) == c
    assert db.resolve_cluster_id(c) == c
    db.delete_face_cluster(c)
    assert db.resolve_cluster_id(a) is None
    assert db.resolve_cluster_id(4242) is None


def test_merge_map_cycle_guard(db):
    db.record_cluster_merge([1], 2)
    db.record_cluster_merge([2], 1)
    m = db.get_cluster_merge_map()
    assert set(m) == {1, 2}
    assert db.resolve_cluster_id(1) is None


# ==================== cluster fields / stash ids ====================


def test_create_cluster_with_stash_ids_pinned(db):
    cid = db.create_face_cluster(name="M", status="matched", performer_name="Jane",
                                 pinned=True, stash_ids=["b", "a", "a"])
    c = db.get_face_cluster(cid)
    assert c["pinned"] == 1
    assert c["performer_id"] is None
    assert db.get_cluster_stash_ids(cid) == ["a", "b"]
    cid2 = db.create_face_cluster()
    assert db.get_face_cluster(cid2)["pinned"] == 0
    assert db.get_cluster_stash_ids(cid2) == []


def test_get_face_cluster_includes_stash_ids(db):
    cid = db.create_face_cluster(stash_ids=["z", "m"])
    c = db.get_face_cluster(cid)
    assert c["stash_ids"] == ["m", "z"]
    assert "pinned" in c
    assert db.get_face_cluster(9999) is None
    listed = {x["id"]: x for x in db.list_face_clusters()}
    assert "pinned" in listed[cid]
    assert "pinned" in db.get_clusters_by_status("open")[0]


def test_update_face_cluster_pinned(db):
    cid = db.create_face_cluster()
    assert db.update_face_cluster(cid, pinned=True)
    assert db.get_face_cluster(cid)["pinned"] == 1
    assert db.update_face_cluster(cid, name="x")
    assert db.get_face_cluster(cid)["pinned"] == 1
    assert db.update_face_cluster(cid, pinned=False)
    assert db.get_face_cluster(cid)["pinned"] == 0


def test_set_cluster_stash_ids_replace_and_append(db):
    cid = db.create_face_cluster(stash_ids=["a"])
    assert db.set_cluster_stash_ids(cid, ["b", "c"]) == 2
    assert db.get_cluster_stash_ids(cid) == ["b", "c"]
    assert db.set_cluster_stash_ids(cid, ["c", "d"], replace=False) == 1
    assert db.get_cluster_stash_ids(cid) == ["b", "c", "d"]
    assert db.set_cluster_stash_ids(cid, [], replace=True) == 0
    assert db.get_cluster_stash_ids(cid) == []
    with pytest.raises(ValueError):
        db.set_cluster_stash_ids(4242, ["x"])
    other = db.create_face_cluster(stash_ids=["q"])
    assert db.get_all_cluster_stash_ids() == {other: {"q"}}


def test_stash_id_cluster_map_priority_assigned_over_ignored_over_matched_over_open(db):
    small_open = db.create_face_cluster(status="open", stash_ids=["x", "y", "z", "w"])
    big_open = db.create_face_cluster(status="open", stash_ids=["w"])
    db.add_faces_to_cluster(big_open, [_add_face(db, scene_id=1), _add_face(db, scene_id=2)])
    matched = db.create_face_cluster(status="matched", stash_ids=["x", "y"])
    assigned = db.create_face_cluster(status="assigned", performer_id="42", stash_ids=["x"])
    ignored = db.create_face_cluster(status="ignored", stash_ids=["x", "v"])
    tie_a = db.create_face_cluster(status="open", stash_ids=["t"])
    tie_b = db.create_face_cluster(status="open", stash_ids=["t"])
    m = db.get_stash_id_cluster_map()
    assert m["x"] == assigned
    assert m["y"] == matched
    assert m["z"] == small_open
    assert m["w"] == big_open
    assert m["t"] == min(tie_a, tie_b)
    # ignored groups keep their identity (so it is not re-seeded as a matched
    # suggestion), ranked right after assigned
    assert m["v"] == ignored
    assert db.get_stash_id_cluster_map(statuses=("ignored", "matched", "open"))["x"] == ignored
    assert db.get_stash_id_cluster_map(statuses=("open",))["x"] == small_open


# ==================== rejections ====================


def test_rejections_roundtrip(db):
    f1, f2 = _add_face(db, scene_id=1), _add_face(db, scene_id=2)
    n = db.add_face_rejections([f1, f2], cluster_id=7, performer_id="42", stash_ids=["u1", "u2"])
    assert n == 8
    assert db.add_face_rejections([f1], cluster_id=7) == 0
    assert db.add_face_rejections([f1], cluster_id=None, performer_id=None) == 0
    r = db.get_face_rejections()
    assert r[f1] == {"clusters": {7}, "performers": {"42"}, "stash_ids": {"u1", "u2"}}
    assert set(r) == {f1, f2}
    assert set(db.get_face_rejections([f2])) == {f2}
    assert db.get_face_rejections([]) == {}
    assert db.clear_face_rejections([f1]) == 4
    assert f1 not in db.get_face_rejections()


def test_rejections_cascade_on_face_delete(db):
    fid = _add_face(db, scene_id=3)
    db.add_face_rejections([fid], cluster_id=1)
    db.delete_library_faces_for_scene(3)
    assert db.get_face_rejections() == {}


# ==================== misc helpers ====================


def test_clear_cluster_members(db):
    cid = db.create_face_cluster()
    db.add_faces_to_cluster(cid, [_add_face(db, scene_id=1), _add_face(db, scene_id=2)])
    assert db.clear_cluster_members(cid) == 2
    assert _members(db, cid) == []
    assert db.get_face_cluster(cid) is not None


def test_delete_empty_clusters_only_given_statuses(db):
    empty_open = db.create_face_cluster(status="open", pinned=True)
    empty_matched = db.create_face_cluster(status="matched")
    empty_assigned = db.create_face_cluster(status="assigned")
    empty_banned = db.create_face_cluster(status="banned")
    full_open = db.create_face_cluster(status="open")
    db.add_faces_to_cluster(full_open, [_add_face(db)])
    # v14: pinned empty groups are kept unless asked for (see round 3, #5/#8)
    assert db.delete_empty_clusters() == 1
    assert db.get_face_cluster(empty_open) is not None
    assert db.get_face_cluster(empty_matched) is None
    assert db.delete_empty_clusters(("open",), include_pinned=True) == 1
    assert db.get_face_cluster(empty_open) is None
    assert db.get_face_cluster(empty_assigned) is not None
    assert db.get_face_cluster(empty_banned) is not None
    assert db.get_face_cluster(full_open) is not None
    assert db.delete_empty_clusters(("banned",)) == 1
    assert db.get_face_cluster(empty_banned) is None


def test_get_scene_ids_for_faces(db):
    a = _add_face(db, scene_id=5, frame=0)
    b = _add_face(db, scene_id=5, frame=1)
    c = _add_face(db, scene_id=2)
    assert db.get_scene_ids_for_faces([a, b, c]) == [2, 5]
    assert db.get_scene_ids_for_faces([]) == []


def test_bbox_iou_basic():
    a = {"x": 0, "y": 0, "w": 10, "h": 10}
    assert rdb._bbox_iou(a, a) == pytest.approx(1.0)
    assert rdb._bbox_iou(a, {"x": 20, "y": 20, "w": 10, "h": 10}) == 0
    assert rdb._bbox_iou(a, {"x": 5, "y": 0, "w": 10, "h": 10}) == pytest.approx(50 / 150)
    assert rdb._bbox_iou(a, {"x": 0, "y": 0, "w": 0, "h": 10}) == 0
    assert rdb._bbox_iou({"x": 0, "y": 0, "w": 0, "h": 0}, {"x": 0, "y": 0, "w": 0, "h": 0}) == 0


# ==================== upsert_scene_library_faces ====================


def test_upsert_same_bbox_keeps_ids_and_memberships(db):
    first = db.upsert_scene_library_faces(1, [
        _face_dict(frame=0, x=10, y=10, w=100, h=100),
        _face_dict(frame=0, x=300, y=10, w=100, h=100),
    ])
    f0, f1 = first["face_ids"]
    assert first["inserted"] == 2
    assigned = db.create_face_cluster(status="assigned", performer_id="42")
    banned = db.create_face_cluster(status="banned")
    db.add_faces_to_cluster(assigned, [f0])
    db.add_faces_to_cluster(banned, [f1])

    second = db.upsert_scene_library_faces(1, [
        _face_dict(frame=0, x=305, y=15, w=100, h=100, seed=300),
        _face_dict(frame=0, x=15, y=5, w=100, h=100, seed=10),
    ])
    assert second["face_ids"] == [f1, f0]
    assert second["updated"] == 2 and second["inserted"] == 0 and second["deleted"] == 0
    assert _members(db, assigned) == [f0]
    assert _members(db, banned) == [f1]
    assert db.get_library_face(f0)["bbox_x"] == 15


def test_upsert_deletes_vanished_uncurated_and_retains_vanished_curated(db):
    r = db.upsert_scene_library_faces(1, [
        _face_dict(frame=0, x=10), _face_dict(frame=1, x=10), _face_dict(frame=2, x=10),
        _face_dict(frame=3, x=10),
    ])
    plain, in_open, in_pinned, rejected = r["face_ids"]
    open_c = db.create_face_cluster(status="open")
    pinned_c = db.create_face_cluster(status="open", pinned=True)
    db.add_faces_to_cluster(open_c, [in_open])
    db.add_faces_to_cluster(pinned_c, [in_pinned])
    db.add_face_rejections([rejected], cluster_id=99)

    res = db.upsert_scene_library_faces(1, [])
    assert res["deleted"] == 2
    assert res["retained"] == 2
    assert db.get_library_face(plain) is None
    assert db.get_library_face(in_open) is None
    assert db.get_library_face(in_pinned) is not None
    assert db.get_library_face(rejected) is not None
    assert _members(db, pinned_c) == [in_pinned]
    assert rejected in db.get_face_rejections()


def test_upsert_inserts_new_and_returns_parallel_ids(db):
    r1 = db.upsert_scene_library_faces(4, [_face_dict(frame=0, x=10)])
    (old,) = r1["face_ids"]
    r2 = db.upsert_scene_library_faces(4, [
        _face_dict(frame=5, x=10, ts=50.0),
        _face_dict(frame=0, x=12, seed=10),
        _face_dict(frame=0, x=400),
    ])
    ids = r2["face_ids"]
    assert len(ids) == 3
    assert ids[1] == old
    assert ids[0] is not None and ids[2] is not None and len({*ids}) == 3
    assert r2["inserted"] == 2 and r2["updated"] == 1
    assert db.get_library_face(ids[0])["frame_index"] == 5
    assert db.get_library_face(ids[0])["stash_scene_id"] == 4
    # duplicate within the same call collides on UNIQUE key -> None
    r3 = db.upsert_scene_library_faces(9, [_face_dict(frame=0, x=10, w=10, h=10),
                                           _face_dict(frame=0, x=10, w=10, h=10)])
    assert r3["face_ids"][0] is not None and r3["face_ids"][1] is None
    assert r3["inserted"] == 1


def test_upsert_matches_by_timestamp_when_frame_index_shifts(db):
    (fid,) = db.upsert_scene_library_faces(1, [_face_dict(frame=3, ts=30.0, x=10)])["face_ids"]
    r = db.upsert_scene_library_faces(1, [_face_dict(frame=4, ts=30.2, x=11, seed=3010)])
    assert r["face_ids"] == [fid]
    assert db.get_library_face(fid)["frame_index"] == 4
    # timestamp too far apart (even though frame index equal) -> not matched
    r = db.upsert_scene_library_faces(1, [_face_dict(frame=4, ts=35.0, x=11)])
    assert r["face_ids"][0] != fid
    assert r["deleted"] == 1
    # null timestamps fall back to frame_index equality
    (g,) = db.upsert_scene_library_faces(2, [_face_dict(frame=1, ts=None, x=10)])["face_ids"]
    assert db.upsert_scene_library_faces(2, [_face_dict(frame=1, ts=None, x=12, seed=1010)])["face_ids"] == [g]


def test_upsert_stale_crop_paths_excludes_paths_still_in_use(db):
    r = db.upsert_scene_library_faces(1, [
        _face_dict(frame=0, x=10, crop_path="1/a.jpg"),
        _face_dict(frame=1, x=10, crop_path="1/b.jpg"),
        _face_dict(frame=2, x=10, crop_path="1/c.jpg"),
    ])
    a, b, c = r["face_ids"]
    res = db.upsert_scene_library_faces(1, [
        _face_dict(frame=0, x=11, crop_path="1/a2.jpg", seed=10),   # a updated, old path stale
        _face_dict(frame=1, x=12, crop_path="1/b.jpg", seed=1010),  # b same path, not stale
        _face_dict(frame=9, x=10, crop_path="1/c.jpg"),    # new face reuses c's path
    ])
    # c deleted (frame 2 vanished) but its path is referenced by the new row
    assert res["deleted"] == 1
    assert sorted(res["stale_crop_paths"]) == ["1/a.jpg"]
    assert db.get_library_face(a)["crop_path"] == "1/a2.jpg"


def test_upsert_updates_best_match_fields(db):
    (fid,) = db.upsert_scene_library_faces(1, [
        _face_dict(frame=0, x=10, match_id="old", match_name="Old", conf=0.5, db_version="v1")])["face_ids"]
    db.upsert_scene_library_faces(1, [
        _face_dict(frame=0, x=10, match_id="new", match_name="New", conf=0.9, db_version="v2")])
    f = db.get_library_face(fid)
    assert (f["best_match_id"], f["best_match_name"], f["best_match_confidence"], f["db_version"]) == \
        ("new", "New", 0.9, "v2")
    db.upsert_scene_library_faces(1, [_face_dict(frame=0, x=10)])
    f = db.get_library_face(fid)
    assert f["best_match_id"] is None and f["best_match_name"] is None


def test_upsert_greedy_prefers_highest_iou(db):
    r = db.upsert_scene_library_faces(1, [
        _face_dict(frame=0, x=10, y=10, w=100, h=100, ts=1.0),
        _face_dict(frame=1, x=20, y=10, w=100, h=100, ts=1.1),
    ])
    best, other = r["face_ids"]
    # new face is in the same time window as both; IoU 1.0 with `best`, ~0.82 with `other`
    res = db.upsert_scene_library_faces(1, [_face_dict(frame=0, x=10, y=10, w=100, h=100, ts=1.1)])
    assert res["face_ids"] == [best]
    assert res["updated"] == 1 and res["deleted"] == 1 and res["conflicts"] == 0
    assert db.get_library_face(other) is None


def test_upsert_update_collision_is_conflict(db):
    # v14 key: (scene, frame, timestamp, x, y, w, h). A paired update can only
    # collide with a retained row through a duplicate detection: both new
    # faces want keep's key; the first pairs with keep, the second with mover.
    r = db.upsert_scene_library_faces(2, [
        _face_dict(frame=0, x=10, y=10, w=100, h=100, ts=9.0),
        _face_dict(frame=0, x=12, y=10, w=100, h=100, ts=9.0),
    ])
    keep, mover = r["face_ids"]
    db.add_face_rejections([keep], cluster_id=1)
    dup1 = _face_dict(frame=0, x=10, y=10, w=100, h=100, ts=9.0)
    dup2 = _face_dict(frame=0, x=10, y=10, w=100, h=100, ts=9.0, seed=12)   # mover's person
    res = db.upsert_scene_library_faces(2, [dup1, dup2])
    assert res["conflicts"] == 1
    assert res["updated"] == 1
    assert res["face_ids"] == [keep, mover]
    assert db.get_library_face(mover)["bbox_x"] == 12
    assert db.get_library_face(keep) is not None


def test_upsert_same_frame_new_timestamp_is_a_new_row(db):
    # pre-v13 this collided on (scene, frame, x, y) with the kept row and was dropped
    r = db.upsert_scene_library_faces(3, [_face_dict(frame=1, x=10, y=10, ts=10.0)])
    [kept] = r["face_ids"]
    db.add_face_rejections([kept], cluster_id=1)
    res = db.upsert_scene_library_faces(3, [_face_dict(frame=1, x=10, y=10, ts=7.5)])
    assert res["inserted"] == 1 and res["conflicts"] == 0 and res["retained"] == 1
    assert res["face_ids"][0] not in (None, kept)
