"""Repair round 4 (service side): regression tests for the third skeptic pass
on the face-cluster fixes (items S1-S7). Each test fails on the pre-round code
and passes with the fix.
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import face_cluster_service as fcs
import face_clusters_router
from face_cluster_service import FaceClusterService, InvalidClusterOperation
from recommendations_db import RecommendationsDB
from tests.test_fc_fix_round2 import _make_app
from tests.test_fc_fix_round3 import _cj, _node3
from tests.test_fc_fix_service import FakeStash, Person, _add_face, _members, _memberships


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "r4.db")


@pytest.fixture
def svc(db):
    return FaceClusterService(db)


class _SlowReadStash(FakeStash):
    """Scene reads take a while (network), so concurrent taggers overlap."""

    def get_scene_performer_ids_sync(self, scene_id):
        r = super().get_scene_performer_ids_sync(scene_id)
        time.sleep(0.3)
        return r


def _run_all(*calls):
    errors = []

    def wrap(fn):
        try:
            fn()
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(e)

    ts = [threading.Thread(target=wrap, args=(c,), daemon=True) for c in calls]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    assert not errors, errors


# ======================================================================
# S1: scene tagging is a read-modify-write; concurrent taggers must not
# drop each other's performer
# ======================================================================

class TestSceneTagRace:
    def test_concurrent_assigns_sharing_a_scene_keep_both_performers(self, db, svc):
        p, q = Person(90000), Person(91000)
        g1, g2 = db.create_face_cluster(), db.create_face_cluster()
        db.add_faces_to_cluster(g1, [p.face(db, 7)])
        db.add_faces_to_cluster(g2, [q.face(db, 7, frame=3)])
        stash = _SlowReadStash(scene_performers={"7": ["x"]})
        _run_all(lambda: svc.assign_performer(g1, "A", "A", stash),
                 lambda: svc.assign_performer(g2, "B", "B", stash))
        assert sorted(stash.scenes["7"]) == ["A", "B", "x"]

    def test_assign_racing_merge_into_assigned_group_keeps_both(self, db, svc):
        p, q = Person(92000), Person(93000)
        g1 = db.create_face_cluster()
        db.add_faces_to_cluster(g1, [p.face(db, 7)])
        tgt = db.create_face_cluster(status="assigned", performer_id="B", pinned=True)
        db.add_faces_to_cluster(tgt, [q.face(db, 8)])
        src = db.create_face_cluster()
        db.add_faces_to_cluster(src, [q.face(db, 7, frame=2)])
        stash = _SlowReadStash(scene_performers={"7": ["x"]})
        _run_all(lambda: svc.assign_performer(g1, "A", "A", stash),
                 lambda: svc.merge_clusters([src], tgt, stash_client=stash))
        assert sorted(stash.scenes["7"]) == ["A", "B", "x"]


# ======================================================================
# S2: a StashDB-linked performer group wins its id over a group that only
# carries it as a member anchor
# ======================================================================

def test_linked_group_wins_stash_id_over_bigger_anchor_only_group(db, svc):
    look, real = Person(94000), Person(95000)
    u = db.create_face_cluster(status="matched", stash_ids=["uuid-x"])
    db.add_faces_to_cluster(u, [look.face(db, 1 + i, match_id="uuid-x") for i in range(4)])
    stash = FakeStash(performer_stash_ids={"A": [], "B": ["uuid-x"]})
    svc.assign_performer(u, "A", "Created", stash)        # create-and-assign: no StashDB link
    assert db.get_cluster_stash_ids(u) == ["uuid-x"]      # keeps its anchor, unsynced
    lnk = db.create_face_cluster()
    db.add_faces_to_cluster(lnk, [real.face(db, 20)])
    svc.assign_performer(lnk, "B", "Real", stash)         # linked to uuid-x on StashDB
    assert db.get_face_cluster(lnk)["stash_ids_synced"] == 1

    assert db.get_stash_id_cluster_map()["uuid-x"] == lnk
    new = real.face(db, 50, match_id="uuid-x")
    svc.build_clusters(incremental=True, auto_tag=True, stash_client=stash)
    assert _memberships(db, new) == [lnk]
    assert stash.scenes.get("50") == ["B"]


# ======================================================================
# S3: re-assigning after the group left 'assigned' still drops the old
# performer's ids
# ======================================================================

def test_reassign_after_status_change_does_not_leak_old_performer_ids(db, svc):
    p = Person(96000)
    g = db.create_face_cluster()
    db.add_faces_to_cluster(g, [p.face(db, 1 + i) for i in range(3)])
    stash = FakeStash(performer_stash_ids={"41": ["uuid-a"], "43": []})
    svc.assign_performer(g, "41", "Wrong", stash)
    assert db.get_cluster_stash_ids(g) == ["uuid-a"]
    svc.update_cluster(g, status="open")
    svc.assign_performer(g, "43", "Right", stash)
    assert db.get_cluster_stash_ids(g) == []
    assert "uuid-a" in db.get_blocked_stash_ids(g)

    newcomer = _add_face(db, scene_id=77, match_id="uuid-a")
    svc.build_clusters(incremental=True, auto_tag=True, stash_client=stash)
    assert newcomer not in _members(db, g)
    assert "43" not in stash.scenes.get("77", [])


# ======================================================================
# S4: full builds re-check the links of synced performer groups
# ======================================================================

def test_full_build_resyncs_performer_whose_link_changed(db, svc):
    p = Person(97000)
    g = db.create_face_cluster()
    db.add_faces_to_cluster(g, [p.face(db, 1 + i) for i in range(2)])
    stash = FakeStash(performer_stash_ids={"41": ["uuid-a"]})
    svc.assign_performer(g, "41", "P", stash)
    assert db.get_face_cluster(g)["stash_ids_synced"] == 1

    stash.performer_stash_ids["41"] = ["uuid-b"]           # relinked in Stash
    svc.build_clusters(stash_client=stash)                 # full build
    assert db.get_cluster_stash_ids(g) == ["uuid-b"]
    new = p.face(db, 30, match_id="uuid-b")
    svc.build_clusters(incremental=True, stash_client=stash)
    assert _memberships(db, new) == [g]


# ======================================================================
# S5: ejecting wrong-performer faces never rejects their own identity, and
# a member-derived id that lost its majority leaves the group
# ======================================================================

class TestEjectOwnIdentity:
    def test_one_at_a_time_eject_does_not_reject_own_identity(self, db, svc):
        bob, jane = Person(98000), Person(98500)
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        j1, j2 = jane.face(db, 1, match_id="uuid-jane"), jane.face(db, 2, match_id="uuid-jane")
        db.add_faces_to_cluster(a, [bob.face(db, 3), j1, j2])

        svc.eject_faces(a, [j1])
        assert db.get_face_rejections([j1]).get(j1, {}).get("stash_ids", set()) == set()
        svc.eject_faces(a, [j2])
        svc.build_clusters(incremental=True)
        [m] = [c["id"] for c in db.list_face_clusters("matched")]
        assert _memberships(db, j1) == [m] and _memberships(db, j2) == [m]

    def test_eject_after_backfill_drops_lost_member_anchor(self, db, svc):
        bob, jane = Person(99000), Person(99500)
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        j1, j2 = jane.face(db, 1, match_id="uuid-jane"), jane.face(db, 2, match_id="uuid-jane")
        db.add_faces_to_cluster(a, [bob.face(db, 3), j1, j2])
        svc.build_clusters(incremental=True)                 # backfill: 42 gets uuid-jane
        assert db.get_cluster_stash_ids(a) == ["uuid-jane"]

        svc.eject_faces(a, [j1])
        assert db.get_cluster_stash_ids(a) == []
        assert "uuid-jane" in db.get_blocked_stash_ids(a)
        assert db.get_face_rejections([j1]).get(j1, {}).get("stash_ids", set()) == set()
        svc.eject_faces(a, [j2])
        svc.build_clusters(incremental=True)
        assert j1 not in _members(db, a)
        [m] = [c["id"] for c in db.list_face_clusters("matched")]
        assert _memberships(db, j1) == [m] and _memberships(db, j2) == [m]
        assert db.get_cluster_stash_ids(a) == []             # not re-added by backfill

    def test_linked_performer_still_rejects_its_own_id(self, db, svc):
        p = Person(99800)
        g = db.create_face_cluster()
        faces = [p.face(db, 1 + i, match_id="uuid-x") for i in range(3)]
        db.add_faces_to_cluster(g, faces)
        svc.assign_performer(g, "41", "P", FakeStash(performer_stash_ids={"41": ["uuid-x"]}))
        svc.eject_faces(g, [faces[0]])
        assert db.get_face_rejections([faces[0]])[faces[0]]["stash_ids"] == {"uuid-x"}
        assert db.get_cluster_stash_ids(g) == ["uuid-x"]


# ======================================================================
# S6: a performer-less assigned group never absorbs one with a performer
# ======================================================================

@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setattr(face_clusters_router, "get_rec_db", lambda: db)
    monkeypatch.setattr(face_clusters_router, "_service", FaceClusterService(db))
    monkeypatch.setattr(face_clusters_router, "_optional_stash_client", lambda: None)
    return TestClient(_make_app())


class TestPerformerlessAssigned:
    def test_assign_with_empty_performer_id_is_rejected(self, db, client, monkeypatch):
        g = db.create_face_cluster()
        db.add_faces_to_cluster(g, [_add_face(db, scene_id=1)])
        monkeypatch.setattr(face_clusters_router, "get_stash_client", lambda: FakeStash())
        r = client.post(f"/face-groups/{g}/assign", json={"performer_id": "", "performer_name": "X"})
        assert r.status_code == 422
        assert db.get_face_cluster(g)["status"] == "open"

    def test_merge_into_performerless_target_from_performer_group_is_refused(self, db, svc):
        tgt = db.create_face_cluster(status="assigned", performer_id=None, pinned=True)
        db.add_faces_to_cluster(tgt, [_add_face(db, scene_id=1, frame=i) for i in range(5)])
        src = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        f = _add_face(db, scene_id=2)
        db.add_faces_to_cluster(src, [f])
        with pytest.raises(InvalidClusterOperation):
            svc.merge_clusters([src], tgt)
        assert _memberships(db, f) == [src]

    def test_plugin_prefers_assigned_target_with_performer(self, tmp_path):
        [r] = _node3(tmp_path, [["chooseMergeTarget", [[_cj(1, "assigned", 9, None),
                                                        _cj(2, "assigned", 2, "p1")]]]])
        assert r["error"] is None and r["targetId"] == 2


# ======================================================================
# S7: create-and-assign never orphans a performer; cancelled builds stop
# in every long loop
# ======================================================================

class _CreatingStash(FakeStash):
    def __init__(self, on_create=None, **kw):
        super().__init__(**kw)
        self.created: list[str] = []
        self.on_create = on_create

    def create_performer_sync(self, name, **extra):
        self.created.append(name)
        if self.on_create:
            self.on_create()
        return {"id": "900", "name": name}


def test_build_starting_during_performer_create_does_not_orphan_it(db, client, monkeypatch):
    g = db.create_face_cluster()
    db.add_faces_to_cluster(g, [_add_face(db, scene_id=1)])
    taken = []

    def build_starts():          # a build grabs the build lock right after the check
        taken.append(fcs._BUILD_LOCK.acquire(blocking=False))

    stash = _CreatingStash(on_create=build_starts, performer_stash_ids={"900": []})
    monkeypatch.setattr(face_clusters_router, "get_stash_client", lambda: stash)
    try:
        r = client.post(f"/face-groups/{g}/create-and-assign", json={"name": "New"})
    finally:
        if taken and taken[0]:
            fcs._BUILD_LOCK.release()
    assert stash.created == ["New"]
    assert r.status_code == 200, r.text
    c = db.get_face_cluster(g)
    assert c["status"] == "assigned" and c["performer_id"] == "900"


def test_stop_during_pass3_creates_no_more_groups(db, svc, monkeypatch):
    for k in range(4):                       # 4 distinct unmatched people, 3 faces each
        p = Person(100000 + 1000 * k)
        for i in range(3):
            p.face(db, 10 * k + i + 1)
    flag = {"stop": False}
    real = fcs.cluster_library_faces

    def clustering_then_cancel(*a, **k):
        out = real(*a, **k)
        flag["stop"] = True                  # cancel requested while clustering ran
        return out

    monkeypatch.setattr(fcs, "cluster_library_faces", clustering_then_cancel)
    res = svc.build_clusters(incremental=True, should_stop=lambda: flag["stop"])
    assert res["cancelled"] is True
    assert db.list_face_clusters("open") == []


# ======================================================================
# Round 4b: S5 follow-ups (ban keeps the group's identity; a face the user
# rejected from a performer is never tagged with it by a group action),
# anchors cleared by v13 come back, banned faces can be restored from the UI
# ======================================================================

_UNLINKED = {"42": []}          # performer 42 exists but has no StashDB link


class TestBanKeepsGroupIdentity:
    def test_banning_member_anchor_faces_keeps_the_guessed_id(self, db, svc):
        bob = Person(110000)
        stash = FakeStash(performer_stash_ids=dict(_UNLINKED))
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        anchored = [bob.face(db, i, match_id="uuid-bob") for i in range(1, 4)]
        db.add_faces_to_cluster(a, [*anchored, bob.face(db, 4), bob.face(db, 5)])
        svc.build_clusters(incremental=True, stash_client=stash)
        assert db.get_cluster_stash_ids(a) == ["uuid-bob"]

        svc.eject_or_ban(a, anchored[:2], ban=True)          # junk crops of Bob
        assert db.get_cluster_stash_ids(a) == ["uuid-bob"]
        assert "uuid-bob" not in db.get_blocked_stash_ids(a)
        new = [bob.face(db, 10 + i, match_id="uuid-bob") for i in range(3)]
        svc.build_clusters(incremental=True, stash_client=stash)
        assert all(_memberships(db, f) == [a] for f in new)
        assert db.list_face_clusters("matched") == []


class TestRejectedFacesNeverTaggedByGroupActions:
    def _setup(self, db, svc):
        """Unlinked group 42 got uuid-bob from 1 Bob + 2 Jane faces all
        (mis)matched to uuid-bob; the user ejects the Jane faces. They form a
        matched uuid-bob group (their match may be right: S5)."""
        bob, jane = Person(111000), Person(112000)
        stash = FakeStash(performer_stash_ids=dict(_UNLINKED))
        a = db.create_face_cluster(status="assigned", performer_id="42", pinned=True)
        wrong = [jane.face(db, 20 + i, match_id="uuid-bob") for i in range(2)]
        db.add_faces_to_cluster(a, [bob.face(db, 1, match_id="uuid-bob"), *wrong, bob.face(db, 5)])
        svc.build_clusters(incremental=True, stash_client=stash)
        svc.eject_faces(a, wrong)
        svc.build_clusters(incremental=True, stash_client=stash)
        [m] = [c["id"] for c in db.list_face_clusters("matched")]
        assert _members(db, m) == set(wrong)
        extra = [bob.face(db, 30 + i, match_id="uuid-bob") for i in range(2)]
        svc.build_clusters(incremental=True, stash_client=stash)
        assert _members(db, m) == {*wrong, *extra}
        return a, m, wrong, extra

    def test_assign_skips_faces_rejected_from_that_performer(self, db, svc):
        a, m, wrong, extra = self._setup(db, svc)
        st = FakeStash(performer_stash_ids=dict(_UNLINKED))
        svc.assign_performer(m, "42", "Bob", st)
        tagged = {int(s) for s, _ in st.writes}
        assert tagged == {30, 31}                            # never Jane's scenes 20, 21
        assert _members(db, m) == set(extra)
        assert all(_memberships(db, f) == [] for f in wrong)

    def test_merge_into_performer_skips_faces_rejected_from_it(self, db, svc):
        a, m, wrong, extra = self._setup(db, svc)
        st = FakeStash(performer_stash_ids=dict(_UNLINKED))
        svc.merge_clusters([m], a, stash_client=st)
        tagged = {int(s) for s, _ in st.writes}
        assert tagged == {30, 31}
        assert set(extra) <= _members(db, a) and not (set(wrong) & _members(db, a))
        assert all(_memberships(db, f) == [] for f in wrong)


def test_v13_marks_scenes_with_stored_faces_for_re_identify(tmp_path):
    """v13 clears every face's anchor and says the next identify rewrites
    it; the fingerprint job skips current-version scenes, so v13 must
    mark those scenes outdated."""
    import sqlite3
    from tests.test_fc_fix_round2 import _downgrade_to_v12, _raw_face
    path = tmp_path / "m12.db"
    RecommendationsDB(path)
    _downgrade_to_v12(path)
    conn = sqlite3.connect(path)
    _raw_face(conn, 1, match="uuid-a")
    for scene in (1, 2):
        conn.execute("INSERT INTO scene_fingerprints (stash_scene_id, fingerprint_status, db_version) "
                     "VALUES (?, 'complete', '2026.01.30')", (scene,))
    conn.commit()
    conn.close()
    db = RecommendationsDB(path)
    assert db.get_scene_fingerprint(1)["db_version"] is None
    assert db.get_scene_fingerprint(2)["db_version"] == "2026.01.30"


def test_banned_faces_can_be_restored(tmp_path):
    [ok, declined, failed] = _node_unban(tmp_path, [
        [7, True, None], [7, False, None], [7, True, "boom"]])
    assert ok["calls"] == [[7]] and ok["r"] is True
    assert declined["calls"] == [] and declined["r"] is False
    assert failed["r"] is False and "boom" in failed["alerts"][0]
    from tests.test_fc_fix_round3 import _PLUGIN_JS
    js = _PLUGIN_JS.read_text(encoding="utf-8")
    block = js[js.index("function renderBannedList"):js.index("function openGroupDetail")]
    assert "performUnban(" in block and "FaceGroupsAPI" in block


_UNBAN_HARNESS = r"""
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
  const out = [];
  for (const [faceId, answer, failWith] of JSON.parse(process.argv[3])) {
    const calls = [], alerts = [];
    const api = { unban: async ids => { calls.push(ids); if (failWith) throw new Error(failWith); return { unbanned: 1 }; } };
    const r = await FG.performUnban(faceId, api, () => answer, m => alerts.push(m));
    out.push({ r, calls, alerts });
  }
  console.log(JSON.stringify(out));
})();
"""


def _node_unban(tmp_path, cases):
    import json
    import shutil
    import subprocess
    from tests.test_fc_fix_round3 import _PLUGIN_JS
    if shutil.which("node") is None:
        pytest.skip("node is not available")
    h = tmp_path / "hu.js"
    h.write_text(_UNBAN_HARNESS, encoding="utf-8")
    proc = subprocess.run(["node", str(h), str(_PLUGIN_JS), json.dumps(cases)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])
