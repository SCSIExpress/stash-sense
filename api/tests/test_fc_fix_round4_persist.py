"""Repair round 4, persistence side.

P1. A re-identify pairs a new face with an old row only when their
    embeddings agree, so a different person at the same position never
    inherits the old row's membership, ban or rejection.
P2. Crop files are unique per write: a concurrent identify of the same scene
    can never unlink a crop another write still references.
P3. An unclaimed face gets no anchor end to end (real matching + persist);
    add_library_face stays out of production code (it has no anchoring).
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import library_face_persist
import scene_matcher
from face_cluster_service import FaceClusterService
from identification_router import _match_to_response, distance_to_confidence
from library_face_store import LibraryFaceStore
from recommendations_db import RecommendationsDB

SCENE = 9191

_rng = np.random.default_rng(4242)
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


def _face(x: int, y: int = 10, w: int = 60, h: int = 60):
    return NS(bbox={"x": x, "y": y, "w": w, "h": h}, confidence=0.99, yaw=0.0, image=None)


def _frames(*indices: int):
    return [NS(frame_index=i, timestamp_sec=float(i), image=np.full((480, 640, 3), 90, np.uint8))
            for i in indices]


def _bm(sid: str):
    return NS(stashdb_id=sid, name=f"name-{sid}", confidence=0.8)


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "r4p.db")


@pytest.fixture
def svc(db):
    return FaceClusterService(db)


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    s = LibraryFaceStore(tmp_path / "fake")

    def fake_save_crop(scene_id, frame_index, bbox, frame_image, timestamp_sec=None):
        rel = f"{scene_id}/{frame_index}_{bbox['x']}_{bbox['y']}.jpg"
        p = s.base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpg")
        return rel

    monkeypatch.setattr(s, "save_crop", fake_save_crop)
    return s


def _persist(db, store, faces, anchor=None, frame=0):
    """faces: list of (x, person-name). anchor: sid for every face (or None)."""
    detected = [(frame, _face(x)) for x, _p in faces]
    persons = [NS(best_match=_bm(anchor), _face_indices=list(range(len(faces))))] if anchor else []
    return library_face_persist.persist_from_identify(
        scene_id=SCENE,
        extraction_frames=_frames(frame),
        detected_faces=detected,
        embeddings=[_emb(p) for _x, p in faces],
        persons=persons,
        db_version="v1",
        db=db,
        store=store,
    )


def _rows(db):
    with db._connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM library_faces WHERE stash_scene_id = ? ORDER BY id", (SCENE,))]


def _at(db, x):
    return next(r for r in _rows(db) if r["bbox_x"] == x)


# ======================================================================
# P1: a pair needs agreeing embeddings
# ======================================================================

class TestPairNeedsEmbeddingAgreement:
    def test_stranger_at_same_spot_does_not_inherit_assigned_membership(self, db, fake_store):
        _persist(db, fake_store, [(10, "alice")])
        alice = _at(db, 10)
        grp = db.create_face_cluster(name="Alice", status="assigned",
                                     performer_id="1", performer_name="Alice")
        db.add_faces_to_cluster(grp, [alice["id"]])

        # re-identify: a different person stands at (nearly) the same spot
        _persist(db, fake_store, [(12, "stranger")])

        new = _at(db, 12)
        assert new["id"] != alice["id"]
        assert db.get_face_cluster_membership(new["id"]) == []
        kept = db.get_library_face(alice["id"])          # curated row kept untouched
        assert kept is not None and kept["bbox_x"] == 10
        assert kept["facenet_emb"] == alice["facenet_emb"]
        assert [m["id"] for m in db.get_face_cluster_membership(alice["id"])] == [grp]

    def test_stranger_at_same_spot_is_not_silently_banned(self, db, svc, fake_store):
        _persist(db, fake_store, [(10, "junk")])
        junk = _at(db, 10)["id"]
        svc.ban_faces([junk])

        _persist(db, fake_store, [(12, "realface")])

        new = _at(db, 12)["id"]
        assert new != junk
        assert [f["id"] for f in svc.get_banned_faces()] == [junk]

    def test_stranger_at_same_spot_does_not_inherit_rejections(self, db, fake_store):
        _persist(db, fake_store, [(10, "bob")])
        bob = _at(db, 10)["id"]
        db.add_face_rejections([bob], cluster_id=77, stash_ids=["uuid-x"])

        _persist(db, fake_store, [(12, "carol")])

        new = _at(db, 12)["id"]
        assert new != bob
        assert db.get_face_rejections([new]) == {}
        assert bob in db.get_face_rejections([bob])

    def test_unmatched_uncurated_row_is_replaced_and_its_crop_removed(self, db, fake_store):
        _persist(db, fake_store, [(10, "dave")])
        old = _at(db, 10)
        _persist(db, fake_store, [(12, "erin")])
        rows = _rows(db)
        assert [r["bbox_x"] for r in rows] == [12]
        assert rows[0]["id"] != old["id"]
        assert not (fake_store.base / old["crop_path"]).exists()
        assert (fake_store.base / rows[0]["crop_path"]).exists()

    def test_same_person_redetected_still_keeps_its_id(self, db, fake_store):
        _persist(db, fake_store, [(10, "frank")])
        fid = _at(db, 10)["id"]
        _persist(db, fake_store, [(12, "frank")])
        assert [r["id"] for r in _rows(db)] == [fid]


# ======================================================================
# P2: concurrent identifies never unlink each other's crops
# ======================================================================

def test_crop_paths_are_unique_per_write(tmp_path):
    s = LibraryFaceStore(tmp_path)
    img = np.full((200, 200, 3), 50, np.uint8)
    bbox = {"x": 10, "y": 10, "w": 50, "h": 50}
    a = s.save_crop(1, 3, bbox, img, timestamp_sec=2.0)
    b = s.save_crop(1, 3, bbox, img, timestamp_sec=2.0)
    assert a != b
    assert (s.base / a).is_file() and (s.base / b).is_file()


def test_concurrent_identify_keeps_the_crop_it_references(db, tmp_path, monkeypatch):
    """Identify #1 moves the face (X -> Y) and deletes X's old crop after its
    commit; identify #2, racing it, re-detects the face at X, saves its crop
    before #1's unlink and commits pointing at it. The row must still have a
    crop on disk at the end."""
    store = LibraryFaceStore(tmp_path / "real")
    _persist(db, store, [(10, "gina")])

    real_delete = store.delete_crop
    raced = {"done": False}

    def delete_with_race(rel):
        if not raced["done"]:
            raced["done"] = True
            _persist(db, store, [(10, "gina")])       # identify #2 slips in here
        return real_delete(rel)

    monkeypatch.setattr(store, "delete_crop", delete_with_race)
    _persist(db, store, [(40, "gina")])               # identify #1 (IoU still >= 0.5)
    assert raced["done"]

    rows = _rows(db)
    assert len(rows) == 1
    assert rows[0]["crop_path"]
    assert (store.base / rows[0]["crop_path"]).is_file()
    # and no orphan crops were left behind
    on_disk = sorted(p.relative_to(store.base).as_posix() for p in store.base.rglob("*.jpg"))
    assert on_disk == [rows[0]["crop_path"]]


# ======================================================================
# P3: unclaimed faces get no anchor end to end; add_library_face stays a
# raw test/tooling insert
# ======================================================================

def _m(sid, score):
    return NS(stashdb_id=sid, name=f"name-{sid}", combined_score=score,
              facenet_distance=score, arcface_distance=score, country=None,
              image_url=None, universal_id=None, endpoint=None)


def _res(person, matches):
    return NS(embedding=_emb(person), matches=matches, face=None)


_KW = dict(_match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
_MODES = {
    "cluster": lambda r: scene_matcher.cluster_mode_matching(r, recognizer=None, top_k=3, **_KW),
    "hybrid": lambda r: scene_matcher.hybrid_matching(r, recognizer=None, top_k=5, max_distance=0.6, **_KW),
    "frequency": lambda r: scene_matcher.frequency_based_matching(r, top_k=5, **_KW),
    "clustered_frequency": lambda r: scene_matcher.clustered_frequency_matching(
        r, None, top_k=5, max_distance=0.6, **_KW),
}


@pytest.mark.parametrize("mode", list(_MODES))
def test_unclaimed_face_gets_no_anchor_end_to_end(db, fake_store, mode):
    # A in four frames; stranger X once (its top match sZ, sA only second);
    # U once with no matches at all.
    all_results = [(f, _res("A", [_m("sA", 0.2)])) for f in range(4)] + [
        (1, _res("X", [_m("sZ", 0.35), _m("sA", 0.45)])),
        (2, _res("U", [])),
    ]
    persons = _MODES[mode](all_results)
    claimed = {i for p in persons for i in scene_matcher.get_face_indices(p)}

    detected = [(f, _face(10 + 100 * i)) for i, (f, _r) in enumerate(all_results)]
    library_face_persist.persist_from_identify(
        scene_id=SCENE,
        extraction_frames=_frames(*sorted({f for f, _ in detected})),
        detected_faces=detected,
        embeddings=[r.embedding for _f, r in all_results],
        persons=persons,
        face_matches=[r.matches for _f, r in all_results],
        db_version="v1", db=db, store=fake_store,
    )
    by_x = {r["bbox_x"]: r for r in _rows(db)}
    rows = [by_x[10 + 100 * i] for i in range(len(all_results))]

    assert [rows[i]["best_match_id"] for i in range(4)] == ["sA"] * 4
    assert rows[5]["best_match_id"] is None                 # U: nothing to anchor to
    for i in set(range(len(all_results))) - claimed:
        assert rows[i]["best_match_id"] is None, f"unclaimed face {i} anchored"
    if mode in ("hybrid", "frequency"):
        assert 4 not in claimed                               # X really is unclaimed
        assert rows[4]["best_match_id"] is None
    else:
        # X is its own person there: anchored to its own top match, never sA
        assert rows[4]["best_match_id"] == "sZ"


def test_add_library_face_not_used_by_production_code():
    """add_library_face is a raw insert with no anchoring or pairing checks;
    production writes go through persist_from_identify ->
    upsert_scene_library_faces."""
    api = Path(__file__).resolve().parents[1]
    callers = [p.name for p in api.glob("*.py")
               if re.search(r"\badd_library_face\(", p.read_text()) and p.name != "recommendations_db.py"]
    callers += [p.relative_to(api).as_posix() for p in api.glob("*/*.py")
                if not p.parts[-2] == "tests" and re.search(r"\badd_library_face\(", p.read_text())]
    assert callers == []
    src = (api / "recommendations_db.py").read_text()
    assert len(re.findall(r"\.add_library_face\(", src)) == 0
