"""Regression tests for the library-face persistence fixes (review findings 1, 2, 3, 7).

1. Two people in the same frame must each keep their own best-match anchor.
2. A face that no person claimed must get no anchor (no persons[-1] fallback).
3. The face -> person mapping must come from the persons identify actually
   returned, in cluster, frequency and hybrid modes.
7. Re-identifying a scene must keep face ids, so bans and curated cluster
   memberships survive.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

import identification_router
import library_face_persist
import scene_matcher
from face_cluster_service import FaceClusterService
from identification_router import PersonResult, _match_to_response, distance_to_confidence
from library_face_store import LibraryFaceStore
from recommendations_db import RecommendationsDB

SCENE = 4242


# ---------------------------------------------------------------- fixtures

_rng = np.random.default_rng(1234)
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


def _face(x: int, y: int, w: int = 60, h: int = 60, conf: float = 0.99):
    return NS(bbox={"x": x, "y": y, "w": w, "h": h}, confidence=conf, yaw=0.0, image=None)


def _match(sid: str, score: float):
    return NS(stashdb_id=sid, name=f"name-{sid}", combined_score=score,
              facenet_distance=score, arcface_distance=score, country=None,
              image_url=None, universal_id=None, endpoint=None)


def _result(person: str, matches: list):
    return NS(embedding=_emb(person), matches=matches, face=None)


def _bm(sid: str, conf: float = 0.8):
    return NS(stashdb_id=sid, name=f"name-{sid}", confidence=conf)


def _person(sid: str | None, face_indices: list[int]):
    return NS(best_match=_bm(sid) if sid else None, _face_indices=list(face_indices))


def _frames(*indices: int):
    return [NS(frame_index=i, timestamp_sec=float(i), image=np.zeros((480, 640, 3), np.uint8))
            for i in indices]


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "rec.db")


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = LibraryFaceStore(tmp_path)

    def fake_save_crop(scene_id, frame_index, bbox, frame_image, timestamp_sec=None):
        rel = f"{scene_id}/{frame_index}_{bbox['x']}_{bbox['y']}.jpg"
        p = s.base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpg")
        return rel

    monkeypatch.setattr(s, "save_crop", fake_save_crop)
    return s


def _scene_rows(db, scene=SCENE):
    with db._connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM library_faces WHERE stash_scene_id = ? ORDER BY id", (scene,))]


def _row_at(rows, x):
    return next(r for r in rows if r["bbox_x"] == x)


def _persist(db, store, detected, persons, frames=None):
    embeddings = [_emb(f"p{i}") for i in range(len(detected))]
    return library_face_persist.persist_from_identify(
        scene_id=SCENE,
        extraction_frames=frames if frames is not None else _frames(*{f for f, _ in detected}),
        detected_faces=detected,
        embeddings=embeddings,
        persons=persons,
        db_version="v1",
        db=db,
        store=store,
    )


# ------------------------------------------------ findings 1 & 2: anchoring

def test_two_people_same_frame_get_their_own_anchor(db, store):
    detected = [(0, _face(10, 10)), (0, _face(300, 10))]
    # persons in the order identify returned them; A owns face 1, B owns face 0
    persons = [_person("A", [1]), _person("B", [0])]
    assert _persist(db, store, detected, persons) == 2
    rows = _scene_rows(db)
    assert _row_at(rows, 10)["best_match_id"] == "B"
    assert _row_at(rows, 300)["best_match_id"] == "A"
    assert _row_at(rows, 300)["best_match_name"] == "name-A"
    assert _row_at(rows, 300)["best_match_confidence"] == pytest.approx(0.8)


def test_unmapped_face_gets_no_anchor(db, store):
    detected = [(0, _face(10, 10)), (3, _face(50, 50))]
    persons = [_person("P", [0])]
    _persist(db, store, detected, persons)
    rows = _scene_rows(db)
    assert _row_at(rows, 10)["best_match_id"] == "P"
    unmapped = _row_at(rows, 50)
    assert unmapped["best_match_id"] is None
    assert unmapped["best_match_name"] is None
    assert unmapped["best_match_confidence"] is None


def test_person_without_best_match_leaves_face_unanchored(db, store):
    detected = [(0, _face(10, 10))]
    _persist(db, store, detected, [_person(None, [0])])
    assert _scene_rows(db)[0]["best_match_id"] is None


def test_face_person_assignment_first_claim_wins_and_bounds():
    persons = [_person("A", [0, 1, 7, -1]), _person("B", [1, 2])]
    assert library_face_persist.face_person_assignment(persons, n_faces=3) == {0: 0, 1: 0, 2: 1}


def test_screenshot_faces_ignored(db, store):
    # 2 detected faces; the screenshot face sits at all_results index 2 (frame -1)
    detected = [(0, _face(10, 10)), (1, _face(20, 20))]
    persons = [_person("A", [0, 2]), _person("S", [2]), _person("B", [1])]
    assign = library_face_persist.face_person_assignment(persons, n_faces=len(detected))
    assert assign == {0: 0, 1: 2}
    assert _persist(db, store, detected, persons) == 2
    rows = _scene_rows(db)
    assert len(rows) == 2
    assert {r["frame_index"] for r in rows} == {0, 1}
    assert _row_at(rows, 20)["best_match_id"] == "B"


def test_screenshot_results_get_indices_past_detected_faces():
    all_results = [
        (0, _result("A", [_match("sA", 0.2)])),
        (1, _result("A", [_match("sA", 0.22)])),
        (-1, _result("A", [_match("sA", 0.18)])),
    ]
    persons = scene_matcher.cluster_mode_matching(all_results, recognizer=None, top_k=3)
    assert scene_matcher.get_face_indices(persons[0]) == [0, 1, 2]
    assert library_face_persist.face_person_assignment(persons, n_faces=2) == {0: 0, 1: 0}


# ------------------------------------------- finding 3: matching modes

def test_cluster_mode_matching_indices_follow_sorted_persons():
    # A appears first but has fewer faces; B is the more prominent person.
    all_results = [
        (0, _result("A", [_match("sA", 0.2)])),
        (0, _result("B", [_match("sB", 0.25)])),
        (1, _result("B", [_match("sB", 0.25)])),
        (2, _result("A", [_match("sA", 0.21)])),
        (2, _result("B", [_match("sB", 0.24)])),
    ]
    persons = scene_matcher.cluster_mode_matching(all_results, recognizer=None, top_k=3)
    assert [p.best_match.stashdb_id for p in persons] == ["sB", "sA"]
    assert [p.person_id for p in persons] == [0, 1]
    assert scene_matcher.get_face_indices(persons[0]) == [1, 2, 4]
    assert scene_matcher.get_face_indices(persons[1]) == [0, 3]


def test_cluster_mode_matching_merges_clusters_by_match_and_keeps_all_indices():
    # Two embedding clusters with the same top match are merged into one person.
    all_results = [
        (0, _result("A", [_match("sA", 0.2)])),
        (1, _result("A2", [_match("sA", 0.3)])),
        (2, _result("B", [_match("sB", 0.2)])),
    ]
    persons = scene_matcher.cluster_mode_matching(all_results, recognizer=None, top_k=3)
    by_id = {p.best_match.stashdb_id: scene_matcher.get_face_indices(p) for p in persons}
    assert by_id == {"sA": [0, 1], "sB": [2]}


def _legacy_cluster_mode(all_results, cluster_threshold=0.6, top_k=5):
    """Verbatim copy of the pre-fix inline cluster-mode branch of identify_scene."""
    from scene_matcher import aggregate_matches, cluster_faces_by_person, merge_clusters_by_match
    clusters = cluster_faces_by_person(all_results, None, distance_threshold=cluster_threshold)
    clusters = merge_clusters_by_match(clusters)
    persons = []
    used_performers: set[str] = set()
    all_persons = []
    for person_id, cluster in enumerate(clusters):
        aggregated_matches = aggregate_matches(cluster, top_k=top_k)
        all_persons.append((len(cluster), PersonResult(
            person_id=person_id,
            frame_count=len(cluster),
            best_match=aggregated_matches[0] if aggregated_matches else None,
            all_matches=aggregated_matches,
        )))
    all_persons.sort(key=lambda x: x[0], reverse=True)
    for _, person in all_persons:
        if person.best_match:
            if person.best_match.stashdb_id in used_performers:
                for alt_match in person.all_matches[1:]:
                    if alt_match.stashdb_id not in used_performers:
                        person.best_match = alt_match
                        used_performers.add(alt_match.stashdb_id)
                        break
                else:
                    person.best_match = None
            else:
                used_performers.add(person.best_match.stashdb_id)
        person.all_matches = [m for m in person.all_matches if m.stashdb_id not in used_performers or m.stashdb_id == (person.best_match.stashdb_id if person.best_match else None)]
        persons.append(person)
    for i, person in enumerate(persons):
        person.person_id = i
    return persons


def test_cluster_mode_matching_output_identical_to_legacy_branch():
    # Includes a dedup case: C's top match sA is already used by A, so C falls
    # back to its alternative, and an unknown (no-match) cluster.
    def build():
        return [
            (0, _result("A", [_match("sA", 0.2), _match("sQ", 0.4)])),
            (1, _result("A", [_match("sA", 0.22)])),
            (2, _result("A", [_match("sA", 0.21)])),
            # C's per-face top (matches[0]) is sQ, so it is not merged into A,
            # but its aggregated best is sA -> dedup falls back to sQ.
            (0, _result("C", [_match("sQ", 0.3), _match("sA", 0.2)])),
            (3, _result("C", [_match("sQ", 0.31), _match("sA", 0.2), _match("sR", 0.45)])),
            (4, _result("U", [])),
        ]
    results = build()
    new = scene_matcher.cluster_mode_matching(results, recognizer=None, top_k=4)
    old = _legacy_cluster_mode(results, top_k=4)
    assert [p.model_dump() for p in new] == [p.model_dump() for p in old]


def test_clustered_frequency_indices_follow_final_order():
    all_results = [
        (0, _result("U", [])),                         # unknown person
        (0, _result("A", [_match("sA", 0.2)])),
        (1, _result("B", [_match("sB", 0.25)])),
        (2, _result("B", [_match("sB", 0.24)])),
        (3, _result("B", [_match("sB", 0.26)])),
        (3, _result("A", [_match("sA", 0.21)])),
        (4, _result("U", [])),
    ]
    persons = scene_matcher.clustered_frequency_matching(
        all_results, recognizer=None, top_k=3,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    assert [p.best_match.stashdb_id if p.best_match else None for p in persons] == ["sB", "sA", None]
    idx = [scene_matcher.get_face_indices(p) for p in persons]
    assert idx == [[2, 3, 4], [1, 5], [0, 6]]


def test_clustered_frequency_all_used_person_keeps_its_faces():
    # C's only candidate (sA) is taken by the larger A cluster -> best_match None,
    # but C's faces still belong to C, not to A.
    all_results = [
        (0, _result("A", [_match("sA", 0.2)])),
        (1, _result("A", [_match("sA", 0.2)])),
        # C's top match (sLow) is over max_distance, so sA is its only candidate.
        # A different top match keeps merge_clusters_by_match from folding C into A.
        (2, _result("C", [_match("sLow", 0.7), _match("sA", 0.3)])),
    ]
    persons = scene_matcher.clustered_frequency_matching(
        all_results, recognizer=None, top_k=3,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    assert persons[0].best_match.stashdb_id == "sA"
    assert scene_matcher.get_face_indices(persons[0]) == [0, 1]
    assert persons[1].best_match is None
    assert scene_matcher.get_face_indices(persons[1]) == [2]


def test_hybrid_matching_assigns_each_face_to_at_most_one_final_person():
    all_results = [
        (0, _result("A", [_match("sA", 0.2)])),              # 0  cluster A
        (0, _result("B", [_match("sB", 0.25)])),             # 1  cluster B
        (1, _result("A", [_match("sA", 0.22)])),             # 2  cluster A
        (1, _result("B", [_match("sB", 0.24)])),             # 3  cluster B
        (2, _result("D", [_match("sZ", 0.35), _match("sB", 0.4)])),  # 4 own cluster, top sZ
        (3, _result("C", [_match("sX", 0.45)])),             # 5  background, no final person
        (4, _result("E", [_match("sY", 0.6)])),              # 6  over max_distance
    ]
    persons = scene_matcher.hybrid_matching(
        all_results, recognizer=None, top_k=5, max_distance=0.5,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    ids = {p.best_match.stashdb_id: scene_matcher.get_face_indices(p) for p in persons}
    assert set(ids) == {"sA", "sB"}
    assert ids["sA"] == [0, 2]
    # D's own top match is sZ (not a final person): its secondary sB match
    # must not hand it to sB.
    assert ids["sB"] == [1, 3]
    claimed = [i for v in ids.values() for i in v]
    assert len(claimed) == len(set(claimed))
    assign = library_face_persist.face_person_assignment(persons, n_faces=len(all_results))
    assert 5 not in assign and 6 not in assign


def test_frequency_based_matching_attaches_face_claims():
    all_results = [
        (0, _result("A", [_match("sA", 0.2), _match("sB", 0.45)])),
        (1, _result("A", [_match("sA", 0.22)])),
        (1, _result("B", [_match("sB", 0.3)])),
        (2, _result("B", [_match("sB", 0.31)])),
        (3, _result("C", [_match("sX", 0.3)])),   # single appearance, dropped
    ]
    persons = scene_matcher.frequency_based_matching(
        all_results, top_k=5,
        _match_to_response=_match_to_response, _distance_to_confidence=distance_to_confidence)
    ids = {p.best_match.stashdb_id: scene_matcher.get_face_indices(p) for p in persons}
    assert ids == {"sA": [0, 1], "sB": [2, 3]}


def test_face_indices_not_serialized():
    p = PersonResult(person_id=0, frame_count=1, best_match=None, all_matches=[])
    scene_matcher.set_face_indices(p, [3, 1, 3])
    assert scene_matcher.get_face_indices(p) == [1, 3]
    dumped = p.model_dump()
    assert not any("face" in k and "ind" in k for k in dumped)
    assert "face_indices" not in p.model_dump_json()
    assert "_face_indices" not in PersonResult.model_json_schema().get("properties", {})
    assert scene_matcher.get_face_indices(NS()) == []


def test_face_indices_survive_rerank():
    p = PersonResult(person_id=0, frame_count=1, best_match=None, all_matches=[])
    scene_matcher.set_face_indices(p, [0, 2])
    out = scene_matcher._rerank_scene_persons(
        persons=[p], matcher=NS(), body_ratios=None, tattoo_result=None,
        tattoo_scores=None, signals_used=["face"], tattoos_detected=0)
    assert scene_matcher.get_face_indices(out[0]) == [0, 2]


def test_persist_from_identify_signature_has_no_recognizer():
    params = inspect.signature(library_face_persist.persist_from_identify).parameters
    for gone in ("recognizer", "all_results", "cluster_threshold"):
        assert gone not in params
    for needed in ("scene_id", "extraction_frames", "detected_faces", "embeddings", "persons"):
        assert needed in params


def test_face_person_mapping_removed():
    assert not hasattr(library_face_persist, "face_person_mapping")


def test_identify_scene_uses_cluster_mode_matching_and_new_persist_call():
    src = inspect.getsource(identification_router.identify_scene)
    assert "cluster_mode_matching(" in src
    assert "recognizer=_recognizer,\n            cluster_threshold" not in src
    persist_call = src[src.index("_persist_library_faces_off_loop("):]
    persist_call = persist_call[: persist_call.index(")") + 1]
    assert "all_results" not in persist_call
    assert "recognizer" not in persist_call


# --------------------------------------- finding 7: re-identify keeps ids

def test_reidentify_preserves_ban_and_assigned_membership(db, store):
    detected = [(0, _face(10, 10)), (0, _face(300, 10)), (2, _face(100, 100))]
    persons = [_person("A", [0]), _person("B", [1, 2])]
    _persist(db, store, detected, persons)
    before = {r["bbox_x"]: r["id"] for r in _scene_rows(db)}
    f0, f1, f2 = before[10], before[300], before[100]

    assigned = db.create_face_cluster(name="Alice", status="assigned",
                                      performer_id="123", performer_name="Alice")
    db.add_faces_to_cluster(assigned, [f0])
    svc = FaceClusterService(db)
    svc.ban_faces([f1])
    open_c = db.create_face_cluster(status="open")
    db.add_faces_to_cluster(open_c, [f2])

    moved = [(0, _face(14, 12)), (0, _face(304, 8)), (2, _face(96, 104))]
    _persist(db, store, moved, persons)
    after = {r["bbox_x"]: r["id"] for r in _scene_rows(db)}
    assert after == {14: f0, 304: f1, 96: f2}

    assert [m["id"] for m in db.get_face_cluster_membership(f0)] == [assigned]
    assert db.get_face_cluster(assigned)["status"] == "assigned"
    assert [m["id"] for m in db.get_face_cluster_membership(f2)] == [open_c]
    assert [f["id"] for f in svc.get_banned_faces()] == [f1]
    # the row was refreshed in place with the new detection
    r0 = next(r for r in _scene_rows(db) if r["id"] == f0)
    assert r0["crop_path"] == f"{SCENE}/0_14_12.jpg"
    assert r0["best_match_id"] == "A"
    # the replaced crop was removed from disk
    assert not (store.base / f"{SCENE}/0_10_10.jpg").exists()
    assert (store.base / f"{SCENE}/0_14_12.jpg").exists()


def test_reidentify_removes_vanished_uncurated_face_and_its_crop(db, store):
    detected = [(0, _face(10, 10)), (1, _face(200, 200))]
    _persist(db, store, detected, [_person("A", [0, 1])])
    rows = _scene_rows(db)
    gone_crop = _row_at(rows, 200)["crop_path"]
    assert (store.base / gone_crop).exists()
    keep_id = _row_at(rows, 10)["id"]

    _persist(db, store, [(0, _face(10, 10))], [_person("A", [0])])
    rows = _scene_rows(db)
    assert [r["id"] for r in rows] == [keep_id]
    assert not (store.base / gone_crop).exists()
    assert (store.base / rows[0]["crop_path"]).exists()


def test_reidentify_keeps_vanished_curated_face(db, store):
    detected = [(0, _face(10, 10)), (1, _face(200, 200))]
    _persist(db, store, detected, [_person("A", [0, 1])])
    vanished = _row_at(_scene_rows(db), 200)
    FaceClusterService(db).ban_faces([vanished["id"]])
    _persist(db, store, [(0, _face(10, 10))], [_person("A", [0])])
    ids = [r["id"] for r in _scene_rows(db)]
    assert vanished["id"] in ids
    assert (store.base / vanished["crop_path"]).exists()


def test_no_faces_detected_leaves_existing_rows(db, store):
    _persist(db, store, [(0, _face(10, 10))], [_person("A", [0])])
    assert _persist(db, store, [], []) == 0
    assert len(_scene_rows(db)) == 1


# ------------------------------------------------------------- store/crops

def test_delete_crop_rejects_path_traversal(tmp_path):
    s = LibraryFaceStore(tmp_path / "data")
    outside = tmp_path / "data" / "outside.jpg"
    outside.write_bytes(b"x")
    victim = tmp_path / "victim.jpg"
    victim.write_bytes(b"x")
    assert s.delete_crop("../outside.jpg") is False
    assert s.delete_crop("../../victim.jpg") is False
    assert s.delete_crop(str(victim)) is False
    assert s.delete_crop("") is False
    assert outside.exists() and victim.exists()

    inside = s.base / "4" / "abc.jpg"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")
    assert s.delete_crop("4/abc.jpg") is True
    assert not inside.exists()
    assert s.delete_crop("4/abc.jpg") is False  # already gone


def test_crop_failure_still_persists_row(db, store, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(store, "save_crop", boom)
    assert _persist(db, store, [(0, _face(10, 10))], [_person("A", [0])]) == 1
    row = _scene_rows(db)[0]
    assert row["crop_path"] is None
    assert row["best_match_id"] == "A"


def test_missing_frame_still_persists_row_without_timestamp(db, store):
    detected = [(0, _face(10, 10)), (9, _face(20, 20))]
    assert _persist(db, store, detected, [], frames=_frames(0)) == 2
    r9 = next(r for r in _scene_rows(db) if r["frame_index"] == 9)
    assert r9["timestamp_sec"] is None and r9["crop_path"] is None


def test_persist_without_db_is_nonfatal(monkeypatch, store):
    import recommendations_router

    def no_db():
        raise RuntimeError("no db")
    monkeypatch.setattr(recommendations_router, "get_rec_db", no_db)
    n = library_face_persist.persist_from_identify(
        scene_id=SCENE, extraction_frames=_frames(0),
        detected_faces=[(0, _face(1, 1))], embeddings=[_emb("x")],
        persons=[], store=store)
    assert n == 0
