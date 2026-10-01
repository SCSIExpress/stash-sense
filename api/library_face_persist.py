"""Persist per-face data during scene identification.

Called from identification_router after faces are detected and embedded.
Stores embeddings + bbox in stash_sense.db and face crops on disk, so
library-wide clustering can later group faces by person.

The face -> person mapping comes from the persons identify_scene actually
returned: every matching mode tags each PersonResult with the indices (into
all_results) of the faces it covers (scene_matcher.set_face_indices). The
first len(detected_faces) entries of all_results are the detected faces in
order; screenshot faces are appended after them and are never persisted.
A face is anchored only where the face-level evidence agrees (its own top
match is the person's best match), only by the person with the most support
for that performer, and only once per performer per frame.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_store = None


def get_face_store(data_dir: str | None = None):
    """Lazy singleton for the LibraryFaceStore."""
    global _store
    if _store is None:
        from library_face_store import LibraryFaceStore
        base = data_dir or os.environ.get("DATA_DIR", "./data")
        _store = LibraryFaceStore(base)
    return _store


def np_bytes(arr) -> bytes:
    """Serialize float32 array to bytes (contiguous)."""
    import numpy as np
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def _top_match(matches):
    """The lowest-score match of a face, or None."""
    return min(matches, key=lambda m: m.combined_score) if matches else None


def _match_score(matches, stashdb_id: str):
    """The face's own score for stashdb_id (None if it is not in its matches)."""
    scores = [m.combined_score for m in matches or [] if m.stashdb_id == stashdb_id]
    return min(scores) if scores else None


def face_person_assignment(
    persons: list,
    n_faces: int,
    face_frames: list | None = None,
    face_matches: list | None = None,
) -> dict[int, int]:
    """Map detected-face index -> index into persons, for best-match anchoring.

    Uses the face indices each person carries. Indices outside [0, n_faces)
    (screenshot faces, garbage) are ignored. If two persons claim the same
    face the first one wins.

    An anchor puts a face straight into its performer's group with no
    embedding check, so it must be precise:

    - face_matches (per detected face, its match list): a face is kept only
      if the person's best match is the face's OWN top match. Faces a person
      covers only through its cluster or a secondary match (co-stars,
      background people, a rerank that changed the person's identity) get no
      anchor; builds place them by embedding instead.
    - Persons sharing a best match (possible after a rerank): only the one
      with the most anchored faces for it keeps its anchors.
    - face_frames (per detected face, its frame index): a performer is
      anchored on at most one face per frame. When a frame holds several, the
      face with the best own score for it wins (lowest index without matches).
    """
    from scene_matcher import get_face_indices

    out: dict[int, int] = {}
    for k, person in enumerate(persons or []):
        for i in get_face_indices(person):
            if not isinstance(i, int) or i < 0 or i >= n_faces:
                continue
            if i in out:
                if out[i] != k:
                    logger.debug("face %d claimed by persons %d and %d; keeping %d", i, out[i], k, out[i])
                continue
            out[i] = k

    def person_sid(k):
        bm = getattr(persons[k], "best_match", None)
        return bm.stashdb_id if bm is not None else None

    if face_matches is not None:
        for i in list(out):
            sid = person_sid(out[i])
            matches = face_matches[i] if i < len(face_matches) else None
            top = _top_match(matches)
            if sid is None or top is None or top.stashdb_id != sid:
                del out[i]

    # One identity, one anchoring person. A rerank (multi-signal) re-picks each
    # person's best match independently, and frequency / cluster matching keep
    # performers already used by other persons among a person's candidates, so
    # two persons (two different people) can end up with the same best match.
    # Only the person with the most face-level support for that performer
    # anchors; the others' faces are placed by embedding.
    support: dict[object, dict[int, int]] = {}
    for i, k in out.items():
        sid = person_sid(k)
        support.setdefault(sid, {}).setdefault(k, 0)
        support[sid][k] += 1
    owner = {sid: min(by_k, key=lambda k: (-by_k[k], k)) for sid, by_k in support.items()}
    dropped = {i for i, k in out.items() if owner.get(person_sid(k)) != k}
    if dropped:
        logger.debug("faces %s dropped: their person shares its best match with another person",
                     sorted(dropped))
        out = {i: k for i, k in out.items() if i not in dropped}

    if face_frames is not None:
        # At most one face per performer per frame (keyed by the performer, not
        # the person, so no two faces in a frame are ever anchored to one id).
        best: dict[tuple[object, object], int] = {}
        for i in sorted(out):
            k = out[i]
            sid = person_sid(k)
            key = (sid if sid is not None else ("person", k), face_frames[i])
            j = best.get(key)
            if j is None:
                best[key] = i
                continue
            if face_matches is not None:
                si = _match_score(face_matches[i], sid)
                sj = _match_score(face_matches[j], sid)
                if si is not None and (sj is None or si < sj):
                    best[key] = i
        keep = set(best.values())
        out = {i: k for i, k in out.items() if i in keep}
    return out


def persist_library_faces(
    scene_id: int,
    extraction_frames: list,
    detected_faces: list[tuple[int, object]],
    embeddings: list,
    face_person: dict[int, int],
    persons: list,
    db_version: str | None = None,
    db=None,
    store=None,
    face_matches: list | None = None,
) -> int:
    """Save per-face records for one scene.

    Rows are upserted (RecommendationsDB.upsert_scene_library_faces): a face
    re-detected at the same spot with an agreeing embedding keeps its id, so
    cluster memberships, bans and rejections survive re-identification; a
    different person at that spot is a new row. Vanished uncurated faces are
    removed along with their crops. Each write saves fresh crop files, so
    stale crops can be deleted after commit without racing another write.

    Args:
        scene_id: Stash scene ID.
        extraction_frames: ExtractedFrame list (for timestamps and crops).
        detected_faces: list of (frame_index, DetectedFace).
        embeddings: FaceEmbedding list parallel to detected_faces.
        face_person: {detected-face index: index into persons}.
        persons: PersonResult list, for best-match anchoring.
        db_version: face DB version at identify time.
        face_matches: per detected face, its own match list. When given, an
            anchored face stores its own confidence for the anchor
            (1 - its score) instead of the person's aggregate.
        db, store: injectable RecommendationsDB / LibraryFaceStore.

    Returns:
        Number of faces persisted.
    """
    if not detected_faces:
        return 0

    if db is None:
        try:
            from recommendations_router import get_rec_db
            db = get_rec_db()
        except Exception:
            logger.warning("persist_library_faces: no rec db; skipping")
            return 0
    if store is None:
        store = get_face_store()

    persons = persons or []
    face_person = face_person or {}
    frame_by_index = {f.frame_index: f for f in extraction_frames}

    faces: list[dict] = []
    for i, ((frame_idx, face), emb) in enumerate(zip(detected_faces, embeddings)):
        best_match_id = best_match_name = None
        best_match_conf = None
        k = face_person.get(i)
        if k is not None and 0 <= k < len(persons):
            bm = getattr(persons[k], "best_match", None)
            if bm is not None:
                best_match_id = bm.stashdb_id
                best_match_name = bm.name
                best_match_conf = bm.confidence
                if face_matches is not None and i < len(face_matches):
                    own = _match_score(face_matches[i], bm.stashdb_id)
                    if own is not None:
                        best_match_conf = max(0.0, min(1.0, 1.0 - float(own)))

        ts = None
        crop_path = None
        frame = frame_by_index.get(frame_idx)
        if frame is not None:
            ts = frame.timestamp_sec
            try:
                crop_path = store.save_crop(scene_id, frame_idx, face.bbox, frame.image,
                                            timestamp_sec=ts)
            except Exception:
                logger.warning("crop save failed scene=%s frame=%s", scene_id, frame_idx, exc_info=True)

        faces.append({
            "frame_index": frame_idx,
            "timestamp_sec": ts,
            "bbox": face.bbox,
            "det_confidence": face.confidence,
            "yaw": face.yaw,
            "facenet_emb": np_bytes(emb.facenet),
            "arcface_emb": np_bytes(emb.arcface),
            "crop_path": crop_path,
            "best_match_id": best_match_id,
            "best_match_name": best_match_name,
            "best_match_confidence": best_match_conf,
            "db_version": db_version,
        })

    result = db.upsert_scene_library_faces(scene_id, faces)

    for rel in result.get("stale_crop_paths", []):
        try:
            store.delete_crop(rel)
        except Exception:
            logger.warning("stale crop delete failed: %s", rel, exc_info=True)

    saved = sum(1 for fid in result.get("face_ids", []) if fid is not None)
    logger.info(
        "persisted %d/%d library faces for scene %s (updated=%s inserted=%s deleted=%s retained=%s conflicts=%s)",
        saved, len(detected_faces), scene_id, result.get("updated"), result.get("inserted"),
        result.get("deleted"), result.get("retained"), result.get("conflicts"),
    )
    return saved


def persist_from_identify(
    scene_id: int,
    extraction_frames: list,
    detected_faces: list[tuple[int, object]],
    embeddings: list,
    persons: list,
    db_version: str | None = None,
    db=None,
    store=None,
    face_matches: list | None = None,
) -> int:
    """Persist a scene's faces from an identify_scene run.

    Each face is anchored to the best match of the person that claimed it
    (via the persons' face indices), when that best match is also the face's
    own top match (face_matches), the person has the most support among
    persons sharing that best match, and no closer face in the same frame is
    anchored to that performer; other faces get no anchor. face_matches[i] is the match list
    of detected_faces[i] (identify's all_results[i].matches).
    Returns number of faces persisted.
    """
    if not detected_faces:
        return 0
    face_person = face_person_assignment(
        persons,
        len(detected_faces),
        face_frames=[frame for frame, _face in detected_faces],
        face_matches=face_matches,
    )
    return persist_library_faces(
        scene_id=scene_id,
        extraction_frames=extraction_frames,
        detected_faces=detected_faces,
        embeddings=embeddings,
        face_person=face_person,
        persons=persons,
        db_version=db_version,
        db=db,
        store=store,
        face_matches=face_matches,
    )
