"""Persist per-face data during scene identification.

Called from identification_router after faces are detected and embedded.
Stores embeddings + bbox in stash_sense.db and face crops on disk, so
library-wide clustering can later group faces by person.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

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


def persist_library_faces(
    scene_id: int,
    extraction_frames: list,
    detected_faces: list[tuple[int, object]],
    embeddings: list,
    results: list[tuple[int, object]],
    db_version: str | None = None,
    person_of_frame: dict[int, int] | None = None,
    persons: list | None = None,
) -> int:
    """Save per-face records for one scene.

    Args:
        scene_id: Stash scene ID.
        extraction_frames: ExtractedFrame list (for timestamps and crops).
        detected_faces: list of (frame_index, DetectedFace).
        embeddings: FaceEmbedding list parallel to detected_faces.
        results: (frame_index, RecognitionResult) list — used to map a face
            to its PersonResult (for best-match anchoring).
        db_version: face DB version at identify time.
        person_of_frame: optional {face_index: person_id} mapping.
        persons: PersonResult list, for best-match data.

    Returns:
        Number of faces persisted.
    """
    from recommendations_router import get_rec_db

    if not detected_faces:
        return 0

    try:
        db = get_rec_db()
    except Exception:
        logger.warning("persist_library_faces: no rec db; skipping")
        return 0

    store = get_face_store()
    # Face crops come from the frame image; ExtractedFrame keeps .image
    frame_by_index = {f.frame_index: f for f in extraction_frames}

    # Map (frame_index, occurrence) -> RecognitionResult via results list
    result_by_frame: dict[int, list[object]] = {}
    for frame_idx, result in results:
        result_by_frame.setdefault(frame_idx, []).append(result)

    # person_of_frame maps the i-th detected face to a person index
    person_by_face_idx = person_of_frame or {}

    db.delete_library_faces_for_scene(scene_id)

    saved = 0
    for i, ((frame_idx, face), emb) in enumerate(zip(detected_faces, embeddings)):
        best_match_id = best_match_name = None
        best_match_conf = None
        person_idx = person_by_face_idx.get(i)
        if persons is not None and person_idx is not None and person_idx < len(persons):
            bm = getattr(persons[person_idx], "best_match", None)
            if bm is not None:
                best_match_id = bm.stashdb_id
                best_match_name = bm.name
                best_match_conf = bm.confidence

        ts = None
        frame = frame_by_index.get(frame_idx)
        crop_path = None
        if frame is not None:
            ts = frame.timestamp_sec
            try:
                crop_path = store.save_crop(scene_id, frame_idx, face.bbox, frame.image)
            except Exception:
                logger.warning("crop save failed scene=%s frame=%s", scene_id, frame_idx, exc_info=True)

        fid = db.add_library_face(
            stash_scene_id=scene_id,
            frame_index=frame_idx,
            timestamp_sec=ts,
            bbox=face.bbox,
            det_confidence=face.confidence,
            yaw=face.yaw,
            facenet_emb=np_bytes(emb.facenet),
            arcface_emb=np_bytes(emb.arcface),
            crop_path=crop_path,
            best_match_id=best_match_id,
            best_match_name=best_match_name,
            best_match_confidence=best_match_conf,
            db_version=db_version,
        )
        if fid is not None:
            saved += 1

    logger.info("persisted %d/%d library faces for scene %s", saved, len(detected_faces), scene_id)
    return saved


def np_bytes(arr) -> bytes:
    """Serialize float32 array to bytes (contiguous)."""
    import numpy as np
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def face_person_mapping(
    all_results: list[tuple[int, object]],
    recognizer,
    cluster_threshold: float = 0.6,
) -> dict[tuple[int, int], int]:
    """Map (frame_index, occurrence_index_within_frame) -> person_id.

    Re-runs the same greedy clustering + merge used by identify_scene so
    persisted faces line up with the PersonResults the user saw.
    """
    from scene_matcher import cluster_faces_by_person, merge_clusters_by_match

    clusters = cluster_faces_by_person(all_results, recognizer, cluster_threshold)
    clusters = merge_clusters_by_match(clusters)

    # Count occurrences per frame index so faces within a frame map positionally
    mapping: dict[tuple[int, int], int] = {}
    for person_id, cluster in enumerate(clusters):
        per_frame: dict[int, int] = {}
        for frame_idx, _result in cluster:
            occ = per_frame.get(frame_idx, 0)
            mapping[(frame_idx, occ)] = person_id
            per_frame[frame_idx] = occ + 1
    return mapping


def persist_from_identify(
    scene_id: int,
    extraction_frames: list,
    detected_faces: list[tuple[int, object]],
    embeddings: list,
    all_results: list[tuple[int, object]],
    persons: list,
    recognizer,
    cluster_threshold: float,
    db_version: str | None = None,
) -> int:
    """Convenience wrapper: cluster + persist in one call from identify_scene.

    Maps each detected face to its in-scene person cluster, anchors the
    person's best match onto the face rows, and stores crops.
    Returns number of faces persisted.
    """
    if not detected_faces:
        return 0

    # person_by_face_idx is indexed by position in detected_faces
    mapping = face_person_mapping(all_results, recognizer, cluster_threshold)

    # detected_faces order == zip(detected_faces, embeddings) order used in
    # identify_scene; results share that order too. Build face position -> person.
    # results are (frame_idx, result) in the same order as detected_faces.
    occ_counter: dict[int, int] = {}
    person_by_face_idx: dict[int, int] = {}
    for i, (frame_idx, _face) in enumerate(detected_faces):
        occ = occ_counter.get(frame_idx, 0)
        person_by_face_idx[i] = mapping.get((frame_idx, occ), -1)
        occ_counter[frame_idx] = occ + 1

    return persist_library_faces(
        scene_id=scene_id,
        extraction_frames=extraction_frames,
        detected_faces=detected_faces,
        embeddings=embeddings,
        results=all_results,
        db_version=db_version,
        person_of_frame=person_by_face_idx,
        persons=persons,
    )
