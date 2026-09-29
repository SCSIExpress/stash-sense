"""Library-wide face clustering (union-find over a Voyager HNSW index).

Builds a Voyager index over all persisted library-face embeddings, queries
each face's neighbours, and unions faces closer than a distance threshold.
Clusters seeded from identify-time best matches anchor known performers.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

try:
    from voyager import Index as VoyagerIndex
    from voyager import Space
except ImportError:  # pragma: no cover - voyager stubbed in tests
    VoyagerIndex = None
    Space = None


def _face_vector(facenet_emb: bytes, arcface_emb: bytes, mode: str = "concat") -> np.ndarray:
    fn = np.frombuffer(facenet_emb, dtype=np.float32)
    af = np.frombuffer(arcface_emb, dtype=np.float32)
    if mode == "concat":
        return np.concatenate([fn, af])
    return fn if mode == "facenet" else af


def cluster_library_faces(
    faces: list[dict],
    distance_threshold: float = 0.55,
    vector_mode: str = "concat",
) -> list[list[int]]:
    """Cluster faces by embedding similarity.

    Args:
        faces: list of dicts with keys: id, facenet_emb, arcface_emb.
        distance_threshold: max cosine distance to union two faces.
        vector_mode: 'concat' (1024-d), 'facenet' or 'arcface'.

    Returns:
        List of clusters, each a list of face IDs. Singleton clusters included.
    """
    if not faces:
        return []

    n = len(faces)
    ids = [f["id"] for f in faces]
    vectors = np.stack([_face_vector(f["facenet_emb"], f["arcface_emb"], vector_mode) for f in faces])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    # HNSW-assisted neighbour search when voyager is available; brute force otherwise.
    if VoyagerIndex is not None and n > 64:
        index = VoyagerIndex(space=Space.Cosine, num_dimensions=vectors.shape[1])
        index.add_items(vectors, np.arange(n))
        k = min(32, n)
        neighbours, distances = index.query(vectors, k=k)
        for i in range(n):
            for j, d in zip(neighbours[i], distances[i]):
                j = int(j)
                if j != i and d <= distance_threshold:
                    union(i, j)
    else:
        # Block-wise brute force cosine distance
        block = 512
        for start in range(0, n, block):
            chunk = vectors[start : start + block]
            # cosine distance = 1 - dot (vectors are unit-normalized)
            sim = chunk @ vectors.T
            pairs = np.argwhere(1.0 - sim <= distance_threshold)
            for i_local, j in pairs:
                union(start + int(i_local), int(j))

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(ids[i])

    clusters = list(groups.values())
    clusters.sort(key=len, reverse=True)
    return clusters
