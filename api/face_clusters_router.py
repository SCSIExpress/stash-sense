"""REST endpoints for Immich-style face grouping.

Browse face clusters, view face crops, merge/ignore clusters, and assign a
cluster to a Stash performer (bulk-tags all scenes the faces came from).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from face_cluster_service import FaceClusterService
from recommendations_router import get_rec_db, get_stash_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face-groups", tags=["face-groups"])

_service: FaceClusterService | None = None


def get_face_cluster_service() -> FaceClusterService:
    global _service
    if _service is None:
        _service = FaceClusterService(get_rec_db())
    return _service


# ==================== Models ====================


class BuildClustersRequest(BaseModel):
    distance_threshold: float = Field(0.55, ge=0.2, le=1.5)
    min_cluster_size: int = Field(3, ge=1, le=100)
    seed_by_match: bool = True
    replace_existing: bool = False


class AssignPerformerRequest(BaseModel):
    performer_id: str = Field(description="Stash performer ID")
    performer_name: str = Field(description="Performer name (denormalized for display)")


class MergeClustersRequest(BaseModel):
    source_ids: list[int] = Field(description="Cluster IDs to merge from")
    target_id: int = Field(description="Cluster ID to merge into")


class SplitClusterRequest(BaseModel):
    face_ids: list[int]


class UpdateClusterRequest(BaseModel):
    name: str | None = None
    status: str | None = None


class ClusterOut(BaseModel):
    id: int
    name: str | None
    status: str
    performer_id: str | None
    performer_name: str | None
    face_count: int
    scene_count: int


class ClusterListResponse(BaseModel):
    clusters: list[ClusterOut]
    stats: dict


# ==================== Endpoints ====================


@router.get("", response_model=ClusterListResponse)
async def list_clusters(
    status: str | None = Query(None, description="Filter by status: open|matched|assigned|ignored"),
):
    db = get_rec_db()
    service = get_face_cluster_service()
    clusters = db.list_face_clusters(status)
    return ClusterListResponse(
        clusters=[ClusterOut(**{**c, "face_count": c.get("face_count", 0), "scene_count": c.get("scene_count", 0)}) for c in clusters],
        stats=service.stats(),
    )


@router.post("/build")
async def build_clusters(req: BuildClustersRequest):
    """Cluster all library faces. Long-running for large libraries."""
    service = get_face_cluster_service()
    try:
        return service.build_clusters(
            distance_threshold=req.distance_threshold,
            min_cluster_size=req.min_cluster_size,
            seed_by_match=req.seed_by_match,
            replace_existing=req.replace_existing,
        )
    except Exception as e:
        logger.exception("build_clusters failed")
        raise HTTPException(500, str(e))


@router.get("/stats")
async def cluster_stats():
    return get_face_cluster_service().stats()


@router.get("/{cluster_id}")
async def get_cluster(cluster_id: int):
    db = get_rec_db()
    cluster = db.get_face_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(404, f"cluster {cluster_id} not found")
    faces = db.get_representative_faces(cluster_id, limit=60)
    return {
        **cluster,
        "face_count": db.get_cluster_face_count(cluster_id),
        "scene_ids": db.get_cluster_scene_ids(cluster_id),
        "top_matches": db.get_cluster_top_matches(cluster_id),
        "faces": faces,
    }


@router.patch("/{cluster_id}")
async def update_cluster(cluster_id: int, req: UpdateClusterRequest):
    db = get_rec_db()
    if db.get_face_cluster(cluster_id) is None:
        raise HTTPException(404, f"cluster {cluster_id} not found")
    db.update_face_cluster(cluster_id, name=req.name, status=req.status)
    return db.get_face_cluster(cluster_id)


@router.post("/{cluster_id}/assign")
async def assign_performer(cluster_id: int, req: AssignPerformerRequest):
    """Assign cluster to a Stash performer and bulk-tag all its scenes."""
    service = get_face_cluster_service()
    stash = get_stash_client()
    try:
        return service.assign_performer(cluster_id, req.performer_id, req.performer_name, stash)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception("assign_performer failed")
        raise HTTPException(500, str(e))


@router.post("/{cluster_id}/ignore")
async def ignore_cluster(cluster_id: int):
    if not get_face_cluster_service().ignore_cluster(cluster_id):
        raise HTTPException(404, f"cluster {cluster_id} not found")
    return {"ok": True}


@router.delete("/{cluster_id}")
async def delete_cluster(cluster_id: int):
    """Delete a cluster (faces become unassigned again)."""
    if not get_rec_db().delete_face_cluster(cluster_id):
        raise HTTPException(404, f"cluster {cluster_id} not found")
    return {"ok": True}


@router.post("/merge")
async def merge_clusters(req: MergeClustersRequest):
    try:
        return get_face_cluster_service().merge_clusters(req.source_ids, req.target_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/{cluster_id}/split")
async def split_cluster(cluster_id: int, req: SplitClusterRequest):
    db = get_rec_db()
    if db.get_face_cluster(cluster_id) is None:
        raise HTTPException(404, f"cluster {cluster_id} not found")
    new_id = get_face_cluster_service().split_cluster(cluster_id, req.face_ids)
    return {"new_cluster_id": new_id}


@router.get("/{cluster_id}/face/{face_id}/crop")
async def face_crop(cluster_id: int, face_id: int):
    """Serve a stored face crop JPEG."""
    db = get_rec_db()
    face = db.get_library_face(face_id)
    if face is None or face.get("crop_path") is None:
        raise HTTPException(404, "face crop not found")
    from library_face_persist import get_face_store
    data = get_face_store().read_crop(face["crop_path"])
    if data is None:
        raise HTTPException(404, "crop file missing")
    return Response(content=data, media_type="image/jpeg")
