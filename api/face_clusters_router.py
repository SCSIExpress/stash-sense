"""REST endpoints for Immich-style face grouping.

Browse face clusters, view face crops, merge/ignore clusters, and assign a
cluster to a Stash performer (bulk-tags all scenes the faces came from).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from face_cluster_service import (
    BuildInProgress,
    ClusterNotFound,
    FaceClusterService,
    InvalidClusterOperation,
)
from recommendations_router import get_rec_db, get_stash_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/face-groups", tags=["face-groups"])

_service: FaceClusterService | None = None


def get_face_cluster_service() -> FaceClusterService:
    global _service
    if _service is None:
        _service = FaceClusterService(get_rec_db())
    return _service


def _optional_stash_client():
    """Best-effort Stash client: None if not configured, never raises."""
    try:
        return get_stash_client()
    except HTTPException:
        return None
    except Exception:
        logger.warning("could not resolve stash client", exc_info=True)
        return None


# ==================== Models ====================


class BuildClustersRequest(BaseModel):
    distance_threshold: float = Field(0.55, ge=0.2, le=1.5)
    min_cluster_size: int = Field(3, ge=1, le=100)
    seed_by_match: bool = True
    replace_existing: bool = False
    incremental: bool = Field(False, description="Only cluster faces not already in a group; absorb close ones into existing groups by centroid")
    auto_tag: bool = Field(False, description="With incremental: auto-tag scenes of faces absorbed into assigned groups")


class AssignPerformerRequest(BaseModel):
    performer_id: str = Field(min_length=1, description="Stash performer ID")
    performer_name: str = Field(description="Performer name (denormalized for display)")


class _CreatePerformerFailed(Exception):
    """performerCreate failed (nothing was created)."""


class CreateAndAssignRequest(BaseModel):
    name: str = Field(min_length=1, description="Name for the new Stash performer")
    disambiguation: str | None = Field(None, description="Optional disambiguation string")
    favorite: bool = Field(False, description="Mark performer as favorite")


class MergeClustersRequest(BaseModel):
    source_ids: list[int] = Field(description="Cluster IDs to merge from")
    target_id: int = Field(description="Cluster ID to merge into")


class SplitClusterRequest(BaseModel):
    face_ids: list[int]


class UpdateClusterRequest(BaseModel):
    name: str | None = None
    status: str | None = None


class FaceIdsRequest(BaseModel):
    face_ids: list[int]


class EjectRequest(FaceIdsRequest):
    eject_mode: str = Field("eject", description="eject (back to pool) or ban (never cluster again)")


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


@contextmanager
def _service_errors(invalid_status: int = 409):
    """Map service exceptions to HTTP errors. BuildInProgress -> 409: curation
    is refused while a build runs (it would be undone or duplicated)."""
    try:
        yield
    except BuildInProgress as e:
        raise HTTPException(409, str(e))
    except InvalidClusterOperation as e:
        raise HTTPException(invalid_status, str(e))
    except ClusterNotFound as e:
        raise HTTPException(404, str(e))


# ==================== Endpoints ====================


@router.get("", response_model=ClusterListResponse)
def list_clusters(
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
def build_clusters(req: BuildClustersRequest):
    """Cluster library faces. Long-running for large libraries."""
    service = get_face_cluster_service()
    # Always resolve a stash client (best-effort) so assigned-cluster stash ids
    # get backfilled on every build, not just incremental+auto_tag ones.
    # auto_tag itself is still gated by req.auto_tag inside the service.
    stash = _optional_stash_client()
    try:
        return service.build_clusters(
            distance_threshold=req.distance_threshold,
            min_cluster_size=req.min_cluster_size,
            seed_by_match=req.seed_by_match,
            replace_existing=req.replace_existing,
            incremental=req.incremental,
            auto_tag=req.auto_tag,
            stash_client=stash,
        )
    except BuildInProgress as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        logger.exception("build_clusters failed")
        raise HTTPException(500, str(e))


@router.get("/stats")
def cluster_stats():
    return get_face_cluster_service().stats()


@router.get("/{cluster_id}")
def get_cluster(cluster_id: int, limit: int = Query(200, ge=1, le=2000)):
    db = get_rec_db()
    cluster = db.get_face_cluster(cluster_id)
    if cluster is None:
        live = db.resolve_cluster_id(cluster_id)
        if live is not None and live != cluster_id:
            raise HTTPException(404, f"cluster {cluster_id} was merged into cluster {live}")
        raise HTTPException(404, f"cluster {cluster_id} not found")
    faces = db.get_representative_faces(cluster_id, limit=limit)
    return {
        **cluster,
        "face_count": db.get_cluster_face_count(cluster_id),
        "scene_ids": db.get_cluster_scene_ids(cluster_id),
        "top_matches": db.get_cluster_top_matches(cluster_id),
        "faces": faces,
    }


@router.patch("/{cluster_id}")
def update_cluster(cluster_id: int, req: UpdateClusterRequest):
    with _service_errors(invalid_status=400):
        return get_face_cluster_service().update_cluster(cluster_id, name=req.name, status=req.status)


@router.post("/{cluster_id}/assign")
def assign_performer(cluster_id: int, req: AssignPerformerRequest):
    """Assign cluster to a Stash performer and bulk-tag all its scenes."""
    service = get_face_cluster_service()
    stash = get_stash_client()
    try:
        with _service_errors():
            return service.assign_performer(cluster_id, req.performer_id, req.performer_name, stash)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("assign_performer failed")
        raise HTTPException(500, str(e))


@router.post("/{cluster_id}/create-and-assign")
def create_performer_and_assign(cluster_id: int, req: CreateAndAssignRequest):
    """Create a NEW Stash performer named `name`, then assign the cluster to it
    (bulk-tags every scene containing the group's faces). For personal content
    with performers that aren't in the big DBs."""
    db = get_rec_db()
    cluster = db.get_face_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(404, f"cluster {cluster_id} not found")
    if cluster["status"] == "banned":
        raise HTTPException(409, "banned faces cannot be assigned; unban them first")
    stash = get_stash_client()
    created: dict = {}

    def create():
        # runs under the curation lock, after the group is validated: a
        # refused assign (build running, ...) never leaves an orphan performer
        try:
            created.update(stash.create_performer_sync(
                req.name,
                **({"disambiguation": req.disambiguation} if req.disambiguation else {}),
                **({"favorite": True} if req.favorite else {}),
            ))
        except Exception as e:
            logger.exception("performerCreate failed")
            raise _CreatePerformerFailed(str(e)) from e
        return created

    service = get_face_cluster_service()
    try:
        with _service_errors():
            result = service.assign_performer(cluster_id, None, None, stash, create_performer=create)
    except _CreatePerformerFailed as e:
        raise HTTPException(500, f"failed to create performer: {e}")
    except HTTPException as e:
        if not created:
            raise
        raise HTTPException(e.status_code,
                            f"performer created ({created['id']}) but assignment failed: {e.detail}")
    except Exception as e:
        if not created:
            logger.exception("create-and-assign failed")
            raise HTTPException(500, str(e))
        logger.exception("assign after create failed")
        raise HTTPException(500, f"performer created ({created['id']}) but assignment failed: {e}")

    return result


@router.post("/{cluster_id}/ignore")
def ignore_cluster(cluster_id: int):
    with _service_errors():
        if not get_face_cluster_service().ignore_cluster(cluster_id):
            raise HTTPException(404, f"cluster {cluster_id} not found")
    return {"ok": True}


@router.delete("/{cluster_id}")
def delete_cluster(cluster_id: int):
    """Delete a cluster (faces become unassigned again)."""
    with _service_errors():
        if not get_face_cluster_service().delete_cluster(cluster_id):
            raise HTTPException(404, f"cluster {cluster_id} not found")
    return {"ok": True}


@router.post("/merge")
def merge_clusters(req: MergeClustersRequest):
    """Merge sources into target. Merging into an assigned group tags the
    moved faces' scenes with its performer (best-effort Stash client)."""
    with _service_errors():
        return get_face_cluster_service().merge_clusters(
            req.source_ids, req.target_id, stash_client=_optional_stash_client())


@router.post("/{cluster_id}/split")
def split_cluster(cluster_id: int, req: SplitClusterRequest):
    with _service_errors(invalid_status=400):
        new_id = get_face_cluster_service().split_cluster(cluster_id, req.face_ids)
    return {"new_cluster_id": new_id}


@router.post("/{cluster_id}/eject")
def eject_faces(cluster_id: int, req: EjectRequest):
    """Remove faces from a group. mode=eject returns them to the unassigned pool;
    mode=ban additionally excludes them from all future clustering (junk faces)."""
    if req.eject_mode not in ("eject", "ban"):
        raise HTTPException(400, f"invalid eject_mode: {req.eject_mode!r} (must be 'eject' or 'ban')")
    # Eject first, even when banning: eject records the rejection from this
    # group (ban alone would remove the membership without it, and an unban
    # would let the face drift straight back). Only faces of this group are
    # banned; eject and ban are one curation step (no build in between).
    with _service_errors():
        return get_face_cluster_service().eject_or_ban(
            cluster_id, req.face_ids, ban=req.eject_mode == "ban")


@router.post("/unban")
def unban_faces(req: FaceIdsRequest):
    """Return banned faces to the unassigned pool."""
    with _service_errors():
        return {"unbanned": get_face_cluster_service().unban_faces(req.face_ids)}


@router.get("/banned/list")
def list_banned(limit: int = Query(500, ge=1, le=2000)):
    return {"faces": get_face_cluster_service().get_banned_faces(limit)}


@router.get("/{cluster_id}/face/{face_id}/crop")
def face_crop(cluster_id: int, face_id: int):
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
