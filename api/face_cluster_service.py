"""Immich-style face group service: build, browse, merge, assign clusters."""
from __future__ import annotations

import logging
from typing import Optional

from library_clustering import cluster_library_faces
from recommendations_db import RecommendationsDB

logger = logging.getLogger(__name__)


class FaceClusterService:
    def __init__(self, db: RecommendationsDB):
        self.db = db

    def build_clusters(
        self,
        distance_threshold: float = 0.55,
        min_cluster_size: int = 3,
        seed_by_match: bool = True,
        replace_existing: bool = False,
    ) -> dict:
        """Cluster all unassigned library faces.

        Known-performer faces (best_match set at identify time) are grouped
        directly by their match ID (seed anchors); the remaining faces are
        clustered by embedding similarity.

        Args:
            distance_threshold: max cosine distance for union.
            min_cluster_size: smaller similarity clusters are dropped (not stored).
            seed_by_match: group faces with the same best_match_id first.
            replace_existing: delete existing 'open' clusters first.

        Returns:
            {"clusters_created", "faces_assigned", "faces_total"}
        """
        total = self.db.get_library_face_count()
        if total == 0:
            return {"clusters_created": 0, "faces_assigned": 0, "faces_total": 0}

        if replace_existing:
            for c in self.db.list_face_clusters(status="open"):
                self.db.delete_face_cluster(c["id"])

        # Load all faces
        faces: list[dict] = []
        for batch in self.db.iter_library_faces():
            faces.extend(batch)

        assigned = 0
        created = 0

        # Pass 1: seed clusters from identify-time matches
        by_match: dict[str, list[int]] = {}
        unmatched_faces: list[dict] = []
        if seed_by_match:
            for f in faces:
                mid = f.get("best_match_id")
                if mid:
                    by_match.setdefault(mid, []).append(f["id"])
                else:
                    unmatched_faces.append(f)
        else:
            unmatched_faces = faces

        for mid, face_ids in by_match.items():
            if len(face_ids) == 0:
                continue
            sample = next(f for f in faces if f["id"] == face_ids[0])
            name = None
            # best_match_name isn't in iter batch; fetch from one face row
            face_row = self.db.get_library_face(face_ids[0])
            if face_row:
                name = face_row.get("best_match_name")
            cid = self.db.create_face_cluster(
                name=name or f"Matched {mid[:8]}",
                status="matched",
                performer_id=mid,
                performer_name=name,
            )
            self.db.add_faces_to_cluster(cid, face_ids)
            created += 1
            assigned += len(face_ids)

        # Pass 2: cluster the rest by embedding
        if unmatched_faces:
            clusters = cluster_library_faces(unmatched_faces, distance_threshold)
            for cluster_face_ids in clusters:
                if len(cluster_face_ids) < min_cluster_size:
                    continue
                cid = self.db.create_face_cluster(status="open")
                self.db.add_faces_to_cluster(cid, cluster_face_ids)
                created += 1
                assigned += len(cluster_face_ids)

        return {"clusters_created": created, "faces_assigned": assigned, "faces_total": total}

    def assign_performer(
        self,
        cluster_id: int,
        performer_id: str,
        performer_name: str,
        stash_client,
    ) -> dict:
        """Assign a cluster to a Stash performer and tag all its scenes.

        For each scene containing cluster faces: read current performer IDs
        (read-then-write, performer_ids is a full replacement list) and append
        the target performer if missing.
        """
        cluster = self.db.get_face_cluster(cluster_id)
        if cluster is None:
            raise ValueError(f"cluster {cluster_id} not found")

        scene_ids = self.db.get_cluster_scene_ids(cluster_id)
        tagged, skipped, failed = 0, 0, []

        for sid in scene_ids:
            try:
                scene = stash_client.get_scene_performer_ids_sync(str(sid))
                if scene is None:
                    failed.append({"scene_id": sid, "error": "scene not found"})
                    continue
                current = scene.get("performer_ids") or []
                if performer_id in current:
                    skipped += 1
                    continue
                stash_client.update_scene_performers_sync(str(sid), [*current, performer_id])
                tagged += 1
            except Exception as e:
                logger.exception("failed tagging scene %s", sid)
                failed.append({"scene_id": sid, "error": str(e)})

        self.db.update_face_cluster(
            cluster_id,
            status="assigned",
            performer_id=performer_id,
            performer_name=performer_name,
        )

        return {
            "cluster_id": cluster_id,
            "performer_id": performer_id,
            "performer_name": performer_name,
            "scenes_tagged": tagged,
            "scenes_already_tagged": skipped,
            "scenes_failed": failed,
        }

    def merge_clusters(self, source_ids: list[int], target_id: int) -> dict:
        """Move all faces from source clusters into target; delete sources."""
        total_moved = 0
        for sid in source_ids:
            if sid == target_id:
                continue
            rows = self.db.get_representative_faces(sid, limit=10**9)
            face_ids = [r["id"] for r in rows]
            self.db.add_faces_to_cluster(target_id, face_ids)
            self.db.delete_face_cluster(sid)
            total_moved += len(face_ids)

        # Merge metadata: keep target performer info if present, else adopt source's
        target = self.db.get_face_cluster(target_id)
        if target and not target.get("performer_id"):
            # find any surviving source metadata (they're deleted, so nothing to adopt;
            # caller should set performer info explicitly if wanted)
            pass

        return {"target_id": target_id, "merged_sources": source_ids, "faces_moved": total_moved}

    def split_cluster(self, cluster_id: int, face_ids: list[int], new_status: str = "open") -> int:
        """Remove faces from a cluster into a new cluster."""
        cid = self.db.create_face_cluster(status=new_status)
        self.db.remove_faces_from_cluster(cluster_id, face_ids)
        self.db.add_faces_to_cluster(cid, face_ids)
        return cid

    def ignore_cluster(self, cluster_id: int) -> bool:
        """Mark a cluster as ignored (false positives, background faces)."""
        return self.db.update_face_cluster(cluster_id, status="ignored")

    def stats(self) -> dict:
        clusters = self.db.list_face_clusters()
        return {
            "faces_total": self.db.get_library_face_count(),
            "clusters": len(clusters),
            "open": sum(1 for c in clusters if c["status"] == "open"),
            "matched": sum(1 for c in clusters if c["status"] == "matched"),
            "assigned": sum(1 for c in clusters if c["status"] == "assigned"),
            "ignored": sum(1 for c in clusters if c["status"] == "ignored"),
        }
