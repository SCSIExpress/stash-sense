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
        incremental: bool = False,
        auto_tag: bool = False,
        stash_client=None,
    ) -> dict:
        """Cluster library faces.

        Modes:
        - Full build (default): cluster ALL faces from scratch. `replace_existing`
          clears open groups first. Existing assigned/ignored groups are re-seeded
          by anchor so their membership is recovered from scratch.
        - Incremental (incremental=True): only faces not already in a live group
          are clustered. They are first matched against the centroids of existing
          groups (assigned, matched, open — closest centroid within threshold
          wins); leftovers form new open groups. This is how new matches "pick up"
          existing groups, including groups you've assigned or merged.

        With auto_tag=True (incremental only), faces absorbed into an assigned
        group trigger tagging of their scenes for that group's performer.

        Returns:
            {"clusters_created", "faces_assigned", "faces_total", ...mode-specific extras}
        """
        if incremental:
            return self._build_incremental(
                distance_threshold=distance_threshold,
                min_cluster_size=min_cluster_size,
                auto_tag=auto_tag,
                stash_client=stash_client,
            )
        return self._build_full(
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
            seed_by_match=seed_by_match,
            replace_existing=replace_existing,
        )

    def _build_full(
        self,
        distance_threshold: float,
        min_cluster_size: int,
        seed_by_match: bool,
        replace_existing: bool,
    ) -> dict:
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

        # Pass 1: seed clusters from identify-time matches.
        # Faces whose match equals an existing assigned group's performer join
        # that group (rebuilds grow assigned groups); otherwise a matched group
        # is (re)created.
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

        # Map performer_id -> existing assigned group
        assigned_groups = {}
        for c in self.db.get_clusters_by_status("assigned"):
            if c.get("performer_id"):
                assigned_groups[c["performer_id"]] = c["id"]

        for mid, face_ids in by_match.items():
            if len(face_ids) == 0:
                continue
            # Faces whose performer already has an assigned group join it;
            # otherwise a fresh matched group is created for this rebuild.
            existing = assigned_groups.get(mid)
            if existing is not None:
                self.db.add_faces_to_cluster(existing, face_ids)
                assigned += len(face_ids)
                continue
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

    def _build_incremental(
        self,
        distance_threshold: float,
        min_cluster_size: int,
        auto_tag: bool,
        stash_client=None,
    ) -> dict:
        """Cluster only new (unassigned) faces; absorb close ones into existing
        groups by centroid proximity. New small groups stay open; faces absorbed
        into an assigned group can auto-tag their scenes."""
        import numpy as np

        total = self.db.get_library_face_count()
        new_faces: list[dict] = []
        for batch in self.db.iter_library_faces(unassigned_only=True):
            new_faces.extend(batch)

        if not new_faces:
            return {
                "mode": "incremental", "clusters_created": 0, "faces_assigned": 0,
                "faces_total": total, "absorbed": 0, "tagged_scenes": 0,
            }

        # Existing live groups (assigned, matched, open) with their centroids
        live = [c for c in self.db.get_clusters_by_status("assigned", "matched", "open")
                if c["face_count"] > 0]
        centroids: dict[int, np.ndarray] = {}
        for c in live:
            vec = self.db.get_cluster_centroid(c["id"])
            if vec is not None:
                centroids[c["id"]] = vec

        status_of = {c["id"]: c["status"] for c in live}
        performer_of = {c["id"]: (c["performer_id"], c["performer_name"]) for c in live}

        absorbed_by: dict[int, list[int]] = {}  # cluster_id -> face ids
        leftovers: list[dict] = []

        # Pass 1: seed-by-match — faces whose identify-time match equals an
        # existing group's performer join that group outright.
        # Pass 2: centroid proximity for the rest.
        for f in new_faces:
            placed = False
            mid = f.get("best_match_id")
            if mid:
                for cid, (pid, _pname) in performer_of.items():
                    if pid == mid:
                        absorbed_by.setdefault(cid, []).append(f["id"])
                        placed = True
                        break
            if placed:
                continue
            fn = np.frombuffer(f["facenet_emb"], dtype=np.float32)
            af = np.frombuffer(f["arcface_emb"], dtype=np.float32)
            v = np.concatenate([fn, af])
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
                best_cid, best_d = None, float("inf")
                for cid, cent in centroids.items():
                    d = 1.0 - float(np.dot(v, cent))
                    if d < best_d:
                        best_d, best_cid = d, cid
                if best_cid is not None and best_d <= distance_threshold:
                    absorbed_by.setdefault(best_cid, []).append(f["id"])
                    placed = True
            if not placed:
                leftovers.append(f)

        absorbed = 0
        tagged_scenes = 0

        for cid, face_ids in absorbed_by.items():
            self.db.add_faces_to_cluster(cid, face_ids)
            absorbed += len(face_ids)
            # refresh centroid with the new members
            new_cent = self.db.get_cluster_centroid(cid)
            if new_cent is not None:
                centroids[cid] = new_cent
            # auto-tag newly absorbed faces' scenes for assigned groups
            if auto_tag and status_of.get(cid) == "assigned" and stash_client is not None:
                pid, _pname = performer_of.get(cid, (None, None))
                if pid:
                    for fid in face_ids:
                        face = self.db.get_library_face(fid)
                        if face is None:
                            continue
                        sid = str(face["stash_scene_id"])
                        try:
                            scene = stash_client.get_scene_performer_ids_sync(sid)
                            if scene is None:
                                continue
                            current = scene.get("performer_ids") or []
                            if pid not in current:
                                stash_client.update_scene_performers_sync(sid, [*current, pid])
                                tagged_scenes += 1
                        except Exception:
                            logger.exception("auto-tag failed for scene %s", sid)

        # Pass 3: leftovers form new open groups
        created = 0
        if leftovers:
            from library_clustering import cluster_library_faces
            for cluster_face_ids in cluster_library_faces(leftovers, distance_threshold):
                if len(cluster_face_ids) < min_cluster_size:
                    continue
                cid = self.db.create_face_cluster(status="open")
                self.db.add_faces_to_cluster(cid, cluster_face_ids)
                created += 1

        return {
            "mode": "incremental", "clusters_created": created,
            "faces_assigned": absorbed + sum(1 for _ in leftovers),
            "faces_total": total, "faces_new": len(new_faces),
            "absorbed": absorbed, "groups_absorbed_into": len(absorbed_by),
            "tagged_scenes": tagged_scenes,
        }

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
        """Move all faces from source clusters into target; delete sources.

        The merge is recorded in face_cluster_merge_log so rebuilds can
        re-route: if a deleted source's identity would re-form (same seed
        match), its faces are redirected to the surviving target instead.
        """
        total_moved = 0
        for sid in source_ids:
            if sid == target_id:
                continue
            rows = self.db.get_representative_faces(sid, limit=10**9)
            face_ids = [r["id"] for r in rows]
            self.db.add_faces_to_cluster(target_id, face_ids)
            self.db.delete_face_cluster(sid)
            total_moved += len(face_ids)

        self.db.record_cluster_merge(source_ids, target_id)

        return {"target_id": target_id, "merged_sources": source_ids, "faces_moved": total_moved}

    def split_cluster(self, cluster_id: int, face_ids: list[int], new_status: str = "open") -> int:
        """Remove faces from a cluster into a new cluster."""
        cid = self.db.create_face_cluster(status=new_status)
        self.db.remove_faces_from_cluster(cluster_id, face_ids)
        self.db.add_faces_to_cluster(cid, face_ids)
        return cid

    def eject_faces(self, cluster_id: int, face_ids: list[int]) -> int:
        """Remove faces from a group; they return to the unassigned pool and can
        be absorbed into other groups (or new ones) on the next incremental run."""
        removed = self.db.remove_faces_from_cluster(cluster_id, face_ids)
        # Drop now-empty groups (except assigned/ignored ones, which are user state)
        cluster = self.db.get_face_cluster(cluster_id)
        if cluster and cluster["status"] in ("open", "matched") and self.db.get_cluster_face_count(cluster_id) == 0:
            self.db.delete_face_cluster(cluster_id)
        return removed

    def ban_faces(self, face_ids: list[int]) -> int:
        """Eject faces from any live group and put them in a singleton 'banned'
        group — junk detections (background people, wrong boxes) that should
        never be clustered again. Banned faces are excluded from all
        clustering passes."""
        held = 0
        for fid in face_ids:
            for membership in self.db.get_face_cluster_membership(fid):
                if membership["status"] in ("ignored", "banned"):
                    continue
                self.db.remove_faces_from_cluster(membership["id"], [fid])
            # dedupe: a face only needs one banned marker
            existing = [m for m in self.db.get_face_cluster_membership(fid) if m["status"] == "banned"]
            if existing:
                continue
            bid = self.db.create_face_cluster(status="banned", name="Banned faces")
            self.db.add_faces_to_cluster(bid, [fid])
            held += 1
        return held

    def unban_faces(self, face_ids: list[int]) -> int:
        """Return banned faces to the unassigned pool."""
        removed = 0
        for fid in face_ids:
            for membership in self.db.get_face_cluster_membership(fid):
                if membership["status"] == "banned":
                    self.db.delete_face_cluster(membership["id"])
                    removed += 1
        return removed

    def get_banned_faces(self, limit: int = 500) -> list[dict]:
        """Faces currently banned, with crops for review/unban."""
        rows = self.db.get_clusters_by_status("banned")
        out = []
        for c in rows[: limit]:
            for f in self.db.get_representative_faces(c["id"], limit=1):
                f["cluster_id"] = c["id"]
                out.append(f)
        return out

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
