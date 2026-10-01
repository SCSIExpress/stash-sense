"""Face-cluster building as a queue job."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from base_job import BaseJob, JobContext
from face_cluster_service import BuildInProgress, FaceClusterService
from recommendations_router import get_rec_db


def _optional_stash_client():
    """Best-effort Stash client for auto-tag backfill; None if unavailable."""
    try:
        from recommendations_router import get_stash_client
        return get_stash_client()
    except Exception:
        return None


class ClusterLibraryFacesJob(BaseJob):
    """Clusters all library faces into user-browsable groups."""

    async def run(self, context: JobContext, cursor: str | None = None) -> str | None:
        if context.is_stop_requested():
            return None

        db = get_rec_db()
        service = FaceClusterService(db)

        total = db.get_library_face_count()
        await context.report_progress(0, max(1, total))

        if context.is_stop_requested():
            return None

        threshold = 0.55
        min_size = 3
        try:
            from settings import get_setting
            t = get_setting("face_cluster_threshold")
            m = get_setting("face_cluster_min_size")
            if t is not None:
                threshold = float(t)
            if m is not None:
                min_size = int(m)
        except Exception:
            logger.warning("could not read cluster settings; using defaults", exc_info=True)

        stash = _optional_stash_client()

        # Run the (sync, potentially slow) build off the event loop thread so it
        # doesn't stall other async handlers sharing this loop (e.g. the
        # fingerprint job's /identify/scene self-calls). The worker thread
        # cannot be cancelled, so the build polls the stop flag itself.
        try:
            result = await asyncio.to_thread(
                service.build_clusters,
                distance_threshold=threshold,
                min_cluster_size=min_size,
                seed_by_match=True,
                replace_existing=True,
                incremental=True,
                stash_client=stash,
                should_stop=context.is_stop_requested,
            )
        except BuildInProgress as e:
            # Surface the skip (the queue records it as the job's error) rather
            # than reporting a build that never ran as completed.
            logger.info("face cluster build already in progress; skipping this run")
            raise RuntimeError(f"skipped: {e}") from e

        if result.get("cancelled"):
            logger.info("face cluster build stopped on request")

        await context.report_progress(result.get("faces_total", total), max(1, total))
        return None
