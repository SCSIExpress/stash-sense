"""Face-cluster building as a queue job."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

from base_job import BaseJob, JobContext
from face_cluster_service import FaceClusterService
from recommendations_router import get_rec_db


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

        result = service.build_clusters(
            distance_threshold=threshold,
            min_cluster_size=min_size,
            seed_by_match=True,
            replace_existing=True,
        )
        await context.report_progress(result.get("faces_total", total), max(1, total))
        return None
