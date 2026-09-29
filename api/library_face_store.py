"""Face cropping and storage for library face grouping."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LibraryFaceStore:
    """Stores face crops on disk under <data_dir>/library_faces/ as JPEG."""

    def __init__(self, data_dir: str | Path):
        self.base = Path(data_dir) / "library_faces"
        self.base.mkdir(parents=True, exist_ok=True)

    def _shard_dir(self, scene_id: int) -> Path:
        shard = str(scene_id // 1000)
        d = self.base / shard
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_crop(self, scene_id: int, frame_index: int, bbox: dict, frame_image: np.ndarray) -> str:
        """Save a face crop as JPEG, return the relative path."""
        x, y = int(bbox.get("x", 0)), int(bbox.get("y", 0))
        w, h = int(bbox.get("w", 0)), int(bbox.get("h", 0))
        img_h, img_w = frame_image.shape[:2]
        # Clamp bbox to frame bounds with a small margin for context
        mx, my = max(0, x - 4), max(0, y - 4)
        mw, mh = min(img_w - mx, w + 8), min(img_h - my, h + 8)
        if mw <= 0 or mh <= 0:
            raise ValueError(f"invalid bbox {bbox} for frame {img_w}x{img_h}")
        crop = frame_image[my : my + mh, mx : mx + mw]
        # Resize large crops down for storage (max 160px on the long side)
        long_side = max(crop.shape[:2])
        if long_side > 160:
            scale = 160 / long_side
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        key = hashlib.sha1(f"{scene_id}:{frame_index}:{x}:{y}".encode()).hexdigest()[:16]
        rel = f"{scene_id // 1000}/{key}.jpg"
        path = self._shard_dir(scene_id) / f"{key}.jpg"
        # cv2 expects BGR; imwrite returns False (not exception) on failure
        if not cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82]):
            raise OSError(f"cv2.imwrite failed for {path}")
        return rel

    def read_crop(self, rel_path: str) -> bytes | None:
        """Read a stored crop as JPEG bytes."""
        path = self.base / rel_path
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def disk_usage(self) -> int:
        total = 0
        for p in self.base.rglob("*.jpg"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total
