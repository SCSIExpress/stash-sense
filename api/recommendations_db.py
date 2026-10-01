"""
SQLite Database Layer for Stash Sense Recommendations

Stores user-local recommendations, analysis state, and settings.
Separate from the distributed performers.db to allow independent updates.

See: docs/plans/2026-01-28-recommendations-engine-design.md
"""

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 14



_FACE_CLUSTER_V12_DDL = """
    -- A cluster's identity in StashDB/stash-box id space (matched anchor, or the
    -- assigned performer's stash ids).
    CREATE TABLE IF NOT EXISTS face_cluster_stash_ids (
        cluster_id INTEGER NOT NULL REFERENCES face_clusters(id) ON DELETE CASCADE,
        stash_id TEXT NOT NULL,
        PRIMARY KEY (cluster_id, stash_id)
    );
    CREATE INDEX IF NOT EXISTS idx_fc_stash_ids_stash ON face_cluster_stash_ids(stash_id);

    -- Durable rejection memory (eject/split). No FK on ref: cluster refs are
    -- resolved through the merge log at read time.
    CREATE TABLE IF NOT EXISTS face_rejections (
        face_id INTEGER NOT NULL REFERENCES library_faces(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('cluster','performer','stash_id')),
        ref TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (face_id, kind, ref)
    );
"""


# v14: StashDB ids a user decision took away from a group (re-assigning it to
# another performer, or the performer's own StashDB link contradicting the
# group's match anchor). Member-derived anchors never re-add them.
_FACE_CLUSTER_V14_DDL = """
    CREATE TABLE IF NOT EXISTS face_cluster_blocked_stash_ids (
        cluster_id INTEGER NOT NULL REFERENCES face_clusters(id) ON DELETE CASCADE,
        stash_id TEXT NOT NULL,
        PRIMARY KEY (cluster_id, stash_id)
    );
"""


_LIBRARY_FACES_COLUMNS = (
    "id, stash_scene_id, frame_index, timestamp_sec, bbox_x, bbox_y, bbox_w, bbox_h, "
    "det_confidence, yaw, facenet_emb, arcface_emb, crop_path, best_match_id, "
    "best_match_name, best_match_confidence, db_version, created_at"
)


def _library_faces_ddl(table: str = "library_faces") -> str:
    """library_faces as of v13: no inline UNIQUE (see _LIBRARY_FACES_KEY_DDL)."""
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stash_scene_id INTEGER NOT NULL,
        frame_index INTEGER NOT NULL,
        timestamp_sec REAL,
        bbox_x REAL NOT NULL,
        bbox_y REAL NOT NULL,
        bbox_w REAL NOT NULL,
        bbox_h REAL NOT NULL,
        det_confidence REAL NOT NULL,
        yaw REAL,
        facenet_emb BLOB NOT NULL,
        arcface_emb BLOB NOT NULL,
        crop_path TEXT,
        best_match_id TEXT,
        best_match_name TEXT,
        best_match_confidence REAL,
        db_version TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
"""


# One row per detection: the key includes the timestamp, so after a sampling
# change (other num_frames / offsets / duration) a new face at the same frame
# index and box position as a kept (curated) row from the old sampling is a
# different row, not a silently dropped duplicate.
_LIBRARY_FACES_KEY_DDL = """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_lib_faces_key ON library_faces(
        stash_scene_id, frame_index, IFNULL(timestamp_sec, -1.0), bbox_x, bbox_y, bbox_w, bbox_h);
    CREATE INDEX IF NOT EXISTS idx_lib_faces_scene ON library_faces(stash_scene_id);
    CREATE INDEX IF NOT EXISTS idx_lib_faces_match ON library_faces(best_match_id);
"""


def _rebuild_library_faces_v13(conn: sqlite3.Connection) -> None:
    """Rebuild library_faces without the inline UNIQUE(scene, frame, x, y).

    Standard SQLite table rebuild with foreign keys OFF, so the members and
    rejections that reference library_faces(id) are not cascaded away. Ids and
    the AUTOINCREMENT sequence are preserved. Commits.
    """
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        row = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'library_faces'"
        ).fetchone()
        seq = row[0] if row else 0
        conn.execute("DROP TABLE IF EXISTS library_faces_v13")
        conn.executescript("BEGIN;" + _library_faces_ddl("library_faces_v13") + f"""
            INSERT INTO library_faces_v13 ({_LIBRARY_FACES_COLUMNS})
                SELECT {_LIBRARY_FACES_COLUMNS} FROM library_faces;
            DROP TABLE library_faces;
            ALTER TABLE library_faces_v13 RENAME TO library_faces;
        """ + _LIBRARY_FACES_KEY_DDL)
        conn.execute(
            "UPDATE sqlite_sequence SET seq = MAX(seq, ?) WHERE name = 'library_faces'", (seq,)
        )
        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        if bad:
            raise sqlite3.IntegrityError(f"foreign key violations after library_faces rebuild: {bad[:5]}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _dedupe_matched_clusters_v12(conn: sqlite3.Connection) -> None:
    """Collapse duplicate matched clusters per stash_id (left by old full builds).

    Keeps the cluster with the most members (ties: lowest id), copies the others'
    members and stash ids into it, then deletes them.
    """
    dup_sids = [r[0] for r in conn.execute(
        """
        SELECT s.stash_id FROM face_cluster_stash_ids s
        JOIN face_clusters c ON c.id = s.cluster_id
        WHERE c.status = 'matched'
        GROUP BY s.stash_id HAVING COUNT(*) > 1
        """
    ).fetchall()]
    for sid in dup_sids:
        rows = conn.execute(
            """
            SELECT c.id, c.pinned,
                   (SELECT COUNT(*) FROM face_cluster_members m WHERE m.cluster_id = c.id) AS n
            FROM face_clusters c
            JOIN face_cluster_stash_ids s ON s.cluster_id = c.id
            WHERE c.status = 'matched' AND s.stash_id = ?
            ORDER BY n DESC, c.id ASC
            """,
            (sid,),
        ).fetchall()
        if len(rows) < 2:
            continue  # already folded by an earlier stash_id in this loop
        keep = rows[0][0]
        for other, pinned, _n in rows[1:]:
            conn.execute(
                "INSERT OR IGNORE INTO face_cluster_members (cluster_id, face_id) "
                "SELECT ?, face_id FROM face_cluster_members WHERE cluster_id = ?",
                (keep, other),
            )
            conn.execute(
                "INSERT OR IGNORE INTO face_cluster_stash_ids (cluster_id, stash_id) "
                "SELECT ?, stash_id FROM face_cluster_stash_ids WHERE cluster_id = ?",
                (keep, other),
            )
            if pinned:
                conn.execute("UPDATE face_clusters SET pinned = 1 WHERE id = ?", (keep,))
            conn.execute("DELETE FROM face_clusters WHERE id = ?", (other,))


def _unit_face_vec(facenet: bytes | None, arcface: bytes | None):
    """Normalized facenet+arcface concat vector of a stored face (None if unusable)."""
    import numpy as np
    if not facenet or not arcface:
        return None
    v = np.concatenate([np.frombuffer(facenet, dtype=np.float32),
                        np.frombuffer(arcface, dtype=np.float32)])
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def _bbox_iou(a: dict, b: dict) -> float:
    """IoU of two {x, y, w, h} boxes. 0 when either box has zero area."""
    ax, ay, aw, ah = (float(a.get(k, 0) or 0) for k in ("x", "y", "w", "h"))
    bx, by, bw, bh = (float(b.get(k, 0) or 0) for k in ("x", "y", "w", "h"))
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Recommendation:
    """A recommendation for user action."""
    id: int
    type: str
    status: str  # 'pending', 'dismissed', 'resolved'
    target_type: str  # 'scene', 'performer', 'studio', 'file'
    target_id: str
    details: dict
    resolution_action: str | None
    resolution_details: dict | None
    resolved_at: str | None
    confidence: float | None
    source_analysis_id: int | None
    created_at: str
    updated_at: str


@dataclass
class AnalysisRun:
    """Record of an analysis run."""
    id: int
    type: str
    status: str  # 'running', 'completed', 'failed'
    started_at: str
    completed_at: str | None
    items_total: int | None
    items_processed: int | None
    recommendations_created: int
    cursor: str | None
    error_message: str | None


@dataclass
class RecommendationSettings:
    """Settings for a recommendation type."""
    type: str
    enabled: bool
    auto_dismiss_threshold: float | None
    notify: bool
    interval_hours: int | None
    last_run_at: str | None
    next_run_at: str | None
    config: dict | None


class RecommendationsDB:
    """
    SQLite database for recommendations and analysis state.

    Usage:
        db = RecommendationsDB("stash_sense.db")

        # Create a recommendation
        rec_id = db.create_recommendation(
            type="duplicate_performer",
            target_type="performer",
            target_id="123",
            details={"duplicate_ids": ["123", "456"], "suggested_keeper": "123"}
        )

        # Get pending recommendations
        recs = db.get_recommendations(status="pending", type="duplicate_performer")

        # Resolve a recommendation
        db.resolve_recommendation(rec_id, action="merged", details={"kept_id": "123"})
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize database schema if needed."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                self._create_schema(conn)
            else:
                version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
                if version < SCHEMA_VERSION:
                    self._migrate_schema(conn, version)

    def _create_schema(self, conn: sqlite3.Connection):
        """Create the database schema."""
        conn.executescript(f"""
            -- Schema version tracking
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY
            );
            INSERT INTO schema_version (version) VALUES ({SCHEMA_VERSION});

            -- Core recommendations table
            CREATE TABLE recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                details JSON NOT NULL,
                resolution_action TEXT,
                resolution_details JSON,
                resolved_at TEXT,
                confidence REAL,
                source_analysis_id INTEGER REFERENCES analysis_runs(id),
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(type, target_type, target_id)
            );
            CREATE INDEX idx_rec_status ON recommendations(status);
            CREATE INDEX idx_rec_type ON recommendations(type);
            CREATE INDEX idx_rec_target ON recommendations(target_type, target_id);

            -- Track analysis runs
            CREATE TABLE analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                items_total INTEGER,
                items_processed INTEGER,
                recommendations_created INTEGER DEFAULT 0,
                cursor TEXT,
                error_message TEXT
            );
            CREATE INDEX idx_analysis_type_status ON analysis_runs(type, status);

            -- User preferences per recommendation type
            CREATE TABLE recommendation_settings (
                type TEXT PRIMARY KEY,
                enabled INTEGER DEFAULT 1,
                auto_dismiss_threshold REAL,
                notify INTEGER DEFAULT 1,
                interval_hours INTEGER,
                last_run_at TEXT,
                next_run_at TEXT,
                config JSON
            );

            -- Dismissed targets (don't re-recommend)
            CREATE TABLE dismissed_targets (
                type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                dismissed_at TEXT DEFAULT (datetime('now')),
                reason TEXT,
                permanent INTEGER DEFAULT 0,
                PRIMARY KEY (type, target_type, target_id)
            );

            -- Track analysis watermarks for incremental runs
            CREATE TABLE analysis_watermarks (
                type TEXT PRIMARY KEY,
                last_completed_at TEXT,
                last_cursor TEXT,
                last_stash_updated_at TEXT,
                logic_version INTEGER DEFAULT 1
            );

            -- Scene fingerprints for duplicate detection
            CREATE TABLE scene_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_scene_id INTEGER NOT NULL UNIQUE,
                total_faces INTEGER NOT NULL DEFAULT 0,
                frames_analyzed INTEGER NOT NULL DEFAULT 0,
                fingerprint_status TEXT NOT NULL DEFAULT 'pending',
                db_version TEXT,  -- Face recognition DB version used to generate this fingerprint
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_scene_fp_stash_id ON scene_fingerprints(stash_scene_id);
            CREATE INDEX idx_scene_fp_status ON scene_fingerprints(fingerprint_status);
            CREATE INDEX idx_scene_fp_db_version ON scene_fingerprints(db_version);

            -- Face entries within scene fingerprints
            CREATE TABLE scene_fingerprint_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                performer_id TEXT NOT NULL,
                face_count INTEGER NOT NULL DEFAULT 0,
                avg_confidence REAL,
                proportion REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(fingerprint_id, performer_id)
            );
            CREATE INDEX idx_scene_fp_faces_fingerprint ON scene_fingerprint_faces(fingerprint_id);
            CREATE INDEX idx_scene_fp_faces_performer ON scene_fingerprint_faces(performer_id);

            -- Image fingerprints for gallery/image identification
            CREATE TABLE image_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_image_id TEXT NOT NULL UNIQUE,
                gallery_id TEXT,
                faces_detected INTEGER NOT NULL DEFAULT 0,
                db_version TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_image_fp_gallery ON image_fingerprints(gallery_id);

            -- Face entries within image fingerprints
            CREATE TABLE image_fingerprint_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_image_id TEXT NOT NULL REFERENCES image_fingerprints(stash_image_id) ON DELETE CASCADE,
                performer_id TEXT NOT NULL,
                confidence REAL,
                distance REAL,
                bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(stash_image_id, performer_id)
            );
            CREATE INDEX idx_image_fp_faces_image ON image_fingerprint_faces(stash_image_id);
            CREATE INDEX idx_image_fp_faces_performer ON image_fingerprint_faces(performer_id);

            -- Upstream sync snapshots
            CREATE TABLE upstream_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                local_entity_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                stash_box_id TEXT NOT NULL,
                upstream_data JSON NOT NULL,
                upstream_updated_at TEXT NOT NULL,
                fetched_at TEXT DEFAULT (datetime('now')),
                UNIQUE(entity_type, endpoint, stash_box_id)
            );
            CREATE INDEX idx_upstream_entity ON upstream_snapshots(entity_type, endpoint);
            CREATE INDEX idx_upstream_stash_box_id ON upstream_snapshots(stash_box_id);

            -- Per-field monitoring configuration
            CREATE TABLE upstream_field_config (
                endpoint TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                field_name TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                PRIMARY KEY (endpoint, entity_type, field_name)
            );

            -- User settings (key-value store)
            CREATE TABLE user_settings (
                key TEXT PRIMARY KEY,
                value JSON NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            -- Seed default settings
            INSERT INTO user_settings (key, value) VALUES ('normalize_enum_display', 'true');

            -- Duplicate scene candidates (work queue for scoring)
            CREATE TABLE duplicate_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_a_id INTEGER NOT NULL,
                scene_b_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                run_id INTEGER REFERENCES analysis_runs(id),
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(scene_a_id, scene_b_id)
            );
            CREATE INDEX idx_dup_candidates_run ON duplicate_candidates(run_id);
            CREATE INDEX idx_dup_candidates_run_id ON duplicate_candidates(run_id, id);

            -- Job queue
            CREATE TABLE IF NOT EXISTS job_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                priority INTEGER NOT NULL,
                cursor TEXT,
                items_total INTEGER,
                items_processed INTEGER DEFAULT 0,
                error_message TEXT,
                triggered_by TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                started_at TEXT,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
            CREATE INDEX IF NOT EXISTS idx_job_queue_type_status ON job_queue(type, status);

            -- Job schedules
            CREATE TABLE IF NOT EXISTS job_schedules (
                type TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0,
                interval_hours REAL NOT NULL,
                priority INTEGER NOT NULL,
                last_run_at TEXT,
                next_run_at TEXT
            );

            -- Per-face persistence for Immich-style face grouping (see _library_faces_ddl).
            -- User-curated face groups.
            -- performer_id is always a LOCAL Stash performer id (or NULL); a group's
            -- StashDB/stash-box identity lives in face_cluster_stash_ids.
            -- pinned = 1: user-curated, never dissolved by a full rebuild.
            CREATE TABLE IF NOT EXISTS face_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                performer_id TEXT,
                performer_name TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                pinned INTEGER NOT NULL DEFAULT 0,
                -- 1 once an assigned group's performer stash ids were fetched from Stash
                stash_ids_synced INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS face_cluster_members (
                cluster_id INTEGER NOT NULL REFERENCES face_clusters(id) ON DELETE CASCADE,
                face_id INTEGER NOT NULL REFERENCES library_faces(id) ON DELETE CASCADE,
                UNIQUE(cluster_id, face_id)
            );
            CREATE INDEX IF NOT EXISTS idx_fc_members_cluster ON face_cluster_members(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_fc_members_face ON face_cluster_members(face_id);

            -- Durable merge history: source groups folded into a target.
            CREATE TABLE IF NOT EXISTS face_cluster_merge_log (
                source_cluster_id INTEGER PRIMARY KEY,
                target_cluster_id INTEGER NOT NULL,
                merged_at TEXT DEFAULT (datetime('now'))
            );
        """ + _library_faces_ddl() + _LIBRARY_FACES_KEY_DDL + _FACE_CLUSTER_V12_DDL
            + _FACE_CLUSTER_V14_DDL)

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int):
        """Migrate schema from older version."""
        if from_version < 2:
            # Add scene fingerprint tables
            conn.executescript("""
                -- Scene fingerprints for duplicate detection
                CREATE TABLE IF NOT EXISTS scene_fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_scene_id INTEGER NOT NULL UNIQUE,
                    total_faces INTEGER NOT NULL DEFAULT 0,
                    frames_analyzed INTEGER NOT NULL DEFAULT 0,
                    fingerprint_status TEXT NOT NULL DEFAULT 'pending',
                    db_version TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_scene_fp_stash_id ON scene_fingerprints(stash_scene_id);
                CREATE INDEX IF NOT EXISTS idx_scene_fp_status ON scene_fingerprints(fingerprint_status);
                CREATE INDEX IF NOT EXISTS idx_scene_fp_db_version ON scene_fingerprints(db_version);

                -- Face entries within scene fingerprints
                CREATE TABLE IF NOT EXISTS scene_fingerprint_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                    performer_id TEXT NOT NULL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    avg_confidence REAL,
                    proportion REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(fingerprint_id, performer_id)
                );
                CREATE INDEX IF NOT EXISTS idx_scene_fp_faces_fingerprint ON scene_fingerprint_faces(fingerprint_id);
                CREATE INDEX IF NOT EXISTS idx_scene_fp_faces_performer ON scene_fingerprint_faces(performer_id);

                -- Update schema version
                UPDATE schema_version SET version = 3;
            """)

        if from_version == 2:
            # Add db_version column to scene_fingerprints
            conn.executescript("""
                ALTER TABLE scene_fingerprints ADD COLUMN db_version TEXT;
                CREATE INDEX IF NOT EXISTS idx_scene_fp_db_version ON scene_fingerprints(db_version);
                UPDATE schema_version SET version = 3;
            """)

        if from_version < 4:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS upstream_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    local_entity_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    stash_box_id TEXT NOT NULL,
                    upstream_data JSON NOT NULL,
                    upstream_updated_at TEXT NOT NULL,
                    fetched_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(entity_type, endpoint, stash_box_id)
                );
                CREATE INDEX IF NOT EXISTS idx_upstream_entity ON upstream_snapshots(entity_type, endpoint);
                CREATE INDEX IF NOT EXISTS idx_upstream_stash_box_id ON upstream_snapshots(stash_box_id);

                CREATE TABLE IF NOT EXISTS upstream_field_config (
                    endpoint TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    PRIMARY KEY (endpoint, entity_type, field_name)
                );

                ALTER TABLE dismissed_targets ADD COLUMN permanent INTEGER DEFAULT 0;

                UPDATE schema_version SET version = 4;
            """)

        if from_version < 5:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value JSON NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                INSERT OR IGNORE INTO user_settings (key, value) VALUES ('normalize_enum_display', 'true');

                UPDATE schema_version SET version = 5;
            """)

        if from_version < 6:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS image_fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_image_id TEXT NOT NULL UNIQUE,
                    gallery_id TEXT,
                    faces_detected INTEGER NOT NULL DEFAULT 0,
                    db_version TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_image_fp_gallery ON image_fingerprints(gallery_id);

                CREATE TABLE IF NOT EXISTS image_fingerprint_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_image_id TEXT NOT NULL REFERENCES image_fingerprints(stash_image_id) ON DELETE CASCADE,
                    performer_id TEXT NOT NULL,
                    confidence REAL,
                    distance REAL,
                    bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(stash_image_id, performer_id)
                );
                CREATE INDEX IF NOT EXISTS idx_image_fp_faces_image ON image_fingerprint_faces(stash_image_id);
                CREATE INDEX IF NOT EXISTS idx_image_fp_faces_performer ON image_fingerprint_faces(performer_id);

                UPDATE schema_version SET version = 6;
            """)

        if from_version < 7:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS duplicate_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_a_id INTEGER NOT NULL,
                    scene_b_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    run_id INTEGER REFERENCES analysis_runs(id),
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(scene_a_id, scene_b_id)
                );
                CREATE INDEX IF NOT EXISTS idx_dup_candidates_run ON duplicate_candidates(run_id);
                CREATE INDEX IF NOT EXISTS idx_dup_candidates_run_id ON duplicate_candidates(run_id, id);

                UPDATE schema_version SET version = 7;
            """)

        if from_version < 8:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS job_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    priority INTEGER NOT NULL,
                    cursor TEXT,
                    items_total INTEGER,
                    items_processed INTEGER DEFAULT 0,
                    error_message TEXT,
                    triggered_by TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
                CREATE INDEX IF NOT EXISTS idx_job_queue_type_status ON job_queue(type, status);

                CREATE TABLE IF NOT EXISTS job_schedules (
                    type TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    interval_hours REAL NOT NULL,
                    priority INTEGER NOT NULL,
                    last_run_at TEXT,
                    next_run_at TEXT
                );

                UPDATE schema_version SET version = 8;
            """)

        if from_version < 9:
            conn.executescript("""
                ALTER TABLE analysis_watermarks ADD COLUMN logic_version INTEGER DEFAULT 1;

                UPDATE schema_version SET version = 9;
            """)

        if from_version < 10:
            conn.executescript("""
                -- Per-face persistence for Immich-style face grouping.
                -- One row per detected face across the whole library.
                CREATE TABLE IF NOT EXISTS library_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_scene_id INTEGER NOT NULL,
                    frame_index INTEGER NOT NULL,
                    timestamp_sec REAL,
                    bbox_x REAL NOT NULL,
                    bbox_y REAL NOT NULL,
                    bbox_w REAL NOT NULL,
                    bbox_h REAL NOT NULL,
                    det_confidence REAL NOT NULL,
                    yaw REAL,
                    facenet_emb BLOB NOT NULL,
                    arcface_emb BLOB NOT NULL,
                    crop_path TEXT,
                    best_match_id TEXT,
                    best_match_name TEXT,
                    best_match_confidence REAL,
                    db_version TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(stash_scene_id, frame_index, bbox_x, bbox_y)
                );
                CREATE INDEX IF NOT EXISTS idx_lib_faces_scene ON library_faces(stash_scene_id);
                CREATE INDEX IF NOT EXISTS idx_lib_faces_match ON library_faces(best_match_id);

                -- User-curated face groups (the Immich-style "unnamed person" buckets).
                CREATE TABLE IF NOT EXISTS face_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    performer_id TEXT,
                    performer_name TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                -- Faces assigned to clusters.
                CREATE TABLE IF NOT EXISTS face_cluster_members (
                    cluster_id INTEGER NOT NULL REFERENCES face_clusters(id) ON DELETE CASCADE,
                    face_id INTEGER NOT NULL REFERENCES library_faces(id) ON DELETE CASCADE,
                    UNIQUE(cluster_id, face_id)
                );
                CREATE INDEX IF NOT EXISTS idx_fc_members_cluster ON face_cluster_members(cluster_id);
                CREATE INDEX IF NOT EXISTS idx_fc_members_face ON face_cluster_members(face_id);

                UPDATE schema_version SET version = 10;
            """)

        if from_version < 11:
            conn.executescript("""
                -- Durable merge history: source groups that were folded into a target.
                CREATE TABLE IF NOT EXISTS face_cluster_merge_log (
                    source_cluster_id INTEGER PRIMARY KEY,
                    target_cluster_id INTEGER NOT NULL,
                    merged_at TEXT DEFAULT (datetime('now'))
                );

                UPDATE schema_version SET version = 11;
            """)

        if from_version < 12:
            # Idempotent DDL (a failed earlier attempt may have added the column already).
            cols = {r[1] for r in conn.execute("PRAGMA table_info(face_clusters)")}
            if "pinned" not in cols:
                conn.execute(
                    "ALTER TABLE face_clusters ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
                )
            conn.executescript(_FACE_CLUSTER_V12_DDL)
            # Data fix-ups, the matched-cluster dedupe and the version bump share one
            # transaction: executescript leaves the BEGIN open, the connection
            # context manager commits (or rolls back) all of it together.
            conn.executescript("""
                BEGIN;
                -- matched groups: move the StashDB anchor out of performer_id
                INSERT OR IGNORE INTO face_cluster_stash_ids (cluster_id, stash_id)
                    SELECT id, performer_id FROM face_clusters
                    WHERE status = 'matched' AND performer_id IS NOT NULL;
                UPDATE face_clusters SET performer_id = NULL WHERE status = 'matched';
                UPDATE face_clusters SET pinned = 1
                    WHERE status IN ('assigned', 'ignored', 'banned')
                       OR id IN (SELECT target_cluster_id FROM face_cluster_merge_log)
                       OR (status = 'open' AND name IS NOT NULL);
                -- a face in a curated group loses any extra open/matched membership
                DELETE FROM face_cluster_members WHERE rowid IN (
                    SELECT m.rowid FROM face_cluster_members m
                    JOIN face_clusters c ON c.id = m.cluster_id
                    WHERE c.status IN ('open', 'matched') AND EXISTS (
                        SELECT 1 FROM face_cluster_members m2
                        JOIN face_clusters c2 ON c2.id = m2.cluster_id
                        WHERE m2.face_id = m.face_id AND c2.id != c.id
                          AND c2.status IN ('assigned', 'ignored', 'banned')));
                -- ghost empty banned groups left by the old re-identify cascade
                DELETE FROM face_clusters
                    WHERE status = 'banned'
                      AND id NOT IN (SELECT cluster_id FROM face_cluster_members);
            """)
            _dedupe_matched_clusters_v12(conn)
            conn.execute("UPDATE schema_version SET version = 12")

        if from_version < 13:
            # 1. UNIQUE key gains the timestamp (table rebuild, FKs off, own commit).
            _rebuild_library_faces_v13(conn)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(face_clusters)")}
            if "stash_ids_synced" not in cols:
                conn.execute(
                    "ALTER TABLE face_clusters ADD COLUMN stash_ids_synced INTEGER NOT NULL DEFAULT 0"
                )
            # 2. Anchors written before v13 are untrustworthy: the persist step
            #    mapped faces to persons by position (wrong in multi-person
            #    scenes), fell back to persons[-1], and hybrid mode handed faces
            #    to a performer through secondary matches. Clear them (the next
            #    identify of each scene rewrites them) and dissolve the
            #    automatic matched groups seeded from them; their faces return
            #    to the pool and are re-placed by embedding. Pinned (curated)
            #    groups are kept. The fingerprint job skips scenes whose
            #    fingerprint is current, so mark those scenes outdated.
            conn.executescript("""
                BEGIN;
                UPDATE library_faces
                    SET best_match_id = NULL, best_match_name = NULL, best_match_confidence = NULL;
                UPDATE scene_fingerprints SET db_version = NULL, updated_at = datetime('now')
                    WHERE stash_scene_id IN (SELECT DISTINCT stash_scene_id FROM library_faces);
                DELETE FROM face_clusters WHERE status = 'matched' AND pinned = 0;
                UPDATE schema_version SET version = 13;
            """)

        if from_version < 14:
            # Blocked-anchor memory, and the face key gains the box size (two
            # detections sharing a top-left corner at one moment are distinct
            # faces). Widening a UNIQUE key never fails on existing rows.
            conn.executescript("BEGIN;" + _FACE_CLUSTER_V14_DDL + """
                DROP INDEX IF EXISTS uq_lib_faces_key;
            """ + _LIBRARY_FACES_KEY_DDL + """
                -- "synced" now means "the performer is linked on StashDB";
                -- v13 also set it after lookups that found no such link.
                -- Re-check every assigned group once.
                UPDATE face_clusters SET stash_ids_synced = 0;
                UPDATE schema_version SET version = 14;
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with row factory."""
        # Generous busy timeout: a build reading the face pool and a scene
        # persist committing can overlap for a while on large libraries.
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ==================== Recommendations ====================

    def create_recommendation(
        self,
        type: str,
        target_type: str,
        target_id: str,
        details: dict,
        confidence: float | None = None,
        source_analysis_id: int | None = None,
    ) -> int | None:
        """
        Create a recommendation. Returns ID if created, None if duplicate.
        """
        with self._connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO recommendations (
                        type, target_type, target_id, details, confidence, source_analysis_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (type, target_type, target_id, json.dumps(details), confidence, source_analysis_id)
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Already exists
                return None

    def get_recommendation(self, rec_id: int) -> Recommendation | None:
        """Get a recommendation by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM recommendations WHERE id = ?", (rec_id,)
            ).fetchone()
            if row:
                return self._row_to_recommendation(row)
        return None

    def get_recommendations(
        self,
        status: str | None = None,
        type: str | None = None,
        target_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Recommendation]:
        """Get recommendations with optional filtering."""
        query = "SELECT * FROM recommendations WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)
        if target_type:
            query += " AND target_type = ?"
            params.append(target_type)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_recommendation(row) for row in rows]

    def count_recommendations(self, status=None, type=None, target_type=None) -> int:
        """Count recommendations with optional filtering (for pagination totals)."""
        query = "SELECT COUNT(*) FROM recommendations WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)
        if target_type:
            query += " AND target_type = ?"
            params.append(target_type)
        with self._connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    def get_recommendation_by_target(
        self,
        type: str,
        target_type: str,
        target_id: str,
        status: str | None = None,
    ) -> Recommendation | None:
        """Get a recommendation by target (uses idx_rec_target index). Returns first match or None."""
        query = "SELECT * FROM recommendations WHERE type = ? AND target_type = ? AND target_id = ?"
        params: list = [type, target_type, target_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " LIMIT 1"

        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()
            if row:
                return self._row_to_recommendation(row)
        return None

    def get_recommendation_counts(self) -> dict[str, dict[str, int]]:
        """Get counts by type and status."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT type, status, COUNT(*) as count FROM recommendations GROUP BY type, status"
            ).fetchall()

            counts = {}
            for row in rows:
                if row['type'] not in counts:
                    counts[row['type']] = {}
                counts[row['type']][row['status']] = row['count']
            return counts

    def resolve_recommendation(
        self,
        rec_id: int,
        action: str,
        details: dict | None = None,
    ) -> bool:
        """Mark a recommendation as resolved. Returns True if updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET status = 'resolved',
                    resolution_action = ?,
                    resolution_details = ?,
                    resolved_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (action, json.dumps(details) if details else None, rec_id)
            )
            return cursor.rowcount > 0

    def dismiss_recommendation(self, rec_id: int, reason: str | None = None, permanent: bool = False) -> bool:
        """Dismiss a recommendation and add to dismissed_targets."""
        with self._connection() as conn:
            # Get the recommendation first
            row = conn.execute(
                "SELECT type, target_type, target_id FROM recommendations WHERE id = ?",
                (rec_id,)
            ).fetchone()

            if not row:
                return False

            # Mark as dismissed
            conn.execute(
                """
                UPDATE recommendations
                SET status = 'dismissed', updated_at = datetime('now')
                WHERE id = ?
                """,
                (rec_id,)
            )

            # Add to dismissed_targets to prevent re-recommendation
            try:
                conn.execute(
                    """
                    INSERT INTO dismissed_targets (type, target_type, target_id, reason, permanent)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (row['type'], row['target_type'], row['target_id'], reason, int(permanent))
                )
            except sqlite3.IntegrityError:
                pass  # Already dismissed

            return True

    def batch_dismiss_by_type(self, rec_type: str, permanent: bool = False, reason: str | None = None) -> int:
        """Dismiss all pending recommendations of a given type. Returns count dismissed."""
        with self._connection() as conn:
            # Get all pending recs of this type
            rows = conn.execute(
                "SELECT id, type, target_type, target_id FROM recommendations WHERE type = ? AND status = 'pending'",
                (rec_type,)
            ).fetchall()

            if not rows:
                return 0

            rec_ids = [row['id'] for row in rows]

            # Mark all as dismissed
            conn.execute(
                f"UPDATE recommendations SET status = 'dismissed', updated_at = datetime('now') WHERE id IN ({','.join('?' * len(rec_ids))})",
                rec_ids
            )

            # Add to dismissed_targets
            for row in rows:
                try:
                    conn.execute(
                        "INSERT INTO dismissed_targets (type, target_type, target_id, reason, permanent) VALUES (?, ?, ?, ?, ?)",
                        (row['type'], row['target_type'], row['target_id'], reason, int(permanent))
                    )
                except sqlite3.IntegrityError:
                    pass  # Already dismissed

            return len(rec_ids)

    def is_dismissed(self, type: str, target_type: str, target_id: str) -> bool:
        """Check if a target has been dismissed."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ?
                """,
                (type, target_type, target_id)
            ).fetchone()
            return row is not None

    def is_permanently_dismissed(self, type: str, target_type: str, target_id: str) -> bool:
        """Check if a target has been permanently dismissed."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ? AND permanent = 1
                """,
                (type, target_type, target_id)
            ).fetchone()
            return row is not None

    def undismiss(self, type: str, target_type: str, target_id: str):
        """Remove soft dismissals for a target (does not remove permanent dismissals)."""
        with self._connection() as conn:
            conn.execute(
                """
                DELETE FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ? AND permanent = 0
                """,
                (type, target_type, target_id)
            )

    def update_recommendation_details(self, rec_id: int, details: dict) -> bool:
        """Update details on a pending recommendation. Returns True if updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET details = ?, updated_at = datetime('now')
                WHERE id = ? AND status = 'pending'
                """,
                (json.dumps(details), rec_id)
            )
            return cursor.rowcount > 0

    def reopen_recommendation(self, rec_id: int, details: dict) -> bool:
        """Reopen a dismissed recommendation with new details. Returns True if updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET status = 'pending', details = ?,
                    resolution_action = NULL, resolution_details = NULL,
                    resolved_at = NULL, updated_at = datetime('now')
                WHERE id = ? AND status = 'dismissed'
                """,
                (json.dumps(details), rec_id)
            )
            return cursor.rowcount > 0

    def _row_to_recommendation(self, row: sqlite3.Row) -> Recommendation:
        """Convert a database row to a Recommendation object."""
        return Recommendation(
            id=row['id'],
            type=row['type'],
            status=row['status'],
            target_type=row['target_type'],
            target_id=row['target_id'],
            details=json.loads(row['details']),
            resolution_action=row['resolution_action'],
            resolution_details=json.loads(row['resolution_details']) if row['resolution_details'] else None,
            resolved_at=row['resolved_at'],
            confidence=row['confidence'],
            source_analysis_id=row['source_analysis_id'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    # ==================== Analysis Runs ====================

    def start_analysis_run(self, type: str, items_total: int | None = None) -> int:
        """Start a new analysis run. Returns run ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_runs (type, status, started_at, items_total)
                VALUES (?, 'running', datetime('now'), ?)
                """,
                (type, items_total)
            )
            return cursor.lastrowid

    def update_analysis_progress(
        self,
        run_id: int,
        items_processed: int,
        recommendations_created: int,
        cursor: str | None = None,
    ):
        """Update analysis run progress."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET items_processed = ?, recommendations_created = ?, cursor = ?
                WHERE id = ?
                """,
                (items_processed, recommendations_created, cursor, run_id)
            )

    def fail_stale_analysis_runs(self) -> int:
        """Mark any 'running' analysis runs as failed (e.g. after sidecar restart). Returns count."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'failed', completed_at = datetime('now'),
                    error_message = 'Sidecar restarted while analysis was running'
                WHERE status = 'running'
                """
            )
            return cursor.rowcount

    def update_analysis_items_total(self, run_id: int, items_total: int):
        """Update the total items count for an analysis run."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE analysis_runs SET items_total = ? WHERE id = ?",
                (items_total, run_id)
            )

    def complete_analysis_run(self, run_id: int, recommendations_created: int):
        """Mark an analysis run as completed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'completed', completed_at = datetime('now'), recommendations_created = ?
                WHERE id = ?
                """,
                (recommendations_created, run_id)
            )

    def fail_analysis_run(self, run_id: int, error_message: str):
        """Mark an analysis run as failed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'failed', completed_at = datetime('now'), error_message = ?
                WHERE id = ?
                """,
                (error_message, run_id)
            )

    def get_analysis_run(self, run_id: int) -> AnalysisRun | None:
        """Get an analysis run by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row:
                return AnalysisRun(**dict(row))
        return None

    def get_recent_analysis_runs(self, type: str | None = None, limit: int = 20) -> list[AnalysisRun]:
        """Get recent analysis runs."""
        query = "SELECT * FROM analysis_runs"
        params = []

        if type:
            query += " WHERE type = ?"
            params.append(type)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [AnalysisRun(**dict(row)) for row in rows]

    # ==================== Settings ====================

    def get_settings(self, type: str) -> RecommendationSettings | None:
        """Get settings for a recommendation type."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM recommendation_settings WHERE type = ?", (type,)
            ).fetchone()
            if row:
                return RecommendationSettings(
                    type=row['type'],
                    enabled=bool(row['enabled']),
                    auto_dismiss_threshold=row['auto_dismiss_threshold'],
                    notify=bool(row['notify']),
                    interval_hours=row['interval_hours'],
                    last_run_at=row['last_run_at'],
                    next_run_at=row['next_run_at'],
                    config=json.loads(row['config']) if row['config'] else None,
                )
        return None

    def get_all_settings(self) -> list[RecommendationSettings]:
        """Get all recommendation settings."""
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM recommendation_settings").fetchall()
            return [
                RecommendationSettings(
                    type=row['type'],
                    enabled=bool(row['enabled']),
                    auto_dismiss_threshold=row['auto_dismiss_threshold'],
                    notify=bool(row['notify']),
                    interval_hours=row['interval_hours'],
                    last_run_at=row['last_run_at'],
                    next_run_at=row['next_run_at'],
                    config=json.loads(row['config']) if row['config'] else None,
                )
                for row in rows
            ]

    def upsert_settings(
        self,
        type: str,
        enabled: bool | None = None,
        auto_dismiss_threshold: float | None = None,
        notify: bool | None = None,
        interval_hours: int | None = None,
        config: dict | None = None,
    ):
        """Create or update settings for a recommendation type."""
        with self._connection() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT 1 FROM recommendation_settings WHERE type = ?", (type,)
            ).fetchone()

            if existing:
                updates = []
                params = []
                if enabled is not None:
                    updates.append("enabled = ?")
                    params.append(int(enabled))
                if auto_dismiss_threshold is not None:
                    updates.append("auto_dismiss_threshold = ?")
                    params.append(auto_dismiss_threshold)
                if notify is not None:
                    updates.append("notify = ?")
                    params.append(int(notify))
                if interval_hours is not None:
                    updates.append("interval_hours = ?")
                    params.append(interval_hours)
                if config is not None:
                    updates.append("config = ?")
                    params.append(json.dumps(config))

                if updates:
                    params.append(type)
                    conn.execute(
                        f"UPDATE recommendation_settings SET {', '.join(updates)} WHERE type = ?",
                        params
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO recommendation_settings (type, enabled, auto_dismiss_threshold, notify, interval_hours, config)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (type, int(enabled) if enabled is not None else 1,
                     auto_dismiss_threshold, int(notify) if notify is not None else 1,
                     interval_hours, json.dumps(config) if config else None)
                )

    # ==================== Watermarks ====================

    def get_watermark(self, type: str) -> dict | None:
        """Get analysis watermark for incremental runs."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_watermarks WHERE type = ?", (type,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def set_watermark(
        self,
        type: str,
        last_cursor: str | None = None,
        last_stash_updated_at: str | None = None,
        logic_version: int | None = None,
    ):
        """Update analysis watermark."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_watermarks (type, last_completed_at, last_cursor, last_stash_updated_at, logic_version)
                VALUES (?, datetime('now'), ?, ?, COALESCE(?, 1))
                ON CONFLICT(type) DO UPDATE SET
                    last_completed_at = datetime('now'),
                    last_cursor = COALESCE(?, last_cursor),
                    last_stash_updated_at = COALESCE(?, last_stash_updated_at),
                    logic_version = COALESCE(?, logic_version)
                """,
                (type, last_cursor, last_stash_updated_at, logic_version,
                 last_cursor, last_stash_updated_at, logic_version)
            )

    def delete_watermark(self, type: str):
        """Delete a watermark entry."""
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM analysis_watermarks WHERE type = ?", (type,)
            )

    # ==================== Upstream Snapshots ====================

    def upsert_upstream_snapshot(
        self,
        entity_type: str,
        local_entity_id: str,
        endpoint: str,
        stash_box_id: str,
        upstream_data: dict,
        upstream_updated_at: str,
    ) -> int:
        """
        Create or update an upstream snapshot. Returns the snapshot ID.
        Uses upsert on the unique constraint (entity_type, endpoint, stash_box_id).
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO upstream_snapshots (
                    entity_type, local_entity_id, endpoint, stash_box_id,
                    upstream_data, upstream_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, endpoint, stash_box_id) DO UPDATE SET
                    local_entity_id = excluded.local_entity_id,
                    upstream_data = excluded.upstream_data,
                    upstream_updated_at = excluded.upstream_updated_at,
                    fetched_at = datetime('now')
                RETURNING id
                """,
                (entity_type, local_entity_id, endpoint, stash_box_id,
                 json.dumps(upstream_data), upstream_updated_at)
            )
            return cursor.fetchone()[0]

    def get_upstream_snapshot(
        self,
        entity_type: str,
        endpoint: str,
        stash_box_id: str,
    ) -> dict | None:
        """Get an upstream snapshot by its unique key. Returns dict with parsed upstream_data, or None."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM upstream_snapshots
                WHERE entity_type = ? AND endpoint = ? AND stash_box_id = ?
                """,
                (entity_type, endpoint, stash_box_id)
            ).fetchone()
            if row:
                result = dict(row)
                result["upstream_data"] = json.loads(result["upstream_data"])
                return result
        return None

    def delete_snapshots_for_endpoint(self, entity_type: str, endpoint: str) -> int:
        """Delete all upstream snapshots for an entity type + endpoint. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM upstream_snapshots WHERE entity_type = ? AND endpoint = ?",
                (entity_type, endpoint)
            )
            return cursor.rowcount

    # ==================== Upstream Field Config ====================

    def get_enabled_fields(self, endpoint: str, entity_type: str) -> set[str] | None:
        """
        Get the set of enabled field names for an endpoint/entity_type.
        Returns None if no config exists (caller should use defaults).
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT field_name, enabled FROM upstream_field_config
                WHERE endpoint = ? AND entity_type = ?
                """,
                (endpoint, entity_type)
            ).fetchall()
            if not rows:
                return None
            return {row["field_name"] for row in rows if row["enabled"]}

    def set_field_config(self, endpoint: str, entity_type: str, field_configs: dict[str, bool]):
        """
        Set field monitoring configuration for an endpoint/entity_type.
        Replaces all existing config for this endpoint/entity_type.
        field_configs maps field_name -> enabled bool.
        """
        with self._connection() as conn:
            # Delete existing config for this endpoint/entity_type
            conn.execute(
                """
                DELETE FROM upstream_field_config
                WHERE endpoint = ? AND entity_type = ?
                """,
                (endpoint, entity_type)
            )
            # Insert new config rows
            for field_name, enabled in field_configs.items():
                conn.execute(
                    """
                    INSERT INTO upstream_field_config (endpoint, entity_type, field_name, enabled)
                    VALUES (?, ?, ?, ?)
                    """,
                    (endpoint, entity_type, field_name, int(enabled))
                )

    # ==================== User Settings ====================

    def get_user_setting(self, key: str) -> Any | None:
        """Get a user setting by key. Returns the parsed JSON value, or None if not found."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM user_settings WHERE key = ?", (key,)
            ).fetchone()
            if row:
                val = row["value"]
                # SQLite NUMERIC affinity may auto-convert JSON numbers from
                # str to int/float. Return those directly; parse strings as JSON.
                if not isinstance(val, str):
                    return val
                return json.loads(val)
        return None

    def set_user_setting(self, key: str, value: Any):
        """Set a user setting. Creates or updates."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = datetime('now')
                """,
                (key, json.dumps(value))
            )

    def get_all_user_settings(self) -> dict[str, Any]:
        """Get all user settings as a dict."""
        with self._connection() as conn:
            rows = conn.execute("SELECT key, value FROM user_settings").fetchall()
            result = {}
            for row in rows:
                val = row["value"]
                result[row["key"]] = val if not isinstance(val, str) else json.loads(val)
            return result

    def delete_user_setting(self, key: str):
        """Delete a user setting by key."""
        with self._connection() as conn:
            conn.execute("DELETE FROM user_settings WHERE key = ?", (key,))

    # ==================== Endpoint Priorities ====================

    def get_endpoint_priorities(self) -> list[str]:
        """Get the ordered list of endpoint URLs, highest priority first.

        Returns empty list if no priorities are configured.
        """
        result = self.get_user_setting("endpoint_priorities")
        if isinstance(result, list):
            return result
        return []

    def set_endpoint_priorities(self, endpoints: list[str]):
        """Set the endpoint priority order. Index 0 = highest priority."""
        self.set_user_setting("endpoint_priorities", endpoints)

    # ==================== Disabled Endpoints ====================

    def get_disabled_endpoints(self) -> list[str]:
        """Get the list of disabled endpoint URLs."""
        result = self.get_user_setting("disabled_endpoints")
        if isinstance(result, list):
            return result
        return []

    def set_disabled_endpoints(self, endpoints: list[str]):
        """Set the list of disabled endpoint URLs."""
        self.set_user_setting("disabled_endpoints", endpoints)

    def is_endpoint_enabled(self, endpoint: str) -> bool:
        """Check if an endpoint is enabled (not in disabled list)."""
        return endpoint not in self.get_disabled_endpoints()

    # ==================== Scene Fingerprints ====================

    def create_scene_fingerprint(
        self,
        stash_scene_id: int,
        total_faces: int,
        frames_analyzed: int,
        fingerprint_status: str = "pending",
        db_version: str | None = None,
    ) -> int:
        """
        Create or update a scene fingerprint. Returns the fingerprint ID.
        Uses upsert - if fingerprint exists for scene, updates it.

        Args:
            stash_scene_id: The Stash scene ID
            total_faces: Total faces detected in the scene
            frames_analyzed: Number of frames analyzed
            fingerprint_status: Status ('pending', 'complete', 'error')
            db_version: Face recognition DB version used for this fingerprint
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scene_fingerprints (stash_scene_id, total_faces, frames_analyzed, fingerprint_status, db_version)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(stash_scene_id) DO UPDATE SET
                    total_faces = excluded.total_faces,
                    frames_analyzed = excluded.frames_analyzed,
                    fingerprint_status = excluded.fingerprint_status,
                    db_version = excluded.db_version,
                    updated_at = datetime('now')
                RETURNING id
                """,
                (stash_scene_id, total_faces, frames_analyzed, fingerprint_status, db_version)
            )
            return cursor.fetchone()[0]

    def get_scene_fingerprint(self, stash_scene_id: int) -> dict | None:
        """Get a scene fingerprint by stash scene ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scene_fingerprints WHERE stash_scene_id = ?",
                (stash_scene_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_all_scene_fingerprints(self, status: str | None = None) -> list[dict]:
        """Get all scene fingerprints, optionally filtered by status."""
        with self._connection() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM scene_fingerprints WHERE fingerprint_status = ?",
                    (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM scene_fingerprints").fetchall()
            return [dict(row) for row in rows]

    def add_fingerprint_face(
        self,
        fingerprint_id: int,
        performer_id: str,
        face_count: int,
        avg_confidence: float | None = None,
        proportion: float | None = None,
    ) -> int:
        """Add or update a face entry in a scene fingerprint. Returns the face entry ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scene_fingerprint_faces (fingerprint_id, performer_id, face_count, avg_confidence, proportion)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint_id, performer_id) DO UPDATE SET
                    face_count = excluded.face_count,
                    avg_confidence = excluded.avg_confidence,
                    proportion = excluded.proportion
                RETURNING id
                """,
                (fingerprint_id, performer_id, face_count, avg_confidence, proportion)
            )
            return cursor.fetchone()[0]

    def get_fingerprint_faces(self, fingerprint_id: int) -> list[dict]:
        """Get all face entries for a scene fingerprint."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scene_fingerprint_faces WHERE fingerprint_id = ?",
                (fingerprint_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_fingerprint_faces(self, fingerprint_id: int) -> int:
        """Delete all face entries for a scene fingerprint. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM scene_fingerprint_faces WHERE fingerprint_id = ?",
                (fingerprint_id,)
            )
            return cursor.rowcount

    def get_fingerprints_needing_refresh(self, current_db_version: str) -> list[dict]:
        """Get fingerprints that were generated with an older DB version."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scene_fingerprints
                WHERE db_version IS NULL OR db_version != ?
                ORDER BY stash_scene_id
                """,
                (current_db_version,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_scene_ids_without_fingerprints(self, scene_ids: list[int]) -> list[int]:
        """Given a list of scene IDs, return those without fingerprints."""
        if not scene_ids:
            return []
        with self._connection() as conn:
            placeholders = ",".join("?" * len(scene_ids))
            rows = conn.execute(
                f"""
                SELECT stash_scene_id FROM scene_fingerprints
                WHERE stash_scene_id IN ({placeholders})
                """,
                scene_ids
            ).fetchall()
            existing = {row[0] for row in rows}
            return [sid for sid in scene_ids if sid not in existing]

    def get_fingerprint_stats(self, current_db_version: str | None = None) -> dict:
        """Get fingerprint coverage statistics."""
        with self._connection() as conn:
            stats = {}
            stats['total_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints"
            ).fetchone()[0]
            stats['complete_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints WHERE fingerprint_status = 'complete'"
            ).fetchone()[0]
            stats['pending_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints WHERE fingerprint_status = 'pending'"
            ).fetchone()[0]
            stats['error_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints WHERE fingerprint_status = 'error'"
            ).fetchone()[0]

            if current_db_version:
                stats['current_version_count'] = conn.execute(
                    "SELECT COUNT(*) FROM scene_fingerprints WHERE db_version = ?",
                    (current_db_version,)
                ).fetchone()[0]
                stats['needs_refresh_count'] = conn.execute(
                    "SELECT COUNT(*) FROM scene_fingerprints WHERE db_version IS NULL OR db_version != ?",
                    (current_db_version,)
                ).fetchone()[0]

            return stats

    def mark_fingerprints_for_refresh(self, scene_ids: list[int] | None = None) -> int:
        """
        Mark fingerprints for refresh by clearing their db_version.
        If scene_ids is None, marks all fingerprints.
        Returns count of fingerprints marked.
        """
        with self._connection() as conn:
            if scene_ids is None:
                cursor = conn.execute(
                    "UPDATE scene_fingerprints SET db_version = NULL, updated_at = datetime('now')"
                )
            else:
                placeholders = ",".join("?" * len(scene_ids))
                cursor = conn.execute(
                    f"""
                    UPDATE scene_fingerprints
                    SET db_version = NULL, updated_at = datetime('now')
                    WHERE stash_scene_id IN ({placeholders})
                    """,
                    scene_ids
                )
            return cursor.rowcount

    # ==================== Library Faces ====================

    def add_library_face(
        self,
        stash_scene_id: int,
        frame_index: int,
        timestamp_sec: float | None,
        bbox: dict,
        det_confidence: float,
        yaw: float | None,
        facenet_emb: bytes,
        arcface_emb: bytes,
        crop_path: str | None = None,
        best_match_id: str | None = None,
        best_match_name: str | None = None,
        best_match_confidence: float | None = None,
        db_version: str | None = None,
    ) -> int | None:
        """Insert one raw per-face record. Returns face ID, or None if duplicate.

        Low-level insert for tests and tooling only: it applies none of the
        anchoring rules (best_match_id is stored as given) and none of the
        re-identify pairing. Identification writes go through
        library_face_persist.persist_from_identify -> upsert_scene_library_faces.
        """
        with self._connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO library_faces (
                        stash_scene_id, frame_index, timestamp_sec,
                        bbox_x, bbox_y, bbox_w, bbox_h, det_confidence, yaw,
                        facenet_emb, arcface_emb, crop_path,
                        best_match_id, best_match_name, best_match_confidence, db_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stash_scene_id, frame_index, timestamp_sec,
                        bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0),
                        det_confidence, yaw, facenet_emb, arcface_emb, crop_path,
                        best_match_id, best_match_name, best_match_confidence, db_version,
                    )
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def delete_library_faces_for_scene(self, stash_scene_id: int) -> int:
        """Delete all per-face records for a scene (used when re-identifying)."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM library_faces WHERE stash_scene_id = ?",
                (stash_scene_id,)
            )
            return cursor.rowcount

    def get_library_face_count(self) -> int:
        with self._connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM library_faces").fetchone()[0]

    def get_library_face(self, face_id: int) -> dict | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM library_faces WHERE id = ?", (face_id,)).fetchone()
            return dict(row) if row else None

    def get_representative_faces(self, cluster_id: int, limit: int = 12) -> list[dict]:
        """Get sample faces (with crops) for a cluster, best-confidence first."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT lf.id, lf.stash_scene_id, lf.crop_path, lf.det_confidence,
                       lf.best_match_id, lf.best_match_name, lf.best_match_confidence,
                       lf.frame_index, lf.timestamp_sec
                FROM face_cluster_members m
                JOIN library_faces lf ON lf.id = m.face_id
                WHERE m.cluster_id = ?
                ORDER BY lf.det_confidence DESC
                LIMIT ?
                """,
                (cluster_id, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_cluster_scene_ids(self, cluster_id: int) -> list[int]:
        """Distinct scene IDs containing faces of a cluster."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT lf.stash_scene_id
                FROM face_cluster_members m
                JOIN library_faces lf ON lf.id = m.face_id
                WHERE m.cluster_id = ?
                ORDER BY lf.stash_scene_id
                """,
                (cluster_id,)
            ).fetchall()
            return [r[0] for r in rows]

    def get_cluster_face_count(self, cluster_id: int) -> int:
        with self._connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM face_cluster_members WHERE cluster_id = ?",
                (cluster_id,)
            ).fetchone()[0]

    def get_cluster_top_matches(self, cluster_id: int, limit: int = 5) -> list[dict]:
        """Top best-match performers (from identify time) across a cluster's faces."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT best_match_id, best_match_name,
                       COUNT(*) AS n,
                       AVG(best_match_confidence) AS avg_conf
                FROM face_cluster_members m
                JOIN library_faces lf ON lf.id = m.face_id
                WHERE m.cluster_id = ? AND best_match_id IS NOT NULL
                GROUP BY best_match_id
                ORDER BY n DESC
                LIMIT ?
                """,
                (cluster_id, limit)
            ).fetchall()
            return [
                {"performer_id": r[0], "name": r[1], "face_count": r[2], "avg_confidence": r[3]}
                for r in rows
            ]

    # ==================== Face Clusters ====================

    def create_face_cluster(
        self,
        name: str | None = None,
        status: str = "open",
        performer_id: str | None = None,
        performer_name: str | None = None,
        pinned: bool = False,
        stash_ids: list[str] | None = None,
    ) -> int:
        """Create a cluster (and its stash ids) in one transaction.

        performer_id must be a LOCAL Stash performer id or None; StashDB identity
        goes in stash_ids.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO face_clusters (name, status, performer_id, performer_name, pinned)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, status, performer_id, performer_name, 1 if pinned else 0)
            )
            cluster_id = cursor.lastrowid
            if stash_ids:
                conn.executemany(
                    "INSERT OR IGNORE INTO face_cluster_stash_ids (cluster_id, stash_id) VALUES (?, ?)",
                    [(cluster_id, sid) for sid in stash_ids]
                )
            return cluster_id

    def update_face_cluster(
        self,
        cluster_id: int,
        name: str | None = None,
        status: str | None = None,
        performer_id: str | None = None,
        performer_name: str | None = None,
        pinned: bool | None = None,
    ) -> bool:
        """Partial update. Only non-None fields change."""
        with self._connection() as conn:
            sets, params = [], []
            if pinned is not None:
                sets.append("pinned = ?")
                params.append(1 if pinned else 0)
            if name is not None:
                sets.append("name = ?")
                params.append(name)
            if status is not None:
                sets.append("status = ?")
                params.append(status)
            if performer_id is not None:
                sets.append("performer_id = ?")
                params.append(performer_id)
            if performer_name is not None:
                sets.append("performer_name = ?")
                params.append(performer_name)
            if not sets:
                return False
            sets.append("updated_at = datetime('now')")
            params.append(cluster_id)
            cursor = conn.execute(
                f"UPDATE face_clusters SET {', '.join(sets)} WHERE id = ?",
                params
            )
            return cursor.rowcount > 0

    def delete_face_cluster(self, cluster_id: int, only_unpinned: bool = False) -> bool:
        """Delete a cluster (members cascade back to the pool). only_unpinned:
        skip it if it is pinned *at delete time* (a curation that raced ahead
        of a build wins)."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM face_clusters WHERE id = ?" + (" AND pinned = 0" if only_unpinned else ""),
                (cluster_id,))
            return cursor.rowcount > 0

    def delete_cluster_if_empty(self, cluster_id: int) -> bool:
        """Delete the cluster only if it has no members (checked atomically)."""
        with self._connection() as conn:
            return conn.execute(
                "DELETE FROM face_clusters WHERE id = ? AND NOT EXISTS "
                "(SELECT 1 FROM face_cluster_members m WHERE m.cluster_id = ?)",
                (cluster_id, cluster_id),
            ).rowcount > 0

    def get_face_cluster(self, cluster_id: int) -> dict | None:
        """Cluster row (incl. pinned) plus "stash_ids" (sorted list)."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,)).fetchone()
            if not row:
                return None
            cluster = dict(row)
            cluster["stash_ids"] = [r[0] for r in conn.execute(
                "SELECT stash_id FROM face_cluster_stash_ids WHERE cluster_id = ? ORDER BY stash_id",
                (cluster_id,)
            ).fetchall()]
            return cluster

    def list_face_clusters(self, status: str | None = None) -> list[dict]:
        """List clusters with face counts, largest first."""
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT c.*, COUNT(m.face_id) AS face_count,
                           (SELECT COUNT(DISTINCT lf.stash_scene_id)
                            FROM face_cluster_members mm
                            JOIN library_faces lf ON lf.id = mm.face_id
                            WHERE mm.cluster_id = c.id) AS scene_count
                    FROM face_clusters c
                    LEFT JOIN face_cluster_members m ON m.cluster_id = c.id
                    WHERE c.status = ?
                    GROUP BY c.id
                    ORDER BY face_count DESC
                    """,
                    (status,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.*, COUNT(m.face_id) AS face_count,
                           (SELECT COUNT(DISTINCT lf.stash_scene_id)
                            FROM face_cluster_members mm
                            JOIN library_faces lf ON lf.id = mm.face_id
                            WHERE mm.cluster_id = c.id) AS scene_count
                    FROM face_clusters c
                    LEFT JOIN face_cluster_members m ON m.cluster_id = c.id
                    GROUP BY c.id
                    ORDER BY face_count DESC
                    """
                ).fetchall()
            return [dict(r) for r in rows]

    def add_faces_to_cluster(self, cluster_id: int, face_ids: list[int]) -> int:
        """Add faces to a cluster in one transaction. Returns rows actually inserted.

        Raises ValueError if the cluster does not exist. An unknown face_id raises
        sqlite3.IntegrityError (FK) and rolls back the whole call.
        """
        if not face_ids:
            return 0
        added = 0
        with self._connection() as conn:
            if conn.execute("SELECT 1 FROM face_clusters WHERE id = ?", (cluster_id,)).fetchone() is None:
                raise ValueError(f"cluster {cluster_id} not found")
            for fid in face_ids:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO face_cluster_members (cluster_id, face_id) VALUES (?, ?)",
                    (cluster_id, fid)
                )
                added += cursor.rowcount
        return added

    def remove_faces_from_cluster(self, cluster_id: int, face_ids: list[int]) -> int:
        with self._connection() as conn:
            placeholders = ",".join("?" * len(face_ids))
            cursor = conn.execute(
                f"DELETE FROM face_cluster_members WHERE cluster_id = ? AND face_id IN ({placeholders})",
                [cluster_id, *face_ids]
            )
            return cursor.rowcount

    def get_unassigned_face_ids(self, limit: int | None = None) -> list[int]:
        """Face IDs with no membership in any cluster (whatever its status)."""
        with self._connection() as conn:
            query = """
                SELECT lf.id FROM library_faces lf
                WHERE NOT EXISTS (
                    SELECT 1 FROM face_cluster_members m WHERE m.face_id = lf.id
                )
                ORDER BY lf.id
            """
            if limit:
                query += f" LIMIT {int(limit)}"
            return [r[0] for r in conn.execute(query).fetchall()]

    def get_face_cluster_membership(self, face_id: int) -> list[dict]:
        """Clusters a face belongs to."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT c.* FROM face_cluster_members m
                JOIN face_clusters c ON c.id = m.cluster_id
                WHERE m.face_id = ?
                """,
                (face_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_cluster_assignments(self) -> dict[int, list[int]]:
        """Map of face_id -> [cluster_id] for incremental clustering."""
        with self._connection() as conn:
            rows = conn.execute("SELECT cluster_id, face_id FROM face_cluster_members").fetchall()
            mapping: dict[int, list[int]] = {}
            for cluster_id, face_id in rows:
                mapping.setdefault(face_id, []).append(cluster_id)
            return mapping

    def record_cluster_merge(self, source_ids: list[int], target_id: int) -> None:
        """Remember that source clusters were merged into target (for rebuilds)."""
        with self._connection() as conn:
            for sid in source_ids:
                if sid == target_id:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO face_cluster_merge_log (source_cluster_id, target_cluster_id) VALUES (?, ?)",
                    (sid, target_id)
                )

    def get_cluster_merge_map(self) -> dict[int, int]:
        """Map of source_cluster_id -> target_cluster_id (follows chains)."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT source_cluster_id, target_cluster_id FROM face_cluster_merge_log"
            ).fetchall()
            mapping = dict(rows)
        # Resolve chains: source -> mid -> target
        for src in mapping:
            seen = set()
            dst = mapping[src]
            while dst in mapping and dst not in seen:
                seen.add(dst)
                dst = mapping[dst]
            mapping[src] = dst
        return mapping

    def iter_library_faces(
        self, unassigned_only: bool = False, batch_size: int = 500
    ) -> Iterator[list[dict]]:
        """Iterate library faces in batches of dicts.

        Keys: id, facenet_emb, arcface_emb, best_match_id, best_match_name, stash_scene_id.
        unassigned_only=True yields only faces with NO membership in any cluster,
        whatever its status (open/matched/assigned/ignored/banned).
        """
        cols = ("lf.id, lf.facenet_emb, lf.arcface_emb, lf.best_match_id, "
                "lf.best_match_name, lf.stash_scene_id")
        query = f"SELECT {cols} FROM library_faces lf"
        if unassigned_only:
            query += (" WHERE NOT EXISTS (SELECT 1 FROM face_cluster_members m"
                      " WHERE m.face_id = lf.id)")
        query += " ORDER BY lf.id"
        with self._connection() as conn:
            cursor = conn.execute(query)
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield [dict(r) for r in rows]

    def get_cluster_centroid(self, cluster_id: int) -> list | None:
        """Mean 1024-d concat embedding of a cluster's faces (None if empty)."""
        import numpy as np
        vecs = []
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT lf.facenet_emb, lf.arcface_emb
                FROM face_cluster_members m
                JOIN library_faces lf ON lf.id = m.face_id
                WHERE m.cluster_id = ?
                """,
                (cluster_id,)
            ).fetchall()
        for fn, af in rows:
            vecs.append(np.concatenate([
                np.frombuffer(fn, dtype=np.float32),
                np.frombuffer(af, dtype=np.float32),
            ]))
        if not vecs:
            return None
        mean = np.mean(np.stack(vecs), axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            return None
        return (mean / norm).astype(np.float32)

    def get_clusters_by_status(self, *statuses: str) -> list[dict]:
        """Clusters with any of the given statuses, with face counts."""
        placeholders = ",".join("?" * len(statuses))
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, COUNT(m.face_id) AS face_count,
                       (SELECT COUNT(DISTINCT lf.stash_scene_id)
                        FROM face_cluster_members mm
                        JOIN library_faces lf ON lf.id = mm.face_id
                        WHERE mm.cluster_id = c.id) AS scene_count
                FROM face_clusters c
                LEFT JOIN face_cluster_members m ON m.cluster_id = c.id
                WHERE c.status IN ({placeholders})
                GROUP BY c.id
                ORDER BY face_count DESC
                """,
                list(statuses)
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------- cluster stash ids (StashDB / stash-box id space) ----------

    def get_cluster_stash_ids(self, cluster_id: int) -> list[str]:
        with self._connection() as conn:
            return [r[0] for r in conn.execute(
                "SELECT stash_id FROM face_cluster_stash_ids WHERE cluster_id = ? ORDER BY stash_id",
                (cluster_id,)
            ).fetchall()]

    def set_cluster_stash_ids(
        self, cluster_id: int, stash_ids: list[str], replace: bool = True
    ) -> int:
        """Store a cluster's stash ids. replace=True drops existing rows first.

        Returns rows inserted. Raises ValueError if the cluster does not exist.
        """
        with self._connection() as conn:
            if conn.execute("SELECT 1 FROM face_clusters WHERE id = ?", (cluster_id,)).fetchone() is None:
                raise ValueError(f"cluster {cluster_id} not found")
            if replace:
                conn.execute("DELETE FROM face_cluster_stash_ids WHERE cluster_id = ?", (cluster_id,))
            inserted = 0
            for sid in stash_ids:
                inserted += conn.execute(
                    "INSERT OR IGNORE INTO face_cluster_stash_ids (cluster_id, stash_id) VALUES (?, ?)",
                    (cluster_id, sid)
                ).rowcount
            return inserted

    def get_all_cluster_stash_ids(self) -> dict[int, set[str]]:
        with self._connection() as conn:
            mapping: dict[int, set[str]] = {}
            for cid, sid in conn.execute(
                "SELECT cluster_id, stash_id FROM face_cluster_stash_ids"
            ).fetchall():
                mapping.setdefault(cid, set()).add(sid)
            return mapping

    def get_stash_id_cluster_map(
        self, statuses: tuple[str, ...] = ("assigned", "ignored", "matched", "open")
    ) -> dict[str, int]:
        """stash_id -> cluster_id among clusters with the given statuses.

        Ties across clusters: assigned > ignored > matched > open (user
        decisions first: faces of an ignored identity keep landing in the
        ignored group instead of re-seeding a matched suggestion); among
        assigned groups one whose performer is linked to the id on StashDB
        (synced) beats one that only carries it as a member anchor; then more
        faces, then lower id.
        """
        if not statuses:
            return {}
        placeholders = ",".join("?" * len(statuses))
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT s.stash_id, c.id,
                       CASE c.status WHEN 'assigned' THEN 0 WHEN 'ignored' THEN 1
                                     WHEN 'matched' THEN 2 WHEN 'open' THEN 3 ELSE 4 END AS prio,
                       CASE WHEN c.status = 'assigned' AND c.stash_ids_synced = 1
                            THEN 0 ELSE 1 END AS anchor_only,
                       (SELECT COUNT(*) FROM face_cluster_members m WHERE m.cluster_id = c.id) AS n
                FROM face_cluster_stash_ids s
                JOIN face_clusters c ON c.id = s.cluster_id
                WHERE c.status IN ({placeholders})
                ORDER BY s.stash_id, prio ASC, anchor_only ASC, n DESC, c.id ASC
                """,
                list(statuses)
            ).fetchall()
        mapping: dict[str, int] = {}
        for sid, cid, _prio, _anchor_only, _n in rows:
            mapping.setdefault(sid, cid)
        return mapping

    def get_duplicate_stash_id_groups(self) -> list[tuple[int, list[int]]]:
        """(keeper, [unpinned matched/open duplicates]) per stash id shared by
        several live/ignored groups. The keeper is the get_stash_id_cluster_map
        winner; only automatic (unpinned matched/open) groups are listed as
        duplicates, curated groups are never folded."""
        winners = self.get_stash_id_cluster_map()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT s.stash_id, c.id FROM face_cluster_stash_ids s
                JOIN face_clusters c ON c.id = s.cluster_id
                WHERE c.status IN ('matched', 'open') AND c.pinned = 0
                ORDER BY c.id
                """
            ).fetchall()
        out: dict[int, list[int]] = {}
        for sid, cid in rows:
            keeper = winners.get(sid)
            if keeper is not None and keeper != cid and cid not in out.get(keeper, []):
                out.setdefault(keeper, []).append(cid)
        # a group listed as duplicate must not also be a keeper
        dups = {c for v in out.values() for c in v}
        return [(k, v) for k, v in out.items() if k not in dups]

    def fold_cluster_into(
        self, source_id: int, target_id: int, exclude_face_ids: Iterable[int] = ()
    ) -> int | None:
        """Fold an automatic group into another (one transaction).

        Moves members (except exclude_face_ids, which return to the pool) and
        stash ids, logs the merge, deletes the source. Unlike
        move_faces_to_cluster it does not pin the target. Returns faces moved,
        or None if either group no longer exists (nothing changed).
        """
        excluded = list(dict.fromkeys(int(f) for f in exclude_face_ids))
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            n = conn.execute(
                "SELECT COUNT(*) FROM face_clusters WHERE id IN (?, ?)", (source_id, target_id)
            ).fetchone()[0]
            if source_id == target_id or n != 2:
                return None
            # a group a user curated since the duplicate scan is not automatic any more
            if conn.execute(
                "SELECT pinned FROM face_clusters WHERE id = ?", (source_id,)
            ).fetchone()[0]:
                return None
            not_in = ""
            params: list = [target_id, source_id]
            if excluded:
                not_in = f" AND face_id NOT IN ({','.join('?' * len(excluded))})"
                params.extend(excluded)
            moved = conn.execute(
                "INSERT OR IGNORE INTO face_cluster_members (cluster_id, face_id) "
                f"SELECT ?, face_id FROM face_cluster_members WHERE cluster_id = ?{not_in}",
                params,
            ).rowcount
            conn.execute(
                "INSERT OR IGNORE INTO face_cluster_stash_ids (cluster_id, stash_id) "
                "SELECT ?, stash_id FROM face_cluster_stash_ids WHERE cluster_id = ? "
                "AND stash_id NOT IN (SELECT stash_id FROM face_cluster_blocked_stash_ids "
                "WHERE cluster_id = ?)",
                (target_id, source_id, target_id),
            )
            conn.execute(
                "INSERT OR REPLACE INTO face_cluster_merge_log (source_cluster_id, target_cluster_id) "
                "VALUES (?, ?)",
                (source_id, target_id),
            )
            conn.execute("DELETE FROM face_clusters WHERE id = ?", (source_id,))
        return moved

    def place_faces(self, cluster_id: int, face_ids: list[int]) -> int | None:
        """Build-time add: insert only faces that still exist and are still
        pooled (no membership anywhere: a face split, banned or assigned since
        the build read the pool is never put into a second group), into a
        cluster that still exists. Returns rows inserted, or None if the
        cluster is gone.

        Builds read the pool and the cluster list up front; a concurrent
        re-identify can delete pooled faces and a user can delete or merge a
        group before the build writes. Those must not abort the build.
        """
        if not face_ids:
            return 0
        added = 0
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM face_clusters WHERE id = ?", (cluster_id,)).fetchone() is None:
                return None
            for fid in face_ids:
                added += conn.execute(
                    "INSERT OR IGNORE INTO face_cluster_members (cluster_id, face_id) "
                    "SELECT ?, lf.id FROM library_faces lf WHERE lf.id = ? AND NOT EXISTS "
                    "(SELECT 1 FROM face_cluster_members m WHERE m.face_id = lf.id)",
                    (cluster_id, fid),
                ).rowcount
        return added

    # ---------- stash-id sync bookkeeping ----------

    def get_assigned_clusters_needing_stash_sync(self, include_synced: bool = False) -> list[dict]:
        """Assigned groups whose performer stash ids were never fetched, or that
        have no stash ids at all (the performer may have been linked since).
        include_synced: every assigned group with a performer (links can
        change in Stash; full builds re-check them all)."""
        cond = "" if include_synced else (
            " AND (c.stash_ids_synced = 0 OR NOT EXISTS ("
            "SELECT 1 FROM face_cluster_stash_ids s WHERE s.cluster_id = c.id))")
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT c.* FROM face_clusters c "
                "WHERE c.status = 'assigned' AND c.performer_id IS NOT NULL AND c.performer_id != ''"
                f"{cond} ORDER BY c.id"
            ).fetchall()]

    def set_cluster_stash_ids_synced(self, cluster_id: int, synced: bool = True) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE face_clusters SET stash_ids_synced = ? WHERE id = ?",
                (1 if synced else 0, cluster_id),
            )

    def block_cluster_stash_ids(self, cluster_id: int, stash_ids: Iterable[str]) -> int:
        """Remember stash ids a user decision took away from this cluster.
        Member-derived anchors never re-add them. Returns rows inserted."""
        ids = [s for s in stash_ids if s]
        if not ids:
            return 0
        with self._connection() as conn:
            return sum(conn.execute(
                "INSERT OR IGNORE INTO face_cluster_blocked_stash_ids (cluster_id, stash_id) "
                "SELECT ?, ? WHERE EXISTS (SELECT 1 FROM face_clusters WHERE id = ?)",
                (cluster_id, sid, cluster_id),
            ).rowcount for sid in ids)

    def unblock_cluster_stash_ids(self, cluster_id: int, stash_ids: Iterable[str]) -> int:
        ids = [s for s in stash_ids if s]
        if not ids:
            return 0
        with self._connection() as conn:
            return conn.execute(
                "DELETE FROM face_cluster_blocked_stash_ids WHERE cluster_id = ? "
                f"AND stash_id IN ({','.join('?' * len(ids))})",
                (cluster_id, *ids),
            ).rowcount

    def get_blocked_stash_ids(self, cluster_id: int) -> set[str]:
        with self._connection() as conn:
            return {r[0] for r in conn.execute(
                "SELECT stash_id FROM face_cluster_blocked_stash_ids WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchall()}

    def split_faces_to_new_cluster(
        self, cluster_id: int, face_ids: list[int], new_status: str = "open",
    ) -> int:
        """One transaction: create a pinned cluster, move face_ids (members of
        cluster_id) into it and pin the source. Raises ValueError if the source
        is gone or none of the faces are its members. Returns the new id."""
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM face_clusters WHERE id = ?", (cluster_id,)).fetchone() is None:
                raise ValueError(f"cluster {cluster_id} not found")
            ids = list(dict.fromkeys(int(f) for f in face_ids))
            members = [r[0] for r in conn.execute(
                f"SELECT face_id FROM face_cluster_members WHERE cluster_id = ? "
                f"AND face_id IN ({','.join('?' * len(ids))})", (cluster_id, *ids),
            ).fetchall()] if ids else []
            if not members:
                raise ValueError(f"none of the faces are in cluster {cluster_id}")
            new_id = conn.execute(
                "INSERT INTO face_clusters (status, pinned) VALUES (?, 1)", (new_status,)
            ).lastrowid
            ph = ",".join("?" * len(members))
            conn.execute(
                f"UPDATE face_cluster_members SET cluster_id = ? WHERE cluster_id = ? AND face_id IN ({ph})",
                (new_id, cluster_id, *members),
            )
            conn.execute(
                "UPDATE face_clusters SET pinned = 1, updated_at = datetime('now') WHERE id = ?",
                (cluster_id,),
            )
            return new_id

    def ban_face(self, face_id: int) -> bool:
        """One transaction: drop every non-banned membership of the face and,
        unless it is already banned, put it in its own pinned 'banned' group.
        Returns True if a new ban was created (False: already banned or the
        face does not exist)."""
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM library_faces WHERE id = ?", (face_id,)).fetchone() is None:
                return False
            conn.execute(
                "DELETE FROM face_cluster_members WHERE face_id = ? AND cluster_id IN "
                "(SELECT id FROM face_clusters WHERE status != 'banned')", (face_id,))
            if conn.execute(
                "SELECT 1 FROM face_cluster_members m JOIN face_clusters c ON c.id = m.cluster_id "
                "WHERE m.face_id = ? AND c.status = 'banned'", (face_id,),
            ).fetchone():
                return False
            bid = conn.execute(
                "INSERT INTO face_clusters (status, name, pinned) VALUES ('banned', 'Banned faces', 1)"
            ).lastrowid
            conn.execute("INSERT INTO face_cluster_members (cluster_id, face_id) VALUES (?, ?)",
                         (bid, face_id))
            return True

    def get_member_anchor(
        self, cluster_id: int, min_count: int = 2, exclude_face_ids: Iterable[int] = (),
    ) -> str | None:
        """The best_match_id a strict majority of the group's anchored members
        (other than exclude_face_ids) share, if at least min_count of them do;
        else None."""
        excluded = [int(f) for f in exclude_face_ids]
        not_in = f" AND m.face_id NOT IN ({','.join('?' * len(excluded))})" if excluded else ""
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT lf.best_match_id, COUNT(*) AS n
                FROM face_cluster_members m JOIN library_faces lf ON lf.id = m.face_id
                WHERE m.cluster_id = ? AND lf.best_match_id IS NOT NULL{not_in}
                GROUP BY lf.best_match_id ORDER BY n DESC, lf.best_match_id
                """,
                (cluster_id, *excluded),
            ).fetchall()
        if not rows:
            return None
        total = sum(r[1] for r in rows)
        sid, n = rows[0]
        if n >= min_count and n * 2 > total:
            return sid
        return None

    # ---------- merge / resolve ----------

    def move_faces_to_cluster(self, source_ids: list[int], target_id: int) -> dict:
        """Atomically merge source clusters into target (one transaction).

        Moves members and stash ids, logs the merge, pins the target and deletes
        the sources. Raises ValueError (nothing modified) when there are no
        sources left after dropping target_id, or target/sources are missing.
        Returns {"faces_moved": int, "sources_deleted": [ids]}.
        """
        sources: list[int] = []
        for sid in source_ids:
            if sid != target_id and sid not in sources:
                sources.append(sid)
        if not sources:
            raise ValueError("no source clusters")
        placeholders = ",".join("?" * len(sources))
        with self._connection() as conn:
            # write lock first: existence checks and the move are one atomic step
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM face_clusters WHERE id = ?", (target_id,)).fetchone() is None:
                raise ValueError(f"target cluster {target_id} not found")
            found = {r[0] for r in conn.execute(
                f"SELECT id FROM face_clusters WHERE id IN ({placeholders})", sources
            ).fetchall()}
            missing = [sid for sid in sources if sid not in found]
            if missing:
                raise ValueError(f"source clusters not found: {missing}")
            moved = conn.execute(
                f"""
                INSERT OR IGNORE INTO face_cluster_members (cluster_id, face_id)
                SELECT ?, face_id FROM face_cluster_members WHERE cluster_id IN ({placeholders})
                """,
                [target_id, *sources]
            ).rowcount
            conn.execute(
                f"""
                INSERT OR IGNORE INTO face_cluster_stash_ids (cluster_id, stash_id)
                SELECT ?, stash_id FROM face_cluster_stash_ids WHERE cluster_id IN ({placeholders})
                """,
                [target_id, *sources]
            )
            conn.executemany(
                "INSERT OR REPLACE INTO face_cluster_merge_log (source_cluster_id, target_cluster_id) VALUES (?, ?)",
                [(sid, target_id) for sid in sources]
            )
            # a user merge is explicit consent to the sources' identities
            conn.execute(
                "DELETE FROM face_cluster_blocked_stash_ids WHERE cluster_id = ? AND stash_id IN "
                "(SELECT stash_id FROM face_cluster_stash_ids WHERE cluster_id = ?)",
                (target_id, target_id),
            )
            conn.execute(
                "UPDATE face_clusters SET pinned = 1, updated_at = datetime('now') WHERE id = ?",
                (target_id,)
            )
            conn.execute(f"DELETE FROM face_clusters WHERE id IN ({placeholders})", sources)
        return {"faces_moved": moved, "sources_deleted": sources}

    def resolve_cluster_id(self, cluster_id: int) -> int | None:
        """Live cluster id for cluster_id: itself if it exists, else its merge target
        (following chains) if that exists, else None."""
        with self._connection() as conn:
            if conn.execute("SELECT 1 FROM face_clusters WHERE id = ?", (cluster_id,)).fetchone():
                return cluster_id
        final = self.get_cluster_merge_map().get(cluster_id)
        if final is None or final == cluster_id:
            return None
        with self._connection() as conn:
            if conn.execute("SELECT 1 FROM face_clusters WHERE id = ?", (final,)).fetchone():
                return final
        return None

    # ---------- bulk helpers ----------

    def clear_cluster_members(self, cluster_id: int, only_unpinned: bool = False) -> int | None:
        """Remove all members. only_unpinned: do nothing (return None) if the
        cluster is pinned or gone at write time."""
        with self._connection() as conn:
            if only_unpinned:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute(
                    "SELECT 1 FROM face_clusters WHERE id = ? AND pinned = 0", (cluster_id,)
                ).fetchone() is None:
                    return None
            return conn.execute(
                "DELETE FROM face_cluster_members WHERE cluster_id = ?", (cluster_id,)
            ).rowcount

    def delete_empty_clusters(
        self, statuses: tuple[str, ...] = ("open", "matched"), include_pinned: bool = False,
    ) -> int:
        """Delete clusters in these statuses with zero members. Pinned clusters
        are kept unless include_pinned (a pinned empty group may be one a user
        operation is filling right now, e.g. a split or a merge target)."""
        if not statuses:
            return 0
        placeholders = ",".join("?" * len(statuses))
        with self._connection() as conn:
            return conn.execute(
                f"""
                DELETE FROM face_clusters
                WHERE status IN ({placeholders})
                  {"" if include_pinned else "AND pinned = 0"}
                  AND NOT EXISTS (SELECT 1 FROM face_cluster_members m WHERE m.cluster_id = face_clusters.id)
                """,
                list(statuses)
            ).rowcount

    def get_scene_ids_for_faces(self, face_ids: list[int]) -> list[int]:
        """Distinct, sorted scene ids of the given faces."""
        if not face_ids:
            return []
        scenes: set[int] = set()
        ids = list(face_ids)
        with self._connection() as conn:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                placeholders = ",".join("?" * len(chunk))
                scenes.update(r[0] for r in conn.execute(
                    f"SELECT DISTINCT stash_scene_id FROM library_faces WHERE id IN ({placeholders})",
                    chunk
                ).fetchall())
        return sorted(scenes)

    # ---------- rejection memory ----------

    def add_face_rejections(
        self,
        face_ids: list[int],
        cluster_id: int | None = None,
        performer_id: str | None = None,
        stash_ids: Iterable[str] = (),
    ) -> int:
        """Remember that faces must not rejoin a cluster / performer / stash id.

        Returns rows inserted (existing rejections count 0).
        """
        refs: list[tuple[str, str]] = []
        if cluster_id is not None:
            refs.append(("cluster", str(cluster_id)))
        if performer_id is not None:
            refs.append(("performer", str(performer_id)))
        refs.extend(("stash_id", s) for s in stash_ids if s is not None)
        if not face_ids or not refs:
            return 0
        inserted = 0
        with self._connection() as conn:
            for fid in face_ids:
                for kind, ref in refs:
                    inserted += conn.execute(
                        "INSERT OR IGNORE INTO face_rejections (face_id, kind, ref) VALUES (?, ?, ?)",
                        (fid, kind, ref)
                    ).rowcount
        return inserted

    def get_face_rejections(self, face_ids: list[int] | None = None) -> dict[int, dict]:
        """{face_id: {"clusters": set[int], "performers": set[str], "stash_ids": set[str]}}.

        Only faces with at least one rejection appear. face_ids=None means all faces.
        """
        rows: list = []
        with self._connection() as conn:
            if face_ids is None:
                rows = conn.execute("SELECT face_id, kind, ref FROM face_rejections").fetchall()
            else:
                ids = list(face_ids)
                for i in range(0, len(ids), 500):
                    chunk = ids[i:i + 500]
                    placeholders = ",".join("?" * len(chunk))
                    rows.extend(conn.execute(
                        f"SELECT face_id, kind, ref FROM face_rejections WHERE face_id IN ({placeholders})",
                        chunk
                    ).fetchall())
        result: dict[int, dict] = {}
        for fid, kind, ref in rows:
            entry = result.setdefault(fid, {"clusters": set(), "performers": set(), "stash_ids": set()})
            if kind == "cluster":
                try:
                    entry["clusters"].add(int(ref))
                except ValueError:
                    pass
            elif kind == "performer":
                entry["performers"].add(ref)
            elif kind == "stash_id":
                entry["stash_ids"].add(ref)
        return result

    def clear_face_rejections(self, face_ids: list[int]) -> int:
        if not face_ids:
            return 0
        removed = 0
        ids = list(face_ids)
        with self._connection() as conn:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                placeholders = ",".join("?" * len(chunk))
                removed += conn.execute(
                    f"DELETE FROM face_rejections WHERE face_id IN ({placeholders})", chunk
                ).rowcount
        return removed

    # ---------- re-identify upsert ----------

    def upsert_scene_library_faces(
        self,
        stash_scene_id: int,
        faces: list[dict],
        iou_threshold: float = 0.5,
        time_tolerance_sec: float = 0.5,
        emb_threshold: float = 0.3,
        pair_emb_threshold: float = 0.5,
    ) -> dict:
        """Replace a scene's faces with a fresh detection set, keeping face ids.

        Each face dict takes the library_faces columns (frame_index, timestamp_sec,
        bbox, det_confidence, yaw, facenet_emb, arcface_emb, crop_path,
        best_match_id, best_match_name, best_match_confidence, db_version).

        New faces are matched one-to-one to existing rows (same frame + IoU >=
        iou_threshold, greedy by IoU; then, for faces left over, same frame +
        embedding cosine distance <= emb_threshold, which pairs a re-encoded
        file whose boxes moved with the resolution). A box pair also needs the
        embeddings to agree (cosine distance <= pair_emb_threshold): a different
        person at the same spot is a new face, not the old row, so it never
        inherits that row's memberships, ban or rejections; such a pair is
        treated as unmatched on both sides. An identical key (frame, moment
        and box) is the same detection and pairs regardless. Matched rows are updated in place so their
        cluster memberships and rejections survive. Unmatched old rows are deleted
        unless curated (member of an assigned/ignored/banned or pinned cluster, or
        with any rejection), which are retained untouched. Unmatched new faces are
        inserted. All in one transaction.

        Returns {"face_ids": [id | None, ...] parallel to faces, "updated",
        "inserted", "deleted", "retained", "conflicts", "stale_crop_paths"}.
        stale_crop_paths lists crop files no row references any more (deleted
        rows, replaced paths of updated rows, and the new paths of faces that
        could not be stored).
        """
        face_ids: list[int | None] = [None] * len(faces)
        updated = inserted = deleted = retained = conflicts = 0
        candidate_stale: list[str] = []

        with self._connection() as conn:
            # Take the write lock before reading: a ban / assign / eject that
            # commits between computing `curated` and the DELETE would
            # otherwise be cascaded away with a row judged uncurated.
            conn.execute("BEGIN IMMEDIATE")
            old_rows = [dict(r) for r in conn.execute(
                """
                SELECT lf.id, lf.frame_index, lf.timestamp_sec,
                       lf.bbox_x, lf.bbox_y, lf.bbox_w, lf.bbox_h, lf.crop_path,
                       lf.facenet_emb, lf.arcface_emb,
                       (EXISTS (
                            SELECT 1 FROM face_cluster_members m
                            JOIN face_clusters c ON c.id = m.cluster_id
                            WHERE m.face_id = lf.id
                              AND (c.status IN ('assigned', 'ignored', 'banned') OR c.pinned = 1))
                        OR EXISTS (SELECT 1 FROM face_rejections r WHERE r.face_id = lf.id)
                       ) AS curated
                FROM library_faces lf
                WHERE lf.stash_scene_id = ?
                ORDER BY lf.id
                """,
                (stash_scene_id,)
            ).fetchall()]

            def same_frame(new: dict, old: dict) -> bool:
                nt, ot = new.get("timestamp_sec"), old["timestamp_sec"]
                if nt is not None and ot is not None:
                    return abs(float(nt) - float(ot)) <= time_tolerance_sec
                return new.get("frame_index") == old["frame_index"]

            new_vecs = [_unit_face_vec(f.get("facenet_emb"), f.get("arcface_emb")) for f in faces]
            old_vecs = [_unit_face_vec(o["facenet_emb"], o["arcface_emb"]) for o in old_rows]

            def emb_dist(i: int, j: int) -> float | None:
                nv, ov = new_vecs[i], old_vecs[j]
                if nv is None or ov is None or nv.shape != ov.shape:
                    return None
                return 1.0 - float(nv @ ov)

            def same_key(new: dict, old: dict) -> bool:
                # Identical UNIQUE key (frame, moment, box): the same pixels,
                # so the same detection whatever the embeddings say; it could
                # not be stored next to the old row anyway.
                nb = new.get("bbox") or {}
                nt, ot = new.get("timestamp_sec"), old["timestamp_sec"]
                return (new.get("frame_index") == old["frame_index"]
                        and (nt is None) == (ot is None)
                        and (nt is None or float(nt) == float(ot))
                        and all(float(nb.get(k, 0) or 0) == float(old[f"bbox_{k}"])
                                for k in ("x", "y", "w", "h")))

            pairs: list[tuple[float, int, int]] = []
            for i, new in enumerate(faces):
                nb = new.get("bbox") or {}
                for j, old in enumerate(old_rows):
                    if not same_frame(new, old):
                        continue
                    iou = _bbox_iou(nb, {"x": old["bbox_x"], "y": old["bbox_y"],
                                         "w": old["bbox_w"], "h": old["bbox_h"]})
                    if iou >= iou_threshold:
                        d = emb_dist(i, j)
                        if d is not None and d > pair_emb_threshold and not same_key(new, old):
                            continue   # same spot, different person
                        nt, ot = new.get("timestamp_sec"), old["timestamp_sec"]
                        dt = abs(float(nt) - float(ot)) if nt is not None and ot is not None else 0.0
                        pairs.append((iou, dt, i, j))
            # Ties on IoU go to the closest moment: a new face whose key equals
            # an old row's key always pairs with that row, so a paired update
            # never needs a key another row still holds.
            pairs.sort(key=lambda p: (-p[0], p[1], p[2], p[3]))
            new_to_old: dict[int, int] = {}
            used_old: set[int] = set()
            for _iou, _dt, i, j in pairs:
                if i in new_to_old or j in used_old:
                    continue
                new_to_old[i] = j
                used_old.add(j)

            # Fallback for faces the boxes cannot pair (file replaced at another
            # resolution: same moments, scaled boxes): same frame and nearly
            # identical embedding is the same detection.
            emb_pairs: list[tuple[float, int, int]] = []
            for i, new in enumerate(faces):
                if i in new_to_old:
                    continue
                for j, old in enumerate(old_rows):
                    if j in used_old or not same_frame(new, old):
                        continue
                    dist = emb_dist(i, j)
                    if dist is not None and dist <= emb_threshold:
                        emb_pairs.append((dist, i, j))
            for _d, i, j in sorted(emb_pairs):
                if i in new_to_old or j in used_old:
                    continue
                new_to_old[i] = j
                used_old.add(j)

            # Delete first so freed UNIQUE keys can be reused by updates/inserts.
            for j, old in enumerate(old_rows):
                if j in used_old:
                    continue
                if old["curated"]:
                    retained += 1
                    continue
                conn.execute("DELETE FROM library_faces WHERE id = ?", (old["id"],))
                deleted += 1
                if old["crop_path"]:
                    candidate_stale.append(old["crop_path"])

            def update_paired(i: int, j: int) -> None:
                new, old = faces[i], old_rows[j]
                bbox = new.get("bbox") or {}
                conn.execute(
                    """
                    UPDATE library_faces SET
                        frame_index = ?, timestamp_sec = ?,
                        bbox_x = ?, bbox_y = ?, bbox_w = ?, bbox_h = ?,
                        det_confidence = ?, yaw = ?,
                        facenet_emb = ?, arcface_emb = ?, crop_path = ?,
                        best_match_id = ?, best_match_name = ?, best_match_confidence = ?,
                        db_version = ?
                    WHERE id = ?
                    """,
                    (
                        new.get("frame_index"), new.get("timestamp_sec"),
                        bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0),
                        new.get("det_confidence"), new.get("yaw"),
                        new.get("facenet_emb"), new.get("arcface_emb"), new.get("crop_path"),
                        new.get("best_match_id"), new.get("best_match_name"),
                        new.get("best_match_confidence"), new.get("db_version"),
                        old["id"],
                    )
                )

            # The UNIQUE key is checked per statement, so a paired update can
            # collide with another paired row that has not moved yet. Retry the
            # collisions after the others went through, until no progress.
            pending = sorted(new_to_old.items())
            while pending:
                blocked: list[tuple[int, int]] = []
                for i, j in pending:
                    try:
                        update_paired(i, j)
                    except sqlite3.IntegrityError:
                        blocked.append((i, j))
                        continue
                    new, old = faces[i], old_rows[j]
                    updated += 1
                    face_ids[i] = old["id"]
                    if old["crop_path"] and old["crop_path"] != new.get("crop_path"):
                        candidate_stale.append(old["crop_path"])
                if len(blocked) == len(pending):
                    break
                pending = blocked
            else:
                blocked = []

            for i, j in blocked:
                # UNIQUE key collides with a retained row: keep this row's
                # detection data, but refresh its identify-time anchor so a
                # stale best match does not linger.
                new, old = faces[i], old_rows[j]
                conn.execute(
                    """
                    UPDATE library_faces SET best_match_id = ?, best_match_name = ?,
                        best_match_confidence = ?, db_version = ?
                    WHERE id = ?
                    """,
                    (new.get("best_match_id"), new.get("best_match_name"),
                     new.get("best_match_confidence"), new.get("db_version"), old["id"]),
                )
                conflicts += 1
                face_ids[i] = old["id"]
                if new.get("crop_path") and new.get("crop_path") != old["crop_path"]:
                    candidate_stale.append(new["crop_path"])

            for i, new in enumerate(faces):
                if i in new_to_old:
                    continue
                bbox = new.get("bbox") or {}
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO library_faces (
                        stash_scene_id, frame_index, timestamp_sec,
                        bbox_x, bbox_y, bbox_w, bbox_h, det_confidence, yaw,
                        facenet_emb, arcface_emb, crop_path,
                        best_match_id, best_match_name, best_match_confidence, db_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stash_scene_id, new.get("frame_index"), new.get("timestamp_sec"),
                        bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0),
                        new.get("det_confidence"), new.get("yaw"),
                        new.get("facenet_emb"), new.get("arcface_emb"), new.get("crop_path"),
                        new.get("best_match_id"), new.get("best_match_name"),
                        new.get("best_match_confidence"), new.get("db_version"),
                    )
                )
                if cursor.rowcount:
                    face_ids[i] = cursor.lastrowid
                    inserted += 1
                elif new.get("crop_path"):
                    candidate_stale.append(new["crop_path"])

            stale: list[str] = []
            for path in dict.fromkeys(candidate_stale):
                if conn.execute(
                    "SELECT 1 FROM library_faces WHERE crop_path = ? LIMIT 1", (path,)
                ).fetchone() is None:
                    stale.append(path)

        return {
            "face_ids": face_ids,
            "updated": updated,
            "inserted": inserted,
            "deleted": deleted,
            "retained": retained,
            "conflicts": conflicts,
            "stale_crop_paths": stale,
        }

    # ==================== Image Fingerprints ====================

    def create_image_fingerprint(
        self,
        stash_image_id: str,
        gallery_id: str | None = None,
        faces_detected: int = 0,
        db_version: str | None = None,
    ) -> int:
        """
        Create or update an image fingerprint. Returns the fingerprint ID.
        Uses upsert - if fingerprint exists for image, updates it.

        Args:
            stash_image_id: The Stash image ID
            gallery_id: The gallery this image belongs to (optional)
            faces_detected: Number of faces detected in the image
            db_version: Face recognition DB version used for this fingerprint
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO image_fingerprints (stash_image_id, gallery_id, faces_detected, db_version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(stash_image_id) DO UPDATE SET
                    gallery_id = COALESCE(excluded.gallery_id, gallery_id),
                    faces_detected = excluded.faces_detected,
                    db_version = excluded.db_version,
                    updated_at = datetime('now')
                RETURNING id
                """,
                (stash_image_id, gallery_id, faces_detected, db_version)
            )
            return cursor.fetchone()[0]

    def get_image_fingerprint(self, stash_image_id: str) -> dict | None:
        """Get an image fingerprint by stash image ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM image_fingerprints WHERE stash_image_id = ?",
                (stash_image_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_gallery_image_fingerprints(self, gallery_id: str) -> list[dict]:
        """Get all image fingerprints for a gallery."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM image_fingerprints WHERE gallery_id = ?",
                (gallery_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def add_image_fingerprint_face(
        self,
        stash_image_id: str,
        performer_id: str,
        confidence: float | None = None,
        distance: float | None = None,
        bbox_x: float | None = None,
        bbox_y: float | None = None,
        bbox_w: float | None = None,
        bbox_h: float | None = None,
    ) -> int:
        """Add or update a face entry in an image fingerprint. Returns the face entry ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO image_fingerprint_faces (
                    stash_image_id, performer_id, confidence, distance,
                    bbox_x, bbox_y, bbox_w, bbox_h
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stash_image_id, performer_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    distance = excluded.distance,
                    bbox_x = excluded.bbox_x,
                    bbox_y = excluded.bbox_y,
                    bbox_w = excluded.bbox_w,
                    bbox_h = excluded.bbox_h
                RETURNING id
                """,
                (stash_image_id, performer_id, confidence, distance,
                 bbox_x, bbox_y, bbox_w, bbox_h)
            )
            return cursor.fetchone()[0]

    def get_image_fingerprint_faces(self, stash_image_id: str) -> list[dict]:
        """Get all face entries for an image fingerprint."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM image_fingerprint_faces WHERE stash_image_id = ?",
                (stash_image_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_image_fingerprint_faces(self, stash_image_id: str) -> int:
        """Delete all face entries for an image fingerprint. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM image_fingerprint_faces WHERE stash_image_id = ?",
                (stash_image_id,)
            )
            return cursor.rowcount

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connection() as conn:
            stats = {}
            stats['total_recommendations'] = conn.execute(
                "SELECT COUNT(*) FROM recommendations"
            ).fetchone()[0]
            stats['pending_recommendations'] = conn.execute(
                "SELECT COUNT(*) FROM recommendations WHERE status = 'pending'"
            ).fetchone()[0]
            stats['dismissed_count'] = conn.execute(
                "SELECT COUNT(*) FROM dismissed_targets"
            ).fetchone()[0]
            stats['analysis_runs_today'] = conn.execute(
                "SELECT COUNT(*) FROM analysis_runs WHERE date(started_at) = date('now')"
            ).fetchone()[0]
            return stats

    # ==================== Duplicate Candidates ====================

    def insert_candidate(
        self,
        scene_a_id: int,
        scene_b_id: int,
        source: str,
        run_id: int,
    ) -> int | None:
        """Insert a candidate pair. Enforces canonical order (a < b). Returns ID or None if duplicate."""
        a, b = (min(scene_a_id, scene_b_id), max(scene_a_id, scene_b_id))
        with self._connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO duplicate_candidates (scene_a_id, scene_b_id, source, run_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (a, b, source, run_id),
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def insert_candidates_batch(
        self,
        candidates: list[tuple[int, int, str]],
        run_id: int,
    ) -> int:
        """Batch insert candidate pairs. Each tuple is (scene_a_id, scene_b_id, source).
        Enforces canonical order. Returns count inserted."""
        if not candidates:
            return 0
        rows = [(min(a, b), max(a, b), source, run_id) for a, b, source in candidates]
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO duplicate_candidates (scene_a_id, scene_b_id, source, run_id)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            return conn.execute(
                "SELECT COUNT(*) FROM duplicate_candidates WHERE run_id = ?", (run_id,)
            ).fetchone()[0]

    def get_candidates_batch(
        self,
        run_id: int,
        after_id: int = 0,
        limit: int = 100,
    ) -> list[dict]:
        """Get a batch of candidates using cursor-based pagination."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM duplicate_candidates
                WHERE run_id = ? AND id > ?
                ORDER BY id
                LIMIT ?
                """,
                (run_id, after_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def count_candidates(self, run_id: int) -> int:
        """Count candidates for a run."""
        with self._connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM duplicate_candidates WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]

    def clear_candidates(self, run_id: int) -> int:
        """Delete all candidates for a run. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM duplicate_candidates WHERE run_id = ?",
                (run_id,),
            )
            return cursor.rowcount

    def clear_all_candidates(self) -> int:
        """Delete ALL candidates from all runs. The candidates table is an
        ephemeral work queue — old rows block new inserts due to the
        UNIQUE(scene_a_id, scene_b_id) constraint which doesn't include run_id."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM duplicate_candidates")
            return cursor.rowcount

    def clear_orphaned_candidates(self) -> int:
        """Delete candidates with NULL run_id (from broken runs that passed run_id=None).
        Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM duplicate_candidates WHERE run_id IS NULL"
            )
            return cursor.rowcount

    def get_candidate_scene_ids(self, run_id: int) -> set[int]:
        """Get all distinct scene IDs that appear in candidates for a run."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT scene_a_id AS sid FROM duplicate_candidates WHERE run_id = ?
                UNION
                SELECT scene_b_id AS sid FROM duplicate_candidates WHERE run_id = ?
                """,
                (run_id, run_id),
            ).fetchall()
            return {row[0] for row in rows}

    def get_fingerprints_with_faces(
        self,
        scene_ids: set[int] | None = None,
    ) -> dict:
        """
        Load all complete fingerprints with their faces in a single JOIN query.
        Returns dict keyed by str(stash_scene_id) with structure:
        {stash_scene_id, total_faces, frames_analyzed, faces: {performer_id -> {face_count, avg_confidence, proportion}}}
        Optionally filtered to specific scene IDs.
        """
        with self._connection() as conn:
            if scene_ids:
                placeholders = ",".join("?" for _ in scene_ids)
                rows = conn.execute(
                    f"""
                    SELECT sf.stash_scene_id, sf.total_faces, sf.frames_analyzed,
                           sff.performer_id, sff.face_count, sff.avg_confidence, sff.proportion
                    FROM scene_fingerprints sf
                    LEFT JOIN scene_fingerprint_faces sff ON sf.id = sff.fingerprint_id
                    WHERE sf.fingerprint_status = 'complete'
                      AND sf.stash_scene_id IN ({placeholders})
                    ORDER BY sf.stash_scene_id
                    """,
                    list(scene_ids),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT sf.stash_scene_id, sf.total_faces, sf.frames_analyzed,
                           sff.performer_id, sff.face_count, sff.avg_confidence, sff.proportion
                    FROM scene_fingerprints sf
                    LEFT JOIN scene_fingerprint_faces sff ON sf.id = sff.fingerprint_id
                    WHERE sf.fingerprint_status = 'complete'
                    ORDER BY sf.stash_scene_id
                    """,
                ).fetchall()

        # Group by scene
        result = {}
        for row in rows:
            scene_id = str(row["stash_scene_id"])
            if scene_id not in result:
                result[scene_id] = {
                    "stash_scene_id": row["stash_scene_id"],
                    "total_faces": row["total_faces"],
                    "frames_analyzed": row["frames_analyzed"],
                    "faces": {},
                }
            if row["performer_id"] is not None:
                result[scene_id]["faces"][row["performer_id"]] = {
                    "performer_id": row["performer_id"],
                    "face_count": row["face_count"],
                    "avg_confidence": row["avg_confidence"],
                    "proportion": row["proportion"],
                }

        return result

    def generate_face_candidates(self) -> list[tuple[int, int]]:
        """
        Find all scene pairs that share an identified performer via SQL self-join
        on scene_fingerprint_faces. Returns list of (scene_a_id, scene_b_id) tuples
        in canonical order (a < b).
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT sfa.stash_scene_id AS scene_a_id,
                                sfb.stash_scene_id AS scene_b_id
                FROM scene_fingerprint_faces fa
                JOIN scene_fingerprint_faces fb ON fa.performer_id = fb.performer_id
                JOIN scene_fingerprints sfa ON fa.fingerprint_id = sfa.id
                JOIN scene_fingerprints sfb ON fb.fingerprint_id = sfb.id
                WHERE sfa.fingerprint_status = 'complete'
                  AND sfb.fingerprint_status = 'complete'
                  AND sfa.stash_scene_id < sfb.stash_scene_id
                  AND fa.performer_id != 'unknown'
                """
            ).fetchall()
            return [(row[0], row[1]) for row in rows]

    def store_scene_phashes(self, phashes: list[tuple[int, str]]) -> int:
        """
        Store scene phashes in memory for duplicate candidate generation.
        Input: list of (stash_scene_id, phash_hex) tuples.
        Stores parsed data in self._phash_data for generate_phash_candidates().
        Returns count stored.
        """
        self._phash_data: list[tuple[int, int]] = []
        for scene_id, phash_hex in phashes:
            try:
                self._phash_data.append((scene_id, int(phash_hex, 16)))
            except (ValueError, TypeError):
                continue
        return len(self._phash_data)

    def generate_phash_candidates(self, max_distance: int = 10) -> list[tuple[int, int, int]]:
        """
        Find all scene pairs with phash Hamming distance <= max_distance.
        Uses data stored by store_scene_phashes().
        Returns list of (scene_a_id, scene_b_id, hamming_distance) in canonical order.
        """
        data = getattr(self, "_phash_data", None)
        if not data:
            return []

        candidates = []
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                xor = data[i][1] ^ data[j][1]
                dist = bin(xor).count("1")
                if dist <= max_distance:
                    a, b = data[i][0], data[j][0]
                    if a > b:
                        a, b = b, a
                    candidates.append((a, b, dist))

        return candidates

    # ========================================================================
    # Job Queue CRUD
    # ========================================================================

    def submit_job(
        self, type: str, priority: int, triggered_by: str,
        cursor: str | None = None, items_total: int | None = None,
    ) -> int | None:
        """Submit a job to the queue. Returns job ID, or None if duplicate queued."""
        with self._connection() as conn:
            existing = conn.execute(
                "SELECT id FROM job_queue WHERE type = ? AND status IN ('queued', 'running', 'stopping')",
                (type,)
            ).fetchone()
            if existing:
                return None
            cursor_obj = conn.execute(
                """
                INSERT INTO job_queue (type, status, priority, cursor, items_total, triggered_by)
                VALUES (?, 'queued', ?, ?, ?, ?)
                """,
                (type, priority, cursor, items_total, triggered_by)
            )
            return cursor_obj.lastrowid

    def get_job(self, job_id: int) -> dict | None:
        """Get a single job by ID."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM job_queue WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_jobs(self, status: str | None = None, type: str | None = None, limit: int = 50) -> list[dict]:
        """Get jobs with optional filters."""
        query = "SELECT * FROM job_queue WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def get_queued_jobs(self) -> list[dict]:
        """Get all queued jobs ordered by priority (lowest number = highest priority)."""
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM job_queue WHERE status = 'queued' ORDER BY priority ASC, created_at ASC"
            ).fetchall()]

    def start_job(self, job_id: int):
        """Mark a job as running."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE job_queue SET status = 'running', started_at = datetime('now') WHERE id = ?",
                (job_id,)
            )

    def complete_job(self, job_id: int):
        """Mark a job as completed."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE job_queue SET status = 'completed', completed_at = datetime('now') WHERE id = ?",
                (job_id,)
            )

    def fail_job(self, job_id: int, error_message: str):
        """Mark a job as failed."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE job_queue SET status = 'failed', completed_at = datetime('now'), error_message = ? WHERE id = ?",
                (error_message, job_id)
            )

    def cancel_job(self, job_id: int):
        """Cancel a queued job."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE job_queue SET status = 'cancelled', completed_at = datetime('now') WHERE id = ?",
                (job_id,)
            )

    def set_job_status(self, job_id: int, status: str):
        """Set job status directly."""
        with self._connection() as conn:
            conn.execute("UPDATE job_queue SET status = ? WHERE id = ?", (status, job_id))

    def update_job_progress(self, job_id: int, items_processed: int | None = None,
                            items_total: int | None = None, cursor: str | None = None):
        """Update job progress fields. Only updates non-None fields."""
        updates = []
        params = []
        if items_processed is not None:
            updates.append("items_processed = ?")
            params.append(items_processed)
        if items_total is not None:
            updates.append("items_total = ?")
            params.append(items_total)
        if cursor is not None:
            updates.append("cursor = ?")
            params.append(cursor)
        if not updates:
            return
        params.append(job_id)
        with self._connection() as conn:
            conn.execute(f"UPDATE job_queue SET {', '.join(updates)} WHERE id = ?", params)

    def requeue_interrupted_jobs(self) -> int:
        """Re-queue jobs left as running/stopping after a crash. Returns count.

        Clears stale progress fields so re-queued jobs don't appear already-finished.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """UPDATE job_queue
                SET status = 'queued', started_at = NULL, completed_at = NULL,
                    items_processed = 0, items_total = NULL
                WHERE status IN ('running', 'stopping')"""
            )
            return cursor.rowcount

    def delete_terminal_jobs(self) -> int:
        """Delete all completed/failed/cancelled jobs. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM job_queue WHERE status IN ('completed', 'failed', 'cancelled')"
            )
            return cursor.rowcount

    # ========================================================================
    # Job Schedules CRUD
    # ========================================================================

    def upsert_job_schedule(self, type: str, enabled: bool, interval_hours: float, priority: int):
        """Insert or update a job schedule.

        When enabling, sets next_run_at = now + interval to prevent immediate fire.
        When disabling, clears next_run_at.
        """
        with self._connection() as conn:
            if enabled:
                next_run_expr = "datetime('now', '+' || CAST(? * 3600 AS INTEGER) || ' seconds')"
                conn.execute(
                    f"""
                    INSERT INTO job_schedules (type, enabled, interval_hours, priority, next_run_at)
                    VALUES (?, 1, ?, ?, {next_run_expr})
                    ON CONFLICT(type) DO UPDATE SET
                        enabled = 1,
                        interval_hours = excluded.interval_hours,
                        priority = excluded.priority,
                        next_run_at = CASE
                            WHEN job_schedules.enabled = 1 THEN job_schedules.next_run_at
                            ELSE {next_run_expr}
                        END
                    """,
                    (type, interval_hours, priority, interval_hours, interval_hours)
                )
            else:
                conn.execute(
                    """
                    INSERT INTO job_schedules (type, enabled, interval_hours, priority, next_run_at)
                    VALUES (?, 0, ?, ?, NULL)
                    ON CONFLICT(type) DO UPDATE SET
                        enabled = 0,
                        interval_hours = excluded.interval_hours,
                        priority = excluded.priority,
                        next_run_at = NULL
                    """,
                    (type, interval_hours, priority)
                )

    def get_job_schedule(self, type: str) -> dict | None:
        """Get schedule for a job type."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM job_schedules WHERE type = ?", (type,)).fetchone()
            return dict(row) if row else None

    def get_all_job_schedules(self) -> list[dict]:
        """Get all job schedules."""
        with self._connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM job_schedules ORDER BY type").fetchall()]

    def update_schedule_last_run(self, type: str):
        """Update last_run_at to now and calculate next_run_at from interval."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE job_schedules
                SET last_run_at = datetime('now'),
                    next_run_at = datetime('now', '+' || CAST(interval_hours * 3600 AS INTEGER) || ' seconds')
                WHERE type = ?
                """,
                (type,)
            )

    def get_due_schedules(self) -> list[dict]:
        """Get enabled schedules that are past their next_run_at."""
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(
                """
                SELECT * FROM job_schedules
                WHERE enabled = 1 AND (next_run_at IS NULL OR next_run_at <= datetime('now'))
                """
            ).fetchall()]


# Convenience function
def open_recommendations_db(path: str | Path) -> RecommendationsDB:
    """Open or create a recommendations database."""
    return RecommendationsDB(path)
