"""Immich-style face group service: build, browse, merge, assign clusters.

Id spaces
---------
- ``face_clusters.performer_id`` is ALWAYS a local Stash performer id (set by
  assign), or None.
- A face's ``best_match_id`` is a StashDB / stash-box id. A group's identity in
  that space lives in ``face_cluster_stash_ids``; every match lookup goes through
  ``db.get_stash_id_cluster_map()``.

Curated state
-------------
Assigned, ignored, banned and *pinned* groups are user state: builds never
dissolve them, and their faces (which have a membership) are never re-pooled.
Ejected / split faces carry rejection memory (``face_rejections``) so no build
puts them back into the group they were removed from, including after that
group was merged into another (rejections resolve through the merge map).

Concurrency
-----------
A build reads the pool and the groups up front and writes much later (it runs
on a worker thread). Every curation write (assign, update, merge, split,
eject, ban, unban, ignore, delete) and every build hold ``_STATE_LOCK``, so a
curation never interleaves with a build: while a build runs, curation raises
BuildInProgress (HTTP 409, "try again") instead of being silently undone or
duplicated by the build. Network calls to Stash (scene tagging, stash-id
lookups) during assign / merge run outside the lock, after the group is
already pinned. The DB writes a build makes are also conditional (pinned
groups are never dissolved, only pooled faces are placed), so a writer that
bypasses the service cannot corrupt state either.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager

import numpy as np

from library_clustering import cluster_library_faces
from recommendations_db import RecommendationsDB

logger = logging.getLogger(__name__)

# One build at a time, process-wide (HTTP handler and job runner share it).
# Held for the whole build: "a build is running".
_BUILD_LOCK = threading.Lock()
# Serializes builds with curation writes (re-entrant per thread).
_STATE_LOCK = threading.RLock()
_local = threading.local()
# How long a curation write (or a build) waits for an in-flight curation.
LOCK_WAIT_SEC = 10.0
# Scene tagging is read-then-write of a scene's full performer list; assign,
# merge and auto-tag run it concurrently (handlers are in the threadpool), so
# each scene's read-modify-write is serialized process-wide. Never held while
# taking another lock.
_SCENE_TAG_LOCK = threading.Lock()

_LIVE_STATUSES = ("assigned", "matched", "open")
_USER_SETTABLE_STATUSES = {"open", "ignored"}
_CENTROID_CHUNK = 1024


class ClusterNotFound(ValueError):
    """A referenced face cluster does not exist."""


class InvalidClusterOperation(ValueError):
    """The requested cluster operation is not allowed in the current state."""


class BuildInProgress(RuntimeError):
    """A cluster build is running (or face groups are busy); retry later."""


@contextmanager
def curation_lock():
    """Hold the face-group state lock for a curation write.

    Raises BuildInProgress at once while a build runs, or after LOCK_WAIT_SEC
    if another curation keeps the lock. Re-entrant within a thread.
    """
    depth = getattr(_local, "depth", 0)
    if depth == 0:
        if _BUILD_LOCK.locked():
            raise BuildInProgress("a face group build is running; try again when it finishes")
        if not _STATE_LOCK.acquire(timeout=LOCK_WAIT_SEC):
            raise BuildInProgress("face groups are busy; try again")
    else:
        _STATE_LOCK.acquire()
    _local.depth = depth + 1
    try:
        yield
    finally:
        _local.depth = depth
        _STATE_LOCK.release()


def is_build_running() -> bool:
    return _BUILD_LOCK.locked()


def _is_stashdb_endpoint(endpoint) -> bool:
    """Face matches (best_match_id) are StashDB ids."""
    return "stashdb.org" in str(endpoint or "").lower()


def _face_vec(face: dict) -> np.ndarray | None:
    v = np.concatenate([
        np.frombuffer(face["facenet_emb"], dtype=np.float32),
        np.frombuffer(face["arcface_emb"], dtype=np.float32),
    ])
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return None
    return v / norm


class FaceClusterService:
    def __init__(self, db: RecommendationsDB):
        self.db = db

    # ------------------------------------------------------------------ helpers

    def _require(self, cluster_id: int) -> dict:
        cluster = self.db.get_face_cluster(cluster_id)
        if cluster is None:
            raise ClusterNotFound(f"cluster {cluster_id} not found")
        return cluster

    def _members_of(self, cluster_id: int, face_ids: list[int]) -> list[int]:
        """The subset of face_ids that are currently members of cluster_id (order kept)."""
        out: list[int] = []
        seen: set[int] = set()
        for fid in face_ids:
            if fid in seen:
                continue
            seen.add(fid)
            if any(m["id"] == cluster_id for m in self.db.get_face_cluster_membership(fid)):
                out.append(fid)
        return out

    def _reject_from(self, cluster: dict, face_ids: list[int], ban: bool = False) -> None:
        """Remember that face_ids must not rejoin this cluster (or its identity).

        Call before removing the faces. A performer group with no stash ids yet
        (backfill pending, performer not linked) is identified by the StashDB
        id its *remaining* members agree on, so the ejected face cannot re-seed
        a matched group for the same identity. The ejected faces never vote:
        their own match is evidence against this group, not for it, and must
        stay free to take them to that identity's own group.

        A performer group whose performer is not linked on StashDB (unsynced)
        only *guesses* its StashDB id from members, so here:
        - a member-derived id an ejected face carried that no longer has a
          clear majority (>= 2, > 50%) of the remaining members is dropped
          from the group and blocked (backfill never re-adds it);
        - an ejected face is never rejected from its own best match: that is
          its true identity, not this group's.
        Neither applies to a ban (ban=True): banned faces are junk detections,
        no evidence about the group's identity.
        """
        cid = cluster["id"]
        stash_ids = list(cluster.get("stash_ids") or [])
        guessed = (not ban and cluster.get("status") == "assigned"
                   and not cluster.get("stash_ids_synced"))
        own: dict[int, str | None] = {}
        if guessed:
            for fid in face_ids:
                row = self.db.get_library_face(fid)
                own[fid] = row.get("best_match_id") if row else None
            keep = self.db.get_member_anchor(cid, min_count=2, exclude_face_ids=face_ids)
            lost = {s for s in stash_ids if s in set(own.values()) and s != keep}
            if lost:
                self.db.block_cluster_stash_ids(cid, lost)
                stash_ids = [s for s in stash_ids if s not in lost]
                self.db.set_cluster_stash_ids(cid, stash_ids, replace=True)
        if not stash_ids and cluster.get("status") in ("assigned", "matched"):
            derived = self.db.get_member_anchor(cid, min_count=1, exclude_face_ids=face_ids)
            if derived:
                stash_ids = [derived]
        performer = cluster.get("performer_id") if cluster.get("status") == "assigned" else None
        by_ids: dict[tuple[str, ...], list[int]] = {}
        for fid in face_ids:
            ids = tuple(s for s in stash_ids if not (guessed and s == own.get(fid)))
            by_ids.setdefault(ids, []).append(fid)
        for ids, fids in by_ids.items():
            self.db.add_face_rejections(fids, cluster_id=cid, performer_id=performer, stash_ids=ids)

    def _drop_rejected_by_performer(self, cluster_id: int, performer_id, face_ids: list[int]) -> set[int]:
        """Return to the pool the faces among face_ids (members of cluster_id)
        the user rejected from performer_id: a group action (assign, merge)
        never tags a performer onto faces ejected from it. Returns them."""
        if not performer_id or not face_ids:
            return set()
        rej = self.db.get_face_rejections(face_ids)
        out = {f for f, r in rej.items() if str(performer_id) in r["performers"]}
        if out:
            self.db.remove_faces_from_cluster(cluster_id, sorted(out))
        return out

    def _tag_scenes(self, stash_client, performer_id: str, scene_ids: list[int]) -> tuple[int, int, list[dict]]:
        """Read-then-write each distinct scene once, under _SCENE_TAG_LOCK (a
        concurrent tagger of the same scene would otherwise drop our performer
        or we theirs). Returns (tagged, already, failed)."""
        tagged, skipped, failed = 0, 0, []
        for sid in dict.fromkeys(scene_ids):
            try:
                with _SCENE_TAG_LOCK:
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
        return tagged, skipped, failed

    # ------------------------------------------------------------------ build

    def build_clusters(
        self,
        distance_threshold: float = 0.55,
        min_cluster_size: int = 3,
        seed_by_match: bool = True,
        replace_existing: bool = False,
        incremental: bool = False,
        auto_tag: bool = False,
        stash_client=None,
        should_stop=None,
    ) -> dict:
        """Cluster library faces. Raises BuildInProgress if a build is running.

        Both modes place only *pooled* faces: faces with no membership in any
        cluster. Faces in assigned / ignored / banned / pinned groups are never
        re-clustered.

        - Incremental (incremental=True): pooled faces are placed by StashDB
          match into the group carrying that stash id, then by centroid into
          the nearest live group (assigned/matched/open) within threshold, and
          the rest form new open groups of at least min_cluster_size.
        - Full (default): first dissolves the automatic groups -- unpinned open
          groups are deleted, unpinned matched groups are emptied but kept (so
          their ids and stash ids are reused) -- then places the pool the same
          way and deletes groups left empty. Idempotent: a second full build
          yields the same groups and never duplicates a matched group.
          ``replace_existing`` is accepted for API compatibility but ignored: a
          full build always replaces the unpinned automatic groups.

        Before placing, both modes backfill stash ids of assigned groups (from
        Stash when stash_client is given -- a full build re-checks every
        assigned group's links, an incremental one only the unsynced ones --
        else from a clear majority of their members' own matches) and fold automatic groups that duplicate the
        identity of a higher-priority group (e.g. a matched group for a stash
        id an assigned group now carries) into that group.

        Faces are never placed into a group they were ejected / split from
        (rejection memory, resolved through the merge map).

        The build reads the pool and the groups up front and writes later;
        faces deleted by a concurrent re-identify are skipped, and a group
        deleted or merged meanwhile is resolved through the merge map (or its
        faces stay pooled). Such races never abort the build.

        auto_tag (both modes, needs stash_client): faces landing in an assigned
        group tag their scenes (one read-then-write per distinct scene).

        should_stop: optional callable; when it returns True the build stops
        at the next checkpoint (between passes, centroid chunks, dissolved /
        synced / newly formed groups) with "cancelled", so a cancelled job's
        thread releases the build lock promptly. Auto-tagging of faces already
        placed still runs (it is their only chance).

        Returns {"mode", "clusters_created", "matched_created", "faces_assigned",
        "faces_total", "faces_new", "absorbed", "groups_absorbed_into",
        "tagged_scenes", "groups_folded", "cancelled"} (+ "dissolved" in full mode).
        """
        if not _BUILD_LOCK.acquire(blocking=False):
            raise BuildInProgress("a face cluster build is already running")
        stop = should_stop or (lambda: False)
        try:
            # wait for an in-flight curation write to finish; new ones now fail fast
            if not _STATE_LOCK.acquire(timeout=LOCK_WAIT_SEC):
                raise BuildInProgress("face groups are busy with a curation; try again")
            depth = getattr(_local, "depth", 0)
            _local.depth = depth + 1
            try:
                if stop():
                    return {"mode": "incremental" if incremental else "full", **self._empty_result(0),
                            "cancelled": True}
                if incremental:
                    return self._build_incremental(
                        distance_threshold, min_cluster_size, seed_by_match, auto_tag, stash_client, stop,
                    )
                return self._build_full(
                    distance_threshold, min_cluster_size, seed_by_match, auto_tag, stash_client, stop,
                )
            finally:
                _local.depth = depth
                _STATE_LOCK.release()
        finally:
            _BUILD_LOCK.release()

    @staticmethod
    def _empty_result(total: int) -> dict:
        return {
            "clusters_created": 0, "matched_created": 0, "faces_assigned": 0,
            "faces_total": total, "faces_new": 0, "absorbed": 0,
            "groups_absorbed_into": 0, "tagged_scenes": 0, "groups_folded": 0,
            "cancelled": False,
        }

    def _prepare(self, stash_client, moves: dict[int, list[int]], resync_all: bool = False,
                 stop=None) -> int:
        """Stash-id backfill + duplicate folding. Returns groups folded; the
        faces each fold moved are added to moves (keeper -> face ids).
        resync_all (full builds): re-check the Stash links of every assigned
        group, not only the unsynced ones."""
        if stash_client is not None:
            try:
                self.sync_assigned_stash_ids(stash_client, include_synced=resync_all, should_stop=stop)
            except Exception:
                logger.warning("stash id backfill failed; continuing build", exc_info=True)
        if stop is not None and stop():
            return 0
        try:
            self.backfill_member_anchors()
        except Exception:
            logger.warning("member anchor backfill failed; continuing build", exc_info=True)
        return self.fold_duplicate_groups(moves)

    def _finish_build(self, result: dict, placed: dict[int, list[int]], auto_tag: bool,
                      stash_client) -> dict:
        """Auto-tag scenes of faces that landed in assigned groups, whether
        placed by this build or folded in with a duplicate group (either way
        no later build places them again, so this is the only chance)."""
        if auto_tag and stash_client is not None:
            for cid, ids in placed.items():
                c = self.db.get_face_cluster(cid)
                if c is None or c.get("status") != "assigned" or not c.get("performer_id"):
                    continue
                scenes = self.db.get_scene_ids_for_faces(ids)
                tagged, _skipped, _failed = self._tag_scenes(stash_client, str(c["performer_id"]), scenes)
                result["tagged_scenes"] += tagged
        return result

    @staticmethod
    def _merge_moves(*maps: dict[int, list[int]]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for m in maps:
            for cid, ids in m.items():
                out.setdefault(cid, []).extend(ids)
        return out

    def _build_incremental(self, distance_threshold, min_cluster_size, seed_by_match, auto_tag,
                           stash_client, stop) -> dict:
        moves: dict[int, list[int]] = {}
        folded = self._prepare(stash_client, moves, stop=stop)
        result, placed = self._place_pool(distance_threshold, min_cluster_size, seed_by_match, stop)
        result["groups_folded"] = folded
        self._finish_build(result, self._merge_moves(moves, placed), auto_tag, stash_client)
        return {"mode": "incremental", **result}

    def _build_full(self, distance_threshold, min_cluster_size, seed_by_match, auto_tag,
                    stash_client, stop) -> dict:
        # Conditional writes: a group pinned since it was listed is kept.
        dissolved = 0
        for c in self.db.get_clusters_by_status("open"):
            if stop():
                break
            if not c.get("pinned") and self.db.delete_face_cluster(c["id"], only_unpinned=True):
                dissolved += 1
        for c in self.db.get_clusters_by_status("matched"):
            if stop():
                break
            if not c.get("pinned") and self.db.clear_cluster_members(c["id"], only_unpinned=True) is not None:
                dissolved += 1
        moves: dict[int, list[int]] = {}
        folded = self._prepare(stash_client, moves, resync_all=True, stop=stop)
        result, placed = self._place_pool(distance_threshold, min_cluster_size, seed_by_match, stop)
        result["groups_folded"] = folded
        # Curation is locked out for the whole build, so empty pinned groups
        # here are ghosts (their faces were deleted), not groups being filled.
        self.db.delete_empty_clusters(("matched", "open", "banned"), include_pinned=True)
        self._finish_build(result, self._merge_moves(moves, placed), auto_tag, stash_client)
        return {"mode": "full", "dissolved": dissolved, **result}

    def fold_duplicate_groups(self, moves: dict[int, list[int]] | None = None) -> int:
        """Fold unpinned matched/open groups into the group that wins their stash
        id (assigned > ignored > matched > open). Members rejected from the
        winner return to the pool instead. Returns groups folded; if moves is
        given, the moved face ids are recorded there per keeper."""
        folded = 0
        for keeper, dups in self.db.get_duplicate_stash_id_groups():
            for dup in dups:
                members = [f["id"] for f in self.db.get_representative_faces(dup, limit=10**9)]
                rej = self.db.get_face_rejections(members) if members else {}
                keeper_row = self.db.get_face_cluster(keeper)
                if keeper_row is None:
                    break
                keeper_stash = set(keeper_row.get("stash_ids") or [])
                merge_map = self.db.get_cluster_merge_map()
                excluded = []
                for fid, r in rej.items():
                    if (any(merge_map.get(ref, ref) == keeper for ref in r["clusters"])
                            or (keeper_row.get("status") == "assigned"
                                and str(keeper_row.get("performer_id")) in r["performers"])
                            or (keeper_stash & r["stash_ids"])):
                        excluded.append(fid)
                if self.db.fold_cluster_into(dup, keeper, excluded) is not None:
                    folded += 1
                    if moves is not None:
                        ex = set(excluded)
                        moves.setdefault(keeper, []).extend(f for f in members if f not in ex)
        return folded

    def _place_pool(self, distance_threshold, min_cluster_size, seed_by_match,
                    stop) -> tuple[dict, dict[int, list[int]]]:
        """Place every pooled (membership-less) face. Shared by both build modes.
        Returns (result, placed: cluster id -> face ids written)."""
        total = self.db.get_library_face_count()
        pool: list[dict] = []
        for batch in self.db.iter_library_faces(unassigned_only=True):
            pool.extend(batch)

        result = self._empty_result(total)
        result["faces_new"] = len(pool)
        placed: dict[int, list[int]] = {}
        if stop():
            result["cancelled"] = True
            return result, placed
        if not pool:
            return result, placed

        rej = self.db.get_face_rejections([f["id"] for f in pool])
        merge_map = self.db.get_cluster_merge_map()
        cl_stash = self.db.get_all_cluster_stash_ids()
        clusters = {c["id"]: c for c in self.db.get_clusters_by_status(*_LIVE_STATUSES)}
        had_members = {cid for cid, c in clusters.items() if c.get("face_count", 0) > 0}
        stash_map = self.db.get_stash_id_cluster_map()
        created: list[int] = []

        def is_rejected(face_id: int, cluster_id: int) -> bool:
            r = rej.get(face_id)
            if not r:
                return False
            if any(merge_map.get(ref, ref) == cluster_id for ref in r["clusters"]):
                return True
            pid = clusters.get(cluster_id, {}).get("performer_id")
            if pid is not None and str(pid) in r["performers"]:
                return True
            return bool(cl_stash.get(cluster_id, set()) & r["stash_ids"])

        def place(cid: int, ids: list[int]) -> None:
            """Write a placement, tolerating groups/faces that vanished since the snapshot."""
            n = self.db.place_faces(cid, ids)
            if n is None:
                live = self.db.resolve_cluster_id(cid)
                if live is None or live == cid:
                    logger.info("cluster %s vanished during build; %d faces stay pooled", cid, len(ids))
                    return
                if live not in clusters:
                    row = self.db.get_face_cluster(live)
                    if row is None:
                        return
                    clusters[live] = row
                    cl_stash[live] = set(row.get("stash_ids") or [])
                ids = [f for f in ids if not is_rejected(f, live)]
                cid = live
                n = self.db.place_faces(cid, ids)
                if n is None:
                    return
            if n:
                placed.setdefault(cid, []).extend(ids)

        def finish() -> dict:
            for cid in created:
                if self.db.get_cluster_face_count(cid) == 0:
                    self.db.delete_face_cluster(cid)
            placed_count = sum(len(ids) for ids in placed.values())
            result["absorbed"] = sum(len(ids) for cid, ids in placed.items() if cid in had_members)
            result["groups_absorbed_into"] = sum(1 for cid in placed if cid in had_members)
            result["faces_assigned"] = placed_count
            result["clusters_created"] = sum(
                1 for cid in created if self.db.get_face_cluster(cid) is not None)
            return result, placed

        # Pass 1: by StashDB match.
        remaining: list[dict] = []
        by_match: dict[int, list[int]] = {}
        for f in pool:
            fid, x = f["id"], f.get("best_match_id")
            if x:
                cid = stash_map.get(x)
                if cid is not None:
                    if not is_rejected(fid, cid):
                        by_match.setdefault(cid, []).append(fid)
                        continue
                elif seed_by_match and x not in rej.get(fid, {}).get("stash_ids", set()):
                    name = f.get("best_match_name")
                    cid = self.db.create_face_cluster(
                        name=name or f"Matched {x[:8]}", status="matched",
                        performer_id=None, performer_name=name, stash_ids=[x],
                    )
                    created.append(cid)
                    clusters[cid] = {"id": cid, "status": "matched", "performer_id": None,
                                     "performer_name": name, "face_count": 0}
                    stash_map[x] = cid
                    cl_stash[cid] = {x}
                    result["matched_created"] += 1
                    by_match.setdefault(cid, []).append(fid)
                    continue
            remaining.append(f)
        for cid, ids in by_match.items():
            place(cid, ids)
        if stop():
            result["cancelled"] = True
            return finish()

        # Pass 2: nearest non-rejected centroid within threshold (centroids computed once).
        leftovers: list[dict] = []
        cids: list[int] = []
        cents: list[np.ndarray] = []
        if remaining:
            for cid in list(clusters):
                cent = self.db.get_cluster_centroid(cid)
                if cent is not None:
                    cids.append(cid)
                    cents.append(np.asarray(cent, dtype=np.float32))
        if remaining and cents:
            C = np.stack(cents)
            vecs, vec_faces = [], []
            for f in remaining:
                v = _face_vec(f)
                if v is None or v.shape[0] != C.shape[1]:
                    leftovers.append(f)
                else:
                    vecs.append(v)
                    vec_faces.append(f)
            centroid_placed: dict[int, list[int]] = {}
            for start in range(0, len(vec_faces), _CENTROID_CHUNK):
                if stop():
                    result["cancelled"] = True
                    return finish()
                chunk = vec_faces[start:start + _CENTROID_CHUNK]
                D = 1.0 - np.stack(vecs[start:start + _CENTROID_CHUNK]) @ C.T
                for f, row in zip(chunk, D):
                    cand = np.nonzero(row <= distance_threshold)[0]
                    target = None
                    for j in cand[np.argsort(row[cand], kind="stable")]:
                        if not is_rejected(f["id"], cids[j]):
                            target = cids[j]
                            break
                    if target is None:
                        leftovers.append(f)
                    else:
                        centroid_placed.setdefault(target, []).append(f["id"])
            for cid, ids in centroid_placed.items():
                place(cid, ids)
        else:
            leftovers.extend(remaining)
        if stop():
            result["cancelled"] = True
            return finish()

        # Pass 3: leftovers form new open groups.
        if leftovers:
            for group in cluster_library_faces(leftovers, distance_threshold):
                if stop():
                    result["cancelled"] = True
                    break
                if len(group) < min_cluster_size:
                    continue
                cid = self.db.create_face_cluster(status="open")
                created.append(cid)
                n = self.db.place_faces(cid, group)
                if n:
                    placed.setdefault(cid, []).extend(group)

        return finish()

    def _apply_performer_stash_ids(self, cluster_id: int, lookup: list[dict]) -> bool:
        """Set an assigned group's stash ids from its performer's Stash links.

        - Performer linked on StashDB: its links ARE the group's identity. They
          replace the group's ids; ids that drop out (a match anchor the link
          contradicts) are blocked so member anchors never re-add them. The
          group is marked synced.
        - Not linked on StashDB (no links, or other stash-boxes only): its ids
          are unioned in (the group's StashDB anchor is all it has) and the
          group stays unsynced, so every build re-checks until it is linked.

        Returns True if the group's ids changed.
        """
        ids = list(dict.fromkeys(s["stash_id"] for s in lookup if s.get("stash_id")))
        linked = any(_is_stashdb_endpoint(s.get("endpoint")) and s.get("stash_id") for s in lookup)
        before = set(self.db.get_cluster_stash_ids(cluster_id))
        self.db.unblock_cluster_stash_ids(cluster_id, ids)
        if linked:
            self.db.block_cluster_stash_ids(cluster_id, before - set(ids))
            self.db.set_cluster_stash_ids(cluster_id, ids, replace=True)
            self.db.set_cluster_stash_ids_synced(cluster_id, True)
        else:
            self.db.set_cluster_stash_ids(cluster_id, ids, replace=False)
            self.db.set_cluster_stash_ids_synced(cluster_id, False)
        return set(self.db.get_cluster_stash_ids(cluster_id)) != before

    def sync_assigned_stash_ids(self, stash_client, include_synced: bool = False,
                                should_stop=None) -> int:
        """Refresh the stash ids of assigned groups whose performer is not
        known to be linked on StashDB (assign-time lookup failed or found no
        StashDB link, legacy groups) or that have none -- with include_synced,
        of every assigned group (a performer can be relinked in Stash); see
        _apply_performer_stash_ids. Returns groups whose ids changed. Errors
        are logged and swallowed per group. Stops early when should_stop().
        """
        if stash_client is None or not hasattr(stash_client, "get_performer_stash_ids_sync"):
            return 0
        cache: dict[str, list[dict] | None] = {}
        updated = 0
        for c in self.db.get_assigned_clusters_needing_stash_sync(include_synced=include_synced):
            if should_stop is not None and should_stop():
                break
            pid = str(c["performer_id"])
            try:
                if pid not in cache:
                    cache[pid] = stash_client.get_performer_stash_ids_sync(pid)
                if cache[pid] is None:
                    continue  # performer not found: retry on a later build
                if self._apply_performer_stash_ids(c["id"], cache[pid]):
                    updated += 1
            except Exception:
                logger.warning("stash id lookup failed for performer %s (cluster %s)",
                               pid, c["id"], exc_info=True)
        return updated

    def backfill_member_anchors(self) -> int:
        """Assigned groups whose performer is not linked on StashDB (unsynced)
        gain the StashDB id a clear majority (>= 2, > 50%) of their anchored
        members share -- also when the group already has other ids (e.g. the
        performer's links to other stash-boxes) -- unless another assigned
        group carries it or a user decision blocked it for this group.
        Returns groups updated."""
        have = self.db.get_all_cluster_stash_ids()
        assigned = self.db.get_clusters_by_status("assigned")
        assigned_ids: dict[str, int] = {}
        for c in assigned:
            for sid in have.get(c["id"], ()):
                assigned_ids.setdefault(sid, c["id"])
        updated = 0
        for c in assigned:
            if c.get("stash_ids_synced"):
                continue  # the performer's StashDB link is the identity
            sid = self.db.get_member_anchor(c["id"])
            if sid is None or sid in have.get(c["id"], ()):
                continue
            if assigned_ids.get(sid, c["id"]) != c["id"]:
                continue
            if sid in self.db.get_blocked_stash_ids(c["id"]):
                continue
            self.db.set_cluster_stash_ids(c["id"], [sid], replace=False)
            assigned_ids[sid] = c["id"]
            updated += 1
        return updated

    # ------------------------------------------------------------------ assign

    def assign_performer(
        self,
        cluster_id: int,
        performer_id: str,
        performer_name: str,
        stash_client,
        create_performer=None,
    ) -> dict:
        """Assign a cluster to a local Stash performer and tag all its scenes.

        create_performer (create-and-assign): a callable returning the new
        performer {"id", "name"}, called under the curation lock after the
        group is validated, so a refused assign (build running, group missing
        or banned) never leaves an orphan performer; the result then carries
        "created_performer".

        Under the curation lock the group becomes assigned and pinned (so no
        build can dissolve or fold it) before any Stash call; a missing group
        raises ClusterNotFound, a running build BuildInProgress. Then, outside
        the lock, each scene with cluster faces is tagged (read-then-write,
        performer_ids is a full replacement list) and the performer's stash-box
        ids are applied (see _apply_performer_stash_ids). If that lookup fails
        the group stays unsynced and a later build applies them.

        Re-assigning to another performer (also after the group was set open
        or ignored in between) clears the group's stash ids and blocks them:
        they were the old performer's identity.
        """
        with curation_lock():
            prev = self._require(cluster_id)
            if prev["status"] == "banned":
                raise InvalidClusterOperation("banned faces cannot be assigned; unban them first")
            created = None
            if create_performer is not None:
                created = create_performer()
                performer_id, performer_name = str(created["id"]), created["name"]
            # whatever the status now: a group assigned, then set open/ignored,
            # still carries the old performer's id and stash ids
            if prev.get("performer_id") and str(prev["performer_id"]) != str(performer_id):
                old = self.db.get_cluster_stash_ids(cluster_id)
                self.db.block_cluster_stash_ids(cluster_id, old)
                self.db.set_cluster_stash_ids(cluster_id, [], replace=True)
            if not self.db.update_face_cluster(
                cluster_id,
                status="assigned",
                performer_id=performer_id,
                performer_name=performer_name,
                pinned=True,
            ):
                raise ClusterNotFound(f"cluster {cluster_id} not found")
            self.db.set_cluster_stash_ids_synced(cluster_id, False)
            self._drop_rejected_by_performer(cluster_id, performer_id, [
                f["id"] for f in self.db.get_representative_faces(cluster_id, limit=10**9)])
            scene_ids = self.db.get_cluster_scene_ids(cluster_id)

        tagged, skipped, failed = self._tag_scenes(stash_client, performer_id, scene_ids)

        lookup = None
        if hasattr(stash_client, "get_performer_stash_ids_sync"):
            try:
                lookup = stash_client.get_performer_stash_ids_sync(performer_id)
                if lookup is None:
                    logger.warning("performer %s not found in Stash; keeping stash ids of cluster %s",
                                   performer_id, cluster_id)
            except Exception:
                logger.warning("stash id lookup failed for performer %s; keeping stash ids of cluster %s",
                               performer_id, cluster_id, exc_info=True)
        if lookup is not None:
            try:
                with curation_lock():
                    now = self.db.get_face_cluster(cluster_id)
                    if (now is not None and now["status"] == "assigned"
                            and str(now.get("performer_id")) == str(performer_id)):
                        self._apply_performer_stash_ids(cluster_id, lookup)
            except BuildInProgress:
                logger.info("build running; stash ids of cluster %s are applied by a later build",
                            cluster_id)

        out = {
            "cluster_id": cluster_id,
            "performer_id": performer_id,
            "performer_name": performer_name,
            "scenes_tagged": tagged,
            "scenes_already_tagged": skipped,
            "scenes_failed": failed,
            "stash_ids": self.db.get_cluster_stash_ids(cluster_id),
        }
        if created is not None:
            out["created_performer"] = created
        return out

    # ------------------------------------------------------------------ curate

    def update_cluster(self, cluster_id: int, name: str | None = None, status: str | None = None) -> dict:
        """Rename and/or set status open|ignored. Pins the group. Returns the cluster."""
        if status is not None and status not in _USER_SETTABLE_STATUSES:
            raise InvalidClusterOperation(
                f"status must be one of {sorted(_USER_SETTABLE_STATUSES)}, got {status!r}"
            )
        with curation_lock():
            cluster = self._require(cluster_id)
            if status is not None and cluster["status"] == "banned":
                raise InvalidClusterOperation("the status of a banned group cannot be changed; unban it")
            if not self.db.update_face_cluster(cluster_id, name=name, status=status, pinned=True):
                raise ClusterNotFound(f"cluster {cluster_id} not found")
            return self.db.get_face_cluster(cluster_id)

    def merge_clusters(self, source_ids: list[int], target_id: int, stash_client=None) -> dict:
        """Atomically move all faces (and stash ids) of sources into target.

        Validation happens before any change, under the curation lock (no
        assign / build can change a group between the checks and the move):
        missing clusters raise ClusterNotFound; banned clusters, mixing ignored
        with non-ignored groups, an empty source list, or conflicting performer
        assignments raise InvalidClusterOperation. The merge is logged
        (rejections and rebuilds follow it) and the target is pinned.

        - An open target that absorbs a matched source becomes matched (it now
          carries that identity), taking the source's name if it has none.
        - Merging into an assigned target tags the moved faces' scenes with its
          performer when stash_client is given (like assign does), after the
          lock is released.
        """
        with curation_lock():
            target = self._require(target_id)
            sources = list(dict.fromkeys(s for s in source_ids if s != target_id))
            if not sources:
                raise InvalidClusterOperation("no source clusters to merge")
            src_rows = {}
            missing = []
            for sid in sources:
                row = self.db.get_face_cluster(sid)
                if row is None:
                    missing.append(sid)
                else:
                    src_rows[sid] = row
            if missing:
                raise ClusterNotFound(f"clusters not found: {missing}")

            involved = [target, *src_rows.values()]
            if any(c["status"] == "banned" for c in involved):
                raise InvalidClusterOperation("banned clusters cannot be merged")
            ignored = [c["status"] == "ignored" for c in involved]
            if any(ignored) and not all(ignored):
                raise InvalidClusterOperation(
                    "ignored groups can only be merged with other ignored groups; un-ignore it first")
            performers = {str(c["performer_id"]) for c in involved
                          if c["status"] == "assigned" and c.get("performer_id")}
            if len(performers) > 1:
                raise InvalidClusterOperation("clusters are assigned to different performers")
            if target["status"] != "assigned" and any(c["status"] == "assigned" for c in src_rows.values()):
                raise InvalidClusterOperation("merge into the assigned group")
            if not target.get("performer_id") and performers:
                raise InvalidClusterOperation("merge into the group assigned to a performer")

            tgt_performer = target.get("performer_id") if target["status"] == "assigned" else None
            moving = [f["id"] for sid in sources
                      for f in self.db.get_representative_faces(sid, limit=10**9)] if tgt_performer else []

            try:
                res = self.db.move_faces_to_cluster(sources, target_id)
            except (ValueError, sqlite3.IntegrityError) as e:  # raced with a delete
                raise ClusterNotFound(str(e)) from e
            dropped = self._drop_rejected_by_performer(target_id, tgt_performer, moving)
            moved_scenes: list[int] = []
            if tgt_performer and stash_client is not None:
                moved_scenes = self.db.get_scene_ids_for_faces([f for f in moving if f not in dropped])

            matched_src = [c for c in src_rows.values() if c["status"] == "matched"]
            if target["status"] == "open" and matched_src:
                donor = max(matched_src, key=lambda c: c.get("face_count") or 0)
                self.db.update_face_cluster(
                    target_id,
                    status="matched",
                    name=None if target.get("name") else donor.get("name"),
                    performer_name=None if target.get("performer_name") else donor.get("performer_name"),
                )

        out = {"target_id": target_id, "merged_sources": sources, "faces_moved": res["faces_moved"],
               "scenes_tagged": 0}
        if moved_scenes:
            tagged, skipped, failed = self._tag_scenes(
                stash_client, str(target["performer_id"]), moved_scenes)
            out.update(scenes_tagged=tagged, scenes_already_tagged=skipped, scenes_failed=failed)
        return out

    def split_cluster(self, cluster_id: int, face_ids: list[int], new_status: str = "open") -> int:
        """Move faces out of a cluster into a new pinned cluster; they are
        remembered as rejected from the original, and the original is pinned
        (a full build would otherwise dissolve it and let the split faces
        regroup with their former siblings). Create, move and pin are one
        transaction."""
        with curation_lock():
            cluster = self._require(cluster_id)
            members = self._members_of(cluster_id, face_ids)
            if not members:
                raise InvalidClusterOperation(f"none of the faces are in cluster {cluster_id}")
            self._reject_from(cluster, members)
            try:
                return self.db.split_faces_to_new_cluster(cluster_id, members, new_status)
            except ValueError as e:
                raise ClusterNotFound(str(e)) from e

    def eject_faces(self, cluster_id: int, face_ids: list[int], ban: bool = False) -> int:
        """Remove faces from a group back to the pool. They are remembered as
        rejected from this group (and its performer / stash ids), so no build
        puts them back. The group is pinned, or deleted if an automatic group is
        left empty. Returns the number of faces removed."""
        with curation_lock():
            cluster = self._require(cluster_id)
            members = self._members_of(cluster_id, face_ids)
            removed = 0
            if members:
                self._reject_from(cluster, members, ban=ban)
                removed = self.db.remove_faces_from_cluster(cluster_id, members)
            if not (cluster["status"] in ("open", "matched") and self.db.delete_cluster_if_empty(cluster_id)):
                self.db.update_face_cluster(cluster_id, pinned=True)
            return removed

    def eject_or_ban(self, cluster_id: int, face_ids: list[int], ban: bool = False) -> dict:
        """The eject endpoint: eject (recording the rejection from this group),
        then with ban=True ban the faces that were members of it -- all in one
        curation-lock section, so no build sees them pooled in between.
        Returns {"ejected": n} or {"banned": n}."""
        with curation_lock():
            self._require(cluster_id)
            members = self._members_of(cluster_id, face_ids)
            removed = self.eject_faces(cluster_id, face_ids, ban=ban)
            if ban:
                return {"banned": self.ban_faces(members)}
            return {"ejected": removed}

    def ban_faces(self, face_ids: list[int]) -> int:
        """Eject faces from any group (live or ignored) and put them in a singleton 'banned'
        group — junk detections (background people, wrong boxes) that should
        never be clustered again. Banned faces are excluded from all
        clustering passes. Each face is banned in one transaction."""
        with curation_lock():
            return sum(1 for fid in dict.fromkeys(face_ids) if self.db.ban_face(fid))

    def unban_faces(self, face_ids: list[int]) -> int:
        """Return banned faces to the unassigned pool."""
        with curation_lock():
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
        """Mark a cluster as ignored (false positives, background faces). Pins it."""
        with curation_lock():
            c = self.db.get_face_cluster(cluster_id)
            if c is None:
                return False
            if c["status"] == "banned":
                raise InvalidClusterOperation("banned groups cannot be ignored; unban them first")
            return self.db.update_face_cluster(cluster_id, status="ignored", pinned=True)

    def delete_cluster(self, cluster_id: int) -> bool:
        """Delete a group; its faces return to the pool (rejections are kept)."""
        with curation_lock():
            return self.db.delete_face_cluster(cluster_id)

    def stats(self) -> dict:
        clusters = self.db.list_face_clusters()
        by = lambda s: sum(1 for c in clusters if c["status"] == s)  # noqa: E731
        return {
            "faces_total": self.db.get_library_face_count(),
            "clusters": sum(1 for c in clusters if c["status"] != "banned"),
            "open": by("open"),
            "matched": by("matched"),
            "assigned": by("assigned"),
            "ignored": by("ignored"),
            "banned": by("banned"),
        }
