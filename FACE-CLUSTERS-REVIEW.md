# Face Clusters Feature — Review Guide

> **Repo**: `github.com/SCSIExpress/stash-sense` (fork of `carrotwaxr/stash-sense`)
> **Branch**: `face-clusters` — 7 commits, ~3,000 lines added on top of upstream `main`
> **Purpose of this doc**: hand a reviewer (human or AI) a map of what was built, why,
> where it lives, and what to scrutinize.

## What this feature does

Upstream stash-sense matches faces in scenes against a pre-embedded database of ~108k
*known* performers (StashDB etc.). It had no Immich-style workflow: browse faces grouped
by similarity, name/merge the groups, assign them to performers, and have the source
scenes auto-tagged. Unknown faces (amateurs, personal content) were computed and then
discarded. This fork adds that workflow end-to-end:

1. **Persist per-face data** during identification: embeddings (FaceNet512 + ArcFace,
   1024-d concatenated), bbox, frame/timestamp reference, identify-time best match,
   and a face crop JPEG for UI display.
2. **Cluster library-wide**: HNSW-assisted union-find (brute-force fallback) over all
   stored faces; groups seeded from identify-time matches come in named and
   performer-linked (`matched`), the rest are `open`.
3. **User curation**: browse groups with face thumbnails, multi-select merge, eject
   (wrong face → back to the unassigned pool), ban (junk detection → never clustered
   again, reviewable/unban-able), rename, ignore.
4. **Assignment**: assign a group to an existing Stash performer or **create a new
   performer from the group** — both bulk-tag every scene containing the group's faces
   (read-then-write `sceneUpdate`; preserves existing tags; skips already-tagged).
5. **Incremental pickup**: new faces (from the ongoing fingerprint job) are absorbed
   into existing groups by centroid proximity — including assigned and merged groups —
   and can auto-tag their scenes on absorption. Merges are recorded durably so rebuilds
   honor them.

## Commit map (oldest → newest)

| Commit | What it introduced |
|---|---|
| `cb9dc3b` | Core: per-face persistence (schema v10), clustering engine, `/face-groups` API, plugin Face Groups tab, `cluster_library_faces` queue job |
| `433b100` | Fix: face-groups module self-inits (operations-module pattern) |
| `c2612e0` | Nav bar link (`stash-sense-nav.js`) + create-and-assign performer |
| `d610010` | Incremental clustering (centroid absorption), durable merge log (schema v11), auto-tag on absorb |
| `c0cef8a` | Expandable group detail, eject/ban faces, banned list |
| `2a975bf` | Fix: eject's `mode` param key collided with `runPluginOperation`'s operation mode (renamed `eject_mode`) |
| `f0c3e8a` | Fix: performer search modal — `/stash/search-performers` returns a bare array, code read `r.performers` |

## Where to look

### Backend (Python, `api/`)

| File | Role | Review focus |
|---|---|---|
| `recommendations_db.py` | Local SQLite layer. **Schema v10**: `library_faces`, `face_clusters`, `face_cluster_members`. **Schema v11**: `face_cluster_merge_log`. Migration 9→10→11 + fresh-create path (tables exist in BOTH `_create_schema` and `_migrate_schema` — this was a real bug early on). New methods: `add_library_face` (UNIQUE constraint dedupes per scene/frame/bbox), `iter_library_faces(unassigned_only)`, `get_cluster_centroid`, `get_unassigned_face_ids` (excludes ignored + banned), cluster CRUD, `record_cluster_merge`/`get_cluster_merge_map` (chain-resolving). | migration correctness, the UNIQUE-key dedup choice, merge-log chain resolution |
| `library_face_persist.py` | Hook called from `identify_scene` (in `identification_router.py` after the fingerprint save). Re-runs the same greedy clustering + merge the identify response used, maps each detected face to its in-scene person, anchors best-match data, writes crops via `LibraryFaceStore`, and upserts rows. Deletes a scene's previous face rows first (re-identify replaces). | the face→person mapping logic (`face_person_mapping` + positional occurrence counting — this is the subtlest code here), failure isolation (must never break identify) |
| `library_face_store.py` | Face crops on disk: `<DATA_DIR>/library_faces/<scene_id//1000>/<sha1>.jpg`, ~160 px JPEG q82. Bounds-clamped bbox with 4 px margin. | path traversal (key is hashed, scene-id sharded — safe?), silent `cv2.imwrite` failure handling (raises now) |
| `library_clustering.py` | Union-find over cosine distance. Voyager HNSW (`k=32` neighbours) when available; block-wise brute-force below 64 faces or when voyager import fails (no cp313 wheel — deliberate, do not "fix"). Vectors unit-normalized 1024-d concat. | threshold semantics (cosine distance vs the codebase's mixed usage), brute-force memory profile at ~100k faces |
| `face_cluster_service.py` | Orchestration: `_build_full` (from-scratch: seed-by-match anchors → existing assigned groups absorb same-performer faces → similarity clustering for the rest), `_build_incremental` (only unassigned faces; centroid match against live groups; auto-tag absorbed faces of assigned groups; leftovers → new open groups), `assign_performer` (read-then-write tagging loop), `merge_clusters` (moves faces + records merge log), `eject_faces` (empty open/matched groups auto-delete), `ban_faces`/`unban_faces` (singleton banned marker groups). | `_build_incremental` centroid staleness (refreshed per group after absorb), auto-tag failure isolation, `assign_performer` being synchronous (long loops block the HTTP request — known trade-off) |
| `face_clusters_router.py` | REST: `GET /face-groups` (list+stats), `POST /face-groups/build` (full or incremental), `GET/PATCH/DELETE /face-groups/{id}`, `POST .../assign`, `POST .../create-and-assign` (creates Stash performer then assigns), `POST .../eject` (eject_mode eject|ban), `POST /face-groups/unban`, `GET /face-groups/banned/list`, `GET /face-groups/{cid}/face/{fid}/crop` (serves JPEG). Registered in `main.py` inside a try/ImportError guard. | route ordering (`/{cluster_id}` vs literal paths), create-and-assign partial-failure handling (performer created but assign failed → 500 with performer id in message) |
| `jobs/cluster_faces_job.py` | Queue job wrapper (`cluster_library_faces` type; LIGHT resource, weekly-schedulable). Uses incremental mode. | settings fallbacks (`face_cluster_threshold`, `face_cluster_min_size`) |
| `job_models.py`, `queue_manager.py` | Job registration + dispatch branch for the new type. | — |
| `identification_router.py` | One integration point: after `save_scene_fingerprint`, calls `persist_from_identify` inside try/except (non-fatal). | that it truly can't break the hot identify path |
| `stash_client_unified.py` | New sync methods: `get_scene_performer_ids_sync`, `update_scene_performers_sync`, `create_performer_sync`; `search_performers` gained `image_path`. Sync-on-async: these use `_execute_sync` (plain httpx), NOT the rate-limited async `_execute`. | **Known design criticism**: sync calls bypass the rate limiter; bulk assign could hammer Stash. Acceptable for a local box but worth flagging. |

### Plugin (JavaScript, `plugin/`)

| File | Role | Review focus |
|---|---|---|
| `stash-sense-face-groups.js` | The Face Groups tab (fourth tab, self-injecting like Operations). List view (cards, thumbnails, filters by status, select/merge mode, banned list), detail view (full face grid, eject/ban selection, click face → scene), assign modal (search existing + create-new with favorite/disambiguation). | **The params-spread trap**: `SS.runPluginOperation(mode, params)` builds `{mode, ...params}` — any param key named `mode` silently overwrites the operation mode (caused the eject bug, fixed via `eject_mode`). No other call passes `mode`. Also: N+1 crop fetching (one proxy call per face, base64 through Stash) — fine at current scale, could be batched |
| `stash-sense-nav.js` | Adds a "Stash Sense" `<li><a>` to Stash's main `.navbar-nav`; SPA-aware (pushState + popstate), active-state sync via `SS.onNavigate`. | selector fragility across Stash versions; duplicate-injection guard |
| `stash_sense_backend.py` | Plugin's Python exec backend. New: `handle_face_groups` (`fg_*` modes), `sidecar_patch`, crop proxy returns base64 data URL, `fg_create_and_assign`. | error passthrough shape (`{"error": ...}`), crop base64 memory for 200-face details |
| `stash-sense.yml`, `stash-sense.css` | Module registration (load order: core → recs → settings → ops → face-groups → nav → stash-sense.js) + tab/thumb/detail styles. | ⚠ deploy gotcha: Stash only serves files listed in the plugin's `manifest` (written at install time) — any new JS file must be appended there |

### Tests

- `api/tests/test_face_clusters.py` — 25 tests: DB layer (dedup, cascade, membership,
  listing counts), clustering engine (3-people synthetic separation, identical-collapse,
  empty), service (seed-by-match, assign read-then-write incl. already-tagged skip,
  merge + merge-log chain, incremental absorb into assigned groups, auto-tag, banned
  exclusion), schema v11.
- Two pre-existing tests updated: schema-version assertions (9→10→11) and the job
  registry expected-set.
- Full suite: **1,314 passed** (`pytest -m "not heavy"`; heavy/ML tests need GPU/ONNX).

## Known limitations / deliberate trade-offs (already accepted, listed for honesty)

1. **Sampling**: only faces seen during identify enter the corpus (60 frames @ 5–95%,
   ≥40 px, confidence ≥0.5). Rare/blurry faces never cluster. Re-identify refreshes a scene.
2. **Storage**: ~4 KB embeddings + ~5–15 KB crop per face → est. 2–6 GB for a 6.7k-scene
   library. Lives on the NVMe pool; growth documented.
3. **Sync bulk tagging**: assign/auto-tag loops scenes synchronously without the async
   rate limiter. Local-box acceptable; not great library-hygiene.
4. **`matched` groups are re-derived each full rebuild** (from stored per-face best-match
   anchors); assigned/ignored/banned survive rebuilds. Merge log redirects are honored
   via performer-anchor routing, not by cluster id (matched groups get fresh ids).
5. **N+1 crop fetches** in the plugin (base64 through Stash's plugin proxy).
6. No pagination on the banned list beyond limit param.

## Deployment context (why it looks the way it does)

- Runs on an Unraid box (Tower) with **no NVIDIA GPU** — CPU ONNX inference only; the
  CUDA base image falls back at runtime. This is why identify/fingerprint is slow and
  why the fingerprint job runs for days.
- The sidecar's `/data` gets hot-swapped by upstream database releases — that's why the
  fork's own state (faces, clusters) lives in `stash_sense.db` + `library_faces/`, which
  the updater never touches.
- voyager has no Python 3.13 wheel (dev box) — brute-force fallback covers local tests;
  the container (3.11) uses HNSW.

## Suggested review pass

1. `git diff origin/main...face-clusters -- api/recommendations_db.py` — schema/migration.
2. `api/library_face_persist.py` + the call site in `identification_router.py` — the
   face→person mapping and failure isolation.
3. `api/face_cluster_service.py` — full vs incremental build semantics, auto-tag.
4. `api/face_clusters_router.py` — route order, create-and-assign partial failure.
5. `plugin/stash-sense-face-groups.js` — the `{mode, ...params}` spread trap and its
   remaining call sites.
6. Run: `cd api && python -m pytest tests/test_face_clusters.py -v` (or the full suite).