/**
 * Stash Sense Face Groups Module
 * Immich-style face grouping UI: browse unnamed face clusters, merge them,
 * and assign them to Stash performers (bulk-tags all source scenes).
 */
(function() {
  'use strict';

  const SS = window.StashSense;
  if (!SS) {
    console.error('[Stash Sense] Core module not loaded');
    return;
  }

  // ==================== API ====================

  async function apiCall(mode, params = {}) {
    const settings = await SS.getSettings();
    const result = await SS.runPluginOperation(mode, {
      sidecar_url: settings.sidecarUrl,
      ...params,
    });
    if (result && result.error) {
      throw new Error(result.error);
    }
    return result;
  }

  const FaceGroupsAPI = {
    async list(status) { return apiCall('fg_list', status ? { status } : {}); },
    async get(clusterId) { return apiCall('fg_get', { cluster_id: clusterId }); },
    async build(opts = {}) { return apiCall('fg_build', { incremental: true, auto_tag: true, replace_existing: true, ...opts }); },
    async stats() { return apiCall('fg_stats'); },
    async assign(clusterId, performerId, performerName) {
      return apiCall('fg_assign', { cluster_id: clusterId, performer_id: performerId, performer_name: performerName });
    },
    async ignore(clusterId) { return apiCall('fg_ignore', { cluster_id: clusterId }); },
    async remove(clusterId) { return apiCall('fg_delete', { cluster_id: clusterId }); },
    async merge(sourceIds, targetId) { return apiCall('fg_merge', { source_ids: sourceIds, target_id: targetId }); },
    async rename(clusterId, name) { return apiCall('fg_update', { cluster_id: clusterId, name }); },
    async crop(clusterId, faceId) { return apiCall('fg_crop', { cluster_id: clusterId, face_id: faceId }); },
    // NB: never pass a key named `mode` in params — runPluginOperation builds
    // {mode, ...params} and the spread would overwrite the operation mode.
    async eject(clusterId, faceIds, ejectMode) { return apiCall('fg_eject', { cluster_id: clusterId, face_ids: faceIds, eject_mode: ejectMode }); },
    async unban(faceIds) { return apiCall('fg_unban', { face_ids: faceIds }); },
    async banned() { return apiCall('fg_banned'); },
    async searchPerformers(query) { return apiCall('search_performers', { query }); },
    async createAndAssign(clusterId, name, opts = {}) {
      return apiCall('fg_create_and_assign', { cluster_id: clusterId, name, ...opts });
    },
  };

  // ==================== State ====================

  let pollInterval = null;
  let selectionMode = false;
  let selectedClusters = new Set();
  let currentFilter = null; // null = all

  // ==================== Helpers ====================

  function esc(s) { return SS.escapeHtml(String(s == null ? '' : s)); }

  function statusBadge(status) {
    const map = {
      open: ['badge-warning', 'Unnamed'],
      matched: ['badge', 'Matched'],
      assigned: ['badge-success', 'Assigned'],
      ignored: ['badge', 'Ignored'],
    };
    const [cls, label] = map[status] || ['badge', status];
    return `<span class="ss-badge ${cls}">${esc(label)}</span>`;
  }

  async function faceImg(clusterId, face) {
    try {
      const r = await FaceGroupsAPI.crop(clusterId, face.id);
      return r.data_url || '';
    } catch (e) {
      return '';
    }
  }

  // ==================== Cluster List ====================

  async function renderList(container) {
    container.innerHTML = '<div class="ss-loading"><div class="ss-loading-text">Loading face groups…</div></div>';
    let data;
    try {
      data = await FaceGroupsAPI.list(currentFilter);
    } catch (e) {
      container.innerHTML = `<div class="ss-empty-state"><p>Failed to load face groups: ${esc(e.message)}</p></div>`;
      return;
    }

    const { clusters, stats } = data;

    const header = `
      <div class="ss-actions" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <h2 style="margin:0;">Face Groups</h2>
          <span style="opacity:.7;">${stats.faces_total} faces in ${stats.clusters} groups — ${stats.open} unnamed, ${stats.assigned} assigned, ${stats.ignored} ignored</span>
        </div>
        <div style="display:flex;gap:8px;">
          ${selectionMode
            ? `<button class="ss-btn ss-btn-primary" id="fg-merge-btn" ${selectedClusters.size < 2 ? 'disabled' : ''}>Merge (${selectedClusters.size})</button>
               <button class="ss-btn ss-btn-secondary" id="fg-cancel-select">Cancel</button>`
            : `<button class="ss-btn ss-btn-secondary" id="fg-select-toggle">Select / Merge</button>
               <button class="ss-btn ss-btn-secondary" id="fg-banned-btn">Banned faces</button>`}
          <button class="ss-btn ss-btn-primary" id="fg-rebuild">Update Groups</button>
        </div>
      </div>`;

    if (!clusters.length) {
      container.innerHTML = `
        ${header}
        <div class="ss-empty-state">
          <p><strong>No face groups yet.</strong></p>
          <p>Groups are built from faces detected during identification. Identify some scenes (or run Fingerprint Generation in Operations), then click <strong>Rebuild Groups</strong>.</p>
        </div>`;
      wireHeader(container);
      return;
    }

    const filterBar = `
      <div style="display:flex;gap:6px;margin:12px 0;">
        ${['', 'open', 'matched', 'assigned', 'ignored'].map(f => {
          const label = f === '' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1);
          const active = (currentFilter || '') === f;
          return `<button class="ss-btn ss-btn-sm ${active ? 'ss-btn-primary' : 'ss-btn-secondary'}" data-fg-filter="${f}">${label}</button>`;
        }).join('')}
      </div>`;

    const cards = clusters.map(c => `
      <div class="ss-performer-option" data-cluster-id="${c.id}" style="${selectionMode && selectedClusters.has(c.id) ? 'border-color: var(--ss-accent, #7b5cff);' : ''}">
        <div class="ss-fg-card-header">
          ${selectionMode ? `<input type="checkbox" class="fg-select" data-cluster-id="${c.id}" ${selectedClusters.has(c.id) ? 'checked' : ''} />` : ''}
          <div>
            <div class="ss-performer-name">${c.name ? esc(c.name) : `Group #${c.id}`}</div>
            <div>${statusBadge(c.status)} <span class="ss-performer-count">${c.face_count} faces · ${c.scene_count} scenes</span></div>
          </div>
        </div>
        <div class="ss-fg-thumbs" data-cluster-id="${c.id}"><div class="ss-loading-inline">…</div></div>
        ${c.status !== 'assigned' ? `
        <div style="display:flex;gap:6px;padding:0 12px 12px;">
          <button class="ss-btn ss-btn-sm ss-btn-primary fg-assign" data-cluster-id="${c.id}">Assign performer…</button>
          <button class="ss-btn ss-btn-sm ss-btn-secondary fg-ignore" data-cluster-id="${c.id}">Ignore</button>
        </div>` : `
        <div style="padding:0 12px 12px;">
          <span style="opacity:.8;">→ ${esc(c.performer_name || c.performer_id)}</span>
        </div>`}
      </div>`).join('');

    container.innerHTML = `
      ${header}
      ${filterBar}
      <div class="ss-performer-grid" style="grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));">${cards}</div>`;

    wireHeader(container);
    wireCards(container);
    loadThumbs(container);
  }

  function wireHeader(container) {
    const rebuild = container.querySelector('#fg-rebuild');
    if (rebuild) rebuild.addEventListener('click', () => rebuildGroups(container));

    const selectToggle = container.querySelector('#fg-select-toggle');
    if (selectToggle) selectToggle.addEventListener('click', () => {
      selectionMode = true;
      renderList(container);
    });

    const cancel = container.querySelector('#fg-cancel-select');
    if (cancel) cancel.addEventListener('click', () => {
      selectionMode = false;
      selectedClusters.clear();
      renderList(container);
    });

    const bannedBtn = container.querySelector('#fg-banned-btn');
    if (bannedBtn) bannedBtn.addEventListener('click', () => renderBannedList(container));

    const mergeBtn = container.querySelector('#fg-merge-btn');
    if (mergeBtn) mergeBtn.addEventListener('click', () => {
      if (selectedClusters.size < 2) return;
      const ids = [...selectedClusters];
      const targetId = ids[0]; // largest is first in list order
      if (!confirm(`Merge ${ids.length - 1} group(s) into group #${targetId}?`)) return;
      FaceGroupsAPI.merge(ids.filter(i => i !== targetId), targetId)
        .then(() => {
          selectedClusters.clear();
          selectionMode = false;
          renderList(container);
        })
        .catch(e => alert(`Merge failed: ${e.message}`));
    });

    container.querySelectorAll('[data-fg-filter]').forEach(btn => {
      btn.addEventListener('click', () => {
        currentFilter = btn.dataset.fgFilter || null;
        renderList(container);
      });
    });
  }

  function wireCards(container) {
    container.querySelectorAll('.fg-select').forEach(cb => {
      cb.addEventListener('click', (e) => e.stopPropagation());
      cb.addEventListener('change', () => {
        const id = parseInt(cb.dataset.clusterId, 10);
        if (cb.checked) selectedClusters.add(id); else selectedClusters.delete(id);
        renderList(container);
      });
    });

    // clicking a card (outside buttons) opens the detail/expand view
    container.querySelectorAll('.ss-performer-option[data-cluster-id]').forEach(card => {
      card.addEventListener('click', (e) => {
        if (e.target.closest('button') || e.target.closest('input') || selectionMode) return;
        openGroupDetail(parseInt(card.dataset.clusterId, 10), container);
      });
      card.style.cursor = 'pointer';
    });

    container.querySelectorAll('.fg-ignore').forEach(btn => {
      btn.addEventListener('click', () => {
        const id = parseInt(btn.dataset.clusterId, 10);
        if (!confirm('Ignore this group? Its faces will not be suggested again.')) return;
        FaceGroupsAPI.ignore(id).then(() => renderList(container)).catch(e => alert(e.message));
      });
    });

    container.querySelectorAll('.fg-assign').forEach(btn => {
      btn.addEventListener('click', () => openAssignModal(parseInt(btn.dataset.clusterId, 10), container));
    });
  }

  function loadThumbs(container) {
    container.querySelectorAll('.ss-fg-thumbs').forEach(async el => {
      const clusterId = parseInt(el.dataset.clusterId, 10);
      try {
        const detail = await FaceGroupsAPI.get(clusterId);
        const faces = (detail.faces || []).slice(0, 8);
        el.innerHTML = '';
        for (const f of faces) {
          const img = document.createElement('img');
          img.className = 'ss-fg-thumb';
          img.alt = 'face';
          img.loading = 'lazy';
          try {
            const r = await FaceGroupsAPI.crop(clusterId, f.id);
            if (r.data_url) img.src = r.data_url;
          } catch (e) { /* skip */ }
          el.appendChild(img);
        }
        if (!faces.length) el.innerHTML = '<span style="opacity:.5;">No face images stored</span>';
      } catch (e) {
        el.innerHTML = `<span style="opacity:.5;">${esc(e.message)}</span>`;
      }
    });
  }

  // ==================== Group Detail (expand) ====================

  let ejectMode = null; // null | 'eject' | 'ban'
  let ejectedFaces = new Set();

  function renderBannedList(container) {
    const panel = document.getElementById('ss-face-groups') || container;
    panel.innerHTML = '<div class="ss-loading"><div class="ss-loading-text">Loading banned faces…</div></div>';
    FaceGroupsAPI.banned().then(({ faces }) => {
      const header = `
        <div class="ss-actions" style="display:flex;justify-content:space-between;align-items:center;">
          <h2 style="margin:0;">Banned faces (${faces.length})</h2>
          <button class="ss-btn ss-btn-sm ss-btn-secondary" id="fg-back">← All groups</button>
        </div>
        <p style="opacity:.65;">These are excluded from all clustering. Select faces to restore them to the unassigned pool.</p>`;
      if (!faces.length) {
        panel.innerHTML = `${header}<div class="ss-empty-state"><p>No banned faces.</p></div>`;
      } else {
        panel.innerHTML = `${header}<div class="ss-fg-detail-grid">${faces.map(f => `
          <div class="ss-fg-thumb-wrap" style="position:relative;">
            <img class="ss-fg-thumb ss-fg-unban" data-face-id="${f.id}" data-cluster-id="${f.cluster_id}" alt="face" style="cursor:pointer;" />
          </div>`).join('')}</div>`;
        panel.querySelectorAll('.ss-fg-unban').forEach(img => {
          FaceGroupsAPI.crop(parseInt(img.dataset.clusterId, 10), parseInt(img.dataset.faceId, 10))
            .then(r => { if (r.data_url) img.src = r.data_url; }).catch(() => {});
        });
      }
      panel.querySelector('#fg-back').addEventListener('click', () => renderList(container));
    }).catch(e => {
      panel.innerHTML = `<div class="ss-empty-state"><p>${esc(e.message)}</p></div>`;
    });
  }

  function openGroupDetail(clusterId, container) {
    const dashboard = document.getElementById('ss-recommendations');
    const panel = document.getElementById('ss-face-groups') || dashboard;
    panel.innerHTML = '<div class="ss-loading"><div class="ss-loading-text">Loading group…</div></div>';

    FaceGroupsAPI.get(clusterId).then(cluster => {
      renderGroupDetail(cluster, panel, container);
    }).catch(e => {
      panel.innerHTML = `<div class="ss-empty-state"><p>${esc(e.message)}</p></div>`;
    });
  }

  function renderGroupDetail(cluster, panel, container) {
    const c = cluster.id;
    const header = `
      <div class="ss-actions" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
        <div style="display:flex;gap:8px;align-items:center;">
          <button class="ss-btn ss-btn-sm ss-btn-secondary" id="fg-back">← All groups</button>
          <h2 style="margin:0;">${cluster.name ? esc(cluster.name) : `Group #${c}`}</h2>
          ${statusBadge(cluster.status)}
          ${cluster.performer_name ? `<span style="opacity:.8;">→ ${esc(cluster.performer_name)}</span>` : ''}
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          ${ejectMode === null ? `
            <span style="opacity:.7;">Click faces to select</span>
            <button class="ss-btn ss-btn-sm ss-btn-secondary" id="fg-eject-mode">Remove faces</button>
          ` : `
            <span style="opacity:.7;">${ejectedFaces.size} selected</span>
            <button class="ss-btn ss-btn-sm ss-btn-primary" id="fg-eject-confirm" ${ejectedFaces.size ? '' : 'disabled'}>
              ${ejectMode === 'ban' ? 'Ban selected' : 'Remove selected'}
            </button>
            <button class="ss-btn ss-btn-sm ss-btn-secondary" id="fg-eject-cancel">Cancel</button>
          `}
        </div>
      </div>
      <p style="opacity:.65;margin:0 0 12px;">${cluster.face_count} faces across ${cluster.scene_ids ? cluster.scene_ids.length : '?'} scenes.
      ${ejectMode ? (ejectMode === 'ban'
        ? 'Banned faces are junk detections — they will never be clustered again (reviewable via Banned list).'
        : 'Removed faces return to the unassigned pool and may re-group on the next update.') : ''}
      </p>`;

    const faces = cluster.faces || [];
    const thumbGrid = faces.map(f => `
      <div class="ss-fg-thumb-wrap" data-face-id="${f.id}" style="position:relative;">
        <img class="ss-fg-thumb ${ejectedFaces.has(f.id) ? 'ss-fg-selected' : ''}" data-face-id="${f.id}"
             alt="face" style="cursor:pointer; ${ejectedFaces.has(f.id) ? 'outline:3px solid #e04848;outline-offset:-3px;' : ''}" />
        <div style="font-size:.65em;opacity:.6;text-align:center;">scene ${f.stash_scene_id}</div>
      </div>`).join('');

    panel.innerHTML = `
      ${header}
      ${cluster.top_matches && cluster.top_matches.length && cluster.status !== 'assigned' ? `
        <div style="margin-bottom:12px;font-size:.9em;opacity:.75;">
          Identify-time matches: ${cluster.top_matches.map(m => `${esc(m.name)} (${m.face_count})`).join(', ')}
        </div>` : ''}
      <div class="ss-fg-detail-grid">${thumbGrid || '<span style="opacity:.5;">No stored face images.</span>'}</div>`;

    // back
    panel.querySelector('#fg-back').addEventListener('click', () => {
      ejectMode = null; ejectedFaces.clear();
      renderList(container);
    });

    // eject mode toggle
    const modeBtn = panel.querySelector('#fg-eject-mode');
    if (modeBtn) modeBtn.addEventListener('click', () => {
      // two-step: first click arms "remove", a small toggle lets you choose ban
      ejectMode = 'eject';
      renderGroupDetail(cluster, panel, container);
      // offer ban via a confirm-time choice
    });

    const cancelBtn = panel.querySelector('#fg-eject-cancel');
    if (cancelBtn) cancelBtn.addEventListener('click', () => {
      ejectMode = null; ejectedFaces.clear();
      renderGroupDetail(cluster, panel, container);
    });

    const confirmBtn = panel.querySelector('#fg-eject-confirm');
    if (confirmBtn) confirmBtn.addEventListener('click', async () => {
      const ids = [...ejectedFaces];
      if (!ids.length) return;
      let mode = ejectMode;
      if (mode === 'eject') {
        const ban = confirm('Also BAN these faces from ever being clustered again (for junk detections)?\n\nOK = ban permanently, Cancel = just remove from group');
        mode = ban ? 'ban' : 'eject';
      }
      try {
        const r = await FaceGroupsAPI.eject(c, ids, mode);
        alert(mode === 'ban' ? `Banned ${r.banned} face(s).` : `Removed ${r.ejected} face(s).`);
        ejectMode = null; ejectedFaces.clear();
        renderList(container);
      } catch (e) {
        alert(`Failed: ${e.message}`);
      }
    });

    // face click = select (only in eject mode) else open scene
    panel.querySelectorAll('.ss-fg-thumb').forEach(img => {
      img.addEventListener('click', async () => {
        const fid = parseInt(img.dataset.faceId, 10);
        if (ejectMode) {
          if (ejectedFaces.has(fid)) ejectedFaces.delete(fid); else ejectedFaces.add(fid);
          img.classList.toggle('ss-fg-selected');
          img.style.outline = ejectedFaces.has(fid) ? '3px solid #e04848' : '';
          img.style.outlineOffset = '-3px';
          const span = panel.querySelector('#fg-eject-confirm');
          if (span) {
            span.disabled = !ejectedFaces.size;
            span.textContent = `${ejectMode === 'ban' ? 'Ban' : 'Remove'} selected (${ejectedFaces.size})`;
          }
          const selCount = panel.querySelector('.ss-actions span[style*="opacity"]');
          return;
        }
        // default: jump to the scene in Stash
        const face = faces.find(f => f.id === fid);
        if (face) window.location.href = `/scenes/${face.stash_scene_id}`;
      });
    });

    // load crops
    panel.querySelectorAll('.ss-fg-thumb').forEach(img => {
      const fid = parseInt(img.dataset.faceId, 10);
      FaceGroupsAPI.crop(c, fid).then(r => { if (r.data_url) img.src = r.data_url; }).catch(() => {});
    });
  }

  // ==================== Assign Modal ====================

  function openAssignModal(clusterId, container) {
    const overlay = SS.createElement('div', { className: 'ss-modal-overlay' });
    overlay.innerHTML = `
      <div class="ss-modal-content" style="max-width:520px;">
        <div class="ss-modal-header"><h3>Assign performer to group #${clusterId}</h3>
          <button class="ss-modal-close">×</button></div>
        <div class="ss-modal-body">
          <p style="opacity:.8;">All scenes containing faces from this group will be tagged with the chosen performer (scenes keep their existing performers).</p>
          <input type="text" class="ss-input" id="fg-performer-search" placeholder="Search existing performers, or type a new name…" style="width:100%;padding:8px;margin-bottom:8px;" />
          <div style="display:flex;gap:8px;margin-bottom:8px;">
            <button class="ss-btn ss-btn-sm ss-btn-primary" id="fg-create-new" disabled>Create new performer</button>
            <label style="display:flex;align-items:center;gap:4px;font-size:.9em;opacity:.85;">
              <input type="checkbox" id="fg-new-favorite" /> Favorite
            </label>
            <input type="text" class="ss-input" id="fg-new-disambig" placeholder="Disambiguation (optional)" style="flex:1;padding:4px 8px;display:none;" />
          </div>
          <div id="fg-performer-results" style="max-height:300px;overflow-y:auto;"></div>
        </div>
      </div>`;
    document.body.appendChild(overlay);

    const close = () => overlay.remove();
    overlay.querySelector('.ss-modal-close').addEventListener('click', close);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

    const searchInput = overlay.querySelector('#fg-performer-search');
    const results = overlay.querySelector('#fg-performer-results');
    const createBtn = overlay.querySelector('#fg-create-new');
    const favCheck = overlay.querySelector('#fg-new-favorite');
    const disambigInput = overlay.querySelector('#fg-new-disambig');
    let debounce;

    function wireCreate() {
      if (!createBtn) return;
      createBtn.disabled = searchInput.value.trim().length === 0;
      createBtn.onclick = async () => {
        const name = searchInput.value.trim();
        if (!name) return;
        const orig = createBtn.textContent;
        createBtn.disabled = true;
        createBtn.textContent = 'Creating & tagging…';
        try {
          const r = await FaceGroupsAPI.createAndAssign(clusterId, name, {
            favorite: favCheck && favCheck.checked,
            disambiguation: disambigInput && disambigInput.value.trim() || undefined,
          });
          close();
          alert(`Created performer "${r.created_performer.name}" — tagged ${r.scenes_tagged} scene(s)` +
                (r.scenes_already_tagged ? `, ${r.scenes_already_tagged} already tagged` : '') +
                (r.scenes_failed && r.scenes_failed.length ? `, ${r.scenes_failed.length} FAILED` : ''));
          renderList(container);
        } catch (e) {
          alert(`Create failed: ${e.message}`);
          createBtn.disabled = false;
          createBtn.textContent = orig;
        }
      };
    }

    function renderResults(list) {
      if (!list.length) {
        results.innerHTML = '<p style="opacity:.6;padding:8px;">No existing performer matches — use "Create new performer" above.</p>';
        return;
      }
      results.innerHTML = list.map(p => `
        <div class="ss-performer-option fg-performer-pick" data-performer-id="${esc(p.id)}" data-performer-name="${esc(p.name)}"
             style="display:flex;gap:10px;align-items:center;padding:8px;cursor:pointer;margin-bottom:4px;">
          ${p.image_path ? `<img src="${esc(p.image_path)}" style="width:40px;height:40px;border-radius:50%;object-fit:cover;" />` : ''}
          <span class="ss-performer-name">${esc(p.name)}</span>
        </div>`).join('');
      results.querySelectorAll('.fg-performer-pick').forEach(el => {
        el.addEventListener('click', async () => {
          const pid = el.dataset.performerId;
          const pname = el.dataset.performerName;
          el.innerHTML = '<span style="opacity:.7;">Tagging scenes…</span>';
          try {
            const r = await FaceGroupsAPI.assign(clusterId, pid, pname);
            close();
            alert(`Assigned "${pname}" — tagged ${r.scenes_tagged} scene(s)` +
                  (r.scenes_already_tagged ? `, ${r.scenes_already_tagged} already tagged` : '') +
                  (r.scenes_failed && r.scenes_failed.length ? `, ${r.scenes_failed.length} FAILED` : ''));
            renderList(container);
          } catch (e) {
            alert(`Assign failed: ${e.message}`);
          }
        });
      });
    }

    async function doSearch() {
      const q = searchInput.value.trim();
      if (q.length < 2) { results.innerHTML = '<p style="opacity:.6;padding:8px;">Type at least 2 characters…</p>'; return; }
      results.innerHTML = '<div class="ss-loading-inline">Searching…</div>';
      try {
        const r = await FaceGroupsAPI.searchPerformers(q);
        // /stash/search-performers returns a bare array; older shapes may wrap it
        const list = Array.isArray(r) ? r : (r.performers || []);
        renderResults(list);
      } catch (e) {
        results.innerHTML = `<p style="opacity:.6;">${esc(e.message)}</p>`;
      }
    }

    searchInput.addEventListener('input', () => {
      clearTimeout(debounce);
      debounce = setTimeout(doSearch, 350);
      // Create-new affordance: any non-empty input is a candidate new performer
      if (createBtn) {
        createBtn.style.display = '';
        wireCreate();
      }
    });
    wireCreate();
    searchInput.focus();
  }

  // ==================== Rebuild ====================

  async function rebuildGroups(container) {
    if (!confirm('Update face groups?\n\nIncremental: new faces are matched against existing groups (assigned, matched, and open) by similarity and absorbed when close enough — scenes of faces absorbed into assigned groups are auto-tagged with that group\'s performer. Leftover faces form new open groups.')) return;
    rebuildGroupsInProgress(container);
    try {
      const r = await FaceGroupsAPI.build({ incremental: true, auto_tag: true, replace_existing: true });
      const parts = [`absorbed ${r.absorbed} new face(s) into ${r.groups_absorbed_into} existing group(s)`];
      if (r.tagged_scenes) parts.push(`auto-tagged ${r.tagged_scenes} scene(s)`);
      if (r.clusters_created) parts.push(`created ${r.clusters_created} new group(s)`);
      alert(`Face groups updated: ${parts.join('; ')}.`);
    } catch (e) {
      alert(`Rebuild failed: ${e.message}`);
    }
    renderList(container);
  }

  function rebuildGroupsInProgress(container) {
    container.innerHTML = `
      <div class="ss-loading">
        <div class="ss-loading-text">Clustering library faces… this can take a while for large libraries.</div>
      </div>`;
  }

  // ==================== Tab Injection ====================

  function createFaceGroupsContainer() {
    const el = SS.createElement('div', {
      id: 'ss-face-groups',
      className: 'ss-page-panel',
      attrs: { 'data-panel': 'face-groups' },
    });
    return el;
  }

  function renderFaceGroups(panel) {
    renderList(panel);
  }

  function injectFaceGroupsTab() {
    const route = SS.getRoute();
    if (route.type !== 'plugin') return;

    const dashboard = document.getElementById('ss-recommendations');
    if (!dashboard) return;

    const tabBar = dashboard.querySelector('.ss-page-tabs');
    if (!tabBar) return;

    if (document.getElementById('ss-face-groups')) return;
    if (tabBar.querySelector('[data-tab="face-groups"]')) return;

    const initialTab = SS.getTabFromUrl();

    const tab = SS.createElement('button', {
      className: `ss-page-tab ${initialTab === 'face-groups' ? 'active' : ''}`,
      textContent: 'Face Groups',
      attrs: { 'data-tab': 'face-groups' },
    });

    const settingsTabBtn = tabBar.querySelector('[data-tab="settings"]');
    if (settingsTabBtn) {
      tabBar.insertBefore(tab, settingsTabBtn);
    } else {
      tabBar.appendChild(tab);
    }

    const panel = createFaceGroupsContainer();
    panel.style.display = initialTab === 'face-groups' ? '' : 'none';

    const settingsPanel = document.getElementById('ss-settings');
    if (settingsPanel) {
      settingsPanel.parentElement.insertBefore(panel, settingsPanel);
    } else {
      dashboard.appendChild(panel);
    }

    if (initialTab === 'face-groups') {
      tabBar.querySelectorAll('.ss-page-tab').forEach(t => { if (t !== tab) t.classList.remove('active'); });
      dashboard.querySelectorAll('.ss-page-panel').forEach(p => { if (p !== panel) p.style.display = 'none'; });
      panel.dataset.loaded = 'true';
      renderFaceGroups(panel);
    }

    tabBar.addEventListener('click', (e) => {
      const btn = e.target.closest('.ss-page-tab');
      if (!btn) return;
      const tabName = btn.dataset.tab;
      SS.setTabInUrl(tabName);
      panel.style.display = tabName === 'face-groups' ? '' : 'none';
      if (tabName === 'face-groups' && !panel.dataset.loaded) {
        panel.dataset.loaded = 'true';
        renderFaceGroups(panel);
      }
      if (tabName !== 'face-groups') {
        // stop crop fetch storms when navigating away: nothing persistent to stop (loads are bounded)
      }
    });
  }

  // ==================== Init ====================

  function cleanup() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
    const el = document.getElementById('ss-face-groups');
    if (el) el.remove();
    selectionMode = false;
    selectedClusters.clear();
  }

  function init() {
    const tryInject = () => {
      if (SS.getRoute().type === 'plugin') {
        setTimeout(injectFaceGroupsTab, 900); // after settings (600) and operations (800)
      }
    };
    tryInject();
    SS.onNavigate((route) => {
      if (route.type === 'plugin') setTimeout(injectFaceGroupsTab, 900);
    });
    SS.onLeavePlugin(cleanup);
    console.log(`[${SS.PLUGIN_NAME}] Face Groups module loaded`);
  }

  window.StashSenseFaceGroups = { init };

  init();
})();
