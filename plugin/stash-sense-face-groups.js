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
    async build(opts = {}) { return apiCall('fg_build', opts); },
    async stats() { return apiCall('fg_stats'); },
    async assign(clusterId, performerId, performerName) {
      return apiCall('fg_assign', { cluster_id: clusterId, performer_id: performerId, performer_name: performerName });
    },
    async ignore(clusterId) { return apiCall('fg_ignore', { cluster_id: clusterId }); },
    async remove(clusterId) { return apiCall('fg_delete', { cluster_id: clusterId }); },
    async merge(sourceIds, targetId) { return apiCall('fg_merge', { source_ids: sourceIds, target_id: targetId }); },
    async rename(clusterId, name) { return apiCall('fg_update', { cluster_id: clusterId, name }); },
    async crop(clusterId, faceId) { return apiCall('fg_crop', { cluster_id: clusterId, face_id: faceId }); },
    async searchPerformers(query) { return apiCall('search_performers', { query }); },
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
            : `<button class="ss-btn ss-btn-secondary" id="fg-select-toggle">Select / Merge</button>`}
          <button class="ss-btn ss-btn-primary" id="fg-rebuild">Rebuild Groups</button>
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
      cb.addEventListener('change', () => {
        const id = parseInt(cb.dataset.clusterId, 10);
        if (cb.checked) selectedClusters.add(id); else selectedClusters.delete(id);
        renderList(container);
      });
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

  // ==================== Assign Modal ====================

  function openAssignModal(clusterId, container) {
    const overlay = SS.createElement('div', { className: 'ss-modal-overlay' });
    overlay.innerHTML = `
      <div class="ss-modal-content" style="max-width:520px;">
        <div class="ss-modal-header"><h3>Assign performer to group #${clusterId}</h3>
          <button class="ss-modal-close">×</button></div>
        <div class="ss-modal-body">
          <p style="opacity:.8;">All scenes containing faces from this group will be tagged with the chosen performer (scenes keep their existing performers).</p>
          <input type="text" class="ss-input" id="fg-performer-search" placeholder="Search Stash performers…" style="width:100%;padding:8px;margin-bottom:8px;" />
          <div id="fg-performer-results" style="max-height:300px;overflow-y:auto;"></div>
        </div>
      </div>`;
    document.body.appendChild(overlay);

    const close = () => overlay.remove();
    overlay.querySelector('.ss-modal-close').addEventListener('click', close);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

    const searchInput = overlay.querySelector('#fg-performer-search');
    const results = overlay.querySelector('#fg-performer-results');
    let debounce;

    function renderResults(list) {
      if (!list.length) {
        results.innerHTML = '<p style="opacity:.6;padding:8px;">No performers found — create one in Stash first.</p>';
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
        renderResults(r.performers || []);
      } catch (e) {
        results.innerHTML = `<p style="opacity:.6;">${esc(e.message)}</p>`;
      }
    }

    searchInput.addEventListener('input', () => {
      clearTimeout(debounce);
      debounce = setTimeout(doSearch, 350);
    });
    searchInput.focus();
  }

  // ==================== Rebuild ====================

  async function rebuildGroups(container) {
    if (!confirm('Rebuild face groups?\n\nOpen (unnamed) groups are re-clustered from all stored faces. Assigned and ignored groups are kept.')) return;
    rebuildGroupsInProgress(container);
    try {
      const r = await FaceGroupsAPI.build({ replace_existing: true });
      alert(`Built ${r.clusters_created} groups covering ${r.faces_assigned} of ${r.faces_total} faces.`);
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
})();
