/**
 * Stash Sense Nav Module
 * Adds a "Stash Sense" link to Stash's main navigation bar so the plugin
 * dashboard is reachable without a bookmark.
 */
(function() {
  'use strict';

  const SS = window.StashSense;
  if (!SS) {
    console.error('[Stash Sense] Core module not loaded');
    return;
  }

  let injected = false;

  function findNavbar() {
    // Stash's main nav: a .nav with .nav-link items (Scenes, Images, ...)
    const candidates = document.querySelectorAll('.navbar-nav, nav .nav, .main-nav');
    for (const nav of candidates) {
      if (nav.querySelector('a.nav-link')) return nav;
    }
    return null;
  }

  function markActive(nav) {
    const onPluginPage = SS.getRoute().type === 'plugin';
    const link = nav.querySelector('a[data-ss-navlink]');
    if (!link) return;
    link.classList.toggle('active', onPluginPage);
  }

  function injectNav() {
    if (injected) return;
    const nav = findNavbar();
    if (!nav) return;

    if (nav.querySelector('[data-ss-navlink]')) { injected = true; return; }

    const li = document.createElement('li');
    li.className = 'nav-item';

    const a = document.createElement('a');
    a.className = 'nav-link';
    a.setAttribute('data-ss-navlink', 'true');
    a.href = '/plugins/stash-sense';
    a.textContent = 'Stash Sense';
    a.addEventListener('click', (e) => {
      e.preventDefault();
      // Stash is an SPA; full navigation reloads the app which is acceptable,
      // but prefer history + popstate so the SPA router picks it up.
      if (window.location.pathname !== '/plugins/stash-sense') {
        window.history.pushState({}, '', '/plugins/stash-sense');
        window.dispatchEvent(new PopStateEvent('popstate'));
      }
    });

    li.appendChild(a);
    nav.appendChild(li);
    injected = true;

    // Keep active state in sync with SPA navigation
    SS.onNavigate(() => markActive(nav));
    markActive(nav);
  }

  function init() {
    // Nav may not exist yet at plugin load; retry briefly
    let tries = 0;
    const timer = setInterval(() => {
      tries += 1;
      injectNav();
      if (injected || tries > 20) clearInterval(timer);
    }, 500);
  }

  window.StashSenseNav = { init };

  init();
})();
