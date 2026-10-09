// Shared models picker for RealityStream pages.
//
// The model list comes from the checkbox headings in realitystream/models/README.md, e.g.
//   ## <input type="checkbox" id="model-lr" name="model" value="lr"> Logistic Regression (lr)
// so adding a model there adds it everywhere: models/index.html (which shows the README itself),
// cloud/run/page.html and the Cloud Run home page (which render a compact list into #modelSelect).
//
// Every input[name="model"] checkbox on the page stays in sync with the URL hash (#models=lr,rfc),
// which param-input.js copies into the YAML's models: line. Without models in the hash, the
// checkboxes show the models already in the YAML (#paramText pre).
//
// Needs getHash() and goHash() from localsite/js/localsite.js.

(function () {
  // README.md resides in ../models/ relative to this script, wherever the script is hosted
  const README_URL = (function () {
    try {
      return new URL('../models/README.md', document.currentScript.src).href;
    } catch (e) {
      return '/realitystream/models/README.md';
    }
  })();

  let modelsPromise = null;

  // [{value, label}] parsed from the README checkbox headings
  function loadModels() {
    if (!modelsPromise) {
      modelsPromise = fetch(README_URL)
        .then(r => r.text())
        .then(text => {
          const models = [];
          const re = /^#+\s*<input[^>]*name="model"[^>]*value="([^"]+)"[^>]*>\s*(.+)$/gm;
          let m;
          while ((m = re.exec(text)) !== null) {
            models.push({ value: m[1].trim(), label: m[2].trim() });
          }
          return models;
        });
    }
    return modelsPromise;
  }

  function checkboxes() {
    return Array.from(document.querySelectorAll('input[name="model"][type="checkbox"]'));
  }

  // Models listed in the YAML editor, lowercased
  function yamlModels() {
    const pre = document.querySelector('#paramText pre');
    if (!pre || typeof jsyaml === 'undefined') return [];
    try {
      const obj = jsyaml.load(pre.textContent || '') || {};
      let models = obj.models || [];
      if (typeof models === 'string') models = models.split(',');
      return models.map(x => String(x).trim().toLowerCase()).filter(Boolean);
    } catch (e) {
      return [];
    }
  }

  function selectedFromHashOrYaml() {
    const hash = typeof getHash === 'function' ? getHash() : {};
    if (hash.models) {
      return String(hash.models).split(',').map(x => x.trim().toLowerCase()).filter(Boolean);
    }
    return yamlModels();
  }

  // Check the boxes that match the hash (or the YAML)
  function syncFromState() {
    const selected = selectedFromHashOrYaml();
    checkboxes().forEach(cb => {
      cb.checked = selected.includes(cb.value.toLowerCase());
    });
  }

  // Write the checked boxes to #models= so param-input.js updates the YAML
  function syncToHash() {
    const values = checkboxes().filter(cb => cb.checked).map(cb => cb.value);
    if (typeof getHash !== 'function' || typeof goHash !== 'function') return;
    const hash = getHash();
    if (values.length) {
      hash.models = values.join(',');
    } else {
      delete hash.models;
    }
    goHash(hash);
  }

  let bound = false;

  // Keep all model checkboxes on the page in sync with the hash and YAML
  function bind() {
    checkboxes().forEach(cb => {
      if (cb.dataset.rsBound) return;
      cb.dataset.rsBound = '1';
      cb.addEventListener('change', () => {
        // A run needs at least one model, so the last checked box stays checked
        if (!checkboxes().some(c => c.checked)) {
          cb.checked = true;
          cb.title = 'At least one model is required';
          return;
        }
        syncToHash();
      });
    });
    syncFromState();
    if (bound) return;
    bound = true;
    window.addEventListener('hashchange', syncFromState);
    document.addEventListener('hashChangeEvent', syncFromState);
    // param-input.js rewrites the YAML after loading a parameter base
    const pre = document.querySelector('#paramText pre');
    if (pre && typeof MutationObserver !== 'undefined') {
      new MutationObserver(syncFromState).observe(pre, { childList: true, characterData: true, subtree: true });
    }
  }

  // Render a compact checkbox list into the container (default #modelSelect), then bind it
  function render(containerId) {
    const host = document.getElementById(containerId || 'modelSelect');
    if (!host) return Promise.resolve([]);
    return loadModels().then(models => {
      if (!document.getElementById('rsModelSelectStyle')) {
        const style = document.createElement('style');
        style.id = 'rsModelSelectStyle';
        style.textContent = '.rs-model-select{display:flex;flex-wrap:wrap;gap:6px 20px;margin:8px 0}'
          + '.rs-model-option{display:inline-flex;align-items:center;gap:6px;cursor:pointer}';
        document.head.appendChild(style);
      }
      host.classList.add('rs-model-select');
      host.innerHTML = models.map(m =>
        `<label class="rs-model-option"><input type="checkbox" name="model" value="${m.value}"> ${m.label}</label>`
      ).join('');
      bind();
      return models;
    });
  }

  window.RSModelSelect = { loadModels, render, bind, selected: () => checkboxes().filter(cb => cb.checked).map(cb => cb.value) };

  // Pages with a #modelSelect container get the list automatically
  function autoRender() {
    if (document.getElementById('modelSelect')) render('modelSelect');
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoRender);
  } else {
    autoRender();
  }
})();
