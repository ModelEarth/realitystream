// Shared models picker for RealityStream pages.
//
// The model list and each model's details come from realitystream/models/README.md, where every model is a
// checkbox heading followed by its description:
//   ## <input type="checkbox" id="model-lr" name="model" value="lr"> Logistic Regression (lr)
//   - **Type**: ...
// so adding or editing a model there updates every page that uses this picker:
// realitystream/models/ and the RealityStream Cloud Run home page (realitystream/models/home.html).
//
// render() shows the model titles with checkboxes. Each title expands to its details, and
// "Expand all" opens every one. The checkboxes stay in sync with the URL hash (#models=lr,rfc),
// which param-input.js copies into the YAML's models: line. Without models in the hash,
// the checkboxes show the models already in the YAML (#paramText pre).
//
// Needs getHash() and goHash() from localsite/js/localsite.js. Uses showdown for the details when the
// page has loaded it, otherwise a small built-in converter.

(function () {
  // README.md resides in ../models/ relative to this script, wherever the script is hosted
  const README_URL = (function () {
    try {
      return new URL('../models/README.md', document.currentScript.src).href;
    } catch (e) {
      return '/realitystream/models/README.md';
    }
  })();

  const MODEL_HEADING = /^#+\s*<input[^>]*name="model"[^>]*value="([^"]+)"[^>]*>\s*(.+)$/;
  let readmePromise = null;

  // {intro, models: [{value, label, details}], outro} as Markdown, split at the model headings.
  // A model's details run until the next model heading, a horizontal rule, or another heading.
  function loadReadme() {
    if (!readmePromise) {
      readmePromise = fetch(README_URL)
        .then(r => r.text())
        .then(text => {
          const intro = [], outro = [], models = [];
          let current = null, done = false;
          text.split('\n').forEach(line => {
            const m = line.match(MODEL_HEADING);
            if (m && !done) {
              current = { value: m[1].trim(), label: m[2].trim(), lines: [] };
              models.push(current);
            } else if (current && !done && (/^\s*-{3,}\s*$/.test(line) || /^#/.test(line))) {
              done = true;
              outro.push(line);
            } else if (done) {
              outro.push(line);
            } else if (current) {
              current.lines.push(line);
            } else {
              intro.push(line);
            }
          });
          models.forEach(m => { m.details = m.lines.join('\n').trim(); delete m.lines; });
          return { intro: intro.join('\n').trim(), models, outro: outro.join('\n').trim() };
        });
    }
    return readmePromise;
  }

  function loadModels() {
    return loadReadme().then(r => r.models);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // Markdown to HTML: showdown when available, else lists, paragraphs, bold and links
  function mdToHtml(md) {
    if (window.showdown) {
      return new showdown.Converter({ tables: true, simpleLineBreaks: true }).makeHtml(md);
    }
    const inline = s => escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2">$1</a>');
    let html = '', inList = false, para = [];
    const flush = () => { if (para.length) { html += '<p>' + para.join('<br>') + '</p>'; para = []; } };
    md.split('\n').forEach(line => {
      const item = line.match(/^\s*[-*]\s+(.*)$/);
      if (item) {
        flush();
        if (!inList) { html += '<ul>'; inList = true; }
        html += '<li>' + inline(item[1]) + '</li>';
        return;
      }
      if (inList) { html += '</ul>'; inList = false; }
      if (line.trim()) para.push(inline(line.trim())); else flush();
    });
    flush();
    if (inList) html += '</ul>';
    return html;
  }

  // Links in the README are relative to it, so resolve them for pages elsewhere
  function resolveLinks(el) {
    el.querySelectorAll('a[href]').forEach(a => {
      const href = a.getAttribute('href');
      if (!/^([a-z]+:|#|\/\/)/i.test(href)) a.href = new URL(href, README_URL).href;
    });
  }

  // Render README Markdown (for example the intro or outro) into an element
  function renderMarkdown(el, md) {
    if (!el) return;
    el.innerHTML = mdToHtml(md);
    resolveLinks(el);
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

  // Single mode (set by run-panel.js without an API key) allows one model at a time
  let single = false;

  // In single mode, keep only the first checked model and pass that on to the hash and YAML
  function enforceSingle() {
    if (!single) return;
    const checked = checkboxes().filter(cb => cb.checked);
    if (checked.length > 1) {
      checked.slice(1).forEach(cb => { cb.checked = false; });
      syncToHash();
    }
  }

  // Check the boxes that match the hash (or the YAML)
  function syncFromState() {
    const selected = selectedFromHashOrYaml();
    checkboxes().forEach(cb => {
      cb.checked = selected.includes(cb.value.toLowerCase());
    });
    enforceSingle();
  }

  function setSingle(on) {
    single = !!on;
    const note = document.querySelector('.rs-models-single');
    if (note) note.style.display = single ? 'inline' : 'none';
    enforceSingle();
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
        if (single && cb.checked) {
          checkboxes().forEach(c => { if (c !== cb) c.checked = false; });
        }
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

  const STYLE = `
.rs-model-select { margin:8px 0; }
.rs-models-toolbar { margin:0 0 6px; font-size:14px; }
.rs-models-single { margin-left:12px; opacity:.75; }
.rs-models-toolbar button { background:none; border:0; padding:0; color:inherit; text-decoration:underline; cursor:pointer; font:inherit; opacity:.8; }
.rs-model-row { border-top:1px solid rgba(127,127,127,.25); }
.rs-model-row:last-child { border-bottom:1px solid rgba(127,127,127,.25); }
.rs-model-head { display:flex; align-items:center; gap:8px; padding:6px 0; }
.rs-model-option { display:inline-flex; align-items:center; gap:8px; cursor:pointer; flex:1; min-width:0; }
.rs-model-toggle { background:none; border:0; padding:2px 6px; cursor:pointer; color:inherit; font-size:13px; opacity:.75; border-radius:6px; }
.rs-model-toggle:hover { opacity:1; background:rgba(127,127,127,.15); }
.rs-model-toggle .rs-chevron { display:inline-block; transition:transform .15s; }
.rs-model-toggle[aria-expanded="true"] .rs-chevron { transform:rotate(90deg); }
.rs-model-details { padding:0 0 10px 30px; font-size:14px; }
.rs-model-details ul { margin:4px 0; padding-left:18px; }
.rs-model-details p { margin:4px 0; }
`;

  function setExpanded(row, open) {
    const btn = row.querySelector('.rs-model-toggle');
    const details = row.querySelector('.rs-model-details');
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    details.hidden = !open;
  }

  // Render model titles with checkboxes and expandable details into the container (default #modelSelect)
  function render(containerId) {
    const host = document.getElementById(containerId || 'modelSelect');
    if (!host) return Promise.resolve([]);
    return loadModels().then(models => {
      if (!document.getElementById('rsModelSelectStyle')) {
        const style = document.createElement('style');
        style.id = 'rsModelSelectStyle';
        style.textContent = STYLE;
        document.head.appendChild(style);
      }
      host.classList.add('rs-model-select');
      host.innerHTML = '<div class="rs-models-toolbar"><button type="button" class="rs-expand-all">Expand all</button>'
        + `<span class="rs-models-single" style="display:${single ? 'inline' : 'none'}">Without an API key, choose 1 model.</span></div>`
        + models.map(m => `
<div class="rs-model-row">
  <div class="rs-model-head">
    <label class="rs-model-option"><input type="checkbox" name="model" value="${escapeHtml(m.value)}"> ${m.label}</label>
    <button type="button" class="rs-model-toggle" aria-expanded="false" title="Details"><span class="rs-chevron">&#9656;</span> Details</button>
  </div>
  <div class="rs-model-details" hidden>${mdToHtml(m.details)}</div>
</div>`).join('');
      resolveLinks(host);

      const rows = Array.from(host.querySelectorAll('.rs-model-row'));
      const allBtn = host.querySelector('.rs-expand-all');
      const updateAllBtn = () => {
        const allOpen = rows.every(r => !r.querySelector('.rs-model-details').hidden);
        allBtn.textContent = allOpen ? 'Collapse all' : 'Expand all';
      };
      rows.forEach(row => row.querySelector('.rs-model-toggle').addEventListener('click', () => {
        setExpanded(row, row.querySelector('.rs-model-details').hidden);
        updateAllBtn();
      }));
      allBtn.addEventListener('click', () => {
        const open = allBtn.textContent === 'Expand all';
        rows.forEach(row => setExpanded(row, open));
        updateAllBtn();
      });
      bind();
      return models;
    });
  }

  window.RSModelSelect = {
    loadReadme, loadModels, render, renderMarkdown, bind, setSingle,
    selected: () => checkboxes().filter(cb => cb.checked).map(cb => cb.value),
  };

  // Pages with a #modelSelect container get the picker automatically
  function autoRender() {
    if (document.getElementById('modelSelect')) render('modelSelect');
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoRender);
  } else {
    autoRender();
  }
})();
