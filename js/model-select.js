// Shared models picker for RealityStream pages.
//
// The model list and each model's details come from realitystream/models/README.md, where every model is a
// checkbox heading followed by its description:
//   ## <input type="checkbox" id="model-lr" name="model" value="lr"> Logistic Regression (lr)
//   - **Type**: ...
// so adding or editing a model there updates every page that uses this picker:
// realitystream/models/ and the RealityStream Cloud Run home page (realitystream/api/home.html).
//
// render() shows the model titles with checkboxes. Each title expands to its details, and
// "Expand all" opens every one and adds #expand=all to the page URL (so a link loads expanded), and
// "Collapse all" removes it. The checkboxes stay in sync with the URL hash (#models=lr,rfc),
// which param-input.js copies into the YAML's models: line. Without models in the hash,
// the checkboxes show the models already in the YAML (#paramText pre). All of them may be unchecked.
// Each model's details end with an empty .rs-model-params[data-model] slot, where model-params.js puts
// that model's hyperparameter fields.
//
// run-panel.js calls setLocked() (in Active mode until the Team Passphrase is entered) and setNote()
// (the selection limit, shown above the checkboxes).
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
  // A model's details run until the next model heading, a horizontal rule, another heading, or an HTML comment.
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
            } else if (current && !done && (/^\s*-{3,}\s*$/.test(line) || /^#/.test(line) || /^\s*<!--/.test(line))) {
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

  // Keep each short parenthetical phrase in the element's text, such as "(e.g., churn prediction)", on one line
  function nowrapParens(el) {
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) if (/\([^()]{1,32}\)/.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
    nodes.forEach(node => {
      const frag = document.createDocumentFragment();
      let last = 0;
      node.nodeValue.replace(/\([^()]{1,32}\)/g, (match, at) => {
        frag.appendChild(document.createTextNode(node.nodeValue.slice(last, at)));
        const span = document.createElement('span');
        span.className = 'rs-nowrap';
        span.textContent = match;
        frag.appendChild(span);
        last = at + match.length;
      });
      frag.appendChild(document.createTextNode(node.nodeValue.slice(last)));
      node.parentNode.replaceChild(frag, node);
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

  // Locked (set by run-panel.js) greys out the checkboxes; the Details toggles still work
  let locked = false;

  // Check the boxes that match the hash (or the YAML)
  function syncFromState() {
    const selected = selectedFromHashOrYaml();
    checkboxes().forEach(cb => {
      cb.checked = selected.includes(cb.value.toLowerCase());
      cb.disabled = locked;
    });
    announce();
  }

  // Let other scripts (run-panel.js) know the checked models may have changed
  function announce() {
    document.dispatchEvent(new Event('rsModelsChanged'));
  }

  function setLocked(on) {
    locked = !!on;
    checkboxes().forEach(cb => { cb.disabled = locked; });
    document.querySelectorAll('.rs-model-select').forEach(el => el.classList.toggle('rs-models-locked', locked));
  }

  // HTML above the checkboxes, or '' to hide it
  function setNote(html) {
    document.querySelectorAll('.rs-models-note').forEach(note => {
      note.innerHTML = html || '';
      note.hidden = !html;
    });
  }

  // Write the checked boxes to #models= so param-input.js updates the YAML.
  // With none checked, empty the YAML's models: line first, since the YAML is what's shown without #models=.
  function syncToHash() {
    const values = checkboxes().filter(cb => cb.checked).map(cb => cb.value);
    if (typeof goHash !== 'function') return;
    if (values.length) {
      goHash({ models: values.join(',') });
      return;
    }
    const pre = document.querySelector('#paramText pre');
    if (pre && typeof jsyaml !== 'undefined') {
      try {
        const obj = jsyaml.load(pre.textContent || '') || {};
        obj.models = [];
        pre.textContent = typeof convertToYAML === 'function' ? convertToYAML(obj) : jsyaml.dump(obj, { lineWidth: -1 });
      } catch (e) {} // The YAML is being edited and isn't valid yet
    }
    goHash({}, ['models']);
  }

  let bound = false;

  // Keep all model checkboxes on the page in sync with the hash and YAML
  function bind() {
    checkboxes().forEach(cb => {
      if (cb.dataset.rsBound) return;
      cb.dataset.rsBound = '1';
      cb.addEventListener('change', () => {
        syncToHash();
        announce();
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
.rs-models-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:6px 12px; margin:0 0 8px; font-size:14px; }
.rs-models-note { margin:0 0 8px; color:#c2410c; } .dark .rs-models-note { color:#fb923c; }
.rs-models-locked .rs-model-option { opacity:.55; cursor:default; }
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
.rs-model-params:empty { display:none; }
.rs-nowrap { white-space:nowrap; }
`;

  function setExpanded(row, open) {
    if (!row) return;
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
      host.innerHTML = '<div class="rs-models-toolbar"><button type="button" class="rs-expand-all">Expand all</button></div>'
        + '<div class="rs-models-note" hidden></div>'
        + models.map(m => `
<div class="rs-model-row" data-model="${escapeHtml(m.value)}">
  <div class="rs-model-head">
    <label class="rs-model-option"><input type="checkbox" name="model" value="${escapeHtml(m.value)}"> ${m.label}</label>
    <button type="button" class="rs-model-toggle" aria-expanded="false" title="Details"><span class="rs-chevron">&#9656;</span> Details</button>
  </div>
  <div class="rs-model-details" hidden><div class="rs-model-text">${mdToHtml(m.details)}</div><div class="rs-model-params" data-model="${escapeHtml(m.value)}"></div></div>
</div>`).join('');
      resolveLinks(host);
      host.querySelectorAll('.rs-model-option, .rs-model-text').forEach(nowrapParens);

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
        // Without a history entry or a hash change event: only the details open or close
        if (typeof goHashQuiet === 'function') {
          if (open) goHashQuiet({ expand: 'all' }); else goHashQuiet({}, ['expand']);
        }
      });
      // #expand=all opens every model's details when the page loads
      if (typeof getHash === 'function' && getHash().expand === 'all') {
        rows.forEach(row => setExpanded(row, true));
        updateAllBtn();
      }
      bind();
      // model-params.js fills each model's .rs-model-params, and run-panel.js applies the mode
      document.dispatchEvent(new Event('rsModelSelectRendered'));
      return models;
    });
  }

  // Open a model's details (model-params.js, to show a field the API rejected)
  function expand(value) {
    const row = Array.from(document.querySelectorAll('.rs-model-row')).find(r => r.dataset.model === value);
    setExpanded(row, true);
  }

  window.RSModelSelect = {
    loadReadme, loadModels, render, renderMarkdown, bind, setLocked, setNote, expand,
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
