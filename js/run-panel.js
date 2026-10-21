// Shared "Run Models" panel for RealityStream pages: today's usage, SMOTE choice, Run button and results.
// Used by realitystream/models/ and the Cloud Run home page (realitystream/api/home.html).
//
// Runs go to the RealityStream Cloud Run API (api/main.py): this site when the page is served by
// Cloud Run, otherwise RS_RUN_API below (or data-api on #rsRunControls). On localhost, the page uses the
// local API at port 8080 when `python api/main.py` is running there, and ?api= in the page URL picks
// another, for example ?api=https://realitystream-kwr4qrkopq-uc.a.run.app/ to run on Cloud Run.
//
// Each run writes a report folder ({year}/run-{date}T{time}, with an index.html). The local API saves it
// in the webroot's reports folder, and the link after the Run button opens it. With GITHUB_REPORTS_TOKEN
// set, the same folder is pushed to GitHub modelearth/reports, and a line below the button says whether it was.
// The YAML posted to /run is the text of #paramText pre, which param-input.js, model-select.js and
// model-params.js keep up to date.
//
// Preview and Active modes (switch at the top right of the page, remembered per browser):
//   Preview (default)  anyone may check any models and edit every field; Run Models is gray.
//   Active             shows the Team Passphrase field. Until a working passphrase is entered, every
//                      other field is greyed out. Runs are limited to MODEL_LIMIT models (1 without the
//                      passphrase, as the service allows), with a note above the model checkboxes.
//
// Events for model-params.js: rsModeChange (detail.mode, and detail.locked: true in Active mode without a
// valid Team Passphrase) and rsParamError (detail.field and detail.message, when /run rejects a value in the YAML).
//
// Containers (each optional except #rsRunControls):
//   #rsMode         Preview / Active switch and the Team Passphrase field (created above #rsRunControls when missing)
//   #rsUsage        today's run time and cost
//   #rsRunControls  SMOTE choice, Run button and status
//
// The Team Passphrase is the service's REALITYSTREAM_API_KEY (sent as header X-API-Key), shared with the
// team; it isn't a Google key. Without it the service allows 1 model, once per day, and a note explains
// how to run more. A passphrase typed into the field is remembered in this browser (localStorage) until
// "Clear" (which asks to confirm). Once a working passphrase is saved, the field is hidden and only
// "Passphrase saved in browser" and Clear show.
//   #rsResults      results table after a run
//   #rsEndpoints    list of API endpoints
//   [data-rs-layout] the Settings section: an Expand / Condense switch goes at the right of its h2.
//                   Expansive (default) stacks Features above Target with full URLs and shows the YAML; Condensed puts
//                   them side by side and hides their URLs and the YAML. The choice is remembered per browser.

(function () {
  const RS_RUN_API = 'https://realitystream-kwr4qrkopq-uc.a.run.app/';
  // Models per run in Active mode with the Team Passphrase (a deploy on your own Google Cloud account may raise it)
  const MODEL_LIMIT = 2;

  const LOCAL_API_PORT = 8080;

  const $ = id => document.getElementById(id);
  const onLocalhost = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);

  let localApi = ''; // The local API's address, when one answers on localhost (see ready)

  function apiBase() {
    const el = $('rsRunControls');
    if (el && el.dataset.api) return new URL(el.dataset.api, location.href).href.replace(/\/?$/, '/');
    if (onLocalhost) {
      const local = new URLSearchParams(location.search).get('api');
      if (local) return new URL(local, location.href).href.replace(/\/?$/, '/');
      if (localApi) return localApi;
    }
    return /\.run\.app$/.test(location.hostname) ? location.origin + '/' : RS_RUN_API;
  }

  // On localhost without ?api=, use the local API when its /health answers. Pages wait for this
  // (RSRunPanel.ready) before their first call to apiBase().
  const ready = (function detectLocalApi() {
    if (!onLocalhost || new URLSearchParams(location.search).get('api')) return Promise.resolve();
    const base = `${location.protocol}//${location.hostname}:${LOCAL_API_PORT}/`;
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), 1500);
    return fetch(base + 'health', { cache: 'no-store', signal: abort.signal })
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (d && d.service === 'realitystream') localApi = base; })
      .catch(() => {})
      .finally(() => clearTimeout(timer));
  })();

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  const STYLE = `
.rs-usage-when { font-weight:400; }
.rs-usage-bar { height:10px; background:rgba(127,127,127,.2); border-radius:6px; overflow:hidden; margin:10px 0 6px; }
.rs-usage-fill { height:100%; width:0; background:#2f6fde; transition:width .3s; }
.rs-usage-fill.full { background:#c2410c; }
.rs-usage-stats { display:flex; flex-wrap:wrap; gap:8px 24px; opacity:.85; }
.rs-run-label { font-weight:600; margin:16px 0 6px; }
.rs-run-label-note { font-weight:400; }
.rs-report-dest { margin-top:10px; opacity:.85; }
.rs-output-link { font-weight:600; }
.rs-output-none { opacity:.85; }
.rs-github-status { margin-top:8px; }
.rs-smote { display:flex; flex-wrap:wrap; gap:6px 20px; }
.rs-smote label { display:inline-flex; gap:6px; align-items:center; cursor:pointer; }
.rs-run-actions { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-top:18px; padding-bottom:10px; }
/* Button shape shared by Run and Intro; button.rs-btn-shape outranks localsite's pill-shaped .btn.
   Border width and style only, so Intro keeps localsite's .btn-clear border color. */
button.rs-btn-shape { border-width:1px; border-style:solid; border-radius:8px; padding:9px 22px; font-size:16px; line-height:1.25; cursor:pointer; }
.rs-run-button { background:#2f6fde; color:#fff; border-color:transparent; }
.rs-run-button:disabled { opacity:.6; cursor:default; }
.rs-run-button:hover:not(:disabled) { background:#2558b8; }
.rs-run-button.rs-preview, .rs-run-button.rs-preview:disabled { background:#9aa1ab; opacity:1; }
.rs-run-hint { margin-top:16px; }
.rs-run-hint + .rs-run-actions { margin-top:8px; }
/* Preview / Active switch at the top right, aligned left when #rsMode is narrow */
#rsMode { container-type:inline-size; }
.rs-mode-bar { display:flex; flex-wrap:wrap; align-items:center; justify-content:flex-end; gap:8px 14px; margin:0 0 12px; }
@container (max-width:600px) { .rs-mode-bar { justify-content:flex-start; } }
.rs-mode-bar .rs-api-key { margin:0; flex-wrap:wrap; align-items:center; gap:6px 12px; }
.rs-mode-switch { display:inline-flex; border:1px solid rgba(127,127,127,.35); border-radius:8px; overflow:hidden; font-size:14px; }
.rs-mode-switch button { background:transparent; color:inherit; border:0; padding:6px 16px; cursor:pointer; font:inherit; }
.rs-mode-switch button[aria-pressed="true"] { background:#2f6fde; color:#fff; }
/* Greyed out in Active mode until the Team Passphrase is entered (the elements are also inert) */
.rs-locked { opacity:.5; }
/* A page's extra button at the far right of the Run row (Intro on realitystream/models/, which uses
   localsite's transparent .btn-clear with the .rs-btn-shape shape) */
.rs-intro-button { margin-left:auto; }
.rs-layout-switch { margin-left:auto; display:inline-flex; border:1px solid rgba(127,127,127,.35); border-radius:8px; overflow:hidden; font-size:13px; font-weight:400; }
.rs-layout-switch button { background:transparent; color:inherit; border:0; padding:4px 12px; cursor:pointer; font:inherit; }
.rs-layout-switch button[aria-pressed="true"] { background:#2f6fde; color:#fff; }
[data-rs-layout] > h2 { display:flex; align-items:center; gap:12px; }
/* localsite's .flexmain h2::before (hash-link offset) would become a flex item and add space before the title */
[data-rs-layout] > h2::before { display:none; }
.rs-layout-expansive .rs-pickers { flex-direction:column; }
.rs-layout-expansive .rs-picker { width:100%; }
.rs-layout-expansive .rsUrlShort { display:none; }
.rs-layout-expansive .rsUrlFull { display:inline; word-break:break-all; }
.rs-layout-condensed .rsCardFilename + .rsCardTitle { display:none; }
.rs-layout-condensed #paramText { display:none; }
.rs-run-status.error { color:#c2410c; } .rs-run-status.ok { color:var(--color-success, #6aa442); }
.dark .rs-run-status.error { color:#fb923c; } .dark .rs-run-status.ok { color:var(--color-success, #6aa442); }
.rs-free-note { margin-top:14px; opacity:.85; }
.rs-confirm-backdrop { position:fixed; inset:0; background:rgba(0,0,0,.35); display:flex; align-items:center; justify-content:center; z-index:10000; }
.rs-confirm-box { background:#fff; color:#222; border-radius:12px; padding:20px 22px; max-width:calc(100% - 32px); box-shadow:0 8px 30px rgba(0,0,0,.25); font-size:16px; }
.dark .rs-confirm-box { background:#1d2128; color:#eee; }
.rs-confirm-actions { display:flex; gap:10px; justify-content:flex-end; margin-top:16px; }
.rs-confirm-actions button { border:1px solid rgba(127,127,127,.5); border-radius:8px; padding:7px 18px; font-size:15px; cursor:pointer; background:transparent; color:inherit; }
.rs-confirm-actions .rs-confirm-yes { background:#2f6fde; border-color:#2f6fde; color:#fff; }
.rs-confirm-actions .rs-confirm-yes:hover { background:#2558b8; }
.rs-forget-key { background:none; border:0; padding:0; color:inherit; text-decoration:underline; cursor:pointer; font:inherit; opacity:.8; }
.rs-key-state { opacity:.75; }
.rs-key-field { margin:0; display:inline; }
.rs-api-key input { padding:6px 8px; border-radius:8px; border:1px solid rgba(127,127,127,.4); background:transparent; color:inherit; width:260px; max-width:100%; }
.rs-table-wrap { overflow-x:auto; }
.rs-table { width:100%; border-collapse:collapse; font-size:14px; }
.rs-table th, .rs-table td { text-align:left; padding:7px 10px 7px 0; border-top:1px solid rgba(127,127,127,.25); white-space:nowrap; }
.rs-table th { opacity:.75; font-weight:600; }
.rs-table td.wrap { white-space:normal; }
.rs-settings-cell { font-size:12px; opacity:.8; min-width:180px; }
.rs-run-settings { margin:0 0 8px; opacity:.85; }
`;

  let info = null;
  let keyValid = false;
  let keyChecked = false; // Fields lock only after the saved key is checked, so a key holder's fields don't flicker
  let lockedNow = null;   // true in Active mode without a valid Team Passphrase; null until first applied
  const KEY_STORE = 'realitystreamApiKey';
  const MODE_STORE = 'realitystreamMode';
  let mode = 'preview';
  try { mode = localStorage.getItem(MODE_STORE) === 'active' ? 'active' : 'preview'; } catch (e) {}

  function savedKey() {
    try { return localStorage.getItem(KEY_STORE) || ''; } catch (e) { return ''; }
  }
  function saveKey(value) {
    try { value ? localStorage.setItem(KEY_STORE, value) : localStorage.removeItem(KEY_STORE); } catch (e) {}
  }

  function moreRunsHtml() {
    const url = (info && info.more_runs_url) || 'https://model.earth/realitystream/deploy/#googleaccountautomation';
    return `<br>To run more, <a href="${url}" target="_blank" rel="noopener">add a REALITYSTREAM_API_KEY</a> for this Google Cloud project, `
      + `or <a href="${url}" target="_blank" rel="noopener">run on your own cloud account</a>.`;
  }

  // True when the service wants a Team Passphrase and no working one has been entered
  function keyNeeded() {
    return !!(info && info.api_key_enabled) && !keyValid;
  }

  // Models one run may have: 1 without the Team Passphrase, else MODEL_LIMIT
  function modelLimit() {
    return keyNeeded() ? 1 : MODEL_LIMIT;
  }

  function limitText() {
    if (keyNeeded()) return 'Without the Team Passphrase, limit your selection to 1 model.';
    const url = (info && info.more_runs_url) || 'https://model.earth/realitystream/deploy/#googleaccountautomation';
    return `Limit your selection to ${MODEL_LIMIT} models, unless you're using `
      + `<a href="${url}" target="_blank" rel="noopener">your own Google Cloud account</a>.`;
  }

  // Settings fields to grey out while locked: the Settings section except its heading and the Run row,
  // and the SMOTE choice (model-select.js and model-params.js grey out their own fields)
  function lockables() {
    const controls = $('rsRunControls');
    const list = Array.from(document.querySelectorAll('#rsRunControls .rs-smote, #rsRunControls .rs-smote-label'));
    const section = document.querySelector('[data-rs-layout]');
    if (section) {
      Array.from(section.children).forEach(el => {
        if (el.tagName !== 'H2' && !el.contains(controls)) list.push(el);
      });
    }
    return list;
  }

  function lockSettings(locked) {
    lockables().forEach(el => {
      el.classList.toggle('rs-locked', locked);
      if (locked) el.setAttribute('inert', ''); else el.removeAttribute('inert');
    });
    const editor = $('paramText');
    if (editor) editor.setAttribute('contenteditable', locked ? 'false' : 'true');
  }

  // Apply the mode and the Team Passphrase state to the page
  function applyKeyMode() {
    const active = mode === 'active';
    const locked = active && (!keyChecked || keyNeeded());
    document.querySelectorAll('#rsMode .rs-mode-switch button').forEach(b =>
      b.setAttribute('aria-pressed', b.dataset.mode === mode ? 'true' : 'false'));
    const keyRow = document.querySelector('.rs-api-key');
    if (keyRow) keyRow.style.display = active && info && info.api_key_enabled ? 'flex' : 'none';

    if (window.RSModelSelect && RSModelSelect.setLocked) RSModelSelect.setLocked(locked);
    lockSettings(locked);
    if (locked !== lockedNow) {
      lockedNow = locked;
      document.dispatchEvent(new CustomEvent('rsModeChange', { detail: { mode, locked } }));
    }

    const note = document.querySelector('#rsRunControls .rs-free-note');
    if (note) {
      note.style.display = active && keyNeeded() ? 'block' : 'none';
      note.innerHTML = 'Without the Team Passphrase, you can run 1 model once per day. ' + moreRunsHtml();
    }
    updateModelState();
    const stored = !!savedKey();
    const forget = document.querySelector('.rs-forget-key');
    if (forget) forget.style.display = stored ? 'inline' : 'none';
    // A working saved passphrase hides the field; one that isn't accepted stays visible to correct
    const field = document.querySelector('.rs-api-key .rs-key-field');
    if (field) field.style.display = stored && keyValid ? 'none' : '';
    const keyState = document.querySelector('.rs-key-state');
    if (keyState) {
      const typed = keyInputEl() ? keyInputEl().value : '';
      keyState.textContent = !typed ? '' : keyValid ? 'Passphrase saved in browser' : 'Passphrase not accepted';
    }
  }

  // The Team Passphrase input (its form also holds a hidden username input, so select it by class)
  function keyInputEl() {
    return document.querySelector('.rs-api-key .rs-key-input');
  }

  // Ask the service whether the Team Passphrase in the field is valid; with save, store it when it is
  async function checkKey(save) {
    const input = keyInputEl();
    const value = input ? input.value.trim() : '';
    let valid = false;
    if (value) {
      try {
        const r = await fetch(apiBase() + 'key', { headers: { 'X-API-Key': value }, cache: 'no-store' }).then(r => r.json());
        valid = r.key === 'valid';
      } catch (e) {}
    }
    if (save && valid) saveKey(value);
    keyValid = valid;
    keyChecked = true;
    applyKeyMode();
  }

  // Today's totals and the endpoint list come from the API's JSON home
  async function refreshUsage() {
    try {
      info = await fetch(apiBase(), { headers: { Accept: 'application/json' }, cache: 'no-store' }).then(r => r.json());
    } catch (e) {
      if ($('rsUsage')) $('rsUsage').querySelector('.rs-usage-stats').textContent = 'The RealityStream service could not be reached.';
      return;
    }
    const usage = $('rsUsage');
    if (usage) {
      const t = info.today || { seconds: 0, cost_usd: 0, runs: 0 };
      const pct = Math.min(100, 100 * t.cost_usd / info.daily_cost_limit_usd);
      usage.querySelector('.rs-usage-day').textContent = info.day;
      usage.querySelector('.rs-usage-stats').innerHTML =
        `<span><b>${(t.seconds / 60).toFixed(1)}</b> minutes run</span>`
        + `<span><b>${info.minutes_left_today != null ? info.minutes_left_today.toFixed(1) : '?'}</b> of ${Math.round(info.daily_limit_minutes)} minutes left</span>`
        + `<span><b>$${t.cost_usd.toFixed(3)}</b> of $${info.daily_cost_limit_usd.toFixed(2)}</span>`
        + `<span><b>${t.runs}</b> runs</span>`;
      const fill = usage.querySelector('.rs-usage-fill');
      fill.style.width = pct + '%';
      fill.classList.toggle('full', pct >= 100);
    }
    updateReportDest();
    applyKeyMode();
    if ($('rsEndpoints')) {
      $('rsEndpoints').innerHTML = '<div class="rs-table-wrap"><table class="rs-table">'
        + Object.entries(info.endpoints || {}).map(([k, v]) =>
          `<tr><td><code>${escapeHtml(k)}</code></td><td class="wrap">${escapeHtml(v)}</td></tr>`).join('')
        + '</table></div>';
    }
  }

  // Run button and the note above the model checkboxes: "Run Model" when exactly one model is checked,
  // otherwise "Run Models"; gray in Preview mode; the limit note in Active mode when too many are checked
  function updateModelState() {
    const button = document.querySelector('#rsRunControls .rs-run-button');
    if (!button) return;
    const count = window.RSModelSelect ? RSModelSelect.selected().length : 0;
    button.textContent = count === 1 ? 'Run Model' : 'Run Models';
    const preview = mode === 'preview';
    button.classList.toggle('rs-preview', preview);
    if (!running) button.disabled = preview;
    const hint = document.querySelector('#rsRunControls .rs-run-hint');
    if (hint) hint.hidden = !preview;
    if (window.RSModelSelect && RSModelSelect.setNote) {
      RSModelSelect.setNote(!preview && count > modelLimit() ? limitText() : '');
    }
  }

  function setMode(value) {
    mode = value === 'active' ? 'active' : 'preview';
    try { localStorage.setItem(MODE_STORE, mode); } catch (e) {}
    applyKeyMode();
    const input = keyInputEl();
    if (mode === 'active' && keyNeeded() && input && input.offsetParent) input.focus();
  }

  let running = false;

  function setStatus(text, kind, html) {
    const el = document.querySelector('#rsRunControls .rs-run-status');
    if (html) el.innerHTML = html; else el.textContent = text;
    el.className = 'rs-run-status ' + (kind || '');
  }

  const pct = v => (v == null ? '' : (v <= 1 ? (v * 100).toFixed(1) : Number(v).toFixed(1)) + '%');

  // A model's settings, from params_used, as "n_estimators 100, max_depth 8"
  function settingsText(used) {
    return Object.entries(used || {}).map(([k, v]) => `${k} ${Array.isArray(v) ? v.join('×') : v}`).join(', ');
  }

  function showResults(data) {
    const host = $('rsResults');
    if (!host) return;
    const rows = [];
    [['no_smote', 'Without SMOTE'], ['smote', 'With SMOTE']].forEach(([key, label]) => {
      (data[key] || []).forEach(r => rows.push(
        `<tr><td>${escapeHtml(r.model_type || r.model || '')}</td><td>${label}</td><td>${pct(r.accuracy)}</td>`
        + `<td>${pct(r.roc_auc)}</td><td>${pct(r.f1_score)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td>`
        + `<td>${r.time != null ? Number(r.time).toFixed(1) + ' s' : ''}</td>`
        + `<td class="wrap rs-settings-cell">${escapeHtml(settingsText(r.params_used))}</td></tr>`));
    });
    const s = data.settings;
    host.querySelector('.rs-results-body').innerHTML
      = (s ? `<p class="rs-run-settings">Test share ${Math.round(s.test_size * 100)}%, seed ${escapeHtml(s.random_state)}, `
        + `${escapeHtml(s.n_iter)} search tries for XGBoost and MLP</p>` : '')
      + '<div class="rs-table-wrap"><table class="rs-table">'
      + '<tr><th>Model</th><th>Training</th><th>Accuracy</th><th>ROC-AUC</th><th>F1</th><th>Precision</th><th>Recall</th><th>Time</th><th>Settings</th></tr>'
      + (rows.join('') || '<tr><td colspan="9">No models were trained.</td></tr>') + '</table></div>';
    host.style.display = 'block';
  }

  // The report folder: the local copy when this page shares a webroot with the local API, else the
  // copy pushed to GitHub (model.earth/reports shows it once GitHub Pages rebuilds)
  const apiIsLocal = () => /^https?:\/\/(localhost|127\.0\.0\.1)[:/]/.test(apiBase());

  function outputUrl(data) {
    if (data.output_path && onLocalhost && apiIsLocal()) return location.origin + '/' + data.output_path;
    if (data.uploaded_to) return `https://model.earth/reports/${data.uploaded_to}/`;
    return '';
  }

  let lastRun = null; // The last successful run's reply, for showOutput()

  const REPORTS_REPO_LINK = '<a href="https://github.com/modelearth/reports" target="_blank" rel="noopener">modelearth/reports</a>';
  const START_LOCAL_API = 'To save output in your webroot, start the local API: <code>cd realitystream && python api/main.py</code>';

  // A link to add GITHUB_REPORTS_TOKEN, on the local API's keys page
  const keysLink = () => (apiIsLocal()
    ? ` Add it on the <a href="${escapeHtml(apiBase())}keys" target="_blank" rel="noopener">local keys page</a>.` : '');

  // The GitHub token line after the Run button, before a run (from the API's home JSON)
  function tokenStatusHtml() {
    if (!info) return '';
    if (info.github_token === undefined) {
      return 'GitHub token: this Cloud Run revision doesn\'t report it yet, and doesn\'t keep output folders (a redeploy adds both).';
    }
    if (info.github_token) {
      return `GitHub token (GITHUB_REPORTS_TOKEN) is set: runs with the Team Passphrase are also pushed to ${REPORTS_REPO_LINK}.`;
    }
    return `GitHub token (GITHUB_REPORTS_TOKEN) is not set${info.saves_locally ? '' : ' on the Cloud Run service'}, `
      + `so output isn't pushed to ${REPORTS_REPO_LINK}.` + keysLink();
  }

  // After the Run button: the output folder link (or why there's none) and the GitHub token line.
  // data is a run's reply, or null to show the state before a run.
  function showOutput(data) {
    const link = document.querySelector('#rsRunControls .rs-output-link');
    const none = document.querySelector('#rsRunControls .rs-output-none');
    const github = document.querySelector('#rsRunControls .rs-github-status');
    if (!link) return;
    const url = data ? outputUrl(data) : '';
    link.hidden = !url;
    if (url) {
      link.href = url;
      link.textContent = 'Output: ' + (data.output_path || 'reports/' + data.uploaded_to + '/');
    }
    none.hidden = !data || !!url;
    if (data && !url) {
      none.innerHTML = 'No output folder: this run used the Cloud Run service, which keeps its output only by pushing it to GitHub.'
        + (onLocalhost ? ' ' + START_LOCAL_API : '');
    }
    const g = data && data.github;
    if (!data) {
      github.innerHTML = tokenStatusHtml();
    } else if (g && g.message) {
      github.innerHTML = escapeHtml(g.message.replace(/ Pushed to .*$/, ''))
        + (g.pushed_to ? ` Pushed to <a href="https://github.com/modelearth/reports/tree/main/${escapeHtml(g.pushed_to)}" target="_blank" rel="noopener">modelearth/reports/${escapeHtml(g.pushed_to)}</a>.` : '')
        + (!g.pushed_to && g.token === 'missing' ? keysLink() : '');
    } else {
      github.innerHTML = 'GitHub token: this Cloud Run revision doesn\'t report it, so the output wasn\'t pushed (a redeploy adds this).';
    }
    github.hidden = !github.innerHTML;
  }

  // Where runs go, before a run (from the API's home JSON)
  function updateReportDest() {
    const el = document.querySelector('#rsRunControls .rs-report-dest');
    if (!el || !info) return;
    el.innerHTML = info.saves_locally
      ? `Runs use your local API (${escapeHtml(apiBase())}), which saves each run's output in your webroot's <code>reports/{year}/run-{date-time}</code> folder.`
      : 'Runs use the RealityStream Cloud Run service.' + (onLocalhost ? ' ' + START_LOCAL_API : '');
    // refreshUsage() runs again after each run, so keep showing that run's output
    if (!running) showOutput(lastRun);
  }

  async function runModels() {
    if (mode !== 'active') return;
    const pre = document.querySelector('#paramText pre') || $('paramText');
    const yaml = pre ? pre.textContent.trim() : '';
    if (!yaml) {
      setStatus('Add parameters first.', 'error');
      return;
    }
    // With no models checked, the run still writes its report folder (data and baseline, no models)
    const count = window.RSModelSelect ? RSModelSelect.selected().length : 1;
    if (count > modelLimit()) {
      setStatus('', 'error', limitText());
      return;
    }
    const smote = (document.querySelector('#rsRunControls input[name="rs_smote"]:checked') || {}).value || '';
    const url = apiBase() + 'run' + (smote ? '?smote=' + smote : '');
    const headers = { 'Content-Type': 'text/yaml' };
    const key = keyInputEl();
    if (key && key.value.trim()) headers['X-API-Key'] = key.value.trim();

    const button = document.querySelector('#rsRunControls .rs-run-button');
    button.disabled = true;
    running = true;
    const started = Date.now();
    const timer = setInterval(() => setStatus(`Running… ${Math.round((Date.now() - started) / 1000)} s`), 1000);
    setStatus('Running…');
    lastRun = null;
    showOutput(null);
    try {
      const resp = await fetch(url, { method: 'POST', headers, body: yaml });
      const data = await resp.json().catch(() => ({}));
      const secs = Math.round((Date.now() - started) / 1000);
      if (resp.ok && data.status === 'success') {
        showResults(data);
        lastRun = data;
        showOutput(data);
        setStatus(`Finished in ${secs} s.`, 'ok');
      } else if (data.status === 'missing_key') {
        setStatus(`${data.key} is needed. ${data.how_to_get_it || ''}`, 'error');
      } else if (data.more_runs_url) {
        // Limits without the Team Passphrase: link the ways to run more
        const msg = escapeHtml((data.message || '').replace(/To run more,.*$/, '').trim());
        setStatus('', 'error', msg + ' ' + moreRunsHtml());
      } else if (resp.status === 400 && data.field) {
        // A value model-params.js can point to
        setStatus(data.message, 'error');
        document.dispatchEvent(new CustomEvent('rsParamError', { detail: { field: data.field, message: data.message } }));
      } else if (resp.status === 401) {
        setStatus('The Team Passphrase was not accepted. Check it, or choose Clear to run without one.', 'error');
      } else {
        setStatus(data.message || `Run failed (HTTP ${resp.status}).`, 'error');
      }
    } catch (e) {
      setStatus('The service did not respond. Long runs can take several minutes; check today\'s totals and try again.', 'error');
    } finally {
      clearInterval(timer);
      running = false;
      updateModelState();
      refreshUsage();
    }
  }

  function init() {
    const controls = $('rsRunControls');
    if (!controls) return;
    if (!document.getElementById('rsRunPanelStyle')) {
      const style = document.createElement('style');
      style.id = 'rsRunPanelStyle';
      style.textContent = STYLE;
      document.head.appendChild(style);
    }
    if ($('rsUsage')) {
      $('rsUsage').innerHTML = '<h2>Today\'s model run time <span class="rs-usage-when">(<span class="rs-usage-day">…</span>, resets at midnight Eastern)</span></h2>'
        + '<div class="rs-usage-bar"><div class="rs-usage-fill"></div></div><div class="rs-usage-stats">Loading…</div>';
    }
    let modeHost = $('rsMode');
    if (!modeHost) {
      modeHost = document.createElement('div');
      modeHost.id = 'rsMode';
      controls.parentNode.insertBefore(modeHost, controls);
    }
    modeHost.innerHTML = `
<div class="rs-mode-bar">
  <div class="rs-api-key" style="display:none">
    <!-- Its own form, so a browser offering to save the passphrase doesn't pick up another field on the page
         as the username. The hidden username field is blank (if a browser still fills one in, use a space). -->
    <form class="rs-key-field" autocomplete="on">
      <input type="text" name="username" autocomplete="username" value="" tabindex="-1" aria-hidden="true" style="display:none">
      <label>Team Passphrase <input type="password" class="rs-key-input" name="realitystream-passphrase" autocomplete="current-password"></label>
    </form>
    <span class="rs-key-state"></span>
    <button type="button" class="rs-forget-key" style="display:none">Clear</button>
  </div>
  <span class="rs-mode-switch" role="group" aria-label="Mode">
    <button type="button" data-mode="preview">Preview</button><button type="button" data-mode="active">Active</button>
  </span>
</div>`;
    modeHost.querySelector('.rs-mode-switch').addEventListener('click', e => {
      const b = e.target.closest('button');
      if (b) setMode(b.dataset.mode);
    });
    controls.innerHTML = `
<div class="rs-run-label rs-smote-label">SMOTE oversampling <span class="rs-run-label-note">– Adds synthetic examples of the rarer class to balance training</span></div>
<div class="rs-smote">
  <label><input type="radio" name="rs_smote" value="" checked> Without and with SMOTE</label>
  <label><input type="radio" name="rs_smote" value="0"> Without SMOTE</label>
  <label><input type="radio" name="rs_smote" value="1"> With SMOTE</label>
</div>
<div class="rs-free-note" style="display:none"></div>
<div class="rs-run-hint" hidden>Change to <a href="#" class="rs-go-active">active mode</a> to generate output.</div>
<div class="rs-run-actions">
  <button type="button" class="rs-run-button rs-btn-shape">Run Models</button>
  <a class="rs-output-link" target="_blank" rel="noopener" hidden></a>
  <span class="rs-output-none" hidden></span>
  <span class="rs-run-status"></span>
</div>
<div class="rs-github-status" hidden></div>
<div class="rs-report-dest"></div>`;
    controls.querySelector('.rs-run-button').addEventListener('click', runModels);
    controls.querySelector('.rs-go-active').addEventListener('click', e => {
      e.preventDefault();
      setMode('active');
    });
    const keyInput = modeHost.querySelector('.rs-key-input');
    keyInput.value = savedKey();
    // Check the passphrase and save it in this browser when valid: on paste, on Enter, and when the field
    // changes. Enter is handled on the input itself, because the form also holds the hidden username
    // field and so doesn't submit on Enter.
    const storeKey = () => checkKey(true);
    keyInput.addEventListener('change', storeKey);
    keyInput.addEventListener('paste', () => setTimeout(storeKey, 0)); // after the pasted text is in the field
    keyInput.addEventListener('keydown', e => {
      if (e.key !== 'Enter') return;
      e.preventDefault();
      storeKey();
    });
    modeHost.querySelector('.rs-key-field').addEventListener('submit', e => {
      e.preventDefault();
      storeKey();
    });
    modeHost.querySelector('.rs-forget-key').addEventListener('click', () => {
      rsConfirm('Delete your saved passphrase?', () => {
        saveKey('');
        keyInput.value = '';
        checkKey(false);
      });
    });
    // The models picker draws its checkboxes after this, and param-input.js adds Settings fields
    document.addEventListener('rsModelSelectRendered', applyKeyMode);
    document.addEventListener('rsModelsChanged', updateModelState);
    // param-input.js adds rows to the Settings section after loading, which also need locking
    const section = document.querySelector('[data-rs-layout]');
    if (section && typeof MutationObserver !== 'undefined') {
      new MutationObserver(() => { if (lockedNow) lockSettings(true); }).observe(section, { childList: true });
    }
    if ($('rsResults')) {
      $('rsResults').innerHTML = '<h2>Results</h2><div class="rs-results-body"></div>';
      $('rsResults').style.display = 'none';
    }
    initLayoutSwitch();
    applyKeyMode();
    ready.then(refreshUsage).then(() => checkKey(false));
  }

  // In-page confirmation with Yes / No buttons (instead of the browser's confirm dialog)
  function rsConfirm(message, onYes) {
    const backdrop = document.createElement('div');
    backdrop.className = 'rs-confirm-backdrop';
    backdrop.innerHTML = '<div class="rs-confirm-box" role="dialog" aria-modal="true">'
      + '<div class="rs-confirm-message"></div><div class="rs-confirm-actions">'
      + '<button type="button" class="rs-confirm-yes">Yes</button><button type="button" class="rs-confirm-no">No</button></div></div>';
    backdrop.querySelector('.rs-confirm-message').textContent = message;
    const close = () => { backdrop.remove(); document.removeEventListener('keydown', onKey); };
    const onKey = e => { if (e.key === 'Escape') close(); };
    backdrop.querySelector('.rs-confirm-yes').addEventListener('click', () => { close(); onYes(); });
    backdrop.querySelector('.rs-confirm-no').addEventListener('click', close);
    backdrop.addEventListener('click', e => { if (e.target === backdrop) close(); });
    document.addEventListener('keydown', onKey);
    document.body.appendChild(backdrop);
    backdrop.querySelector('.rs-confirm-no').focus();
  }

  const LAYOUT_STORE = 'realitystreamLayout';

  // Expand / Condense switch at the right of the Settings h2 ([data-rs-layout])
  function initLayoutSwitch() {
    const section = document.querySelector('[data-rs-layout]');
    const heading = section && section.querySelector(':scope > h2');
    if (!heading || heading.querySelector('.rs-layout-switch')) return;
    let layout = 'expansive';
    try { layout = localStorage.getItem(LAYOUT_STORE) || 'expansive'; } catch (e) {}
    const sw = document.createElement('span');
    sw.className = 'rs-layout-switch';
    sw.setAttribute('role', 'group');
    sw.setAttribute('aria-label', 'Layout');
    sw.innerHTML = '<button type="button" data-layout="expansive">Expand</button><button type="button" data-layout="condensed">Condense</button>';
    heading.appendChild(sw);
    const apply = value => {
      layout = value === 'condensed' ? 'condensed' : 'expansive';
      section.classList.toggle('rs-layout-expansive', layout === 'expansive');
      section.classList.toggle('rs-layout-condensed', layout === 'condensed');
      sw.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.layout === layout ? 'true' : 'false'));
    };
    sw.addEventListener('click', e => {
      const b = e.target.closest('button');
      if (!b) return;
      apply(b.dataset.layout);
      try { localStorage.setItem(LAYOUT_STORE, layout); } catch (e2) {}
    });
    apply(layout);
  }

  window.RSRunPanel = { refreshUsage, apiBase, ready, setMode, mode: () => mode, isLocked: () => lockedNow };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
