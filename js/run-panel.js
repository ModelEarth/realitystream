// Shared "Run Models" panel for RealityStream pages: today's usage, SMOTE choice, Run button and results.
// Used by realitystream/models/ and the Cloud Run home page (realitystream/models/home.html).
//
// Runs go to the RealityStream Cloud Run API (models/main.py): this site when the page is served by
// Cloud Run, otherwise RS_RUN_API below (or data-api on #rsRunControls). The YAML posted to /run is
// the text of #paramText pre, which param-input.js and model-select.js keep up to date.
//
// Containers (each optional except #rsRunControls):
//   #rsUsage        today's run time and cost
//   #rsRunControls  SMOTE choice, API key, Run button and status
//
// Without an API key the service allows 1 model, once per day, so the models picker switches to
// single-model mode and a note explains how to run more. A key typed into the API key field is
// remembered in this browser (localStorage) until "Forget key".
//   #rsResults      results table after a run
//   #rsEndpoints    list of API endpoints

(function () {
  const RS_RUN_API = 'https://realitystream-kwr4qrkopq-uc.a.run.app/';

  const $ = id => document.getElementById(id);

  function apiBase() {
    const el = $('rsRunControls');
    if (el && el.dataset.api) return new URL(el.dataset.api, location.href).href.replace(/\/?$/, '/');
    return /\.run\.app$/.test(location.hostname) ? location.origin + '/' : RS_RUN_API;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  const STYLE = `
.rs-usage-bar { height:10px; background:rgba(127,127,127,.2); border-radius:6px; overflow:hidden; margin:10px 0 6px; }
.rs-usage-fill { height:100%; width:0; background:#2f6fde; transition:width .3s; }
.rs-usage-fill.full { background:#c2410c; }
.rs-usage-stats { display:flex; flex-wrap:wrap; gap:8px 24px; opacity:.85; }
.rs-run-label { font-weight:600; margin:16px 0 6px; }
.rs-smote { display:flex; flex-wrap:wrap; gap:6px 20px; }
.rs-smote label { display:inline-flex; gap:6px; align-items:center; cursor:pointer; }
.rs-run-actions { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-top:18px; }
.rs-run-button { background:#2f6fde; color:#fff; border:0; border-radius:8px; padding:10px 22px; font-size:16px; cursor:pointer; }
.rs-run-button:disabled { opacity:.6; cursor:default; }
.rs-run-status.error { color:#c2410c; } .rs-run-status.ok { color:#15803d; }
.dark .rs-run-status.error { color:#fb923c; } .dark .rs-run-status.ok { color:#4ade80; }
.rs-free-note { margin-top:14px; opacity:.85; }
.rs-forget-key { background:none; border:0; padding:0; color:inherit; text-decoration:underline; cursor:pointer; font:inherit; opacity:.8; }
.rs-key-state { opacity:.75; }
.rs-api-key input { padding:6px 8px; border-radius:8px; border:1px solid rgba(127,127,127,.4); background:transparent; color:inherit; width:260px; max-width:100%; }
.rs-table-wrap { overflow-x:auto; }
.rs-table { width:100%; border-collapse:collapse; font-size:14px; }
.rs-table th, .rs-table td { text-align:left; padding:7px 10px 7px 0; border-top:1px solid rgba(127,127,127,.25); white-space:nowrap; }
.rs-table th { opacity:.75; font-weight:600; }
.rs-table td.wrap { white-space:normal; }
`;

  let info = null;
  let keyValid = false;
  let keyChecked = false; // Limits apply only after the saved key is checked, so a key holder's models aren't trimmed
  const KEY_STORE = 'realitystreamApiKey';

  function savedKey() {
    try { return localStorage.getItem(KEY_STORE) || ''; } catch (e) { return ''; }
  }
  function saveKey(value) {
    try { value ? localStorage.setItem(KEY_STORE, value) : localStorage.removeItem(KEY_STORE); } catch (e) {}
  }

  function moreRunsHtml() {
    const url = (info && info.more_runs_url) || 'https://model.earth/cloud/run/#googleaccountautomation';
    return `To run more, <a href="${url}" target="_blank" rel="noopener">add a REALITYSTREAM_API_KEY</a> for this Google Cloud project, `
      + `or <a href="${url}" target="_blank" rel="noopener">run on your own cloud account</a>.`;
  }

  // Show the limits that apply: with a valid key any number of models, without one 1 model once per day
  function applyKeyMode() {
    const limited = !!(info && info.api_key_enabled) && !keyValid;
    if (window.RSModelSelect && RSModelSelect.setSingle) RSModelSelect.setSingle(limited);
    const note = document.querySelector('#rsRunControls .rs-free-note');
    if (note) {
      note.style.display = limited ? 'block' : 'none';
      note.innerHTML = 'Without an API key, you can run 1 model once per day. ' + moreRunsHtml();
    }
    const forget = document.querySelector('#rsRunControls .rs-forget-key');
    if (forget) forget.style.display = savedKey() ? 'inline' : 'none';
    const keyState = document.querySelector('#rsRunControls .rs-key-state');
    if (keyState) {
      const typed = document.querySelector('#rsRunControls .rs-api-key input').value;
      keyState.textContent = !typed ? '' : keyValid ? 'Key accepted' : 'Key not accepted';
    }
  }

  // Ask the service whether the key in the API key field is valid
  async function checkKey() {
    const value = document.querySelector('#rsRunControls .rs-api-key input').value.trim();
    let valid = false;
    if (value) {
      try {
        const r = await fetch(apiBase() + 'key', { headers: { 'X-API-Key': value }, cache: 'no-store' }).then(r => r.json());
        valid = r.key === 'valid';
      } catch (e) {}
    }
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
    const keyRow = document.querySelector('#rsRunControls .rs-api-key');
    if (keyRow) keyRow.style.display = info.api_key_enabled ? 'flex' : 'none';
    if (keyChecked) applyKeyMode();
    if ($('rsEndpoints')) {
      $('rsEndpoints').innerHTML = '<div class="rs-table-wrap"><table class="rs-table">'
        + Object.entries(info.endpoints || {}).map(([k, v]) =>
          `<tr><td><code>${escapeHtml(k)}</code></td><td class="wrap">${escapeHtml(v)}</td></tr>`).join('')
        + '</table></div>';
    }
  }

  function setStatus(text, kind, html) {
    const el = document.querySelector('#rsRunControls .rs-run-status');
    if (html) el.innerHTML = html; else el.textContent = text;
    el.className = 'rs-run-status ' + (kind || '');
  }

  const pct = v => (v == null ? '' : (v <= 1 ? (v * 100).toFixed(1) : Number(v).toFixed(1)) + '%');

  function showResults(data) {
    const host = $('rsResults');
    if (!host) return;
    const rows = [];
    [['no_smote', 'Without SMOTE'], ['smote', 'With SMOTE']].forEach(([key, label]) => {
      (data[key] || []).forEach(r => rows.push(
        `<tr><td>${escapeHtml(r.model_type || r.model || '')}</td><td>${label}</td><td>${pct(r.accuracy)}</td>`
        + `<td>${pct(r.roc_auc)}</td><td>${pct(r.f1_score)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td>`
        + `<td>${r.time != null ? Number(r.time).toFixed(1) + ' s' : ''}</td></tr>`));
    });
    host.querySelector('.rs-results-body').innerHTML = '<div class="rs-table-wrap"><table class="rs-table">'
      + '<tr><th>Model</th><th>Training</th><th>Accuracy</th><th>ROC-AUC</th><th>F1</th><th>Precision</th><th>Recall</th><th>Time</th></tr>'
      + (rows.join('') || '<tr><td colspan="8">No models were trained.</td></tr>') + '</table></div>';
    host.style.display = 'block';
  }

  async function runModels() {
    const pre = document.querySelector('#paramText pre') || $('paramText');
    const yaml = pre ? pre.textContent.trim() : '';
    if (!yaml) {
      setStatus('Add parameters first.', 'error');
      return;
    }
    const smote = (document.querySelector('#rsRunControls input[name="rs_smote"]:checked') || {}).value || '';
    const url = apiBase() + 'run' + (smote ? '?smote=' + smote : '');
    const headers = { 'Content-Type': 'text/yaml' };
    const key = document.querySelector('#rsRunControls .rs-api-key input');
    if (key && key.value) headers['X-API-Key'] = key.value;

    const button = document.querySelector('#rsRunControls .rs-run-button');
    button.disabled = true;
    const started = Date.now();
    const timer = setInterval(() => setStatus(`Running… ${Math.round((Date.now() - started) / 1000)} s`), 1000);
    setStatus('Running…');
    try {
      const resp = await fetch(url, { method: 'POST', headers, body: yaml });
      const data = await resp.json().catch(() => ({}));
      const secs = Math.round((Date.now() - started) / 1000);
      if (resp.ok && data.status === 'success') {
        showResults(data);
        setStatus(`Finished in ${secs} s.`, 'ok');
      } else if (data.status === 'missing_key') {
        setStatus(`${data.key} is needed. ${data.how_to_get_it || ''}`, 'error');
      } else if (data.more_runs_url) {
        // Limits without an API key: link the ways to run more
        const msg = escapeHtml((data.message || '').replace(/To run more,.*$/, '').trim());
        setStatus('', 'error', msg + ' ' + moreRunsHtml());
      } else if (resp.status === 401) {
        setStatus('The API key was not accepted. Check it, or choose Forget key to run without one.', 'error');
      } else {
        setStatus(data.message || `Run failed (HTTP ${resp.status}).`, 'error');
      }
    } catch (e) {
      setStatus('The service did not respond. Long runs can take several minutes; check today\'s totals and try again.', 'error');
    } finally {
      clearInterval(timer);
      button.disabled = false;
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
      $('rsUsage').innerHTML = '<h2>Today\'s model run time (<span class="rs-usage-day">…</span>, resets at midnight Eastern)</h2>'
        + '<div class="rs-usage-bar"><div class="rs-usage-fill"></div></div><div class="rs-usage-stats">Loading…</div>';
    }
    controls.innerHTML = `
<div class="rs-run-label">SMOTE oversampling</div>
<div class="rs-smote">
  <label><input type="radio" name="rs_smote" value="" checked> Without and with SMOTE</label>
  <label><input type="radio" name="rs_smote" value="0"> Without SMOTE</label>
  <label><input type="radio" name="rs_smote" value="1"> With SMOTE</label>
</div>
<div class="rs-run-actions rs-api-key" style="display:none">
  <label>API key <input type="password" autocomplete="off" placeholder="Optional"></label>
  <span class="rs-key-state"></span>
  <button type="button" class="rs-forget-key" style="display:none">Forget key</button>
</div>
<div class="rs-free-note" style="display:none"></div>
<div class="rs-run-actions">
  <button type="button" class="rs-run-button">Run Models</button>
  <span class="rs-run-status"></span>
</div>`;
    controls.querySelector('.rs-run-button').addEventListener('click', runModels);
    const keyInput = controls.querySelector('.rs-api-key input');
    keyInput.value = savedKey();
    // Remember a typed key in this browser, then check it
    keyInput.addEventListener('change', () => { saveKey(keyInput.value.trim()); checkKey(); });
    controls.querySelector('.rs-forget-key').addEventListener('click', () => {
      saveKey('');
      keyInput.value = '';
      checkKey();
    });
    if ($('rsResults')) {
      $('rsResults').innerHTML = '<h2>Results</h2><div class="rs-results-body"></div>';
      $('rsResults').style.display = 'none';
    }
    refreshUsage().then(checkKey);
  }

  window.RSRunPanel = { refreshUsage, apiBase };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
