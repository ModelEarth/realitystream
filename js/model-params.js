// Common Settings panel for RealityStream pages: the YAML's split, search and hyperparameters sections.
// Used by realitystream/models/ and the Cloud Run home page (realitystream/api/home.html).
//
// The fields come from the API's GET /model-params (api/model-params.yaml), the file the API validates
// runs against, so this panel and the service agree. When the service has no such endpoint (an older
// deploy), the panel stays hidden.
//
// The panel edits the YAML in #paramText pre, as model-select.js does for models:, and copies each
// changed value into the URL hash (for example #hyperparameters.rfc.n_estimators=200), so a shared link
// keeps it. Only values that differ from the defaults are written, and "Default" removes one again.
// Editing the YAML by hand updates the fields.
//
// Settings shared by more than one model (split, and search for xgboost and mlp) stay in the panel. Each
// model's hyperparameters go in that model's details in the Models panel (model-select.js's
// .rs-model-params slot, shown by the "N Parameters" button after its description).
//
// In Active mode without the Team Passphrase every field is greyed out. run-panel.js sends rsModeChange
// when that changes, and rsParamError when /run rejects a value, which this panel marks on that field.
//
// Container: #modelParams, a panel section (the Panel Menu Toggle goes at its upper right).
// Needs js-yaml, getHash and goHashQuiet from localsite.js, and run-panel.js loaded first (RSRunPanel).

(function () {
  const $ = id => document.getElementById(id);
  const SECTIONS = ['split', 'search', 'hyperparameters'];
  const ALIASES = { logisticregression: 'lr', randomforest: 'rfc', randombitsforest: 'rbf', xgb: 'xgboost' };
  // Form layout: fields whose help begins beside the field name, after a dash. The first sentence of the help
  // goes there and the rest stays below the input, with either the service's help text or a newer one.
  // When nothing is left below, the range goes beside the name too.
  const BESIDE_NAME = ['hyperparameters.lr.C', 'hyperparameters.svm.C', 'hyperparameters.lr.l1_ratio'];
  // Fields whose "Used when ..." note (shown while another field's value makes them inactive) goes beside the
  // field name, after a dash, instead of below the input
  const NOTE_BESIDE_NAME = ['hyperparameters.svm.degree'];
  // Fields the form words its own way, in place of the service's help and after text: beside the field name
  // (after a dash) and below the input (with the range)
  const LAYOUT_TEXT = {
    'search.n_iter': { beside: 'RandomizedSearchCV tries for XGBoost and MLP', below: 'Each with 5-fold cross-validation.' },
  };
  const SOURCE_URL = 'https://github.com/ModelEarth/realitystream/blob/main/api/model-params.yaml';

  let schema = null;     // GET /model-params
  let locked = false;    // Active mode without the Team Passphrase (run-panel.js)
  let paramError = null; // {field, message} from the last rejected run

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // Escaped text with each short parenthetical phrase, such as "(0.001 to 1000)", kept on one line
  function textHtml(s) {
    return escapeHtml(s).replace(/\([^()]{1,32}\)/g, '<span class="rs-nowrap">$&</span>');
  }

  const STYLE = `
#modelParams { position:relative; }
#modelParams > h2 { padding-right:40px; }
.rs-mp-menu { position:absolute; top:14px; right:14px; }
.rs-mp-note { margin:0 0 12px; opacity:.85; }
.rs-mp-note.warn { opacity:1; color:#c2410c; } .dark .rs-mp-note.warn { color:#fb923c; }
.rs-mp-intro { margin:0 0 12px; opacity:.85; }
/* 8px more between a model's bullets and its fields */
.rs-model-params { margin:10px 0 4px; padding-top:8px; }
/* At most 2 columns (each at least half the width less the gap), 1 when narrower than 2 × 210px */
.rs-mp-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(max(210px, calc((100% - 20px) / 2)), 1fr)); gap:14px 20px; }
.rs-mp-field { min-width:0; }
/* The field name above the input (for the BESIDE_NAME fields, such as C, followed by "– " and their help's first sentence) */
.rs-mp-top { margin-bottom:4px; font-size:13px; }
.rs-mp-name { font-weight:600; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; }
.rs-mp-brief { font-weight:400; font-size:13px; opacity:.8; }
/* Search / Fixed and Default, at the right of the input */
.rs-mp-mode { display:inline-flex; flex:none; border:1px solid rgba(127,127,127,.35); border-radius:6px; overflow:hidden; font-size:12px; }
.rs-mp-mode button { background:transparent; color:inherit; border:0; padding:5px 8px; cursor:pointer; font:inherit; }
.rs-mp-mode button[aria-pressed="true"] { background:#2f6fde; color:#fff; }
.rs-mp-row { display:flex; align-items:center; gap:8px; }
.rs-mp-row input, .rs-mp-row select { flex:1; min-width:0; padding:6px 8px; border-radius:8px; border:1px solid rgba(127,127,127,.4);
  background:transparent; color:inherit; font:inherit; font-size:14px; }
.dark .rs-mp-row select option { background:#1d2128; }
.rs-mp-row input:disabled, .rs-mp-row select:disabled { opacity:.55; }
.rs-mp-field.changed .rs-mp-row input, .rs-mp-field.changed .rs-mp-row select { border-color:#2f6fde; }
.rs-mp-field.error .rs-mp-row input, .rs-mp-field.error .rs-mp-row select { border-color:#c2410c; }
.rs-mp-default { flex:none; background:transparent; border:1px solid rgba(127,127,127,.35); border-radius:6px; padding:5px 10px;
  color:inherit; cursor:pointer; font:inherit; font-size:12px; }
.rs-mp-default:hover:not(:disabled) { background:rgba(127,127,127,.15); }
/* Below the input, each on its own line: the help, the rest of the description with the range, then why the
   field is inactive or searched, or an error */
.rs-mp-helptext { opacity:.75; }
.rs-mp-after { font-size:13px; margin-top:4px; }
/* Short parenthetical phrases, such as a field's range, stay on one line */
.rs-mp-range, .rs-nowrap { white-space:nowrap; }
/* A field's explain note, on its own row after the field */
.rs-mp-explain { grid-column:1 / -1; margin:0; font-size:13px; }
/* As light as the help text; an error stays full strength */
.rs-mp-msg { opacity:.75; }
.rs-mp-msg:empty { display:none; }
.rs-mp-field.error .rs-mp-msg { color:#c2410c; opacity:1; } .dark .rs-mp-field.error .rs-mp-msg { color:#fb923c; }
`;

  // ---- YAML ----

  function pre() {
    return document.querySelector('#paramText pre');
  }

  function readYaml() {
    const el = pre();
    if (!el || typeof jsyaml === 'undefined') return null;
    try {
      const obj = jsyaml.load(el.textContent || '');
      return obj && typeof obj === 'object' ? obj : {};
    } catch (e) {
      return null; // The YAML is being edited and isn't valid yet
    }
  }

  function canonical(model) {
    const key = String(model).trim().toLowerCase();
    return ALIASES[key] || key;
  }

  // The YAML's split, search and hyperparameters, with model aliases (xgb) turned into keys
  function settingsOf(obj) {
    const out = { split: {}, search: {}, hyperparameters: {} };
    ['split', 'search'].forEach(s => { if (obj && obj[s] && typeof obj[s] === 'object') Object.assign(out[s], obj[s]); });
    const hyper = obj && obj.hyperparameters;
    if (hyper && typeof hyper === 'object') {
      Object.entries(hyper).forEach(([model, values]) => {
        if (values && typeof values === 'object') out.hyperparameters[canonical(model)] = Object.assign({}, values);
      });
    }
    return out;
  }

  function getAt(settings, path) {
    const [section, a, b] = path;
    return section === 'hyperparameters' ? (settings.hyperparameters[a] || {})[b] : settings[section][a];
  }

  // Set (or with undefined, remove) one value in the YAML, and mirror it in the hash
  function writeValue(path, value) {
    const el = pre();
    const obj = readYaml();
    if (!el || !obj) {
      setNote('Fix the YAML below first; it isn\'t valid YAML right now.', true);
      return false;
    }
    const settings = settingsOf(obj);
    const [section, a, b] = path;
    if (section === 'hyperparameters') {
      const model = settings.hyperparameters[a] || {};
      if (value === undefined) delete model[b]; else model[b] = value;
      if (Object.keys(model).length) settings.hyperparameters[a] = model; else delete settings.hyperparameters[a];
    } else if (value === undefined) {
      delete settings[section][a];
    } else {
      settings[section][a] = value;
    }
    SECTIONS.forEach(s => {
      if (Object.keys(settings[s]).length) obj[s] = settings[s]; else delete obj[s];
    });
    el.textContent = typeof convertToYAML === 'function' ? convertToYAML(obj) : jsyaml.dump(obj, { lineWidth: -1 });
    if (typeof updateResetButtonVisibility === 'function') updateResetButtonVisibility();

    const key = path.join('.');
    if (typeof goHashQuiet === 'function') {
      if (value === undefined) goHashQuiet({}, [key]);
      else goHashQuiet({ [key]: Array.isArray(value) ? value.join(',') : value });
    }
    return true;
  }

  // After a hand edit of the YAML, make the hash's settings match it (so a later hash change can't undo it)
  function syncHashFromYaml() {
    const obj = readYaml();
    if (!obj || typeof getHash !== 'function' || typeof goHashQuiet !== 'function') return;
    const remove = Object.keys(getHash()).filter(k => SECTIONS.includes(k.split('.')[0]));
    const add = {};
    const settings = settingsOf(obj);
    ['split', 'search'].forEach(s => Object.entries(settings[s]).forEach(([k, v]) => { add[`${s}.${k}`] = v; }));
    Object.entries(settings.hyperparameters).forEach(([m, values]) =>
      Object.entries(values).forEach(([k, v]) => { add[`hyperparameters.${m}.${k}`] = Array.isArray(v) ? v.join(',') : v; }));
    goHashQuiet(add, remove.filter(k => !(k in add)));
  }

  // ---- Fields ----

  // Fields in this panel and in the models' details
  function allFields() {
    return document.querySelectorAll('#modelParams .rs-mp-field, .rs-model-params .rs-mp-field');
  }

  function groupFields(path) {
    return path[0] === 'hyperparameters' ? schema.hyperparameters[path[1]] : schema[path[0]];
  }

  function rangeText(spec) {
    if (spec.type === 'choice') return '';
    if (spec.type === 'layers') return `1 to 3 layers of ${spec.min} to ${spec.max}`;
    return `${spec.min} to ${spec.max}`;
  }

  function fieldHtml(path, spec) {
    const id = 'rsmp-' + path.join('-');
    let control;
    if (spec.type === 'choice') {
      control = `<select id="${id}">` + spec.options.map(o => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join('') + '</select>';
    } else if (spec.type === 'layers') {
      control = `<input id="${id}" type="text" inputmode="numeric" placeholder="100 or 50, 50">`;
    } else {
      const step = spec.step || (spec.type === 'int' ? 1 : 'any');
      control = `<input id="${id}" type="number" min="${spec.min}" max="${spec.max}" step="${step}">`;
    }
    const mode = spec.search
      ? '<span class="rs-mp-mode" role="group" aria-label="Search or fixed"><button type="button" data-mode="search">Search</button><button type="button" data-mode="fixed">Fixed</button></span>'
      : '';
    const range = rangeText(spec);
    const rangeHtml = range ? `<span class="rs-mp-range">(${escapeHtml(range)})</span>` : '';
    let help = spec.help || '', after = spec.after, beside = '';
    const own = LAYOUT_TEXT[path.join('.')];
    if (own) {
      [beside, help, after] = [own.beside, own.below, ''];
    } else if (BESIDE_NAME.includes(path.join('.')) && help) {
      beside = (help.match(/^.*?\.(?=\s|$)/) || [help])[0];
      help = help.slice(beside.length).trim();
    }
    // Lines below the input: the help, then the after text; the range ends the last of them
    const lines = [help, after].filter(Boolean).map(textHtml);
    beside = beside ? textHtml(beside) : '';
    if (rangeHtml && beside && !lines.length) beside += ' ' + rangeHtml;
    else if (rangeHtml) lines.length ? (lines[lines.length - 1] += ' ' + rangeHtml) : lines.push(rangeHtml);
    return `<div class="rs-mp-field" data-path="${path.join('.')}">
  <div class="rs-mp-top"><label class="rs-mp-name" for="${id}">${escapeHtml(path[path.length - 1])}</label>${beside ? ` <span class="rs-mp-brief">– ${beside}</span>` : ''}${NOTE_BESIDE_NAME.includes(path.join('.')) ? ' <span class="rs-mp-brief rs-mp-note-beside"></span>' : ''}</div>
  <div class="rs-mp-row">${control}${mode}<button type="button" class="rs-mp-default" title="Remove from the YAML, so the default applies">Default</button></div>
  <div class="rs-mp-after">${lines.map(line => `<div class="rs-mp-helptext">${line}</div>`).join('')}<div class="rs-mp-msg"></div></div>
</div>`;
  }

  // A grid of fields; prefixes such as ['split'] or ['hyperparameters', 'rfc'] name the fields' sections
  function gridHtml(prefixes) {
    const fields = prefixes.flatMap(prefix => Object.entries(groupFields(prefix)).map(([name, spec]) => fieldHtml(prefix.concat(name), spec)
      + (spec.explain ? `<p class="rs-mp-explain">${textHtml(spec.explain)}</p>` : '')));
    return `<div class="rs-mp-grid">${fields.join('')}</div>`;
  }

  // Split and search in this panel, and each model's hyperparameters in its details (when not there yet)
  function build() {
    const host = $('modelParams');
    if (!host || !schema) return;
    const body = host.querySelector('.rs-mp-body');
    if (!body.querySelector('.rs-mp-field')) {
      body.innerHTML = gridHtml([['split'], ['search']]);
      body.querySelectorAll('.rs-mp-field').forEach(bindField);
    }
    document.querySelectorAll('.rs-model-params[data-model]').forEach(slot => {
      const key = canonical(slot.dataset.model);
      if (!schema.hyperparameters[key] || slot.querySelector('.rs-mp-field')) return;
      slot.innerHTML = gridHtml([['hyperparameters', key]]);
      slot.querySelectorAll('.rs-mp-field').forEach(bindField);
    });
    fill();
    // model-select.js counts each model's fields for its "N Parameters" button
    document.dispatchEvent(new Event('rsModelParamsBuilt'));
  }

  function fieldPath(field) {
    return field.dataset.path.split('.');
  }

  function controlOf(field) {
    return field.querySelector('input, select');
  }

  // Parse and range-check a control's text; returns {value} or {error}
  function parseControl(spec, text) {
    text = String(text).trim();
    if (spec.type === 'choice') return { value: text };
    if (spec.type === 'layers') {
      const sizes = text.split(/[\s,]+/).filter(Boolean).map(Number);
      if (!sizes.length || sizes.length > 3 || sizes.some(n => !Number.isInteger(n) || n < spec.min || n > spec.max)) {
        return { error: `Give 1 to 3 whole numbers from ${spec.min} to ${spec.max}, such as 100 or 50, 50.` };
      }
      return { value: sizes };
    }
    if (text === '') return { error: 'Enter a number.' };
    const n = Number(text);
    if (!Number.isFinite(n) || (spec.type === 'int' && !Number.isInteger(n))) {
      return { error: spec.type === 'int' ? 'Enter a whole number.' : 'Enter a number.' };
    }
    if (n < spec.min || n > spec.max) return { error: `Use ${spec.min} to ${spec.max}.` };
    return { value: n };
  }

  function sameValue(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
  }


  function bindField(field) {
    const path = fieldPath(field);
    const spec = groupFields(path)[path[path.length - 1]];
    const control = controlOf(field);
    control.addEventListener('change', () => {
      const parsed = parseControl(spec, control.value);
      field.dataset.error = parsed.error || '';
      if (paramError && paramError.field === field.dataset.path) paramError = null;
      if (parsed.error) {
        fill();
        return;
      }
      // A value equal to the default is removed, except a searched field set to Fixed, which keeps it
      const value = !spec.search && sameValue(parsed.value, spec.default) ? undefined : parsed.value;
      if (writeValue(path, value)) fill();
    });
    field.querySelector('.rs-mp-default').addEventListener('click', () => {
      field.dataset.error = '';
      if (writeValue(path, undefined)) fill();
    });
    field.querySelectorAll('.rs-mp-mode button').forEach(button => button.addEventListener('click', () => {
      if (locked) return;
      if (button.dataset.mode === 'search') {
        if (writeValue(path, undefined)) fill();
      } else {
        const parsed = parseControl(spec, control.value);
        if (writeValue(path, parsed.error ? spec.default : parsed.value)) fill();
      }
    }));
  }

  function setNote(text, warn) {
    const note = document.querySelector('#modelParams .rs-mp-note');
    if (!note) return;
    note.textContent = text || '';
    note.style.display = text ? 'block' : 'none';
    note.classList.toggle('warn', !!warn);
  }

  // Show the YAML's values (or the defaults) in the fields, and which options and fields apply
  function fill() {
    const host = $('modelParams');
    if (!host || !schema) return;
    const obj = readYaml();
    const settings = settingsOf(obj || {});
    const hasSettings = SECTIONS.some(s => Object.keys(settings[s]).length);
    if (locked && hasSettings) {
      setNote('This YAML changes model settings, which need the Team Passphrase. Enter it at the top of the page, or choose Reset to defaults from this panel\'s menu.', true);
    } else if (locked) {
      setNote('Enter the Team Passphrase at the top of the page to change model settings. Runs without it use the defaults.');
    } else {
      setNote(obj ? '' : 'The YAML below isn\'t valid right now, so these fields show defaults.', !obj);
    }

    allFields().forEach(field => {
      const path = fieldPath(field);
      const fields = groupFields(path);
      const name = path[path.length - 1];
      const spec = fields[name];
      const control = controlOf(field);
      const set = getAt(settings, path);
      const value = set === undefined ? spec.default : set;
      const effective = other => {
        const v = getAt(settings, path.slice(0, -1).concat(other));
        return v === undefined ? fields[other].default : v;
      };

      if (document.activeElement !== control) {
        control.value = Array.isArray(value) ? value.join(', ') : value;
      }
      const searched = !!spec.search && set === undefined;
      field.dataset.mode = spec.search ? (searched ? 'search' : 'fixed') : '';
      field.querySelectorAll('.rs-mp-mode button').forEach(b => {
        b.setAttribute('aria-pressed', b.dataset.mode === field.dataset.mode ? 'true' : 'false');
        b.disabled = locked;
      });
      field.classList.toggle('changed', set !== undefined);
      // Shown only for a value set in the YAML; otherwise the input fills the column
      field.querySelector('.rs-mp-default').style.display = set !== undefined ? '' : 'none';

      // only_with: the field applies only when another field has certain values
      let inactiveReason = '';
      Object.entries(spec.only_with || {}).forEach(([other, allowed]) => {
        if (!allowed.includes(effective(other))) inactiveReason = `Used when ${other} is ${allowed.join(' or ')}.`;
      });

      // valid_with: grey out options that don't work with another field's value, in both directions
      if (spec.type === 'choice') {
        Array.from(control.options).forEach(option => { option.disabled = false; option.title = ''; });
        Object.entries(spec.valid_with || {}).forEach(([other, table]) => {
          const allowed = table[effective(other)];
          Array.from(control.options).forEach(option => {
            if (allowed && !allowed.includes(option.value)) {
              option.disabled = true;
              const works = Object.keys(table).filter(k => table[k].includes(option.value));
              option.title = `Needs ${other} ${works.join(' or ')}`;
            }
          });
        });
        Object.entries(fields).forEach(([otherName, otherSpec]) => {
          const table = (otherSpec.valid_with || {})[name];
          if (!table) return;
          const current = effective(otherName);
          Array.from(control.options).forEach(option => {
            if (table[option.value] && !table[option.value].includes(current)) {
              option.disabled = true;
              option.title = `Doesn't support ${otherName} ${current}`;
            }
          });
        });
      }

      control.disabled = locked || searched || !!inactiveReason;
      field.querySelector('.rs-mp-default').disabled = locked;
      // A value the service rejected, or one typed out of range, else why the field is inactive
      const error = (paramError && paramError.field === field.dataset.path ? paramError.message : '') || field.dataset.error || '';
      field.classList.toggle('error', !!error);
      const noteBeside = field.querySelector('.rs-mp-note-beside');
      if (noteBeside) noteBeside.textContent = inactiveReason ? '– ' + inactiveReason : '';
      field.querySelector('.rs-mp-msg').textContent = error || (noteBeside ? '' : inactiveReason)
        || (searched ? 'Searched: RandomizedSearchCV picks the value.' : '');
      const below = field.querySelector('.rs-mp-after');
      below.hidden = !below.textContent.trim(); // No empty line below the input
    });
  }

  // Remove every setting from the YAML and the hash
  function resetAll() {
    const el = pre();
    const obj = readYaml();
    if (!el || !obj) return;
    SECTIONS.forEach(s => delete obj[s]);
    el.textContent = typeof convertToYAML === 'function' ? convertToYAML(obj) : jsyaml.dump(obj, { lineWidth: -1 });
    if (typeof updateResetButtonVisibility === 'function') updateResetButtonVisibility();
    if (typeof getHash === 'function' && typeof goHashQuiet === 'function') {
      goHashQuiet({}, Object.keys(getHash()).filter(k => SECTIONS.includes(k.split('.')[0])));
    }
    paramError = null;
    allFields().forEach(f => { f.dataset.error = ''; });
    fill();
  }

  function addMenu() {
    if (typeof addPanelMenu !== 'function' || $('modelParamsMenuControl')) return;
    addPanelMenu({
      panelType: 'Content',
      targetPanelId: 'modelParams',
      containerSelector: '#modelParams .rs-mp-menu',
      inline: true,
      menuItems: [
        { label: 'Reset to defaults', action: 'rsmp-reset', icon: 'restart_alt' },
        { label: 'About these settings', action: 'link', icon: 'info', href: SOURCE_URL },
      ],
      onAction: action => {
        if (action !== 'rsmp-reset') return false;
        resetAll(); // Also allowed without the Team Passphrase, to clear settings a shared link brought in
        return true;
      },
    }).render();
  }

  // ---- Start ----

  function init() {
    const host = $('modelParams');
    if (!host || !window.RSRunPanel) return;
    host.style.display = 'none';
    // After run-panel.js has checked for a local API
    Promise.resolve(RSRunPanel.ready).then(() => fetch(RSRunPanel.apiBase() + 'model-params', { cache: 'no-store' }))
      .then(r => (r.ok ? r.json() : null))
      .catch(() => null)
      .then(data => {
        if (!data || !data.hyperparameters) return; // An older service without /model-params
        schema = data;
        if (!document.getElementById('rsModelParamsStyle')) {
          const style = document.createElement('style');
          style.id = 'rsModelParamsStyle';
          style.textContent = STYLE;
          document.head.appendChild(style);
        }
        host.innerHTML = '<h2>Common Settings</h2><div class="rs-mp-menu"></div><p class="rs-mp-note"></p>'
          + '<p class="rs-mp-intro">Train/test split and search, for all models. Each model\'s hyperparameters are under Parameters, in the Models panel.</p>'
          + '<div class="rs-mp-body"></div>';
        host.style.display = '';
        addMenu();
        if (RSRunPanel.isLocked) locked = !!RSRunPanel.isLocked();
        build();

        // model-select.js draws the models' details (with their .rs-model-params slots) after loading README.md
        document.addEventListener('rsModelSelectRendered', build);
        document.addEventListener('rsModeChange', e => {
          locked = !!(e.detail && e.detail.locked);
          fill();
        });
        document.addEventListener('rsParamError', e => {
          paramError = e.detail && e.detail.field ? e.detail : null;
          fill();
          const field = paramError && Array.from(allFields()).find(f => f.dataset.path === paramError.field);
          if (!field) return;
          const slot = field.closest('.rs-model-params');
          if (slot && window.RSModelSelect && RSModelSelect.expand) RSModelSelect.expand(slot.dataset.model);
          if (field.scrollIntoView) field.scrollIntoView({ block: 'center', behavior: 'smooth' });
        });
        // param-input.js and model-select.js rewrite the YAML; a hand edit also updates the hash
        const el = pre();
        if (el && typeof MutationObserver !== 'undefined') {
          new MutationObserver(fill).observe(el, { childList: true, characterData: true, subtree: true });
        }
        const editor = $('paramText');
        if (editor) editor.addEventListener('input', syncHashFromYaml);
      });
  }

  window.RSModelParams = { refresh: () => fill() };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
