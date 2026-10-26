/* DYNAMITE Run Explorer.
 *
 * Data: data/manifest.js (DYNX.manifest([...]) with one summary per run) and
 * data/<run>/run.js (DYNX.add({...})), written by
 * scripts/dynamite_explorer/extract_run.py and build_manifest.py.
 * Script tags rather than fetch, so the page also works from file://.
 */
'use strict';

const $ = (sel, el = document) => el.querySelector(sel);
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
};
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 3) => {
  if (v == null || !isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
  return (+v.toFixed(d)).toString();
};
const fmtChi = v => (v == null || !isFinite(v)) ? '–' : v.toFixed(1);
const log10 = Math.log10;

/* ------------------------------------------------------------------ data */
function loadRun(name) {
  if (DYNX.runs[name]) return Promise.resolve(DYNX.runs[name]);
  return new Promise((resolve, reject) => {
    (DYNX._wait[name] = DYNX._wait[name] || []).push(resolve);
    if (!document.querySelector(`script[data-run="${CSS.escape(name)}"]`)) {
      const s = document.createElement('script');
      s.src = `data/${encodeURIComponent(name)}/run.js`;
      s.dataset.run = name;
      s.onerror = () => reject(new Error(`could not load data/${name}/run.js`));
      document.head.appendChild(s);
    }
  }).then(prepareRun);
}

const PAR_LABEL = {
  'm-bh': 'log M_BH [M☉]', 'a-bh': 'a_BH', 'f-dh': 'log f_DM = log M200/M*', 'c-dh': 'c_DM',
  'q-stars': 'q (C/A)', 'p-stars': 'p (B/A)', 'u-stars': 'u (σ_obs/σ_intr)', 'ml': 'M/L',
  theta: 'θ [deg]', phi: 'φ [deg]', psi: 'ψ [deg]',
};
const parLabel = n => PAR_LABEL[n] || n;
const CHI2S = ['kinmapchi2', 'kinchi2', 'chi2'];

function prepareRun(run) {
  if (run._prepared) return run;
  const cols = run.models.columns;
  run.ci = Object.fromEntries(cols.map((c, i) => [c, i]));
  run.parInfo = Object.fromEntries(run.parameters.map(p => [p.name, p]));
  run.freePars = run.parameters.filter(p => !p.fixed).map(p => p.name);
  run.allPars = run.parameters.map(p => p.name);
  run.done = run.models.rows.map(r => r[run.ci.all_done] === true || r[run.ci.all_done] === 'True');
  run.detailRows = Object.keys(run.details || {}).map(Number);
  run.sel = run.best_row;
  run.geomByName = Object.fromEntries((run.geometry || []).map(g => [g.name, decodeGeometry(g)]));
  run._prepared = true;
  return run;
}

// display value of a parameter (log10 for logarithmic parameters)
function pval(run, row, name) {
  const v = run.models.rows[row][run.ci[name]];
  if (v == null) return null;
  const info = run.parInfo[name];
  return info && info.logarithmic ? log10(v) : v;
}
const cval = (run, row, col) => run.models.rows[row][run.ci[col]];

function decodeGeometry(g) {
  const bytes = Uint8Array.from(atob(g.grid), c => c.charCodeAt(0));
  return Object.assign({}, g, {ids: new Uint16Array(bytes.buffer)});
}
function decodeOrbit(o) {
  const bytes = Uint8Array.from(atob(o.pts), c => c.charCodeAt(0));
  const q = new Int16Array(bytes.buffer);
  const f = new Float32Array(q.length);
  const s = o.scale / 32767;
  for (let i = 0; i < q.length; i++) f[i] = q[i] * s;
  return f;
}

function chiThreshold(run) {
  return run.n_obs2 ? Math.sqrt(run.n_obs2) : null;
}
function bestChi(run, which) {
  let best = Infinity;
  run.models.rows.forEach((r, i) => {
    const v = r[run.ci[which]];
    if (run.done[i] && v != null && v < best) best = v;
  });
  return best;
}

/* ------------------------------------------------------------- colormaps */
function hexToRgb(hx) {
  hx = hx.replace('#', '');
  return [0, 2, 4].map(i => parseInt(hx.slice(i, i + 2), 16));
}
function makeCmap(stops) {
  const rgb = stops.map(hexToRgb);
  return t => {
    t = Math.min(1, Math.max(0, isFinite(t) ? t : 0.5));
    const x = t * (rgb.length - 1), i = Math.min(rgb.length - 2, Math.floor(x)), f = x - i;
    return rgb[i].map((c, k) => Math.round(c + (rgb[i + 1][k] - c) * f));
  };
}
const SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
// low -> high magnitude; on the dark surface high values are the light end
const seqStops = () => isDark() ? SEQ.slice().reverse() : SEQ.slice();
function divStops() {
  const dark = isDark();
  return dark
    ? ['#9ec5f4', '#3987e5', '#1c5cab', '#383835', '#a83a3a', '#e66767', '#f4b3b3']
    : ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f4b3b3', '#e34948', '#8f1f1f'];
}
const isDark = () => getComputedStyle(document.documentElement).colorScheme === 'dark';
const plotlyScale = stops => stops.map((c, i) => [i / (stops.length - 1), c]);

const TYPE_NAMES = {1: 'x-tube', 2: 'y-tube', 3: 'z-tube', 4: 'box', 5: 'irregular', 0: 'unknown'};
const typeColor = t => ({3: css('--series-1'), 1: css('--series-2'), 4: css('--series-3'),
  5: css('--series-4'), 2: css('--series-5')}[t] || css('--text-muted'));
const TYPE_ORDER = [3, 1, 4, 5, 2];

/* --------------------------------------------------------------- plotly */
function baseLayout(extra = {}) {
  const ink = css('--text-secondary'), grid = css('--grid');
  const axis = {gridcolor: grid, zerolinecolor: css('--axis'), linecolor: css('--axis'),
    tickfont: {color: ink, size: 11}, title: {font: {color: ink, size: 12}}};
  const lay = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {family: 'system-ui, -apple-system, Segoe UI, sans-serif', color: ink, size: 12},
    margin: {l: 56, r: 16, t: 24, b: 44}, hovermode: 'closest',
    legend: {orientation: 'h', y: -0.18, font: {size: 11}},
    hoverlabel: {bgcolor: css('--surface-1'), bordercolor: css('--border'), font: {color: css('--text-primary')}},
  };
  for (const k of Object.keys(extra)) if (/^[xy]axis\d*$/.test(k)) extra[k] = deepMerge(structuredClone(axis), extra[k]);
  if (!extra.xaxis) lay.xaxis = axis;
  if (!extra.yaxis) lay.yaxis = structuredClone(axis);
  return deepMerge(lay, extra);
}
function deepMerge(a, b) {
  for (const [k, v] of Object.entries(b)) {
    if (v && typeof v === 'object' && !Array.isArray(v) && a[k] && typeof a[k] === 'object') deepMerge(a[k], v);
    else a[k] = v;
  }
  return a;
}
const PCONF = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']};
function plot(el, traces, layout) { Plotly.react(el, traces, baseLayout(layout), PCONF); return el; }

/* ------------------------------------------------------------------ router */
const app = $('#app');
let cleanup = [];
function route() {
  cleanup.forEach(f => { try { f(); } catch (e) { /* ignore */ } });
  cleanup = [];
  const hash = location.hash.slice(1) || '/';
  const [path, query] = hash.split('?');
  const params = new URLSearchParams(query || '');
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  document.querySelectorAll('[data-nav]').forEach(a => a.classList.toggle('active',
    (parts[0] || 'overview') === a.dataset.nav || (parts[0] === 'run' && a.dataset.nav === 'overview')));
  app.innerHTML = '';
  if (parts[0] === 'run' && parts[1]) return pageRun(parts[1], parts[2] || 'summary', params);
  if (parts[0] === 'compare') return pageCompare(params);
  return pageOverview();
}
window.addEventListener('hashchange', route);

function setHashQuiet(hash) { history.replaceState(null, '', hash); }

/* --------------------------------------------------------------- tooltip */
const tip = $('#tooltip');
function showTip(html, ev) {
  tip.innerHTML = html; tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, hgt = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 4) x = ev.clientX - w - pad;
  if (y + hgt > innerHeight - 4) y = ev.clientY - hgt - pad;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => { tip.hidden = true; };

/* ================================================================ overview */
function pageOverview() {
  const runs = DYNX.manifestData || [];
  $('#manifest-info').textContent = runs.length ? `${runs.length} runs` : '';
  if (!runs.length) {
    app.append(h('div', {class: 'card'}, h('h2', {}, 'No runs found'),
      h('p', {class: 'secondary'}, 'data/manifest.js is missing or empty. Run scripts/dynamite_explorer/sync.sh to pull extracted runs from the cluster.')));
    return;
  }
  const selected = new Set(JSON.parse(localStorageGet('dynx-compare') || '[]'));
  const head = h('div', {class: 'card'},
    h('h1', {}, 'DYNAMITE runs'),
    h('p', {class: 'secondary'}, 'Every directory with an all_models.ecsv on the cluster. Pick runs to compare, or open a run to see its χ² landscape, kinematic fits, intrinsic shape and orbits.'),
  );
  app.append(head);

  const allPars = ['m-bh', 'f-dh', 'q-stars', 'p-stars', 'u-stars', 'ml', 'theta', 'phi', 'psi'];
  let sortKey = 'name', sortDir = 1, filter = '';
  const tableCard = h('div', {class: 'card'});
  const ctl = h('div', {class: 'controls'},
    h('label', {}, 'Filter ', h('input', {type: 'text', placeholder: 'run name…', oninput: e => { filter = e.target.value.toLowerCase(); drawTable(); }})),
    h('label', {}, h('input', {type: 'checkbox', id: 'only-done', checked: true, onchange: drawTable}), 'hide runs without finished models'),
    h('span', {class: 'spacer'}),
    h('button', {class: 'primary', onclick: () => {
      localStorageSet('dynx-compare', JSON.stringify([...selected]));
      location.hash = '#/compare?runs=' + [...selected].map(encodeURIComponent).join(',');
    }}, 'Compare selected'),
  );
  const wrap = h('div', {class: 'tablewrap'});
  tableCard.append(ctl, wrap, h('p', {class: 'note'},
    'χ²/N: the run\'s selection χ² (which_chi2) divided by the number of kinematic constraints N = n_GH × n_bins. Runs fit different data sets, so absolute χ² values are only comparable between runs with the same kinematics.'));
  app.append(tableCard);

  const dots = h('div', {class: 'card'}, h('h2', {}, 'Best-fit parameters across runs'),
    h('p', {class: 'secondary small'}, 'One dot per run: the minimum of each run\'s which_chi2. Bars span the models within Δχ² ≤ √(2N) of the minimum where that is known.'),
    h('div', {class: 'plot', id: 'dots'}));
  app.append(dots);

  function rowsFiltered() {
    const onlyDone = $('#only-done') ? $('#only-done').checked : true;
    return runs.filter(r => (!onlyDone || r.n_done > 0) && r.name.toLowerCase().includes(filter));
  }
  function sortVal(r, k) {
    if (k === 'name') return r.name;
    if (k in (r.best || {})) return r.best[k];
    return r[k];
  }
  function drawTable() {
    const rows = rowsFiltered().slice().sort((a, b) => {
      const va = sortVal(a, sortKey), vb = sortVal(b, sortKey);
      if (va == null) return 1; if (vb == null) return -1;
      return (va > vb ? 1 : va < vb ? -1 : 0) * sortDir;
    });
    const cols = [['', ''], ['name', 'Run'], ['kin', 'Kinematics'], ['n_done', 'Done'], ['n_models', 'Models'],
      ['chi2_red', 'χ²/N'], ['chi2_sel', 'best χ²'], ...allPars.map(p => [p, parLabel(p).replace(/ \[.*\]/, '')]), ['extracted', 'Extracted']];
    const thead = h('tr', {}, cols.map(([k, l]) => h('th', {class: k === 'name' || k === 'kin' ? 'left' : '', onclick: () => {
      if (!k) return; if (sortKey === k) sortDir *= -1; else { sortKey = k; sortDir = 1; } drawTable();
    }}, l + (sortKey === k ? (sortDir > 0 ? ' ↑' : ' ↓') : ''))));
    const tbody = rows.map(r => {
      const cb = h('input', {type: 'checkbox', checked: selected.has(r.name), onclick: e => {
        e.stopPropagation(); e.target.checked ? selected.add(r.name) : selected.delete(r.name);
      }});
      return h('tr', {class: 'clickable', onclick: () => { location.hash = `#/run/${encodeURIComponent(r.name)}`; }},
        h('td', {}, cb),
        h('td', {class: 'left'}, h('a', {href: `#/run/${encodeURIComponent(r.name)}`}, r.name)),
        h('td', {class: 'left secondary'}, (r.kin || []).map(k => `${k.name} (${k.n_bins})`).join(', ')),
        h('td', {}, r.n_done), h('td', {}, r.n_models),
        h('td', {}, r.chi2_red != null ? r.chi2_red.toFixed(2) : '–'),
        h('td', {title: r.which_chi2}, fmtChi(r.chi2_sel)),
        allPars.map(p => h('td', {}, r.best ? fmt(r.best[p], 3) : '–')),
        h('td', {class: 'muted'}, (r.extracted || '').slice(0, 16).replace('T', ' ')));
    });
    wrap.replaceChildren(h('table', {class: 'data'}, h('thead', {}, thead), h('tbody', {}, tbody)));
    drawDots(rows);
  }
  function drawDots(rows) {
    rows = rows.filter(r => r.best);
    const pars = allPars.filter(p => rows.some(r => r.best[p] != null));
    const n = pars.length, ncol = Math.min(n, 3), nrow = Math.ceil(n / ncol);
    const traces = [], layout = {grid: {rows: nrow, columns: ncol, pattern: 'independent', xgap: 0.08, ygap: 0.18},
      showlegend: false, height: Math.max(260, nrow * (40 + rows.length * 16)), margin: {l: 230, r: 16, t: 30, b: 30}};
    pars.forEach((p, i) => {
      const ax = i === 0 ? '' : i + 1;
      const x = [], y = [], lo = [], hi = [], txt = [];
      rows.forEach(r => {
        x.push(r.best[p]); y.push(r.name);
        const rg = (r.range1 || {})[p];
        lo.push(rg ? r.best[p] - rg[0] : 0); hi.push(rg ? rg[1] - r.best[p] : 0);
        txt.push(`${r.name}<br>${parLabel(p)} = ${fmt(r.best[p], 4)}${rg ? `<br>Δχ²≤√2N: ${fmt(rg[0], 3)} – ${fmt(rg[1], 3)}` : ''}`);
      });
      traces.push({type: 'scatter', mode: 'markers', x, y, xaxis: 'x' + ax, yaxis: 'y' + ax,
        marker: {size: 9, color: css('--series-1'), line: {width: 2, color: css('--surface-1')}},
        error_x: {type: 'data', symmetric: false, array: hi, arrayminus: lo, color: css('--series-1'), thickness: 1.5, width: 0},
        hovertext: txt, hoverinfo: 'text'});
      layout['xaxis' + ax] = {title: {text: parLabel(p)}, automargin: true};
      layout['yaxis' + ax] = {showticklabels: i % ncol === 0, automargin: true, type: 'category', categoryorder: 'array', categoryarray: rows.map(r => r.name).reverse()};
    });
    plot($('#dots'), traces, layout);
  }
  drawTable();
}

function localStorageGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* ignore */ } }

/* ===================================================================== run */
const RUN_TABS = [['summary', 'Summary'], ['chi2', 'χ² landscape'], ['kin', 'Kinematic maps'],
  ['shape', 'Intrinsic shape'], ['orbits', 'Orbit distribution'], ['orbits3d', '3D orbits'],
  ['plots', 'DYNAMITE plots'], ['config', 'Config']];

async function pageRun(name, tab, params) {
  app.append(h('p', {class: 'muted'}, `Loading ${name}…`));
  let run;
  try { run = await loadRun(name); } catch (e) {
    app.replaceChildren(h('div', {class: 'card warn'}, e.message)); return;
  }
  if (params.get('model') != null && run.models.rows[+params.get('model')]) run.sel = +params.get('model');
  app.innerHTML = '';
  const best = run.best_row;
  const header = h('div', {class: 'card'},
    h('div', {class: 'muted small'}, h('a', {href: '#/'}, 'Runs'), ' / '),
    h('h1', {}, run.name),
    h('div', {class: 'secondary small mono'}, run.path),
    h('div', {style: 'margin-top:10px'},
      stat('models', run.n_models), stat('finished', run.n_done),
      stat('with details', run.detailRows.length),
      stat(`best ${run.which_chi2}`, best != null ? fmtChi(cval(run, best, run.which_chi2)) : '–'),
      run.n_obs2 ? stat('N constraints', run.n_obs2 / 2) : null,
      (run.geometry || []).map(g => stat(g.name, `${g.n_bins} bins`))),
    modelPicker(run, () => renderTab()),
  );
  const tabs = h('div', {class: 'tabs'}, RUN_TABS.map(([k, l]) =>
    h('a', {href: `#/run/${encodeURIComponent(name)}/${k}`, class: k === tab ? 'active' : ''}, l)));
  const body = h('div');
  app.append(header, tabs, body);
  function renderTab() {
    cleanup.forEach(f => { try { f(); } catch (e) { /* ignore */ } });
    cleanup = [];
    body.innerHTML = '';
    const f = {summary: tabSummary, chi2: tabChi2, kin: tabKin, shape: tabShape, orbits: tabOrbits,
      orbits3d: tabOrbits3d, plots: tabPlots, config: tabConfig}[tab] || tabSummary;
    f(run, body, renderTab);
  }
  run._rerender = renderTab;
  renderTab();
}

function stat(k, v) { return h('span', {class: 'stat'}, h('span', {class: 'v'}, v), h('span', {class: 'k'}, k)); }

function modelLabel(run, row) {
  const d = cval(run, row, 'directory');
  return `#${row} ${d} · ${run.which_chi2} ${fmtChi(cval(run, row, run.which_chi2))}`;
}
function modelPicker(run, onChange) {
  const rows = run.detailRows.slice().sort((a, b) => cval(run, a, run.which_chi2) - cval(run, b, run.which_chi2));
  if (!rows.length) return h('p', {class: 'note'}, 'No model details were extracted for this run (no finished models with orbit libraries on disk).');
  const sel = h('select', {onchange: e => {
    run.sel = +e.target.value;
    setHashQuiet(location.hash.split('?')[0] + `?model=${run.sel}`);
    onChange();
  }}, rows.map(r => h('option', {value: r, selected: r === run.sel}, (r === run.best_row ? '★ ' : '') + modelLabel(run, r))));
  return h('div', {class: 'controls', style: 'margin:10px 0 0'}, h('label', {}, 'Model ', sel),
    h('span', {class: 'muted small'}, `${rows.length} models with kinematic maps and orbit data; the χ² tab covers all ${run.n_models}.`));
}
const detail = run => (run.details || {})[run.sel];

/* -------------------------------------------------------------- summary */
function tabSummary(run, body) {
  const thr = chiThreshold(run);
  const which = run.which_chi2;
  const best = run.best_row;
  const chiMin = bestChi(run, which);
  const parsShown = run.allPars.concat(['theta', 'phi', 'psi']);
  const ranges = {};
  if (thr) {
    for (const p of parsShown) {
      const vals1 = [], vals3 = [];
      run.models.rows.forEach((r, i) => {
        const c = r[run.ci[which]];
        if (!run.done[i] || c == null) return;
        const v = pval(run, i, p);
        if (v == null || !isFinite(v)) return;
        if (c - chiMin <= thr) vals1.push(v);
        if (c - chiMin <= 3 * thr) vals3.push(v);
      });
      ranges[p] = [vals1.length ? [Math.min(...vals1), Math.max(...vals1)] : null,
        vals3.length ? [Math.min(...vals3), Math.max(...vals3)] : null, vals1.length, vals3.length];
    }
  }
  const sel = run.sel;
  const rows = parsShown.map(p => {
    const info = run.parInfo[p] || {};
    const rg = ranges[p] || [];
    return h('tr', {},
      h('td', {}, parLabel(p)),
      h('td', {class: 'left secondary'}, info.fixed ? 'fixed' : (p in run.parInfo ? 'free' : 'derived')),
      h('td', {}, best != null ? fmt(pval(run, best, p), 4) : '–'),
      h('td', {}, sel != null && sel !== best ? fmt(pval(run, sel, p), 4) : '·'),
      h('td', {}, rg[0] ? `${fmt(rg[0][0], 3)} – ${fmt(rg[0][1], 3)}` : '–'),
      h('td', {}, rg[1] ? `${fmt(rg[1][0], 3)} – ${fmt(rg[1][1], 3)}` : '–'),
      h('td', {class: 'muted'}, info.lo != null ? `${info.lo} … ${info.hi}` : ''),
      h('td', {class: 'muted'}, info.step != null ? `${info.step} / ${info.minstep}` : ''));
  });
  const parCard = h('div', {class: 'card'}, h('h2', {}, 'Parameters'),
    h('div', {class: 'tablewrap'}, h('table', {class: 'data'},
      h('thead', {}, h('tr', {}, ['Parameter', 'Status', 'Best', 'Selected model', 'Δχ² ≤ √2N', 'Δχ² ≤ 3√2N', 'Bounds', 'Step / min'].map(t => h('th', {class: t === 'Status' ? 'left' : ''}, t)))),
      h('tbody', {}, rows))),
    h('p', {class: 'note'}, thr
      ? `Ranges from the finished models within the Δχ² thresholds DYNAMITE uses (√(2N) = ${thr.toFixed(1)} for ${which}; ${ranges[parsShown[0]]?.[2] ?? 0} and ${ranges[parsShown[0]]?.[3] ?? 0} models). Grid ranges, not formal confidence intervals. θ, φ, ψ are the viewing angles implied by (p, q, u).`
      : 'The number of constraints is unknown for this run (no model details were extracted), so no Δχ² ranges are shown.'));

  const chiCard = h('div', {class: 'card'}, h('h2', {}, 'χ² of the best and selected model'));
  const chiRows = [best, sel].filter((v, i, a) => v != null && a.indexOf(v) === i).map(r => h('tr', {},
    h('td', {}, r === best ? '★ best' : 'selected'), h('td', {class: 'left mono'}, cval(run, r, 'directory')),
    CHI2S.map(c => h('td', {}, fmtChi(cval(run, r, c)))),
    h('td', {}, cval(run, r, 'which_iter'))));
  chiCard.append(h('table', {class: 'data'},
    h('thead', {}, h('tr', {}, ['', 'Directory', 'kinmapχ²', 'kinχ²', 'χ² (total)', 'iteration'].map(t => h('th', {}, t)))),
    h('tbody', {}, chiRows)));
  const det = detail(run);
  if (det) {
    chiCard.append(h('h3', {style: 'margin-top:14px'}, 'Map χ² by kinematic set and moment (selected model)'),
      h('table', {class: 'data'},
        h('thead', {}, h('tr', {}, ['Set', 'bins', 'v', 'σ', 'h3', 'h4', 'total', 'total / (4 n_bins)'].map(t => h('th', {}, t)))),
        h('tbody', {}, det.kin.map(k => h('tr', {}, h('td', {}, k.name), h('td', {}, k.n_bins),
          ['v', 'sigma', 'h3', 'h4'].map(m => h('td', {}, fmtChi(k.chi2[m]))),
          h('td', {}, fmtChi(k.chi2_total)), h('td', {}, (k.chi2_total / (4 * k.n_bins)).toFixed(2)))))),
      h('p', {class: 'note'}, 'Recomputed from the per-bin data and model Gauss–Hermite moments: Σ((data − model)/error)². Their sum reproduces DYNAMITE\'s kinmapchi2.'));
  }
  const s = run.settings || {};
  const ob = s.orblib || {}, ws = s.weight_solver || {}, ps = s.parameter_space || {};
  const setCard = h('div', {class: 'card'}, h('h2', {}, 'Settings'),
    kv([
      ['Orbit library', `nE=${ob.nE}, nI2=${ob.nI2}, nI3=${ob.nI3}, dithering=${ob.dithering}`],
      ['Radial range', `log r = ${ob.logrmin} … ${ob.logrmax} arcsec`],
      ['Weight solver', `${ws.type}${ws.nnls_solver ? ' / ' + ws.nnls_solver : ''}, regularisation ${ws.regularisation}, n_GH ${ws.number_GH}`],
      ['GH systematic errors', ws.GH_sys_err],
      ['Parameter search', `${ps.generator_type}, which_chi2 = ${ps.which_chi2}`],
      ['Stopping', JSON.stringify(ps.stopping_criteria || {})],
      ['Halo / BH', `${s.dh_type || '–'} / ${s.bh_type || '–'}`],
      ['Distance', `${s.distMPc} Mpc`],
      ['Kinematics', Object.entries(s.kinematics || {}).map(([k, v]) => `${k}: ${v.datafile}`).join('; ')],
      ['Config', run.config_file], ['Input', run.input_dir], ['Input note', run.input_note],
      ['DYNAMITE', run.dynamite_version], ['Extracted', run.extracted],
    ]));
  body.append(h('div', {class: 'row'}, parCard), h('div', {class: 'row'}, chiCard, setCard));
}
function kv(pairs) {
  return h('table', {class: 'data'}, h('tbody', {}, pairs.filter(p => p[1] != null && p[1] !== 'undefined').map(([k, v]) =>
    h('tr', {}, h('td', {class: 'secondary'}, k), h('td', {class: 'left', style: 'white-space:normal'}, String(v))))));
}

/* ---------------------------------------------------------- chi2 tab */
function tabChi2(run, body) {
  let which = run.which_chi2, logy = true, onlyNear = false;
  const pars = run.freePars.concat(['theta', 'phi', 'psi']);
  const allX = run.allPars.concat(['theta', 'phi', 'psi', 'which_iter']);
  let px = pars.find(p => p === 'q-stars') || pars[0], py = pars.find(p => p === 'p-stars') || pars[1] || pars[0];
  const ctl = h('div', {class: 'controls'},
    h('label', {}, 'χ² ', h('select', {onchange: e => { which = e.target.value; draw(); }},
      CHI2S.map(c => h('option', {value: c, selected: c === which}, c + (c === run.which_chi2 ? ' (used for search)' : ''))))),
    h('label', {}, h('input', {type: 'checkbox', checked: logy, onchange: e => { logy = e.target.checked; draw(); }}), 'log Δχ²'),
    h('label', {}, h('input', {type: 'checkbox', onchange: e => { onlyNear = e.target.checked; draw(); }}), 'only Δχ² ≤ 10 √2N'),
  );
  const c1 = h('div', {class: 'card'}, h('h2', {}, 'Δχ² against each parameter'),
    h('p', {class: 'secondary small'}, 'Each dot is a finished model. Ringed dots have kinematic maps and orbit data; click one to select it. Dashed lines: Δχ² = √(2N) and 3√(2N).'),
    ctl, h('div', {class: 'plot', id: 'chi-par'}));
  const pairCtl = h('div', {class: 'controls'},
    h('label', {}, 'x ', h('select', {onchange: e => { px = e.target.value; drawPair(); }}, allX.map(p => h('option', {value: p, selected: p === px}, parLabel(p))))),
    h('label', {}, 'y ', h('select', {onchange: e => { py = e.target.value; drawPair(); }}, allX.map(p => h('option', {value: p, selected: p === py}, parLabel(p))))));
  const c2 = h('div', {class: 'card'}, h('h2', {}, 'Parameter pairs'),
    h('p', {class: 'secondary small'}, 'Colour: Δχ², capped at 3√(2N); the colour bar runs from best (0) to worst. Grey: models outside that range.'),
    pairCtl, h('div', {class: 'plot', id: 'chi-pair', style: 'height:480px'}));
  const c3 = h('div', {class: 'card'}, h('h2', {}, 'Progress'), h('div', {class: 'plot', id: 'chi-prog', style: 'height:300px'}));
  const c4 = h('div', {class: 'card'}, h('h2', {}, 'Models'), h('div', {class: 'tablewrap', id: 'model-table'}));
  body.append(c1, h('div', {class: 'grid2'}, c2, c3), c4);

  const thr = chiThreshold(run);
  function finished() {
    const chiMin = bestChi(run, which);
    const out = [];
    run.models.rows.forEach((r, i) => {
      const c = r[run.ci[which]];
      if (!run.done[i] || c == null) return;
      const d = c - chiMin;
      if (onlyNear && thr && d > 10 * thr) return;
      out.push({i, d});
    });
    return out;
  }
  function draw() {
    const fin = finished();
    const n = pars.length, ncol = Math.min(3, n), nrow = Math.ceil(n / ncol);
    const traces = [], layout = {grid: {rows: nrow, columns: ncol, pattern: 'independent', ygap: 0.22, xgap: 0.08},
      showlegend: false, height: 250 * nrow + 40, shapes: []};
    const det = new Set(run.detailRows);
    const floor = logy ? Math.max(0.1, (thr || 1) * 0.01) : 0;
    pars.forEach((p, k) => {
      const ax = k === 0 ? '' : k + 1;
      const pts = fin.map(f => ({x: pval(run, f.i, p), y: Math.max(f.d, floor), i: f.i})).filter(q => q.x != null && isFinite(q.x));
      traces.push({type: pts.length > 3000 ? 'scattergl' : 'scatter', mode: 'markers', xaxis: 'x' + ax, yaxis: 'y' + ax,
        x: pts.map(q => q.x), y: pts.map(q => q.y), customdata: pts.map(q => q.i),
        hovertemplate: `${parLabel(p)} %{x}<br>Δχ² %{y:.1f}<br>row %{customdata}<extra></extra>`,
        marker: {size: 6, color: css('--series-1'), opacity: 0.55}});
      const dp = pts.filter(q => det.has(q.i));
      traces.push({type: 'scatter', mode: 'markers', xaxis: 'x' + ax, yaxis: 'y' + ax,
        x: dp.map(q => q.x), y: dp.map(q => q.y), customdata: dp.map(q => q.i), hoverinfo: 'skip',
        marker: {size: 11, color: 'rgba(0,0,0,0)', line: {width: 2, color: dp.map(q => q.i === run.sel ? css('--series-2') : css('--text-primary'))}}});
      layout['xaxis' + ax] = {title: {text: parLabel(p)}};
      layout['yaxis' + ax] = {type: logy ? 'log' : 'linear', title: {text: k % ncol === 0 ? 'Δ' + which : ''}};
      if (thr) [1, 3].forEach(m => layout.shapes.push({type: 'line', xref: `x${ax} domain`, yref: 'y' + ax, x0: 0, x1: 1,
        y0: m * thr, y1: m * thr, line: {color: css('--axis'), width: 1, dash: 'dash'}}));
    });
    const el = plot($('#chi-par'), traces, layout);
    el.removeAllListeners && el.removeAllListeners('plotly_click');
    el.on('plotly_click', ev => selectFromPlot(run, ev));
    drawPair(); drawProgress(); drawModelTable();
  }
  function drawPair() {
    const fin = finished();
    const cap = thr ? 3 * thr : Math.max(...fin.map(f => f.d)) || 1;
    const inn = fin.filter(f => f.d <= cap), out = fin.filter(f => f.d > cap);
    const val = (i, p) => p === 'which_iter' ? cval(run, i, p) : pval(run, i, p);
    inn.sort((a, b) => b.d - a.d);
    const traces = [
      {type: 'scattergl', mode: 'markers', x: out.map(f => val(f.i, px)), y: out.map(f => val(f.i, py)), hoverinfo: 'skip',
        marker: {size: 6, color: css('--grid')}, name: 'outside'},
      {type: 'scatter', mode: 'markers', x: inn.map(f => val(f.i, px)), y: inn.map(f => val(f.i, py)), customdata: inn.map(f => f.i),
        text: inn.map(f => f.d.toFixed(1)),
        hovertemplate: `${parLabel(px)} %{x}<br>${parLabel(py)} %{y}<br>Δχ² %{text}<br>row %{customdata}<extra></extra>`,
        marker: {size: 10, color: inn.map(f => f.d), cmin: 0, cmax: cap, colorscale: plotlyScale(seqStops().reverse()),
          line: {width: 1, color: css('--surface-1')}, colorbar: {title: {text: 'Δχ²', side: 'right'}, thickness: 10, len: 0.8}}},
    ];
    if (run.sel != null) traces.push({type: 'scatter', mode: 'markers', x: [val(run.sel, px)], y: [val(run.sel, py)], hoverinfo: 'skip',
      marker: {size: 16, color: 'rgba(0,0,0,0)', line: {width: 2, color: css('--series-2')}}, name: 'selected'});
    const el = plot($('#chi-pair'), traces, {showlegend: false, xaxis: {title: {text: parLabel(px)}}, yaxis: {title: {text: parLabel(py)}}});
    el.removeAllListeners && el.removeAllListeners('plotly_click');
    el.on('plotly_click', ev => selectFromPlot(run, ev));
  }
  function drawProgress() {
    const fin = finished().sort((a, b) => a.i - b.i);
    let best = Infinity;
    const runBest = fin.map(f => (best = Math.min(best, f.d)));
    plot($('#chi-prog'), [
      {type: 'scatter', mode: 'markers', x: fin.map(f => cval(run, f.i, 'which_iter')), y: fin.map(f => Math.max(f.d, 0.1)),
        marker: {size: 5, color: css('--series-1'), opacity: 0.5}, name: 'model', hovertemplate: 'iter %{x}<br>Δχ² %{y:.1f}<extra></extra>'},
      {type: 'scatter', mode: 'lines', x: fin.map(f => cval(run, f.i, 'which_iter')), y: runBest.map(v => Math.max(v, 0.1)),
        line: {width: 2, color: css('--series-2'), shape: 'hv'}, name: 'running minimum'},
    ], {xaxis: {title: {text: 'iteration'}}, yaxis: {type: 'log', title: {text: 'Δ' + which}}, legend: {y: -0.3}});
  }
  let sortCol = which, sortDir = 1;
  function drawModelTable() {
    const cols = ['directory', ...run.allPars, 'theta', 'phi', 'psi', ...CHI2S, 'which_iter'];
    const idx = run.models.rows.map((_, i) => i);
    const key = i => cols.includes(sortCol) && sortCol !== 'directory' ? (run.parInfo[sortCol] ? pval(run, i, sortCol) : cval(run, i, sortCol)) : cval(run, i, 'directory');
    idx.sort((a, b) => {
      const va = key(a), vb = key(b);
      if (va == null) return 1; if (vb == null) return -1;
      return (va > vb ? 1 : va < vb ? -1 : 0) * sortDir;
    });
    const det = new Set(run.detailRows);
    const shown = idx.slice(0, 400);
    $('#model-table').replaceChildren(h('table', {class: 'data'},
      h('thead', {}, h('tr', {}, h('th', {}, ''), cols.map(c => h('th', {class: c === 'directory' ? 'left' : '', onclick: () => {
        if (sortCol === c) sortDir *= -1; else { sortCol = c; sortDir = 1; } drawModelTable();
      }}, (c === 'directory' ? 'Directory' : parLabel(c).replace(/ \(.*\)/, '')) + (sortCol === c ? (sortDir > 0 ? ' ↑' : ' ↓') : ''))))),
      h('tbody', {}, shown.map(i => h('tr', {class: (det.has(i) ? 'clickable ' : '') + (i === run.sel ? 'sel' : ''),
        onclick: det.has(i) ? () => selectModel(run, i) : null},
        h('td', {title: det.has(i) ? 'details available' : ''}, det.has(i) ? '●' : (run.done[i] ? '' : '…')),
        cols.map(c => h('td', {class: c === 'directory' ? 'left mono' : ''},
          c === 'directory' ? cval(run, i, c) : c === 'which_iter' ? cval(run, i, c) : run.parInfo[c] ? fmt(pval(run, i, c), 4) : CHI2S.includes(c) ? fmtChi(cval(run, i, c)) : fmt(cval(run, i, c), 3))))))),
    );
    if (idx.length > 400) $('#model-table').append(h('p', {class: 'note'}, `Showing the first 400 of ${idx.length} rows in this sort order.`));
  }
  draw();
}
function selectFromPlot(run, ev) {
  const pt = ev.points && ev.points[0];
  if (!pt || pt.customdata == null) return;
  const i = +pt.customdata;
  if (run.details && run.details[i]) selectModel(run, i);
}
function selectModel(run, i) {
  run.sel = i;
  const base = location.hash.split('?')[0];
  location.hash = base + `?model=${i}`;
}

/* --------------------------------------------------------- kinematic maps */
const MOMENTS = [['v', 'V [km/s]', true], ['sigma', 'σ [km/s]', false], ['h3', 'h3', true], ['h4', 'h4', true]];

function tabKin(run, body) {
  const det = detail(run);
  if (!det) { body.append(noDetails(run)); return; }
  const sets = det.kin.map(k => k.name);
  let set = sets[0], resScale = 3, zoom = 1;
  const ctl = h('div', {class: 'controls'},
    h('label', {}, 'Kinematic set ', h('select', {onchange: e => { set = e.target.value; draw(); }}, sets.map(s => h('option', {value: s}, s)))),
    h('label', {}, 'Residual range ±', h('select', {onchange: e => { resScale = +e.target.value; draw(); }}, [2, 3, 5, 10].map(v => h('option', {value: v, selected: v === 3}, v + 'σ')))),
    h('label', {}, 'Zoom ', h('input', {type: 'range', min: 1, max: 12, step: 0.5, value: 1, oninput: e => { zoom = +e.target.value; draw(); }})),
  );
  const mapsEl = h('div');
  const histEl = h('div', {class: 'plot', style: 'height:260px'});
  const radEl = h('div', {class: 'plot', style: 'height:300px'});
  body.append(h('div', {class: 'card'}, h('h2', {}, 'Data, model and residuals'),
    h('p', {class: 'secondary small'}, 'Columns: observed moments, the model\'s Gauss–Hermite fit, and (data − model)/error. Data and model share one colour scale per row. Hover a pixel for the bin values.'),
    ctl, mapsEl),
  h('div', {class: 'grid2'},
    h('div', {class: 'card'}, h('h2', {}, 'Residual distributions'), histEl,
      h('p', {class: 'note'}, 'Normalised residuals (data − model)/error per bin. A unit Gaussian (dashed) is what errors that describe the scatter would give.')),
    h('div', {class: 'card'}, h('h2', {}, 'Moments against radius'), radEl,
      h('p', {class: 'note'}, 'Bin radius from the flux-weighted pixel centroid, in the data frame.'))));
  function draw() {
    const k = det.kin.find(x => x.name === set);
    const g = run.geomByName[set];
    mapsEl.replaceChildren(kinMapGrid(g, k, {resScale, zoom, columns: ['data', 'model', 'resid']}));
    drawResidHist(histEl, k);
    drawRadial(radEl, g, k);
  }
  draw();
}

function noDetails(run) {
  return h('div', {class: 'card'}, h('p', {class: 'secondary'},
    run.detailRows.length ? 'Pick one of the models with extracted details in the selector above.' :
      'This run has no extracted model details: either no model has finished, or the orbit libraries of the finished models are no longer on disk.'));
}

// geometry helpers: bin centroids
function binCentroids(g) {
  if (g._cent) return g._cent;
  const n = g.n_bins, sx = new Float64Array(n + 1), sy = new Float64Array(n + 1), cnt = new Float64Array(n + 1);
  for (let j = 0; j < g.ny; j++) for (let i = 0; i < g.nx; i++) {
    const b = g.ids[j * g.nx + i];
    if (!b) continue;
    sx[b] += g.minx + (i + 0.5) * g.dx; sy[b] += g.miny + (j + 0.5) * g.dx; cnt[b]++;
  }
  const a = g.angle * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  const out = [];
  for (let b = 1; b <= n; b++) {
    const x = sx[b] / cnt[b], y = sy[b] / cnt[b];
    out.push({x: x * c - y * s, y: x * s + y * c, r: Math.hypot(x, y), npix: cnt[b]});
  }
  return (g._cent = out);
}

function robustRange(arrs, sym) {
  const vals = arrs.flat().filter(v => v != null && isFinite(v)).sort((a, b) => a - b);
  if (!vals.length) return [0, 1];
  const q = p => vals[Math.min(vals.length - 1, Math.floor(p * (vals.length - 1)))];
  if (sym) {
    const med = q(0.5), lim = Math.max(Math.abs(q(0.02) - (med)), Math.abs(q(0.98) - med));
    // velocity maps are centred on the median, h3/h4 on zero
    const c = Math.abs(med) < lim * 0.25 ? 0 : med;
    return [c - lim, c + lim];
  }
  return [q(0.02), q(0.98)];
}

/* Draw one map. vals is per bin (index 0 = bin 1). */
function drawBinMap(canvas, g, vals, range, cmap, zoom) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!g._img) {
    g._img = document.createElement('canvas');
    g._img.width = g.nx; g._img.height = g.ny;
  }
  const im = g._img.getContext('2d').createImageData(g.nx, g.ny);
  const lut = vals.map(v => (v == null || !isFinite(v)) ? null : cmap((v - range[0]) / (range[1] - range[0])));
  for (let p = 0; p < g.ids.length; p++) {
    const b = g.ids[p];
    if (!b) continue;
    const c = lut[b - 1];
    if (!c) continue;
    im.data[4 * p] = c[0]; im.data[4 * p + 1] = c[1]; im.data[4 * p + 2] = c[2]; im.data[4 * p + 3] = 255;
  }
  g._img.getContext('2d').putImageData(im, 0, 0);
  const T = mapTransform(g, W, H, zoom);
  ctx.imageSmoothingEnabled = false;
  ctx.setTransform(T.a, T.b, T.c, T.d, T.e, T.f);
  ctx.drawImage(g._img, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  // centre cross and scale bar
  ctx.strokeStyle = css('--text-muted'); ctx.lineWidth = 1;
  const [cx, cy] = T.toScreen(0, 0);
  ctx.beginPath(); ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy); ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4); ctx.stroke();
  const target = (W / T.s) / 4, pow = Math.pow(10, Math.floor(log10(target)));
  const len = [1, 2, 5, 10].map(m => m * pow).filter(v => v <= target).pop() || pow;
  ctx.fillStyle = css('--text-secondary'); ctx.font = '10px system-ui';
  ctx.fillRect(8, H - 10, len * T.s, 2);
  ctx.fillText(`${len}″`, 8, H - 14);
}
// pixel grid -> rotated plot coordinates (major axis along x) -> canvas
function mapTransform(g, W, H, zoom = 1) {
  const a = g.angle * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
  const corners = [[g.minx, g.miny], [g.minx + g.nx * g.dx, g.miny], [g.minx, g.miny + g.ny * g.dx], [g.minx + g.nx * g.dx, g.miny + g.ny * g.dx]]
    .map(([x, y]) => [x * ca - y * sa, x * sa + y * ca]);
  const xs = corners.map(c => c[0]), ys = corners.map(c => c[1]);
  const ext = Math.max(Math.max(...xs.map(Math.abs)), Math.max(...ys.map(Math.abs)));
  const s = Math.min(W, H) / (2 * ext) * zoom * 0.98;
  const cx = W / 2, cy = H / 2;
  return {
    s, a: s * g.dx * ca, b: -s * g.dx * sa, c: -s * g.dx * sa, d: -s * g.dx * ca,
    e: cx + s * (ca * g.minx - sa * g.miny), f: cy - s * (sa * g.minx + ca * g.miny),
    toScreen: (xr, yr) => [cx + s * xr, cy - s * yr],
    toData: (px, py) => {
      const xr = (px - cx) / s, yr = -(py - cy) / s;
      return [xr * ca + yr * sa, -xr * sa + yr * ca];   // rotate back
    },
    toPlot: (px, py) => [(px - cx) / s, -(py - cy) / s],
  };
}
function binAt(g, T, px, py) {
  const [x, y] = T.toData(px, py);
  const i = Math.floor((x - g.minx) / g.dx), j = Math.floor((y - g.miny) / g.dx);
  if (i < 0 || j < 0 || i >= g.nx || j >= g.ny) return 0;
  return g.ids[j * g.nx + i];
}

function colorbar(range, cmap, label) {
  const c = h('canvas', {width: 200, height: 8});
  const ctx = c.getContext('2d');
  for (let x = 0; x < 200; x++) { const col = cmap(x / 199); ctx.fillStyle = `rgb(${col})`; ctx.fillRect(x, 0, 1, 8); }
  return h('div', {class: 'cbar'}, h('span', {}, fmt(range[0], 2)), c, h('span', {}, fmt(range[1], 2)), h('span', {}, label));
}

/* Map grid. columns: list of {key,label,vals(m)->array,kind:'value'|'resid'} or shorthand strings. */
function kinMapGrid(g, k, opts) {
  const {resScale = 3, zoom = 1} = opts;
  const cols = opts.columns.map(c => typeof c === 'string' ? {data: {label: 'Data', get: m => k.data[m], kind: 'value', src: k},
    model: {label: 'Model', get: m => k.model[m], kind: 'value', src: k},
    resid: {label: '(data − model)/err', get: m => resid(k, m), kind: 'resid', src: k}}[c] : c);
  const ncol = cols.length;
  const grid = h('div', {class: 'mapgrid', style: `grid-template-columns: 18px repeat(${ncol}, minmax(0, 1fr))`});
  grid.append(h('div'), ...cols.map(c => h('div', {class: 'lbl'}, c.label)));
  const seqMap = makeCmap(seqStops()), divMap = makeCmap(divStops());
  const size = 300;
  for (const [m, label, sym] of MOMENTS) {
    if (!k.data[m]) continue;
    const valueCols = cols.filter(c => c.kind === 'value');
    const vr = robustRange(valueCols.map(c => c.get(m) || []).concat([k.data[m]]), sym);
    grid.append(h('div', {class: 'rowlbl'}, label));
    for (const c of cols) {
      const isRes = c.kind === 'resid';
      const range = isRes ? [-resScale, resScale] : vr;
      const cmap = isRes || sym ? divMap : seqMap;
      const cv = h('canvas', {width: size * devicePixelRatio, height: size * devicePixelRatio});
      const vals = c.get(m) || [];
      const geo = c.geom || g;
      drawBinMap(cv, geo, vals, range, cmap, zoom);
      const T = mapTransform(geo, cv.width, cv.height, zoom);
      const src = c.src || k;
      cv.addEventListener('mousemove', ev => {
        const r = cv.getBoundingClientRect();
        const px = (ev.clientX - r.left) * cv.width / r.width, py = (ev.clientY - r.top) * cv.height / r.height;
        const b = binAt(geo, T, px, py);
        if (!b) { hideTip(); return; }
        const i = b - 1, [xr, yr] = T.toPlot(px, py);
        const d = src.data[m]?.[i], e = src.data['d' + m]?.[i], mo = src.model[m]?.[i];
        showTip(`<b>bin ${b}</b> · (${xr.toFixed(2)}″, ${yr.toFixed(2)}″)<br>${label}: data ${fmt(d, 3)} ± ${fmt(e, 3)}<br>model ${fmt(mo, 3)}<br>resid ${fmt((d - mo) / e, 2)}σ`, ev);
      });
      cv.addEventListener('mouseleave', hideTip);
      grid.append(h('div', {class: 'mapcell'}, cv, colorbar(range, cmap, isRes ? 'σ' : '')));
    }
  }
  return grid;
}
const resid = (k, m) => (k.data[m] || []).map((d, i) => (d - k.model[m][i]) / k.data['d' + m][i]);

function drawResidHist(el, k) {
  const traces = [];
  const xs = [];
  for (let x = -6; x <= 6; x += 0.1) xs.push(x);
  MOMENTS.forEach(([m, label], i) => {
    if (!k.data[m]) return;
    const r = resid(k, m).filter(isFinite);
    const ax = i === 0 ? '' : i + 1;
    traces.push({type: 'histogram', x: r.map(v => Math.max(-6, Math.min(6, v))), xbins: {start: -6, end: 6, size: 0.5},
      histnorm: 'probability density', xaxis: 'x' + ax, yaxis: 'y' + ax, name: label,
      marker: {color: css('--series-1'), line: {width: 1, color: css('--surface-1')}},
      hovertemplate: `${label}<br>%{x}: %{y:.2f}<extra></extra>`});
    traces.push({type: 'scatter', mode: 'lines', x: xs, y: xs.map(x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI)),
      xaxis: 'x' + ax, yaxis: 'y' + ax, line: {dash: 'dash', width: 1.5, color: css('--text-secondary')}, hoverinfo: 'skip'});
  });
  const lay = {grid: {rows: 1, columns: 4, pattern: 'independent', xgap: 0.06}, showlegend: false, margin: {l: 40, r: 10, t: 28, b: 36}, annotations: []};
  MOMENTS.forEach(([m, label], i) => {
    const ax = i === 0 ? '' : i + 1;
    const r = resid(k, m).filter(isFinite);
    const rms = Math.sqrt(r.reduce((s, v) => s + v * v, 0) / r.length);
    lay['xaxis' + ax] = {range: [-6, 6], title: {text: label.split(' ')[0]}};
    lay['yaxis' + ax] = {showticklabels: i === 0};
    lay.annotations.push({xref: `x${ax} domain`, yref: `y${ax} domain`, x: 0.02, y: 1.12, showarrow: false, xanchor: 'left',
      text: `rms ${rms.toFixed(2)}`, font: {size: 11}});
  });
  plot(el, traces, lay);
}

function drawRadial(el, g, k) {
  const cent = binCentroids(g);
  const traces = [], lay = {grid: {rows: 1, columns: 4, pattern: 'independent', xgap: 0.07}, showlegend: false, margin: {l: 44, r: 10, t: 10, b: 40}};
  MOMENTS.forEach(([m, label], i) => {
    if (!k.data[m]) return;
    const ax = i === 0 ? '' : i + 1;
    const r = cent.map(c => c.r);
    traces.push({type: 'scatter', mode: 'markers', x: r, y: k.data[m], xaxis: 'x' + ax, yaxis: 'y' + ax, name: 'data',
      marker: {size: 4, color: css('--text-muted'), opacity: 0.6}, hovertemplate: 'r %{x:.2f}″<br>data %{y:.3f}<extra></extra>'});
    traces.push({type: 'scatter', mode: 'markers', x: r, y: k.model[m], xaxis: 'x' + ax, yaxis: 'y' + ax, name: 'model',
      marker: {size: 4, color: css('--series-2')}, hovertemplate: 'r %{x:.2f}″<br>model %{y:.3f}<extra></extra>'});
    lay['xaxis' + ax] = {type: 'log', dtick: 1, title: {text: 'r [″]'}};
    lay['yaxis' + ax] = {title: {text: label}};
  });
  plot(el, traces, lay);
}

/* --------------------------------------------------------- intrinsic shape */
function tabShape(run, body) {
  const q = run.qpt;
  if (!q) { body.append(h('div', {class: 'card'}, h('p', {class: 'secondary'}, 'No intrinsic shape profile (no finished model).'))); return; }
  let logx = true;
  const el = h('div', {class: 'plot', style: 'height:440px'});
  const ctl = h('div', {class: 'controls'}, h('label', {}, h('input', {type: 'checkbox', checked: true, onchange: e => { logx = e.target.checked; draw(); }}), 'log radius'));
  const det = detail(run);
  const g = det && det.orbits3d && det.orbits3d.gauss;
  const bestRow = q.rows[0];
  body.append(h('div', {class: 'grid2'},
    h('div', {class: 'card'}, h('h2', {}, 'Intrinsic axis ratios p(r), q(r) and triaxiality T(r)'),
      h('p', {class: 'secondary small'}, `Isodensity axis ratios of the deprojected MGE along the principal axes, as in DYNAMITE's triaxial_qpt plot. Lines: best model (row ${bestRow}). Bands: mean ± std over the ${q.n} finished models within 3√(2N) of the minimum χ².`),
      ctl, el),
    h('div', {class: 'card'}, h('h2', {}, 'Viewing angles and deprojected Gaussians'),
      kv([['Best (p, q, u)', `${fmt(pval(run, bestRow, 'p-stars'), 4)}, ${fmt(pval(run, bestRow, 'q-stars'), 4)}, ${fmt(pval(run, bestRow, 'u-stars'), 4)}`],
        ['θ, φ, ψ', `${fmt(cval(run, bestRow, 'theta'), 2)}°, ${fmt(cval(run, bestRow, 'phi'), 2)}°, ${fmt(cval(run, bestRow, 'psi'), 2)}°`],
        ['flattest observed q′', fmt(run.mge.qobs_min, 4)]]),
      g ? h('div', {class: 'tablewrap', style: 'margin-top:10px'}, h('table', {class: 'data'},
        h('thead', {}, h('tr', {}, ['#', 'σ_obs ″', 'q′', 'σ_intr ″', 'p_j', 'q_j', 'T_j', 'M_j [M☉]'].map(t => h('th', {}, t)))),
        h('tbody', {}, g.p.map((p, j) => h('tr', {}, h('td', {}, j + 1), h('td', {}, fmt(run.mge.sigma[j], 3)), h('td', {}, fmt(run.mge.q[j], 3)),
          h('td', {}, fmt(g.sigma[j], 3)), h('td', {}, fmt(p, 3)), h('td', {}, fmt(g.q[j], 3)),
          h('td', {}, fmt((1 - p * p) / (1 - g.q[j] * g.q[j]), 3)), h('td', {}, fmt(g.mass[j], 3))))))) : null,
      g ? h('p', {class: 'note'}, 'Per-Gaussian values for the selected model\'s orbit library (from its parameters_pot.in), with the M/L the library was integrated at.') : null)));
  function draw() {
    const r = q.r;
    const band = (m, s, col, name) => [
      {type: 'scatter', mode: 'lines', x: r, y: m.map((v, i) => v + s[i]), line: {width: 0}, hoverinfo: 'skip', showlegend: false},
      {type: 'scatter', mode: 'lines', x: r, y: m.map((v, i) => v - s[i]), fill: 'tonexty', fillcolor: col + '33', line: {width: 0}, hoverinfo: 'skip', showlegend: false},
    ];
    const traces = [
      ...band(q.p_m, q.p_s, css('--series-1')), ...band(q.q_m, q.q_s, css('--series-2')), ...band(q.T_m, q.T_s, css('--series-3')),
      {type: 'scatter', mode: 'lines', x: r, y: q.best.p, name: 'p = B/A', line: {width: 2, color: css('--series-1')}},
      {type: 'scatter', mode: 'lines', x: r, y: q.best.q, name: 'q = C/A', line: {width: 2, color: css('--series-2')}},
      {type: 'scatter', mode: 'lines', x: r, y: q.best.T, name: 'T = (1−p²)/(1−q²)', line: {width: 2, color: css('--series-3')}},
    ];
    plot(el, traces, {xaxis: {type: logx ? 'log' : 'linear', dtick: logx ? 1 : undefined, title: {text: 'r [arcsec]'}}, yaxis: {range: [0, 1.1], title: {text: 'p, q, T'}},
      hovermode: 'x unified', legend: {y: -0.2}});
  }
  draw();
}

/* -------------------------------------------------------- orbit distribution */
function tabOrbits(run, body) {
  const det = detail(run);
  if (!det || !det.orbit_dist) { body.append(noDetails(run)); return; }
  const od = det.orbit_dist;
  const total = od.w.reduce((a, b) => a + (b || 0), 0) || 1;
  const frac = t => od.w.reduce((s, w, i) => s + (od.type[i] === t ? w : 0), 0) / total;
  const nz = od.w.filter(w => w > 0).length;
  const stats = h('div', {}, TYPE_ORDER.map(t => stat(TYPE_NAMES[t], (100 * frac(t)).toFixed(1) + '%')),
    stat('orbits with weight > 0', `${nz} / ${od.w.length}`));
  const circEl = h('div', {class: 'plot', style: 'height:420px'});
  const famEl = h('div', {class: 'plot', style: 'height:420px'});
  const betaEl = h('div', {class: 'plot', style: 'height:320px'});
  const sigEl = h('div', {class: 'plot', style: 'height:320px'});
  const vcEl = h('div', {class: 'plot', style: 'height:320px'});
  let ycol = 'lmd_z';
  body.append(
    h('div', {class: 'card'}, h('h2', {}, 'Orbit weights by family'), stats,
      h('p', {class: 'note'}, 'Family per orbit bundle: the majority of DYNAMITE\'s per-dither classification (sign changes of the angular momentum components during integration). Weights are the NNLS solution of the selected model.')),
    h('div', {class: 'grid2'},
      h('div', {class: 'card'}, h('h2', {}, 'Circularity distribution'),
        h('div', {class: 'controls'}, h('label', {}, 'y ', h('select', {onchange: e => { ycol = e.target.value; drawCirc(); }},
          h('option', {value: 'lmd_z'}, 'λz (short-axis circularity)'), h('option', {value: 'lmd_x'}, 'λx (long-axis circularity)')))),
        circEl, h('p', {class: 'note'}, 'Orbit weight summed in bins of time-averaged radius and circularity λ = ⟨L⟩/(⟨r⟩ V_rms), as in DYNAMITE\'s orbit_plot. Tube orbits enter twice, once per sense of rotation.')),
      h('div', {class: 'card'}, h('h2', {}, 'Orbit families against radius'), famEl,
        h('p', {class: 'note'}, 'Fraction of the orbit weight in each radial bin carried by each family.'))),
    h('div', {class: 'grid3'},
      h('div', {class: 'card'}, h('h2', {}, 'Velocity anisotropy'), betaEl,
        h('p', {class: 'note'}, 'β_r = 1 − (σ_θ² + σ_φ²)/(2σ_r²) and β_z = 1 − σ_z²/σ_R², mass-weighted over each radial shell of the intrinsic moment grid (first octant).')),
      h('div', {class: 'card'}, h('h2', {}, 'Intrinsic dispersions'), sigEl),
      h('div', {class: 'card'}, h('h2', {}, 'Circular velocity (along the major axis)'), vcEl,
        h('p', {class: 'note'}, 'From the ported potential, scaled to the selected model\'s M/L. Only in the model that has 3D orbits (the best one).'))));
  function drawCirc() {
    const x = [], y = [], z = [];
    od.r.forEach((r, i) => { if (od.w[i] > 0 && r > 0) { x.push(log10(r)); y.push(od[ycol][i]); z.push(od.w[i]); } });
    plot(circEl, [{type: 'histogram2d', x, y, z, histfunc: 'sum', nbinsx: 40, nbinsy: 30,
      colorscale: plotlyScale([css('--surface-1')].concat(seqStops())),
      colorbar: {title: {text: 'weight', side: 'right'}, thickness: 10},
      hovertemplate: 'log r %{x}<br>λ %{y}<br>weight %{z:.3f}<extra></extra>'}],
    {xaxis: {title: {text: 'log₁₀ ⟨r⟩ [arcsec]'}}, yaxis: {title: {text: ycol === 'lmd_z' ? 'λz' : 'λx'}, range: [-1.05, 1.05]}});
  }
  function drawFam() {
    const lr = od.r.map(r => r > 0 ? log10(r) : null).filter(v => v != null);
    const lo = Math.floor(Math.min(...lr) * 4) / 4, hi = Math.ceil(Math.max(...lr) * 4) / 4;
    const nb = Math.max(4, Math.round((hi - lo) / 0.25));
    const edges = Array.from({length: nb + 1}, (_, i) => lo + i * (hi - lo) / nb);
    const sums = TYPE_ORDER.map(() => new Float64Array(nb)), tot = new Float64Array(nb);
    od.r.forEach((r, i) => {
      if (!(r > 0) || !(od.w[i] > 0)) return;
      const b = Math.min(nb - 1, Math.floor((log10(r) - lo) / (hi - lo) * nb));
      const ti = TYPE_ORDER.indexOf(od.type[i]);
      if (ti >= 0) sums[ti][b] += od.w[i];
      tot[b] += od.w[i];
    });
    const xc = edges.slice(0, -1).map((e, i) => Math.pow(10, (e + edges[i + 1]) / 2));
    const traces = TYPE_ORDER.map((t, ti) => ({type: 'scatter', mode: 'lines+markers', x: xc,
      y: Array.from(sums[ti]).map((s, b) => tot[b] > 0 ? s / tot[b] : null), name: TYPE_NAMES[t],
      line: {width: 2, color: typeColor(t)}, marker: {size: 8, color: typeColor(t), line: {width: 2, color: css('--surface-1')}},
      hovertemplate: `${TYPE_NAMES[t]}: %{y:.2f}<extra></extra>`})).filter(tr => tr.y.some(v => v > 0));
    traces.push({type: 'bar', x: xc, y: Array.from(tot), name: 'total weight', yaxis: 'y2', marker: {color: css('--grid')}, hoverinfo: 'skip', showlegend: false});
    plot(famEl, traces, {xaxis: {type: 'log', dtick: 1, anchor: 'y2', title: {text: '⟨r⟩ [arcsec]'}}, yaxis: {range: [0, 1], title: {text: 'weight fraction'}, domain: [0.3, 1]},
      yaxis2: {domain: [0, 0.22], title: {text: 'weight'}, gridcolor: css('--grid'), tickfont: {size: 10}}, hovermode: 'x unified', legend: {y: -0.25}});
  }
  function drawProfiles() {
    const ip = det.intrinsic;
    if (!ip) { betaEl.replaceChildren(h('p', {class: 'muted'}, 'No intrinsic moments.')); sigEl.replaceChildren(); return; }
    plot(betaEl, [
      {type: 'scatter', mode: 'lines+markers', x: ip.r, y: ip.beta_r, name: 'β_r', line: {width: 2, color: css('--series-1')}, marker: {size: 8}},
      {type: 'scatter', mode: 'lines+markers', x: ip.r, y: ip.beta_z, name: 'β_z', line: {width: 2, color: css('--series-2')}, marker: {size: 8}},
    ], {xaxis: {type: 'log', dtick: 1, title: {text: 'r [arcsec]'}}, yaxis: {title: {text: 'β'}, zeroline: true}, hovermode: 'x unified', legend: {y: -0.3}});
    plot(sigEl, [
      ['sigma_r', 'σ_r', 1], ['sigma_th', 'σ_θ', 2], ['sigma_ph', 'σ_φ', 3], ['mean_vph', '⟨v_φ⟩', 4]].map(([k, l, s]) =>
      ({type: 'scatter', mode: 'lines+markers', x: ip.r, y: ip[k], name: l, line: {width: 2, color: css('--series-' + s), dash: k.startsWith('mean') ? 'dot' : 'solid'}, marker: {size: 7}})),
    {xaxis: {type: 'log', dtick: 1, title: {text: 'r [arcsec]'}}, yaxis: {title: {text: 'km/s'}}, hovermode: 'x unified', legend: {y: -0.3}});
  }
  function drawVc() {
    const o = det.orbits3d;
    if (!o) { vcEl.replaceChildren(h('p', {class: 'muted'}, 'Select the best model to see its circular-velocity curve.')); return; }
    const r = o.vc.r;
    plot(vcEl, [['total', 'total', 'text-primary'], ['stars', 'stars', 'series-1'], ['bh', 'black hole', 'series-2'], ['dm', 'dark halo', 'series-3']].map(([k, l, c]) =>
      ({type: 'scatter', mode: 'lines', x: r, y: o.vc.x[k], name: l, line: {width: 2, color: css('--' + c)}})),
    {xaxis: {type: 'log', dtick: 1, title: {text: 'r [arcsec]'}}, yaxis: {title: {text: 'v_c [km/s]'}}, hovermode: 'x unified', legend: {y: -0.3}});
  }
  drawCirc(); drawFam(); drawProfiles(); drawVc();
}

/* --------------------------------------------------------------- 3D orbits */
function tabOrbits3d(run, body) {
  let det = detail(run);
  if (!det || !det.orbits3d) {
    const bestDet = run.details && run.details[run.best_row];
    if (bestDet && bestDet.orbits3d) det = bestDet;
  }
  if (!det || !det.orbits3d) { body.append(h('div', {class: 'card'}, h('p', {class: 'secondary'}, '3D orbits were integrated for the best model only, and this run has none.'))); return; }
  if (typeof THREE === 'undefined') { body.append(h('div', {class: 'card warn'}, 'three.js did not load (offline?).')); return; }
  const O = det.orbits3d;
  const orbits = O.orbits.map((o, k) => Object.assign({k}, o));
  const ang = O.angles;
  const deg = Math.PI / 180;
  const th = ang.theta * deg, ph = ang.phi * deg, ps = ang.psi * deg;
  // sky basis in intrinsic coordinates (projection_n in orblib_f): x', y', LOS
  const ex = new THREE.Vector3(-Math.sin(ph), Math.cos(ph), 0);
  const ey = new THREE.Vector3(-Math.cos(th) * Math.cos(ph), -Math.cos(th) * Math.sin(ph), Math.sin(th));
  const ez = new THREE.Vector3(Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th));
  // rotation by psi into the frame of the data maps (commented formula in project_n)
  const skyRight = ex.clone().multiplyScalar(Math.sin(ps)).add(ey.clone().multiplyScalar(-Math.cos(ps)));
  const skyUp = ex.clone().multiplyScalar(Math.cos(ps)).add(ey.clone().multiplyScalar(Math.sin(ps)));

  const rs = orbits.map(o => o.r).filter(r => r > 0);
  const lrMin = Math.floor(log10(Math.min(...rs)) * 10) / 10, lrMax = Math.ceil(log10(Math.max(...rs)) * 10) / 10;
  const state = {n: Math.min(60, orbits.length), lo: lrMin, hi: lrMax, color: 'type', mirror: false, animate: false,
    types: new Set([1, 2, 3, 4, 5, 0]), solo: null, shells: true};

  const sceneEl = h('div', {class: 'scene'});
  const overlay = h('div', {class: 'overlay'});
  const legend = h('div', {class: 'legend3d'});
  sceneEl.append(overlay, legend);
  const listEl = h('div', {class: 'tablewrap'});
  const numIn = h('input', {type: 'range', min: 1, max: orbits.length, value: state.n, oninput: e => { state.n = +e.target.value; nLbl.textContent = state.n; rebuild(); }});
  const nLbl = h('span', {}, state.n);
  const rLo = h('input', {type: 'range', min: lrMin, max: lrMax, step: 0.05, value: lrMin, oninput: e => { state.lo = +e.target.value; rLbl(); rebuild(true); }});
  const rHi = h('input', {type: 'range', min: lrMin, max: lrMax, step: 0.05, value: lrMax, oninput: e => { state.hi = +e.target.value; rLbl(); rebuild(true); }});
  const rTxt = h('span', {class: 'small secondary'});
  const rLbl = () => { rTxt.textContent = `${fmt(Math.pow(10, state.lo), 3)}″ – ${fmt(Math.pow(10, state.hi), 3)}″`; };
  rLbl();
  const typeBoxes = TYPE_ORDER.map(t => h('label', {}, h('input', {type: 'checkbox', checked: true, onchange: e => {
    e.target.checked ? state.types.add(t) : state.types.delete(t); rebuild();
  }}), h('span', {class: 'swatch', style: `background:${typeColor(t)}`}), TYPE_NAMES[t]));
  const viewBtns = [['sky', 'Sky (as observed)'], ['x', 'Along x'], ['y', 'Along y'], ['z', 'Along z (face-on)'], ['iso', 'Oblique']]
    .map(([v, l]) => h('button', {onclick: () => setView(v)}, l));
  const side = h('div', {class: 'side3d'},
    h('div', {class: 'card'}, h('h3', {}, 'Selection'),
      h('div', {class: 'controls'}, h('label', {}, 'Top ', numIn, nLbl, ' orbits by weight')),
      h('div', {class: 'controls'}, h('label', {}, 'log ⟨r⟩ from ', rLo)), h('div', {class: 'controls'}, h('label', {}, 'to ', rHi), rTxt),
      h('div', {class: 'controls', style: 'flex-direction:column;align-items:flex-start;gap:4px'}, typeBoxes)),
    h('div', {class: 'card'}, h('h3', {}, 'Display'),
      h('div', {class: 'controls'}, h('label', {}, 'Colour ', h('select', {onchange: e => { state.color = e.target.value; rebuild(); }},
        [['type', 'orbit family'], ['r', 'radius'], ['lmd_z', 'λz'], ['w', 'weight']].map(([v, l]) => h('option', {value: v}, l))))),
      h('div', {class: 'controls'},
        h('label', {}, h('input', {type: 'checkbox', onchange: e => { state.mirror = e.target.checked; rebuild(); }}), '8-fold mirror (as DYNAMITE)'),
        h('label', {}, h('input', {type: 'checkbox', checked: true, onchange: e => { state.shells = e.target.checked; rebuild(); }}), 'isodensity shells'),
        h('label', {}, h('input', {type: 'checkbox', onchange: e => { state.animate = e.target.checked; }}), 'animate')),
      h('div', {class: 'controls'}, viewBtns)),
    h('div', {class: 'card'}, h('h3', {}, 'Orbits shown'), h('p', {class: 'note', style: 'margin-top:0'}, 'Click a row to show only that orbit; click again to show all.'), listEl));
  body.append(h('div', {class: 'card'},
    h('h2', {}, 'Orbits in the intrinsic (deprojected) frame'),
    h('p', {class: 'secondary small'}, `Model ${det.directory.split('/output/models/')[1] || det.directory}. x is the long axis, y the intermediate, z the short axis. ${O.orbits.length} most heavily weighted orbit bundles (${(100 * O.weight_fraction).toFixed(1)}% of the total weight; ${O.n_nonzero} of ${O.n_total} orbits have non-zero weight). The arrow points to the observer at (θ, φ, ψ) = (${fmt(ang.theta, 1)}°, ${fmt(ang.phi, 1)}°, ${fmt(ang.psi, 1)}°). Drag to rotate, scroll to zoom, right-drag to pan.`),
    h('div', {class: 'viewer3d'}, sceneEl, side),
    h('p', {class: 'note'}, O.note + ' Each orbit is the centre dither of its bundle; line opacity scales with weight. The animation advances every orbit through its 15 periods in the same time, so it shows the sense of motion, not relative speeds.')));

  // ---- three.js scene
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setClearColor(new THREE.Color(css('--scene-bg')));
  sceneEl.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1e6);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const world = new THREE.Group();
  scene.add(world);
  const orbitGroup = new THREE.Group(), helperGroup = new THREE.Group();
  world.add(orbitGroup, helperGroup);
  const geoms = orbits.map(o => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(decodeOrbit(o), 3));
    return g;
  });
  const dot = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({size: 5, sizeAttenuation: false, vertexColors: true}));
  scene.add(dot);
  let shown = [], extent = 1, frame = 0;

  function colorFor(o, rng) {
    if (state.color === 'type') return new THREE.Color(typeColor(o.type));
    let t;
    if (state.color === 'r') t = (log10(o.r) - rng.lr0) / (rng.lr1 - rng.lr0 || 1);
    else if (state.color === 'w') t = (log10(o.w) - rng.lw0) / (rng.lw1 - rng.lw0 || 1);
    else t = (o.lmd_z + 1) / 2;
    const c = (state.color === 'lmd_z' ? makeCmap(divStops()) : makeCmap(isDark() ? seqStops().slice(0, -1) : seqStops().slice(2)))(t);
    return new THREE.Color(`rgb(${c[0]},${c[1]},${c[2]})`);
  }
  function rebuild(refit) {
    orbitGroup.clear(); helperGroup.clear();
    const cand = orbits.slice(0, state.n).filter(o => state.types.has(o.type) && log10(o.r) >= state.lo - 1e-9 && log10(o.r) <= state.hi + 1e-9);
    shown = state.solo != null ? cand.filter(o => o.k === state.solo) : cand;
    const rng = {lr0: Math.min(...shown.map(o => log10(o.r))), lr1: Math.max(...shown.map(o => log10(o.r))),
      lw0: Math.min(...shown.map(o => log10(o.w))), lw1: Math.max(...shown.map(o => log10(o.w)))};
    const wmax = Math.max(...shown.map(o => o.w), 1e-12);
    const signs = state.mirror ? [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]] : [[1, 1, 1]];
    extent = 0;
    for (const o of shown) {
      const mat = new THREE.LineBasicMaterial({color: colorFor(o, rng), transparent: true,
        opacity: state.solo != null ? 1 : 0.25 + 0.75 * Math.sqrt(o.w / wmax)});
      for (const s of signs) {
        const line = new THREE.Line(geoms[o.k], mat);
        line.scale.set(...s);
        orbitGroup.add(line);
      }
      extent = Math.max(extent, o.scale);
    }
    extent = extent || 1;
    addHelpers();
    const pos = new Float32Array(shown.length * 3), col = new Float32Array(shown.length * 3);
    shown.forEach((o, i) => { const c = colorFor(o, rng); col.set([c.r, c.g, c.b], 3 * i); });
    dot.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    dot.geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
    legend.replaceChildren(...(state.color === 'type' ? TYPE_ORDER.filter(t => shown.some(o => o.type === t)).map(t =>
      h('span', {}, h('span', {class: 'swatch', style: `background:${typeColor(t)}`}), TYPE_NAMES[t])) :
      [h('span', {class: 'secondary'}, {r: 'colour: log radius (pale = ' + (isDark() ? 'outer)' : 'inner)'), w: 'colour: log weight (pale = ' + (isDark() ? 'high)' : 'low)'), lmd_z: 'colour: λz (blue −1 … red +1)'}[state.color])]));
    overlay.textContent = `${shown.length} orbits · ${(100 * shown.reduce((s, o) => s + o.w, 0)).toFixed(1)}% of weight · extent ${fmt(extent, 3)}″`;
    drawList();
    if (refit) fit();
  }
  function textSprite(text, color) {
    const c = document.createElement('canvas'); c.width = 256; c.height = 64;
    const ctx = c.getContext('2d'); ctx.font = '28px system-ui'; ctx.fillStyle = color; ctx.fillText(text, 4, 40);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(c), depthTest: false, transparent: true}));
    sp.scale.set(extent * 0.32, extent * 0.08, 1);
    return sp;
  }
  function addHelpers() {
    const L = extent * 1.15, ink = css('--text-muted');
    const axMat = new THREE.LineBasicMaterial({color: new THREE.Color(ink)});
    [['x (long)', [1, 0, 0]], ['y (intermediate)', [0, 1, 0]], ['z (short)', [0, 0, 1]]].forEach(([lbl, v]) => {
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...v.map(c => -c * L)), new THREE.Vector3(...v.map(c => c * L))]);
      helperGroup.add(new THREE.Line(g, axMat));
      const sp = textSprite(lbl, ink); sp.position.set(...v.map(c => c * L * 1.08)); helperGroup.add(sp);
    });
    const arrow = new THREE.ArrowHelper(ez, new THREE.Vector3(0, 0, 0), L * 1.1, new THREE.Color(css('--series-2')), L * 0.08, L * 0.04);
    helperGroup.add(arrow);
    const spo = textSprite('to observer', css('--series-2')); spo.position.copy(ez.clone().multiplyScalar(L * 1.25)); helperGroup.add(spo);
    // isodensity shells from the best model's p(r), q(r): three principal-plane ellipses each
    if (state.shells && run.qpt) {
      const q = run.qpt, radii = [0.3, 0.6, 0.95].map(f => extent * f);
      const mat = new THREE.LineBasicMaterial({color: new THREE.Color(css('--text-secondary')), transparent: true, opacity: 0.35});
      for (const R of radii) {
        const i = Math.max(0, Math.min(q.r.length - 1, q.r.findIndex(r => r >= R)));
        const pp = q.best.p[i] ?? 1, qq = q.best.q[i] ?? 1;
        const ell = (fn) => { const pts = []; for (let k = 0; k <= 96; k++) { const t = 2 * Math.PI * k / 96; pts.push(fn(Math.cos(t), Math.sin(t))); } return new THREE.BufferGeometry().setFromPoints(pts); };
        helperGroup.add(new THREE.Line(ell((c, s) => new THREE.Vector3(R * c, R * pp * s, 0)), mat));
        helperGroup.add(new THREE.Line(ell((c, s) => new THREE.Vector3(R * c, 0, R * qq * s)), mat));
        helperGroup.add(new THREE.Line(ell((c, s) => new THREE.Vector3(0, R * pp * c, R * qq * s)), mat));
      }
    }
  }
  function setView(v) {
    const D = extent * 4.2;
    const dirs = {x: [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 1)], y: [new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)],
      z: [new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0)], sky: [ez, skyUp],
      iso: [new THREE.Vector3(1, 0.8, 0.6).normalize(), new THREE.Vector3(0, 0, 1)]};
    const [d, up] = dirs[v];
    controls.target.set(0, 0, 0);
    camera.up.copy(up);
    camera.position.copy(d.clone().multiplyScalar(D));
    camera.near = extent / 1000; camera.far = extent * 100; camera.updateProjectionMatrix();
    controls.update();
  }
  function fit() { setView('iso'); }
  function drawList() {
    listEl.replaceChildren(h('table', {class: 'data'},
      h('thead', {}, h('tr', {}, ['', 'family', '⟨r⟩ ″', 'λz', 'weight %', 'pro/retro'].map(t => h('th', {}, t)))),
      h('tbody', {}, orbits.slice(0, state.n).filter(o => state.types.has(o.type)).map(o => h('tr', {
        class: 'clickable' + (state.solo === o.k ? ' sel' : ''),
        onclick: () => { state.solo = state.solo === o.k ? null : o.k; rebuild(); }},
      h('td', {}, h('span', {class: 'swatch', style: `background:${typeColor(o.type)}`})),
      h('td', {class: 'left'}, `${TYPE_NAMES[o.type]} (${o.lib})`), h('td', {}, fmt(o.r, 3)), h('td', {}, fmt(o.lmd_z, 2)),
      h('td', {}, (100 * o.w).toFixed(2)), h('td', {}, o.lib === 'tube' ? `${(100 * o.w_pro).toFixed(1)}/${(100 * o.w_retro).toFixed(1)}` : '–'))))));
  }
  function resize() {
    const w = sceneEl.clientWidth, hh = sceneEl.clientHeight;
    renderer.setSize(w, hh, false);
    renderer.domElement.style.width = w + 'px'; renderer.domElement.style.height = hh + 'px';
    camera.aspect = w / hh; camera.updateProjectionMatrix();
  }
  let raf;
  function loop() {
    raf = requestAnimationFrame(loop);
    controls.update();
    dot.visible = state.animate;
    if (state.animate && shown.length) {
      frame++;
      const pos = dot.geometry.attributes.position;
      shown.forEach((o, i) => {
        const arr = geoms[o.k].attributes.position.array, n = arr.length / 3;
        const j = Math.floor((frame * 0.5) % n);
        pos.array[3 * i] = arr[3 * j]; pos.array[3 * i + 1] = arr[3 * j + 1]; pos.array[3 * i + 2] = arr[3 * j + 2];
      });
      pos.needsUpdate = true;
    }
    renderer.render(scene, camera);
  }
  const ro = new ResizeObserver(resize); ro.observe(sceneEl);
  rebuild(); resize(); fit(); loop();
  cleanup.push(() => { cancelAnimationFrame(raf); ro.disconnect(); renderer.dispose(); geoms.forEach(g => g.dispose()); });
}

/* ------------------------------------------------------------ plots/config */
function tabPlots(run, body) {
  if (!run.plots || !run.plots.length) { body.append(h('div', {class: 'card'}, h('p', {class: 'secondary'}, 'No PNG plots in this run\'s plots/ directory.'))); return; }
  body.append(h('div', {class: 'card'}, h('p', {class: 'secondary small', style: 'margin-top:0'}, 'The PNGs DYNAMITE (or your scripts) wrote to the run\'s plots/ directory, copied as they were at extraction. They may predate the latest models.'),
    h('div', {class: 'gallery'}, run.plots.map(p => {
      const src = `data/${encodeURIComponent(run.name)}/plots/${encodeURIComponent(p)}`;
      return h('figure', {}, h('a', {href: src, target: '_blank'}, h('img', {src, loading: 'lazy', alt: p})), h('figcaption', {}, p));
    }))));
}
function tabConfig(run, body) {
  body.append(h('div', {class: 'card'}, h('h2', {}, run.config_file), h('pre', {class: 'yaml'}, run.config_text)));
}

/* ================================================================= compare */
async function pageCompare(params) {
  const all = DYNX.manifestData || [];
  let names = (params.get('runs') || '').split(',').filter(Boolean).map(decodeURIComponent);
  if (!names.length) names = JSON.parse(localStorageGet('dynx-compare') || '[]');
  const picker = h('div', {class: 'checklist'}, all.filter(r => r.n_done > 0).map(r => h('label', {},
    h('input', {type: 'checkbox', value: r.name, checked: names.includes(r.name)}), ' ', r.name, h('span', {class: 'muted small'}, ` (${r.n_done} done)`))));
  app.append(h('div', {class: 'card'}, h('h1', {}, 'Compare runs'),
    h('p', {class: 'secondary'}, 'Best-fit parameters, intrinsic shapes and kinematic residuals side by side.'),
    picker, h('div', {class: 'controls', style: 'margin-top:10px'}, h('button', {class: 'primary', onclick: () => {
      const sel = [...picker.querySelectorAll('input:checked')].map(i => i.value);
      localStorageSet('dynx-compare', JSON.stringify(sel));
      location.hash = '#/compare?runs=' + sel.map(encodeURIComponent).join(',');
    }}, 'Compare'))));
  if (!names.length) return;
  const status = h('p', {class: 'muted'}, 'Loading…');
  app.append(status);
  const runs = [];
  for (const n of names) { try { runs.push(await loadRun(n)); } catch (e) { status.textContent = e.message; } }
  status.remove();
  if (!runs.length) return;

  // parameter table
  const pars = ['m-bh', 'f-dh', 'q-stars', 'p-stars', 'u-stars', 'ml', 'theta', 'phi', 'psi'];
  app.append(h('div', {class: 'card'}, h('h2', {}, 'Best models'), h('div', {class: 'tablewrap'}, h('table', {class: 'data'},
    h('thead', {}, h('tr', {}, ['Run', 'which χ²', 'best', 'χ²/N', ...pars.map(parLabel), 'kinematics'].map(t => h('th', {}, t)))),
    h('tbody', {}, runs.map(r => {
      const b = r.best_row;
      const N = r.n_obs2 ? r.n_obs2 / 2 : null;
      return h('tr', {}, h('td', {}, h('a', {href: `#/run/${encodeURIComponent(r.name)}`}, r.name)), h('td', {}, r.which_chi2),
        h('td', {}, b != null ? fmtChi(cval(r, b, r.which_chi2)) : '–'), h('td', {}, b != null && N ? (cval(r, b, r.which_chi2) / N).toFixed(2) : '–'),
        pars.map(p => h('td', {}, b != null && (r.ci[p] != null) ? fmt(pval(r, b, p), 4) : '–')),
        h('td', {class: 'left'}, (r.geometry || []).map(g => `${g.name} (${g.n_bins})`).join(', ')));
    }))))));

  // qpt overlay
  const qEl = h('div', {class: 'plot', style: 'height:420px'});
  app.append(h('div', {class: 'card'}, h('h2', {}, 'Intrinsic shape of the best models'),
    h('p', {class: 'secondary small'}, 'Solid: p(r) = B/A. Dashed: q(r) = C/A. One colour per run, in the order listed above (up to 8).'), qEl));
  const traces = [];
  runs.slice(0, 8).forEach((r, i) => {
    if (!r.qpt) return;
    const c = css('--series-' + (i + 1));
    traces.push({type: 'scatter', mode: 'lines', x: r.qpt.r, y: r.qpt.best.p, name: r.name + ' p', legendgroup: r.name, line: {color: c, width: 2}});
    traces.push({type: 'scatter', mode: 'lines', x: r.qpt.r, y: r.qpt.best.q, name: r.name + ' q', legendgroup: r.name, line: {color: c, width: 2, dash: 'dash'}});
  });
  plot(qEl, traces, {xaxis: {type: 'log', dtick: 1, title: {text: 'r [arcsec]'}}, yaxis: {range: [0, 1.05], title: {text: 'axis ratio'}}, legend: {y: -0.25}});

  // kinematic comparison
  const withDet = runs.filter(r => r.detailRows.length);
  if (withDet.length) {
    const opts = [];
    withDet.forEach(r => r.detailRows.forEach(row => opts.push([r, row])));
    const optEls = () => opts.map(([r, row], i) => h('option', {value: i}, `${r.name} · ${modelLabel(r, row)}`));
    let a = 0, b = Math.min(1, opts.length - 1);
    const bestIdx = r => opts.findIndex(([rr, row]) => rr === r && row === r.best_row);
    if (withDet.length > 1) { a = bestIdx(withDet[0]); b = bestIdx(withDet[1]); }
    const selA = h('select', {onchange: e => { a = +e.target.value; draw(); }}, optEls());
    const selB = h('select', {onchange: e => { b = +e.target.value; draw(); }}, optEls());
    selA.value = a; selB.value = b;
    const setSel = h('select', {onchange: draw});
    let zoom = 1;
    const mapsEl = h('div');
    app.append(h('div', {class: 'card'}, h('h2', {}, 'Kinematic residuals: model A against model B'),
      h('p', {class: 'secondary small'}, 'Same kinematic set from two models (any runs). Each model is shown against its own run\'s data and binning; residual scale ±3σ.'),
      h('div', {class: 'controls'}, h('label', {}, 'A ', selA), h('label', {}, 'B ', selB)),
      h('div', {class: 'controls'}, h('label', {}, 'Kinematic set ', setSel),
        h('label', {}, 'Zoom ', h('input', {type: 'range', min: 1, max: 12, step: 0.5, value: 1, oninput: e => { zoom = +e.target.value; draw(); }}))),
      mapsEl));
    function draw() {
      const [ra, rowA] = opts[a], [rb, rowB] = opts[b];
      const ka = ra.details[rowA].kin, kb = rb.details[rowB].kin;
      const common = ka.map(k => k.name).filter(n => kb.some(k => k.name === n));
      const cur = setSel.value;
      setSel.replaceChildren(...common.map(n => h('option', {value: n, selected: n === cur}, n)));
      if (!common.length) { mapsEl.replaceChildren(h('p', {class: 'warn'}, 'These two models share no kinematic set name.')); return; }
      const name = setSel.value || common[0];
      const KA = ka.find(k => k.name === name), KB = kb.find(k => k.name === name);
      const GA = ra.geomByName[name], GB = rb.geomByName[name];
      const cols = [
        {label: 'Data (A)', get: m => KA.data[m], kind: 'value', src: KA, geom: GA},
        {label: 'Model A', get: m => KA.model[m], kind: 'value', src: KA, geom: GA},
        {label: 'Model B', get: m => KB.model[m], kind: 'value', src: KB, geom: GB},
        {label: 'Resid A', get: m => resid(KA, m), kind: 'resid', src: KA, geom: GA},
        {label: 'Resid B', get: m => resid(KB, m), kind: 'resid', src: KB, geom: GB},
      ];
      mapsEl.replaceChildren(
        h('table', {class: 'data', style: 'margin-bottom:10px'}, h('thead', {}, h('tr', {}, ['', 'v', 'σ', 'h3', 'h4', 'total'].map(t => h('th', {}, t)))),
          h('tbody', {}, [['A', KA], ['B', KB]].map(([l, K]) => h('tr', {}, h('td', {}, `${l}: map χ² (${K.n_bins} bins)`),
            ['v', 'sigma', 'h3', 'h4'].map(m => h('td', {}, fmtChi(K.chi2[m]))), h('td', {}, fmtChi(K.chi2_total)))))),
        kinMapGrid(GA, KA, {zoom, columns: cols}));
    }
    draw();
  }
}

/* ------------------------------------------------------------------- boot */
$('#theme-toggle').addEventListener('click', () => {
  const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.dataset.theme = cur === 'dark' ? 'light' : 'dark';
  localStorageSet('dynx-theme', document.documentElement.dataset.theme);
  route();
});
const savedTheme = localStorageGet('dynx-theme');
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
route();
