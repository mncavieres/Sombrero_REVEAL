"""Write dynamite_explorer/data/manifest.js from the extracted run.js files.

One summary per run for the overview page: best model of the run's
which_chi2, its parameters (log10 for logarithmic ones, as in the YAML),
the parameter ranges within Δχ² <= sqrt(2N), and chi2/N.

    python build_manifest.py [data_dir]
"""
import json
import math
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / 'dynamite_explorer' / 'data'
SHOWN = ['m-bh', 'f-dh', 'q-stars', 'p-stars', 'u-stars', 'ml', 'theta', 'phi', 'psi']


def load(js):
    text = js.read_text()
    start, end = text.index('('), text.rindex(')')
    return json.loads(text[start + 1:end])


def summarise(d):
    cols = d['models']['columns']
    ci = {c: i for i, c in enumerate(cols)}
    rows = d['models']['rows']
    logpar = {p['name'] for p in d['parameters'] if p.get('logarithmic')}
    which = d['which_chi2']

    def val(r, p):
        v = rows[r][ci[p]] if p in ci else None
        if v is None:
            return None
        return math.log10(v) if p in logpar and v > 0 else v

    out = dict(name=d['name'], path=d['path'], n_models=d['n_models'], n_done=d['n_done'],
               which_chi2=which, extracted=d.get('extracted'),
               kin=[dict(name=g['name'], n_bins=g['n_bins']) for g in d.get('geometry') or []],
               n_details=len(d.get('details') or {}),
               has_orbits=any('orbits3d' in v for v in (d.get('details') or {}).values()))
    b = d.get('best_row')
    if b is None:
        return out
    chi_min = rows[b][ci[which]]
    out['chi2_sel'] = chi_min
    out['best'] = {p: val(b, p) for p in SHOWN}
    n_obs2 = d.get('n_obs2')
    if n_obs2:
        out['chi2_red'] = chi_min / (n_obs2 / 2)
        thr = math.sqrt(n_obs2)
        done = [i for i, r in enumerate(rows) if r[ci['all_done']] in (True, 'True')
                and r[ci[which]] is not None and r[ci[which]] - chi_min <= thr]
        out['range1'] = {}
        for p in SHOWN:
            vals = [val(i, p) for i in done]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if vals:
                out['range1'][p] = [min(vals), max(vals)]
    return out


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    runs = []
    for js in sorted(data.glob('*/run.js')):
        try:
            runs.append(summarise(load(js)))
        except Exception as exc:
            print(f'skipping {js.parent.name}: {exc}')
    (data / 'manifest.js').write_text('DYNX.manifest(' + json.dumps(runs, separators=(',', ':')) + ');\n')
    print(f'{data / "manifest.js"}: {len(runs)} runs')


if __name__ == '__main__':
    main()
