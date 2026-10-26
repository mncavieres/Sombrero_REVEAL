"""Extract one DYNAMITE run into a compact data file for the HTML explorer.

Run on the cluster with the DYNAMITE environment, for a directory that holds
``all_models.ecsv`` (the run's output directory)::

    python extract_run.py --run-dir /path/to/run/output --name my_run \
        --out /path/to/explorer_data --work /path/to/scratch

Writes ``<out>/<name>/run.js`` and copies the run's existing PNG plots to
``<out>/<name>/plots/``.

Read-only on the run. DYNAMITE's ``Configuration`` rewrites
``all_models.ecsv`` and deletes orbit libraries of unfinished models when it
loads an output directory, so it is never pointed at the real run. Each model
is read from a sandbox under ``--work``: a copy of the input directory, a
filtered ``all_models.ecsv``, copies of the small orblib files, and symlinks
to the large ``*.bz2`` orbit libraries (DYNAMITE only reads those, by
decompressing them into the sandbox).

Contents of ``run.js`` (``DYNX.add({...})``):

* the full ``all_models`` table plus the viewing angles implied by (p, q, u);
* parsed configuration, number of observables, best model;
* for the best ``--n-best`` models (all models if the run has at most
  ``--all-below`` finished ones): per-bin data and model Gauss-Hermite
  kinematics for every kinematic set, intrinsic velocity-moment profiles,
  and the orbit-weight distribution (r, circularities, orbit type, weight);
* the bin geometry of every kinematic set (pixel grid of bin ids);
* intrinsic p(r), q(r), T(r) for the best model and for the models within
  3 sqrt(2 N_obs) of the minimum chi2 (as in DYNAMITE's ``qpu_plot``);
* re-integrated 3D orbits of the most heavily weighted orbit bundles of the
  best model, with circular-velocity curves (see ``orbit_potential.py``).
"""
import argparse
import base64
import datetime
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table
from scipy.io import FortranFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orbit_potential as op  # noqa: E402

MAX_PLOT_BYTES = 4_000_000
ORBIT_TYPES = {1: 'x-tube', 2: 'y-tube', 3: 'z-tube', 4: 'box', 5: 'irregular'}


# ----------------------------------------------------------------- helpers
def clean(obj):
    """Make numpy/astropy values JSON-safe (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def rounded(a, sig=6):
    """List of floats rounded to ``sig`` significant digits (smaller JSON)."""
    a = np.asarray(a, dtype=float)
    out = []
    for v in a.ravel():
        out.append(float(f'{v:.{sig}g}') if np.isfinite(v) else None)
    return out


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode('ascii')


def find_config(run_dir, table):
    """The YAML DYNAMITE copied into a model directory, else one nearby."""
    models = run_dir / 'models'
    for d in table['directory']:
        cand = sorted((models / str(d)).glob('*.yaml'))
        if cand:
            return cand[0]
    for base in (run_dir, run_dir.parent):
        for c in sorted(base.glob('*.yaml')):
            try:
                cfg = yaml.safe_load(open(c))
                out = Path(cfg['io_settings']['output_directory']).resolve()
                if out == run_dir.resolve():
                    return c
            except Exception:
                pass
    raise FileNotFoundError(f'no DYNAMITE config found for {run_dir}')


def find_input_dir(cfg, run_dir):
    """The run's input directory; second value says if a substitute was used."""
    configured = Path(cfg['io_settings']['input_directory'])
    cands = [configured, run_dir / 'input', run_dir.parent / 'input']
    for c in cands:
        if c.is_dir():
            return c, None
    # the run whose input the config names may since have been renamed,
    # e.g. 'small_nirspec_muse_run' -> 'small_nirspec_muse_run copy'
    base = configured.parent
    for c in sorted(base.parent.glob(base.name + '*/' + configured.name)):
        if c.is_dir():
            return c, f'configured input {configured} is missing; used {c}'
    raise FileNotFoundError(f'input directory not found, tried {cands}')


def par_list(cfg):
    """Free/fixed parameters in DYNAMITE naming (e.g. 'm-bh', 'ml')."""
    out = []
    for comp, cdef in cfg['system_components'].items():
        for pname, pdef in (cdef.get('parameters') or {}).items():
            out.append(dict(name=f'{pname}-{comp}', **_pinfo(pdef)))
    for pname, pdef in (cfg.get('system_parameters') or {}).items():
        out.append(dict(name=pname, **_pinfo(pdef)))
    return out


def _pinfo(pdef):
    g = pdef.get('par_generator_settings', {}) or {}
    return dict(fixed=bool(pdef.get('fixed', False)),
                logarithmic=bool(pdef.get('logarithmic', False)),
                value=pdef.get('value'), lo=g.get('lo'), hi=g.get('hi'),
                step=g.get('step'), minstep=g.get('minstep'),
                latex=pdef.get('LaTeX', ''))


def viewing_angles(p, q, u, qobs):
    """Vectorised port of TriaxialVisibleComponent.triax_pqu2tpp."""
    p, q, u = (np.asarray(v, dtype=float) for v in (p, q, u))
    u = np.where(u == 1, u * (1 - np.finfo(float).epsneg), u)
    u = np.where(np.isclose(u, p), p, u)
    p2, q2, u2, o2 = p ** 2, q ** 2, u ** 2, qobs ** 2
    with np.errstate(all='ignore'):
        t = (1 - p2) / (1 - q2)
        ok = ((t >= 0) & (t <= 1) & (np.maximum(q / qobs, p) < u)
              & (u <= np.minimum(p / qobs, 1)) & (p > 0) & (q > 0) & (u > 0) & (u <= 1))
        w1 = (u2 - q2) * (o2 * u2 - q2) / ((1 - q2) * (p2 - q2))
        w2 = (u2 - p2) * (p2 - o2 * u2) * (1 - q2) / ((1 - u2) * (1 - o2 * u2) * (p2 - q2))
        w3 = (1 - o2 * u2) * (p2 - o2 * u2) * (u2 - q2) / ((1 - u2) * (u2 - p2) * (o2 * u2 - q2))
        theta = np.where(ok & (w1 >= 0), np.degrees(np.arccos(np.sqrt(w1))), np.nan)
        phi = np.where(ok & (w2 >= 0), np.degrees(np.arctan(np.sqrt(w2))), np.nan)
        psi = np.where(ok & (w3 >= 0), 180 - np.degrees(np.arctan(np.sqrt(w3))), np.nan)
    return theta, phi, psi


def intrinsic_shape_profile(mge, theta, phi, psi, r):
    """p(r), q(r) of isodensity surfaces along the principal axes.

    Same construction as DYNAMITE's ``Plotter.pqintr_mge_v2``: find where the
    density along y (z) equals the density at radius r along x.
    """
    qobs, sig = np.asarray(mge['q']), np.asarray(mge['sigma'])
    pj, qj, ratio = op.deproject_mge(qobs, theta, phi, psi, np.asarray(mge['PA_twist']))
    sig_i = sig * ratio
    dens = np.asarray(mge['I']) * qobs * sig ** 2 / (sig_i ** 3 * pj * qj)
    rr = np.geomspace(r.min() / 3, r.max() * 3, 4000)

    def rho(x, scale):
        return np.sum(dens[None] * np.exp(-0.5 * (x[:, None] / (sig_i * scale)[None]) ** 2), 1)

    rx = rho(r, np.ones_like(pj))
    ry, rz = rho(rr, pj), rho(rr, qj)
    # densities decrease outwards: interpolate radius as a function of log rho
    y_iso = np.interp(-np.log(rx), -np.log(ry), rr)
    z_iso = np.interp(-np.log(rx), -np.log(rz), rr)
    p_r, q_r = y_iso / r, z_iso / r
    with np.errstate(all='ignore'):
        T = (1 - p_r ** 2) / (1 - q_r ** 2)
    return p_r, q_r, T


# ----------------------------------------------------------------- sandbox
def build_sandbox(run_dir, table, rows, cfg, input_dir, work):
    """Isolated copy of the run for the models in ``rows``. Returns config path."""
    sb = Path(tempfile.mkdtemp(prefix='dynx_', dir=work))
    shutil.copytree(input_dir, sb / 'input', symlinks=False)
    out = sb / 'output'
    (out / 'models').mkdir(parents=True)
    (out / 'plots').mkdir()
    orblib_dirs = sorted({str(table['directory'][i]).split('/')[0] for i in rows})
    keep = [i for i, d in enumerate(table['directory'])
            if str(d).split('/')[0] in orblib_dirs and bool(table['all_done'][i])]
    for i in rows:
        if i not in keep:
            keep.append(i)
    keep = sorted(keep)
    sub = table[keep]
    sub.write(out / cfg['io_settings']['all_models_file'], format='ascii.ecsv')
    p_mass = run_dir / 'mass_aper.ecsv'
    if p_mass.is_file():
        shutil.copy2(p_mass, out / 'mass_aper.ecsv')
    for od in orblib_dirs:
        src, dst = run_dir / 'models' / od, out / 'models' / od
        (dst / 'datfil').mkdir(parents=True)
        shutil.copytree(src / 'infil', dst / 'infil')
        for f in (src / 'datfil').iterdir():
            if f.name.endswith('.tmp') or f.name.endswith('_done') or not f.is_file():
                continue
            if f.suffix in ('.bz2', '.npz'):
                os.symlink(f, dst / 'datfil' / f.name)
            elif f.stat().st_size < 50_000_000:
                shutil.copy2(f, dst / 'datfil' / f.name)
        for d in sub['directory']:
            d = str(d)
            if d.startswith(od + '/') and (run_dir / 'models' / d).is_dir():
                shutil.copytree(run_dir / 'models' / d, out / 'models' / d)
    cfg = dict(cfg)
    cfg['io_settings'] = dict(cfg['io_settings'])
    cfg['io_settings']['input_directory'] = str(sb / 'input') + '/'
    cfg['io_settings']['output_directory'] = str(out) + '/'
    cfg['multiprocessing_settings'] = dict(cfg.get('multiprocessing_settings') or {})
    cfg['multiprocessing_settings']['ncpus'] = 1
    cfg_path = sb / 'config.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return sb, cfg_path


# ----------------------------------------------------------- per model data
def read_orbtypes(mod_dir, fileroot, norb, ndith):
    """Orbit types per dither from datfil/<fileroot>_qgrid.dat.bz2, or None."""
    src = Path(mod_dir) / 'datfil' / f'{fileroot}_qgrid.dat.bz2'
    if not src.exists():
        return None
    tmp = Path(mod_dir) / 'datfil' / f'{fileroot}_qgrid_types.dat'
    subprocess.run(f'bunzip2 -c "{src}" > "{tmp}"', shell=True, check=True)
    try:
        f = FortranFile(str(tmp), 'r')
        f.read_ints(np.int32)
        f.read_ints(np.int32)
        for _ in range(3):
            f.read_reals(float)
        types = np.zeros((norb, ndith ** 3), dtype=int)
        for j in range(norb):
            f.read_ints(np.int32)
            types[j] = f.read_ints(np.int32)
            f.read_reals(float)
        f.close()
    finally:
        tmp.unlink()
    return types


def intrinsic_profiles(orblib, weights):
    """Mass-weighted spherical and cylindrical dispersion profiles vs r."""
    construct, grid = orblib.get_model_intrinsic_moment_constructor()
    m = construct(weights)
    dens = m[..., 0]
    x, y, z = m[..., 1], m[..., 2], m[..., 3]
    v1 = m[..., 4:7]
    M = np.zeros(dens.shape + (3, 3))
    M[..., 0, 0], M[..., 1, 1], M[..., 2, 2] = m[..., 7], m[..., 8], m[..., 9]
    M[..., 0, 1] = M[..., 1, 0] = m[..., 10]
    M[..., 1, 2] = M[..., 2, 1] = m[..., 11]
    M[..., 0, 2] = M[..., 2, 0] = m[..., 12]
    R = np.hypot(x, y)
    r = np.sqrt(R ** 2 + z ** 2)
    with np.errstate(all='ignore'):
        e_r = np.stack([x / r, y / r, z / r], -1)
        e_th = np.stack([x * z / (R * r), y * z / (R * r), -R / r], -1)
        e_ph = np.stack([-y / R, x / R, np.zeros_like(R)], -1)
        e_R = np.stack([x / R, y / R, np.zeros_like(R)], -1)
        e_z = np.zeros_like(e_r)
        e_z[..., 2] = 1

        def disp(e):
            mean = np.sum(v1 * e, -1)
            second = np.einsum('...i,...ij,...j->...', e, M, e)
            return mean, second - mean ** 2

        prof = {}
        w = np.where(np.isfinite(dens), dens, 0)
        shell = w.sum(axis=(1, 2))
        for name, e in (('r', e_r), ('th', e_th), ('ph', e_ph), ('R', e_R), ('z', e_z)):
            mean, var = disp(e)
            var = np.where(np.isfinite(var), var, 0)
            mean = np.where(np.isfinite(mean), mean, 0)
            prof[f'sigma_{name}'] = np.sqrt((w * var).sum(axis=(1, 2)) / shell)
            prof[f'mean_v{name}'] = (w * mean).sum(axis=(1, 2)) / shell
        rmean = (w * np.where(np.isfinite(r), r, 0)).sum(axis=(1, 2)) / shell
    edges = np.asarray(grid[0])
    out = dict(r=rounded(rmean), r_edges=rounded(edges), mass=rounded(shell),
               **{k: rounded(v) for k, v in prof.items()})
    sr, st, sp = prof['sigma_r'], prof['sigma_th'], prof['sigma_ph']
    with np.errstate(all='ignore'):
        out['beta_r'] = rounded(1 - (st ** 2 + sp ** 2) / (2 * sr ** 2))
        out['beta_z'] = rounded(1 - prof['sigma_z'] ** 2 / prof['sigma_R'] ** 2)
    return out


def bundle_table(orblib, weights, pars, mod_dir, kpa):
    """One row per orbit (weight index): library, bundle, type, r, circularity."""
    nE, nI2, nI3, nd = pars['nE'], pars['nI2'], pars['nI3'], pars['dithering']
    norb = nE * nI2 * nI3
    orblib.read_orbit_property_file()
    props = orblib.orb_properties
    r_arcsec = np.asarray(props['r'].to('km').value).mean(1) / kpa
    lmd_z = np.asarray(props['lmd_z'].value).mean(1)
    lmd_x = np.asarray(props['lmd_x'].value).mean(1)
    lmd = np.asarray(props['lmd'].value).mean(1)
    n = len(weights)
    lib = np.array([0] * (2 * norb) + [1] * norb)              # 0 tube, 1 box
    bundle = np.concatenate([np.repeat(np.arange(norb), 2), np.arange(norb)])
    flip = np.concatenate([np.tile([0, 1], norb), np.zeros(norb, int)])
    otype = np.zeros(n, int)
    for k, root in enumerate(('orblib', 'orblibbox')):
        try:
            t = read_orbtypes(mod_dir, root, norb, nd)
        except Exception:
            t = None
        if t is None:
            continue
        maj = np.array([np.bincount(row, minlength=6).argmax() for row in t])
        if k == 0:
            otype[:2 * norb] = np.repeat(maj, 2)
        else:
            otype[2 * norb:] = maj
    iE, rem = np.divmod(bundle, nI2 * nI3)
    iI2, iI3 = np.divmod(rem, nI3)
    return dict(lib=lib, bundle=bundle, flip=flip, type=otype, iE=iE, iI2=iI2, iI3=iI3,
                r=r_arcsec, lmd_z=lmd_z, lmd_x=lmd_x, lmd=lmd, w=np.asarray(weights))


def _integrate_job(args):
    pot_pars, row, kpa, n_periods, n_out = args
    try:
        pot = op.DynamitePotential(pot_pars)
        t, pos, vel, period = op.integrate_orbit(pot, row, kpa, n_periods=n_periods,
                                                 n_out=n_out)
        av = op.orbit_averages(pos, vel)
        return pos, av, period
    except Exception as exc:  # keep going if one orbit fails
        return None, str(exc), None


def orbit_set(bt, pars, mod_dir, n_orbits, procs, velscale):
    """Integrate the most heavily weighted bundles of the best model."""
    nd = pars['dithering']
    w = bt['w']
    # combine each tube orbit with its velocity-flipped twin
    keys = {}
    for k in np.nonzero(w > 0)[0]:
        key = (int(bt['lib'][k]), int(bt['bundle'][k]))
        keys.setdefault(key, []).append(k)
    # The outer bundles carry most of the mass, so a plain top-N by weight has
    # no orbits near the centre. Take the heaviest few of every energy shell
    # first, then fill up with the heaviest of the rest.
    by_weight = sorted(keys.items(), key=lambda kv: -w[kv[1]].sum())
    per_shell = max(3, n_orbits // (2 * pars['nE']))
    picked, count = [], {}
    for kv in by_weight:
        iE = int(bt['iE'][kv[1][0]])
        if count.get(iE, 0) < per_shell:
            picked.append(kv)
            count[iE] = count.get(iE, 0) + 1
    picked_keys = {kv[0] for kv in picked}
    picked += [kv for kv in by_weight if kv[0] not in picked_keys][:max(0, n_orbits - len(picked))]
    ranked = sorted(picked, key=lambda kv: -w[kv[1]].sum())
    total = w.sum()
    ics = {0: op.read_begin(Path(mod_dir) / 'datfil' / 'begin.dat'),
           1: op.read_begin(Path(mod_dir) / 'datfil' / 'beginbox.dat')}
    pot = op.DynamitePotential(pars)
    jobs, meta = [], []
    for (lib, b), ks in ranked:
        k0 = ks[0]
        shape, data = ics[lib]
        row = data[op.bundle_ic_row(int(bt['iE'][k0]), int(bt['iI2'][k0]),
                                    int(bt['iI3'][k0]), shape, nd)]
        jobs.append((pars, row, pot.kpa, 15, 1500))
        wpro = float(sum(w[k] for k in ks if bt['flip'][k] == 0))
        wret = float(sum(w[k] for k in ks if bt['flip'][k] == 1))
        meta.append(dict(lib='tube' if lib == 0 else 'box', bundle=b, iE=int(bt['iE'][k0]),
                         iI2=int(bt['iI2'][k0]), iI3=int(bt['iI3'][k0]),
                         w=(wpro + wret) / total, w_pro=wpro / total, w_retro=wret / total,
                         type=int(bt['type'][k0]), r=float(bt['r'][k0]),
                         lmd_z=float(bt['lmd_z'][k0]), lmd_x=float(bt['lmd_x'][k0])))
    with mp.Pool(procs) as pool:
        results = pool.map(_integrate_job, jobs)
    orbits = []
    for m, (pos, av, period) in zip(meta, results):
        if pos is None:
            logging.getLogger('dynx').warning(f'orbit {m} failed: {av}')
            continue
        scale = float(np.abs(pos).max()) or 1.0
        q16 = np.round(pos / scale * 32767).astype(np.int16)
        m.update(scale=scale, n=len(pos), pts=b64(q16), r_int=float(av['r']),
                 type_int=int(av['type']), period=float(period) / velscale)
        orbits.append(m)
    # circular velocity curves along the three axes
    rgrid = np.geomspace(10 ** pars['logrmin'], 10 ** pars['logrmax'], 60)
    vc = {ax: {k: rounded(v * velscale, 5) for k, v in pot.circular_velocity(rgrid, i).items()}
          for i, ax in enumerate('xyz')}
    return dict(
        orbits=orbits, weight_fraction=float(sum(o['w'] for o in orbits)),
        n_nonzero=int((w > 0).sum()), n_total=int(len(w)),
        gauss=dict(p=rounded(pot.p), q=rounded(pot.q), sigma=rounded(pot.sig),
                   mass=rounded(pot.mass_gauss * 1.0)),
        mstars=float(pot.mstars), mbh=float(pot.mbh), nfw=pot.nfw,
        vc=dict(r=rounded(rgrid), **vc), velscale=velscale,
        note='Orbits re-integrated from DYNAMITE initial conditions (centre '
             'dither of each bundle) for 15 periods; potential ported from '
             'triaxpotent.f90/dmpotent.f90. Positions in arcsec, intrinsic frame. '
             'Selection: the heaviest bundles of every energy shell, then the '
             'heaviest of the rest.')


def model_details(config, row, with_orbits, n_orbits, procs):
    """Kinematics, intrinsic profiles, orbit distribution for one table row."""
    from dynamite import analysis

    model = config.all_models.get_model_from_row(row)
    orblib = model.get_orblib()
    model.get_weights(orblib)
    weights = np.asarray(model.weights)
    orblib.read_losvd_histograms()
    model.get_orblib = lambda: orblib  # reuse the LOSVDs for every kin set
    stars = config.system.get_unique_triaxial_visible_component()
    kin_out = []
    for k, kin in enumerate(stars.kinematic_data):
        a = analysis.Analysis(config=config, model=model, kin_set=k)
        gh = a.get_gh_model_kinematic_maps(model=model, kin_set=k,
                                           v_sigma_option='fit', weights=weights)
        data = kin.data
        mod = {c: rounded(gh[c]) for c in gh.colnames}
        dat = {c: rounded(data[c]) for c in ('v', 'dv', 'sigma', 'dsigma',
                                             'h3', 'dh3', 'h4', 'dh4') if c in data.colnames}
        chi2 = {}
        for c in ('v', 'sigma', 'h3', 'h4'):
            if c in data.colnames and c in gh.colnames:
                res = (np.asarray(data[c]) - np.asarray(gh[c])) / np.asarray(data['d' + c])
                chi2[c] = float(np.nansum(res ** 2))
        kin_out.append(dict(name=kin.name, data=dat, model=mod, chi2=chi2,
                            chi2_total=float(sum(chi2.values())), n_bins=len(data)))
    mod_dir = model.directory_noml
    pars = op.read_parameters_pot(mod_dir + 'infil/parameters_pot.in')
    kpa = pars['distance'] * 1e6 * np.tan(np.pi / 648e3) * op.PARSEC_KM
    det = dict(row=int(row), directory=str(model.directory), kin=kin_out,
               velscale=float(orblib.velocity_scaling_factor),
               weight_meta=clean(_weights_meta(model)))
    try:
        det['intrinsic'] = intrinsic_profiles(orblib, weights)
    except Exception:
        logging.getLogger('dynx').warning('intrinsic moments failed:\n' + traceback.format_exc())
    try:
        bt = bundle_table(orblib, weights, pars, mod_dir, kpa)
        det['orbit_dist'] = dict(
            lib=bt['lib'].astype(int).tolist(), type=bt['type'].astype(int).tolist(),
            iE=bt['iE'].tolist(), r=rounded(bt['r'], 4), lmd_z=rounded(bt['lmd_z'], 4),
            lmd_x=rounded(bt['lmd_x'], 4), w=rounded(bt['w'] / bt['w'].sum(), 4))
        if with_orbits:
            det['orbits3d'] = orbit_set(bt, pars, mod_dir, n_orbits, procs,
                                        float(orblib.velocity_scaling_factor))
            det['orbits3d']['angles'] = dict(theta=pars['theta'], phi=pars['phi'],
                                             psi=pars['psi'])
    except Exception:
        logging.getLogger('dynx').warning('orbit distribution failed:\n' + traceback.format_exc())
    return det


def _weights_meta(model):
    f = Path(model.directory) / 'orbit_weights.ecsv'
    if f.exists():
        return dict(Table.read(f, format='ascii.ecsv').meta)
    return {}


# ----------------------------------------------------------------- driver
def kin_geometry(config):
    stars = config.system.get_unique_triaxial_visible_component()
    out = []
    for kin in stars.kinematic_data:
        dp = kin.dp_args
        nx, ny, dx = int(dp['n_x']), int(dp['n_y']), float(dp['dx'])
        i = np.round((dp['x'] - dp['min_x']) / dx - 0.5).astype(int)
        j = np.round((dp['y'] - dp['min_y']) / dx - 0.5).astype(int)
        grid = np.zeros((ny, nx), dtype=np.uint16)
        grid[j, i] = np.asarray(dp['idx_bin_to_pix']) + 1
        psf = kin.PSF if hasattr(kin, 'PSF') else None
        out.append(dict(name=kin.name, type=kin.type, nx=nx, ny=ny, dx=dx,
                        minx=float(dp['min_x']), miny=float(dp['min_y']),
                        angle=float(dp['angle']), grid=b64(grid),
                        n_bins=int(len(kin.data)), psf=clean(psf)))
    return out


def extract(args):
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out) / args.name
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    (out_dir / 'plots').mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('dynx')
    table = Table.read(run_dir / 'all_models.ecsv', format='ascii.ecsv')
    try:
        cfg_path = find_config(run_dir, table)
        cfg = yaml.safe_load(open(cfg_path))
        input_dir, input_note = find_input_dir(cfg, run_dir)
        if input_note:
            log.warning(input_note)
    except FileNotFoundError as exc:
        # nothing finished and no config copy: still show the model table
        log.warning(f'{exc}; writing the model table only')
        cfg = cfg_path = input_dir = input_note = None
    if cfg is not None:
        pars = par_list(cfg)
        which = cfg['parameter_space_settings'].get('which_chi2', 'kinchi2')
    else:
        names = table.colnames[:table.colnames.index('chi2')]
        pars = [dict(name=n, fixed=False, logarithmic=n in ('m-bh', 'f-dh'), value=None,
                     lo=None, hi=None, step=None, minstep=None, latex='') for n in names]
        which = 'kinmapchi2'
    done = np.array([bool(v) for v in table['all_done']])
    chi = np.array(table[which], dtype=float)
    order = [i for i in np.argsort(np.where(done & np.isfinite(chi), chi, np.inf))
             if done[i] and np.isfinite(chi[i])]
    n_done = len(order)

    def on_disk(i):
        d = run_dir / 'models' / str(table['directory'][i])
        return ((d / 'orbit_weights.ecsv').is_file()
                and (d.parent / 'infil' / 'parameters_pot.in').is_file()
                and (d.parent / 'datfil').is_dir())
    # orbit libraries of some finished models may have been deleted since
    available = [i for i in order if on_disk(i)]
    n_avail = len(available)

    def library_bytes(i):
        datfil = (run_dir / 'models' / str(table['directory'][i])).parent / 'datfil'
        return sum(f.stat().st_size for f in datfil.glob('*.bz2'))
    # reading a large orbit library takes minutes per model: only take every
    # finished model of small runs with light libraries
    light = n_avail and library_bytes(available[0]) < args.heavy_bytes
    n_detail = n_avail if (n_avail <= args.all_below and light) else min(args.n_best, n_avail)
    detail_rows = available[:n_detail] if cfg is not None else []

    if cfg is not None:
        mge = Table.read(input_dir / cfg['system_components']['stars']['mge_pot'],
                         format='ascii.ecsv')
        qobs = float(np.min(mge['q']))
        theta, phi, psi = viewing_angles(table['p-stars'], table['q-stars'],
                                         table['u-stars'], qobs)
    else:
        mge, qobs = None, None
        theta = phi = psi = np.full(len(table), np.nan)
    cfg_or_empty = cfg or {'system_attributes': {}, 'system_components': {'stars': {}}}
    cols = [c for c in table.colnames]
    rows_out = []
    for i in range(len(table)):
        r = []
        for c in cols:
            v = table[c][i]
            r.append(v.item() if hasattr(v, 'item') else v)
        r += [theta[i], phi[i], psi[i]]
        rows_out.append(r)
    result = dict(
        name=args.name, path=str(run_dir), config_file=str(cfg_path) if cfg_path else None,
        input_dir=str(input_dir) if input_dir else None, input_note=input_note, extracted=datetime.datetime.now().isoformat(timespec='seconds'),
        which_chi2=which, parameters=pars,
        settings=dict(orblib=cfg_or_empty.get('orblib_settings'),
                      weight_solver=cfg_or_empty.get('weight_solver_settings'),
                      parameter_space=cfg_or_empty.get('parameter_space_settings'),
                      distMPc=cfg_or_empty['system_attributes'].get('distMPc'),
                      kinematics={k: v for k, v in cfg_or_empty['system_components']['stars']
                                  .get('kinematics', {}).items()},
                      dh_type=cfg_or_empty['system_components'].get('dh', {}).get('type'),
                      bh_type=cfg_or_empty['system_components'].get('bh', {}).get('type')),
        config_text=open(cfg_path).read() if cfg_path else '',
        mge=dict(I=rounded(mge['I']), sigma=rounded(mge['sigma']), q=rounded(mge['q']),
                 qobs_min=qobs) if mge is not None else None,
        models=dict(columns=cols + ['theta', 'phi', 'psi'], rows=rows_out),
        n_models=len(table), n_done=n_done, n_on_disk=n_avail, best_row=int(order[0]) if order else None,
        details={}, geometry=None, qpt=None,
    )
    try:
        import dynamite as dyn
        result['dynamite_version'] = dyn.__version__
    except Exception:
        dyn = None

    # copy existing DYNAMITE plots
    plots = []
    for pdir in (run_dir / 'plots',):
        if pdir.is_dir():
            for f in sorted(pdir.glob('*.png')):
                if f.stat().st_size <= MAX_PLOT_BYTES:
                    shutil.copy2(f, out_dir / 'plots' / f.name)
                    plots.append(f.name)
    result['plots'] = plots

    if detail_rows and dyn is not None:
        sb, sb_cfg = build_sandbox(run_dir, table, detail_rows, cfg, input_dir, work)
        try:
            config = dyn.config_reader.Configuration(str(sb_cfg), reset_logging=True,
                                                     reset_existing_output=False)
            # the sandbox config differs from the model backups only in paths
            logging.getLogger().setLevel(logging.ERROR)
            logging.getLogger('dynamite').setLevel(logging.ERROR)
            log.setLevel(logging.WARNING)
            sub = config.all_models.table
            # velocity scaling: ml of the orbit library is the upsilon it was built with
            orig = config.all_models.get_ml_of_original_orblib

            def ml_orblib(model_id, _t=sub):
                d = str(_t['directory'][model_id]).split('/')[0]
                f = Path(config.settings.io_settings['model_directory']) / d / 'infil' / 'parameters_pot.in'
                try:
                    return op.read_parameters_pot(f)['upsilon']
                except Exception:
                    return orig(model_id)
            config.all_models.get_ml_of_original_orblib = ml_orblib
            result['n_obs2'] = float(config.get_2n_obs())
            result['geometry'] = kin_geometry(config)
            for rank, row in enumerate(detail_rows):
                d = str(table['directory'][row])
                srow = [i for i, x in enumerate(sub['directory']) if str(x) == d]
                if not srow:
                    log.warning(f'{d} not in sandbox table, skipped')
                    continue
                log.warning(f'[{args.name}] model {rank + 1}/{len(detail_rows)}: {d}')
                try:
                    det = model_details(config, srow[0], rank == 0,
                                        args.n_orbits, args.procs)
                    det['row'] = int(row)
                    result['details'][str(row)] = det
                except Exception:
                    log.warning(f'model {d} failed:\n' + traceback.format_exc())
        finally:
            if not args.keep_sandbox:
                shutil.rmtree(sb, ignore_errors=True)

    # intrinsic shape profiles (qpu_plot equivalent)
    if order and cfg is not None:
        rmax = 10 ** cfg['orblib_settings']['logrmax']
        rmin = 10 ** cfg['orblib_settings']['logrmin']
        r = np.geomspace(max(rmin, 1e-3), rmax, 80)
        n_obs2 = result.get('n_obs2') or np.nan
        chlim = np.sqrt(n_obs2) if np.isfinite(n_obs2) else 0
        within = [i for i in order if chi[i] <= chi[order[0]] + 3 * chlim]
        if len(within) < 3:
            within = order[:3]
        within = within[:200]
        prof = []
        for i in within:
            if not np.isfinite(theta[i]):
                continue
            prof.append(intrinsic_shape_profile(mge, theta[i], phi[i], psi[i], r))
        if prof:
            P = np.array([p[0] for p in prof])
            Q = np.array([p[1] for p in prof])
            T = np.array([p[2] for p in prof])
            result['qpt'] = dict(
                r=rounded(r), n=len(prof), rows=[int(i) for i in within],
                best=dict(p=rounded(P[0]), q=rounded(Q[0]), T=rounded(T[0])),
                p_m=rounded(P.mean(0)), p_s=rounded(P.std(0)),
                q_m=rounded(Q.mean(0)), q_s=rounded(Q.std(0)),
                T_m=rounded(np.nanmean(T, 0)), T_s=rounded(np.nanstd(T, 0)))

    js = 'DYNX.add(' + json.dumps(clean(result), separators=(',', ':')) + ');\n'
    (out_dir / 'run.js').write_text(js)
    summary = dict(name=args.name, path=str(run_dir), n_models=len(table), n_done=n_done,
                   which_chi2=which, best_row=result['best_row'],
                   n_details=len(result['details']), size=len(js),
                   extracted=result['extracted'])
    (out_dir / 'summary.json').write_text(json.dumps(clean(summary)))
    log.warning(f'[{args.name}] wrote {out_dir}/run.js ({len(js) / 1e6:.1f} MB)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-dir', required=True, help='directory with all_models.ecsv')
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True, help='scratch directory for sandboxes')
    ap.add_argument('--n-best', type=int, default=3)
    ap.add_argument('--all-below', type=int, default=40,
                    help='extract every finished model if the run has at most this many')
    ap.add_argument('--heavy-bytes', type=float, default=400e6,
                    help='compressed orbit-library size above which only --n-best models are read')
    ap.add_argument('--n-orbits', type=int, default=200)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--keep-sandbox', action='store_true')
    args = ap.parse_args()
    log = logging.getLogger('dynx')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    log.addHandler(handler)
    log.propagate = False
    log.setLevel(logging.WARNING)
    extract(args)


if __name__ == '__main__':
    main()
