# DYNAMITE Run Explorer

A static HTML page for browsing every DYNAMITE run on the cluster: χ² landscapes,
best-fit parameters, data/model/residual kinematic maps, intrinsic shape
p(r), q(r), T(r), orbit-weight distributions, and the most heavily weighted
orbits drawn in 3D in the deprojected frame.

## Open it

```bash
python3 -m http.server 8765 --directory dynamite_explorer
```

Then open <http://localhost:8765>. Opening `index.html` directly also works:
data is loaded through script tags, not `fetch`. Plotly and three.js come from
CDNs, so the page needs internet.

## Update the data

```bash
scripts/dynamite_explorer/sync.sh --extract
```

This copies the scripts to `mia-node-05`, extracts every run under
`/nexus/posix0/MIA-astro-env/nneum/macavieres/sombrero` whose `all_models.ecsv`
changed since the last extraction, rsyncs the result into
`dynamite_explorer/data/` and rebuilds `data/manifest.js`. Without `--extract`
it only pulls what is already extracted. Use another node with
`NODE=mia-node-03`, and re-extract named runs with
`scripts/dynamite_explorer/sync.sh --extract -- --only <run> --force`.

Cluster layout (`.../sombrero/dynamite_explorer/`): `scripts/`, `data/`
(one folder per run plus `logs/`), `work/` (temporary sandboxes).

## What gets extracted, and how

`scripts/dynamite_explorer/extract_run.py`, per run (a directory with
`all_models.ecsv`):

* The whole `all_models` table, plus the viewing angles (θ, φ, ψ) implied by
  each (p, q, u), using the same formula as DYNAMITE's `triax_pqu2tpp`.
* For the best models by the run's `which_chi2` (3 by default; every finished
  model when a run has at most 40 and its orbit library is under 400 MB):
  per-bin data and model Gauss–Hermite moments for each kinematic set,
  intrinsic dispersion and anisotropy profiles, and the orbit-weight
  distribution (radius, circularities, orbit family per bundle).
* Intrinsic p(r), q(r), T(r) for the best model and the spread over the
  models within 3√(2N) of the minimum, as in DYNAMITE's `qpu_plot`.
* For the best model, about 200 orbits re-integrated in 3D (below).
* The PNGs in the run's `plots/` directory, and the config file text.

**Read-only on the runs.** Loading a DYNAMITE `Configuration` on an output
directory calls `AllModels.update_model_table()`, which can rewrite
`all_models.ecsv` and delete the orbit libraries of unfinished models. The
extractor never does that on a real run: each model is read inside a sandbox
in `work/` with a copied input directory, a filtered `all_models.ecsv`, copies
of the small orblib files and symlinks to the large `*.bz2` libraries, which
DYNAMITE only decompresses into the sandbox.

**Checks.** The map χ² recomputed from the extracted per-bin moments,
Σ((data − model)/error)² over v, σ, h3, h4 and all kinematic sets, reproduces
DYNAMITE's `kinmapchi2` (15851.3 for `muse500_first_pqu_fullyfree`, row 67).

### 3D orbits

DYNAMITE keeps only time-averaged projections of each orbit, not its path.
`scripts/dynamite_explorer/orbit_potential.py` re-integrates orbits from
DYNAMITE's own initial conditions (`begin.dat` for tubes, `beginbox.dat` for
boxes; the centre dither of each bundle) in a NumPy port of the Fortran
potential: the deprojected MGE (`triaxpotent.f90`, Cappellari 2002 eq. 12),
the Plummer black hole and the NFW halo (`dmpotent.f90`), with the parameters
from the model's `infil/parameters_pot.in`. Each is integrated for 15 periods.
The selection takes the heaviest bundles of every energy shell first, so the
centre is represented, then fills up with the heaviest of the rest.

Validation against DYNAMITE's `orbclass.out`: time-averaged radii agree to
0.1–1.5% for tubes and boxes, and Lz agrees for z-tubes. Energy is
conserved to about 1e-7. The Fortran integrator uses its own interpolated
potential and 200 periods, so a re-integrated orbit has the same family and
extent but not exactly the same path.

The sky view uses DYNAMITE's projection (`project_n` in `orblib_f`):
x' = −sin φ x + cos φ y, y' = −cos θ cos φ x − cos θ sin φ y + sin θ z, then
the ψ rotation, so that the major axis lies along the horizontal as in the
kinematic maps.

## Files

| Path | Role |
| --- | --- |
| `dynamite_explorer/index.html`, `app.js`, `style.css` | the viewer |
| `dynamite_explorer/data/<run>/run.js` | one run's extracted data |
| `dynamite_explorer/data/manifest.js` | run summaries for the overview |
| `scripts/dynamite_explorer/extract_run.py` | cluster-side extraction of one run |
| `scripts/dynamite_explorer/run_all.py` | finds all runs and extracts them in parallel |
| `scripts/dynamite_explorer/orbit_potential.py` | potential port and orbit integrator |
| `scripts/dynamite_explorer/build_manifest.py` | local manifest builder |
| `scripts/dynamite_explorer/sync.sh` | extract on the cluster and pull |
