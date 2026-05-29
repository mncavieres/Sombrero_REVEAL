# BAYES-LOSVD NIRSpec Workspace

This folder is dedicated to fitting Sombrero JWST/NIRSpec stellar kinematics
with the upstream `jfalconbarroso/BAYES-LOSVD` code.

The upstream repo is cloned in:

```bash
bayeslosvd_nirspec/BAYES-LOSVD
```

The NIRSpec overlay uses the AGN-subtracted G235H cube:

```bash
Data/IFU/david_subs/g235h_agn_sub.fits
```

## Layout

- `BAYES-LOSVD/`: upstream code clone.
- `config/`: Sombrero/NIRSpec instrument reader, LSF, mask, and preproc config.
- `scripts/stage_nirspec_bayeslosvd.py`: copies the local overlay into the
  upstream clone and links the AGN-subtracted cube into `BAYES-LOSVD/data/`.
- `scripts/make_phoenix_bayes_templates.py`: converts local PHOENIX templates
  into the simple linear-wavelength FITS format expected by BAYES-LOSVD.
- `scripts/run_preproc.sh`: stages the overlay and runs BAYES-LOSVD
  preprocessing.
- `scripts/run_bin0_smoke.sh`: runs one BAYES-LOSVD bin as a smoke test.
- `scripts/build_kinematics_dashboard.py`: creates a self-contained HTML
  dashboard for clicking bins, comparing LOSVDs, and switching between
  bin-center and cubic-interpolated maps. Dashboard kinematic maps are derived
  from the normalized LOSVD in each bin, not from raw model parameters.

## Environment

BAYES-LOSVD is an older script-based Python/Stan project. Its docs recommend
Python 3.6-era dependencies and `pystan` 2.x. A reproducible starting point is:

```bash
/opt/miniconda3/bin/conda env create -p bayeslosvd_nirspec/env -f bayeslosvd_nirspec/environment.yml
/opt/miniconda3/bin/conda activate bayeslosvd_nirspec/env
```

On Apple Silicon, `pystan` 2.x can be the hard part. If conda cannot solve it
natively, use an x86_64/Rosetta conda environment.

## First Run

Stage the NIRSpec files into the upstream clone:

```bash
python bayeslosvd_nirspec/scripts/stage_nirspec_bayeslosvd.py
```

Prepare a BAYES-compatible PHOENIX template library:

```bash
python bayeslosvd_nirspec/scripts/make_phoenix_bayes_templates.py --max-templates 120
```

Run preprocessing:

```bash
bayeslosvd_nirspec/scripts/run_preproc.sh
```

Run one bin as a Stan smoke test:

```bash
bayeslosvd_nirspec/scripts/run_bin0_smoke.sh
```

Full fitting should be launched from `BAYES-LOSVD/scripts`, following the
upstream workflow:

```bash
cd bayeslosvd_nirspec/BAYES-LOSVD/scripts
python bayes_losvd_run.py -f ../preproc_data/sombrero_nirspec_g235h_agn_sub.hdf5 -b all -i 500 -c 1 -n 4 -t SP
```

Results will appear under:

```bash
bayeslosvd_nirspec/BAYES-LOSVD/results/
```

## Interactive Dashboard

After producing the vmax1200 MAP result HDF5 files, build the main dashboard with
the Gaussian and GH34 sanity products:

```bash
bayeslosvd_nirspec/env_x86/bin/python bayeslosvd_nirspec/scripts/build_kinematics_dashboard.py \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-GaussianMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-GaussianMAP_results.hdf5 GaussianMAP_vmax1200 \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-GH34MAP/sombrero_nirspec_g235h_agn_sub_vmax1200-GH34MAP_results.hdf5 GH34MAP_vmax1200_sanity \
  --output bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_dashboard.html
```

The older non-vmax1200 result files use a narrower `xvel=-700..700 km/s`
grid. Keep the unconstrained GHfree MAP in a separate diagnostic dashboard
unless explicitly comparing that pathological free-LOSVD solution.
