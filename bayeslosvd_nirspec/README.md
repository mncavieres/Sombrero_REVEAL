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
- `scripts/make_xsl_bayes_templates.py`: converts the pPXF XSL SPS NIR
  templates into the same BAYES-LOSVD FITS template format and writes the XSL
  template LSF table.
- `scripts/run_preproc.sh`: stages the overlay and runs BAYES-LOSVD
  preprocessing.
- `scripts/run_bin0_smoke.sh`: runs one BAYES-LOSVD bin as a smoke test.
- `scripts/build_kinematics_dashboard.py`: creates a self-contained HTML
  dashboard for clicking bins, comparing LOSVDs, and switching between
  bin-center and cubic-interpolated maps. Dashboard kinematic maps are derived
  from the normalized LOSVD in each bin, not from raw model parameters.
- `scripts/plot_spectral_fit_axis_checkplots.py`: plots observed spectra,
  BAYES best fits, residuals, and selected-bin locations for the IFU peak plus
  bins along the rotated major/minor axes.

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

Prepare a BAYES-compatible XSL template library from the local pPXF XSL SPS
model:

```bash
bayeslosvd_nirspec/env_x86/bin/python bayeslosvd_nirspec/scripts/make_xsl_bayes_templates.py --force
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

## Reproducing Runs

Every BAYES-LOSVD run launched through the patched entry points now writes a
runnable shell file plus a JSON manifest next to the product:

- preprocessing: `preproc_data/<run>_reproduce.sh` and
  `preproc_data/<run>_run_manifest.json`
- upstream sampler: `results/<run>-<fit>/reproduce_run.sh` and
  `results/<run>-<fit>/run_manifest.json`
- MAP wrappers: `results/<run>/reproduce_run.sh` and
  `results/<run>/run_manifest.json`
- dashboard/checkplot helpers: `<output>_reproduce.sh` or
  `reproduce_checkplots.sh`, plus the matching manifest JSON

The shell file reruns the original command from the original working directory.
The manifest records the command, Python executable, selected environment
variables, git commit/status, input/output file metadata, and snapshots of small
configuration files.

Existing products can be backfilled with:

```bash
bayeslosvd_nirspec/env_x86/bin/python bayeslosvd_nirspec/scripts/backfill_reproduction_files.py
```

For products created before run manifests existed, the backfilled command is
reconstructed from the HDF5 attributes, filenames, available preproc configs,
and local runner defaults; those manifests are marked as reconstructed.

## Interactive Dashboard

After producing the vmax1200 MAP result HDF5 files, build the main dashboard with
the Gaussian/GH34 sanity products plus the free/non-regularized and RW-regularized
test products:

```bash
bayeslosvd_nirspec/env_x86/bin/python bayeslosvd_nirspec/scripts/build_kinematics_dashboard.py \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-GaussianMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-GaussianMAP_results.hdf5 GaussianMAP_vmax1200 \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-GH34MAP/sombrero_nirspec_g235h_agn_sub_vmax1200-GH34MAP_results.hdf5 GH34MAP_vmax1200_sanity \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-GHfreeMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-GHfreeMAP_results.hdf5 GHfreeMAP_vmax1200_nonreg_DIAGNOSTIC \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP_results.hdf5 RWregularizedMAP_vmax1200 \
  --output bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_dashboard.html
```

Use the dashboard's `Regularization` segmented control to switch the result
dropdown between all products, non-regularized products, and regularized products.
The `Sigma edge outliers` toggle masks bins whose LOSVD-derived dispersion is
higher than the maximum dispersion within `r <= 1 arcsec` for the selected
result.

Dashboard HTML files also include an `Import Maps` file picker. It accepts JSON
result payloads in the dashboard's internal format, or CSV map tables with
`x,y,vel,sigma,h3,h4` columns. The `x,y` coordinates are interpreted in the
rotated dashboard map frame, where the major axis is horizontal. Imported CSV
maps are appended to the existing result dropdown and get browser-side
interpolated grids for the `Interpolated` view.

The older non-vmax1200 result files use a narrower `xvel=-700..700 km/s`
grid and should not be mixed into the current comparison dashboards.

## Regularized Test Product

For a non-parametric LOSVD regularization test, run the random-walk smoothing
model on the same vmax1200 preprocessed cube:

```bash
bayeslosvd_nirspec/env_x86/bin/python bayeslosvd_nirspec/scripts/run_regularized_map.py \
  --preproc bayeslosvd_nirspec/BAYES-LOSVD/preproc_data/sombrero_nirspec_g235h_agn_sub_vmax1200.hdf5 \
  --output bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP_results.hdf5 \
  --fit-type RW \
  --iter 2000 \
  --retries 2 \
  --restart \
  --init-result bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-GaussianMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-GaussianMAP_results.hdf5
```

The spatial-regularized variant keeps the same vmax1200 preprocessing and RW
velocity-space LOSVD prior, then adds a fixed spatial prior that pulls each bin
toward the neighbor-averaged LOSVD from the completed `RWregularizedMAP`
solution. The production run used the 6 nearest neighboring bins, a Gaussian
spatial weighting scale of `0.35 arcsec`, and a per-velocity-channel prior width
of `0.02`:

```bash
bayeslosvd_nirspec/env_x86/bin/python bayeslosvd_nirspec/scripts/run_regularized_map.py \
  --preproc bayeslosvd_nirspec/BAYES-LOSVD/preproc_data/sombrero_nirspec_g235h_agn_sub_vmax1200.hdf5 \
  --output bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-RWspatialRegularizedMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-RWspatialRegularizedMAP_results.hdf5 \
  --fit-type RWSPATIAL \
  --output-fit-type RWspatialRegularized \
  --iter 2000 \
  --retries 2 \
  --restart \
  --init-result bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP_results.hdf5 \
  --spatial-prior-result bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP_results.hdf5 \
  --spatial-k 6 \
  --spatial-scale 0.35 \
  --spatial-sigma 0.02
```

Then build the comparison dashboard with Gaussian, GH34 sanity, GHfree
non-regularized, RW regularized, and RW+spatial regularized products:

```bash
bayeslosvd_nirspec/env_x86/bin/python bayeslosvd_nirspec/scripts/build_kinematics_dashboard.py \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-GaussianMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-GaussianMAP_results.hdf5 GaussianMAP_vmax1200 \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-GH34MAP/sombrero_nirspec_g235h_agn_sub_vmax1200-GH34MAP_results.hdf5 GH34MAP_vmax1200_sanity \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-GHfreeMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-GHfreeMAP_results.hdf5 GHfreeMAP_vmax1200_nonreg_DIAGNOSTIC \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-RWregularizedMAP_results.hdf5 RWregularizedMAP_vmax1200 \
  --results bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200-RWspatialRegularizedMAP/sombrero_nirspec_g235h_agn_sub_vmax1200-RWspatialRegularizedMAP_results.hdf5 RWspatialRegularizedMAP_vmax1200 \
  --output bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_vmax1200_regularization_test_dashboard.html
```

## Paper-Style SP Comparison

The comparison setup matching the quoted BAYES-LOSVD test uses:

- target S/N `100`
- minimum spaxel S/N `1`
- `vmax=700 km/s` and `velscale=60 km/s`
- PHOENIX fixed comparison run: `npca=5`
- PHOENIX/XSL pca995 runs: choose the minimum number of PCA components
  preserving at least `99.5%` of the template variance
- independent LOSVD prior model `SP`, i.e. `losvd ~ normal(0, sigma)`
- PHOENIX regularized comparison run: spatial random-walk LOSVD model `RW`

In BAYES-LOSVD this velocity setup produces 23 LOSVD samples from `-660` to
`+660 km/s` in 60 km/s steps. With corrected PCA accounting, the PHOENIX
`pca995` preprocessor selected 6 components, preserving `99.543%` of the
template variance.

The completed products are:

```bash
bayeslosvd_nirspec/BAYES-LOSVD/preproc_data/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_npca5.hdf5
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_npca5-SPpaperMAP/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_npca5-SPpaperMAP_results.hdf5
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_npca5_paper_test_dashboard.html
```

The completed PHOENIX `pca995` products are:

```bash
bayeslosvd_nirspec/BAYES-LOSVD/preproc_data/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix.hdf5
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix-SPpaperMAP/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix-SPpaperMAP_results.hdf5
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix-SPpaperMAP/bayeslosvd_sppaper_phoenix_pca995_map_checkplots.png
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix-SPpaperMAP/bayeslosvd_sppaper_phoenix_pca995_spectral_fit_axis_checkplots.png
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix-RWpaperRegularizedMAP/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix-RWpaperRegularizedMAP_results.hdf5
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix-RWpaperRegularizedMAP/bayeslosvd_rwpaper_phoenix_pca995_map_checkplots.png
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix-RWpaperRegularizedMAP/bayeslosvd_rwpaper_phoenix_pca995_spectral_fit_axis_checkplots.png
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_template_comparison_dashboard.html
```

The `SPpaper` dashboard entry is marked as regularized/prior-constrained by
the dashboard, but it is not the spatial random-walk model. It is the
independent equal-uncertainty LOSVD prior described above. The
`RWpaper_PHOENIX` entry is the velocity-space random-walk LOSVD counterpart
using the same PHOENIX `pca995` preprocessed data and initialization from the
non-regularized PHOENIX paper-style MAP result. Its run can be reproduced from:

```bash
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_phoenix-RWpaperRegularizedMAP/reproduce_run.sh
```

The XSL paper-style reproduction uses the XSL SPS library distributed with the
local pPXF environment. The automatic PCA target selected 5 components, which
preserve `99.612%` of the XSL template variance. BAYES-LOSVD warned that some
XSL template LSF values are broader than the science LSF, so those pixels use
the package's tiny smoothing floor during template convolution.

The completed XSL products are:

```bash
bayeslosvd_nirspec/BAYES-LOSVD/templates/XSL_NIRSPEC_G235H/
bayeslosvd_nirspec/BAYES-LOSVD/preproc_data/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_xsl.hdf5
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_xsl-SPpaperMAP/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_xsl-SPpaperMAP_results.hdf5
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_xsl_paper_test_dashboard.html
bayeslosvd_nirspec/BAYES-LOSVD/results/sombrero_nirspec_g235h_agn_sub_snr100_v700_dv60_pca995_xsl-SPpaperMAP/bayeslosvd_sppaper_xsl_pca995_spectral_fit_axis_checkplots.png
```
