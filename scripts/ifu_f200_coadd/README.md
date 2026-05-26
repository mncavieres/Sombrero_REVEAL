# IFU/NIRCam Coadd Tool

This directory contains an isolated command-line tool for building a JWST/NIRCam
imaging mosaic and patching its central region with a synthetic NIRCam-filter
image made from a NIRSpec IFU cube. It defaults to F200W, but the MAST mode can
switch to another compatible NIRCam filter when F200W is unavailable.

The workflow keeps the notebook choices that matter for this product:

- collapse the IFU cube through the active NIRCam throughput curve
- erode the finite IFU footprint before reprojection
- use `reproject_interp` for mosaicking, alignment reprojection, and final patching
- run the chi2 alignment from a local copy of the existing IFU alignment logic
- scale the IFU patch to the finite NIRCam mosaic overlap before replacing pixels

Dependencies are listed in `requirements.txt`, and a conda recipe is provided
in `environment.yml`.

The local repo environment created for this tool lives at:

```bash
conda activate /Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/venvs/ifu_f200_coadd
```

Example:

```bash
/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/venvs/ifu_f200_coadd/bin/python \
  scripts/ifu_f200_coadd/build_ifu_f200_mosaic.py \
  --ifu Data/IFU/david_subs/g235h_agn_sub.fits \
  --f200 \
    Data/f090_f200/jw06565-o002_t001_nircam_clear-f200w/jw06565-o002_t001_nircam_clear-f200w_i2d.fits \
    Data/f090_f200/jw06565-o003_t001_nircam_clear-f200w/jw06565-o003_t001_nircam_clear-f200w_i2d.fits \
    Data/f090_f200/jw06565-o005_t002_nircam_clear-f200w/jw06565-o005_t002_nircam_clear-f200w_i2d.fits \
  --imaging-filter F200W \
  --output-dir Data/IFU/f200_coadd_tool \
  --overwrite
```

MAST discovery mode:

```bash
/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/venvs/ifu_f200_coadd/bin/python \
  scripts/ifu_f200_coadd/build_ifu_f200_mosaic.py \
  --ifu Data/IFU/david_subs/g235h_agn_sub.fits \
  --mast-query \
  --mast-filter F200W \
  --output-dir Data/IFU/f200_coadd_tool/mast_test \
  --overwrite
```

Use `--mast-dry-run` to write the MAST observation/product tables without
downloading or mosaicking. By default the MAST cone-search center is computed
from the IFU celestial footprint, and the automatic radius is at least
`--mast-min-radius-arcsec 180` so the Sombrero three-tile NIRCam mosaic is found.
Pass `--mast-radius-arcsec` to force a different radius. Matching products are
then filtered to the continuous footprint component seeded by any product whose
footprint overlaps the IFU center. Use `--mast-no-connectivity` to disable that
graph selection.

When `--mast-query` is used, the tool first prints the inferred IFU band and the
NIRCam filters whose throughput curves fit inside the IFU wavelength range. It
then prints the filters actually found by the MAST query. If the requested
filter is missing but compatible filters are present, the default behavior is to
ask interactively which filter to use. For non-interactive runs, use
`--mast-filter-fallback auto`, `--mast-filter-fallback off`, or
`--mast-fallback-filter FILTER`.

Example dry run with automatic fallback:

```bash
/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/venvs/ifu_f200_coadd/bin/python \
  scripts/ifu_f200_coadd/build_ifu_f200_mosaic.py \
  --ifu Data/IFU/david_subs/g235h_agn_sub.fits \
  --mast-query \
  --mast-filter F200W \
  --mast-filter-fallback auto \
  --mast-dry-run \
  --output-dir Data/IFU/f200_coadd_tool/mast_test \
  --overwrite
```

The reference compatibility table generated from
`Data/for_antoine/nircam_throughputs/mean_throughputs` is
`nircam_nirspec_filter_compatibility.ecsv`. With the default 0.75 response
fraction threshold, the nominal compatible filters are:

- G235: F182M, F187N, F200W, F210M, F212N, F250M, F277W, F300M
- G395: F300M, F322W2, F323N, F335M, F356W, F360M, F405N, F410M, F430M, F444W, F460M, F466N, F470N, F480M

Main outputs:

- `f200w_ifu_f200w_mosaic.fits`: NIRCam images coadded alone
- `f200w_ifu_nirspec_f200w_raw.fits`: synthetic IFU image before chi2 alignment
- `f200w_ifu_nirspec_f200w_aligned.fits`: synthetic IFU image with aligned WCS
- `f200w_ifu_f200w_ifu_patched_mosaic.fits`: final mosaic with the scaled IFU patch
- `f200w_ifu_run_summary.json`: paths, software versions, IFU erosion info, alignment offset, scaling factor, and profile settings
- `f200w_ifu_scale_pixels.ecsv`: every finite NIRCam/IFU overlap pixel used for flux scaling
- `f200w_ifu_axis_profiles.ecsv`: NIRCam, IFU unscaled, IFU scaled, and patched profiles along major/minor cuts
- `f200w_ifu_scaling_checkplot.png`: visual scaling and profile checkplot
- `f200w_ifu_diagnostic_cutout.fits`: compact cutout around the IFU patch with NIRCam, IFU unscaled/scaled, final patched cutout, footprint, and masks

If MAST fallback switches to a non-F200W filter and the default prefix is used,
the prefix and filter tag change accordingly, for example `f277w_ifu_f277w_mosaic.fits`.

If `--align-center-sky` is omitted, the tool uses the center of the synthetic
IFU image WCS as the alignment cutout center. Use `--scaling False` to patch
without rescaling the IFU, `--scale-mode none` to keep scale diagnostics with a
unit factor, or `--scale-factor VALUE` to force a manual factor.
The profile PA defaults to `90` degrees east of north; pass `--profile-pa-deg`
to set the actual galaxy major-axis PA used in the diagnostic cuts.

By default, exact-zero NIRCam reference pixels are excluded from the chi2
alignment and flux-scale fits. This matters for products where saturated or
masked cores are stored as `0.0` instead of NaN; otherwise the alignment can
move the IFU peak away from the galaxy center to reduce residuals in the
masked hole. Use `--no-align-mask-reference-zero` or
`--no-scale-mask-reference-zero` to restore the old behavior.

For explicitly supplied non-F200W images, pass `--imaging-filter FILTER` so the
tool picks the matching throughput curve and writes the correct filter tag.
