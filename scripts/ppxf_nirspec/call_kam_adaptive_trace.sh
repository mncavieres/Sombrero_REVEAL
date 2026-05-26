cd /Users/mncavieres/Documents/2026-1/Sombrero_REVEAL

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp \
/opt/miniconda3/envs/ppxf/bin/python -u scripts/ppxf_nirspec/ppxf_nirspec_phoenix_powerbin_kinematics_lsf_table.py \
  --cube-path /Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/kam_adaptive_trace_step/f170lp_g235h-f170lp_s3d.fits \
  --output-dir Data/ppxf_nirspec/adaptive_trace_step_kam_sn120_lsf \
  --lsf-table-path scripts/ppxf_nirspec/jwst_nirspec_g235h_disp.fits \
  --target-sn 120 \
  --fit-windows-rest-um 2.1-2.4 \
  --expected-template-count 0 \
  --n-plot-bins 8 \
  --check-plot-radius-arcsec 0.5 \
  --n-processes 6
