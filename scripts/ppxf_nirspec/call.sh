cd /Users/mncavieres/Documents/2026-1/Sombrero_REVEAL

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp \
/opt/miniconda3/envs/ppxf/bin/python -u scripts/ppxf_nirspec/ppxf_nirspec_phoenix_powerbin_kinematics_lsf_table.py \
  --cube-path /Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/david_subs/g235h_agn_sub.fits \
  --output-dir Data/ppxf_nirspec/agn_sub_powerbin_sn120_wavelength_lsf \
  --lsf-table-path scripts/ppxf_nirspec/jwst_nirspec_g235h_disp.fits \
  --target-sn 120 \
  --fit-windows-rest-um 2.1-2.398 \
  --expected-template-count 0 \
  --n-plot-bins 8 \
  --n-processes 6
