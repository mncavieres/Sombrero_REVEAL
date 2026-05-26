cd /Users/mncavieres/Documents/2026-1/Sombrero_REVEAL

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp \
/opt/miniconda3/envs/ppxf/bin/python -u scripts/ppxf_nirspec/ppxf_nirspec_phoenix_powerbin_degree_study.py \
  --cube-path /Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/antoine/sombrero_nirspec_1150_p1293_g235_wicked.fits \
  --study-dir Data/ppxf_nirspec/antoine_wicked_powerbin_sn120_lsf_degree_study \
  --lsf-table-path scripts/ppxf_nirspec/jwst_nirspec_g235h_disp.fits \
  --target-sn 200 \
  --fit-windows-rest-um 2.1-2.4 \
  --expected-template-count 0 \
  --n-processes 6 \
  --n-plot-bins 0 \
  --check-plot-radius-arcsec 0.5 \
  --case-pairs "0:0,0:5,0:10,5:0,5:5,5:10,10:0,10:5,10:6,10:10"
