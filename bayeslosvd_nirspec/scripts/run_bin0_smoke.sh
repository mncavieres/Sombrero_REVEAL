#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${BAYESLOSVD_PYTHON:-python}"
BAYES_ROOT="$ROOT/bayeslosvd_nirspec/BAYES-LOSVD"
FIT_TYPE="${BAYESLOSVD_FIT_TYPE:-SP}"
NITER="${BAYESLOSVD_NITER:-200}"
NCHAIN="${BAYESLOSVD_NCHAIN:-1}"

cd "$BAYES_ROOT/scripts"
"$PYTHON" bayes_losvd_run.py \
  -f ../preproc_data/sombrero_nirspec_g235h_agn_sub.hdf5 \
  -b 0 \
  -i "$NITER" \
  -c "$NCHAIN" \
  -n 1 \
  -v 1 \
  -p 1 \
  -t "$FIT_TYPE"
