#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${BAYESLOSVD_PYTHON:-python}"
BAYES_ROOT="$ROOT/bayeslosvd_nirspec/BAYES-LOSVD"

"$PYTHON" "$ROOT/bayeslosvd_nirspec/scripts/stage_nirspec_bayeslosvd.py"

cd "$BAYES_ROOT/scripts"
"$PYTHON" bayes_losvd_preproc_data.py -c ../config_files/nirspec_g235h_agn_sub_preproc.properties
