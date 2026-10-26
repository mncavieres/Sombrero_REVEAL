#!/usr/bin/env bash
# Extract DYNAMITE runs on the cluster and pull them into dynamite_explorer/data.
#
#   scripts/dynamite_explorer/sync.sh            # pull what is already extracted
#   scripts/dynamite_explorer/sync.sh --extract  # (re)extract changed runs first
#   NODE=mia-node-03 scripts/dynamite_explorer/sync.sh --extract -- --only nirspec_only
#
# Extraction only re-runs runs whose all_models.ecsv changed since the last one.
set -euo pipefail

NODE=${NODE:-mia-node-05}
ROOT=/nexus/posix0/MIA-astro-env/nneum/macavieres/sombrero
REMOTE=$ROOT/dynamite_explorer
PY=/nexus/posix0/MIA-astro-env/nneum/macavieres/conda_envs/dynamite_py310/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
LOCAL=$(cd "$HERE/../.." && pwd)/dynamite_explorer/data

if [[ "${1:-}" == "--extract" ]]; then
  shift
  [[ "${1:-}" == "--" ]] && shift
  ssh "$NODE" "mkdir -p $REMOTE/scripts"
  scp -q "$HERE"/orbit_potential.py "$HERE"/extract_run.py "$HERE"/run_all.py "$NODE:$REMOTE/scripts/"
  ssh "$NODE" "cd $REMOTE && $PY -W ignore scripts/run_all.py --root $ROOT --out data --work work -j 4 --procs 6 $*"
fi

mkdir -p "$LOCAL"
rsync -a --delete --exclude logs --exclude manifest.js "$NODE:$REMOTE/data/" "$LOCAL/"
python3 "$HERE/build_manifest.py" "$LOCAL"
echo "Open $(dirname "$LOCAL")/index.html"
