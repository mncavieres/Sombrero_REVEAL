"""Find every DYNAMITE run under a root directory and extract each one.

A run is any directory holding ``all_models.ecsv`` (outside ``models/``).
Its explorer name is its path relative to the root, with a trailing
``/output`` dropped, ``/`` turned into ``__`` and spaces into ``_``.

    python run_all.py --root /nexus/.../sombrero --out data --work work -j 4

Runs whose ``all_models.ecsv`` is older than the existing extraction are
skipped unless ``--force``. Writes ``<out>/manifest.js`` at the end.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def discover(root, max_depth=5):
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        # never descend into orbit libraries or our own output
        dirnames[:] = [d for d in dirnames
                       if d not in ('models', 'plots', 'input', 'datfil', 'infil')
                       and not d.startswith('dynamite_explorer') and depth < max_depth]
        if 'all_models.ecsv' in filenames:
            runs.append(Path(dirpath))
    return sorted(runs)


def run_name(run_dir, root):
    rel = run_dir.relative_to(root)
    parts = list(rel.parts)
    if parts and parts[-1] == 'output' and len(parts) > 1:
        parts = parts[:-1]
    return '__'.join(parts).replace(' ', '_')


def needs_update(run_dir, out_dir):
    summary = out_dir / 'summary.json'
    if not summary.exists():
        return True
    return (run_dir / 'all_models.ecsv').stat().st_mtime > summary.stat().st_mtime


def write_manifest(out):
    runs = []
    for s in sorted(Path(out).glob('*/summary.json')):
        runs.append(json.loads(s.read_text()))
    (Path(out) / 'manifest.js').write_text(
        'DYNX.manifest(' + json.dumps(runs, separators=(',', ':')) + ');\n')
    print(f'manifest: {len(runs)} runs')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('-j', '--jobs', type=int, default=3)
    ap.add_argument('--procs', type=int, default=6, help='orbit integration workers per run')
    ap.add_argument('--only', nargs='*', help='run names to (re)extract')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('extra', nargs=argparse.REMAINDER, help='passed to extract_run.py')
    args = ap.parse_args()
    root, out, work = Path(args.root).resolve(), Path(args.out).resolve(), Path(args.work).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    todo = []
    for rd in discover(root):
        name = run_name(rd, root)
        if args.only and name not in args.only:
            continue
        if args.force or args.only or needs_update(rd, out / name):
            todo.append((rd, name))
        else:
            print(f'up to date: {name}')
    print(f'{len(todo)} runs to extract')

    def job(item):
        rd, name = item
        cmd = [args.python, '-W', 'ignore', str(HERE / 'extract_run.py'), '--run-dir', str(rd),
               '--name', name, '--out', str(out), '--work', str(work),
               '--procs', str(args.procs)] + [a for a in args.extra if a != '--']
        with open(out / 'logs' / f'{name}.log', 'w') as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        print(f'{"ok " if rc == 0 else "FAILED"} {name}', flush=True)
        return rc

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(job, todo))
    write_manifest(out)


if __name__ == '__main__':
    main()
