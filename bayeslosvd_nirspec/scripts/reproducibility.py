#!/usr/bin/env python3
"""Helpers for writing reproducible BAYES-LOSVD run files."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


RELEVANT_ENV_VARS = (
    "MPLCONFIGDIR",
    "PYTHONPATH",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _run_text(command, cwd=None):
    try:
        return subprocess.check_output(command, cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _git_info(cwd):
    root = _run_text(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    if not root:
        return None

    status_short = _run_text(["git", "status", "--short"], cwd=root) or ""
    status_lines = status_short.splitlines()
    max_lines = 200
    return {
        "root": root,
        "commit": _run_text(["git", "rev-parse", "HEAD"], cwd=root),
        "branch": _run_text(["git", "branch", "--show-current"], cwd=root),
        "dirty": bool(status_lines),
        "status_short": status_lines[:max_lines],
        "status_short_truncated": len(status_lines) > max_lines,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path, cwd, hash_limit_bytes=10_000_000):
    if path is None:
        return None

    raw = Path(path)
    absolute = raw if raw.is_absolute() else Path(cwd) / raw
    info = {
        "path": str(raw),
        "absolute_path": str(absolute.resolve()) if absolute.exists() else str(absolute),
        "exists": absolute.exists(),
    }
    if not absolute.exists():
        return info

    st = absolute.stat()
    info.update(
        {
            "size_bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        }
    )
    if absolute.is_file() and st.st_size <= hash_limit_bytes:
        info["sha256"] = _sha256(absolute)
    return info


def _snapshot_text_file(path, cwd, max_bytes=1_000_000):
    if path is None:
        return None
    raw = Path(path)
    absolute = raw if raw.is_absolute() else Path(cwd) / raw
    if not absolute.exists() or not absolute.is_file() or absolute.stat().st_size > max_bytes:
        return None
    try:
        return {
            "path": str(raw),
            "absolute_path": str(absolute.resolve()),
            "text": absolute.read_text(encoding="utf-8", errors="replace"),
        }
    except Exception:
        return None


def _command_from_argv(argv):
    argv = list(sys.argv if argv is None else argv)
    return " ".join(shlex.quote(str(part)) for part in [sys.executable, *argv])


def _export_lines(env):
    lines = []
    for key, value in env.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    return lines


def write_reproduction_files(
    output_dir,
    *,
    run_name,
    argv=None,
    cwd=None,
    input_paths=None,
    config_paths=None,
    output_paths=None,
    extra=None,
    run_file_name="reproduce_run.sh",
    manifest_name="run_manifest.json",
):
    """Write a shell run file and JSON manifest into an output directory."""

    cwd_path = Path(cwd or os.getcwd()).resolve()
    outdir = Path(output_dir)
    if not outdir.is_absolute():
        outdir = cwd_path / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    command = _command_from_argv(argv)
    env = {key: os.environ[key] for key in RELEVANT_ENV_VARS if key in os.environ}
    generated = datetime.now(timezone.utc).isoformat()
    input_paths = [path for path in (input_paths or []) if path]
    config_paths = [path for path in (config_paths or []) if path]
    output_paths = [path for path in (output_paths or []) if path]

    manifest = {
        "run_name": run_name,
        "generated_utc": generated,
        "cwd": str(cwd_path),
        "python_executable": sys.executable,
        "argv": list(sys.argv if argv is None else argv),
        "command": command,
        "environment": env,
        "git": _git_info(cwd_path),
        "inputs": [_file_info(path, cwd_path) for path in input_paths],
        "outputs": [_file_info(path, cwd_path) for path in output_paths],
        "config_snapshots": [
            snapshot
            for snapshot in (_snapshot_text_file(path, cwd_path) for path in config_paths)
            if snapshot is not None
        ],
        "extra": extra or {},
    }

    manifest_path = outdir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    shell_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Reproduce BAYES-LOSVD run: {run_name}",
        f"# Generated UTC: {generated}",
        f"# Manifest: {manifest_path.name}",
        f"cd {shlex.quote(str(cwd_path))}",
        "",
        *_export_lines(env),
        "",
        command,
        "",
    ]
    run_path = outdir / run_file_name
    run_path.write_text("\n".join(shell_lines), encoding="utf-8")
    run_path.chmod(run_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return run_path, manifest_path
