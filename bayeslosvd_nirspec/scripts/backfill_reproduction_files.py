#!/usr/bin/env python3
"""Backfill reproduce scripts/manifests for existing BAYES-LOSVD products."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import toml

from reproducibility import write_reproduction_files


REPO = Path(__file__).resolve().parents[2]
BAYES_ROOT = REPO / "bayeslosvd_nirspec" / "BAYES-LOSVD"
PREPROC_DIR = BAYES_ROOT / "preproc_data"
RESULTS_DIR = BAYES_ROOT / "results"
CONFIG_DIRS = [BAYES_ROOT / "config_files", REPO / "bayeslosvd_nirspec" / "config"]
STAN_DIR = BAYES_ROOT / "scripts" / "stan_model"
MAP_RUNNER = REPO / "bayeslosvd_nirspec" / "scripts" / "run_ghfree_map.py"
REG_RUNNER = REPO / "bayeslosvd_nirspec" / "scripts" / "run_regularized_map.py"
DASHBOARD_RUNNER = REPO / "bayeslosvd_nirspec" / "scripts" / "build_kinematics_dashboard.py"
CHECKPLOT_RUNNER = REPO / "bayeslosvd_nirspec" / "scripts" / "plot_ghfree_map.py"
CUBE_PATH = REPO / "Data" / "IFU" / "david_subs" / "g235h_agn_sub.fits"
MGE_TABLE = REPO / "Data" / "mge_NAGN_0deg_pa_positive_gauss" / "mge_luminosity_table.csv"


FIT_CACHE = {
    "Gaussian": {
        "cache": STAN_DIR / "test-cache-gaussian.pkl",
        "stan": STAN_DIR / "bayes-losvd_model_Gaussian.stan",
    },
    "GHfree": {
        "cache": STAN_DIR / "test-cache-gh.pkl",
        "stan": STAN_DIR / "bayes-losvd_model_gh_full_series.stan",
    },
    "GH34": {
        "cache": STAN_DIR / "test-cache-gh34.pkl",
        "stan": STAN_DIR / "bayes-losvd_model_GH34.stan",
    },
}


def _as_int(value):
    return int(np.asarray(value).item())


def _preproc_signature(path):
    with h5py.File(path, "r") as handle:
        return {
            "nbins": _as_int(handle["in/nbins"]),
            "nvel": _as_int(handle["in/nvel"]),
            "ntemp": _as_int(handle["in/ntemp"]),
            "xvel_min": float(np.nanmin(handle["in/xvel"])),
            "xvel_max": float(np.nanmax(handle["in/xvel"])),
        }


def _result_signature(handle):
    return {
        "nbins": _as_int(handle["in/nbins"]),
        "nvel": _as_int(handle["in/nvel"]),
        "ntemp": _as_int(handle["in/ntemp"]),
        "xvel_min": float(np.nanmin(handle["in/xvel"])),
        "xvel_max": float(np.nanmax(handle["in/xvel"])),
    }


def _find_preproc_for_result(result_path, preproc_signatures):
    with h5py.File(result_path, "r") as handle:
        sig = _result_signature(handle)
    matches = [
        path
        for path, preproc_sig in preproc_signatures.items()
        if all(preproc_sig[key] == sig[key] for key in ["nbins", "nvel", "ntemp", "xvel_min", "xvel_max"])
    ]
    if len(matches) == 1:
        return matches[0]
    stem = result_path.name.replace("_results.hdf5", "")
    for path in preproc_signatures:
        if path.stem in stem:
            return path
    return matches[0] if matches else None


def _find_config(run_name):
    for config_dir in CONFIG_DIRS:
        for path in sorted(config_dir.glob("*preproc.properties")):
            try:
                config = toml.load(path)
            except Exception:
                continue
            if run_name in config:
                return path, config[run_name]
    return None, {}


def _rel_to_bayes_scripts(path):
    if path is None:
        return None
    try:
        return path.resolve().relative_to((BAYES_ROOT / "scripts").resolve())
    except ValueError:
        return path


def _is_relative_to(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _known_gaussian_for(preproc_path):
    if preproc_path is None:
        return None
    run_name = preproc_path.stem
    candidate = RESULTS_DIR / f"{run_name}-GaussianMAP" / f"{run_name}-GaussianMAP_results.hdf5"
    return candidate if candidate.exists() else None


def backfill_preproc():
    written = []
    cwd = BAYES_ROOT / "scripts"
    for hdf5 in sorted(PREPROC_DIR.glob("*.hdf5")):
        run_name = hdf5.stem
        config_path, config = _find_config(run_name)
        config_arg = Path("../config_files") / config_path.name if config_path and _is_relative_to(config_path, BAYES_ROOT / "config_files") else config_path
        cube = Path("../data") / str(config.get("filename", "g235h_agn_sub.fits"))
        mask = Path("../config_files") / str(config.get("mask_file", "nirspec_g235h_stellar.mask"))
        run_file, manifest = write_reproduction_files(
            PREPROC_DIR,
            run_name=run_name,
            argv=["bayes_losvd_preproc_data.py", "-c", str(config_arg or "")],
            cwd=cwd,
            input_paths=[config_arg, cube, mask],
            config_paths=[config_arg],
            output_paths=[hdf5, hdf5.with_suffix(".pdf")],
            extra={
                "runner": "bayes_losvd_preproc_data.py",
                "reconstructed": True,
                "reconstruction_note": "Backfilled from matching preproc config and preproc output filename.",
                "config_section": run_name,
                "config_values": config,
            },
            run_file_name=f"{run_name}_reproduce.sh",
            manifest_name=f"{run_name}_run_manifest.json",
        )
        with h5py.File(hdf5, "a") as handle:
            handle.attrs["reproduce_script"] = str(run_file)
            handle.attrs["run_manifest"] = str(manifest)
            handle.attrs["reproduction_reconstructed"] = True
        written.append(run_file)
    return written


def _fit_family(fit_type, regularization):
    if regularization in {"RW", "SP"}:
        return "regularized"
    if str(fit_type).startswith("RW") or str(fit_type).startswith("SP"):
        return "regularized"
    return "map"


def _result_bins_arg(handle):
    nbins = _as_int(handle["in/nbins"])
    keys = sorted(int(key) for key in handle["out"].keys())
    return "all" if len(keys) == nbins and keys == list(range(nbins)) else ",".join(str(key) for key in keys)


def backfill_results():
    written = []
    preproc_signatures = {path: _preproc_signature(path) for path in sorted(PREPROC_DIR.glob("*.hdf5"))}
    for result in sorted(RESULTS_DIR.glob("*/*_results.hdf5")):
        preproc = _find_preproc_for_result(result, preproc_signatures)
        with h5py.File(result, "r") as handle:
            fit_type = str(handle.attrs.get("fit_type", "")).strip() or result.parent.name
            regularization = str(handle.attrs.get("regularization", "")).strip()
            optimizer_iter = int(handle.attrs.get("optimizer_iter", 1000))
            bins_arg = _result_bins_arg(handle)

        with h5py.File(result, "r") as handle:
            reconstructed = "run_manifest" not in handle.attrs
        extra = {
            "reconstructed": reconstructed,
            "reconstruction_note": "Backfilled from HDF5 attributes, result filename, available preproc files, and local runner defaults.",
            "optimizer_iter": optimizer_iter,
            "fit_type": fit_type,
            "regularization": regularization or None,
            "bins": bins_arg,
        }

        family = _fit_family(fit_type, regularization)
        if family == "regularized":
            fit_arg = regularization if regularization in {"RW", "SP"} else ("SP" if fit_type.startswith("SP") else "RW")
            argv = [
                str(REG_RUNNER),
                "--preproc",
                str(preproc),
                "--output",
                str(result),
                "--fit-type",
                fit_arg,
                "--iter",
                str(optimizer_iter),
                "--retries",
                "2",
                "--restart",
            ]
            if bins_arg != "all":
                argv.extend(["--bins", bins_arg])
            if fit_type != f"{fit_arg}regularized":
                argv.extend(["--output-fit-type", fit_type])
            init_result = None
            if fit_arg == "RW" and "smoke" not in result.parent.name:
                init_result = _known_gaussian_for(preproc)
                if init_result is not None:
                    argv.extend(["--init-result", str(init_result)])
            code = STAN_DIR / f"bayes-losvd_model_{fit_arg}.stan"
            inputs = [preproc, code, init_result]
            extra.update({"runner": "run_regularized_map.py", "fit_arg": fit_arg})
        else:
            cache_info = FIT_CACHE.get(fit_type, FIT_CACHE["GHfree"])
            argv = [
                str(MAP_RUNNER),
                "--preproc",
                str(preproc),
                "--model-cache",
                str(cache_info["cache"]),
                "--output",
                str(result),
                "--fit-type",
                fit_type,
                "--iter",
                str(optimizer_iter),
                "--retries",
                "3",
                "--restart",
            ]
            init_result = None
            if fit_type != "Gaussian":
                init_result = _known_gaussian_for(preproc)
                if init_result is not None:
                    argv.extend(["--init-result", str(init_result)])
            inputs = [preproc, cache_info["cache"], cache_info["stan"], init_result]
            extra.update({"runner": "run_ghfree_map.py", "model_cache": str(cache_info["cache"])})

        run_file, manifest = write_reproduction_files(
            result.parent,
            run_name=result.stem,
            argv=argv,
            cwd=REPO,
            input_paths=inputs,
            output_paths=[result],
            extra=extra,
        )
        with h5py.File(result, "a") as handle:
            handle.attrs["reproduce_script"] = str(run_file)
            handle.attrs["run_manifest"] = str(manifest)
            handle.attrs["reproduction_reconstructed"] = bool(reconstructed)
        written.append(run_file)
    return written


def _extract_dashboard_payload(path):
    text = path.read_text(encoding="utf-8")
    marker = "const DATA = "
    start = text.index(marker) + len(marker)
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    return payload


def _resolve_source(source):
    path = Path(source)
    if path.is_absolute():
        return path
    repo_path = REPO / path
    if repo_path.exists():
        return repo_path
    results_path = RESULTS_DIR / path
    if results_path.exists():
        return results_path
    return path


def backfill_dashboards():
    written = []
    for html in sorted(RESULTS_DIR.glob("*.html")):
        payload = _extract_dashboard_payload(html)
        argv = [str(DASHBOARD_RUNNER)]
        input_paths = []
        labels = []
        for result in payload.get("results", []):
            source = _resolve_source(result["source"])
            label = result.get("label", Path(source).parent.name)
            labels.append(label)
            input_paths.append(source)
            argv.extend(["--results", str(source), label])

        image_payload = payload.get("image") or {}
        mge_payload = payload.get("mge") or {}
        image_source = _resolve_source(image_payload.get("source", CUBE_PATH))
        mge_source = _resolve_source(mge_payload.get("source", MGE_TABLE))
        input_paths.extend([image_source, mge_source])
        argv.extend(["--cube", str(image_source), "--mge-table", str(mge_source), "--output", str(html)])
        run_file, _ = write_reproduction_files(
            html.parent,
            run_name=html.stem,
            argv=argv,
            cwd=REPO,
            input_paths=input_paths,
            output_paths=[html],
            extra={
                "runner": "build_kinematics_dashboard.py",
                "reconstructed": True,
                "reconstruction_note": "Backfilled from the dashboard embedded DATA payload.",
                "result_labels": labels,
                "grid_size": payload.get("gridSize"),
                "image_grid_size": payload.get("imageGridSize"),
            },
            run_file_name=f"{html.stem}_reproduce.sh",
            manifest_name=f"{html.stem}_run_manifest.json",
        )
        written.append(run_file)
    return written


def backfill_checkplots():
    written = []
    for result in sorted(RESULTS_DIR.glob("*/*_results.hdf5")):
        pngs = sorted(result.parent.glob("*.png"))
        if not pngs:
            continue
        central = [path for path in pngs if "r1p00arcsec" in path.name]
        argv = [str(CHECKPLOT_RUNNER), "--results", str(result), "--output-dir", str(result.parent)]
        if central:
            argv.extend(["--central-radius", "1.0"])
        run_file, _ = write_reproduction_files(
            result.parent,
            run_name=f"{result.stem}_checkplots",
            argv=argv,
            cwd=REPO,
            input_paths=[result],
            output_paths=pngs,
            extra={
                "runner": "plot_ghfree_map.py",
                "reconstructed": True,
                "reconstruction_note": "Backfilled from checkplot PNGs present in the result directory.",
                "central_radius": 1.0 if central else None,
            },
            run_file_name="reproduce_checkplots.sh",
            manifest_name="checkplots_run_manifest.json",
        )
        written.append(run_file)
    return written


def main():
    written = []
    written.extend(backfill_preproc())
    written.extend(backfill_results())
    written.extend(backfill_dashboards())
    written.extend(backfill_checkplots())
    for path in written:
        print(path)
    print(f"wrote {len(written)} reproduction scripts", file=sys.stderr)


if __name__ == "__main__":
    main()
