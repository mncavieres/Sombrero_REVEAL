#!/usr/bin/env python3
"""Fast GHfree MAP pass for the NIRSpec BAYES-LOSVD preprocessed cube."""

from __future__ import annotations

import argparse
import contextlib
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np


def _five_row(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(5, float(arr))
    return np.tile(arr, (5, 1))


def _as_scalar(dataset):
    value = np.asarray(dataset)
    return value.item() if value.shape == () else value


def _result_value(group, name):
    if name not in group:
        return None
    value = np.asarray(group[name])
    if value.ndim == 0:
        return value.item()
    if value.shape[0] > 2:
        return value[2]
    return value.ravel()[0]


def _load_init_values(path):
    if path is None:
        return {}
    values = {}
    with h5py.File(path, "r") as handle:
        if "out" not in handle:
            return values
        for key in handle["out"].keys():
            group = handle[f"out/{key}"]
            values[int(key)] = {
                name: _result_value(group, name)
                for name in ["vel", "sigma", "h1", "h2", "h3", "h4", "weights", "coefs"]
                if name in group
            }
    return values


def _init_scalar(value, default, lower, upper, jitter=0.0):
    if value is None:
        out = default
    else:
        arr = np.asarray(value, dtype=float)
        out = float(arr.ravel()[0]) if arr.size else default
        if not np.isfinite(out):
            out = default
    out += jitter
    return float(np.clip(out, lower, upper))


def _init_vector(value, size, jitter_scale=0.0, lower=-1.5, upper=1.5):
    if value is None:
        out = np.zeros(size, dtype=float)
    else:
        out = np.asarray(value, dtype=float).ravel()
        if out.size != size:
            out = np.zeros(size, dtype=float)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if jitter_scale > 0:
        out = out + np.random.default_rng(12345 + size).normal(0.0, jitter_scale, size=size)
    return np.clip(out, lower, upper)


def _make_init(common, bin_id, fit_type, init_values, attempt):
    vals = init_values.get(bin_id, {})
    xvel = np.asarray(common["xvel"], dtype=float)
    vel_margin = max(1e-3, float(common["velscale"]) * 0.05)
    sigma_lower = float(common["velscale"]) / 2.0 + 1e-3
    sigma_upper = 500.0 - 1e-3
    rng = np.random.default_rng(1000 * attempt + bin_id)
    vel_jitter = 0.0 if attempt == 0 else rng.normal(0.0, 25.0)
    sigma_jitter = 0.0 if attempt == 0 else rng.normal(0.0, 20.0)
    init = {
        "vel": _init_scalar(vals.get("vel"), 0.0, float(np.min(xvel)) + vel_margin, float(np.max(xvel)) - vel_margin, vel_jitter),
        "sigma": _init_scalar(vals.get("sigma"), 220.0, sigma_lower, sigma_upper, sigma_jitter),
        "weights": _init_vector(vals.get("weights"), int(common["ntemp"]), jitter_scale=0.02 if attempt > 0 else 0.0),
        "coefs": _init_vector(vals.get("coefs"), int(common["porder"]) + 1, jitter_scale=0.02 if attempt > 0 else 0.0),
    }
    if fit_type.lower() == "ghfree":
        for name in ["h1", "h2", "h3", "h4"]:
            init[name] = _init_scalar(vals.get(name), 0.0, -0.3, 0.3, rng.normal(0.0, 0.03) if attempt > 0 else 0.0)
    elif fit_type.lower() in {"gh34", "gh34map"}:
        for name in ["h3", "h4"]:
            init[name] = _init_scalar(vals.get(name), 0.0, -0.25, 0.25, rng.normal(0.0, 0.03) if attempt > 0 else 0.0)
    return init


@contextlib.contextmanager
def _quiet_fds():
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


def _load_common(preproc):
    group = preproc["in"]
    return {
        "npix_obs": _as_scalar(group["npix_obs"]),
        "ntemp": _as_scalar(group["ntemp"]),
        "nvel": _as_scalar(group["nvel"]),
        "npix_temp": _as_scalar(group["npix_temp"]),
        "mask": np.asarray(group["mask"]) + 1,
        "nmask": _as_scalar(group["nmask"]),
        "porder": _as_scalar(group["porder"]),
        "templates": np.asarray(group["templates"]),
        "mean_template": np.asarray(group["mean_template"]),
        "velscale": _as_scalar(group["velscale"]),
        "xvel": np.asarray(group["xvel"]),
        "spec_obs_all": np.asarray(group["spec_obs"]),
        "sigma_obs_all": np.asarray(group["sigma_obs"]),
    }


def _make_data(common, bin_id):
    data = {
        key: val
        for key, val in common.items()
        if key not in {"spec_obs_all", "sigma_obs_all"}
    }
    data["spec_obs"] = common["spec_obs_all"][:, bin_id]
    data["sigma_obs"] = common["sigma_obs_all"][:, bin_id]
    return data


def _write_failure(group, common, reason):
    group.attrs["fit_failed"] = True
    group.attrs["failure_reason"] = str(reason)
    npix = int(common["npix_obs"])
    nvel = int(common["nvel"])
    for name in ["vel", "sigma", "h1", "h2", "h3", "h4"]:
        group.create_dataset(name, data=np.full(5, np.nan), compression="gzip")
    for name, size in {
        "losvd": nvel,
        "spec": int(common["npix_temp"]),
        "conv_spec": npix,
        "poly": npix,
        "bestfit": npix,
    }.items():
        group.create_dataset(name, data=np.full((5, size), np.nan), compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preproc", required=True, type=Path)
    parser.add_argument("--model-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-type", default="GHfree")
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--init-result", type=Path, default=None, help="Optional result HDF5 used to initialize matching bins.")
    args = parser.parse_args()

    if args.output.exists() and args.restart:
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.model_cache.open("rb") as handle:
        model = pickle.load(handle)
    init_values = _load_init_values(args.init_result)

    with h5py.File(args.preproc, "r") as preproc, h5py.File(args.output, "a") as out:
        if "in" not in out:
            preproc.copy("in", out)
        out.attrs["fit_type"] = args.fit_type
        out.attrs["fit_mode"] = "MAP optimization"
        out.attrs["optimizer_iter"] = args.iter
        out_group = out.require_group("out")
        common = _load_common(preproc)
        nbins = int(_as_scalar(preproc["in/nbins"]))

        for bin_id in range(nbins):
            key = str(bin_id)
            if key in out_group and not args.restart:
                continue
            if key in out_group:
                del out_group[key]

            grp = out_group.create_group(key)
            data = _make_data(common, bin_id)
            result = None
            last_error = None
            for attempt in range(args.retries + 1):
                try:
                    with _quiet_fds():
                        result = model.optimizing(
                            data=data,
                            iter=args.iter,
                            init=_make_init(common, bin_id, args.fit_type, init_values, attempt),
                            as_vector=False,
                            verbose=False,
                            seed=100000 + 1000 * attempt + bin_id,
                        )
                    break
                except Exception as exc:
                    last_error = exc

            if result is None:
                _write_failure(grp, common, last_error)
            else:
                pars = result["par"]
                grp.attrs["fit_failed"] = False
                grp.attrs["optimizer_value"] = float(result["value"])
                for name, value in pars.items():
                    grp.create_dataset(name, data=_five_row(value), compression="gzip")

                if (bin_id + 1) % 25 == 0 or bin_id == nbins - 1:
                    print(f"completed {bin_id + 1}/{nbins} bins", flush=True)


if __name__ == "__main__":
    main()
