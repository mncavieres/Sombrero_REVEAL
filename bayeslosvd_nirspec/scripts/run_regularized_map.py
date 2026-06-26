#!/usr/bin/env python3
"""Fast MAP pass for regularized BAYES-LOSVD NIRSpec kinematics."""

from __future__ import annotations

import argparse
import contextlib
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pystan
import pystan.api

from reproducibility import write_reproduction_files


FIT_CODES = {
    "RW": "bayes-losvd_model_RW.stan",
    "RWSPATIAL": "bayes-losvd_model_RW_spatial.stan",
    "SP": "bayes-losvd_model_SP.stan",
}


def _five_row(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(5, float(arr))
    return np.tile(arr, (5, 1))


def _as_scalar(dataset):
    value = np.asarray(dataset)
    return value.item() if value.shape == () else value


def _median_row(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.item()
    if arr.shape[0] > 2:
        return arr[2]
    return arr.ravel()


def _normalise_losvd(losvd):
    arr = np.asarray(losvd, dtype=float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 1e-10, None)
    total = np.sum(arr)
    if not np.isfinite(total) or total <= 0:
        arr = np.ones_like(arr)
        total = np.sum(arr)
    return arr / total


def _load_init_values(path):
    if path is None:
        return {}, None
    values = {}
    with h5py.File(path, "r") as handle:
        xvel = np.asarray(handle["in/xvel"], dtype=float) if "in/xvel" in handle else None
        if "out" not in handle:
            return values, xvel
        for key in handle["out"].keys():
            group = handle[f"out/{key}"]
            values[int(key)] = {
                name: _median_row(group[name])
                for name in ["losvd", "weights", "coefs"]
                if name in group
            }
    return values, xvel


def _load_losvd_array(path, target_xvel, nbins):
    if path is None:
        return None, None
    losvds = np.full((nbins, len(target_xvel)), np.nan, dtype=float)
    good = np.zeros(nbins, dtype=bool)
    with h5py.File(path, "r") as handle:
        source_xvel = np.asarray(handle["in/xvel"], dtype=float) if "in/xvel" in handle else target_xvel
        order = np.argsort(source_xvel)
        for key in handle.get("out", {}).keys():
            bin_id = int(key)
            if bin_id < 0 or bin_id >= nbins:
                continue
            group = handle[f"out/{key}"]
            if "losvd" not in group:
                continue
            losvd = _median_row(group["losvd"])
            losvd = np.asarray(losvd, dtype=float).ravel()
            if losvd.size == len(source_xvel):
                losvd = np.interp(target_xvel, source_xvel[order], losvd[order], left=1e-10, right=1e-10)
            if losvd.size != len(target_xvel):
                continue
            losvds[bin_id] = _normalise_losvd(losvd)
            good[bin_id] = np.all(np.isfinite(losvds[bin_id]))
    return losvds, good


def _fallback_losvd(xvel):
    return _normalise_losvd(np.exp(-0.5 * (np.asarray(xvel, dtype=float) / 220.0) ** 2))


def _build_spatial_priors(preproc, target_losvds, target_good, k, scale_arcsec):
    group = preproc["in"]
    xbin = np.asarray(group["xbin"], dtype=float)
    ybin = np.asarray(group["ybin"], dtype=float)
    xvel = np.asarray(group["xvel"], dtype=float)
    nbins = int(_as_scalar(group["nbins"]))
    nvel = len(xvel)
    priors = np.full((nbins, nvel), np.nan, dtype=float)
    neighbor_ids = np.full((nbins, k), -1, dtype=int)
    neighbor_weights = np.zeros((nbins, k), dtype=float)
    fallback = _fallback_losvd(xvel)

    if target_losvds is None or target_good is None or not np.any(target_good):
        priors[:] = fallback
        return priors, neighbor_ids, neighbor_weights

    valid = target_good & np.isfinite(xbin) & np.isfinite(ybin)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        priors[:] = fallback
        return priors, neighbor_ids, neighbor_weights

    scale = float(scale_arcsec)
    if not np.isfinite(scale) or scale <= 0:
        finite_xy = np.column_stack([xbin[valid_ids], ybin[valid_ids]])
        distances = []
        for xy in finite_xy:
            d = np.hypot(finite_xy[:, 0] - xy[0], finite_xy[:, 1] - xy[1])
            d = np.sort(d[d > 0])
            if d.size:
                distances.append(d[0])
        scale = float(np.nanmedian(distances) * 2.5) if distances else 0.35

    for bin_id in range(nbins):
        dx = xbin[valid_ids] - xbin[bin_id]
        dy = ybin[valid_ids] - ybin[bin_id]
        dist = np.hypot(dx, dy)
        not_self = valid_ids != bin_id
        candidates = valid_ids[not_self]
        cand_dist = dist[not_self]

        if candidates.size == 0:
            if valid[bin_id]:
                priors[bin_id] = target_losvds[bin_id]
            else:
                priors[bin_id] = fallback
            continue

        nearest_order = np.argsort(cand_dist)[:k]
        chosen = candidates[nearest_order]
        chosen_dist = cand_dist[nearest_order]
        weights = np.exp(-0.5 * (chosen_dist / scale) ** 2)
        if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
            weights = np.ones_like(chosen_dist)
        weights = weights / np.sum(weights)
        prior = np.sum(target_losvds[chosen] * weights[:, None], axis=0)
        priors[bin_id] = _normalise_losvd(prior)
        neighbor_ids[bin_id, : len(chosen)] = chosen
        neighbor_weights[bin_id, : len(chosen)] = weights

    return priors, neighbor_ids, neighbor_weights


def _init_vector(value, size, jitter_scale=0.0, lower=-1.5, upper=1.5, seed=12345):
    if value is None:
        out = np.zeros(size, dtype=float)
    else:
        out = np.asarray(value, dtype=float).ravel()
        if out.size != size:
            out = np.zeros(size, dtype=float)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if jitter_scale > 0:
        out = out + np.random.default_rng(seed).normal(0.0, jitter_scale, size=size)
    return np.clip(out, lower, upper)


def _init_losvd(common, vals, init_xvel, attempt, bin_id):
    xvel = np.asarray(common["xvel"], dtype=float)
    losvd = vals.get("losvd")
    if losvd is None:
        losvd = np.exp(-0.5 * (xvel / 220.0) ** 2)
    else:
        losvd = np.asarray(losvd, dtype=float).ravel()
        if init_xvel is not None and losvd.size == len(init_xvel) and losvd.size != len(xvel):
            order = np.argsort(init_xvel)
            losvd = np.interp(xvel, init_xvel[order], losvd[order], left=1e-10, right=1e-10)
        if losvd.size != len(xvel):
            losvd = np.exp(-0.5 * (xvel / 220.0) ** 2)
    losvd = _normalise_losvd(losvd)
    if attempt > 0:
        rng = np.random.default_rng(200000 + 1000 * attempt + bin_id)
        losvd = _normalise_losvd(losvd + rng.normal(0.0, 0.002, size=losvd.size))
    return losvd


def _make_init(common, bin_id, init_values, init_xvel, attempt):
    vals = init_values.get(bin_id, {})
    ntemp = int(common["ntemp"])
    porder = int(common["porder"])
    return {
        "losvd": _init_losvd(common, vals, init_xvel, attempt, bin_id),
        "sigma": float(np.clip(0.035 + 0.015 * attempt, 1e-4, 0.9)),
        "weights": _init_vector(
            vals.get("weights"),
            ntemp,
            jitter_scale=0.02 if attempt > 0 else 0.0,
            seed=300000 + bin_id + attempt,
        ),
        "coefs": _init_vector(
            vals.get("coefs"),
            porder + 1,
            jitter_scale=0.02 if attempt > 0 else 0.0,
            seed=400000 + bin_id + attempt,
        ),
    }


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


def _make_data(common, bin_id, spatial_prior=None, spatial_sigma=None):
    data = {
        key: val
        for key, val in common.items()
        if key not in {"spec_obs_all", "sigma_obs_all"}
    }
    data["spec_obs"] = common["spec_obs_all"][:, bin_id]
    data["sigma_obs"] = common["sigma_obs_all"][:, bin_id]
    if spatial_prior is not None:
        data["losvd_spatial_prior"] = spatial_prior
        data["sigma_spatial"] = spatial_sigma
    return data


def _write_failure(group, common, reason):
    group.attrs["fit_failed"] = True
    group.attrs["failure_reason"] = str(reason)
    npix = int(common["npix_obs"])
    nvel = int(common["nvel"])
    ntemp = int(common["ntemp"])
    porder = int(common["porder"])
    group.create_dataset("sigma", data=np.full(5, np.nan), compression="gzip")
    group.create_dataset("losvd", data=np.full((5, nvel), np.nan), compression="gzip")
    group.create_dataset("weights", data=np.full((5, ntemp), np.nan), compression="gzip")
    group.create_dataset("coefs", data=np.full((5, porder + 1), np.nan), compression="gzip")
    for name, size in {
        "spec": int(common["npix_temp"]),
        "conv_spec": npix,
        "poly": npix,
        "bestfit": npix,
    }.items():
        group.create_dataset(name, data=np.full((5, size), np.nan), compression="gzip")


def _default_cache_paths(stan_dir, fit_type):
    tag = fit_type.lower()
    return stan_dir / f"test-cache-{tag}.pkl", stan_dir / f"test-stanc-{tag}.pkl"


def _load_or_compile_model(codefile, model_cache, stanc_cache):
    if model_cache.exists():
        with model_cache.open("rb") as handle:
            return pickle.load(handle)

    code = codefile.read_text()
    if stanc_cache.exists():
        with stanc_cache.open("rb") as handle:
            stanc_ret = pickle.load(handle)
    else:
        stanc_ret = pystan.api.stanc(model_code=code)
        with stanc_cache.open("wb") as handle:
            pickle.dump(stanc_ret, handle)

    model = pystan.StanModel(stanc_ret=stanc_ret, extra_compile_args=["-w"])
    with model_cache.open("wb") as handle:
        pickle.dump(model, handle)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preproc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-type", choices=sorted(FIT_CODES), default="RW")
    parser.add_argument("--iter", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--bins", default="all", help="Comma-separated bin IDs, or 'all'.")
    parser.add_argument("--init-result", type=Path, default=None)
    parser.add_argument("--model-cache", type=Path, default=None)
    parser.add_argument("--stanc-cache", type=Path, default=None)
    parser.add_argument(
        "--spatial-prior-result",
        type=Path,
        default=None,
        help="Result HDF5 used to build neighbor-averaged LOSVD priors for RWSPATIAL.",
    )
    parser.add_argument("--spatial-k", type=int, default=6, help="Number of nearest neighboring bins for RWSPATIAL.")
    parser.add_argument(
        "--spatial-scale",
        type=float,
        default=0.35,
        help="Gaussian weighting scale in arcsec for RWSPATIAL neighbor averaging. Use <=0 for an adaptive scale.",
    )
    parser.add_argument(
        "--spatial-sigma",
        type=float,
        default=0.02,
        help="Per-velocity-channel sigma of the fixed spatial LOSVD prior for RWSPATIAL.",
    )
    parser.add_argument(
        "--output-fit-type",
        default=None,
        help="Optional fit_type label to store in the output HDF5 attributes.",
    )
    args = parser.parse_args()

    repo_scripts = Path(__file__).resolve().parents[1] / "BAYES-LOSVD" / "scripts"
    stan_dir = repo_scripts / "stan_model"
    codefile = stan_dir / FIT_CODES[args.fit_type]
    default_model_cache, default_stanc_cache = _default_cache_paths(stan_dir, args.fit_type)
    model_cache = args.model_cache or default_model_cache
    stanc_cache = args.stanc_cache or default_stanc_cache

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reproduction_kwargs = {
        "run_name": args.output.stem,
        "input_paths": [args.preproc, args.init_result, args.spatial_prior_result, codefile, model_cache, stanc_cache],
        "output_paths": [args.output],
        "extra": {
            "runner": "run_regularized_map.py",
            "fit_type": args.fit_type,
            "output_fit_type": args.output_fit_type,
            "model_code": str(codefile),
            "model_cache": str(model_cache),
            "stanc_cache": str(stanc_cache),
            "spatial_prior_result": str(args.spatial_prior_result) if args.spatial_prior_result else None,
            "spatial_k": args.spatial_k,
            "spatial_scale": args.spatial_scale,
            "spatial_sigma": args.spatial_sigma,
        },
    }
    run_file, manifest_file = write_reproduction_files(args.output.parent, **reproduction_kwargs)
    if args.output.exists() and args.restart:
        args.output.unlink()

    model = _load_or_compile_model(codefile, model_cache, stanc_cache)
    init_values, init_xvel = _load_init_values(args.init_result)

    with h5py.File(args.preproc, "r") as preproc, h5py.File(args.output, "a") as out:
        if "in" not in out:
            preproc.copy("in", out)
        out.attrs["fit_type"] = args.output_fit_type or f"{args.fit_type}regularized"
        out.attrs["fit_mode"] = "MAP optimization"
        out.attrs["regularization"] = args.fit_type
        out.attrs["optimizer_iter"] = args.iter
        out.attrs["model_code"] = str(codefile)
        out.attrs["reproduce_script"] = str(run_file)
        out.attrs["run_manifest"] = str(manifest_file)
        out_group = out.require_group("out")
        common = _load_common(preproc)
        nbins = int(_as_scalar(preproc["in/nbins"]))
        spatial_priors = None
        spatial_neighbor_ids = None
        spatial_neighbor_weights = None
        if args.fit_type == "RWSPATIAL":
            prior_result = args.spatial_prior_result or args.init_result
            if prior_result is None:
                raise ValueError("RWSPATIAL requires --spatial-prior-result or --init-result.")
            if args.spatial_k < 1:
                raise ValueError("--spatial-k must be at least 1.")
            if args.spatial_sigma <= 0:
                raise ValueError("--spatial-sigma must be positive.")
            prior_losvds, prior_good = _load_losvd_array(prior_result, np.asarray(common["xvel"], dtype=float), nbins)
            spatial_priors, spatial_neighbor_ids, spatial_neighbor_weights = _build_spatial_priors(
                preproc,
                prior_losvds,
                prior_good,
                args.spatial_k,
                args.spatial_scale,
            )
            out.attrs["spatial_regularization"] = "fixed_neighbor_losvd_prior"
            out.attrs["spatial_prior_result"] = str(prior_result)
            out.attrs["spatial_k"] = args.spatial_k
            out.attrs["spatial_scale_arcsec"] = args.spatial_scale
            out.attrs["spatial_sigma"] = args.spatial_sigma
            if "spatial_prior" in out:
                del out["spatial_prior"]
            prior_group = out.create_group("spatial_prior")
            prior_group.create_dataset("losvd", data=spatial_priors, compression="gzip")
            prior_group.create_dataset("neighbor_ids", data=spatial_neighbor_ids, compression="gzip")
            prior_group.create_dataset("neighbor_weights", data=spatial_neighbor_weights, compression="gzip")

        if args.bins == "all":
            bin_ids = list(range(nbins))
        else:
            bin_ids = [int(part) for part in args.bins.split(",") if part.strip()]

        for count, bin_id in enumerate(bin_ids, start=1):
            key = str(bin_id)
            if key in out_group and not args.restart:
                continue
            if key in out_group:
                del out_group[key]

            grp = out_group.create_group(key)
            data = _make_data(
                common,
                bin_id,
                spatial_prior=None if spatial_priors is None else spatial_priors[bin_id],
                spatial_sigma=None if spatial_priors is None else args.spatial_sigma,
            )
            result = None
            last_error = None
            for attempt in range(args.retries + 1):
                try:
                    with _quiet_fds():
                        result = model.optimizing(
                            data=data,
                            iter=args.iter,
                            init=_make_init(common, bin_id, init_values, init_xvel, attempt),
                            as_vector=False,
                            verbose=False,
                            seed=500000 + 1000 * attempt + bin_id,
                        )
                    break
                except Exception as exc:
                    last_error = exc

            if result is None:
                _write_failure(grp, common, last_error)
            else:
                grp.attrs["fit_failed"] = False
                grp.attrs["optimizer_value"] = float(result["value"])
                for name, value in result["par"].items():
                    if name == "lp__":
                        continue
                    grp.create_dataset(name, data=_five_row(value), compression="gzip")

            if count % 25 == 0 or count == len(bin_ids):
                print(f"completed {count}/{len(bin_ids)} bins", flush=True)

    write_reproduction_files(args.output.parent, **reproduction_kwargs)
    print(args.output)


if __name__ == "__main__":
    main()
