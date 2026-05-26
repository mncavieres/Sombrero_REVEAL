#!/usr/bin/env python3
"""
Generate JAM model maps for both MUSE and NIRSpec using the same parameter set.

The parameter vector comes from the best-fit MUSE logistic-anisotropy run:
    [M_BH, r_a, beta_0, beta_inf, alpha, M/L]

This script restores that MUSE run, extracts the current maximum-likelihood
sample, and evaluates the same JAM model on:
    - the MUSE pPXF kinematics
    - the NIRSpec/Antoine CSV kinematics

For each instrument it writes:
    - model LOSV map
    - model sigma map, derived from sqrt(max(Vrms^2 - Vlos^2, 0))
    - model Vrms map
    - a 3x3 observed/model/residual overview figure
    - an NPZ bundle with the arrays and metadata
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dynesty import DynamicNestedSampler


SCRIPT_DIR = Path(__file__).resolve().parent
JAMPY_DIR = SCRIPT_DIR / "jampy"
MUSE_JAM_DIR = SCRIPT_DIR / "muse_jam"

if str(MUSE_JAM_DIR) not in sys.path:
    sys.path.insert(0, str(MUSE_JAM_DIR))
if str(JAMPY_DIR) not in sys.path:
    sys.path.insert(0, str(JAMPY_DIR))

import nested_free_muse as muse


base = muse.base

OUTPUT_ROOT = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/"
    "shared_muse_logistic_maps"
)
MUSE_LOGISTIC_RUN_DIR = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/"
    "muse_logistic_beta_rwalk"
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def safe_restore_results(checkpoint_path: Path) -> object:
    with tempfile.NamedTemporaryFile(
        prefix="dynesty_shared_logistic_",
        suffix=".save",
        dir=checkpoint_path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)

    try:
        shutil.copy2(checkpoint_path, tmp_path)
        sampler = DynamicNestedSampler.restore(str(tmp_path))
        return sampler.results
    finally:
        tmp_path.unlink(missing_ok=True)


def load_best_muse_logistic_params() -> tuple[np.ndarray, Path, str]:
    results_npz = MUSE_LOGISTIC_RUN_DIR / "nested_bh_beta_ml_results.npz"
    if results_npz.exists():
        with np.load(results_npz, allow_pickle=True) as data:
            if "best_params" not in data:
                raise KeyError(f"'best_params' missing from {results_npz}")
            return np.asarray(data["best_params"], dtype=float), results_npz, "results_npz"

    checkpoint_path = MUSE_LOGISTIC_RUN_DIR / "checkpoint.save"
    if checkpoint_path.exists():
        results = safe_restore_results(checkpoint_path)
        if len(results.samples) == 0:
            raise RuntimeError(f"No samples found in checkpoint {checkpoint_path}")
        return (
            np.asarray(base.get_best_fit_parameters(results), dtype=float),
            checkpoint_path,
            "checkpoint",
        )

    raise FileNotFoundError(
        "No MUSE logistic results were found in "
        f"{MUSE_LOGISTIC_RUN_DIR}. Expected nested_bh_beta_ml_results.npz "
        "or checkpoint.save."
    )


def derived_sigma_model(vrms_model: np.ndarray, vlos_model: np.ndarray) -> np.ndarray:
    sig2 = np.maximum(np.asarray(vrms_model) ** 2 - np.asarray(vlos_model) ** 2, 0.0)
    return np.sqrt(sig2)


def percentile_limits(values: np.ndarray, *, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    good = values[np.isfinite(values)]
    if good.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(good, low))
    vmax = float(np.nanpercentile(good, high))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def save_model_maps(
    out_dir: Path,
    kin,
    *,
    model_vlos: np.ndarray,
    model_sigma: np.ndarray,
    model_vrms: np.ndarray,
    instrument_label: str,
) -> None:
    vmax_vlos = base.safe_symmetric_limit(
        np.r_[kin.vlos_rf[kin.goodbins], model_vlos[kin.goodbins]],
        fallback=1.0,
    )
    sigma_vmin, sigma_vmax = percentile_limits(
        np.r_[kin.sigma[kin.goodbins], model_sigma[kin.goodbins]]
    )
    vrms_vmin, vrms_vmax = percentile_limits(
        np.r_[kin.vrms[kin.goodbins], model_vrms[kin.goodbins]]
    )

    base.save_interpolated_map(
        out_dir / "model_vlos.png",
        kin.xbin,
        kin.ybin,
        model_vlos,
        title=f"{instrument_label} JAM model LOSV",
        cbar_label="km/s",
        cmap="RdBu_r",
        vmin=-vmax_vlos,
        vmax=vmax_vlos,
    )
    base.save_interpolated_map(
        out_dir / "model_sigma.png",
        kin.xbin,
        kin.ybin,
        model_sigma,
        title=f"{instrument_label} JAM model sigma",
        cbar_label="km/s",
        cmap="viridis",
        vmin=max(0.0, sigma_vmin),
        vmax=sigma_vmax,
    )
    base.save_interpolated_map(
        out_dir / "model_vrms.png",
        kin.xbin,
        kin.ybin,
        model_vrms,
        title=f"{instrument_label} JAM model Vrms",
        cbar_label="km/s",
        cmap="viridis",
        vmin=max(0.0, vrms_vmin),
        vmax=vrms_vmax,
    )


def save_overview_figure(
    out_dir: Path,
    kin,
    *,
    model_vlos: np.ndarray,
    model_sigma: np.ndarray,
    model_vrms: np.ndarray,
    instrument_label: str,
    param_summary: str,
) -> None:
    quantities = [
        ("Vlos", kin.vlos_rf, model_vlos, "RdBu_r", True),
        ("Sigma", kin.sigma, model_sigma, "viridis", False),
        ("Vrms", kin.vrms, model_vrms, "viridis", False),
    ]
    extent = (
        float(np.nanmin(kin.xbin)),
        float(np.nanmax(kin.xbin)),
        float(np.nanmin(kin.ybin)),
        float(np.nanmax(kin.ybin)),
    )

    fig, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)

    for row, (label, observed, model, cmap, symmetric) in enumerate(quantities):
        residual = observed - model
        _, _, obs_grid = base.interpolate_to_grid(kin.xbin, kin.ybin, observed)
        _, _, model_grid = base.interpolate_to_grid(kin.xbin, kin.ybin, model)
        _, _, resid_grid = base.interpolate_to_grid(kin.xbin, kin.ybin, residual)

        if symmetric:
            vmax = base.safe_symmetric_limit(
                np.r_[observed[kin.goodbins], model[kin.goodbins]],
                fallback=1.0,
            )
            data_vmin, data_vmax = -vmax, vmax
        else:
            data_vmin, data_vmax = percentile_limits(
                np.r_[observed[kin.goodbins], model[kin.goodbins]]
            )
            data_vmin = max(0.0, data_vmin)

        resid_lim = base.safe_symmetric_limit(residual[kin.goodbins], fallback=1.0)

        panels = [
            ("Observed", obs_grid, data_vmin, data_vmax),
            ("Model", model_grid, data_vmin, data_vmax),
            ("Residual", resid_grid, -resid_lim, resid_lim),
        ]

        for col, (panel_label, grid, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(
                grid,
                origin="lower",
                extent=extent,
                cmap=cmap if panel_label != "Residual" else "RdBu_r",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("X (arcsec)")
            ax.set_ylabel("Y (arcsec)")
            ax.set_title(f"{label} {panel_label}")
            fig.colorbar(im, ax=ax, label="km/s")

    fig.suptitle(f"{instrument_label} with MUSE logistic best fit\n{param_summary}")
    fig.savefig(out_dir / "overview_observed_model_residual.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_reduced_chi2(residual: np.ndarray, errors: np.ndarray, goodbins: np.ndarray) -> float:
    mask = np.asarray(goodbins, dtype=bool) & np.isfinite(residual) & np.isfinite(errors) & (errors > 0)
    if not np.any(mask):
        return float("nan")
    chi2 = np.sum((np.asarray(residual)[mask] / np.asarray(errors)[mask]) ** 2)
    return float(chi2 / np.count_nonzero(mask))


def generate_products(instrument_name: str, cfg, kin, mge, best_params: np.ndarray) -> dict:
    out_dir = OUTPUT_ROOT / instrument_name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    _, model_vlos, resid_vlos = base.compute_bestfit_losv_model(cfg, kin, mge, best_params)
    _, model_vrms, resid_vrms = base.compute_bestfit_vrms_model(cfg, kin, mge, best_params)
    model_sigma = derived_sigma_model(model_vrms, model_vlos)
    resid_sigma = np.asarray(kin.sigma, dtype=float) - model_sigma

    summary = base.summarize_best_params(cfg, best_params, len(mge.surf_lum))
    param_summary = (
        f"Mbh={summary['best_bh_mass']:.3e} Msun, "
        f"M/L={summary['best_ml']:.3f}, "
        f"r_a={summary['best_beta_ra']:.3f}, "
        f"beta_0={summary['best_beta_0']:.3f}, "
        f"beta_inf={summary['best_beta_inf']:.3f}, "
        f"alpha={summary['best_beta_alpha']:.3f}"
    )

    save_model_maps(
        out_dir,
        kin,
        model_vlos=model_vlos,
        model_sigma=model_sigma,
        model_vrms=model_vrms,
        instrument_label=instrument_name,
    )
    save_overview_figure(
        out_dir,
        kin,
        model_vlos=model_vlos,
        model_sigma=model_sigma,
        model_vrms=model_vrms,
        instrument_label=instrument_name,
        param_summary=param_summary,
    )

    np.savez(
        out_dir / "model_products.npz",
        instrument=np.array(instrument_name),
        beta_prescription=np.array("logistic"),
        best_params=np.asarray(best_params, dtype=float),
        best_bh_mass=np.array(summary["best_bh_mass"], dtype=float),
        best_ml=np.array(summary["best_ml"], dtype=float),
        best_beta_logistic=np.asarray(summary["best_beta_logistic"], dtype=float),
        xbin=np.asarray(kin.xbin, dtype=float),
        ybin=np.asarray(kin.ybin, dtype=float),
        goodbins=np.asarray(kin.goodbins, dtype=bool),
        observed_vlos_rf=np.asarray(kin.vlos_rf, dtype=float),
        observed_sigma=np.asarray(kin.sigma, dtype=float),
        observed_vrms=np.asarray(kin.vrms, dtype=float),
        observed_vlos_err=np.asarray(kin.vlos_err, dtype=float),
        observed_sigma_err=np.asarray(kin.sigma_err, dtype=float),
        observed_vrms_err=np.asarray(kin.vrms_err, dtype=float),
        model_vlos=np.asarray(model_vlos, dtype=float),
        model_sigma=np.asarray(model_sigma, dtype=float),
        model_vrms=np.asarray(model_vrms, dtype=float),
        residual_vlos=np.asarray(resid_vlos, dtype=float),
        residual_sigma=np.asarray(resid_sigma, dtype=float),
        residual_vrms=np.asarray(resid_vrms, dtype=float),
    )

    reduced_chi2 = {
        "vlos": compute_reduced_chi2(resid_vlos, kin.vlos_err, kin.goodbins),
        "sigma": compute_reduced_chi2(resid_sigma, kin.sigma_err, kin.goodbins),
        "vrms": compute_reduced_chi2(resid_vrms, kin.vrms_err, kin.goodbins),
    }

    return {
        "instrument": instrument_name,
        "output_dir": str(out_dir),
        "rotation_deg": float(cfg.rotation_deg),
        "pixsize_arcsec": float(cfg.pixsize_arcsec),
        "sigmapsf_arcsec": float(cfg.sigmapsf_arcsec),
        "n_bins": int(len(kin.xbin)),
        "n_goodbins": int(np.count_nonzero(kin.goodbins)),
        "reduced_chi2": reduced_chi2,
    }


def build_muse_logistic_cfg() -> muse.Config:
    cfg = muse.Config()
    cfg.beta_prescription = "logistic"
    cfg.output_dir = MUSE_LOGISTIC_RUN_DIR
    cfg.sample_method = "rwalk"
    cfg.bound_method = "multi"
    cfg.walks = 32
    cfg.bootstrap = 20
    cfg.beta_min = -4.0
    cfg.beta_max = 0.99
    cfg.beta_ra_min = 0.2
    cfg.beta_ra_max = 30.0
    cfg.beta_alpha_min = 0.5
    cfg.beta_alpha_max = 5.0
    return cfg


def build_nirspec_logistic_cfg() -> base.Config:
    cfg = base.Config()
    cfg = base.Config(
        output_dir=cfg.output_dir,
        kin_path=cfg.kin_path,
        mge_solution_path=cfg.mge_solution_path,
        mge_luminosity_path=cfg.mge_luminosity_path,
        rotation_deg=cfg.rotation_deg,
        redshift=cfg.redshift,
        distance_mpc=cfg.distance_mpc,
        inclination_deg=cfg.inclination_deg,
        sigmapsf_arcsec=cfg.sigmapsf_arcsec,
        pixsize_arcsec=cfg.pixsize_arcsec,
        pixel_scale_arcsec=cfg.pixel_scale_arcsec,
        nlive=cfg.nlive,
        nprocs=cfg.nprocs,
        dlogz_init=cfg.dlogz_init,
        checkpoint_every_sec=cfg.checkpoint_every_sec,
        checkpoint_filename=cfg.checkpoint_filename,
        bound_method=cfg.bound_method,
        sample_method=cfg.sample_method,
        walks=cfg.walks,
        bootstrap=cfg.bootstrap,
        bh_mass_min=cfg.bh_mass_min,
        bh_mass_max=cfg.bh_mass_max,
        beta_min=-4.0,
        beta_max=0.99,
        ml_min=cfg.ml_min,
        ml_max=cfg.ml_max,
        beta_prescription="logistic",
        beta_ra_min=0.2,
        beta_ra_max=30.0,
        beta_alpha_min=0.5,
        beta_alpha_max=5.0,
    )
    return cfg


def main() -> None:
    setup_logging()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    best_params, source_path, source_kind = load_best_muse_logistic_params()
    logging.info("Loaded MUSE logistic best-fit parameters from %s (%s)", source_path, source_kind)
    logging.info("Best-fit theta = %s", np.array2string(best_params, precision=6, separator=", "))

    muse_cfg = build_muse_logistic_cfg()
    nirspec_cfg = build_nirspec_logistic_cfg()

    logging.info("Loading shared MGE inputs")
    mge = base.load_mge_inputs(nirspec_cfg)

    logging.info("Loading MUSE kinematics")
    muse_kin = muse.load_kinematics(muse_cfg)

    logging.info("Loading NIRSpec kinematics")
    nirspec_kin = base.load_kinematics(nirspec_cfg)

    summaries = {
        "source_kind": source_kind,
        "source_path": str(source_path),
        "best_params": [float(x) for x in best_params],
        "parameter_order": ["M_BH", "r_a", "beta_0", "beta_inf", "alpha", "M/L"],
    }
    summaries["muse"] = generate_products("MUSE", muse_cfg, muse_kin, mge, best_params)
    summaries["nirspec"] = generate_products("NIRSpec", nirspec_cfg, nirspec_kin, mge, best_params)

    with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    logging.info("Wrote outputs to %s", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
