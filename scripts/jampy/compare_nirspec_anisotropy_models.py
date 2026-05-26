#!/usr/bin/env python3
"""
Compare the NIRSpec JAM free, logistic, and constant anisotropy fits.

This mirrors the MUSE comparison script but uses the NIRSpec JAM result sets
from the non-symmetrized run tree:
    - free anisotropy:      Data/jam_models/no_symmetrization/nirspec_free_beta
    - logistic anisotropy:  Data/jam_models/no_symmetrization/nirspec_logistic_beta
    - constant anisotropy:  Data/jam_models/no_symmetrization/nirspec_constant_beta

The logistic run is checkpoint-only in this workspace, so the comparison falls
back to the current maximum-likelihood sample in that checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import logging
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from dynesty import DynamicNestedSampler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

NIRSPEC_JAM_DIR = Path(__file__).resolve().parents[1] / "nirspec_jam"
if str(NIRSPEC_JAM_DIR) not in sys.path:
    sys.path.insert(0, str(NIRSPEC_JAM_DIR))

import nested_free as base

import nested_free_nirspec as nirspec


@dataclass
class ModelSpec:
    name: str
    label: str
    color: str
    cfg: base.Config


@dataclass
class ModelComparison:
    spec: ModelSpec
    source_kind: str
    source_path: Path
    best_params: np.ndarray
    best_bh_mass: float
    bh_mass_p16: float
    bh_mass_p50: float
    bh_mass_p84: float
    best_ml: float
    model_vrms: np.ndarray
    residual_vrms: np.ndarray
    chi2: float
    reduced_chi2: float
    jam_reported_reduced_chi2: float


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def make_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="free",
            label="Free beta",
            color="tab:blue",
            cfg=nirspec.make_free_config(),
        ),
        ModelSpec(
            name="logistic",
            label="Logistic beta",
            color="tab:orange",
            cfg=nirspec.make_logistic_config(),
        ),
        ModelSpec(
            name="constant",
            label="Constant beta",
            color="tab:green",
            cfg=nirspec.make_constant_config(),
        ),
    ]


def output_dir() -> Path:
    path = nirspec.NO_SYM_OUTPUT_ROOT / "nirspec_anisotropy_comparison"
    path.mkdir(parents=True, exist_ok=True)
    return path


def restore_checkpoint_safely(checkpoint_path: Path, work_dir: Path):
    with tempfile.NamedTemporaryFile(
        prefix="dynesty_compare_",
        suffix=".save",
        dir=work_dir,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        shutil.copy2(checkpoint_path, tmp_path)
        sampler = DynamicNestedSampler.restore(str(tmp_path))
        return sampler.results
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def weighted_quantile(values: np.ndarray, quantiles: list[float], weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.full_like(quantiles, np.nan, dtype=float)

    values = values[good]
    weights = weights[good]
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, values)


def posterior_bh_quantiles(samples: np.ndarray, logwt: np.ndarray, logz: np.ndarray) -> tuple[float, float, float]:
    samples = np.asarray(samples, dtype=float)
    logwt = np.asarray(logwt, dtype=float)
    logz = np.asarray(logz, dtype=float)

    log_weights = logwt - float(logz[-1])
    log_weights -= np.nanmax(log_weights)
    weights = np.exp(log_weights)

    q16, q50, q84 = weighted_quantile(samples[:, 0], [0.15865, 0.5, 0.84135], weights)
    return float(q16), float(q50), float(q84)


def load_model_state_from_npz(results_path: Path):
    with np.load(results_path, allow_pickle=True) as data:
        if "best_params" not in data:
            raise KeyError(f"'best_params' not found in {results_path}")
        best_params = np.asarray(data["best_params"], dtype=float)
        samples = np.asarray(data["samples"], dtype=float)
        logwt = np.asarray(data["logwt"], dtype=float)
        logz = np.asarray(data["logz"], dtype=float)
    return best_params, samples, logwt, logz


def load_model_state_from_checkpoint(checkpoint_path: Path, work_dir: Path):
    results = restore_checkpoint_safely(checkpoint_path, work_dir)
    if len(results.samples) == 0:
        raise RuntimeError(f"Checkpoint contains zero samples: {checkpoint_path}")
    best_params = np.asarray(base.get_best_fit_parameters(results), dtype=float)
    samples = np.asarray(results.samples, dtype=float)
    logwt = np.asarray(results.logwt, dtype=float)
    logz = np.asarray(results.logz, dtype=float)
    return best_params, samples, logwt, logz


def resolve_model_state(spec: ModelSpec, work_dir: Path):
    results_npz = spec.cfg.output_dir / "nested_bh_beta_ml_results.npz"
    if results_npz.exists():
        best_params, samples, logwt, logz = load_model_state_from_npz(results_npz)
        return best_params, samples, logwt, logz, "results_npz", results_npz

    checkpoint = spec.cfg.output_dir / spec.cfg.checkpoint_filename
    if checkpoint.exists():
        best_params, samples, logwt, logz = load_model_state_from_checkpoint(checkpoint, work_dir)
        return best_params, samples, logwt, logz, "checkpoint", checkpoint

    raise FileNotFoundError(
        f"No final results NPZ or checkpoint found for {spec.label} in {spec.cfg.output_dir}"
    )


def compute_model_comparison(spec: ModelSpec, kin, mge, work_dir: Path) -> ModelComparison:
    best_params, samples, logwt, logz, source_kind, source_path = resolve_model_state(spec, work_dir)
    out, model, residual = base.compute_bestfit_vrms_model(spec.cfg, kin, mge, best_params)

    good = np.asarray(kin.goodbins, dtype=bool)
    chi2 = float(np.sum((residual[good] / kin.vrms_err[good]) ** 2))
    reduced_chi2 = chi2 / float(np.count_nonzero(good))
    summary = base.summarize_best_params(spec.cfg, best_params, len(mge.surf_lum))
    bh_mass_p16, bh_mass_p50, bh_mass_p84 = posterior_bh_quantiles(samples, logwt, logz)

    return ModelComparison(
        spec=spec,
        source_kind=source_kind,
        source_path=source_path,
        best_params=best_params,
        best_bh_mass=float(summary["best_bh_mass"]),
        bh_mass_p16=bh_mass_p16,
        bh_mass_p50=bh_mass_p50,
        bh_mass_p84=bh_mass_p84,
        best_ml=float(summary["best_ml"]),
        model_vrms=np.asarray(model, dtype=float),
        residual_vrms=np.asarray(residual, dtype=float),
        chi2=chi2,
        reduced_chi2=reduced_chi2,
        jam_reported_reduced_chi2=float(getattr(out, "chi2", np.nan)),
    )


def prepare_grid(x: np.ndarray, y: np.ndarray, values: np.ndarray):
    _, _, grid = base.interpolate_to_grid(x, y, values)
    extent = (np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y))
    return grid, extent


def save_map_comparison(kin, comparisons: list[ModelComparison], out_dir: Path) -> None:
    ncols = len(comparisons)
    fig, axes = plt.subplots(3, ncols, figsize=(5.0 * ncols, 11.5), constrained_layout=True)
    axes = np.asarray(axes)

    data_grid, extent = prepare_grid(kin.xbin, kin.ybin, kin.vrms)
    model_grids = [prepare_grid(kin.xbin, kin.ybin, comp.model_vrms)[0] for comp in comparisons]
    residual_grids = [prepare_grid(kin.xbin, kin.ybin, comp.residual_vrms)[0] for comp in comparisons]

    vrms_values = [data_grid[np.isfinite(data_grid)]]
    vrms_values.extend(grid[np.isfinite(grid)] for grid in model_grids if np.any(np.isfinite(grid)))
    vrms_all = np.concatenate(vrms_values)
    vrms_vmin = float(np.nanpercentile(vrms_all, 1.0))
    vrms_vmax = float(np.nanpercentile(vrms_all, 99.0))

    residual_values = [grid[np.isfinite(grid)] for grid in residual_grids if np.any(np.isfinite(grid))]
    residual_all = np.concatenate(residual_values)
    residual_lim = base.safe_symmetric_limit(residual_all, fallback=1.0)

    ims_model = []
    ims_resid = []

    for col, comp in enumerate(comparisons):
        ax_data = axes[0, col]
        ax_data.imshow(
            data_grid,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="auto",
            vmin=vrms_vmin,
            vmax=vrms_vmax,
        )
        ax_data.set_title(
            f"{comp.spec.label}\n"
            f"$\\chi^2={comp.chi2:.1f}$, "
            f"$\\chi^2_\\nu={comp.reduced_chi2:.2f}$",
            color=comp.spec.color,
        )
        ax_data.set_xlabel("X (arcsec)")
        ax_data.set_ylabel("Y (arcsec)")
        if col == 0:
            ax_data.text(
                0.03,
                0.97,
                "Observed Vrms",
                transform=ax_data.transAxes,
                ha="left",
                va="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        ax_model = axes[1, col]
        im_model = ax_model.imshow(
            model_grids[col],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="auto",
            vmin=vrms_vmin,
            vmax=vrms_vmax,
        )
        ims_model.append(im_model)
        ax_model.set_xlabel("X (arcsec)")
        ax_model.set_ylabel("Y (arcsec)")
        if col == 0:
            ax_model.text(
                0.03,
                0.97,
                "Best-fit model",
                transform=ax_model.transAxes,
                ha="left",
                va="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        ax_resid = axes[2, col]
        im_resid = ax_resid.imshow(
            residual_grids[col],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-residual_lim,
            vmax=residual_lim,
        )
        ims_resid.append(im_resid)
        ax_resid.set_xlabel("X (arcsec)")
        ax_resid.set_ylabel("Y (arcsec)")
        if col == 0:
            ax_resid.text(
                0.03,
                0.97,
                "Residual (data - model)",
                transform=ax_resid.transAxes,
                ha="left",
                va="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    fig.colorbar(ims_model[0], ax=axes[0:2, :].ravel().tolist(), label="Vrms (km/s)", shrink=0.95)
    fig.colorbar(ims_resid[0], ax=axes[2, :].ravel().tolist(), label="Residual (km/s)", shrink=0.95)
    fig.suptitle("NIRSpec JAM anisotropy comparison: data, model, and residuals", fontsize=15)
    fig.savefig(out_dir / "nirspec_vrms_data_model_residual_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_chi2_plot(comparisons: list[ModelComparison], out_dir: Path) -> None:
    labels = [comp.spec.label for comp in comparisons]
    colors = [comp.spec.color for comp in comparisons]
    chi2 = np.array([comp.chi2 for comp in comparisons], dtype=float)
    reduced = np.array([comp.reduced_chi2 for comp in comparisons], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    axes[0].bar(labels, chi2, color=colors)
    axes[0].set_ylabel(r"$\chi^2$")
    axes[0].set_title("Absolute chi-square")
    axes[0].tick_params(axis="x", rotation=20)
    for idx, value in enumerate(chi2):
        axes[0].text(idx, value, f"{value:.0f}", ha="center", va="bottom")

    axes[1].bar(labels, reduced, color=colors)
    axes[1].set_ylabel(r"Reduced $\chi^2$")
    axes[1].set_title("Reduced chi-square")
    axes[1].tick_params(axis="x", rotation=20)
    for idx, value in enumerate(reduced):
        axes[1].text(idx, value, f"{value:.2f}", ha="center", va="bottom")

    fig.suptitle("NIRSpec JAM anisotropy model chi-square comparison", fontsize=15)
    fig.savefig(out_dir / "nirspec_anisotropy_chi2_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_bh_mass_plot(comparisons: list[ModelComparison], out_dir: Path) -> None:
    labels = [comp.spec.label for comp in comparisons]
    colors = [comp.spec.color for comp in comparisons]
    best_bh_mass = np.array([comp.best_bh_mass for comp in comparisons], dtype=float)
    bh_mass_p16 = np.array([comp.bh_mass_p16 for comp in comparisons], dtype=float)
    bh_mass_p50 = np.array([comp.bh_mass_p50 for comp in comparisons], dtype=float)
    bh_mass_p84 = np.array([comp.bh_mass_p84 for comp in comparisons], dtype=float)
    yerr = np.vstack([bh_mass_p50 - bh_mass_p16, bh_mass_p84 - bh_mass_p50])
    xpos = np.arange(len(comparisons), dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    for idx, comp in enumerate(comparisons):
        ax.errorbar(
            xpos[idx],
            bh_mass_p50[idx],
            yerr=yerr[:, idx][:, None],
            fmt="o",
            ms=8,
            lw=1.8,
            capsize=5,
            color=colors[idx],
        )

    ax.scatter(
        xpos,
        best_bh_mass,
        marker="x",
        s=70,
        color="black",
        linewidths=1.8,
        label="Best fit",
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_ylabel(r"$M_{\rm BH}$ ($M_\odot$)")
    ax.set_title("Black-hole mass by anisotropy prescription")
    ax.set_xticks(xpos, labels, rotation=20)
    ax.legend(frameon=False)

    for idx in range(len(comparisons)):
        ax.text(xpos[idx], best_bh_mass[idx], f"{best_bh_mass[idx]:.2e}", ha="center", va="bottom")

    fig.savefig(out_dir / "nirspec_bestfit_bh_mass_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(comparisons: list[ModelComparison], out_dir: Path) -> None:
    outpath = out_dir / "nirspec_anisotropy_comparison_summary.csv"
    with outpath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_name",
                "model_label",
                "source_kind",
                "source_path",
                "best_bh_mass",
                "bh_mass_p16",
                "bh_mass_p50",
                "bh_mass_p84",
                "best_ml",
                "chi2",
                "reduced_chi2",
                "jam_reported_reduced_chi2",
            ]
        )
        for comp in comparisons:
            writer.writerow(
                [
                    comp.spec.name,
                    comp.spec.label,
                    comp.source_kind,
                    str(comp.source_path),
                    f"{comp.best_bh_mass:.12e}",
                    f"{comp.bh_mass_p16:.12e}",
                    f"{comp.bh_mass_p50:.12e}",
                    f"{comp.bh_mass_p84:.12e}",
                    f"{comp.best_ml:.12e}",
                    f"{comp.chi2:.12e}",
                    f"{comp.reduced_chi2:.12e}",
                    f"{comp.jam_reported_reduced_chi2:.12e}",
                ]
            )


def main() -> None:
    setup_logging()
    out_dir = output_dir()
    specs = make_model_specs()

    logging.info("Loading common NIRSpec kinematics and MGE inputs")
    kin = base.load_kinematics(specs[0].cfg)
    mge = base.load_mge_inputs(specs[0].cfg)

    comparisons = []
    for spec in specs:
        logging.info("Processing %s", spec.label)
        comparisons.append(compute_model_comparison(spec, kin, mge, out_dir))

    save_map_comparison(kin, comparisons, out_dir)
    save_chi2_plot(comparisons, out_dir)
    save_bh_mass_plot(comparisons, out_dir)
    save_summary_csv(comparisons, out_dir)

    logging.info("Wrote comparison products to %s", out_dir)


if __name__ == "__main__":
    main()
