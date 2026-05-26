"""
Evaluate how the NIRSpec JAM Vrms model changes across a grid of constant beta.

This is a lightweight diagnostic script: it does not sample any parameters.
Instead, it keeps the black-hole mass and stellar M/L fixed to the best-fit
values from an existing NIRSpec free-beta run and only varies the constant
anisotropy beta.

Default beta grid:
    beta = 1.0, 0.5, 0.0, ..., -5.0

Outputs:
    - a multi-panel figure with the observed Vrms map plus one JAM model map
      per beta value,
    - a radial-profile + chi2 summary figure,
    - a CSV table summarizing chi2 for each beta,
    - an NPZ file with all evaluated models.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nested_free as base


@dataclass(frozen=True)
class Config:
    output_dir: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/nirspec_beta_sweep"
    )

    reference_results_path: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/free_beta_2/nested_bh_beta_ml_results.npz"
    )

    kin_path: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/antoine/M104_stellar_Kin.csv"
    )

    mge_solution_path: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_solution.csv"
    )

    mge_luminosity_path: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_luminosity_table.csv"
    )

    rotation_deg: float = -18.0
    redshift: float = 0.003633
    distance_mpc: float = 9.55
    inclination_deg: float = 87.0

    sigmapsf_arcsec: float = 0.1
    pixsize_arcsec: float = 0.1
    pixel_scale_arcsec: float = 0.031

    beta_start: float = 1.0
    beta_stop: float = -5.0
    beta_step: float = -0.5
    beta_upper_clip: float = 0.99


@dataclass
class BetaSweepResult:
    requested_beta: float
    evaluated_beta: float
    chi2: float
    reduced_chi2: float
    jam_reported_chi2: float
    model: np.ndarray
    residual: np.ndarray


def build_beta_grid(cfg: Config) -> np.ndarray:
    if cfg.beta_step == 0:
        raise ValueError("beta_step must be non-zero")

    n_steps_float = (cfg.beta_stop - cfg.beta_start) / cfg.beta_step
    n_steps = int(round(n_steps_float)) + 1
    if n_steps <= 0:
        raise ValueError(
            f"Invalid beta grid: start={cfg.beta_start}, stop={cfg.beta_stop}, step={cfg.beta_step}"
        )

    beta_values = cfg.beta_start + cfg.beta_step * np.arange(n_steps, dtype=float)
    beta_values[-1] = cfg.beta_stop
    return beta_values


def load_reference_parameters(cfg: Config) -> tuple[float, float]:
    with np.load(cfg.reference_results_path, allow_pickle=True) as data:
        if "best_bh_mass" in data and "best_ml" in data:
            bh_mass = float(data["best_bh_mass"])
            ml_fit = float(data["best_ml"])
        elif "best_params" in data:
            best_params = np.asarray(data["best_params"], dtype=float)
            bh_mass = float(best_params[0])
            ml_fit = float(best_params[-1])
        else:
            raise KeyError(
                f"Could not find best_bh_mass/best_ml or best_params in {cfg.reference_results_path}"
            )

    return bh_mass, ml_fit


def sanitize_beta(beta_requested: float, cfg: Config) -> float:
    if beta_requested >= cfg.beta_upper_clip:
        logging.warning(
            "Requested beta=%.2f is at/above the JAM stability limit; evaluating at beta=%.2f instead",
            beta_requested,
            cfg.beta_upper_clip,
        )
        return cfg.beta_upper_clip
    return beta_requested


def evaluate_vrms_model(
    cfg: Config,
    kin,
    mge,
    bh_mass: float,
    ml_fit: float,
    beta_value: float,
):
    beta = np.full(len(mge.surf_lum), beta_value, dtype=float)

    out = base.jam.axi.proj(
        surf_lum=mge.surf_lum,
        sigma_lum=mge.sigma_lum,
        qobs_lum=mge.q_obs_lum,
        surf_pot=mge.surf_lum * ml_fit,
        sigma_pot=mge.sigma_lum,
        qobs_pot=mge.q_obs_lum,
        inc=cfg.inclination_deg,
        mbh=bh_mass,
        distance=cfg.distance_mpc,
        xbin=kin.xbin,
        ybin=kin.ybin,
        align="cyl",
        analytic_los=True,
        beta=beta,
        data=kin.vrms,
        errors=kin.vrms_err,
        flux_obs=None,
        gamma=None,
        goodbins=kin.goodbins,
        interp=True,
        kappa=None,
        sigmapsf=cfg.sigmapsf_arcsec,
        normpsf=np.array([1.0]),
        pixsize=cfg.pixsize_arcsec,
        pixang=cfg.rotation_deg,
        logistic=False,
        ml=1.0,
        moment="zz",
        epsrel=1e-2,
        plot=False,
        quiet=True,
    )

    model = np.asarray(out.model, dtype=float)
    residual = kin.vrms - model
    norm_residual = residual[kin.goodbins] / kin.vrms_err[kin.goodbins]
    chi2 = float(norm_residual @ norm_residual)
    reduced_chi2 = chi2 / float(np.count_nonzero(kin.goodbins))
    jam_reported_chi2 = float(getattr(out, "chi2", np.nan))

    return model, residual, chi2, reduced_chi2, jam_reported_chi2


def run_beta_sweep(cfg: Config, kin, mge, bh_mass: float, ml_fit: float) -> list[BetaSweepResult]:
    results: list[BetaSweepResult] = []

    for beta_requested in build_beta_grid(cfg):
        beta_eval = sanitize_beta(float(beta_requested), cfg)
        logging.info("Evaluating constant beta=%0.2f", beta_eval)

        model, residual, chi2, reduced_chi2, jam_reported_chi2 = evaluate_vrms_model(
            cfg=cfg,
            kin=kin,
            mge=mge,
            bh_mass=bh_mass,
            ml_fit=ml_fit,
            beta_value=beta_eval,
        )

        results.append(
            BetaSweepResult(
                requested_beta=float(beta_requested),
                evaluated_beta=float(beta_eval),
                chi2=chi2,
                reduced_chi2=reduced_chi2,
                jam_reported_chi2=jam_reported_chi2,
                model=model,
                residual=residual,
            )
        )

    return results


def prepare_map_grid(x: np.ndarray, y: np.ndarray, values: np.ndarray):
    _, _, grid = base.interpolate_to_grid(x, y, values)
    extent = (np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y))
    return grid, extent


def should_exclude_from_shared_colorbar(cfg: Config, result: BetaSweepResult) -> bool:
    return np.isclose(result.evaluated_beta, cfg.beta_upper_clip)


def make_panel_title(result: BetaSweepResult) -> str:
    if np.isclose(result.requested_beta, result.evaluated_beta):
        title = rf"$\beta={result.requested_beta:.1f}$"
    else:
        title = rf"$\beta={result.requested_beta:.1f}$" + "\n" + rf"(eval {result.evaluated_beta:.2f})"
    return title + "\n" + rf"$\chi^2_\nu={result.reduced_chi2:.2f}$"


def save_model_map_grid(
    cfg: Config,
    kin,
    beta_results: list[BetaSweepResult],
    bh_mass: float,
    ml_fit: float,
) -> None:
    obs_grid, extent = prepare_map_grid(kin.xbin, kin.ybin, kin.vrms)
    model_grids = [prepare_map_grid(kin.xbin, kin.ybin, result.model)[0] for result in beta_results]

    grids_for_shared_colorbar = [obs_grid]
    for result, grid in zip(beta_results, model_grids):
        if not should_exclude_from_shared_colorbar(cfg, result):
            grids_for_shared_colorbar.append(grid)

    finite_values = [
        grid[np.isfinite(grid)]
        for grid in grids_for_shared_colorbar
        if np.any(np.isfinite(grid))
    ]
    all_values = np.concatenate(finite_values)
    vmin = float(np.nanpercentile(all_values, 1.0))
    vmax = float(np.nanpercentile(all_values, 99.0))

    n_panels = 1 + len(model_grids)
    ncols = 4
    nrows = int(ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.1 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    im = axes[0].imshow(
        obs_grid,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Observed Vrms")
    axes[0].set_xlabel("X (arcsec)")
    axes[0].set_ylabel("Y (arcsec)")

    for ax, result, grid in zip(axes[1:], beta_results, model_grids):
        im_panel = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(make_panel_title(result))
        ax.set_xlabel("X (arcsec)")
        ax.set_ylabel("Y (arcsec)")
        if should_exclude_from_shared_colorbar(cfg, result):
            ax.text(
                0.03,
                0.03,
                "Excluded from shared color scale",
                transform=ax.transAxes,
                fontsize=8,
                color="black",
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.colorbar(im, ax=axes[:n_panels].tolist(), label="km/s", shrink=0.92)
    fig.suptitle(
        f"NIRSpec JAM constant-beta sweep | fixed M_BH={bh_mass:.3e} Msun | fixed M/L={ml_fit:.3f}",
        fontsize=14,
    )
    fig.savefig(cfg.output_dir / "vrms_model_grid_vs_beta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_map_grid_individual_colorbars(
    cfg: Config,
    kin,
    beta_results: list[BetaSweepResult],
    bh_mass: float,
    ml_fit: float,
) -> None:
    obs_grid, extent = prepare_map_grid(kin.xbin, kin.ybin, kin.vrms)
    model_grids = [prepare_map_grid(kin.xbin, kin.ybin, result.model)[0] for result in beta_results]

    n_panels = 1 + len(model_grids)
    ncols = 4
    nrows = int(ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    obs_values = obs_grid[np.isfinite(obs_grid)]
    obs_vmin = float(np.nanpercentile(obs_values, 1.0))
    obs_vmax = float(np.nanpercentile(obs_values, 99.0))
    im_obs = axes[0].imshow(
        obs_grid,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        aspect="auto",
        vmin=obs_vmin,
        vmax=obs_vmax,
    )
    axes[0].set_title("Observed Vrms")
    axes[0].set_xlabel("X (arcsec)")
    axes[0].set_ylabel("Y (arcsec)")
    fig.colorbar(im_obs, ax=axes[0], label="km/s", shrink=0.88)

    for ax, result, grid in zip(axes[1:], beta_results, model_grids):
        panel_values = grid[np.isfinite(grid)]
        panel_vmin = float(np.nanpercentile(panel_values, 1.0))
        panel_vmax = float(np.nanpercentile(panel_values, 99.0))
        im_panel = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            aspect="auto",
            vmin=panel_vmin,
            vmax=panel_vmax,
        )
        ax.set_title(make_panel_title(result))
        ax.set_xlabel("X (arcsec)")
        ax.set_ylabel("Y (arcsec)")
        fig.colorbar(im_panel, ax=ax, label="km/s", shrink=0.88)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"NIRSpec JAM constant-beta sweep (individual colorbars) | fixed M_BH={bh_mass:.3e} Msun | fixed M/L={ml_fit:.3f}",
        fontsize=14,
    )
    fig.savefig(
        cfg.output_dir / "vrms_model_grid_vs_beta_individual_colorbars.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_profile_and_chi2_plot(
    cfg: Config,
    kin,
    beta_results: list[BetaSweepResult],
    bh_mass: float,
    ml_fit: float,
) -> None:
    radius = np.hypot(kin.xbin, kin.ybin)
    order = np.argsort(radius)
    radius_sorted = radius[order]
    vrms_sorted = kin.vrms[order]
    vrms_err_sorted = kin.vrms_err[order]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    ax0 = axes[0]
    ax0.errorbar(
        radius_sorted,
        vrms_sorted,
        yerr=vrms_err_sorted,
        fmt="o",
        ms=3,
        lw=0.7,
        color="k",
        alpha=0.45,
        label="Observed Vrms",
    )

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.05, 0.95, len(beta_results)))
    for color, result in zip(colors, beta_results):
        model_sorted = result.model[order]
        label = f"beta={result.requested_beta:.1f}"
        if not np.isclose(result.requested_beta, result.evaluated_beta):
            label += f" (eval {result.evaluated_beta:.2f})"
        ax0.plot(radius_sorted, model_sorted, color=color, lw=1.6, label=label)

    ax0.set_xlabel("Projected radius (arcsec)")
    ax0.set_ylabel("Vrms (km/s)")
    ax0.set_title("Vrms profiles for constant beta grid")
    ax0.legend(fontsize=8, ncol=2, frameon=False)

    ax1 = axes[1]
    requested_beta = np.array([result.requested_beta for result in beta_results], dtype=float)
    reduced_chi2 = np.array([result.reduced_chi2 for result in beta_results], dtype=float)
    best_idx = int(np.nanargmin(reduced_chi2))

    ax1.plot(requested_beta, reduced_chi2, marker="o", color="tab:blue", lw=1.5)
    ax1.scatter(
        requested_beta[best_idx],
        reduced_chi2[best_idx],
        color="tab:red",
        s=60,
        zorder=3,
        label=f"Best in grid: beta={requested_beta[best_idx]:.1f}",
    )
    ax1.set_xlabel("Requested constant beta")
    ax1.set_ylabel(r"Reduced $\chi^2$")
    ax1.set_title("Vrms misfit versus constant beta")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)

    fig.suptitle(
        f"NIRSpec beta sweep | fixed M_BH={bh_mass:.3e} Msun | fixed M/L={ml_fit:.3f}",
        fontsize=14,
    )
    fig.savefig(cfg.output_dir / "vrms_profiles_and_chi2_vs_beta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_table(cfg: Config, beta_results: list[BetaSweepResult]) -> None:
    header = "requested_beta,evaluated_beta,chi2,reduced_chi2,jam_reported_chi2"
    rows = np.array(
        [
            [
                result.requested_beta,
                result.evaluated_beta,
                result.chi2,
                result.reduced_chi2,
                result.jam_reported_chi2,
            ]
            for result in beta_results
        ],
        dtype=float,
    )
    np.savetxt(
        cfg.output_dir / "beta_sweep_summary.csv",
        rows,
        delimiter=",",
        header=header,
        comments="",
    )


def save_models_npz(cfg: Config, kin, beta_results: list[BetaSweepResult], bh_mass: float, ml_fit: float) -> None:
    save_dict = {
        "requested_beta": np.array([result.requested_beta for result in beta_results], dtype=float),
        "evaluated_beta": np.array([result.evaluated_beta for result in beta_results], dtype=float),
        "chi2": np.array([result.chi2 for result in beta_results], dtype=float),
        "reduced_chi2": np.array([result.reduced_chi2 for result in beta_results], dtype=float),
        "jam_reported_chi2": np.array([result.jam_reported_chi2 for result in beta_results], dtype=float),
        "vrms_models": np.array([result.model for result in beta_results], dtype=float),
        "vrms_residuals": np.array([result.residual for result in beta_results], dtype=float),
        "vrms_data": np.asarray(kin.vrms, dtype=float),
        "vrms_err": np.asarray(kin.vrms_err, dtype=float),
        "xbin": np.asarray(kin.xbin, dtype=float),
        "ybin": np.asarray(kin.ybin, dtype=float),
        "goodbins": np.asarray(kin.goodbins, dtype=bool),
        "fixed_bh_mass": np.array(bh_mass, dtype=float),
        "fixed_ml": np.array(ml_fit, dtype=float),
    }
    np.savez(cfg.output_dir / "beta_sweep_models.npz", **save_dict)


def main() -> None:
    base.setup_logging()
    cfg = Config()
    base.ensure_output_dir(cfg.output_dir)

    logging.info("Loading NIRSpec kinematics")
    kin = base.load_kinematics(cfg)

    logging.info("Loading MGE inputs")
    mge = base.load_mge_inputs(cfg)

    bh_mass, ml_fit = load_reference_parameters(cfg)
    logging.info(
        "Using fixed reference parameters from %s: M_BH=%.6e Msun, M/L=%.6f",
        cfg.reference_results_path,
        bh_mass,
        ml_fit,
    )

    beta_results = run_beta_sweep(cfg, kin, mge, bh_mass=bh_mass, ml_fit=ml_fit)

    save_model_map_grid(cfg, kin, beta_results, bh_mass=bh_mass, ml_fit=ml_fit)
    save_model_map_grid_individual_colorbars(cfg, kin, beta_results, bh_mass=bh_mass, ml_fit=ml_fit)
    save_profile_and_chi2_plot(cfg, kin, beta_results, bh_mass=bh_mass, ml_fit=ml_fit)
    save_summary_table(cfg, beta_results)
    save_models_npz(cfg, kin, beta_results, bh_mass=bh_mass, ml_fit=ml_fit)

    best_result = min(beta_results, key=lambda result: result.reduced_chi2)
    logging.info(
        "Best constant-beta value in this grid: requested beta=%.2f, evaluated beta=%.2f, reduced chi2=%.4f",
        best_result.requested_beta,
        best_result.evaluated_beta,
        best_result.reduced_chi2,
    )
    logging.info("Done")


if __name__ == "__main__":
    main()
