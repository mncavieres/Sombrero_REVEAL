"""
Fit JAM models to the Sombrero kinematics with dynesty.

Free parameters:
    1) Black-hole mass M_BH
    2) Constant anisotropy beta
    3) Stellar mass-to-light ratio M/L

Important modeling choice:
    The free M/L rescales the stellar mass model only through

        surf_pot = surf_lum * ml_fit

    while keeping

        ml = 1.0

    inside jam.axi.proj so the BH mass is not additionally rescaled by JAM.

This script also:
    - rotates the IFU kinematics into the photometric frame,
    - builds Vrms and its uncertainty,
    - checkpoints dynesty every ~10 minutes,
    - resumes from checkpoint when possible,
    - saves posterior samples to NPZ,
    - makes a corner plot when enough samples exist,
      otherwise falls back to pairwise scatter plots,
    - makes a best-fit LOSV comparison plot using jam.axi.proj(moment='z').
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import griddata

import jampy as jam
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty.pool import Pool


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    output_dir: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/free_ml_beta_bh"
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

    nlive: int = 100
    nprocs: int = 8
    dlogz_init: float = 1.0
    checkpoint_every_sec: float = 600.0
    checkpoint_filename: str = "checkpoint.save"

    bh_mass_min: float = 1e7
    bh_mass_max: float = 1e10

    beta_min: float = -2.0
    beta_max: float = 0.99

    ml_min: float = 0.1
    ml_max: float = 2.0


@dataclass
class Kinematics:
    table: Table
    xbin: np.ndarray
    ybin: np.ndarray
    vlos_obs: np.ndarray
    vlos_err: np.ndarray
    vlos_rf: np.ndarray
    sigma: np.ndarray
    sigma_err: np.ndarray
    vrms: np.ndarray
    vrms_err: np.ndarray
    goodbins: np.ndarray


@dataclass
class MGEInputs:
    surf_lum: np.ndarray
    sigma_lum: np.ndarray
    q_obs_lum: np.ndarray


# -----------------------------------------------------------------------------
# Constants for MJy/sr -> Lsun/pc^2 conversion
# -----------------------------------------------------------------------------

ARCSEC2_PER_SR = (180.0 / np.pi * 3600.0) ** 2
MAG_ARCSEC2_TO_LSUN_PC2 = 21.572
M_SUN_AB_F200W = 4.93


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def checkpoint_path(cfg: Config) -> Path:
    return cfg.output_dir / cfg.checkpoint_filename


def rotate_coordinates(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return x_rot, y_rot


def safe_symmetric_limit(values: np.ndarray, fallback: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if not np.any(good):
        return fallback
    vmax = np.nanmax(np.abs(values[good]))
    if not np.isfinite(vmax) or vmax <= 0:
        return fallback
    return float(vmax)


def interpolate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_size: int = 120,
    preferred_method: str = "cubic",
):
    xi = np.linspace(np.nanmin(x), np.nanmax(x), grid_size)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), grid_size)
    xx, yy = np.meshgrid(xi, yi)

    methods = [preferred_method, "linear", "nearest"]
    last = None
    for method in methods:
        try:
            zz = griddata((x, y), values, (xx, yy), method=method, fill_value=np.nan)
            if np.any(np.isfinite(zz)):
                return xx, yy, zz
            last = zz
        except Exception:
            continue

    return xx, yy, last


def save_interpolated_map(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    _, _, zz = interpolate_to_grid(x, y, values)
    extent = (np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y))

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        zz,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")
    ax.set_title(title)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# MGE helpers
# -----------------------------------------------------------------------------

def gaussian_peak_from_total_counts(total_counts: np.ndarray, sigma_pix: np.ndarray, q_obs: np.ndarray) -> np.ndarray:
    return total_counts / (2.0 * np.pi * sigma_pix**2 * q_obs)


def mjysr_to_lsun_pc2(mu_mjysr: np.ndarray, m_sun_ab: float = M_SUN_AB_F200W) -> np.ndarray:
    jy_arcsec2 = mu_mjysr * 1e6 / ARCSEC2_PER_SR
    mu_ab = -2.5 * np.log10(jy_arcsec2 / 3631.0)
    return 10.0 ** (-0.4 * (mu_ab - m_sun_ab - MAG_ARCSEC2_TO_LSUN_PC2))


def make_jam_mge_from_table(
    mge_tab: Table,
    *,
    total_col: str = "total_counts",
    sigma_pix_col: str = "sigma_pix",
    q_col: str = "q_obs",
    pixel_scale_arcsec: float = 0.031,
    m_sun_ab: float = M_SUN_AB_F200W,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_pix = np.asarray(mge_tab[sigma_pix_col], dtype=float)
    q_obs = np.asarray(mge_tab[q_col], dtype=float)
    total_counts = np.asarray(mge_tab[total_col], dtype=float)

    peak_mjysr = gaussian_peak_from_total_counts(total_counts, sigma_pix, q_obs)
    surf_lum = mjysr_to_lsun_pc2(peak_mjysr, m_sun_ab=m_sun_ab)
    sigma_arcsec = sigma_pix * pixel_scale_arcsec

    return surf_lum, sigma_arcsec, q_obs


def load_mge_inputs(cfg: Config) -> MGEInputs:
    """
    Preserve the original script's behavior:
      - compute surf_lum from mge_solution.csv
      - replace sigma_lum and q_obs_lum with the values from
        mge_luminosity_table.csv
    """
    mge_solution = Table.read(cfg.mge_solution_path)
    surf_lum, _, _ = make_jam_mge_from_table(
        mge_solution,
        pixel_scale_arcsec=cfg.pixel_scale_arcsec,
        m_sun_ab=M_SUN_AB_F200W,
    )

    mge_table = Table.read(cfg.mge_luminosity_path)
    sigma_lum = np.asarray(mge_table["sigma_arcsec"], dtype=float)
    q_obs_lum = np.asarray(mge_table["q_obs"], dtype=float)

    if len(surf_lum) != len(sigma_lum):
        raise ValueError(
            "Mismatch between number of MGE Gaussians in mge_solution.csv "
            "and mge_luminosity_table.csv"
        )

    return MGEInputs(
        surf_lum=np.asarray(surf_lum, dtype=float),
        sigma_lum=sigma_lum,
        q_obs_lum=q_obs_lum,
    )


# -----------------------------------------------------------------------------
# Kinematics
# -----------------------------------------------------------------------------

def compute_rest_frame_vlos(losv: np.ndarray, z: float) -> np.ndarray:
    c_kms = 299792.458
    return losv - z * c_kms


def compute_vrms_and_error(
    vlos_rf: np.ndarray,
    vlos_err: np.ndarray,
    sigma: np.ndarray,
    sigma_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vrms = np.sqrt(vlos_rf**2 + sigma**2)
    numer = np.sqrt((vlos_rf * vlos_err) ** 2 + (sigma * sigma_err) ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        vrms_err = np.divide(
            numer,
            vrms,
            out=np.full_like(vrms, np.nan, dtype=float),
            where=vrms > 0,
        )

    return vrms, vrms_err


def load_kinematics(cfg: Config) -> Kinematics:
    kin_table = Table.read(cfg.kin_path, format="csv", delimiter=";")

    x = np.asarray(kin_table["X"], dtype=float)
    y = np.asarray(kin_table["Y"], dtype=float)
    x_rot, y_rot = rotate_coordinates(x, y, cfg.rotation_deg)

    kin_table["X_rot"] = x_rot
    kin_table["Y_rot"] = y_rot

    vlos_obs = np.asarray(kin_table["LOSV"], dtype=float)
    vlos_err = np.asarray(kin_table["LOSV_err"], dtype=float)
    sigma = np.asarray(kin_table["sigma"], dtype=float)
    sigma_err = np.asarray(kin_table["sigma_err"], dtype=float)

    vlos_rf = compute_rest_frame_vlos(vlos_obs, cfg.redshift)
    vrms, vrms_err = compute_vrms_and_error(vlos_rf, vlos_err, sigma, sigma_err)

    goodbins = (
        np.isfinite(x_rot)
        & np.isfinite(y_rot)
        & np.isfinite(vlos_rf)
        & np.isfinite(vlos_err)
        & np.isfinite(sigma)
        & np.isfinite(sigma_err)
        & np.isfinite(vrms)
        & np.isfinite(vrms_err)
        & (vlos_err > 0)
        & (vrms_err > 0)
    )

    return Kinematics(
        table=kin_table,
        xbin=x_rot,
        ybin=y_rot,
        vlos_obs=vlos_obs,
        vlos_err=vlos_err,
        vlos_rf=vlos_rf,
        sigma=sigma,
        sigma_err=sigma_err,
        vrms=vrms,
        vrms_err=vrms_err,
        goodbins=goodbins,
    )


def save_kinematic_maps(cfg: Config, kin: Kinematics) -> None:
    save_interpolated_map(
        cfg.output_dir / "kinematic_maps_LOSV.png",
        kin.xbin,
        kin.ybin,
        kin.vlos_obs,
        title="Observed LOSV",
        cbar_label="km/s",
    )
    save_interpolated_map(
        cfg.output_dir / "kinematic_maps_sigma.png",
        kin.xbin,
        kin.ybin,
        kin.sigma,
        title="Observed sigma",
        cbar_label="km/s",
    )
    save_interpolated_map(
        cfg.output_dir / "kinematic_maps_vlos_compensated.png",
        kin.xbin,
        kin.ybin,
        kin.vlos_rf,
        title="Observed LOSV (rest frame)",
        cbar_label="km/s",
    )
    save_interpolated_map(
        cfg.output_dir / "kinematic_maps_vrms.png",
        kin.xbin,
        kin.ybin,
        kin.vrms,
        title="Observed Vrms",
        cbar_label="km/s",
    )


# -----------------------------------------------------------------------------
# Pickle-safe dynesty callables for multiprocessing
# -----------------------------------------------------------------------------

class JamVrmsLogLikelihood:
    """
    Pickle-safe top-level likelihood callable for dynesty multiprocessing.
    """

    def __init__(self, cfg: Config, kin: Kinematics, mge: MGEInputs):
        self.inclination_deg = cfg.inclination_deg
        self.distance_mpc = cfg.distance_mpc
        self.sigmapsf_arcsec = cfg.sigmapsf_arcsec
        self.pixsize_arcsec = cfg.pixsize_arcsec
        self.rotation_deg = cfg.rotation_deg

        self.surf_lum = np.asarray(mge.surf_lum, dtype=float)
        self.sigma_lum = np.asarray(mge.sigma_lum, dtype=float)
        self.qobs_lum = np.asarray(mge.q_obs_lum, dtype=float)

        self.xbin = np.asarray(kin.xbin, dtype=float)
        self.ybin = np.asarray(kin.ybin, dtype=float)
        self.vrms = np.asarray(kin.vrms, dtype=float)
        self.vrms_err = np.asarray(kin.vrms_err, dtype=float)
        self.goodbins = np.asarray(kin.goodbins, dtype=bool)

    def __call__(self, theta: np.ndarray) -> float:
        bh_mass, beta_scalar, ml_fit = theta
        beta = np.full(self.surf_lum.shape, beta_scalar, dtype=float)

        try:
            out = jam.axi.proj(
                surf_lum=self.surf_lum,
                sigma_lum=self.sigma_lum,
                qobs_lum=self.qobs_lum,
                surf_pot=self.surf_lum * ml_fit,
                sigma_pot=self.sigma_lum,
                qobs_pot=self.qobs_lum,
                inc=self.inclination_deg,
                mbh=bh_mass,
                distance=self.distance_mpc,
                xbin=self.xbin,
                ybin=self.ybin,
                align="cyl",
                analytic_los=False,
                beta=beta,
                data=self.vrms,
                errors=self.vrms_err,
                flux_obs=None,
                gamma=None,
                goodbins=self.goodbins,
                interp=True,
                kappa=None,
                sigmapsf=self.sigmapsf_arcsec,
                normpsf=np.array([1.0]),
                pixsize=self.pixsize_arcsec,
                pixang=self.rotation_deg,
                logistic=False,
                ml=1.0,
                moment="zz",
                epsrel=1e-2,
                plot=False,
                quiet=True,
            )
        except Exception as exc:
            logging.debug("JAM failed at theta=%s with error: %s", theta, exc)
            return -np.inf

        chi2 = getattr(out, "chi2", np.inf)
        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2


class UniformPriorTransform:
    """
    Pickle-safe top-level prior transform callable for dynesty multiprocessing.
    """

    def __init__(self, cfg: Config):
        self.bh_mass_min = cfg.bh_mass_min
        self.bh_mass_max = cfg.bh_mass_max
        self.beta_min = cfg.beta_min
        self.beta_max = cfg.beta_max
        self.ml_min = cfg.ml_min
        self.ml_max = cfg.ml_max

    def __call__(self, utheta: np.ndarray) -> np.ndarray:
        bh_mass = self.bh_mass_min + utheta[0] * (self.bh_mass_max - self.bh_mass_min)
        beta = self.beta_min + utheta[1] * (self.beta_max - self.beta_min)
        ml_fit = self.ml_min + utheta[2] * (self.ml_max - self.ml_min)
        return np.array([bh_mass, beta, ml_fit], dtype=float)


# -----------------------------------------------------------------------------
# Posterior / diagnostics
# -----------------------------------------------------------------------------

def get_best_fit_parameters(results) -> np.ndarray:
    idx = np.argmax(np.asarray(results.logl))
    return np.asarray(results.samples[idx], dtype=float)


def get_samples_for_plotting(results) -> np.ndarray:
    if hasattr(results, "equal_samples") and callable(results.equal_samples):
        try:
            eq = np.asarray(results.equal_samples(), dtype=float)
            if eq.ndim == 2 and eq.shape[0] > 0:
                return eq
        except Exception:
            pass
    return np.asarray(results.samples, dtype=float)


def save_posterior_plot(cfg: Config, results) -> None:
    samples_plot = get_samples_for_plotting(results)
    outpath = cfg.output_dir / "posterior_samples.png"

    enough_for_corner = (
        samples_plot.ndim == 2
        and samples_plot.shape[1] == 3
        and samples_plot.shape[0] >= 20
    )

    if enough_for_corner:
        try:
            fig, _ = dyplot.cornerplot(
                results,
                show_titles=True,
                labels=[
                    r"$M_{\rm BH}\,[M_\odot]$",
                    r"$\beta$",
                    r"$M/L$",
                ],
                title_kwargs={"x": 0.65},
            )
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return
        except Exception as exc:
            logging.warning("Corner plot failed; using scatter fallback: %s", exc)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    if samples_plot.ndim == 2 and samples_plot.shape[0] > 0 and samples_plot.shape[1] >= 3:
        axes[0].scatter(samples_plot[:, 0], samples_plot[:, 1], s=8, alpha=0.6)
        axes[0].set_xscale("log")
        axes[0].set_xlabel(r"$M_{\rm BH}\,[M_\odot]$")
        axes[0].set_ylabel(r"$\beta$")
        axes[0].set_title(r"$M_{\rm BH}$ vs $\beta$")

        axes[1].scatter(samples_plot[:, 0], samples_plot[:, 2], s=8, alpha=0.6)
        axes[1].set_xscale("log")
        axes[1].set_xlabel(r"$M_{\rm BH}\,[M_\odot]$")
        axes[1].set_ylabel(r"$M/L$")
        axes[1].set_title(r"$M_{\rm BH}$ vs $M/L$")

        axes[2].scatter(samples_plot[:, 1], samples_plot[:, 2], s=8, alpha=0.6)
        axes[2].set_xlabel(r"$\beta$")
        axes[2].set_ylabel(r"$M/L$")
        axes[2].set_title(r"$\beta$ vs $M/L$")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No samples available", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_bestfit_losv_model(
    cfg: Config,
    kin: Kinematics,
    mge: MGEInputs,
    best_params: np.ndarray,
):
    bh_mass, beta_scalar, ml_fit = best_params
    beta = np.full(mge.surf_lum.shape, beta_scalar, dtype=float)

    out = jam.axi.proj(
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
        analytic_los=False,
        beta=beta,
        data=kin.vlos_rf,
        errors=kin.vlos_err,
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
        moment="z",
        epsrel=1e-2,
        plot=False,
        quiet=True,
    )

    model = np.asarray(out.model, dtype=float)
    resid = kin.vlos_rf - model
    return out, model, resid


def save_losv_bestfit_plot(
    cfg: Config,
    kin: Kinematics,
    mge: MGEInputs,
    best_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    out, model, resid = compute_bestfit_losv_model(cfg, kin, mge, best_params)

    _, _, obs_grid = interpolate_to_grid(kin.xbin, kin.ybin, kin.vlos_rf)
    _, _, mod_grid = interpolate_to_grid(kin.xbin, kin.ybin, model)
    _, _, res_grid = interpolate_to_grid(kin.xbin, kin.ybin, resid)

    vmax = safe_symmetric_limit(np.r_[kin.vlos_rf[kin.goodbins], model[kin.goodbins]], fallback=1.0)
    rmax = safe_symmetric_limit(resid[kin.goodbins], fallback=1.0)

    extent = (np.nanmin(kin.xbin), np.nanmax(kin.xbin), np.nanmin(kin.ybin), np.nanmax(kin.ybin))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        obs_grid,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[0].set_title("Observed LOSV (rest frame)")
    axes[0].set_xlabel("X (arcsec)")
    axes[0].set_ylabel("Y (arcsec)")
    fig.colorbar(im0, ax=axes[0], label="km/s")

    im1 = axes[1].imshow(
        mod_grid,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[1].set_title("Best-fit JAM LOSV model")
    axes[1].set_xlabel("X (arcsec)")
    axes[1].set_ylabel("Y (arcsec)")
    fig.colorbar(im1, ax=axes[1], label="km/s")

    im2 = axes[2].imshow(
        res_grid,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-rmax,
        vmax=rmax,
        aspect="auto",
    )
    axes[2].set_title("Residual (data - model)")
    axes[2].set_xlabel("X (arcsec)")
    axes[2].set_ylabel("Y (arcsec)")
    fig.colorbar(im2, ax=axes[2], label="km/s")

    fig.savefig(cfg.output_dir / "bestfit_losv_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    best_kappa = getattr(out, "kappa", None)
    if best_kappa is not None:
        best_kappa = np.asarray(best_kappa)

    return model, resid, best_kappa


# -----------------------------------------------------------------------------
# dynesty run
# -----------------------------------------------------------------------------

def run_sampler(cfg: Config, log_likelihood_fn, prior_transform_fn):
    cp_file = checkpoint_path(cfg)

    try:
        with Pool(cfg.nprocs, log_likelihood_fn, prior_transform_fn) as pool:
            sampler = None
            restored = False

            if cp_file.exists():
                try:
                    logging.info("Restoring sampler from %s", cp_file)
                    sampler = DynamicNestedSampler.restore(str(cp_file), pool=pool)
                    restored = True
                except Exception as exc:
                    logging.warning(
                        "Could not restore checkpoint %s (%s). Starting a fresh run instead.",
                        cp_file,
                        exc,
                    )
                    sampler = None

            if sampler is None:
                logging.info("Starting a new parallel sampler")
                sampler = DynamicNestedSampler(
                    pool.loglike,
                    pool.prior_transform,
                    ndim=3,
                    nlive=cfg.nlive,
                    pool=pool,
                    queue_size=cfg.nprocs,
                    use_pool={"prior_transform": False},
                )

            if restored:
                sampler.run_nested(
                    dlogz_init=cfg.dlogz_init,
                    checkpoint_file=str(cp_file),
                    checkpoint_every=cfg.checkpoint_every_sec,
                    resume=True,
                )
            else:
                sampler.run_nested(
                    dlogz_init=cfg.dlogz_init,
                    checkpoint_file=str(cp_file),
                    checkpoint_every=cfg.checkpoint_every_sec,
                )

            return sampler.results

    except Exception as exc:
        logging.warning("Parallel pool failed (%s). Falling back to serial run.", exc)

        sampler = None
        restored = False

        if cp_file.exists():
            try:
                logging.info("Restoring serial sampler from %s", cp_file)
                sampler = DynamicNestedSampler.restore(str(cp_file))
                restored = True
            except Exception as restore_exc:
                logging.warning(
                    "Could not restore checkpoint %s in serial mode (%s). Starting fresh.",
                    cp_file,
                    restore_exc,
                )
                sampler = None

        if sampler is None:
            logging.info("Starting a new serial sampler")
            sampler = DynamicNestedSampler(
                log_likelihood_fn,
                prior_transform_fn,
                ndim=3,
                nlive=cfg.nlive,
            )

        if restored:
            sampler.run_nested(
                dlogz_init=cfg.dlogz_init,
                checkpoint_file=str(cp_file),
                checkpoint_every=cfg.checkpoint_every_sec,
                resume=True,
            )
        else:
            sampler.run_nested(
                dlogz_init=cfg.dlogz_init,
                checkpoint_file=str(cp_file),
                checkpoint_every=cfg.checkpoint_every_sec,
            )

        return sampler.results


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_results(cfg: Config, results, kin: Kinematics, mge: MGEInputs) -> None:
    best_params = get_best_fit_parameters(results)

    losv_model, losv_resid, best_kappa = save_losv_bestfit_plot(cfg, kin, mge, best_params)
    save_posterior_plot(cfg, results)

    save_dict = {
        "samples": np.asarray(results.samples),
        "logl": np.asarray(results.logl),
        "logwt": np.asarray(results.logwt),
        "logz": np.asarray(results.logz),
        "logzerr": np.asarray(results.logzerr),
        "best_params": np.asarray(best_params),
        "best_bh_mass": float(best_params[0]),
        "best_beta": float(best_params[1]),
        "best_ml": float(best_params[2]),
        "vlos_data_rf": np.asarray(kin.vlos_rf),
        "vlos_model_bestfit": np.asarray(losv_model),
        "vlos_residual_bestfit": np.asarray(losv_resid),
        "xbin": np.asarray(kin.xbin),
        "ybin": np.asarray(kin.ybin),
        "goodbins": np.asarray(kin.goodbins),
    }

    if best_kappa is not None:
        save_dict["bestfit_kappa_losv"] = np.asarray(best_kappa)

    if hasattr(results, "equal_samples") and callable(results.equal_samples):
        try:
            save_dict["equal_samples"] = np.asarray(results.equal_samples())
        except Exception:
            pass

    np.savez(cfg.output_dir / "nested_bh_beta_ml_results.npz", **save_dict)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    cfg = Config()

    ensure_output_dir(cfg.output_dir)

    logging.info("Loading and preparing kinematics")
    kin = load_kinematics(cfg)
    save_kinematic_maps(cfg, kin)

    logging.info("Loading MGE inputs")
    mge = load_mge_inputs(cfg)

    logging.info("Building likelihood and priors")
    log_likelihood_fn = JamVrmsLogLikelihood(cfg, kin, mge)
    prior_transform_fn = UniformPriorTransform(cfg)

    logging.info(
        "Running dynesty with checkpointing every %.0f seconds (~%.1f min)",
        cfg.checkpoint_every_sec,
        cfg.checkpoint_every_sec / 60.0,
    )
    results = run_sampler(cfg, log_likelihood_fn, prior_transform_fn)

    logging.info("Saving samples, posterior plot, and LOSV best-fit comparison")
    save_results(cfg, results, kin, mge)

    logging.info("Done")


if __name__ == "__main__":
    main()