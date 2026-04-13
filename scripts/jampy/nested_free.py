"""
Fit JAM models to the Sombrero kinematics with dynesty.

Free parameters:
    1) Black-hole mass M_BH
    2) Anisotropy beta prescription:
         - "constant": one beta for all MGE Gaussians
         - "logistic": beta = [r_a, beta_0, beta_inf, alpha]
         - "free": one beta per MGE Gaussian
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
    - makes a best-fit LOSV comparison plot using jam.axi.proj(moment='z'),
    - makes a best-fit Vrms comparison plot using jam.axi.proj(moment='zz').
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
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/free_beta_2"
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

    nlive: int = 200
    nprocs: int = 8
    dlogz_init: float = 0.0001
    checkpoint_every_sec: float = 30.0
    checkpoint_filename: str = "checkpoint.save"

    bh_mass_min: float = 1e6
    bh_mass_max: float = 1e10

    beta_min: float = -15.0
    beta_max: float = 0.99

    ml_min: float = 0.1
    ml_max: float = 2.0

    # Beta prescription:
    #   "constant" -> one beta shared by all MGE Gaussians
    #   "logistic" -> beta = [r_a, beta_0, beta_inf, alpha]
    #   "free"     -> one beta per MGE Gaussian
    beta_prescription: str = "free"

    # Logistic beta priors
    # r_a is in arcsec, alpha > 0
    beta_ra_min: float = 0.001
    beta_ra_max: float = 300.0
    beta_alpha_min: float = 0.1
    beta_alpha_max: float = 10.0


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


def safe_positive_limit(values: np.ndarray, fallback: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if not np.any(good):
        return fallback
    vmax = np.nanmax(values[good])
    if not np.isfinite(vmax) or vmax <= 0:
        return fallback
    return float(vmax)


def get_beta_parameter_count(cfg: Config, n_mge: int) -> int:
    if cfg.beta_prescription == "constant":
        return 1
    if cfg.beta_prescription == "logistic":
        return 4
    if cfg.beta_prescription == "free":
        return n_mge
    raise ValueError(
        f"Unknown beta_prescription='{cfg.beta_prescription}'. "
        "Use 'constant', 'logistic', or 'free'."
    )


def get_ndim(cfg: Config, n_mge: int) -> int:
    return 1 + get_beta_parameter_count(cfg, n_mge) + 1


def unpack_theta(theta: np.ndarray, cfg: Config, n_mge: int):
    """
    theta layout:
        constant: [bh_mass, beta, ml]
        logistic: [bh_mass, r_a, beta_0, beta_inf, alpha, ml]
        free:     [bh_mass, beta_1, ..., beta_n, ml]
    """
    theta = np.asarray(theta, dtype=float)

    bh_mass = float(theta[0])
    beta_count = get_beta_parameter_count(cfg, n_mge)
    beta_params = theta[1:1 + beta_count]
    ml_fit = float(theta[1 + beta_count])

    expected_size = 1 + beta_count + 1
    if theta.size != expected_size:
        raise ValueError(f"Expected theta size {expected_size}, got {theta.size}")

    if cfg.beta_prescription == "constant":
        beta_scalar = float(beta_params[0])
        beta = np.full(n_mge, beta_scalar, dtype=float)
        logistic = False

    elif cfg.beta_prescription == "logistic":
        beta = np.asarray(beta_params, dtype=float)  # [r_a, beta_0, beta_inf, alpha]
        logistic = True

    elif cfg.beta_prescription == "free":
        beta = np.asarray(beta_params, dtype=float)
        logistic = False

    else:
        raise ValueError(
            f"Unknown beta_prescription='{cfg.beta_prescription}'. "
            "Use 'constant', 'logistic', or 'free'."
        )

    return bh_mass, beta, ml_fit, logistic


def get_parameter_labels(cfg: Config, n_mge: int) -> list[str]:
    labels = [r"$M_{\rm BH}\,[M_\odot]$"]

    if cfg.beta_prescription == "constant":
        labels += [r"$\beta$"]

    elif cfg.beta_prescription == "logistic":
        labels += [
            r"$r_a$",
            r"$\beta_0$",
            r"$\beta_\infty$",
            r"$\alpha$",
        ]

    elif cfg.beta_prescription == "free":
        labels += [rf"$\beta_{{{i+1}}}$" for i in range(n_mge)]

    else:
        raise ValueError(
            f"Unknown beta_prescription='{cfg.beta_prescription}'. "
            "Use 'constant', 'logistic', or 'free'."
        )

    labels += [r"$M/L$"]
    return labels


def summarize_best_params(cfg: Config, best_params: np.ndarray, n_mge: int) -> dict:
    bh_mass, beta, ml_fit, logistic = unpack_theta(best_params, cfg, n_mge)

    summary = {
        "best_bh_mass": float(bh_mass),
        "best_ml": float(ml_fit),
        "best_logistic_flag": bool(logistic),
    }

    if cfg.beta_prescription == "constant":
        summary["best_beta"] = float(beta[0])

    elif cfg.beta_prescription == "logistic":
        summary["best_beta_logistic"] = np.asarray(beta)
        summary["best_beta_ra"] = float(beta[0])
        summary["best_beta_0"] = float(beta[1])
        summary["best_beta_inf"] = float(beta[2])
        summary["best_beta_alpha"] = float(beta[3])

    elif cfg.beta_prescription == "free":
        summary["best_beta_array"] = np.asarray(beta)

    return summary


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

    fig, ax = plt.subplots(figsize=(10, 8))
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
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
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
        self.cfg = cfg

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

        self.n_mge = len(self.surf_lum)

    def __call__(self, theta: np.ndarray) -> float:
        try:
            bh_mass, beta, ml_fit, logistic = unpack_theta(theta, self.cfg, self.n_mge)
        except Exception as exc:
            logging.debug("Theta unpack failed for theta=%s: %s", theta, exc)
            return -np.inf

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
                analytic_los=True,
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
                logistic=logistic,
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

    def __init__(self, cfg: Config, n_mge: int):
        self.cfg = cfg
        self.n_mge = n_mge

    def __call__(self, utheta: np.ndarray) -> np.ndarray:
        utheta = np.asarray(utheta, dtype=float)

        out = []

        bh_mass = self.cfg.bh_mass_min + utheta[0] * (self.cfg.bh_mass_max - self.cfg.bh_mass_min)
        out.append(bh_mass)

        idx = 1

        if self.cfg.beta_prescription == "constant":
            beta = self.cfg.beta_min + utheta[idx] * (self.cfg.beta_max - self.cfg.beta_min)
            out.append(beta)
            idx += 1

        elif self.cfg.beta_prescription == "logistic":
            r_a = self.cfg.beta_ra_min + utheta[idx] * (self.cfg.beta_ra_max - self.cfg.beta_ra_min)
            idx += 1

            beta_0 = self.cfg.beta_min + utheta[idx] * (self.cfg.beta_max - self.cfg.beta_min)
            idx += 1

            beta_inf = self.cfg.beta_min + utheta[idx] * (self.cfg.beta_max - self.cfg.beta_min)
            idx += 1

            alpha = self.cfg.beta_alpha_min + utheta[idx] * (self.cfg.beta_alpha_max - self.cfg.beta_alpha_min)
            idx += 1

            out.extend([r_a, beta_0, beta_inf, alpha])

        elif self.cfg.beta_prescription == "free":
            for _ in range(self.n_mge):
                beta_i = self.cfg.beta_min + utheta[idx] * (self.cfg.beta_max - self.cfg.beta_min)
                out.append(beta_i)
                idx += 1

        else:
            raise ValueError(
                f"Unknown beta_prescription='{self.cfg.beta_prescription}'. "
                "Use 'constant', 'logistic', or 'free'."
            )

        ml_fit = self.cfg.ml_min + utheta[idx] * (self.cfg.ml_max - self.cfg.ml_min)
        out.append(ml_fit)
        idx += 1

        if idx != len(utheta):
            raise ValueError(f"Used {idx} prior parameters but got {len(utheta)}")

        return np.array(out, dtype=float)


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


def save_posterior_plot(cfg: Config, results, n_mge: int) -> None:
    samples_plot = get_samples_for_plotting(results)
    outpath = cfg.output_dir / "posterior_samples.png"
    labels = get_parameter_labels(cfg, n_mge)
    ndim = len(labels)

    allow_corner = ndim <= 8

    enough_for_corner = (
        allow_corner
        and samples_plot.ndim == 2
        and samples_plot.shape[1] == ndim
        and samples_plot.shape[0] >= 20
    )

    if enough_for_corner:
        try:
            fig, _ = dyplot.cornerplot(
                results,
                show_titles=True,
                labels=labels,
                title_kwargs={"x": 0.65},
            )
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return
        except Exception as exc:
            logging.warning("Corner plot failed; using scatter fallback: %s", exc)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    if samples_plot.ndim == 2 and samples_plot.shape[0] > 0 and samples_plot.shape[1] >= 3:
        pairs = [(0, 1), (0, 2), (1, 2)]
        for ax, (i, j) in zip(axes, pairs):
            ax.scatter(samples_plot[:, i], samples_plot[:, j], s=8, alpha=0.6)
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            if i == 0:
                ax.set_xscale("log")
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
    bh_mass, beta, ml_fit, logistic = unpack_theta(best_params, cfg, len(mge.surf_lum))

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
        logistic=logistic,
        ml=1.0,
        moment="z",
        epsrel=1e-2,
        plot=False,
        quiet=False,
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


def compute_bestfit_vrms_model(
    cfg: Config,
    kin: Kinematics,
    mge: MGEInputs,
    best_params: np.ndarray,
):
    bh_mass, beta, ml_fit, logistic = unpack_theta(best_params, cfg, len(mge.surf_lum))

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
        logistic=logistic,
        ml=1.0,
        moment="zz",
        epsrel=1e-2,
        plot=False,
        quiet=True,
    )

    model = np.asarray(out.model, dtype=float)
    resid = kin.vrms - model
    return out, model, resid


def save_vrms_bestfit_plot(
    cfg: Config,
    kin: Kinematics,
    mge: MGEInputs,
    best_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out, model, resid = compute_bestfit_vrms_model(cfg, kin, mge, best_params)

    _, _, obs_grid = interpolate_to_grid(kin.xbin, kin.ybin, kin.vrms)
    _, _, mod_grid = interpolate_to_grid(kin.xbin, kin.ybin, model)
    _, _, res_grid = interpolate_to_grid(kin.xbin, kin.ybin, resid)

    vmax = safe_positive_limit(np.r_[kin.vrms[kin.goodbins], model[kin.goodbins]], fallback=1.0)
    rmax = safe_symmetric_limit(resid[kin.goodbins], fallback=1.0)

    extent = (np.nanmin(kin.xbin), np.nanmax(kin.xbin), np.nanmin(kin.ybin), np.nanmax(kin.ybin))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        obs_grid,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=200.0,
        vmax=vmax,
        aspect="auto",
    )
    axes[0].set_title("Observed Vrms")
    axes[0].set_xlabel("X (arcsec)")
    axes[0].set_ylabel("Y (arcsec)")
    fig.colorbar(im0, ax=axes[0], label="km/s")

    im1 = axes[1].imshow(
        mod_grid,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=200.0,
        vmax=vmax,
        aspect="auto",
    )
    axes[1].set_title("Best-fit JAM Vrms model")
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

    fig.savefig(cfg.output_dir / "bestfit_vrms_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return model, resid


# -----------------------------------------------------------------------------
# dynesty run
# -----------------------------------------------------------------------------

def run_sampler(cfg: Config, log_likelihood_fn, prior_transform_fn, n_mge: int):
    cp_file = checkpoint_path(cfg)
    ndim = get_ndim(cfg, n_mge)

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
                logging.info("Starting a new parallel sampler with ndim=%d", ndim)
                sampler = DynamicNestedSampler(
                    pool.loglike,
                    pool.prior_transform,
                    ndim=ndim,
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
            logging.info("Starting a new serial sampler with ndim=%d", ndim)
            sampler = DynamicNestedSampler(
                log_likelihood_fn,
                prior_transform_fn,
                ndim=ndim,
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
    n_mge = len(mge.surf_lum)
    best_params = get_best_fit_parameters(results)

    losv_model, losv_resid, best_kappa = save_losv_bestfit_plot(cfg, kin, mge, best_params)
    vrms_model, vrms_resid = save_vrms_bestfit_plot(cfg, kin, mge, best_params)

    save_posterior_plot(cfg, results, n_mge)

    save_dict = {
        "samples": np.asarray(results.samples),
        "logl": np.asarray(results.logl),
        "logwt": np.asarray(results.logwt),
        "logz": np.asarray(results.logz),
        "logzerr": np.asarray(results.logzerr),
        "best_params": np.asarray(best_params),
        "beta_prescription": np.array(cfg.beta_prescription),
        "parameter_labels": np.array(get_parameter_labels(cfg, n_mge), dtype=object),
        "vlos_data_rf": np.asarray(kin.vlos_rf),
        "vlos_model_bestfit": np.asarray(losv_model),
        "vlos_residual_bestfit": np.asarray(losv_resid),
        "vrms_data": np.asarray(kin.vrms),
        "vrms_model_bestfit": np.asarray(vrms_model),
        "vrms_residual_bestfit": np.asarray(vrms_resid),
        "xbin": np.asarray(kin.xbin),
        "ybin": np.asarray(kin.ybin),
        "goodbins": np.asarray(kin.goodbins),
    }

    save_dict.update(summarize_best_params(cfg, best_params, n_mge))

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

    n_mge = len(mge.surf_lum)
    ndim = get_ndim(cfg, n_mge)

    logging.info(
        "Building likelihood and priors with beta_prescription='%s' and ndim=%d",
        cfg.beta_prescription,
        ndim,
    )
    log_likelihood_fn = JamVrmsLogLikelihood(cfg, kin, mge)
    prior_transform_fn = UniformPriorTransform(cfg, n_mge)

    logging.info(
        "Running dynesty with checkpointing every %.0f seconds (~%.1f min)",
        cfg.checkpoint_every_sec,
        cfg.checkpoint_every_sec / 60.0,
    )
    results = run_sampler(cfg, log_likelihood_fn, prior_transform_fn, n_mge)

    logging.info("Saving samples, posterior plot, and LOSV/Vrms best-fit comparisons")
    save_results(cfg, results, kin, mge)

    logging.info("Done")


if __name__ == "__main__":
    main()