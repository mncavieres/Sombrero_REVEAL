"""
Refactored JAM + dynesty fit for the Sombrero galaxy.

Main changes:
- central configuration in one place
- split workflow into small functions
- remove repeated plotting / I/O patterns
- make MGE loading explicit
- add basic error handling in the likelihood
- use pathlib instead of raw strings
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    output_dir: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/constant_beta_bh"
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
    ml: float = 0.86

    sigmapsf_arcsec: float = 0.1
    pixsize_arcsec: float = 0.1
    pixel_scale_arcsec: float = 0.031

    nlive: int = 100
    checkpoint_every: int = 100
    dlogz_init: float = 1.0

    bh_mass_min: float = 1e7
    bh_mass_max: float = 1e10
    beta_min: float = -2
    beta_max: float = 1


@dataclass
class Kinematics:
    table: Table
    xbin: np.ndarray
    ybin: np.ndarray
    vlos_rf: np.ndarray
    sigma: np.ndarray
    vrms: np.ndarray
    vrms_err: np.ndarray


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
# Utilities
# -----------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rotate_coordinates(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return x_rot, y_rot


def plot_map(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str = "RdBu_r",
    cbar_label: str = "Value",
    grid_size: int = 100,
    interpolation_method: str = "cubic",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Interpolate scattered data onto a regular grid and display as an image.
    """
    xi = np.linspace(np.min(x), np.max(x), grid_size)
    yi = np.linspace(np.min(y), np.max(y), grid_size)
    xx, yy = np.meshgrid(xi, yi)

    zz = griddata(
        (x, y),
        values,
        (xx, yy),
        method=interpolation_method,
        fill_value=np.nan,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        zz,
        extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")
    return fig, ax


def save_map(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str = "RdBu_r",
    cbar_label: str = "Value",
) -> None:
    fig, _ = plot_map(x, y, values, cmap=cmap, cbar_label=cbar_label)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# MGE helpers
# -----------------------------------------------------------------------------

def gaussian_peak_from_total_counts(total_counts: np.ndarray, sigma_pix: np.ndarray, q_obs: np.ndarray) -> np.ndarray:
    """
    Convert integrated 2D Gaussian counts into peak surface brightness.
    """
    return total_counts / (2.0 * np.pi * sigma_pix**2 * q_obs)


def mjysr_to_lsun_pc2(mu_mjysr: np.ndarray, m_sun_ab: float = M_SUN_AB_F200W) -> np.ndarray:
    """
    Convert surface brightness from MJy/sr to Lsun/pc^2.
    """
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
    """
    Build JAM luminosity arrays from an MGE table with counts.
    """
    sigma_pix = np.asarray(mge_tab[sigma_pix_col], dtype=float)
    q_obs = np.asarray(mge_tab[q_col], dtype=float)
    total_counts = np.asarray(mge_tab[total_col], dtype=float)

    peak_mjysr = gaussian_peak_from_total_counts(total_counts, sigma_pix, q_obs)
    surf_lum = mjysr_to_lsun_pc2(peak_mjysr, m_sun_ab=m_sun_ab)
    sigma_arcsec = sigma_pix * pixel_scale_arcsec

    return surf_lum, sigma_arcsec, q_obs


def load_mge_inputs(cfg: Config) -> MGEInputs:
    """
    Load the MGE inputs used by JAM.

    Preference:
    1. If mge_luminosity_table.csv has precomputed luminosities, use them.
    2. Otherwise derive luminosities from mge_solution.csv.
    """
    lum_table = Table.read(cfg.mge_luminosity_path)

    sigma_lum = np.asarray(lum_table["sigma_arcsec"], dtype=float)
    q_obs_lum = np.asarray(lum_table["q_obs"], dtype=float)

    if "luminosity_Lsun" in lum_table.colnames:
        surf_lum = np.asarray(lum_table["luminosity_Lsun"], dtype=float)
    else:
        logging.info("No luminosity_Lsun column found; deriving surf_lum from mge_solution.csv")
        mge_solution = Table.read(cfg.mge_solution_path)
        surf_lum, _, _ = make_jam_mge_from_table(
            mge_solution,
            pixel_scale_arcsec=cfg.pixel_scale_arcsec,
            m_sun_ab=M_SUN_AB_F200W,
        )

    return MGEInputs(
        surf_lum=surf_lum,
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
    vrms_err = np.sqrt((vlos_rf * vlos_err) ** 2 + (sigma * sigma_err) ** 2) / vrms
    return vrms, vrms_err


def load_kinematics(cfg: Config) -> Kinematics:
    kin_table = Table.read(cfg.kin_path, format="csv", delimiter=";")

    x = np.asarray(kin_table["X"], dtype=float)
    y = np.asarray(kin_table["Y"], dtype=float)

    x_rot, y_rot = rotate_coordinates(x, y, cfg.rotation_deg)
    kin_table["X_rot"] = x_rot
    kin_table["Y_rot"] = y_rot

    losv = np.asarray(kin_table["LOSV"], dtype=float)
    losv_err = np.asarray(kin_table["LOSV_err"], dtype=float)
    sigma = np.asarray(kin_table["sigma"], dtype=float)
    sigma_err = np.asarray(kin_table["sigma_err"], dtype=float)

    vlos_rf = compute_rest_frame_vlos(losv, cfg.redshift)
    vrms, vrms_err = compute_vrms_and_error(vlos_rf, losv_err, sigma, sigma_err)

    return Kinematics(
        table=kin_table,
        xbin=x_rot,
        ybin=y_rot,
        vlos_rf=vlos_rf,
        sigma=sigma,
        vrms=vrms,
        vrms_err=vrms_err,
    )


def save_diagnostic_maps(cfg: Config, kin: Kinematics) -> None:
    save_map(
        cfg.output_dir / "kinematic_maps_LOSV.png",
        kin.xbin,
        kin.ybin,
        np.asarray(kin.table["LOSV"], dtype=float),
        cbar_label="LOSV (km/s)",
    )
    save_map(
        cfg.output_dir / "kinematic_maps_sigma.png",
        kin.xbin,
        kin.ybin,
        kin.sigma,
        cbar_label="sigma (km/s)",
    )
    save_map(
        cfg.output_dir / "kinematic_maps_vlos_compensated.png",
        kin.xbin,
        kin.ybin,
        kin.vlos_rf,
        cbar_label="Vlos Compensated (km/s)",
    )
    save_map(
        cfg.output_dir / "kinematic_maps_vrms.png",
        kin.xbin,
        kin.ybin,
        kin.vrms,
        cbar_label="Vrms (km/s)",
    )


# -----------------------------------------------------------------------------
# JAM likelihood and priors
# -----------------------------------------------------------------------------

def make_log_likelihood(cfg: Config, kin: Kinematics, mge: MGEInputs):
    def log_likelihood(theta: np.ndarray) -> float:
        bh_mass, beta_scalar = theta
        beta = np.full(mge.surf_lum.shape, beta_scalar, dtype=float)

        try:
            out = jam.axi.proj(
                surf_lum=mge.surf_lum,
                sigma_lum=mge.sigma_lum,
                qobs_lum=mge.q_obs_lum,
                surf_pot=mge.surf_lum * cfg.ml,
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
                data=kin.vrms,
                errors=kin.vrms_err,
                flux_obs=None,
                gamma=None,
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
                quiet=True,
            )
        except Exception as exc:
            logging.debug("JAM evaluation failed for theta=%s: %s", theta, exc)
            return -np.inf

        chi2 = getattr(out, "chi2", np.inf)
        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2

    return log_likelihood


def prior_transform(utheta: np.ndarray, cfg: Config) -> np.ndarray:
    bh_mass = cfg.bh_mass_min + utheta[0] * (cfg.bh_mass_max - cfg.bh_mass_min)
    beta = cfg.beta_min + utheta[1] * (cfg.beta_max - cfg.beta_min)
    return np.array([bh_mass, beta])


def make_prior_transform(cfg: Config):
    def _prior_transform(utheta: np.ndarray) -> np.ndarray:
        return prior_transform(utheta, cfg)

    return _prior_transform


# -----------------------------------------------------------------------------
# Sampling and output
# -----------------------------------------------------------------------------

def run_sampler(cfg: Config, log_likelihood_fn, prior_transform_fn):
    sampler = DynamicNestedSampler(
        log_likelihood_fn,
        prior_transform_fn,
        ndim=2,
        nlive=cfg.nlive,
    )

    sampler.run_nested(
        checkpoint_every=cfg.checkpoint_every,
        checkpoint_file=str(cfg.output_dir / "checkpoint.save"),
        dlogz_init=cfg.dlogz_init,
    )

    return sampler.results


def save_results(cfg: Config, results) -> None:
    np.savez(
        cfg.output_dir / "nested_beta_bh_results.npz",
        samples=results.samples,
        logl=results.logl,
        logwt=results.logwt,
        logz=results.logz,
        logzerr=results.logzerr,
    )

    fig, _ = dyplot.cornerplot(
        results,
        show_titles=True,
        title_kwargs={"x": 0.65},
    )
    fig.savefig(cfg.output_dir / "corner_plot.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    cfg = Config()
    ensure_output_dir(cfg.output_dir)

    logging.info("Loading kinematics")
    kin = load_kinematics(cfg)
    save_diagnostic_maps(cfg, kin)

    logging.info("Loading MGE inputs")
    mge = load_mge_inputs(cfg)

    logging.info("Building likelihood")
    log_likelihood_fn = make_log_likelihood(cfg, kin, mge)
    prior_transform_fn = make_prior_transform(cfg)

    logging.info("Running dynesty")
    results = run_sampler(cfg, log_likelihood_fn, prior_transform_fn)

    logging.info("Saving outputs")
    save_results(cfg, results)

    logging.info("Done")


if __name__ == "__main__":
    main()