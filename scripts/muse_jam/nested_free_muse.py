"""
Fit JAM models to the Sombrero MUSE pPXF kinematics with dynesty.

This script mirrors the workflow in scripts/jampy/nested_free.py while
replacing the JWST/NIRSpec CSV kinematic loader with a reader for the MUSE
pPXF products saved as .npz/.fits files.

Key differences relative to nested_free.py:
    - reads stellar kinematics from the pPXF BIN_RESULTS products,
    - assumes the MUSE velocities are already in the galaxy rest frame,
    - assumes the MUSE maps are already aligned with the major axis,
    - defaults to the MUSE spaxel size of 0.2 arcsec,
    - leaves PSF convolution disabled until a MUSE PSF sigma is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table


JAMPY_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "jampy"
if str(JAMPY_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(JAMPY_SCRIPT_DIR))

import nested_free as base


@dataclass
class Config:
    output_dir: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/muse_free_beta"
    )

    ppxf_products_dir: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Plots/ppxfppxf_c30_emiles_refactored"
    )
    ppxf_npz_glob: str = "*_ppxf_products_*.npz"
    ppxf_fits_glob: str = "*_ppxf_products_*.fits"
    prefer_npz: bool = True

    mge_solution_path: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_solution.csv"
    )

    mge_luminosity_path: Path = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_luminosity_table.csv"
    )

    rotation_deg: float = 0.0
    velocities_are_restframe: bool = True
    redshift: float = 0.003633
    distance_mpc: float = 9.55
    inclination_deg: float = 87.0

    sigmapsf_arcsec: float = 0.0
    pixsize_arcsec: float = 0.2
    pixel_scale_arcsec: float = 0.031

    nlive: int = 200
    nprocs: int = 8
    dlogz_init: float = 0.0001
    checkpoint_every_sec: float = 30.0
    checkpoint_filename: str = "checkpoint.save"
    bound_method: str = "multi"
    sample_method: str = "unif"
    walks: int = 32
    bootstrap: int = 20

    bh_mass_min: float = 1e6
    bh_mass_max: float = 1e10

    beta_min: float = -15.0
    beta_max: float = 0.99

    ml_min: float = 0.1
    ml_max: float = 2.0

    beta_prescription: str = "free"

    beta_ra_min: float = 0.001
    beta_ra_max: float = 300.0
    beta_alpha_min: float = 0.1
    beta_alpha_max: float = 10.0


def resolve_unique_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        formatted = "\n".join(f"  - {match}" for match in matches)
        raise ValueError(
            f"Expected exactly one file matching '{pattern}' in {directory}, found:\n{formatted}"
        )
    return matches[0]


def resolve_ppxf_products_path(cfg: Config) -> Path:
    npz_path = resolve_unique_file(cfg.ppxf_products_dir, cfg.ppxf_npz_glob)
    fits_path = resolve_unique_file(cfg.ppxf_products_dir, cfg.ppxf_fits_glob)

    if cfg.prefer_npz and npz_path is not None:
        return npz_path
    if fits_path is not None:
        return fits_path
    if npz_path is not None:
        return npz_path

    raise FileNotFoundError(
        f"No pPXF product files matching '{cfg.ppxf_npz_glob}' or "
        f"'{cfg.ppxf_fits_glob}' were found in {cfg.ppxf_products_dir}"
    )


def log_source_metadata(cfg: Config, source_path: Path) -> None:
    fits_path = source_path if source_path.suffix == ".fits" else source_path.with_suffix(".fits")
    if not fits_path.exists():
        return

    hdr = fits.getheader(fits_path, 0)
    hdr_pixsize = hdr.get("PIXSIZE")
    hdr_redshift = hdr.get("REDSHFT")
    hdr_nbins = hdr.get("NBINS")

    logging.info(
        "MUSE product metadata: NBINS=%s, PIXSIZE=%s arcsec, REDSHIFT=%s",
        hdr_nbins,
        hdr_pixsize,
        hdr_redshift,
    )

    if hdr_pixsize is not None and not np.isclose(float(hdr_pixsize), cfg.pixsize_arcsec):
        logging.warning(
            "Configured pixsize_arcsec=%.6f differs from FITS header PIXSIZE=%.6f",
            cfg.pixsize_arcsec,
            float(hdr_pixsize),
        )

    if hdr_redshift is not None and not np.isclose(float(hdr_redshift), cfg.redshift):
        logging.warning(
            "Configured redshift=%.6f differs from FITS header REDSHIFT=%.6f",
            cfg.redshift,
            float(hdr_redshift),
        )


def build_kinematics(
    cfg: Config,
    kin_table: Table,
    *,
    x_col: str,
    y_col: str,
    v_col: str,
    v_err_col: str,
    sigma_col: str,
    sigma_err_col: str,
):
    # Use the raw pPXF bin results directly. No symmetrization is applied to
    # Vlos, sigma, or Vrms before fitting or plotting.
    x = np.asarray(kin_table[x_col], dtype=float)
    y = np.asarray(kin_table[y_col], dtype=float)
    x_rot, y_rot = base.rotate_coordinates(x, y, cfg.rotation_deg)

    vlos_obs = np.asarray(kin_table[v_col], dtype=float)
    vlos_err = np.asarray(kin_table[v_err_col], dtype=float)
    sigma = np.asarray(kin_table[sigma_col], dtype=float)
    sigma_err = np.asarray(kin_table[sigma_err_col], dtype=float)

    if cfg.velocities_are_restframe:
        vlos_rf = np.asarray(vlos_obs, dtype=float)
    else:
        vlos_rf = base.compute_rest_frame_vlos(vlos_obs, cfg.redshift)

    vrms, vrms_err = base.compute_vrms_and_error(vlos_rf, vlos_err, sigma, sigma_err)

    kin_table = kin_table.copy()
    kin_table["X_rot"] = x_rot
    kin_table["Y_rot"] = y_rot
    kin_table["Vrms"] = vrms
    kin_table["Vrms_err"] = vrms_err

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
        & (sigma_err > 0)
        & (vrms_err > 0)
    )

    return base.Kinematics(
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


def load_kinematics_from_npz(path: Path, cfg: Config):
    required_keys = (
        "bin_id",
        "xbin",
        "ybin",
        "velbin",
        "velerr_bin",
        "sigbin",
        "sigerr_bin",
    )

    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise KeyError(f"Missing required NPZ keys in {path}: {missing}")

        kin_table = Table()
        kin_table["BIN_ID"] = np.asarray(data["bin_id"], dtype=int)
        kin_table["X_ARCSEC"] = np.asarray(data["xbin"], dtype=float)
        kin_table["Y_ARCSEC"] = np.asarray(data["ybin"], dtype=float)
        kin_table["V_KMS"] = np.asarray(data["velbin"], dtype=float)
        kin_table["VERR_KMS"] = np.asarray(data["velerr_bin"], dtype=float)
        kin_table["SIGMA_KMS"] = np.asarray(data["sigbin"], dtype=float)
        kin_table["SIGERR_KMS"] = np.asarray(data["sigerr_bin"], dtype=float)

    return build_kinematics(
        cfg,
        kin_table,
        x_col="X_ARCSEC",
        y_col="Y_ARCSEC",
        v_col="V_KMS",
        v_err_col="VERR_KMS",
        sigma_col="SIGMA_KMS",
        sigma_err_col="SIGERR_KMS",
    )


def load_kinematics_from_fits(path: Path, cfg: Config):
    kin_table = Table.read(path, hdu="BIN_RESULTS")
    return build_kinematics(
        cfg,
        kin_table,
        x_col="X_ARCSEC",
        y_col="Y_ARCSEC",
        v_col="V_KMS",
        v_err_col="VERR_KMS",
        sigma_col="SIGMA_KMS",
        sigma_err_col="SIGERR_KMS",
    )


def load_kinematics(cfg: Config):
    source_path = resolve_ppxf_products_path(cfg)
    logging.info("Loading MUSE pPXF products from %s", source_path)
    log_source_metadata(cfg, source_path)

    if source_path.suffix == ".npz":
        return load_kinematics_from_npz(source_path, cfg)
    if source_path.suffix == ".fits":
        return load_kinematics_from_fits(source_path, cfg)

    raise ValueError(f"Unsupported pPXF product format: {source_path}")


def run_with_config(cfg: Config) -> None:
    base.ensure_output_dir(cfg.output_dir)

    logging.info("Loading and preparing MUSE kinematics")
    if cfg.rotation_deg == 0.0:
        logging.info("rotation_deg=0.0: using the MUSE maps as already aligned to the major axis")
    if cfg.velocities_are_restframe:
        logging.info("velocities_are_restframe=True: skipping the redshift velocity subtraction")
    if cfg.sigmapsf_arcsec <= 0:
        logging.warning(
            "sigmapsf_arcsec=%.3f disables PSF convolution. "
            "Set this to the MUSE PSF sigma in arcsec when ready.",
            cfg.sigmapsf_arcsec,
        )

    kin = load_kinematics(cfg)
    base.save_kinematic_maps(cfg, kin)

    logging.info("Loading MGE inputs")
    mge = base.load_mge_inputs(cfg)

    n_mge = len(mge.surf_lum)
    ndim = base.get_ndim(cfg, n_mge)

    logging.info(
        "Building likelihood and priors with beta_prescription='%s', ndim=%d, bound='%s', sample='%s'",
        cfg.beta_prescription,
        ndim,
        cfg.bound_method,
        cfg.sample_method,
    )
    log_likelihood_fn = base.JamVrmsLogLikelihood(cfg, kin, mge)
    prior_transform_fn = base.UniformPriorTransform(cfg, n_mge)

    logging.info(
        "Running dynesty with checkpointing every %.0f seconds (~%.1f min)",
        cfg.checkpoint_every_sec,
        cfg.checkpoint_every_sec / 60.0,
    )
    results = base.run_sampler(cfg, log_likelihood_fn, prior_transform_fn, n_mge)

    logging.info("Saving samples, posterior plot, and LOSV/Vrms best-fit comparisons")
    base.save_results(cfg, results, kin, mge)

    logging.info("Done")


def main() -> None:
    base.setup_logging()
    run_with_config(Config())


if __name__ == "__main__":
    main()
