#!/usr/bin/env python3
"""
Convolve the AGN-subtracted NIRSpec G235H cube to the MUSE spatial resolution,
fit the central aperture with pPXF/XSL, and compare to the existing MUSE XSL fit.

The defaults are set for the Sombrero REVEAL workspace:

    Data/IFU/david_subs/g235h_agn_sub.fits
    Plots/ppxfppxf_c30_xsl/c30_DATACUBE_normppxf_skycont_Part1_0000_ppxf_products_xsl.fits

This script intentionally fits a central aperture rather than every NIRSpec
spaxel. After PSF matching to a roughly 2 arcsec MUSE seeing disk, the native
0.1 arcsec NIRSpec spaxels are highly correlated, and the aperture comparison is
the cleaner resolution test.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
import sys
import tempfile
from importlib import resources
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_xsl_muse_resolution"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from ppxf import sps_util
from ppxf.ppxf import ppxf, robust_sigma
from scipy import ndimage

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import ppxf_nirspec_phoenix_kinematics as base


C = 299792.458
FWHM_PER_SIGMA = 2.35482004503
ROOT = Path("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL")

DEFAULT_CUBE = ROOT / "Data/IFU/david_subs/g235h_agn_sub.fits"
DEFAULT_MUSE_CUBE = ROOT / "Data/MUSE/c30_cubes/c30_DATACUBE_normppxf_skycont_Part1_0000.fits"
DEFAULT_MUSE_XSL = (
    ROOT
    / "Plots/ppxfppxf_c30_xsl/c30_DATACUBE_normppxf_skycont_Part1_0000_ppxf_products_xsl.fits"
)
DEFAULT_LSF_TABLE = ROOT / "scripts/ppxf_nirspec/jwst_nirspec_g235h_disp.fits"
DEFAULT_OUTPUT_DIR = ROOT / "Data/ppxf_nirspec/agn_sub_xsl_muse_spatial_resolution"


@dataclass(frozen=True)
class XslLibrary:
    templates: np.ndarray
    lam_temp: np.ndarray
    age_grid: np.ndarray
    metal_grid: np.ndarray
    reg_dim: tuple[int, int]
    template_path: Path
    template_count: int
    data_r_min: float
    data_r_med: float
    data_r_max: float
    data_fwhm_med_ang: float
    xsl_fwhm_med_ang: float


@dataclass(frozen=True)
class RunConfig:
    cube_path: Path
    muse_cube_path: Path
    muse_xsl_path: Path
    output_dir: Path
    xsl_template_path: Path
    lsf_table_path: Path
    redshift: float
    central_radius_arcsec: float
    muse_spatial_fwhm_arcsec: float | None
    nirspec_spatial_fwhm_arcsec: float
    lsf_mode: str
    resolving_power: float
    fit_windows_rest_um: tuple[tuple[float, float], ...]
    mask_windows_rest_um: tuple[tuple[float, float], ...]
    degree: int
    mdegree: int
    pop_mdegree: int
    regul_start: float
    regul_max: float
    regul_bracket_steps: int
    regul_bisect_steps: int
    moments: int
    bias: float
    start_sigma: float
    max_abs_velocity: float
    min_sigma: float
    max_sigma: float
    min_wave_finite_frac: float
    min_spaxel_finite_frac: float
    min_log_pixel_fraction: float
    min_goodpixels: int
    template_velocity_margin_kms: float
    wave_crpix_mode: str
    reuse_convolved_cube: bool


def default_xsl_path() -> Path:
    return Path(resources.files("ppxf") / "sps_models" / "spectra_xsl_9.0.npz")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan
    return float(np.sum(weights[good] * values[good]) / np.sum(weights[good]))


def finite_weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = finite_weighted_mean(values, weights)
    if not np.isfinite(mean):
        return np.nan
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan
    var = np.sum(weights[good] * (values[good] - mean) ** 2) / np.sum(weights[good])
    return float(np.sqrt(max(var, 0.0)))


def infer_muse_spatial_fwhm_arcsec(path: Path) -> float | None:
    if not path.is_file():
        return None
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
        values = []
        for key in (
            "HIERARCH ESO TEL AMBI FWHM START",
            "HIERARCH ESO TEL AMBI FWHM END",
            "ESO TEL AMBI FWHM START",
            "ESO TEL AMBI FWHM END",
        ):
            if key in hdr:
                values.append(float(hdr[key]))
    values = [v for v in values if np.isfinite(v) and v > 0]
    if not values:
        return None
    return float(np.mean(values))


def find_science_hdu_index(hdul: fits.HDUList) -> int:
    for name in ("SCI", "DATA"):
        if name in hdul and getattr(hdul[name], "data", None) is not None:
            return hdul.index_of(name)
    for i, hdu in enumerate(hdul):
        if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) == 3:
            return i
    raise ValueError("Could not find a 3D science cube")


def convolve_cube_to_fwhm(
    input_path: Path,
    output_path: Path,
    target_fwhm_arcsec: float,
    native_fwhm_arcsec: float,
    chunk_nwave: int = 128,
) -> tuple[Path, float]:
    with fits.open(input_path, memmap=False) as hdul:
        sci_idx = find_science_hdu_index(hdul)
        sci_hdu = hdul[sci_idx]
        header = sci_hdu.header.copy()
        pixsize = base.derive_pixsize_arcsec(header)
        cube = base.orient_cube_nlam_first(np.asarray(sci_hdu.data, dtype=np.float32), header)
        native_shape = sci_hdu.data.shape

        kernel_fwhm = np.sqrt(max(target_fwhm_arcsec**2 - native_fwhm_arcsec**2, 0.0))
        sigma_pix = kernel_fwhm / FWHM_PER_SIGMA / pixsize if kernel_fwhm > 0 else 0.0
        convolved = np.empty_like(cube, dtype=np.float32)

        if sigma_pix <= 0:
            convolved[:] = cube
        else:
            sigma = (0.0, sigma_pix, sigma_pix)
            for start in range(0, cube.shape[0], chunk_nwave):
                stop = min(start + chunk_nwave, cube.shape[0])
                chunk = cube[start:stop]
                valid = np.isfinite(chunk)
                filled = np.where(valid, chunk, 0.0)
                numerator = ndimage.gaussian_filter(
                    filled,
                    sigma=sigma,
                    mode="constant",
                    cval=0.0,
                    truncate=4.0,
                )
                denominator = ndimage.gaussian_filter(
                    valid.astype(np.float32),
                    sigma=sigma,
                    mode="constant",
                    cval=0.0,
                    truncate=4.0,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    out = numerator / denominator
                out[denominator <= 1e-6] = np.nan
                convolved[start:stop] = out.astype(np.float32)

        if native_shape != convolved.shape:
            data_out = np.moveaxis(convolved, 0, -1)
        else:
            data_out = convolved

        out_hdus = fits.HDUList([hdu.copy() for hdu in hdul])
        out_hdus[sci_idx].data = data_out.astype(np.float32)
        out_hdus[sci_idx].header["MUSEFWHM"] = (float(target_fwhm_arcsec), "target spatial FWHM [arcsec]")
        out_hdus[sci_idx].header["NIRFWHM"] = (float(native_fwhm_arcsec), "assumed native NIRSpec FWHM [arcsec]")
        out_hdus[sci_idx].header["KERFWHM"] = (float(kernel_fwhm), "Gaussian kernel FWHM [arcsec]")
        out_hdus[sci_idx].header["KERSIGPX"] = (float(sigma_pix), "Gaussian kernel sigma [spaxels]")
        out_hdus[sci_idx].header.add_history(
            "Spatially convolved with a NaN-normalized circular Gaussian for MUSE PSF matching."
        )
        out_hdus.writeto(output_path, overwrite=True)

    return output_path, float(sigma_pix)


def make_base_config(cfg: RunConfig, cube_path: Path) -> base.Config:
    return base.Config(
        cube_path=cube_path.resolve(),
        output_dir=cfg.output_dir.resolve(),
        phoenix_dir=cfg.xsl_template_path.parent.resolve(),
        phoenix_wave_path=cfg.xsl_template_path.resolve(),
        template_list_path=None,
        redshift=float(cfg.redshift),
        resolving_power=float(cfg.resolving_power),
        template_resolving_power=np.nan,
        fit_windows_rest_um=cfg.fit_windows_rest_um,
        mask_windows_rest_um=cfg.mask_windows_rest_um,
        teff_min=np.nan,
        teff_max=np.nan,
        feh_min=np.nan,
        feh_max=np.nan,
        logg_min=np.nan,
        logg_max=np.nan,
        expected_template_count=0,
        strict_template_count=False,
        degree=int(cfg.degree),
        mdegree=int(cfg.mdegree),
        moments=int(cfg.moments),
        bias=float(cfg.bias),
        start_sigma=float(cfg.start_sigma),
        max_abs_velocity=float(cfg.max_abs_velocity),
        min_sigma=float(cfg.min_sigma),
        max_sigma=float(cfg.max_sigma),
        min_wave_finite_frac=float(cfg.min_wave_finite_frac),
        min_spaxel_finite_frac=float(cfg.min_spaxel_finite_frac),
        min_log_pixel_fraction=float(cfg.min_log_pixel_fraction),
        min_goodpixels=int(cfg.min_goodpixels),
        top_template_frac=0.10,
        csv_min_sn=0.0,
        n_plot_spaxels=1,
        max_spaxels=None,
        template_velocity_margin_kms=float(cfg.template_velocity_margin_kms),
        wave_crpix_mode=str(cfg.wave_crpix_mode),
    )


def resolving_power_curve(cfg: RunConfig, lam_native_ang: np.ndarray) -> tuple[np.ndarray, str]:
    if cfg.lsf_mode == "fixed":
        return np.full_like(lam_native_ang, cfg.resolving_power, dtype=float), f"fixed R={cfg.resolving_power:.1f}"
    if cfg.lsf_mode != "table":
        raise ValueError(f"Unsupported lsf mode: {cfg.lsf_mode}")
    with fits.open(cfg.lsf_table_path, memmap=False) as hdul:
        tab = hdul[1].data
        wave_um = np.asarray(tab["WAVELENGTH"], dtype=float)
        r_values = np.asarray(tab["R"], dtype=float)
    obs_um = lam_native_ang / 1e4 * (1.0 + cfg.redshift)
    r_interp = np.interp(obs_um, wave_um, r_values, left=r_values[0], right=r_values[-1])
    return r_interp, str(cfg.lsf_table_path)


def load_xsl_library(cfg: RunConfig, cube_data: base.CubeData) -> XslLibrary:
    if not cfg.xsl_template_path.is_file():
        raise FileNotFoundError(f"Missing XSL template file: {cfg.xsl_template_path}")

    fit_lam = cube_data.lam_gal_ang[cube_data.fit_mask_log]
    vel_pad = cfg.max_abs_velocity + cfg.template_velocity_margin_kms
    lam_min = float(np.nanmin(fit_lam) * np.exp(-vel_pad / C))
    lam_max = float(np.nanmax(fit_lam) * np.exp(vel_pad / C))

    with np.load(cfg.xsl_template_path) as data:
        native_lam = np.asarray(data["lam"], dtype=float)
        native_fwhm = np.asarray(data["fwhm"], dtype=float)

    r_native, _ = resolving_power_curve(cfg, native_lam)
    fwhm_gal = {"lam": native_lam, "fwhm": native_lam / r_native}
    norm_range = [
        min(lo for lo, _ in cfg.fit_windows_rest_um) * 1e4,
        max(hi for _, hi in cfg.fit_windows_rest_um) * 1e4,
    ]
    sps = sps_util.sps_lib(
        cfg.xsl_template_path,
        cube_data.velscale,
        fwhm_gal=fwhm_gal,
        lam_range=[lam_min, lam_max],
        norm_range=norm_range,
    )
    npix, *reg_dim = sps.templates.shape
    templates = np.asarray(sps.templates.reshape(npix, -1), dtype=float)
    templates = base.normalize_columns(templates)

    band = (native_lam >= lam_min) & (native_lam <= lam_max)
    data_fwhm_band = native_lam[band] / r_native[band]
    return XslLibrary(
        templates=templates,
        lam_temp=np.asarray(sps.lam_temp, dtype=float),
        age_grid=np.asarray(sps.age_grid, dtype=float),
        metal_grid=np.asarray(sps.metal_grid, dtype=float),
        reg_dim=(int(reg_dim[0]), int(reg_dim[1])),
        template_path=cfg.xsl_template_path,
        template_count=int(templates.shape[1]),
        data_r_min=float(np.nanmin(r_native[band])),
        data_r_med=float(np.nanmedian(r_native[band])),
        data_r_max=float(np.nanmax(r_native[band])),
        data_fwhm_med_ang=float(np.nanmedian(data_fwhm_band)),
        xsl_fwhm_med_ang=float(np.nanmedian(native_fwhm[band])),
    )


def xsl_mean_population(weights: np.ndarray | None, library: XslLibrary) -> tuple[float, float]:
    if weights is None:
        return np.nan, np.nan
    weights = np.asarray(weights, dtype=float)
    if weights.size != library.template_count:
        return np.nan, np.nan
    weights = weights.reshape(library.reg_dim)
    total = np.nansum(weights)
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    logage_grid = np.log10(library.age_grid) + 9.0
    logage = np.nansum(weights * logage_grid) / total
    metal = np.nansum(weights * library.metal_grid) / total
    return float(logage), float(metal)


def fit_population_regularized(
    cfg: RunConfig,
    cube_data: base.CubeData,
    library: XslLibrary,
    galaxy_in: np.ndarray,
    kin_result: base.FitResult,
    fit_mask_in: np.ndarray,
) -> tuple[object | None, dict[str, float]]:
    if kin_result.sol_rel is None or kin_result.clean_mask is None:
        return None, {
            "pop_ok": False,
            "pop_message": "missing_kinematics",
            "pop_logage_yr": np.nan,
            "pop_metal": np.nan,
            "pop_chi2": np.nan,
            "pop_regul": np.nan,
            "pop_target_dchi2": np.nan,
            "pop_achieved_dchi2": np.nan,
        }

    fit_mask = np.asarray(kin_result.clean_mask, dtype=bool) & fit_mask_in & np.isfinite(galaxy_in)
    if np.count_nonzero(fit_mask) < cfg.min_goodpixels:
        return None, {
            "pop_ok": False,
            "pop_message": "too_few_goodpixels",
            "pop_logage_yr": np.nan,
            "pop_metal": np.nan,
            "pop_chi2": np.nan,
            "pop_regul": np.nan,
            "pop_target_dchi2": np.nan,
            "pop_achieved_dchi2": np.nan,
        }

    galaxy, _ = base.normalize_galaxy(galaxy_in, fit_mask)
    if kin_result.bestfit is not None:
        resid = galaxy[fit_mask] - kin_result.bestfit[fit_mask]
        noise_val = robust_sigma(resid, zero=1)
    else:
        noise_val = base.robust_noise_value(galaxy, fit_mask)
    if not np.isfinite(noise_val) or noise_val <= 0:
        noise_val = base.robust_noise_value(galaxy, fit_mask)
    noise = base.safe_positive(np.full_like(galaxy, noise_val, dtype=float))
    kin_sol = [float(kin_result.sol_rel[0]), float(kin_result.sol_rel[1])]

    def run_pop(regul: float, noise_vec: np.ndarray):
        return ppxf(
            library.templates,
            galaxy,
            noise_vec,
            cube_data.velscale,
            kin_sol,
            moments=-2,
            degree=-1,
            mdegree=cfg.pop_mdegree,
            lam=cube_data.lam_gal_ang,
            lam_temp=library.lam_temp,
            mask=fit_mask,
            regul=regul,
            reg_dim=library.reg_dim,
            quiet=True,
        )

    try:
        pp0 = run_pop(0.0, noise)
        noise = noise * np.sqrt(float(pp0.chi2))
        pp0 = run_pop(0.0, noise)

        target_dchi2 = float(np.sqrt(2.0 * pp0.goodpixels.size))

        def dchi2(pp) -> float:
            return float((pp.chi2 - pp0.chi2) * pp0.goodpixels.size)

        best_pp = pp0
        best_regul = 0.0
        lo_reg = 0.0
        hi_reg = float(cfg.regul_start)
        hi_pp = None

        for _ in range(cfg.regul_bracket_steps):
            hi_pp = run_pop(hi_reg, noise)
            if dchi2(hi_pp) >= target_dchi2:
                break
            best_pp = hi_pp
            best_regul = hi_reg
            lo_reg = hi_reg
            hi_reg = min(hi_reg * 2.0, cfg.regul_max)
            if hi_reg >= cfg.regul_max:
                hi_pp = run_pop(hi_reg, noise)
                break

        if hi_pp is not None and dchi2(hi_pp) >= target_dchi2 and hi_reg > 0:
            for _ in range(cfg.regul_bisect_steps):
                mid_reg = 0.5 * hi_reg if lo_reg <= 0 else float(np.sqrt(lo_reg * hi_reg))
                mid_pp = run_pop(mid_reg, noise)
                if dchi2(mid_pp) <= target_dchi2:
                    best_pp = mid_pp
                    best_regul = mid_reg
                    lo_reg = mid_reg
                else:
                    hi_reg = mid_reg

        logage, metal = xsl_mean_population(best_pp.weights[: library.template_count], library)
        achieved = dchi2(best_pp)
        best_pp.regul_used = float(best_regul)
        return best_pp, {
            "pop_ok": True,
            "pop_message": "ok",
            "pop_logage_yr": float(logage),
            "pop_metal": float(metal),
            "pop_chi2": float(best_pp.chi2),
            "pop_regul": float(best_regul),
            "pop_target_dchi2": float(target_dchi2),
            "pop_achieved_dchi2": float(achieved),
        }
    except Exception as exc:
        return None, {
            "pop_ok": False,
            "pop_message": f"population_failed:{exc}",
            "pop_logage_yr": np.nan,
            "pop_metal": np.nan,
            "pop_chi2": np.nan,
            "pop_regul": np.nan,
            "pop_target_dchi2": np.nan,
            "pop_achieved_dchi2": np.nan,
        }


def select_central_indices(cube_data: base.CubeData, radius_arcsec: float) -> np.ndarray:
    radius = np.hypot(cube_data.x, cube_data.y)
    mask = np.isfinite(radius) & (radius <= radius_arcsec)
    return np.flatnonzero(mask)


def fit_central_nirspec(
    cfg: RunConfig,
    cube_data: base.CubeData,
    library: XslLibrary,
) -> tuple[base.FitResult, object | None, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    central_idx = select_central_indices(cube_data, cfg.central_radius_arcsec)
    if central_idx.size == 0:
        raise ValueError("No NIRSpec spaxels fall inside the requested central aperture")

    galaxy = np.nanmean(cube_data.spectra_log[:, central_idx], axis=1)
    valid_frac = np.nanmean(cube_data.valid_frac_log[:, central_idx], axis=1)
    fit_mask = cube_data.fit_mask_log & (valid_frac >= cfg.min_log_pixel_fraction)
    fit_cfg = make_base_config(cfg, cfg.cube_path)
    result = base.ppxf_fit_spectrum(
        library.templates,
        galaxy,
        cube_data.velscale,
        [0.0, cfg.start_sigma],
        fit_mask,
        cube_data.lam_gal_ang,
        library.lam_temp,
        fit_cfg,
    )
    pop_result, pop_summary = fit_population_regularized(cfg, cube_data, library, galaxy, result, fit_mask)
    summary = {
        "n_spaxels": int(central_idx.size),
        "radius_arcsec": float(cfg.central_radius_arcsec),
        "v_rel_kms": float(result.sol_rel[0]) if result.sol_rel is not None else np.nan,
        "v_rel_err_kms": float(result.err_rel[0]) if result.err_rel is not None else np.nan,
        "losv_kms": float(result.losv),
        "losv_err_kms": float(result.losv_err),
        "sigma_kms": float(result.sigma),
        "sigma_err_kms": float(result.sigma_err),
        "vrms_kms": float(result.vrms),
        "vrms_err_kms": float(result.vrms_err),
        "h3": float(result.h3),
        "h3_err": float(result.h3_err),
        "h4": float(result.h4),
        "h4_err": float(result.h4_err),
        "sn": float(result.sn),
        "chi2": float(result.chi2),
        "logage_yr": float(pop_summary["pop_logage_yr"]),
        "metal": float(pop_summary["pop_metal"]),
        **pop_summary,
        "message": result.message,
        "ok": bool(result.ok),
    }
    return result, pop_result, summary, galaxy, fit_mask, central_idx


def muse_central_summary(path: Path, radius_arcsec: float) -> dict[str, float]:
    with fits.open(path, memmap=False) as hdul:
        spax = hdul["SPAXELS"].data
        x = np.asarray(spax["X_ARCSEC"], dtype=float)
        y = np.asarray(spax["Y_ARCSEC"], dtype=float)
        signal = np.asarray(spax["SIGNAL"], dtype=float)
        v = np.asarray(spax["V_KMS"], dtype=float)
        sigma = np.asarray(spax["SIGMA_KMS"], dtype=float)
        logage = np.asarray(spax["LOGAGE_YR"], dtype=float)
        metal = np.asarray(spax["MEAN_METAL"], dtype=float)
        bin_id = np.asarray(spax["BIN_ID"], dtype=int)
    radius = np.hypot(x, y)
    mask = np.isfinite(radius) & (radius <= radius_arcsec)
    weights = np.where(np.isfinite(signal) & (signal > 0), signal, 0.0)
    vrms = np.sqrt(v**2 + sigma**2)
    return {
        "n_spaxels": int(np.count_nonzero(mask)),
        "n_bins": int(np.unique(bin_id[mask]).size) if np.any(mask) else 0,
        "radius_arcsec": float(radius_arcsec),
        "v_rel_kms_weighted_mean": finite_weighted_mean(v[mask], weights[mask]),
        "v_rel_kms_weighted_std": finite_weighted_std(v[mask], weights[mask]),
        "sigma_kms_weighted_mean": finite_weighted_mean(sigma[mask], weights[mask]),
        "sigma_kms_weighted_std": finite_weighted_std(sigma[mask], weights[mask]),
        "vrms_kms_weighted_mean": finite_weighted_mean(vrms[mask], weights[mask]),
        "vrms_kms_weighted_std": finite_weighted_std(vrms[mask], weights[mask]),
        "logage_yr_weighted_mean": finite_weighted_mean(logage[mask], weights[mask]),
        "logage_yr_weighted_std": finite_weighted_std(logage[mask], weights[mask]),
        "metal_weighted_mean": finite_weighted_mean(metal[mask], weights[mask]),
        "metal_weighted_std": finite_weighted_std(metal[mask], weights[mask]),
        "signal_sum": float(np.nansum(weights[mask])),
    }


def write_comparison_csv(path: Path, nirspec: dict[str, float], muse: dict[str, float]) -> None:
    rows = [
        ("V_REL_KMS", nirspec["v_rel_kms"], nirspec["v_rel_err_kms"], muse["v_rel_kms_weighted_mean"], muse["v_rel_kms_weighted_std"]),
        ("SIGMA_KMS", nirspec["sigma_kms"], nirspec["sigma_err_kms"], muse["sigma_kms_weighted_mean"], muse["sigma_kms_weighted_std"]),
        ("VRMS_KMS", nirspec["vrms_kms"], nirspec["vrms_err_kms"], muse["vrms_kms_weighted_mean"], muse["vrms_kms_weighted_std"]),
        ("H3", nirspec["h3"], nirspec["h3_err"], np.nan, np.nan),
        ("H4", nirspec["h4"], nirspec["h4_err"], np.nan, np.nan),
        ("LOGAGE_YR", nirspec["logage_yr"], np.nan, muse["logage_yr_weighted_mean"], muse["logage_yr_weighted_std"]),
        ("METAL", nirspec["metal"], np.nan, muse["metal_weighted_mean"], muse["metal_weighted_std"]),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "nirspec_value", "nirspec_error", "muse_value", "muse_spatial_std", "delta_nirspec_minus_muse"],
        )
        writer.writeheader()
        for metric, nv, ne, mv, ms in rows:
            writer.writerow(
                {
                    "metric": metric,
                    "nirspec_value": nv,
                    "nirspec_error": ne,
                    "muse_value": mv,
                    "muse_spatial_std": ms,
                    "delta_nirspec_minus_muse": nv - mv if np.isfinite(nv) and np.isfinite(mv) else np.nan,
                }
            )


def plot_fit(
    outpath: Path,
    lam: np.ndarray,
    galaxy: np.ndarray,
    result: base.FitResult,
    fit_mask: np.ndarray,
    title: str,
) -> None:
    galaxy_norm, _ = base.normalize_galaxy(galaxy, fit_mask)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(lam, galaxy_norm, color="0.25", lw=0.8, label="NIRSpec aperture")
    if result.bestfit is not None:
        ax.plot(lam, result.bestfit, color="tab:blue", lw=1.0, label="pPXF/XSL")
    ax.set_xlabel("Rest wavelength [Angstrom]")
    ax.set_ylabel("Normalized flux")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(path: Path, nirspec: dict[str, float], muse: dict[str, float]) -> None:
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    kin_labels = ["V", "sigma", "vrms"]
    x = np.arange(len(kin_labels))
    nir_kin = [nirspec["v_rel_kms"], nirspec["sigma_kms"], nirspec["vrms_kms"]]
    muse_kin = [muse["v_rel_kms_weighted_mean"], muse["sigma_kms_weighted_mean"], muse["vrms_kms_weighted_mean"]]
    muse_kin_err = [muse["v_rel_kms_weighted_std"], muse["sigma_kms_weighted_std"], muse["vrms_kms_weighted_std"]]
    ax.bar(x - width / 2, nir_kin, width, label="NIRSpec convolved")
    ax.bar(x + width / 2, muse_kin, width, yerr=muse_kin_err, label="MUSE XSL", alpha=0.85)
    ax.set_xticks(x, kin_labels)
    ax.set_ylabel("km/s")
    ax.set_title("Central kinematics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.axhline(0, color="0.2", lw=0.8)
    ax.bar(np.arange(3), np.asarray(nir_kin) - np.asarray(muse_kin), color="tab:purple")
    ax.set_xticks(np.arange(3), kin_labels)
    ax.set_ylabel("NIRSpec - MUSE [km/s]")
    ax.set_title("Kinematic differences")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    pop_labels = ["log age", "[M/H]"]
    x = np.arange(len(pop_labels))
    nir_pop = [nirspec["logage_yr"], nirspec["metal"]]
    muse_pop = [muse["logage_yr_weighted_mean"], muse["metal_weighted_mean"]]
    muse_pop_err = [muse["logage_yr_weighted_std"], muse["metal_weighted_std"]]
    ax.bar(x - width / 2, nir_pop, width, label="NIRSpec convolved")
    ax.bar(x + width / 2, muse_pop, width, yerr=muse_pop_err, label="MUSE XSL", alpha=0.85)
    ax.set_xticks(x, pop_labels)
    ax.set_title("Population values")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.axhline(0, color="0.2", lw=0.8)
    ax.bar(np.arange(2), np.asarray(nir_pop) - np.asarray(muse_pop), color="tab:green")
    ax.set_xticks(np.arange(2), pop_labels)
    ax.set_ylabel("NIRSpec - MUSE")
    ax.set_title("Population differences")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--muse-cube-path", type=Path, default=DEFAULT_MUSE_CUBE)
    parser.add_argument("--muse-xsl-path", type=Path, default=DEFAULT_MUSE_XSL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xsl-template-path", type=Path, default=default_xsl_path())
    parser.add_argument("--lsf-table-path", type=Path, default=DEFAULT_LSF_TABLE)
    parser.add_argument("--redshift", type=float, default=0.003633)
    parser.add_argument("--central-radius-arcsec", type=float, default=2.0)
    parser.add_argument("--muse-spatial-fwhm-arcsec", type=float, default=None)
    parser.add_argument("--nirspec-spatial-fwhm-arcsec", type=float, default=0.10)
    parser.add_argument("--lsf-mode", choices=("table", "fixed"), default="table")
    parser.add_argument("--resolving-power", type=float, default=2700.0)
    parser.add_argument("--fit-windows-rest-um", type=str, default="2.10-2.40")
    parser.add_argument(
        "--mask-windows-rest-um",
        type=str,
        default="2.117-2.127,2.161-2.171,2.219-2.228,2.316-2.326",
    )
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--mdegree", type=int, default=0)
    parser.add_argument("--pop-mdegree", type=int, default=8)
    parser.add_argument("--regul-start", type=float, default=100.0)
    parser.add_argument("--regul-max", type=float, default=5.0e4)
    parser.add_argument("--regul-bracket-steps", type=int, default=10)
    parser.add_argument("--regul-bisect-steps", type=int, default=10)
    parser.add_argument("--moments", type=int, default=4)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--start-sigma", type=float, default=220.0)
    parser.add_argument("--max-abs-velocity", type=float, default=700.0)
    parser.add_argument("--min-sigma", type=float, default=10.0)
    parser.add_argument("--max-sigma", type=float, default=700.0)
    parser.add_argument("--min-wave-finite-frac", type=float, default=0.35)
    parser.add_argument("--min-spaxel-finite-frac", type=float, default=0.60)
    parser.add_argument("--min-log-pixel-fraction", type=float, default=0.70)
    parser.add_argument("--min-goodpixels", type=int, default=180)
    parser.add_argument(
        "--template-velocity-margin-kms",
        type=float,
        default=6000.0,
        help="Padding for the XSL template wavelength range; the fixed-kinematics population pass needs a generous margin.",
    )
    parser.add_argument("--wave-crpix-mode", choices=("fits", "first_pixel"), default="fits")
    parser.add_argument("--reuse-convolved-cube", action="store_true")
    args = parser.parse_args()

    return RunConfig(
        cube_path=args.cube_path.expanduser().resolve(),
        muse_cube_path=args.muse_cube_path.expanduser().resolve(),
        muse_xsl_path=args.muse_xsl_path.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        xsl_template_path=args.xsl_template_path.expanduser().resolve(),
        lsf_table_path=args.lsf_table_path.expanduser().resolve(),
        redshift=float(args.redshift),
        central_radius_arcsec=float(args.central_radius_arcsec),
        muse_spatial_fwhm_arcsec=args.muse_spatial_fwhm_arcsec,
        nirspec_spatial_fwhm_arcsec=float(args.nirspec_spatial_fwhm_arcsec),
        lsf_mode=str(args.lsf_mode),
        resolving_power=float(args.resolving_power),
        fit_windows_rest_um=base.parse_windows(args.fit_windows_rest_um),
        mask_windows_rest_um=base.parse_windows(args.mask_windows_rest_um),
        degree=int(args.degree),
        mdegree=int(args.mdegree),
        pop_mdegree=int(args.pop_mdegree),
        regul_start=float(args.regul_start),
        regul_max=float(args.regul_max),
        regul_bracket_steps=int(args.regul_bracket_steps),
        regul_bisect_steps=int(args.regul_bisect_steps),
        moments=int(args.moments),
        bias=float(args.bias),
        start_sigma=float(args.start_sigma),
        max_abs_velocity=float(args.max_abs_velocity),
        min_sigma=float(args.min_sigma),
        max_sigma=float(args.max_sigma),
        min_wave_finite_frac=float(args.min_wave_finite_frac),
        min_spaxel_finite_frac=float(args.min_spaxel_finite_frac),
        min_log_pixel_fraction=float(args.min_log_pixel_fraction),
        min_goodpixels=int(args.min_goodpixels),
        template_velocity_margin_kms=float(args.template_velocity_margin_kms),
        wave_crpix_mode=str(args.wave_crpix_mode),
        reuse_convolved_cube=bool(args.reuse_convolved_cube),
    )


def main() -> None:
    cfg = parse_args()
    outdir = ensure_dir(cfg.output_dir)

    muse_fwhm = cfg.muse_spatial_fwhm_arcsec
    if muse_fwhm is None:
        muse_fwhm = infer_muse_spatial_fwhm_arcsec(cfg.muse_cube_path)
    if muse_fwhm is None or not np.isfinite(muse_fwhm) or muse_fwhm <= 0:
        raise ValueError("Could not infer MUSE spatial FWHM; pass --muse-spatial-fwhm-arcsec")

    conv_stem = f"{cfg.cube_path.stem}_muse_spatial_fwhm{muse_fwhm:.2f}arcsec".replace(".", "p")
    convolved_cube_path = outdir / f"{conv_stem}.fits"
    if cfg.reuse_convolved_cube and convolved_cube_path.is_file():
        with fits.open(convolved_cube_path, memmap=False) as hdul:
            sci_idx = find_science_hdu_index(hdul)
            sigma_pix = float(hdul[sci_idx].header.get("KERSIGPX", np.nan))
        print(f"Reusing convolved cube: {convolved_cube_path}")
    else:
        print(
            f"Convolving {cfg.cube_path.name} to MUSE FWHM={muse_fwhm:.3f} arcsec "
            f"(native NIRSpec FWHM={cfg.nirspec_spatial_fwhm_arcsec:.3f} arcsec)"
        )
        convolved_cube_path, sigma_pix = convolve_cube_to_fwhm(
            cfg.cube_path,
            convolved_cube_path,
            target_fwhm_arcsec=float(muse_fwhm),
            native_fwhm_arcsec=cfg.nirspec_spatial_fwhm_arcsec,
        )
        print(f"Saved convolved cube: {convolved_cube_path}")

    fit_cfg = make_base_config(cfg, convolved_cube_path)
    print(f"Reading convolved cube for pPXF: {convolved_cube_path}")
    cube_data = base.read_cube_data(fit_cfg)
    print(
        f"Loaded {cube_data.valid_spaxel_indices.size} valid spaxels; "
        f"velscale={cube_data.velscale:.3f} km/s; pixel={cube_data.pixsize_arcsec:.3f} arcsec"
    )

    print(f"Loading XSL templates: {cfg.xsl_template_path}")
    library = load_xsl_library(cfg, cube_data)
    print(
        f"Using {library.template_count} XSL templates; "
        f"NIRSpec R median={library.data_r_med:.1f}; "
        f"data FWHM median={library.data_fwhm_med_ang:.3f} A"
    )

    result, pop_result, nirspec_summary, galaxy, fit_mask, central_idx = fit_central_nirspec(cfg, cube_data, library)
    print(
        f"NIRSpec central fit: ok={result.ok}, V={nirspec_summary['v_rel_kms']:.2f}, "
        f"sigma={nirspec_summary['sigma_kms']:.2f}, S/N={nirspec_summary['sn']:.1f}, "
        f"population ok={nirspec_summary['pop_ok']}"
    )

    muse_summary = muse_central_summary(cfg.muse_xsl_path, cfg.central_radius_arcsec)

    fit_plot = outdir / "nirspec_convolved_xsl_central_fit.png"
    plot_fit(
        fit_plot,
        cube_data.lam_gal_ang,
        galaxy,
        result,
        fit_mask,
        title=(
            f"NIRSpec AGN-sub XSL central R<={cfg.central_radius_arcsec:.1f}\" "
            f"after MUSE PSF match | sigma={nirspec_summary['sigma_kms']:.1f} km/s"
        ),
    )

    comparison_csv = outdir / "nirspec_muse_central_2arcsec_xsl_comparison.csv"
    write_comparison_csv(comparison_csv, nirspec_summary, muse_summary)
    comparison_plot = outdir / "nirspec_muse_central_2arcsec_xsl_comparison.png"
    plot_comparison(comparison_plot, nirspec_summary, muse_summary)

    run_config = asdict(cfg)
    for key, val in list(run_config.items()):
        if isinstance(val, Path):
            run_config[key] = str(val)
        elif isinstance(val, tuple):
            run_config[key] = list(val)

    payload = {
        "run_config": run_config,
        "muse_spatial_fwhm_arcsec_used": float(muse_fwhm),
        "nirspec_kernel_sigma_pix": float(sigma_pix) if np.isfinite(sigma_pix) else None,
        "convolved_cube_path": str(convolved_cube_path),
        "xsl_library": {
            "template_path": str(library.template_path),
            "template_count": int(library.template_count),
            "reg_dim": list(library.reg_dim),
            "data_r_min": library.data_r_min,
            "data_r_med": library.data_r_med,
            "data_r_max": library.data_r_max,
            "data_fwhm_med_ang": library.data_fwhm_med_ang,
            "xsl_fwhm_med_ang": library.xsl_fwhm_med_ang,
        },
        "nirspec_central": nirspec_summary,
        "muse_central": muse_summary,
        "outputs": {
            "fit_plot": str(fit_plot),
            "comparison_csv": str(comparison_csv),
            "comparison_plot": str(comparison_plot),
        },
    }
    json_path = outdir / "nirspec_muse_central_2arcsec_xsl_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    summary_path = outdir / "run_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "NIRSpec AGN-subtracted XSL pPXF after MUSE spatial PSF matching",
                f"Input cube: {cfg.cube_path}",
                f"Convolved cube: {convolved_cube_path}",
                f"MUSE XSL product: {cfg.muse_xsl_path}",
                f"Central aperture radius: {cfg.central_radius_arcsec:.3f} arcsec",
                f"MUSE FWHM used: {muse_fwhm:.3f} arcsec",
                f"Assumed native NIRSpec FWHM: {cfg.nirspec_spatial_fwhm_arcsec:.3f} arcsec",
                f"Kernel sigma: {sigma_pix:.3f} NIRSpec spaxels" if np.isfinite(sigma_pix) else "Kernel sigma: reused",
                f"pPXF setup: XSL, moments={cfg.moments}, degree={cfg.degree}, mdegree={cfg.mdegree}, pop_mdegree={cfg.pop_mdegree}, bias={cfg.bias}",
                f"NIRSpec population regularization: regul={nirspec_summary['pop_regul']:.3g}, target_dchi2={nirspec_summary['pop_target_dchi2']:.2f}, achieved_dchi2={nirspec_summary['pop_achieved_dchi2']:.2f}",
                f"NIRSpec central spaxels: {nirspec_summary['n_spaxels']}",
                f"MUSE central spaxels: {muse_summary['n_spaxels']} across {muse_summary['n_bins']} bins",
                f"NIRSpec V/sigma/vrms: {nirspec_summary['v_rel_kms']:.2f}, {nirspec_summary['sigma_kms']:.2f}, {nirspec_summary['vrms_kms']:.2f} km/s",
                f"MUSE weighted V/sigma/vrms: {muse_summary['v_rel_kms_weighted_mean']:.2f}, {muse_summary['sigma_kms_weighted_mean']:.2f}, {muse_summary['vrms_kms_weighted_mean']:.2f} km/s",
                f"Comparison CSV: {comparison_csv}",
                f"Comparison JSON: {json_path}",
                f"Fit plot: {fit_plot}",
                f"Comparison plot: {comparison_plot}",
            ]
        )
        + "\n"
    )

    print(f"Saved fit plot       : {fit_plot}")
    print(f"Saved comparison CSV : {comparison_csv}")
    print(f"Saved comparison JSON: {json_path}")
    print(f"Saved comparison plot: {comparison_plot}")
    print(f"Saved run summary    : {summary_path}")


if __name__ == "__main__":
    main()
