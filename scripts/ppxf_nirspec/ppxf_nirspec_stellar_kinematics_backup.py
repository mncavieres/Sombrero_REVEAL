#!/usr/bin/env python3
"""
Stellar-kinematics pPXF workflow for JWST/NIRSpec IFU cubes.

This script is adapted from the working MUSE workflow in
`scripts/ppxf/ppxf_refactored_full_musecube.py`, but it is specialized to
NIRSpec IFU data and focused on stellar kinematics only:

    - build a high-S/N global E-MILES optimal template
    - fit each valid spaxel for [V, sigma, h3, h4]
    - write JAM-friendly CSV, FITS, NPZ, and diagnostic plots

The defaults target the AGN-subtracted Sombrero G235H cube:
`Data/IFU/david_subs/g235h_agn_sub.fits`
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from urllib import request

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib


C = 299792.458  # km/s
GAUSS_FWHM_PER_SIGMA = 2.35482004503

DEFAULT_CUBE_PATH = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/david_subs/g235h_agn_sub.fits"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/ppxf_nirspec/agn_substracted_david"
)
DEFAULT_LSF_TABLE_PATH = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/scripts/ppxf_nirspec/jwst_nirspec_g235h_disp.fits"
)


@dataclass(frozen=True)
class Config:
    cube_path: Path
    output_dir: Path
    redshift: float
    sps_name: str
    lsf_mode: str
    lsf_table_path: Path | None
    resolving_power: float
    fit_windows_rest_um: tuple[tuple[float, float], ...]
    mask_windows_rest_um: tuple[tuple[float, float], ...]
    min_wave_finite_frac: float
    min_spaxel_finite_frac: float
    min_log_pixel_fraction: float
    top_template_frac: float
    degree: int
    mdegree: int
    moments: int
    bias: float
    start_sigma: float
    max_abs_velocity: float
    min_sigma: float
    max_sigma: float
    min_goodpixels: int
    n_plot_spaxels: int
    csv_min_sn: float


@dataclass
class CubeData:
    cube_path: Path
    header: fits.Header
    cube_shape: tuple[int, int, int]
    map_shape: tuple[int, int]
    pixsize_arcsec: float
    wave_obs_um: np.ndarray
    wave_rest_um: np.ndarray
    wave_rest_ang: np.ndarray
    spectra_log: np.ndarray
    valid_frac_log: np.ndarray
    ln_lam_gal: np.ndarray
    lam_gal_ang: np.ndarray
    velscale: float
    fit_mask_log: np.ndarray
    x: np.ndarray
    y: np.ndarray
    row: np.ndarray
    col: np.ndarray
    signal: np.ndarray
    noise_proxy: np.ndarray
    spaxel_finite_frac: np.ndarray
    valid_spaxel_indices: np.ndarray
    valid_spaxel_mask: np.ndarray
    signal_map: np.ndarray
    noise_proxy_map: np.ndarray
    center_row: int
    center_col: int


@dataclass(frozen=True)
class ResolutionInfo:
    lsf_mode: str
    lsf_source: str
    resolving_power_min: float
    resolving_power_med: float
    resolving_power_max: float
    sigma_inst_kms: float
    sigma_inst_min_kms: float
    sigma_inst_max_kms: float
    sigma_template_kms: float
    sigma_template_eff_kms: float
    template_r_eff: float
    template_broader_than_data: bool
    template_broader_fraction: float


@dataclass
class FitResult:
    ok: bool
    message: str
    goodpixels: np.ndarray | None
    clean_mask: np.ndarray | None
    sol_rel: np.ndarray | None
    err_rel: np.ndarray | None
    losv_abs: float
    losv_err: float
    sigma_raw: float
    sigma_raw_err: float
    sigma: float
    sigma_err: float
    vrms: float
    vrms_err: float
    h3: float
    h3_err: float
    h4: float
    h4_err: float
    sn: float
    chi2: float
    bestfit: np.ndarray | None
    noise_vector: np.ndarray | None


def parse_windows(text: str) -> tuple[tuple[float, float], ...]:
    text = text.strip()
    if not text:
        return ()

    windows: list[tuple[float, float]] = []
    for chunk in text.split(","):
        lo_str, hi_str = chunk.strip().split("-", maxsplit=1)
        lo = float(lo_str)
        hi = float(hi_str)
        if hi <= lo:
            raise ValueError(f"Invalid wavelength window '{chunk}'")
        windows.append((lo, hi))
    return tuple(windows)


def parse_optional_path(text: str | None) -> Path | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    if text.lower() == "none":
        return None
    return Path(text).expanduser().resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def mad_std(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis)
    return 1.4826 * mad


def estimate_noise_from_differences(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    diff = np.diff(arr, axis=axis)
    return mad_std(diff, axis=axis) / np.sqrt(2.0)


def safe_positive(a: np.ndarray, fill_value: float | None = None) -> np.ndarray:
    out = np.asarray(a, dtype=float).copy()
    bad = ~np.isfinite(out) | (out <= 0)
    if np.any(bad):
        if fill_value is None:
            good = out[~bad]
            fill_value = float(np.nanmedian(good)) if good.size else 1.0
        out[bad] = fill_value
    return out


def estimate_spectrum_noise(galaxy: np.ndarray, mask: np.ndarray | None = None) -> float:
    galaxy = np.asarray(galaxy, dtype=float)
    if mask is None:
        x = galaxy[np.isfinite(galaxy)]
    else:
        x = galaxy[np.isfinite(galaxy) & mask]
    if x.size < 10:
        x = galaxy[np.isfinite(galaxy)]
    if x.size < 10:
        return 1.0

    noise = estimate_noise_from_differences(x, axis=0)
    if not np.isfinite(noise) or noise <= 0:
        noise = robust_sigma(x, zero=1)
    if not np.isfinite(noise) or noise <= 0:
        noise = np.nanstd(x)
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.0
    return float(noise)


def correct_ppxf_errors(pp) -> np.ndarray | None:
    if getattr(pp, "error", None) is None:
        return None
    return np.asarray(pp.error, dtype=float) * np.sqrt(pp.chi2)


def compute_vrms_and_error(
    vrel: float,
    vrel_err: float,
    sigma: float,
    sigma_err: float,
) -> tuple[float, float]:
    vrms = float(np.sqrt(vrel**2 + sigma**2))
    numer = float(np.sqrt((vrel * vrel_err) ** 2 + (sigma * sigma_err) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        vrms_err = numer / vrms if vrms > 0 else np.nan
    return vrms, vrms_err


def apply_resolution_correction(
    sigma_raw: float,
    sigma_raw_err: float,
    resolution_info: ResolutionInfo,
) -> tuple[float, float]:
    corr2 = sigma_raw**2 + resolution_info.sigma_template_eff_kms**2 - resolution_info.sigma_inst_kms**2
    sigma_corr = float(np.sqrt(max(corr2, 0.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_corr_err = float(sigma_raw * sigma_raw_err / sigma_corr) if sigma_corr > 0 else np.nan
    return sigma_corr, sigma_corr_err


def clip_outliers(galaxy: np.ndarray, bestfit: np.ndarray, mask: np.ndarray) -> np.ndarray:
    good = mask.copy()
    while True:
        scale = galaxy[good] @ bestfit[good] / np.sum(bestfit[good] ** 2)
        resid = galaxy[good] - scale * bestfit[good]
        err = robust_sigma(resid, zero=1)
        new_good = good.copy()
        new_good[good] = np.abs(resid) < 3 * err
        if np.array_equal(new_good, good):
            break
        good = new_good
    return good


def orient_cube_nlam_first(arr: np.ndarray, header: fits.Header) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {arr.shape}")

    nlam = header.get("NAXIS3")
    if nlam is None:
        return arr
    if arr.shape[0] == nlam:
        return arr
    if arr.shape[-1] == nlam:
        return np.moveaxis(arr, -1, 0)
    return arr


def find_science_hdu(hdul: fits.HDUList):
    for name in ("SCI", "DATA"):
        if name in hdul:
            return hdul[name]
    for hdu in hdul:
        if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) == 3:
            return hdu
    raise ValueError("Could not find a 3D science cube")


def find_error_cube(hdul: fits.HDUList) -> np.ndarray | None:
    for extname in ("ERR", "ERROR", "SIGMA", "VAR", "IVAR"):
        if extname not in hdul:
            continue
        hdu = hdul[extname]
        arr = orient_cube_nlam_first(np.asarray(hdu.data, dtype=float), hdu.header)
        if extname in ("ERR", "ERROR", "SIGMA"):
            return np.clip(arr, 0, None)
        if extname == "VAR":
            return np.sqrt(np.clip(arr, 0, None))
        if extname == "IVAR":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.sqrt(np.divide(1.0, arr, where=arr > 0))
    return None


def find_dq_cube(hdul: fits.HDUList) -> np.ndarray | None:
    if "DQ" in hdul:
        hdu = hdul["DQ"]
        return orient_cube_nlam_first(np.asarray(hdu.data), hdu.header)
    return None


def derive_wave_axis_um(header: fits.Header, nlam: int) -> np.ndarray:
    cd3 = header.get("CD3_3", header.get("CDELT3"))
    crval3 = header.get("CRVAL3")
    if cd3 is None or crval3 is None:
        raise KeyError("Could not determine wavelength solution from FITS header")
    return crval3 + cd3 * np.arange(nlam, dtype=float)


def derive_pixsize_arcsec(header: fits.Header) -> float:
    if "CD1_1" in header:
        return abs(float(header["CD1_1"])) * 3600.0
    if "CDELT1" in header:
        return abs(float(header["CDELT1"])) * 3600.0
    if "PIXAR_A2" in header:
        return float(np.sqrt(header["PIXAR_A2"]))
    raise KeyError("Could not determine spatial pixel size from FITS header")


def build_wave_mask(values: np.ndarray, windows: tuple[tuple[float, float], ...]) -> np.ndarray:
    mask = np.zeros_like(values, dtype=bool)
    for lo, hi in windows:
        mask |= (values >= lo) & (values <= hi)
    return mask


def fill_invalid_spectrum(spec: np.ndarray) -> np.ndarray:
    spec = np.asarray(spec, dtype=float)
    good = np.isfinite(spec)
    if good.all():
        return spec
    if np.count_nonzero(good) < 2:
        fill = np.nanmedian(spec[good]) if np.any(good) else 0.0
        return np.full_like(spec, fill, dtype=float)

    idx = np.flatnonzero(good)
    return np.interp(np.arange(spec.size), idx, spec[idx])


def select_largest_blob(mask2d: np.ndarray) -> np.ndarray:
    labels, nlab = ndimage.label(mask2d.astype(bool))
    if nlab == 0:
        return np.zeros_like(mask2d, dtype=bool)
    sizes = np.bincount(labels.ravel())[1:]
    label = int(np.argmax(sizes) + 1)
    return labels == label


def read_cube_data(cfg: Config) -> CubeData:
    with fits.open(cfg.cube_path, memmap=False) as hdul:
        sci_hdu = find_science_hdu(hdul)
        cube = orient_cube_nlam_first(np.asarray(sci_hdu.data, dtype=float), sci_hdu.header)
        err_cube = find_error_cube(hdul)
        dq_cube = find_dq_cube(hdul)
        header = sci_hdu.header.copy()

    nlam, ny, nx = cube.shape
    wave_obs_um = derive_wave_axis_um(header, nlam)
    wave_rest_um = wave_obs_um / (1.0 + cfg.redshift)
    wave_rest_ang = wave_rest_um * 1e4
    pixsize_arcsec = derive_pixsize_arcsec(header)

    fit_mask_lin = build_wave_mask(wave_rest_um, cfg.fit_windows_rest_um)
    finite_frac_wave = np.mean(np.isfinite(cube[fit_mask_lin]), axis=(1, 2))
    wave_fit_finite_ok = np.zeros_like(wave_rest_um, dtype=bool)
    wave_fit_finite_ok[fit_mask_lin] = finite_frac_wave >= cfg.min_wave_finite_frac
    wave_mask_lin = build_wave_mask(wave_rest_um, cfg.fit_windows_rest_um)
    if cfg.mask_windows_rest_um:
        wave_mask_lin &= ~build_wave_mask(wave_rest_um, cfg.mask_windows_rest_um)
    wave_mask_lin &= wave_fit_finite_ok

    contig_lo = min(lo for lo, _ in cfg.fit_windows_rest_um)
    contig_hi = max(hi for _, hi in cfg.fit_windows_rest_um)
    contig_lin = (wave_rest_um >= contig_lo) & (wave_rest_um <= contig_hi)
    if np.count_nonzero(contig_lin) < cfg.min_goodpixels:
        raise ValueError("Contiguous fit window contains too few pixels")

    cube_contig = cube[contig_lin, :, :]
    valid_cube = np.isfinite(cube_contig)

    signal_map = np.nanmedian(cube[wave_mask_lin], axis=0)
    if err_cube is not None:
        noise_proxy_map = np.nanmedian(err_cube[fit_mask_lin], axis=0)
    else:
        noise_proxy_map = estimate_noise_from_differences(cube[fit_mask_lin], axis=0)
    noise_proxy_map = safe_positive(np.asarray(noise_proxy_map, dtype=float))

    spaxel_finite_frac = np.mean(np.isfinite(cube[wave_mask_lin]), axis=0)
    valid_spaxel_mask = (
        np.isfinite(signal_map)
        & np.isfinite(noise_proxy_map)
        & (noise_proxy_map > 0)
        & (spaxel_finite_frac >= cfg.min_spaxel_finite_frac)
    )

    if dq_cube is not None:
        dq_frac = np.mean(np.asarray(dq_cube[fit_mask_lin]) != 0, axis=0)
        valid_spaxel_mask &= dq_frac < 0.5

    valid_spaxel_mask = select_largest_blob(valid_spaxel_mask)
    valid_spaxel_indices = np.flatnonzero(valid_spaxel_mask.ravel())
    if valid_spaxel_indices.size == 0:
        raise ValueError("No valid NIRSpec spaxels survived the selection cuts")

    center_index = int(np.nanargmax(np.where(valid_spaxel_mask, signal_map, -np.inf)))
    center_row, center_col = np.unravel_index(center_index, signal_map.shape)
    row2d, col2d = np.indices((ny, nx))

    x_all = (col2d - center_col) * pixsize_arcsec
    y_all = (row2d - center_row) * pixsize_arcsec

    spectra_lin = cube_contig.reshape(np.count_nonzero(contig_lin), -1)[:, valid_spaxel_indices]
    valid_lin = valid_cube.reshape(np.count_nonzero(contig_lin), -1)[:, valid_spaxel_indices]
    spectra_lin_filled = np.empty_like(spectra_lin)
    for j in range(spectra_lin.shape[1]):
        spectra_lin_filled[:, j] = fill_invalid_spectrum(spectra_lin[:, j])

    lam_range_contig_ang = [
        float(wave_rest_ang[contig_lin][0]),
        float(wave_rest_ang[contig_lin][-1]),
    ]
    velscale0 = float(np.min(C * np.diff(np.log(wave_rest_ang[contig_lin]))))
    spectra_log, ln_lam_gal, velscale = util.log_rebin(
        lam_range_contig_ang,
        spectra_lin_filled,
        velscale=velscale0,
    )
    valid_frac_log, _, _ = util.log_rebin(
        lam_range_contig_ang,
        valid_lin.astype(float),
        velscale=velscale0,
        flux=False,
    )
    fit_mask_log, _, _ = util.log_rebin(
        lam_range_contig_ang,
        wave_mask_lin[contig_lin].astype(float),
        velscale=velscale0,
        flux=False,
    )
    fit_mask_log = np.asarray(fit_mask_log > 0.5, dtype=bool)
    lam_gal_ang = np.exp(ln_lam_gal)

    return CubeData(
        cube_path=cfg.cube_path,
        header=header,
        cube_shape=cube.shape,
        map_shape=(ny, nx),
        pixsize_arcsec=pixsize_arcsec,
        wave_obs_um=wave_obs_um,
        wave_rest_um=wave_rest_um,
        wave_rest_ang=wave_rest_ang,
        spectra_log=np.asarray(spectra_log, dtype=float),
        valid_frac_log=np.asarray(valid_frac_log, dtype=float),
        ln_lam_gal=np.asarray(ln_lam_gal, dtype=float),
        lam_gal_ang=np.asarray(lam_gal_ang, dtype=float),
        velscale=float(velscale),
        fit_mask_log=fit_mask_log,
        x=x_all.ravel()[valid_spaxel_indices],
        y=y_all.ravel()[valid_spaxel_indices],
        row=row2d.ravel()[valid_spaxel_indices] + 1,
        col=col2d.ravel()[valid_spaxel_indices] + 1,
        signal=signal_map.ravel()[valid_spaxel_indices],
        noise_proxy=noise_proxy_map.ravel()[valid_spaxel_indices],
        spaxel_finite_frac=spaxel_finite_frac.ravel()[valid_spaxel_indices],
        valid_spaxel_indices=valid_spaxel_indices,
        valid_spaxel_mask=valid_spaxel_mask,
        signal_map=np.asarray(signal_map, dtype=float),
        noise_proxy_map=np.asarray(noise_proxy_map, dtype=float),
        center_row=int(center_row),
        center_col=int(center_col),
    )


def load_sps_library(
    cfg: Config,
    lam_min_ang: float,
    lam_max_ang: float,
    velscale: float,
) -> tuple[lib.sps_lib, tuple[int, ...], ResolutionInfo]:
    ppxf_dir = resources.files("ppxf")
    basename = f"spectra_{cfg.sps_name}_9.0.npz"
    filename = ppxf_dir / "sps_models" / basename
    if not filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, filename)

    with np.load(filename) as a:
        lam_native = np.asarray(a["lam"], dtype=float)
        fwhm_tem_native = np.asarray(a["fwhm"], dtype=float)
    band = (lam_native >= lam_min_ang) & (lam_native <= lam_max_ang)
    if not np.any(band):
        raise ValueError("Template library does not overlap the requested wavelength range")
    lam_band = lam_native[band]
    fwhm_tem_band = fwhm_tem_native[band]

    if cfg.lsf_mode == "fixed":
        resolving_power_native = np.full_like(lam_native, cfg.resolving_power, dtype=float)
        lsf_source = f"fixed R={cfg.resolving_power:.2f}"
    elif cfg.lsf_mode == "table":
        if cfg.lsf_table_path is None:
            raise ValueError("lsf_mode='table' requires --lsf-table-path")
        with fits.open(cfg.lsf_table_path) as hdul:
            tab = hdul[1].data
            wave_curve_um = np.asarray(tab["WAVELENGTH"], dtype=float)
            resolving_power_curve = np.asarray(tab["R"], dtype=float)
        obs_band_um = lam_band / 1e4 * (1.0 + cfg.redshift)
        if obs_band_um.min() < wave_curve_um.min() or obs_band_um.max() > wave_curve_um.max():
            raise ValueError(
                "Requested fit window falls outside the wavelength coverage of the LSF table: "
                f"{cfg.lsf_table_path}"
            )
        obs_native_um = lam_native / 1e4 * (1.0 + cfg.redshift)
        resolving_power_native = np.interp(
            obs_native_um,
            wave_curve_um,
            resolving_power_curve,
            left=resolving_power_curve[0],
            right=resolving_power_curve[-1],
        )
        lsf_source = str(cfg.lsf_table_path)
    else:
        raise ValueError(f"Unsupported lsf mode '{cfg.lsf_mode}'")

    resolving_power_band = resolving_power_native[band]
    fwhm_gal_native = lam_native / resolving_power_native
    fwhm_gal_band = fwhm_gal_native[band]
    fwhm_eff_band = np.maximum(fwhm_tem_band, fwhm_gal_band)
    sigma_inst_band = C * (fwhm_gal_band / lam_band) / GAUSS_FWHM_PER_SIGMA
    sigma_tem_band = C * (fwhm_tem_band / lam_band) / GAUSS_FWHM_PER_SIGMA
    sigma_eff_band = C * (fwhm_eff_band / lam_band) / GAUSS_FWHM_PER_SIGMA
    resolution_info = ResolutionInfo(
        lsf_mode=cfg.lsf_mode,
        lsf_source=lsf_source,
        resolving_power_min=float(np.nanmin(resolving_power_band)),
        resolving_power_med=float(np.nanmedian(resolving_power_band)),
        resolving_power_max=float(np.nanmax(resolving_power_band)),
        sigma_inst_kms=float(np.nanmedian(sigma_inst_band)),
        sigma_inst_min_kms=float(np.nanmin(sigma_inst_band)),
        sigma_inst_max_kms=float(np.nanmax(sigma_inst_band)),
        sigma_template_kms=float(np.nanmedian(sigma_tem_band)),
        sigma_template_eff_kms=float(np.nanmedian(sigma_eff_band)),
        template_r_eff=float(np.nanmedian(lam_band / fwhm_tem_band)),
        template_broader_than_data=bool(np.any(fwhm_tem_band > fwhm_gal_band)),
        template_broader_fraction=float(np.mean(fwhm_tem_band > fwhm_gal_band)),
    )

    fwhm_gal = {
        "lam": lam_native,
        "fwhm": fwhm_gal_native,
    }
    lam_range = [lam_min_ang - 500.0, lam_max_ang + 500.0]
    sps = lib.sps_lib(filename, velscale, fwhm_gal, lam_range=lam_range)

    npix, *reg_dim = sps.templates.shape
    sps.templates /= np.nanmedian(sps.templates)
    sps.templates = sps.templates.reshape(npix, -1)
    return sps, tuple(reg_dim), resolution_info


def fit_moments2(
    templates: np.ndarray,
    galaxy: np.ndarray,
    velscale: float,
    start: list[float],
    fit_mask: np.ndarray,
    lam: np.ndarray,
    lam_temp: np.ndarray,
    degree: int,
    mdegree: int,
):
    mask = fit_mask.copy()
    noise0 = np.full_like(galaxy, estimate_spectrum_noise(galaxy, mask=mask))
    pp = ppxf(
        templates,
        galaxy,
        noise0,
        velscale,
        start,
        moments=2,
        degree=degree,
        mdegree=mdegree,
        lam=lam,
        lam_temp=lam_temp,
        mask=mask,
        quiet=True,
    )

    mask = clip_outliers(galaxy, pp.bestfit, mask) & fit_mask
    resid = galaxy[mask] - pp.bestfit[mask]
    noise = np.full_like(galaxy, robust_sigma(resid, zero=1))
    noise = safe_positive(noise)

    pp = ppxf(
        templates,
        galaxy,
        noise,
        velscale,
        pp.sol,
        moments=2,
        degree=degree,
        mdegree=mdegree,
        lam=lam,
        lam_temp=lam_temp,
        mask=mask,
        quiet=True,
    )
    noise *= np.sqrt(pp.chi2)
    pp = ppxf(
        templates,
        galaxy,
        noise,
        velscale,
        pp.sol,
        moments=2,
        degree=degree,
        mdegree=mdegree,
        lam=lam,
        lam_temp=lam_temp,
        mask=mask,
        quiet=True,
    )

    pp.clean_mask = mask
    pp.noise_vector = noise
    pp.error_corr = correct_ppxf_errors(pp)
    if np.ndim(templates) == 1:
        pp.optimal_template = np.asarray(templates, dtype=float)
    else:
        pp.optimal_template = templates @ pp.weights
    resid = (pp.galaxy - pp.bestfit)[pp.goodpixels]
    pp.sn = np.nanmedian(pp.galaxy[pp.goodpixels]) / robust_sigma(resid, zero=1)
    return pp


def fit_spaxel_kinematics(
    template: np.ndarray,
    lam_temp_template: np.ndarray,
    galaxy: np.ndarray,
    valid_frac_log: np.ndarray,
    cube_data: CubeData,
    resolution_info: ResolutionInfo,
    cfg: Config,
    start_sigma: float,
) -> FitResult:
    fit_mask = cube_data.fit_mask_log & (valid_frac_log >= cfg.min_log_pixel_fraction)
    if np.count_nonzero(fit_mask) < cfg.min_goodpixels:
        return FitResult(
            ok=False,
            message="too_few_goodpixels",
            goodpixels=None,
            clean_mask=fit_mask,
            sol_rel=None,
            err_rel=None,
            losv_abs=np.nan,
            losv_err=np.nan,
            sigma_raw=np.nan,
            sigma_raw_err=np.nan,
            sigma=np.nan,
            sigma_err=np.nan,
            vrms=np.nan,
            vrms_err=np.nan,
            h3=np.nan,
            h3_err=np.nan,
            h4=np.nan,
            h4_err=np.nan,
            sn=np.nan,
            chi2=np.nan,
            bestfit=None,
            noise_vector=None,
        )

    try:
        pp2 = fit_moments2(
            template,
            galaxy,
            cube_data.velscale,
            [0.0, start_sigma],
            fit_mask,
            cube_data.lam_gal_ang,
            lam_temp_template,
            degree=cfg.degree,
            mdegree=cfg.mdegree,
        )
    except Exception as exc:  # pragma: no cover - fit failures are data-dependent
        return FitResult(
            ok=False,
            message=f"moments2_failed:{exc}",
            goodpixels=None,
            clean_mask=fit_mask,
            sol_rel=None,
            err_rel=None,
            losv_abs=np.nan,
            losv_err=np.nan,
            sigma_raw=np.nan,
            sigma_raw_err=np.nan,
            sigma=np.nan,
            sigma_err=np.nan,
            vrms=np.nan,
            vrms_err=np.nan,
            h3=np.nan,
            h3_err=np.nan,
            h4=np.nan,
            h4_err=np.nan,
            sn=np.nan,
            chi2=np.nan,
            bestfit=None,
            noise_vector=None,
        )

    mask = pp2.clean_mask.copy()
    startn = [float(pp2.sol[0]), float(pp2.sol[1])]
    bounds = [
        [-cfg.max_abs_velocity, cfg.max_abs_velocity],
        [cfg.min_sigma, cfg.max_sigma],
    ]
    if cfg.moments > 2:
        startn.extend([0.0] * (cfg.moments - 2))
        bounds.extend([[-0.3, 0.3]] * (cfg.moments - 2))
    noise = pp2.noise_vector.copy()

    try:
        pp4 = ppxf(
            template,
            galaxy,
            noise,
            cube_data.velscale,
            startn,
            bias=cfg.bias,
            bounds=bounds,
            moments=cfg.moments,
            degree=cfg.degree,
            mdegree=cfg.mdegree,
            lam=cube_data.lam_gal_ang,
            lam_temp=lam_temp_template,
            mask=mask,
            quiet=True,
        )
        noise *= np.sqrt(pp4.chi2)
        pp4 = ppxf(
            template,
            galaxy,
            noise,
            cube_data.velscale,
            pp4.sol,
            bias=cfg.bias,
            bounds=bounds,
            moments=cfg.moments,
            degree=cfg.degree,
            mdegree=cfg.mdegree,
            lam=cube_data.lam_gal_ang,
            lam_temp=lam_temp_template,
            mask=mask,
            quiet=True,
        )
    except Exception as exc:  # pragma: no cover - fit failures are data-dependent
        return FitResult(
            ok=False,
            message=f"moments4_failed:{exc}",
            goodpixels=getattr(pp2, "goodpixels", None),
            clean_mask=mask,
            sol_rel=None,
            err_rel=None,
            losv_abs=np.nan,
            losv_err=np.nan,
            sigma_raw=np.nan,
            sigma_raw_err=np.nan,
            sigma=np.nan,
            sigma_err=np.nan,
            vrms=np.nan,
            vrms_err=np.nan,
            h3=np.nan,
            h3_err=np.nan,
            h4=np.nan,
            h4_err=np.nan,
            sn=np.nan,
            chi2=np.nan,
            bestfit=None,
            noise_vector=None,
        )

    err_rel = correct_ppxf_errors(pp4)
    sol_rel = np.asarray(pp4.sol, dtype=float)
    redshift_fit = (1.0 + cfg.redshift) * np.exp(sol_rel[0] / C) - 1.0
    losv_abs = C * redshift_fit
    losv_err = (1.0 + redshift_fit) * float(err_rel[0]) if err_rel is not None else np.nan
    sigma_raw = float(sol_rel[1])
    sigma_raw_err = float(err_rel[1]) if err_rel is not None else np.nan
    sigma, sigma_err = apply_resolution_correction(sigma_raw, sigma_raw_err, resolution_info)
    vrel = float(sol_rel[0])
    vrel_err = float(err_rel[0]) if err_rel is not None else np.nan
    vrms, vrms_err = compute_vrms_and_error(vrel, vrel_err, sigma, sigma_err)

    resid = (pp4.galaxy - pp4.bestfit)[pp4.goodpixels]
    sn = np.nanmedian(pp4.galaxy[pp4.goodpixels]) / robust_sigma(resid, zero=1)

    return FitResult(
        ok=True,
        message="ok",
        goodpixels=np.asarray(pp4.goodpixels, dtype=int),
        clean_mask=mask,
        sol_rel=sol_rel,
        err_rel=err_rel,
        losv_abs=float(losv_abs),
        losv_err=float(losv_err),
        sigma_raw=sigma_raw,
        sigma_raw_err=sigma_raw_err,
        sigma=sigma,
        sigma_err=sigma_err,
        vrms=vrms,
        vrms_err=vrms_err,
        h3=float(sol_rel[2]) if sol_rel.size > 2 else np.nan,
        h3_err=float(err_rel[2]) if err_rel is not None and err_rel.size > 2 else np.nan,
        h4=float(sol_rel[3]) if sol_rel.size > 3 else np.nan,
        h4_err=float(err_rel[3]) if err_rel is not None and err_rel.size > 3 else np.nan,
        sn=float(sn),
        chi2=float(pp4.chi2),
        bestfit=np.asarray(pp4.bestfit, dtype=float),
        noise_vector=np.asarray(noise, dtype=float),
    )


def plot_global_fit(
    outdir: Path,
    lam_ang: np.ndarray,
    galaxy: np.ndarray,
    bestfit: np.ndarray,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(lam_ang, galaxy, lw=0.8, label="Global spectrum")
    ax.plot(lam_ang, bestfit, lw=1.0, label="pPXF best fit")
    ax.set_title(title)
    ax.set_xlabel("Rest wavelength [Angstrom]")
    ax.set_ylabel("Flux [MJy/sr]")
    ax.set_xlim(float(np.min(lam_ang)), float(np.max(lam_ang)))
    ax.legend(loc="upper right")
    outpath = outdir / "global_template_fit.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_spaxel_fit(outdir: Path, lam_ang: np.ndarray, galaxy: np.ndarray, result: FitResult, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(lam_ang, galaxy, lw=0.8, label="Galaxy")
    if result.bestfit is not None:
        ax.plot(lam_ang, result.bestfit, lw=1.0, label="pPXF best fit")
    ax.set_xlabel("Rest wavelength [Angstrom]")
    ax.set_ylabel("Flux [MJy/sr]")
    ax.set_title(name)
    ax.legend(loc="upper right")
    outpath = outdir / f"{name}.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_maps(outdir: Path, cube_data: CubeData, maps: dict[str, np.ndarray]) -> Path:
    extent = (
        cube_data.x.min() - 0.5 * cube_data.pixsize_arcsec,
        cube_data.x.max() + 0.5 * cube_data.pixsize_arcsec,
        cube_data.y.min() - 0.5 * cube_data.pixsize_arcsec,
        cube_data.y.max() + 0.5 * cube_data.pixsize_arcsec,
    )

    panels = [
        ("VREL_MAP", "V - systemic [km/s]", "RdBu_r"),
        ("SIGMA_MAP", "sigma [km/s]", "inferno"),
        ("VRMS_MAP", "vrms [km/s]", "magma"),
        ("H3_MAP", "h3", "RdBu_r"),
        ("H4_MAP", "h4", "RdBu_r"),
        ("SN_MAP", "S/N", "viridis"),
        ("GOODFIT_MAP", "goodmask", "gray"),
        ("CHI2_MAP", "chi2", "cividis"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, title, cmap) in zip(axes, panels):
        im = ax.imshow(
            maps[key],
            origin="lower",
            extent=extent,
            cmap=cmap,
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("X [arcsec]")
        ax.set_ylabel("Y [arcsec]")
    for ax in axes[len(panels):]:
        ax.axis("off")
    outpath = outdir / "kinematics_maps.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def save_csv(outpath: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "X",
        "Y",
        "LOSV",
        "LOSV_err",
        "sigma_ppxf",
        "sigma_ppxf_err",
        "sigma",
        "sigma_err",
        "Vrms",
        "Vrms_err",
        "h3",
        "h4",
        "ROW",
        "COL",
        "V_REL_KMS",
        "V_REL_ERR_KMS",
        "H3_ERR",
        "H4_ERR",
        "SN",
        "CHI2",
        "GOODPIX_FRAC",
        "SPAXEL_FINITE_FRAC",
    ]
    with outpath.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_fits_products(
    outpath: Path,
    cfg: Config,
    cube_data: CubeData,
    resolution_info: ResolutionInfo,
    table_rows: list[dict[str, object]],
    maps: dict[str, np.ndarray],
    global_template: np.ndarray,
    global_template_lam: np.ndarray,
    global_bestfit: np.ndarray,
) -> None:
    hdr = fits.Header()
    hdr["OBJECT"] = cfg.cube_path.stem[:68]
    hdr["SPSMOD"] = cfg.sps_name
    hdr["REDSHFT"] = float(cfg.redshift)
    hdr["LSFMODE"] = resolution_info.lsf_mode[:8]
    hdr["RPOWER"] = float(resolution_info.resolving_power_med)
    hdr["RPOWMIN"] = float(resolution_info.resolving_power_min)
    hdr["RPOWMAX"] = float(resolution_info.resolving_power_max)
    hdr["MOMENTS"] = int(cfg.moments)
    hdr["BIAS"] = float(cfg.bias)
    hdr["DEGREE"] = int(cfg.degree)
    hdr["MDEGREE"] = int(cfg.mdegree)
    hdr["PIXSIZE"] = (float(cube_data.pixsize_arcsec), "arcsec")
    hdr["SIGINSMN"] = (float(resolution_info.sigma_inst_min_kms), "km/s")
    hdr["SIGINST"] = (float(resolution_info.sigma_inst_kms), "km/s")
    hdr["SIGINSMX"] = (float(resolution_info.sigma_inst_max_kms), "km/s")
    hdr["SIGTEMP"] = (float(resolution_info.sigma_template_kms), "km/s")
    hdr["SIGTEFF"] = (float(resolution_info.sigma_template_eff_kms), "km/s")
    hdr["RTEMP"] = (float(resolution_info.template_r_eff), "Median E-MILES resolving power")
    hdr["TBROADD"] = int(resolution_info.template_broader_than_data)
    hdr["TBRFRAC"] = float(resolution_info.template_broader_fraction)
    hdr["LAMMIN"] = float(np.min(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["LAMMAX"] = float(np.max(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["CENROW"] = int(cube_data.center_row + 1)
    hdr["CENCOL"] = int(cube_data.center_col + 1)

    cols = []
    for key in (
        "ROW",
        "COL",
        "X",
        "Y",
        "SIGNAL",
        "NOISE_PROXY",
        "SPAXEL_FINITE_FRAC",
        "GOODFIT",
        "GOODPIX_FRAC",
        "V_REL_KMS",
        "V_REL_ERR_KMS",
        "LOSV",
        "LOSV_err",
        "SIGMA_PPXF",
        "SIGMA_PPXF_ERR",
        "sigma",
        "sigma_err",
        "VRMS",
        "VRMS_ERR",
        "h3",
        "H3_ERR",
        "h4",
        "H4_ERR",
        "SN",
        "CHI2",
    ):
        values = np.array([row[key] for row in table_rows])
        if values.dtype.kind in ("U", "S", "O"):
            raise TypeError(f"Unexpected string column '{key}'")
        if values.dtype.kind == "b":
            fmt = "L"
        elif np.issubdtype(values.dtype, np.integer):
            fmt = "J"
        else:
            fmt = "D"
        cols.append(fits.Column(name=key.upper(), format=fmt, array=values))
    kin_hdu = fits.BinTableHDU.from_columns(cols, name="KIN_RESULTS")

    hdus = [
        fits.PrimaryHDU(header=hdr),
        kin_hdu,
        fits.ImageHDU(data=maps["VREL_MAP"].astype(np.float32), name="VREL_MAP"),
        fits.ImageHDU(data=maps["SIGMA_MAP"].astype(np.float32), name="SIGMA_MAP"),
        fits.ImageHDU(data=maps["VRMS_MAP"].astype(np.float32), name="VRMS_MAP"),
        fits.ImageHDU(data=maps["H3_MAP"].astype(np.float32), name="H3_MAP"),
        fits.ImageHDU(data=maps["H4_MAP"].astype(np.float32), name="H4_MAP"),
        fits.ImageHDU(data=maps["SN_MAP"].astype(np.float32), name="SN_MAP"),
        fits.ImageHDU(data=maps["CHI2_MAP"].astype(np.float32), name="CHI2_MAP"),
        fits.ImageHDU(data=maps["GOODFIT_MAP"].astype(np.int16), name="GOODFIT_MAP"),
        fits.ImageHDU(data=cube_data.signal_map.astype(np.float32), name="SIGNAL_MAP"),
        fits.ImageHDU(data=cube_data.noise_proxy_map.astype(np.float32), name="NOISE_MAP"),
        fits.ImageHDU(data=cube_data.lam_gal_ang.astype(np.float32), name="LAMBDA_REST"),
        fits.ImageHDU(data=cube_data.fit_mask_log.astype(np.int16), name="FIT_MASK"),
        fits.ImageHDU(data=global_template.astype(np.float32), name="GLOBAL_TMPL"),
        fits.ImageHDU(data=np.asarray(global_template_lam, dtype=np.float32), name="TMPL_LAMBDA"),
        fits.ImageHDU(data=global_bestfit.astype(np.float32), name="GLOBAL_FIT"),
    ]
    fits.HDUList(hdus).writeto(outpath, overwrite=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Fit NIRSpec IFU stellar kinematics with pPXF",
    )
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--redshift", type=float, default=0.003633)
    parser.add_argument("--sps-name", type=str, default="emiles")
    parser.add_argument("--lsf-mode", choices=("fixed", "table"), default="fixed")
    parser.add_argument("--lsf-table-path", type=str, default=str(DEFAULT_LSF_TABLE_PATH))
    parser.add_argument("--resolving-power", type=float, default=2700.0)
    parser.add_argument("--fit-windows-rest-um", type=str, default="2.10-2.42")
    parser.add_argument(
        "--mask-windows-rest-um",
        type=str,
        default="2.117-2.127,2.161-2.171,2.219-2.228,2.316-2.326",
    )
    parser.add_argument("--min-wave-finite-frac", type=float, default=0.35)
    parser.add_argument("--min-spaxel-finite-frac", type=float, default=0.60)
    parser.add_argument("--min-log-pixel-fraction", type=float, default=0.70)
    parser.add_argument("--top-template-frac", type=float, default=0.10)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--mdegree", type=int, default=0)
    parser.add_argument("--moments", type=int, default=4)
    parser.add_argument("--bias", type=float, default=0.10)
    parser.add_argument("--start-sigma", type=float, default=220.0)
    parser.add_argument("--max-abs-velocity", type=float, default=600.0)
    parser.add_argument("--min-sigma", type=float, default=30.0)
    parser.add_argument("--max-sigma", type=float, default=500.0)
    parser.add_argument("--min-goodpixels", type=int, default=180)
    parser.add_argument("--n-plot-spaxels", type=int, default=8)
    parser.add_argument("--csv-min-sn", type=float, default=10.0)

    args = parser.parse_args()
    return Config(
        cube_path=args.cube_path.resolve(),
        output_dir=args.output_dir.resolve(),
        redshift=float(args.redshift),
        sps_name=args.sps_name,
        lsf_mode=str(args.lsf_mode),
        lsf_table_path=parse_optional_path(args.lsf_table_path),
        resolving_power=float(args.resolving_power),
        fit_windows_rest_um=parse_windows(args.fit_windows_rest_um),
        mask_windows_rest_um=parse_windows(args.mask_windows_rest_um),
        min_wave_finite_frac=float(args.min_wave_finite_frac),
        min_spaxel_finite_frac=float(args.min_spaxel_finite_frac),
        min_log_pixel_fraction=float(args.min_log_pixel_fraction),
        top_template_frac=float(args.top_template_frac),
        degree=int(args.degree),
        mdegree=int(args.mdegree),
        moments=int(args.moments),
        bias=float(args.bias),
        start_sigma=float(args.start_sigma),
        max_abs_velocity=float(args.max_abs_velocity),
        min_sigma=float(args.min_sigma),
        max_sigma=float(args.max_sigma),
        min_goodpixels=int(args.min_goodpixels),
        n_plot_spaxels=int(args.n_plot_spaxels),
        csv_min_sn=float(args.csv_min_sn),
    )


def main() -> None:
    cfg = parse_args()
    outdir = ensure_dir(cfg.output_dir)

    cube_data = read_cube_data(cfg)
    sps, _, resolution_info = load_sps_library(
        cfg,
        lam_min_ang=float(np.min(cube_data.lam_gal_ang[cube_data.fit_mask_log])),
        lam_max_ang=float(np.max(cube_data.lam_gal_ang[cube_data.fit_mask_log])),
        velscale=float(cube_data.velscale),
    )

    n_valid = cube_data.valid_spaxel_indices.size
    n_template = max(20, int(np.ceil(cfg.top_template_frac * n_valid)))
    template_sel = np.argsort(cube_data.signal)[-n_template:]
    global_mask = cube_data.fit_mask_log & (
        np.nanmean(cube_data.valid_frac_log[:, template_sel], axis=1) >= cfg.min_log_pixel_fraction
    )
    global_spec = np.nanmean(cube_data.spectra_log[:, template_sel], axis=1)
    pp_global = fit_moments2(
        sps.templates,
        global_spec,
        cube_data.velscale,
        [0.0, cfg.start_sigma],
        global_mask,
        cube_data.lam_gal_ang,
        sps.lam_temp,
        degree=cfg.degree,
        mdegree=cfg.mdegree,
    )
    global_template = np.asarray(pp_global.optimal_template, dtype=float)

    plot_global_fit(
        outdir,
        cube_data.lam_gal_ang,
        global_spec,
        np.asarray(pp_global.bestfit, dtype=float),
        title=(
            f"{cfg.cube_path.stem} global template | "
            f"sigma={pp_global.sol[1]:.1f} km/s | S/N={pp_global.sn:.1f}"
        ),
    )

    start_sigma = float(pp_global.sol[1]) if np.isfinite(pp_global.sol[1]) else cfg.start_sigma
    preview_order = np.argsort(cube_data.signal)[::-1]
    preview_done = 0

    table_rows: list[dict[str, object]] = []
    map_shape = cube_data.map_shape
    maps = {
        "VREL_MAP": np.full(map_shape, np.nan, dtype=float),
        "SIGMA_MAP": np.full(map_shape, np.nan, dtype=float),
        "VRMS_MAP": np.full(map_shape, np.nan, dtype=float),
        "H3_MAP": np.full(map_shape, np.nan, dtype=float),
        "H4_MAP": np.full(map_shape, np.nan, dtype=float),
        "SN_MAP": np.full(map_shape, np.nan, dtype=float),
        "CHI2_MAP": np.full(map_shape, np.nan, dtype=float),
        "GOODFIT_MAP": np.zeros(map_shape, dtype=int),
    }

    for j in tqdm(range(n_valid), desc="Fitting NIRSpec spaxels"):
        galaxy = cube_data.spectra_log[:, j]
        result = fit_spaxel_kinematics(
            template=global_template,
            lam_temp_template=sps.lam_temp,
            galaxy=galaxy,
            valid_frac_log=cube_data.valid_frac_log[:, j],
            cube_data=cube_data,
            resolution_info=resolution_info,
            cfg=cfg,
            start_sigma=start_sigma,
        )

        if result.ok and result.sol_rel is not None:
            start_sigma = float(result.sol_rel[1])

        row = int(cube_data.row[j])
        col = int(cube_data.col[j])
        map_row = row - 1
        map_col = col - 1
        goodpix_frac = (
            float(result.goodpixels.size / np.count_nonzero(result.clean_mask))
            if result.goodpixels is not None and result.clean_mask is not None and np.count_nonzero(result.clean_mask) > 0
            else np.nan
        )
        goodfit = bool(
            result.ok
            and np.isfinite(result.losv_abs)
            and np.isfinite(result.sigma)
            and np.isfinite(result.h3)
            and np.isfinite(result.h4)
            and np.isfinite(result.sn)
            and result.sn >= cfg.csv_min_sn
            and result.sigma >= cfg.min_sigma
            and result.sigma <= cfg.max_sigma
        )

        table_row = {
            "ROW": row,
            "COL": col,
            "X": float(cube_data.x[j]),
            "Y": float(cube_data.y[j]),
            "SIGNAL": float(cube_data.signal[j]),
            "NOISE_PROXY": float(cube_data.noise_proxy[j]),
            "SPAXEL_FINITE_FRAC": float(cube_data.spaxel_finite_frac[j]),
            "GOODFIT": goodfit,
            "GOODPIX_FRAC": goodpix_frac,
            "V_REL_KMS": float(result.sol_rel[0]) if result.sol_rel is not None else np.nan,
            "V_REL_ERR_KMS": float(result.err_rel[0]) if result.err_rel is not None else np.nan,
            "LOSV": float(result.losv_abs),
            "LOSV_err": float(result.losv_err),
            "SIGMA_PPXF": float(result.sigma_raw),
            "SIGMA_PPXF_ERR": float(result.sigma_raw_err),
            "sigma": float(result.sigma),
            "sigma_err": float(result.sigma_err),
            "VRMS": float(result.vrms),
            "VRMS_ERR": float(result.vrms_err),
            "h3": float(result.h3),
            "H3_ERR": float(result.h3_err),
            "h4": float(result.h4),
            "H4_ERR": float(result.h4_err),
            "SN": float(result.sn),
            "CHI2": float(result.chi2),
        }
        table_rows.append(table_row)

        if result.ok and result.sol_rel is not None:
            maps["VREL_MAP"][map_row, map_col] = float(result.sol_rel[0])
            maps["SIGMA_MAP"][map_row, map_col] = float(result.sigma)
            maps["VRMS_MAP"][map_row, map_col] = float(result.vrms)
            maps["H3_MAP"][map_row, map_col] = float(result.h3)
            maps["H4_MAP"][map_row, map_col] = float(result.h4)
            maps["SN_MAP"][map_row, map_col] = float(result.sn)
            maps["CHI2_MAP"][map_row, map_col] = float(result.chi2)
            maps["GOODFIT_MAP"][map_row, map_col] = int(goodfit)

            if preview_done < cfg.n_plot_spaxels and j in preview_order[: max(cfg.n_plot_spaxels * 4, 20)]:
                plot_spaxel_fit(
                    outdir,
                    cube_data.lam_gal_ang,
                    galaxy,
                    result,
                    name=f"spaxel_fit_r{row:02d}_c{col:02d}",
                )
                preview_done += 1

    plot_maps(outdir, cube_data, maps)

    good_csv_rows = [
        {
            "X": row["X"],
            "Y": row["Y"],
            "LOSV": row["LOSV"],
            "LOSV_err": row["LOSV_err"],
            "sigma_ppxf": row["SIGMA_PPXF"],
            "sigma_ppxf_err": row["SIGMA_PPXF_ERR"],
            "sigma": row["sigma"],
            "sigma_err": row["sigma_err"],
            "Vrms": row["VRMS"],
            "Vrms_err": row["VRMS_ERR"],
            "h3": row["h3"],
            "h4": row["h4"],
            "ROW": row["ROW"],
            "COL": row["COL"],
            "V_REL_KMS": row["V_REL_KMS"],
            "V_REL_ERR_KMS": row["V_REL_ERR_KMS"],
            "H3_ERR": row["H3_ERR"],
            "H4_ERR": row["H4_ERR"],
            "SN": row["SN"],
            "CHI2": row["CHI2"],
            "GOODPIX_FRAC": row["GOODPIX_FRAC"],
            "SPAXEL_FINITE_FRAC": row["SPAXEL_FINITE_FRAC"],
        }
        for row in table_rows
        if row["GOODFIT"]
    ]

    base = outdir / f"{cfg.cube_path.stem}_stellar_kinematics"
    csv_path = base.with_suffix(".csv")
    fits_path = base.with_suffix(".fits")
    npz_path = base.with_suffix(".npz")
    json_path = outdir / "run_config.json"
    summary_path = outdir / "run_summary.txt"

    save_csv(csv_path, good_csv_rows)
    save_fits_products(
        fits_path,
        cfg,
        cube_data,
        resolution_info,
        table_rows,
        maps,
        global_template=global_template,
        global_template_lam=np.asarray(sps.lam_temp, dtype=float),
        global_bestfit=np.asarray(pp_global.bestfit, dtype=float),
    )
    np.savez_compressed(
        npz_path,
        lam_gal_ang=cube_data.lam_gal_ang,
        fit_mask_log=cube_data.fit_mask_log,
        x=cube_data.x,
        y=cube_data.y,
        row=cube_data.row,
        col=cube_data.col,
        signal=cube_data.signal,
        noise_proxy=cube_data.noise_proxy,
        spaxel_finite_frac=cube_data.spaxel_finite_frac,
        table_rows=np.array(table_rows, dtype=object),
        vrel_map=maps["VREL_MAP"],
        sigma_map=maps["SIGMA_MAP"],
        vrms_map=maps["VRMS_MAP"],
        h3_map=maps["H3_MAP"],
        h4_map=maps["H4_MAP"],
        sn_map=maps["SN_MAP"],
        chi2_map=maps["CHI2_MAP"],
        goodfit_map=maps["GOODFIT_MAP"],
        global_template=global_template,
        global_template_lam=np.asarray(sps.lam_temp, dtype=float),
        global_bestfit=np.asarray(pp_global.bestfit, dtype=float),
    )

    config_dict = {
        "cube_path": str(cfg.cube_path),
        "output_dir": str(cfg.output_dir),
        "redshift": cfg.redshift,
        "sps_name": cfg.sps_name,
        "lsf_mode": cfg.lsf_mode,
        "lsf_table_path": str(cfg.lsf_table_path) if cfg.lsf_table_path is not None else None,
        "resolving_power": cfg.resolving_power,
        "resolving_power_min": resolution_info.resolving_power_min,
        "resolving_power_med": resolution_info.resolving_power_med,
        "resolving_power_max": resolution_info.resolving_power_max,
        "sigma_inst_kms": resolution_info.sigma_inst_kms,
        "sigma_inst_min_kms": resolution_info.sigma_inst_min_kms,
        "sigma_inst_max_kms": resolution_info.sigma_inst_max_kms,
        "sigma_template_kms": resolution_info.sigma_template_kms,
        "sigma_template_eff_kms": resolution_info.sigma_template_eff_kms,
        "template_r_eff": resolution_info.template_r_eff,
        "template_broader_than_data": resolution_info.template_broader_than_data,
        "template_broader_fraction": resolution_info.template_broader_fraction,
        "fit_windows_rest_um": list(cfg.fit_windows_rest_um),
        "mask_windows_rest_um": list(cfg.mask_windows_rest_um),
        "min_wave_finite_frac": cfg.min_wave_finite_frac,
        "min_spaxel_finite_frac": cfg.min_spaxel_finite_frac,
        "min_log_pixel_fraction": cfg.min_log_pixel_fraction,
        "top_template_frac": cfg.top_template_frac,
        "degree": cfg.degree,
        "mdegree": cfg.mdegree,
        "moments": cfg.moments,
        "bias": cfg.bias,
        "start_sigma": cfg.start_sigma,
        "max_abs_velocity": cfg.max_abs_velocity,
        "min_sigma": cfg.min_sigma,
        "max_sigma": cfg.max_sigma,
        "min_goodpixels": cfg.min_goodpixels,
        "n_plot_spaxels": cfg.n_plot_spaxels,
        "csv_min_sn": cfg.csv_min_sn,
    }
    json_path.write_text(json.dumps(config_dict, indent=2))

    n_total = len(table_rows)
    n_good = len(good_csv_rows)
    median_sn = float(np.nanmedian([row["SN"] for row in good_csv_rows])) if n_good else np.nan
    median_sigma = float(np.nanmedian([row["sigma"] for row in good_csv_rows])) if n_good else np.nan
    systemic_redshift_fit = (1.0 + cfg.redshift) * np.exp(float(pp_global.sol[0]) / C) - 1.0
    systemic_losv_fit = C * systemic_redshift_fit
    summary_lines = [
        f"Cube: {cfg.cube_path}",
        f"Output dir: {cfg.output_dir}",
        f"Total fitted spaxels: {n_total}",
        f"CSV good-fit spaxels: {n_good}",
        f"Global template V_rel: {float(pp_global.sol[0]):.2f} km/s",
        f"Global template sigma: {float(pp_global.sol[1]):.2f} km/s",
        f"LSF mode: {resolution_info.lsf_mode}",
        f"LSF source: {resolution_info.lsf_source}",
        f"Adopted resolving power range: {resolution_info.resolving_power_min:.1f} to {resolution_info.resolving_power_max:.1f}",
        f"Adopted resolving power median: {resolution_info.resolving_power_med:.1f}",
        f"Adopted instrumental sigma range: {resolution_info.sigma_inst_min_kms:.2f} to {resolution_info.sigma_inst_max_kms:.2f} km/s",
        f"Adopted instrumental sigma median: {resolution_info.sigma_inst_kms:.2f} km/s",
        f"E-MILES template sigma: {resolution_info.sigma_template_kms:.2f} km/s",
        f"Effective template sigma in fit: {resolution_info.sigma_template_eff_kms:.2f} km/s",
        f"E-MILES effective resolving power: {resolution_info.template_r_eff:.1f}",
        f"Template broader than data: {resolution_info.template_broader_than_data}",
        f"Template broader fraction: {resolution_info.template_broader_fraction:.3f}",
        f"Fitted systemic redshift: {systemic_redshift_fit:.9f}",
        f"Fitted systemic LOSV: {systemic_losv_fit:.2f} km/s",
        f"Median good-fit S/N: {median_sn:.2f}",
        f"Median good-fit sigma: {median_sigma:.2f} km/s",
        f"CSV: {csv_path}",
        f"FITS: {fits_path}",
        f"NPZ: {npz_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved CSV  : {csv_path}")
    print(f"Saved FITS : {fits_path}")
    print(f"Saved NPZ  : {npz_path}")
    print(f"Saved JSON : {json_path}")
    print(f"Saved text : {summary_path}")
    print(f"Good-fit CSV rows: {n_good}/{n_total}")


if __name__ == "__main__":
    main()
