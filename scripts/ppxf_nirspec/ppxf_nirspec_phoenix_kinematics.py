#!/usr/bin/env python3
"""
Fit JWST/NIRSpec stellar kinematics with pPXF and high-resolution PHOENIX templates.

This script is set up to reproduce the procedure described for NIRSpec:

- pPXF kinematic fits for [V, sigma, h3, h4]
- PHOENIX/ACES high-resolution stellar templates, R ~ 100000
- template selection from the downloaded PHOENIX files, with default cuts
  matching the stated Padova-isochrone selection envelope:
  Teff = 3000-6700 K, [Fe/H] = -2 to +1, logg = 0-4
- convolution of templates to the NIRSpec high-resolution modes, R ~ 2700
- additive Legendre polynomial degree 10
- multiplicative Legendre polynomial degree 6
- pPXF bias = 0 and regul = 0

By default the script targets the existing G235H AGN-subtracted cube and the
local PHOENIX directory:

    Data/IFU/david_subs/g235h_agn_sub.fits
    Data/phoenix_high_res/

If you download exactly the 57 selected PHOENIX templates into that directory,
the default filename-based selection will use those 57 templates. If the folder
contains a larger PHOENIX grid, pass --template-list with one filename per line
to force the exact Padova-selected set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_phoenix"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util


C = 299792.458  # km/s
FWHM_PER_SIGMA = 2.35482004503

ROOT = Path("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL")
DEFAULT_CUBE_PATH = ROOT / "Data/IFU/david_subs/g235h_agn_sub.fits"
DEFAULT_PHOENIX_DIR = ROOT / "Data/phoenix_high_res"
DEFAULT_PHOENIX_WAVE = DEFAULT_PHOENIX_DIR / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
DEFAULT_OUTPUT_DIR = ROOT / "Data/ppxf_nirspec/phoenix_g235h"

PHOENIX_RE = re.compile(
    r"lte(?P<teff>\d{5})-(?P<logg>\d\.\d{2})(?P<feh>[+-]\d\.\d)"
)


@dataclass(frozen=True)
class TemplateMeta:
    path: Path
    teff: float
    logg: float
    feh: float


@dataclass(frozen=True)
class Config:
    cube_path: Path
    output_dir: Path
    phoenix_dir: Path
    phoenix_wave_path: Path
    template_list_path: Path | None
    redshift: float
    resolving_power: float
    template_resolving_power: float
    fit_windows_rest_um: tuple[tuple[float, float], ...]
    mask_windows_rest_um: tuple[tuple[float, float], ...]
    teff_min: float
    teff_max: float
    feh_min: float
    feh_max: float
    logg_min: float
    logg_max: float
    expected_template_count: int
    strict_template_count: bool
    degree: int
    mdegree: int
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
    top_template_frac: float
    csv_min_sn: float
    n_plot_spaxels: int
    max_spaxels: int | None
    template_velocity_margin_kms: float
    wave_crpix_mode: str = "fits"


@dataclass
class CubeData:
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


@dataclass
class TemplateLibrary:
    templates: np.ndarray
    lam_temp: np.ndarray
    meta: list[TemplateMeta]
    sigma_inst_kms: float
    sigma_template_kms: float
    sigma_conv_kms: float
    sigma_conv_pix: float
    lam_min_ang: float
    lam_max_ang: float


@dataclass
class FitResult:
    ok: bool
    message: str
    sol_rel: np.ndarray | None
    err_rel: np.ndarray | None
    losv: float
    losv_err: float
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
    goodpixels: np.ndarray | None
    clean_mask: np.ndarray | None
    bestfit: np.ndarray | None
    weights: np.ndarray | None


def parse_windows(text: str) -> tuple[tuple[float, float], ...]:
    text = text.strip()
    if not text or text.lower() == "none":
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
    if not text or text.lower() == "none":
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
    diff = np.diff(np.asarray(arr, dtype=float), axis=axis)
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


def robust_noise_value(galaxy: np.ndarray, mask: np.ndarray) -> float:
    good = np.isfinite(galaxy) & mask
    x = galaxy[good]
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
    return np.asarray(pp.error, dtype=float) * np.sqrt(float(pp.chi2))


def compute_vrms_and_error(
    vrel: float,
    vrel_err: float,
    sigma: float,
    sigma_err: float,
) -> tuple[float, float]:
    vrms = float(np.sqrt(vrel**2 + sigma**2))
    if not np.isfinite(vrms) or vrms <= 0:
        return vrms, np.nan
    numer = float(np.sqrt((vrel * vrel_err) ** 2 + (sigma * sigma_err) ** 2))
    return vrms, numer / vrms


def clip_outliers(galaxy: np.ndarray, bestfit: np.ndarray, mask: np.ndarray) -> np.ndarray:
    good = mask.copy()
    while True:
        if np.count_nonzero(good) < 10:
            return good
        denom = np.sum(bestfit[good] ** 2)
        if not np.isfinite(denom) or denom <= 0:
            return good
        scale = galaxy[good] @ bestfit[good] / denom
        resid = galaxy[good] - scale * bestfit[good]
        err = robust_sigma(resid, zero=1)
        if not np.isfinite(err) or err <= 0:
            return good
        new_good = good.copy()
        new_good[good] = np.abs(resid) < 3.0 * err
        if np.array_equal(new_good, good):
            return good
        good = new_good


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
        if name in hdul and getattr(hdul[name], "data", None) is not None:
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
    if "DQ" not in hdul:
        return None
    hdu = hdul["DQ"]
    return orient_cube_nlam_first(np.asarray(hdu.data), hdu.header)


def wavelength_to_um(values: np.ndarray, unit: str | None) -> np.ndarray:
    unit_l = (unit or "um").strip().lower()
    if unit_l in ("um", "micron", "microns", "micrometer", "micrometers"):
        return values
    if unit_l in ("angstrom", "angstroms", "aa", "a"):
        return values / 1e4
    if unit_l in ("nm", "nanometer", "nanometers"):
        return values / 1e3
    if unit_l in ("m", "meter", "meters"):
        return values * 1e6
    return values


def derive_wave_axis_um(header: fits.Header, nlam: int, crpix_mode: str = "fits") -> np.ndarray:
    cd3 = header.get("CD3_3", header.get("CDELT3"))
    crval3 = header.get("CRVAL3")
    crpix3 = float(header.get("CRPIX3", 1.0))
    if cd3 is None or crval3 is None:
        raise KeyError("Could not determine wavelength solution from FITS header")
    pix = np.arange(nlam, dtype=float) + 1.0
    mode = (crpix_mode or "fits").strip().lower().replace("-", "_")
    if mode in ("fits", "standard"):
        wave = float(crval3) + (pix - crpix3) * float(cd3)
    elif mode in ("first_pixel", "crval_first", "ignore_crpix"):
        wave = float(crval3) + (pix - 1.0) * float(cd3)
    else:
        raise ValueError(
            "Invalid wave_crpix_mode. Use 'fits' for CRVAL+(pix-CRPIX)*CDELT "
            "or 'first_pixel' for CRVAL+(pix-1)*CDELT."
        )
    return wavelength_to_um(wave, header.get("CUNIT3"))


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


def parse_phoenix_filename(path: Path) -> TemplateMeta | None:
    match = PHOENIX_RE.search(path.name)
    if match is None:
        return None
    return TemplateMeta(
        path=path,
        teff=float(match.group("teff")),
        logg=float(match.group("logg")),
        feh=float(match.group("feh")),
    )


def read_template_list(path: Path, phoenix_dir: Path) -> list[Path]:
    out: list[Path] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = Path(line).expanduser()
        if not item.is_absolute():
            item = phoenix_dir / item
        out.append(item.resolve())
    return out


def discover_templates(cfg: Config) -> list[TemplateMeta]:
    if cfg.template_list_path is not None:
        paths = read_template_list(cfg.template_list_path, cfg.phoenix_dir)
    else:
        paths = sorted(
            p for p in cfg.phoenix_dir.rglob("*")
            if p.name != cfg.phoenix_wave_path.name
            and not any("Alpha=" in part for part in p.parts)
            and (p.name.endswith(".fits") or p.name.endswith(".fits.gz"))
            and p.name.startswith("lte")
        )

    meta: list[TemplateMeta] = []
    skipped: list[Path] = []
    for path in paths:
        parsed = parse_phoenix_filename(path)
        if parsed is None:
            skipped.append(path)
            continue
        if not path.is_file():
            raise FileNotFoundError(f"PHOENIX template listed but not found: {path}")
        if (
            cfg.teff_min <= parsed.teff <= cfg.teff_max
            and cfg.logg_min <= parsed.logg <= cfg.logg_max
            and cfg.feh_min <= parsed.feh <= cfg.feh_max
        ):
            meta.append(parsed)

    meta.sort(key=lambda m: (m.feh, m.logg, m.teff, m.path.name))

    if skipped:
        print(f"Skipped {len(skipped)} PHOENIX-like files with unparsed filenames")
    if cfg.expected_template_count > 0 and len(meta) != cfg.expected_template_count:
        msg = (
            f"Selected {len(meta)} PHOENIX templates, expected "
            f"{cfg.expected_template_count}. "
            "If your PHOENIX folder contains a larger grid, pass --template-list "
            "with the exact 57 Padova-selected filenames."
        )
        if cfg.strict_template_count:
            raise ValueError(msg)
        print("WARNING:", msg)
    if not meta:
        raise ValueError(f"No PHOENIX templates selected from {cfg.phoenix_dir}")
    return meta


def read_cube_data(cfg: Config) -> CubeData:
    with fits.open(cfg.cube_path, memmap=False) as hdul:
        sci_hdu = find_science_hdu(hdul)
        cube = orient_cube_nlam_first(np.asarray(sci_hdu.data, dtype=float), sci_hdu.header)
        err_cube = find_error_cube(hdul)
        dq_cube = find_dq_cube(hdul)
        header = sci_hdu.header.copy()

    nlam, ny, nx = cube.shape
    wave_obs_um = derive_wave_axis_um(header, nlam, getattr(cfg, "wave_crpix_mode", "fits"))
    wave_rest_um = wave_obs_um / (1.0 + cfg.redshift)
    wave_rest_ang = wave_rest_um * 1e4
    pixsize_arcsec = derive_pixsize_arcsec(header)

    fit_mask_lin = build_wave_mask(wave_rest_um, cfg.fit_windows_rest_um)
    if np.count_nonzero(fit_mask_lin) < cfg.min_goodpixels:
        raise ValueError("Requested fit windows contain too few linear pixels")

    finite_frac_wave = np.mean(np.isfinite(cube[fit_mask_lin]), axis=(1, 2))
    wave_fit_finite_ok = np.zeros_like(wave_rest_um, dtype=bool)
    wave_fit_finite_ok[fit_mask_lin] = finite_frac_wave >= cfg.min_wave_finite_frac
    wave_mask_lin = fit_mask_lin & wave_fit_finite_ok
    if cfg.mask_windows_rest_um:
        wave_mask_lin &= ~build_wave_mask(wave_rest_um, cfg.mask_windows_rest_um)

    contig_lo = min(lo for lo, _ in cfg.fit_windows_rest_um)
    contig_hi = max(hi for _, hi in cfg.fit_windows_rest_um)
    contig_lin = (wave_rest_um >= contig_lo) & (wave_rest_um <= contig_hi)
    if np.count_nonzero(contig_lin) < cfg.min_goodpixels:
        raise ValueError("Contiguous fit span contains too few pixels")

    signal_map = np.nanmedian(cube[wave_mask_lin], axis=0)
    if err_cube is not None:
        noise_proxy_map = np.nanmedian(err_cube[wave_mask_lin], axis=0)
    else:
        noise_proxy_map = estimate_noise_from_differences(cube[wave_mask_lin], axis=0)
    noise_proxy_map = safe_positive(noise_proxy_map)

    spaxel_finite_frac = np.mean(np.isfinite(cube[wave_mask_lin]), axis=0)
    valid_spaxel_mask = (
        np.isfinite(signal_map)
        & np.isfinite(noise_proxy_map)
        & (noise_proxy_map > 0)
        & (spaxel_finite_frac >= cfg.min_spaxel_finite_frac)
    )
    if dq_cube is not None:
        dq_frac = np.mean(np.asarray(dq_cube[wave_mask_lin]) != 0, axis=0)
        valid_spaxel_mask &= dq_frac < 0.5
    valid_spaxel_mask = select_largest_blob(valid_spaxel_mask)
    valid_spaxel_indices = np.flatnonzero(valid_spaxel_mask.ravel())
    if valid_spaxel_indices.size == 0:
        raise ValueError("No valid spaxels survived the selection cuts")

    if cfg.max_spaxels is not None:
        order = np.argsort(signal_map.ravel()[valid_spaxel_indices])[::-1]
        valid_spaxel_indices = valid_spaxel_indices[order[: cfg.max_spaxels]]

    center_index = int(np.nanargmax(np.where(valid_spaxel_mask, signal_map, -np.inf)))
    center_row, center_col = np.unravel_index(center_index, signal_map.shape)
    row2d, col2d = np.indices((ny, nx))
    x_all = (col2d - center_col) * pixsize_arcsec
    y_all = (row2d - center_row) * pixsize_arcsec

    cube_contig = cube[contig_lin, :, :]
    valid_cube = np.isfinite(cube_contig)
    spectra_lin = cube_contig.reshape(np.count_nonzero(contig_lin), -1)[:, valid_spaxel_indices]
    valid_lin = valid_cube.reshape(np.count_nonzero(contig_lin), -1)[:, valid_spaxel_indices]
    spectra_lin_filled = np.empty_like(spectra_lin)
    for j in range(spectra_lin.shape[1]):
        spectra_lin_filled[:, j] = fill_invalid_spectrum(spectra_lin[:, j])

    wave_contig_ang = wave_rest_ang[contig_lin]
    velscale0 = float(np.min(C * np.diff(np.log(wave_contig_ang))))
    spectra_log, ln_lam_gal, velscale = util.log_rebin(
        wave_contig_ang,
        spectra_lin_filled,
        velscale=velscale0,
        flux=False,
    )
    valid_frac_log, _, _ = util.log_rebin(
        wave_contig_ang,
        valid_lin.astype(float),
        velscale=velscale0,
        flux=False,
    )
    fit_mask_log, _, _ = util.log_rebin(
        wave_contig_ang,
        wave_mask_lin[contig_lin].astype(float),
        velscale=velscale0,
        flux=False,
    )
    fit_mask_log = np.asarray(fit_mask_log > 0.5, dtype=bool)
    lam_gal_ang = np.exp(ln_lam_gal)

    return CubeData(
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


def normalize_columns(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    scale = np.nanmedian(out, axis=0)
    scale = safe_positive(scale, fill_value=1.0)
    out /= scale
    out[~np.isfinite(out)] = 0.0
    return out


def load_phoenix_templates(cfg: Config, cube_data: CubeData) -> TemplateLibrary:
    meta = discover_templates(cfg)
    with fits.open(cfg.phoenix_wave_path, memmap=False) as hdul:
        wave_all = np.asarray(hdul[0].data, dtype=float).ravel()
    if wave_all.size == 0:
        raise ValueError(f"Empty PHOENIX wavelength array: {cfg.phoenix_wave_path}")

    fit_lam = cube_data.lam_gal_ang[cube_data.fit_mask_log]
    vel_pad = cfg.max_abs_velocity + cfg.template_velocity_margin_kms
    lam_min = float(np.nanmin(fit_lam) * np.exp(-vel_pad / C))
    lam_max = float(np.nanmax(fit_lam) * np.exp(vel_pad / C))
    wave_sel = (wave_all >= lam_min) & (wave_all <= lam_max)
    if np.count_nonzero(wave_sel) < 10:
        raise ValueError("PHOENIX wavelength file does not overlap the requested fit range")
    wave = wave_all[wave_sel]

    flux = np.empty((wave.size, len(meta)), dtype=np.float32)
    for j, item in enumerate(tqdm(meta, desc="Reading PHOENIX templates")):
        with fits.open(item.path, memmap=True) as hdul:
            data = np.asarray(hdul[0].data[wave_sel], dtype=np.float32)
        flux[:, j] = fill_invalid_spectrum(data)

    templates_log, ln_lam_temp, velscale_temp = util.log_rebin(
        wave,
        flux,
        velscale=cube_data.velscale,
        flux=False,
    )
    if not np.isclose(velscale_temp, cube_data.velscale, rtol=0, atol=1e-6):
        raise RuntimeError("Template and galaxy velocity scales do not match")

    sigma_inst = C / cfg.resolving_power / FWHM_PER_SIGMA
    sigma_template = C / cfg.template_resolving_power / FWHM_PER_SIGMA
    sigma_conv = np.sqrt(max(sigma_inst**2 - sigma_template**2, 0.0))
    sigma_conv_pix = sigma_conv / cube_data.velscale
    if sigma_conv_pix > 0:
        templates_log = gaussian_filter1d(
            templates_log,
            sigma=sigma_conv_pix,
            axis=0,
            mode="nearest",
        )
    templates_log = normalize_columns(templates_log)

    return TemplateLibrary(
        templates=np.asarray(templates_log, dtype=float),
        lam_temp=np.exp(ln_lam_temp),
        meta=meta,
        sigma_inst_kms=float(sigma_inst),
        sigma_template_kms=float(sigma_template),
        sigma_conv_kms=float(sigma_conv),
        sigma_conv_pix=float(sigma_conv_pix),
        lam_min_ang=float(wave[0]),
        lam_max_ang=float(wave[-1]),
    )


def normalize_galaxy(galaxy: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, float]:
    scale = np.nanmedian(galaxy[fit_mask & np.isfinite(galaxy)])
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanmedian(galaxy[np.isfinite(galaxy)])
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return np.asarray(galaxy, dtype=float) / scale, float(scale)


def ppxf_fit_spectrum(
    templates: np.ndarray,
    galaxy_in: np.ndarray,
    velscale: float,
    start: list[float],
    fit_mask_in: np.ndarray,
    lam: np.ndarray,
    lam_temp: np.ndarray,
    cfg: Config,
) -> FitResult:
    fit_mask = fit_mask_in & np.isfinite(galaxy_in)
    if np.count_nonzero(fit_mask) < cfg.min_goodpixels:
        return empty_result("too_few_goodpixels", fit_mask)

    galaxy, _ = normalize_galaxy(galaxy_in, fit_mask)
    noise0 = np.full_like(galaxy, robust_noise_value(galaxy, fit_mask))
    noise0 = safe_positive(noise0)

    try:
        pp0 = ppxf(
            templates,
            galaxy,
            noise0,
            velscale,
            start[:2],
            bounds=[
                [-cfg.max_abs_velocity, cfg.max_abs_velocity],
                [cfg.min_sigma, cfg.max_sigma],
            ],
            moments=2,
            degree=cfg.degree,
            mdegree=cfg.mdegree,
            lam=lam,
            lam_temp=lam_temp,
            mask=fit_mask,
            regul=0,
            quiet=True,
        )
        clean_mask = clip_outliers(galaxy, pp0.bestfit, fit_mask) & fit_mask
        if np.count_nonzero(clean_mask) < cfg.min_goodpixels:
            return empty_result("too_few_pixels_after_clip", clean_mask)

        resid = galaxy[clean_mask] - pp0.bestfit[clean_mask]
        noise_val = robust_sigma(resid, zero=1)
        if not np.isfinite(noise_val) or noise_val <= 0:
            noise_val = robust_noise_value(galaxy, clean_mask)
        noise = np.full_like(galaxy, noise_val)
        noise = safe_positive(noise)

        pp2 = ppxf(
            templates,
            galaxy,
            noise,
            velscale,
            pp0.sol,
            bounds=[
                [-cfg.max_abs_velocity, cfg.max_abs_velocity],
                [cfg.min_sigma, cfg.max_sigma],
            ],
            moments=2,
            degree=cfg.degree,
            mdegree=cfg.mdegree,
            lam=lam,
            lam_temp=lam_temp,
            mask=clean_mask,
            regul=0,
            quiet=True,
        )

        start4 = [float(pp2.sol[0]), float(pp2.sol[1]), 0.0, 0.0]
        bounds4 = [
            [-cfg.max_abs_velocity, cfg.max_abs_velocity],
            [cfg.min_sigma, cfg.max_sigma],
            [-0.3, 0.3],
            [-0.3, 0.3],
        ]
        pp4 = ppxf(
            templates,
            galaxy,
            noise,
            velscale,
            start4,
            bias=cfg.bias,
            bounds=bounds4,
            moments=cfg.moments,
            degree=cfg.degree,
            mdegree=cfg.mdegree,
            lam=lam,
            lam_temp=lam_temp,
            mask=clean_mask,
            regul=0,
            quiet=True,
        )
        noise *= np.sqrt(float(pp4.chi2))
        pp4 = ppxf(
            templates,
            galaxy,
            noise,
            velscale,
            pp4.sol,
            bias=cfg.bias,
            bounds=bounds4,
            moments=cfg.moments,
            degree=cfg.degree,
            mdegree=cfg.mdegree,
            lam=lam,
            lam_temp=lam_temp,
            mask=clean_mask,
            regul=0,
            quiet=True,
        )
    except Exception as exc:  # pragma: no cover - depends on the input data
        return empty_result(f"ppxf_failed:{exc}", fit_mask)

    sol = np.asarray(pp4.sol, dtype=float)
    err = correct_ppxf_errors(pp4)
    redshift_fit = (1.0 + cfg.redshift) * np.exp(sol[0] / C) - 1.0
    losv = C * redshift_fit
    losv_err = (1.0 + redshift_fit) * float(err[0]) if err is not None else np.nan
    sigma = float(sol[1])
    sigma_err = float(err[1]) if err is not None else np.nan
    vrel_err = float(err[0]) if err is not None else np.nan
    vrms, vrms_err = compute_vrms_and_error(float(sol[0]), vrel_err, sigma, sigma_err)

    resid = (pp4.galaxy - pp4.bestfit)[pp4.goodpixels]
    rsig = robust_sigma(resid, zero=1)
    sn = np.nanmedian(pp4.galaxy[pp4.goodpixels]) / rsig if rsig > 0 else np.nan

    return FitResult(
        ok=True,
        message="ok",
        sol_rel=sol,
        err_rel=err,
        losv=float(losv),
        losv_err=float(losv_err),
        sigma=sigma,
        sigma_err=sigma_err,
        vrms=float(vrms),
        vrms_err=float(vrms_err),
        h3=float(sol[2]) if sol.size > 2 else np.nan,
        h3_err=float(err[2]) if err is not None and err.size > 2 else np.nan,
        h4=float(sol[3]) if sol.size > 3 else np.nan,
        h4_err=float(err[3]) if err is not None and err.size > 3 else np.nan,
        sn=float(sn),
        chi2=float(pp4.chi2),
        goodpixels=np.asarray(pp4.goodpixels, dtype=int),
        clean_mask=np.asarray(clean_mask, dtype=bool),
        bestfit=np.asarray(pp4.bestfit, dtype=float),
        weights=np.asarray(pp4.weights[: templates.shape[1]], dtype=float),
    )


def empty_result(message: str, clean_mask: np.ndarray | None = None) -> FitResult:
    return FitResult(
        ok=False,
        message=message,
        sol_rel=None,
        err_rel=None,
        losv=np.nan,
        losv_err=np.nan,
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
        goodpixels=None,
        clean_mask=clean_mask,
        bestfit=None,
        weights=None,
    )


def make_maps(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "VREL_MAP": np.full(shape, np.nan, dtype=float),
        "LOSV_MAP": np.full(shape, np.nan, dtype=float),
        "SIGMA_MAP": np.full(shape, np.nan, dtype=float),
        "VRMS_MAP": np.full(shape, np.nan, dtype=float),
        "H3_MAP": np.full(shape, np.nan, dtype=float),
        "H4_MAP": np.full(shape, np.nan, dtype=float),
        "SN_MAP": np.full(shape, np.nan, dtype=float),
        "CHI2_MAP": np.full(shape, np.nan, dtype=float),
        "GOODFIT_MAP": np.zeros(shape, dtype=int),
    }


def plot_fit(
    outpath: Path,
    lam: np.ndarray,
    galaxy: np.ndarray,
    result: FitResult,
    title: str,
) -> None:
    fit_mask = result.clean_mask if result.clean_mask is not None else np.isfinite(galaxy)
    galaxy_norm, _ = normalize_galaxy(galaxy, fit_mask)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(lam, galaxy_norm, lw=0.8, color="0.2", label="Galaxy")
    if result.bestfit is not None:
        ax.plot(lam, result.bestfit, lw=1.0, color="tab:blue", label="pPXF")
    ax.set_xlabel("Rest wavelength [Angstrom]")
    ax.set_ylabel("Normalized flux")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
        ("VRMS_MAP", "Vrms [km/s]", "magma"),
        ("H3_MAP", "h3", "RdBu_r"),
        ("H4_MAP", "h4", "RdBu_r"),
        ("SN_MAP", "S/N", "viridis"),
        ("GOODFIT_MAP", "good fit", "gray"),
        ("CHI2_MAP", "chi2", "cividis"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, title, cmap) in zip(axes, panels):
        im = ax.imshow(maps[key], origin="lower", extent=extent, cmap=cmap, aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("X [arcsec]")
        ax.set_ylabel("Y [arcsec]")
    outpath = outdir / "phoenix_kinematics_maps.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def write_template_manifest(path: Path, library: TemplateLibrary) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "path", "teff", "logg", "feh"])
        writer.writeheader()
        for item in library.meta:
            writer.writerow(
                {
                    "filename": item.path.name,
                    "path": str(item.path),
                    "teff": item.teff,
                    "logg": item.logg,
                    "feh": item.feh,
                }
            )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "X",
        "Y",
        "LOSV",
        "LOSV_err",
        "V_REL_KMS",
        "V_REL_ERR_KMS",
        "sigma",
        "sigma_err",
        "Vrms",
        "Vrms_err",
        "h3",
        "h3_err",
        "h4",
        "h4_err",
        "ROW",
        "COL",
        "SN",
        "CHI2",
        "GOODPIX_FRAC",
        "SPAXEL_FINITE_FRAC",
        "GOODFIT",
        "MESSAGE",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_fits(
    path: Path,
    cfg: Config,
    cube_data: CubeData,
    library: TemplateLibrary,
    table_rows: list[dict[str, object]],
    maps: dict[str, np.ndarray],
) -> None:
    hdr = fits.Header()
    hdr["OBJECT"] = cfg.cube_path.stem[:68]
    hdr["REDSHFT"] = float(cfg.redshift)
    hdr["RPOWER"] = float(cfg.resolving_power)
    hdr["RTEMPL"] = float(cfg.template_resolving_power)
    hdr["SIGINST"] = (float(library.sigma_inst_kms), "km/s")
    hdr["SIGTEMP"] = (float(library.sigma_template_kms), "km/s")
    hdr["SIGCONV"] = (float(library.sigma_conv_kms), "km/s")
    hdr["DEGREE"] = int(cfg.degree)
    hdr["MDEGREE"] = int(cfg.mdegree)
    hdr["MOMENTS"] = int(cfg.moments)
    hdr["BIAS"] = float(cfg.bias)
    hdr["REGUL"] = 0.0
    hdr["NTMPL"] = len(library.meta)
    hdr["LAMMIN"] = float(np.nanmin(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["LAMMAX"] = float(np.nanmax(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["PIXSIZE"] = (float(cube_data.pixsize_arcsec), "arcsec")
    hdr["CENROW"] = int(cube_data.center_row + 1)
    hdr["CENCOL"] = int(cube_data.center_col + 1)

    numeric_cols = []
    table_keys = [
        "ROW",
        "COL",
        "X",
        "Y",
        "LOSV",
        "LOSV_err",
        "V_REL_KMS",
        "V_REL_ERR_KMS",
        "sigma",
        "sigma_err",
        "Vrms",
        "Vrms_err",
        "h3",
        "h3_err",
        "h4",
        "h4_err",
        "SN",
        "CHI2",
        "GOODPIX_FRAC",
        "SPAXEL_FINITE_FRAC",
        "GOODFIT",
    ]
    for key in table_keys:
        values = np.array([row[key] for row in table_rows])
        if values.dtype.kind == "b":
            fmt = "L"
        elif np.issubdtype(values.dtype, np.integer):
            fmt = "J"
        else:
            fmt = "D"
        numeric_cols.append(fits.Column(name=key.upper(), format=fmt, array=values))
    msg = np.array([str(row["MESSAGE"])[:80] for row in table_rows])
    numeric_cols.append(fits.Column(name="MESSAGE", format="80A", array=msg))
    kin_hdu = fits.BinTableHDU.from_columns(numeric_cols, name="KIN_RESULTS")

    meta_cols = [
        fits.Column(name="FILENAME", format="160A", array=np.array([m.path.name for m in library.meta])),
        fits.Column(name="TEFF", format="D", array=np.array([m.teff for m in library.meta])),
        fits.Column(name="LOGG", format="D", array=np.array([m.logg for m in library.meta])),
        fits.Column(name="FEH", format="D", array=np.array([m.feh for m in library.meta])),
    ]
    tmpl_hdu = fits.BinTableHDU.from_columns(meta_cols, name="TEMPLATE_META")

    hdus = [
        fits.PrimaryHDU(header=hdr),
        kin_hdu,
        tmpl_hdu,
        fits.ImageHDU(data=maps["VREL_MAP"].astype(np.float32), name="VREL_MAP"),
        fits.ImageHDU(data=maps["LOSV_MAP"].astype(np.float32), name="LOSV_MAP"),
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
        fits.ImageHDU(data=library.lam_temp.astype(np.float32), name="TEMPLATE_LAMBDA"),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def config_to_json(cfg: Config, library: TemplateLibrary, cube_data: CubeData) -> dict[str, object]:
    data = asdict(cfg)
    for key, val in list(data.items()):
        if isinstance(val, Path):
            data[key] = str(val)
        elif isinstance(val, tuple):
            data[key] = list(val)
    data.update(
        {
            "n_templates": len(library.meta),
            "sigma_inst_kms": library.sigma_inst_kms,
            "sigma_template_kms": library.sigma_template_kms,
            "sigma_conv_kms": library.sigma_conv_kms,
            "sigma_conv_pix": library.sigma_conv_pix,
            "velscale_kms": cube_data.velscale,
            "cube_shape": cube_data.cube_shape,
            "map_shape": cube_data.map_shape,
        }
    )
    return data


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Fit NIRSpec stellar kinematics with pPXF and PHOENIX templates.",
    )
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phoenix-dir", type=Path, default=DEFAULT_PHOENIX_DIR)
    parser.add_argument("--phoenix-wave-path", type=Path, default=DEFAULT_PHOENIX_WAVE)
    parser.add_argument("--template-list", type=str, default=None)
    parser.add_argument("--redshift", type=float, default=0.003633)
    parser.add_argument("--resolving-power", type=float, default=2700.0)
    parser.add_argument("--template-resolving-power", type=float, default=100000.0)
    parser.add_argument("--fit-windows-rest-um", type=str, default="2.10-2.40")
    parser.add_argument(
        "--mask-windows-rest-um",
        type=str,
        default="2.117-2.127,2.161-2.171,2.219-2.228,2.316-2.326",
    )
    parser.add_argument("--teff-min", type=float, default=3000.0)
    parser.add_argument("--teff-max", type=float, default=6700.0)
    parser.add_argument("--feh-min", type=float, default=-2.0)
    parser.add_argument("--feh-max", type=float, default=1.0)
    parser.add_argument("--logg-min", type=float, default=0.0)
    parser.add_argument("--logg-max", type=float, default=4.0)
    parser.add_argument("--expected-template-count", type=int, default=57)
    parser.add_argument("--strict-template-count", action="store_true")
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--mdegree", type=int, default=6)
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
    parser.add_argument("--top-template-frac", type=float, default=0.10)
    parser.add_argument("--csv-min-sn", type=float, default=10.0)
    parser.add_argument("--n-plot-spaxels", type=int, default=8)
    parser.add_argument("--max-spaxels", type=int, default=None)
    parser.add_argument("--template-velocity-margin-kms", type=float, default=1500.0)
    parser.add_argument(
        "--wave-crpix-mode",
        choices=("fits", "first_pixel"),
        default="fits",
        help=(
            "How to interpret the spectral WCS. 'fits' uses CRVAL3+(pix-CRPIX3)*CDELT3. "
            "'first_pixel' uses CRVAL3+(pix-1)*CDELT3 for cubes whose CRVAL3 is already "
            "the first wavelength sample."
        ),
    )

    args = parser.parse_args()
    return Config(
        cube_path=args.cube_path.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        phoenix_dir=args.phoenix_dir.expanduser().resolve(),
        phoenix_wave_path=args.phoenix_wave_path.expanduser().resolve(),
        template_list_path=parse_optional_path(args.template_list),
        redshift=float(args.redshift),
        resolving_power=float(args.resolving_power),
        template_resolving_power=float(args.template_resolving_power),
        fit_windows_rest_um=parse_windows(args.fit_windows_rest_um),
        mask_windows_rest_um=parse_windows(args.mask_windows_rest_um),
        teff_min=float(args.teff_min),
        teff_max=float(args.teff_max),
        feh_min=float(args.feh_min),
        feh_max=float(args.feh_max),
        logg_min=float(args.logg_min),
        logg_max=float(args.logg_max),
        expected_template_count=int(args.expected_template_count),
        strict_template_count=bool(args.strict_template_count),
        degree=int(args.degree),
        mdegree=int(args.mdegree),
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
        top_template_frac=float(args.top_template_frac),
        csv_min_sn=float(args.csv_min_sn),
        n_plot_spaxels=int(args.n_plot_spaxels),
        max_spaxels=args.max_spaxels,
        template_velocity_margin_kms=float(args.template_velocity_margin_kms),
        wave_crpix_mode=str(args.wave_crpix_mode),
    )


def main() -> None:
    cfg = parse_args()
    outdir = ensure_dir(cfg.output_dir)

    print(f"Reading cube: {cfg.cube_path}")
    cube_data = read_cube_data(cfg)
    print(
        f"Log-rebinned cube: {cube_data.spectra_log.shape[0]} pixels, "
        f"{cube_data.spectra_log.shape[1]} spectra, velscale={cube_data.velscale:.3f} km/s"
    )

    print(f"Loading PHOENIX templates from: {cfg.phoenix_dir}")
    library = load_phoenix_templates(cfg, cube_data)
    print(
        f"Using {len(library.meta)} templates; convolved R={cfg.template_resolving_power:.0f} "
        f"to R={cfg.resolving_power:.0f} "
        f"(sigma_conv={library.sigma_conv_kms:.2f} km/s)"
    )

    n_valid = cube_data.valid_spaxel_indices.size
    n_global = max(1, int(np.ceil(cfg.top_template_frac * n_valid)))
    global_sel = np.argsort(cube_data.signal)[-n_global:]
    global_spec = np.nanmean(cube_data.spectra_log[:, global_sel], axis=1)
    global_valid_frac = np.nanmean(cube_data.valid_frac_log[:, global_sel], axis=1)
    global_mask = cube_data.fit_mask_log & (global_valid_frac >= cfg.min_log_pixel_fraction)
    global_result = ppxf_fit_spectrum(
        library.templates,
        global_spec,
        cube_data.velscale,
        [0.0, cfg.start_sigma],
        global_mask,
        cube_data.lam_gal_ang,
        library.lam_temp,
        cfg,
    )
    plot_fit(
        outdir / "phoenix_global_fit.png",
        cube_data.lam_gal_ang,
        global_spec,
        global_result,
        title=(
            f"Global PHOENIX pPXF fit | "
            f"V={global_result.sol_rel[0]:.1f} km/s, "
            f"sigma={global_result.sigma:.1f} km/s"
            if global_result.sol_rel is not None
            else f"Global PHOENIX pPXF fit failed: {global_result.message}"
        ),
    )
    start_sigma = global_result.sigma if np.isfinite(global_result.sigma) else cfg.start_sigma

    maps = make_maps(cube_data.map_shape)
    rows: list[dict[str, object]] = []
    preview_order = np.argsort(cube_data.signal)[::-1]
    preview_candidates = set(preview_order[: max(cfg.n_plot_spaxels * 4, 20)])
    preview_done = 0

    for j in tqdm(range(n_valid), desc="Fitting NIRSpec spaxels"):
        galaxy = cube_data.spectra_log[:, j]
        fit_mask = cube_data.fit_mask_log & (cube_data.valid_frac_log[:, j] >= cfg.min_log_pixel_fraction)
        result = ppxf_fit_spectrum(
            library.templates,
            galaxy,
            cube_data.velscale,
            [0.0, start_sigma],
            fit_mask,
            cube_data.lam_gal_ang,
            library.lam_temp,
            cfg,
        )
        if result.ok and np.isfinite(result.sigma):
            start_sigma = result.sigma

        row = int(cube_data.row[j])
        col = int(cube_data.col[j])
        map_row = row - 1
        map_col = col - 1
        goodpix_frac = (
            float(result.goodpixels.size / np.count_nonzero(result.clean_mask))
            if result.goodpixels is not None
            and result.clean_mask is not None
            and np.count_nonzero(result.clean_mask) > 0
            else np.nan
        )
        vrel = float(result.sol_rel[0]) if result.sol_rel is not None else np.nan
        vrel_err = float(result.err_rel[0]) if result.err_rel is not None else np.nan
        goodfit = bool(
            result.ok
            and np.isfinite(vrel)
            and np.isfinite(result.sigma)
            and np.isfinite(result.h3)
            and np.isfinite(result.h4)
            and np.isfinite(result.sn)
            and result.sn >= cfg.csv_min_sn
            and cfg.min_sigma <= result.sigma <= cfg.max_sigma
        )

        row_dict = {
            "ROW": row,
            "COL": col,
            "X": float(cube_data.x[j]),
            "Y": float(cube_data.y[j]),
            "LOSV": float(result.losv),
            "LOSV_err": float(result.losv_err),
            "V_REL_KMS": vrel,
            "V_REL_ERR_KMS": vrel_err,
            "sigma": float(result.sigma),
            "sigma_err": float(result.sigma_err),
            "Vrms": float(result.vrms),
            "Vrms_err": float(result.vrms_err),
            "h3": float(result.h3),
            "h3_err": float(result.h3_err),
            "h4": float(result.h4),
            "h4_err": float(result.h4_err),
            "SN": float(result.sn),
            "CHI2": float(result.chi2),
            "GOODPIX_FRAC": goodpix_frac,
            "SPAXEL_FINITE_FRAC": float(cube_data.spaxel_finite_frac[j]),
            "GOODFIT": goodfit,
            "MESSAGE": result.message,
        }
        rows.append(row_dict)

        if result.ok and result.sol_rel is not None:
            maps["VREL_MAP"][map_row, map_col] = vrel
            maps["LOSV_MAP"][map_row, map_col] = float(result.losv)
            maps["SIGMA_MAP"][map_row, map_col] = float(result.sigma)
            maps["VRMS_MAP"][map_row, map_col] = float(result.vrms)
            maps["H3_MAP"][map_row, map_col] = float(result.h3)
            maps["H4_MAP"][map_row, map_col] = float(result.h4)
            maps["SN_MAP"][map_row, map_col] = float(result.sn)
            maps["CHI2_MAP"][map_row, map_col] = float(result.chi2)
            maps["GOODFIT_MAP"][map_row, map_col] = int(goodfit)

            if preview_done < cfg.n_plot_spaxels and j in preview_candidates:
                plot_fit(
                    outdir / f"phoenix_spaxel_fit_r{row:02d}_c{col:02d}.png",
                    cube_data.lam_gal_ang,
                    galaxy,
                    result,
                    title=f"Spaxel row={row}, col={col}",
                )
                preview_done += 1

    plot_maps(outdir, cube_data, maps)

    base = outdir / f"{cfg.cube_path.stem}_phoenix_kinematics"
    csv_path = base.with_suffix(".csv")
    all_csv_path = base.with_name(base.name + "_all").with_suffix(".csv")
    fits_path = base.with_suffix(".fits")
    npz_path = base.with_suffix(".npz")
    json_path = outdir / "phoenix_run_config.json"
    summary_path = outdir / "phoenix_run_summary.txt"
    manifest_path = outdir / "selected_phoenix_templates.csv"

    good_rows = [row for row in rows if row["GOODFIT"]]
    write_csv(csv_path, good_rows)
    write_csv(all_csv_path, rows)
    write_template_manifest(manifest_path, library)
    save_fits(fits_path, cfg, cube_data, library, rows, maps)
    np.savez_compressed(
        npz_path,
        lam_gal_ang=cube_data.lam_gal_ang,
        lam_temp=library.lam_temp,
        fit_mask_log=cube_data.fit_mask_log,
        x=cube_data.x,
        y=cube_data.y,
        row=cube_data.row,
        col=cube_data.col,
        table_rows=np.array(rows, dtype=object),
        vrel_map=maps["VREL_MAP"],
        losv_map=maps["LOSV_MAP"],
        sigma_map=maps["SIGMA_MAP"],
        vrms_map=maps["VRMS_MAP"],
        h3_map=maps["H3_MAP"],
        h4_map=maps["H4_MAP"],
        sn_map=maps["SN_MAP"],
        chi2_map=maps["CHI2_MAP"],
        goodfit_map=maps["GOODFIT_MAP"],
        template_filenames=np.array([m.path.name for m in library.meta]),
        template_teff=np.array([m.teff for m in library.meta]),
        template_logg=np.array([m.logg for m in library.meta]),
        template_feh=np.array([m.feh for m in library.meta]),
    )
    json_path.write_text(json.dumps(config_to_json(cfg, library, cube_data), indent=2) + "\n")

    n_good = len(good_rows)
    med_sn = float(np.nanmedian([row["SN"] for row in good_rows])) if good_rows else np.nan
    med_sigma = float(np.nanmedian([row["sigma"] for row in good_rows])) if good_rows else np.nan
    summary_lines = [
        "PHOENIX pPXF NIRSpec stellar kinematics",
        f"Cube: {cfg.cube_path}",
        f"Output dir: {cfg.output_dir}",
        f"PHOENIX dir: {cfg.phoenix_dir}",
        f"PHOENIX wavelength file: {cfg.phoenix_wave_path}",
        f"Templates used: {len(library.meta)}",
        f"Resolving power: data R={cfg.resolving_power:.1f}, template R={cfg.template_resolving_power:.1f}",
        f"Template convolution sigma: {library.sigma_conv_kms:.3f} km/s ({library.sigma_conv_pix:.3f} pix)",
        f"pPXF setup: moments={cfg.moments}, degree={cfg.degree}, mdegree={cfg.mdegree}, bias={cfg.bias}, regul=0",
        f"Fit windows rest um: {cfg.fit_windows_rest_um}",
        f"Masked windows rest um: {cfg.mask_windows_rest_um}",
        f"Total fitted spectra: {len(rows)}",
        f"Good-fit spectra: {n_good}",
        f"Median good-fit S/N: {med_sn:.2f}",
        f"Median good-fit sigma: {med_sigma:.2f} km/s",
        f"Good-fit CSV: {csv_path}",
        f"All-fits CSV: {all_csv_path}",
        f"FITS: {fits_path}",
        f"NPZ: {npz_path}",
        f"Template manifest: {manifest_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved CSV  : {csv_path}")
    print(f"Saved all  : {all_csv_path}")
    print(f"Saved FITS : {fits_path}")
    print(f"Saved NPZ  : {npz_path}")
    print(f"Saved JSON : {json_path}")
    print(f"Saved text : {summary_path}")
    print(f"Good-fit rows: {n_good}/{len(rows)}")


if __name__ == "__main__":
    main()
