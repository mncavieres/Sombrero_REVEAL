#!/usr/bin/env python3
"""
Calibrate MUSE pPXF kinematics onto the NIRSpec pPXF scale.

The calibration is a per-moment affine transform,

    NIRSpec = multiplier * MUSE + additive,

fit at the NIRSpec bin centers after rotating the NIRSpec coordinates into the
MUSE frame.  The script writes a calibrated copy of the MUSE FITS product and
checkplots showing principal-axis major/minor profiles for VLOS, sigma, h3, and
h4.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_muse_to_nirspec_calibration"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUSE_FITS = (
    ROOT
    / "Data/compiled_results/muse/emiles_ppxf_muse/"
    "c30_DATACUBE_normppxf_skycont_Part1_0000_ppxf_products_emiles_dustmasked.fits"
)
DEFAULT_NIRSPEC_FITS = (
    ROOT
    / "Data/compiled_results/nirspec/nirspec_agnsub_kinematics_ppxf_phoenix_lsf_sn120/"
    "g235h_agn_sub_phoenix_powerbin_lsf_sn120_kinematics.fits"
)
DEFAULT_OUTPUT_DIR = DEFAULT_MUSE_FITS.parent / "muse_to_nirspec_calibration"


@dataclass(frozen=True)
class QuantitySpec:
    name: str
    label: str
    unit: str
    muse_column: str
    muse_error_column: str
    muse_map: str
    muse_error_map: str
    nirspec_column: str
    nirspec_error_column: str
    cmap: str


QUANTITIES = [
    QuantitySpec(
        name="vlos",
        label="VLOS",
        unit="km/s",
        muse_column="V_KMS",
        muse_error_column="VERR_KMS",
        muse_map="VEL_MAP",
        muse_error_map="VELERR_MAP",
        nirspec_column="V_REL_KMS",
        nirspec_error_column="V_REL_ERR_KMS",
        cmap="RdBu_r",
    ),
    QuantitySpec(
        name="sigma",
        label="sigma",
        unit="km/s",
        muse_column="SIGMA_KMS",
        muse_error_column="SIGERR_KMS",
        muse_map="SIGMA_MAP",
        muse_error_map="SIGERR_MAP",
        nirspec_column="SIGMA",
        nirspec_error_column="SIGMA_ERR",
        cmap="inferno",
    ),
    QuantitySpec(
        name="h3",
        label="h3",
        unit="",
        muse_column="H3",
        muse_error_column="H3_ERR",
        muse_map="H3_MAP",
        muse_error_map="H3ERR_MAP",
        nirspec_column="H3",
        nirspec_error_column="H3_ERR",
        cmap="RdBu_r",
    ),
    QuantitySpec(
        name="h4",
        label="h4",
        unit="",
        muse_column="H4",
        muse_error_column="H4_ERR",
        muse_map="H4_MAP",
        muse_error_map="H4ERR_MAP",
        nirspec_column="H4",
        nirspec_error_column="H4_ERR",
        cmap="RdBu_r",
    ),
]


@dataclass
class MuseProduct:
    path: Path
    x: np.ndarray
    y: np.ndarray
    signal: np.ndarray
    signal_map: np.ndarray
    x_map: np.ndarray
    y_map: np.ndarray
    values: dict[str, np.ndarray]
    errors: dict[str, np.ndarray]
    maps: dict[str, np.ndarray]
    error_maps: dict[str, np.ndarray]
    x_grid: np.ndarray
    y_grid: np.ndarray


@dataclass
class NirspecProduct:
    path: Path
    x_native: np.ndarray
    y_native: np.ndarray
    x: np.ndarray
    y: np.ndarray
    signal: np.ndarray
    goodfit: np.ndarray
    values: dict[str, np.ndarray]
    errors: dict[str, np.ndarray]


@dataclass
class SignalMapData:
    label: str
    x: np.ndarray
    y: np.ndarray
    signal: np.ndarray


@dataclass(frozen=True)
class SignalGeometry:
    label: str
    center_mode: str
    center_x: float
    center_y: float
    brightest_x: float
    brightest_y: float
    brightest_signal: float
    centroid_x: float
    centroid_y: float
    principal_axis_deg: float
    principal_axis_percentile: float
    n_axis_pixels: int


@dataclass(frozen=True)
class SpatialAlignment:
    mode: str
    muse_center_x: float
    muse_center_y: float
    nirspec_center_x: float
    nirspec_center_y: float
    muse_principal_axis_deg: float
    nirspec_principal_axis_deg: float
    base_rotation_deg: float
    final_rotation_deg: float
    center_shift_x_arcsec: float
    center_shift_y_arcsec: float


@dataclass(frozen=True)
class FitResult:
    quantity: str
    label: str
    multiplier: float
    additive: float
    multiplier_err: float
    additive_err: float
    n_initial: int
    n_used: int
    rms_used: float
    robust_sigma_used: float
    median_residual_used: float
    clip_sigma: float


@dataclass(frozen=True)
class OrientationCandidate:
    rotation_deg: float
    vlos_multiplier: float
    vlos_additive: float
    pearson_r: float
    n_initial: int
    n_used: int
    rms_used: float
    selected: bool = False


@dataclass(frozen=True)
class ProfileRow:
    quantity: str
    dataset: str
    axis: str
    principal_angle_deg: float
    radius_arcsec: float
    median: float
    p16: float
    p84: float
    n: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rotate_xy(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def transform_xy(
    x: np.ndarray,
    y: np.ndarray,
    rotation_deg: float,
    source_center: tuple[float, float],
    target_center: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(x, dtype=float) - float(source_center[0])
    y0 = np.asarray(y, dtype=float) - float(source_center[1])
    xr, yr = rotate_xy(x0, y0, rotation_deg)
    return xr + float(target_center[0]), yr + float(target_center[1])


def project_to_angle(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates parallel/perpendicular to an axis at angle_deg."""
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    major = x * cos_t + y * sin_t
    minor = -x * sin_t + y * cos_t
    return major, minor


def normalize_axis_angle(angle_deg: float) -> float:
    """Normalize an axis angle, where theta and theta+180 are equivalent."""
    angle = ((angle_deg + 90.0) % 180.0) - 90.0
    if angle == -90.0:
        return 90.0
    return float(angle)


def finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values > 0.0)


def safe_percentile(values: np.ndarray, pct: float, default: float = np.nan) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.nanpercentile(values, pct))


def weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    clip_sigma: float,
    max_iter: int = 8,
) -> tuple[FitResult, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        mask &= np.isfinite(yerr) & (yerr > 0.0)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Need at least three matched samples for an affine fit")

    fit_mask = mask.copy()
    params = np.array([1.0, 0.0], dtype=float)
    for _ in range(max_iter):
        xi = x[fit_mask]
        yi = y[fit_mask]
        if yerr is None:
            weights = np.ones_like(xi)
        else:
            weights = 1.0 / np.square(yerr[fit_mask])

        design = np.column_stack([xi, np.ones_like(xi)])
        sqrt_w = np.sqrt(weights)
        aw = design * sqrt_w[:, None]
        yw = yi * sqrt_w
        params = np.linalg.lstsq(aw, yw, rcond=None)[0]

        residual = y - (params[0] * x + params[1])
        resid_used = residual[fit_mask]
        center = float(np.nanmedian(resid_used))
        robust_sigma = 1.4826 * float(np.nanmedian(np.abs(resid_used - center)))
        if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
            robust_sigma = float(np.nanstd(resid_used))
        if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
            break
        next_mask = mask & (np.abs(residual - center) <= clip_sigma * robust_sigma)
        if np.array_equal(next_mask, fit_mask):
            break
        if np.count_nonzero(next_mask) < 3:
            break
        fit_mask = next_mask

    xi = x[fit_mask]
    yi = y[fit_mask]
    if yerr is None:
        weights = np.ones_like(xi)
    else:
        weights = 1.0 / np.square(yerr[fit_mask])
    design = np.column_stack([xi, np.ones_like(xi)])
    sqrt_w = np.sqrt(weights)
    aw = design * sqrt_w[:, None]
    yw = yi * sqrt_w
    params = np.linalg.lstsq(aw, yw, rcond=None)[0]
    residual = y - (params[0] * x + params[1])
    resid_used = residual[fit_mask]

    cov = np.full((2, 2), np.nan, dtype=float)
    try:
        normal = aw.T @ aw
        cov = np.linalg.inv(normal)
        dof = max(1, xi.size - 2)
        chi2 = float(np.sum(np.square((yi - (params[0] * xi + params[1])) * sqrt_w)))
        cov *= chi2 / dof
    except np.linalg.LinAlgError:
        pass

    center = float(np.nanmedian(resid_used))
    robust_sigma = 1.4826 * float(np.nanmedian(np.abs(resid_used - center)))
    rms = float(np.sqrt(np.nanmean(np.square(resid_used))))

    dummy = FitResult(
        quantity="",
        label="",
        multiplier=float(params[0]),
        additive=float(params[1]),
        multiplier_err=float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan,
        additive_err=float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan,
        n_initial=int(np.count_nonzero(mask)),
        n_used=int(np.count_nonzero(fit_mask)),
        rms_used=rms,
        robust_sigma_used=float(robust_sigma),
        median_residual_used=center,
        clip_sigma=float(clip_sigma),
    )
    return dummy, fit_mask


def make_weighted_axis_angle(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    signal: np.ndarray,
    mode: str,
) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mode == "quantity":
        vals = np.asarray(values, dtype=float)
        mask &= np.isfinite(vals)
        weights = np.abs(vals - np.nanmedian(vals[mask])) if np.any(mask) else np.ones_like(vals)
    elif mode == "signal":
        sig = np.asarray(signal, dtype=float)
        mask &= np.isfinite(sig) & (sig > 0.0)
        weights = sig
    else:
        weights = np.ones_like(x, dtype=float)

    if np.count_nonzero(mask) < 3:
        return 0.0

    weights = np.asarray(weights, dtype=float)
    weights = np.where(mask & np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if not np.any(weights > 0.0):
        weights = np.where(mask, 1.0, 0.0)
    weights = weights / np.sum(weights)

    x0 = float(np.sum(weights * x))
    y0 = float(np.sum(weights * y))
    dx = x - x0
    dy = y - y0
    cov = np.array(
        [
            [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
            [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
        ],
        dtype=float,
    )
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, int(np.argmax(eigvals))]
    angle = np.rad2deg(np.arctan2(vec[1], vec[0]))
    return normalize_axis_angle(float(angle))


def map_coordinates_from_header(header: fits.Header, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    pixsize = float(header.get("PIXSIZE", 1.0))
    center_row = float(header.get("CENROW", (ny + 1) / 2.0)) - 1.0
    center_col = float(header.get("CENCOL", (nx + 1) / 2.0)) - 1.0
    row, col = np.indices(shape, dtype=float)
    return (col - center_col) * pixsize, (row - center_row) * pixsize


def measure_signal_geometry(
    data: SignalMapData,
    principal_axis_percentile: float,
    center_mode: str,
) -> SignalGeometry:
    x = np.asarray(data.x, dtype=float)
    y = np.asarray(data.y, dtype=float)
    signal = np.asarray(data.signal, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(signal) & (signal > 0.0)
    if not np.any(good):
        raise ValueError(f"{data.label} signal map has no positive finite pixels")

    good_indices = np.flatnonzero(good.ravel())
    signal_flat = signal.ravel()
    brightest_flat = good_indices[np.nanargmax(signal_flat[good_indices])]
    brightest_index = np.unravel_index(brightest_flat, signal.shape)
    brightest_x = float(x[brightest_index])
    brightest_y = float(y[brightest_index])
    brightest_signal = float(signal[brightest_index])

    threshold = float(np.nanpercentile(signal[good], principal_axis_percentile))
    axis_mask = good & (signal >= threshold)
    if np.count_nonzero(axis_mask) < 3:
        axis_mask = good

    weights = np.where(axis_mask, signal, 0.0).astype(float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if not np.any(weights > 0.0):
        weights = np.where(axis_mask, 1.0, 0.0)
    weights /= np.sum(weights)

    centroid_x = float(np.sum(weights * x))
    centroid_y = float(np.sum(weights * y))
    dx = x - centroid_x
    dy = y - centroid_y
    cov = np.array(
        [
            [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
            [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
        ],
        dtype=float,
    )
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, int(np.argmax(eigvals))]
    pa = normalize_axis_angle(float(np.rad2deg(np.arctan2(vec[1], vec[0]))))

    if center_mode == "brightest":
        center_x = brightest_x
        center_y = brightest_y
    elif center_mode == "centroid":
        center_x = centroid_x
        center_y = centroid_y
    elif center_mode == "origin":
        center_x = 0.0
        center_y = 0.0
    else:
        raise ValueError(f"Unknown center mode '{center_mode}'")

    return SignalGeometry(
        label=data.label,
        center_mode=center_mode,
        center_x=float(center_x),
        center_y=float(center_y),
        brightest_x=brightest_x,
        brightest_y=brightest_y,
        brightest_signal=brightest_signal,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        principal_axis_deg=pa,
        principal_axis_percentile=float(principal_axis_percentile),
        n_axis_pixels=int(np.count_nonzero(axis_mask)),
    )


def load_nirspec_signal_map(path: Path) -> SignalMapData:
    with fits.open(path, memmap=True) as hdul:
        signal = np.asarray(hdul["SIGNAL_MAP"].data, dtype=float)
        x, y = map_coordinates_from_header(hdul[0].header, signal.shape)
    return SignalMapData(label="NIRSpec", x=x, y=y, signal=signal)


def build_spatial_alignment(
    muse_geometry: SignalGeometry,
    nirspec_geometry: SignalGeometry,
    mode: str,
    manual_rotation_deg: float,
) -> SpatialAlignment:
    if mode == "signal":
        base_rotation = normalize_axis_angle(
            muse_geometry.principal_axis_deg - nirspec_geometry.principal_axis_deg
        )
    elif mode == "manual":
        base_rotation = float(manual_rotation_deg)
    else:
        raise ValueError(f"Unknown spatial alignment mode '{mode}'")

    rotated_nirspec_center_x, rotated_nirspec_center_y = rotate_xy(
        np.array([nirspec_geometry.center_x]),
        np.array([nirspec_geometry.center_y]),
        base_rotation,
    )
    shift_x = float(muse_geometry.center_x - rotated_nirspec_center_x[0])
    shift_y = float(muse_geometry.center_y - rotated_nirspec_center_y[0])
    return SpatialAlignment(
        mode=mode,
        muse_center_x=float(muse_geometry.center_x),
        muse_center_y=float(muse_geometry.center_y),
        nirspec_center_x=float(nirspec_geometry.center_x),
        nirspec_center_y=float(nirspec_geometry.center_y),
        muse_principal_axis_deg=float(muse_geometry.principal_axis_deg),
        nirspec_principal_axis_deg=float(nirspec_geometry.principal_axis_deg),
        base_rotation_deg=float(base_rotation),
        final_rotation_deg=float(base_rotation),
        center_shift_x_arcsec=shift_x,
        center_shift_y_arcsec=shift_y,
    )


def with_final_rotation(alignment: SpatialAlignment, final_rotation_deg: float) -> SpatialAlignment:
    rotated_center_x, rotated_center_y = rotate_xy(
        np.array([alignment.nirspec_center_x]),
        np.array([alignment.nirspec_center_y]),
        final_rotation_deg,
    )
    shift_x = float(alignment.muse_center_x - rotated_center_x[0])
    shift_y = float(alignment.muse_center_y - rotated_center_y[0])
    return SpatialAlignment(
        mode=alignment.mode,
        muse_center_x=alignment.muse_center_x,
        muse_center_y=alignment.muse_center_y,
        nirspec_center_x=alignment.nirspec_center_x,
        nirspec_center_y=alignment.nirspec_center_y,
        muse_principal_axis_deg=alignment.muse_principal_axis_deg,
        nirspec_principal_axis_deg=alignment.nirspec_principal_axis_deg,
        base_rotation_deg=alignment.base_rotation_deg,
        final_rotation_deg=float(final_rotation_deg),
        center_shift_x_arcsec=shift_x,
        center_shift_y_arcsec=shift_y,
    )


def load_muse_product(path: Path) -> MuseProduct:
    with fits.open(path, memmap=True) as hdul:
        spax = hdul["SPAXELS"].data
        row = np.asarray(spax["ROW"], dtype=int) - 1
        col = np.asarray(spax["COL"], dtype=int) - 1
        x = np.asarray(spax["X_ARCSEC"], dtype=float)
        y = np.asarray(spax["Y_ARCSEC"], dtype=float)
        signal = np.asarray(spax["SIGNAL"], dtype=float)

        values = {q.name: np.asarray(spax[q.muse_column], dtype=float) for q in QUANTITIES}
        errors = {q.name: np.asarray(spax[q.muse_error_column], dtype=float) for q in QUANTITIES}
        maps = {q.name: np.asarray(hdul[q.muse_map].data, dtype=float) for q in QUANTITIES}
        error_maps = {q.name: np.asarray(hdul[q.muse_error_map].data, dtype=float) for q in QUANTITIES}
        signal_map = np.asarray(hdul["SIGNAL_MAP"].data, dtype=float)

        map_shape = maps["vlos"].shape
        x_map = np.full(map_shape, np.nan, dtype=float)
        y_map = np.full(map_shape, np.nan, dtype=float)
        good = (row >= 0) & (row < map_shape[0]) & (col >= 0) & (col < map_shape[1])
        x_map[row[good], col[good]] = x[good]
        y_map[row[good], col[good]] = y[good]
        x_grid = np.nanmedian(x_map, axis=0)
        y_grid = np.nanmedian(y_map, axis=1)

    if not np.all(np.isfinite(x_grid)) or not np.all(np.isfinite(y_grid)):
        raise ValueError(f"Could not infer a regular MUSE x/y grid from {path}")

    return MuseProduct(
        path=path,
        x=x,
        y=y,
        signal=signal,
        signal_map=signal_map,
        x_map=x_map,
        y_map=y_map,
        values=values,
        errors=errors,
        maps=maps,
        error_maps=error_maps,
        x_grid=x_grid,
        y_grid=y_grid,
    )


def load_nirspec_product(
    path: Path,
    rotation_deg: float,
    goodfit_only: bool,
    source_center: tuple[float, float] = (0.0, 0.0),
    target_center: tuple[float, float] = (0.0, 0.0),
) -> NirspecProduct:
    with fits.open(path, memmap=True) as hdul:
        tab = hdul["BIN_RESULTS"].data
        x_native = np.asarray(tab["X"], dtype=float)
        y_native = np.asarray(tab["Y"], dtype=float)
        x, y = transform_xy(
            x_native,
            y_native,
            rotation_deg,
            source_center=source_center,
            target_center=target_center,
        )
        values = {q.name: np.asarray(tab[q.nirspec_column], dtype=float) for q in QUANTITIES}
        errors = {q.name: np.asarray(tab[q.nirspec_error_column], dtype=float) for q in QUANTITIES}
        signal = np.asarray(tab["SN"], dtype=float) if "SN" in tab.names else np.ones_like(x)
        if "GOODFIT" in tab.names:
            goodfit = np.asarray(tab["GOODFIT"], dtype=bool)
        else:
            goodfit = np.ones_like(x, dtype=bool)

    if not goodfit_only:
        goodfit = np.ones_like(goodfit, dtype=bool)

    return NirspecProduct(
        path=path,
        x_native=x_native,
        y_native=y_native,
        x=x,
        y=y,
        signal=signal,
        goodfit=goodfit,
        values=values,
        errors=errors,
    )


def regular_grid_sample(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    data: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    fill_nearest: bool,
) -> np.ndarray:
    x_order = np.argsort(x_grid)
    y_order = np.argsort(y_grid)
    x_sorted = x_grid[x_order]
    y_sorted = y_grid[y_order]
    data_sorted = data[np.ix_(y_order, x_order)]
    points = np.column_stack([y_query, x_query])

    linear = RegularGridInterpolator(
        (y_sorted, x_sorted),
        data_sorted,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    sampled = np.asarray(linear(points), dtype=float)
    if fill_nearest and np.any(~np.isfinite(sampled)):
        nearest = RegularGridInterpolator(
            (y_sorted, x_sorted),
            data_sorted,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )
        fill = np.asarray(nearest(points), dtype=float)
        bad = ~np.isfinite(sampled) & np.isfinite(fill)
        sampled[bad] = fill[bad]
    return sampled


def sample_muse_at_nirspec(
    muse: MuseProduct,
    nirspec: NirspecProduct,
    fill_nearest: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = {}
    errors = {}
    for q in QUANTITIES:
        values[q.name] = regular_grid_sample(
            muse.x_grid,
            muse.y_grid,
            muse.maps[q.name],
            nirspec.x,
            nirspec.y,
            fill_nearest=fill_nearest,
        )
        errors[q.name] = regular_grid_sample(
            muse.x_grid,
            muse.y_grid,
            muse.error_maps[q.name],
            nirspec.x,
            nirspec.y,
            fill_nearest=fill_nearest,
        )
    return values, errors


def evaluate_vlos_orientation(
    muse: MuseProduct,
    nirspec_path: Path,
    rotation_deg: float,
    goodfit_only: bool,
    fill_nearest: bool,
    clip_sigma: float,
    error_mode: str,
    source_center: tuple[float, float],
    target_center: tuple[float, float],
) -> tuple[NirspecProduct, dict[str, np.ndarray], dict[str, np.ndarray], OrientationCandidate]:
    nirspec = load_nirspec_product(
        nirspec_path,
        rotation_deg=rotation_deg,
        goodfit_only=goodfit_only,
        source_center=source_center,
        target_center=target_center,
    )
    muse_at_nirspec, muse_errors_at_nirspec = sample_muse_at_nirspec(
        muse,
        nirspec,
        fill_nearest=fill_nearest,
    )
    goodfit = np.asarray(nirspec.goodfit, dtype=bool)
    x = muse_at_nirspec["vlos"][goodfit]
    y = nirspec.values["vlos"][goodfit]
    yerr = combined_error(
        nirspec.errors["vlos"][goodfit],
        muse_errors_at_nirspec["vlos"][goodfit],
        error_mode,
    )
    fit, used = weighted_linear_fit(x, y, yerr, clip_sigma=clip_sigma)
    finite = np.isfinite(x) & np.isfinite(y)
    pearson_r = np.nan
    if np.count_nonzero(finite) >= 2:
        pearson_r = float(np.corrcoef(x[finite], y[finite])[0, 1])
    candidate = OrientationCandidate(
        rotation_deg=float(rotation_deg),
        vlos_multiplier=float(fit.multiplier),
        vlos_additive=float(fit.additive),
        pearson_r=pearson_r,
        n_initial=int(fit.n_initial),
        n_used=int(np.count_nonzero(used)),
        rms_used=float(fit.rms_used),
    )
    return nirspec, muse_at_nirspec, muse_errors_at_nirspec, candidate


def choose_vlos_orientation(
    candidates: list[
        tuple[NirspecProduct, dict[str, np.ndarray], dict[str, np.ndarray], OrientationCandidate]
    ],
) -> tuple[NirspecProduct, dict[str, np.ndarray], dict[str, np.ndarray], list[OrientationCandidate]]:
    if not candidates:
        raise ValueError("No VLOS orientation candidates were supplied")

    def sort_key(item):
        candidate = item[3]
        positive = candidate.vlos_multiplier > 0.0 and candidate.pearson_r > 0.0
        corr = candidate.pearson_r if np.isfinite(candidate.pearson_r) else -np.inf
        return (positive, corr, -candidate.rms_used)

    selected_item = max(candidates, key=sort_key)
    selected_rotation = selected_item[3].rotation_deg
    marked_candidates = [
        OrientationCandidate(
            rotation_deg=item[3].rotation_deg,
            vlos_multiplier=item[3].vlos_multiplier,
            vlos_additive=item[3].vlos_additive,
            pearson_r=item[3].pearson_r,
            n_initial=item[3].n_initial,
            n_used=item[3].n_used,
            rms_used=item[3].rms_used,
            selected=item[3].rotation_deg == selected_rotation,
        )
        for item in candidates
    ]
    return selected_item[0], selected_item[1], selected_item[2], marked_candidates


def combined_error(
    nirspec_err: np.ndarray,
    muse_err: np.ndarray,
    mode: str,
) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "nirspec":
        return np.asarray(nirspec_err, dtype=float)
    return np.sqrt(np.square(nirspec_err) + np.square(muse_err))


def fit_calibrations(
    nirspec: NirspecProduct,
    muse_at_nirspec: dict[str, np.ndarray],
    muse_errors_at_nirspec: dict[str, np.ndarray],
    clip_sigma: float,
    error_mode: str,
) -> tuple[dict[str, FitResult], dict[str, np.ndarray]]:
    results: dict[str, FitResult] = {}
    used_masks: dict[str, np.ndarray] = {}
    for q in QUANTITIES:
        yerr = combined_error(nirspec.errors[q.name], muse_errors_at_nirspec[q.name], error_mode)
        fit, used = weighted_linear_fit(
            muse_at_nirspec[q.name],
            nirspec.values[q.name],
            yerr,
            clip_sigma=clip_sigma,
        )
        results[q.name] = FitResult(
            quantity=q.name,
            label=q.label,
            multiplier=fit.multiplier,
            additive=fit.additive,
            multiplier_err=fit.multiplier_err,
            additive_err=fit.additive_err,
            n_initial=fit.n_initial,
            n_used=fit.n_used,
            rms_used=fit.rms_used,
            robust_sigma_used=fit.robust_sigma_used,
            median_residual_used=fit.median_residual_used,
            clip_sigma=fit.clip_sigma,
        )
        used_masks[q.name] = used
    return results, used_masks


def apply_calibration(values: np.ndarray, fit: FitResult) -> np.ndarray:
    return fit.multiplier * np.asarray(values, dtype=float) + fit.additive


def write_calibrated_muse_fits(
    muse_path: Path,
    output_path: Path,
    fits_by_quantity: dict[str, FitResult],
    spatial_alignment: SpatialAlignment,
    muse_geometry: SignalGeometry,
    nirspec_geometry: SignalGeometry,
    auto_vlos_180_align: bool,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    with fits.open(muse_path, memmap=False) as hdul:
        for q in QUANTITIES:
            fit = fits_by_quantity[q.name]
            for table_name in ("BIN_RESULTS", "SPAXELS"):
                if table_name in hdul and q.muse_column in hdul[table_name].data.names:
                    hdul[table_name].data[q.muse_column] = apply_calibration(
                        hdul[table_name].data[q.muse_column],
                        fit,
                    )
                if table_name in hdul and q.muse_error_column in hdul[table_name].data.names:
                    hdul[table_name].data[q.muse_error_column] = (
                        abs(fit.multiplier) * np.asarray(hdul[table_name].data[q.muse_error_column], dtype=float)
                    )
            if q.muse_map in hdul:
                hdul[q.muse_map].data = apply_calibration(hdul[q.muse_map].data, fit).astype(np.float32)
            if q.muse_error_map in hdul:
                hdul[q.muse_error_map].data = (
                    abs(fit.multiplier) * np.asarray(hdul[q.muse_error_map].data, dtype=float)
                ).astype(np.float32)

        primary = hdul[0].header
        primary["M2NCAL"] = (True, "MUSE kinematics affine-calibrated to NIRSpec")
        primary["M2NROTB"] = (float(spatial_alignment.base_rotation_deg), "Base NIRSpec-to-MUSE rotation deg")
        primary["M2NROTF"] = (float(spatial_alignment.final_rotation_deg), "Final NIRSpec-to-MUSE rotation deg")
        primary["M2NAUTO"] = (bool(auto_vlos_180_align), "Auto-tested 180 deg VLOS orientation")
        primary["M2NMCX"] = (float(muse_geometry.center_x), "MUSE alignment center x arcsec")
        primary["M2NMCY"] = (float(muse_geometry.center_y), "MUSE alignment center y arcsec")
        primary["M2NNCX"] = (float(nirspec_geometry.center_x), "NIRSpec alignment center x arcsec")
        primary["M2NNCY"] = (float(nirspec_geometry.center_y), "NIRSpec alignment center y arcsec")
        primary["M2NMPA"] = (float(muse_geometry.principal_axis_deg), "MUSE signal principal axis deg")
        primary["M2NNPA"] = (float(nirspec_geometry.principal_axis_deg), "NIRSpec signal principal axis deg")
        primary["M2NTX"] = (float(spatial_alignment.center_shift_x_arcsec), "NIRSpec-to-MUSE x translation arcsec")
        primary["M2NTY"] = (float(spatial_alignment.center_shift_y_arcsec), "NIRSpec-to-MUSE y translation arcsec")
        primary.add_history(
            f"NIRSpec coordinates used rotation {spatial_alignment.final_rotation_deg:.8g} deg "
            f"(base {spatial_alignment.base_rotation_deg:.8g} deg)."
        )
        primary.add_history(
            f"Spatial alignment centers: MUSE ({muse_geometry.center_x:.8g}, "
            f"{muse_geometry.center_y:.8g}), NIRSpec ({nirspec_geometry.center_x:.8g}, "
            f"{nirspec_geometry.center_y:.8g})."
        )
        for q in QUANTITIES:
            fit = fits_by_quantity[q.name]
            prefix = {"vlos": "M2NV", "sigma": "M2NS", "h3": "M2H3", "h4": "M2H4"}[q.name]
            primary[f"{prefix}M"] = (float(fit.multiplier), f"{q.label} multiplier")
            primary[f"{prefix}A"] = (float(fit.additive), f"{q.label} additive")
            primary.add_history(
                f"{q.label}: calibrated as {fit.multiplier:.8g} * MUSE + {fit.additive:.8g}"
            )

        calib_table = Table(
            {
                "quantity": [fit.quantity for fit in fits_by_quantity.values()],
                "label": [fit.label for fit in fits_by_quantity.values()],
                "multiplier": [fit.multiplier for fit in fits_by_quantity.values()],
                "additive": [fit.additive for fit in fits_by_quantity.values()],
                "multiplier_err": [fit.multiplier_err for fit in fits_by_quantity.values()],
                "additive_err": [fit.additive_err for fit in fits_by_quantity.values()],
                "n_initial": [fit.n_initial for fit in fits_by_quantity.values()],
                "n_used": [fit.n_used for fit in fits_by_quantity.values()],
                "rms_used": [fit.rms_used for fit in fits_by_quantity.values()],
                "robust_sigma_used": [fit.robust_sigma_used for fit in fits_by_quantity.values()],
                "median_residual_used": [fit.median_residual_used for fit in fits_by_quantity.values()],
                "clip_sigma": [fit.clip_sigma for fit in fits_by_quantity.values()],
            }
        )
        hdul.append(fits.BinTableHDU(calib_table, name="M2N_CALIB"))
        hdul.writeto(output_path, overwrite=overwrite)


def profile_edges(coord: np.ndarray, bin_width: float) -> np.ndarray:
    coord = np.asarray(coord, dtype=float)
    finite = coord[np.isfinite(coord)]
    if finite.size == 0:
        return np.array([], dtype=float)
    lo = np.floor(np.nanmin(finite) / bin_width) * bin_width
    hi = np.ceil(np.nanmax(finite) / bin_width) * bin_width
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.array([], dtype=float)
    return np.arange(lo, hi + 1.5 * bin_width, bin_width)


def binned_profile(
    coord: np.ndarray,
    values: np.ndarray,
    bin_width: float,
    min_per_bin: int,
) -> list[tuple[float, float, float, float, int]]:
    edges = profile_edges(coord, bin_width)
    if edges.size < 2:
        return []
    coord = np.asarray(coord, dtype=float)
    values = np.asarray(values, dtype=float)
    rows: list[tuple[float, float, float, float, int]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (coord >= lo) & (coord < hi) & np.isfinite(values)
        n = int(np.count_nonzero(mask))
        if n < min_per_bin:
            continue
        vals = values[mask]
        rows.append(
            (
                float(0.5 * (lo + hi)),
                float(np.nanmedian(vals)),
                float(np.nanpercentile(vals, 16.0)),
                float(np.nanpercentile(vals, 84.0)),
                n,
            )
        )
    return rows


def build_profile_rows(
    muse: MuseProduct,
    nirspec: NirspecProduct,
    fits_by_quantity: dict[str, FitResult],
    used_masks: dict[str, np.ndarray],
    axis_weight_mode: str,
    slit_half_width_arcsec: float,
    radial_bin_width_arcsec: float,
    min_per_bin: int,
    padding_arcsec: float,
    axis_angle_deg: float | None,
) -> tuple[list[ProfileRow], dict[str, float]]:
    rows: list[ProfileRow] = []
    angles: dict[str, float] = {}

    xlim = (
        float(np.nanmin(nirspec.x) - padding_arcsec),
        float(np.nanmax(nirspec.x) + padding_arcsec),
    )
    ylim = (
        float(np.nanmin(nirspec.y) - padding_arcsec),
        float(np.nanmax(nirspec.y) + padding_arcsec),
    )
    muse_region = (
        np.isfinite(muse.x)
        & np.isfinite(muse.y)
        & (muse.x >= xlim[0])
        & (muse.x <= xlim[1])
        & (muse.y >= ylim[0])
        & (muse.y <= ylim[1])
    )

    for q in QUANTITIES:
        fit = fits_by_quantity[q.name]
        if axis_angle_deg is None:
            used = used_masks[q.name]
            angle = make_weighted_axis_angle(
                nirspec.x[used],
                nirspec.y[used],
                nirspec.values[q.name][used],
                nirspec.signal[used],
                mode=axis_weight_mode,
            )
        else:
            angle = normalize_axis_angle(axis_angle_deg)
        angles[q.name] = angle

        datasets = [
            ("NIRSpec", nirspec.x, nirspec.y, nirspec.values[q.name], nirspec.goodfit),
            ("MUSE raw", muse.x, muse.y, muse.values[q.name], muse_region),
            (
                "MUSE calibrated",
                muse.x,
                muse.y,
                apply_calibration(muse.values[q.name], fit),
                muse_region,
            ),
        ]
        for label, x, y, values, base_mask in datasets:
            major_coord, minor_coord = project_to_angle(x, y, angle)
            axis_specs = {
                "major": (major_coord, np.abs(minor_coord) <= slit_half_width_arcsec),
                "minor": (minor_coord, np.abs(major_coord) <= slit_half_width_arcsec),
            }
            for axis, (coord, axis_mask) in axis_specs.items():
                mask = np.asarray(base_mask, dtype=bool) & axis_mask & np.isfinite(values)
                for radius, med, p16, p84, n in binned_profile(
                    coord[mask],
                    np.asarray(values, dtype=float)[mask],
                    radial_bin_width_arcsec,
                    min_per_bin,
                ):
                    rows.append(
                        ProfileRow(
                            quantity=q.name,
                            dataset=label,
                            axis=axis,
                            principal_angle_deg=float(angle),
                            radius_arcsec=radius,
                            median=med,
                            p16=p16,
                            p84=p84,
                            n=n,
                        )
                    )
    return rows, angles


def rows_to_arrays(
    rows: list[ProfileRow],
    quantity: str,
    dataset: str,
    axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [r for r in rows if r.quantity == quantity and r.dataset == dataset and r.axis == axis]
    selected.sort(key=lambda item: item.radius_arcsec)
    return (
        np.array([r.radius_arcsec for r in selected], dtype=float),
        np.array([r.median for r in selected], dtype=float),
        np.array([r.p16 for r in selected], dtype=float),
        np.array([r.p84 for r in selected], dtype=float),
        np.array([r.n for r in selected], dtype=int),
    )


def plot_axis_profiles(
    outpath: Path,
    rows: list[ProfileRow],
    angles: dict[str, float],
    fits_by_quantity: dict[str, FitResult],
    slit_half_width_arcsec: float,
) -> None:
    fig, axes = plt.subplots(
        len(QUANTITIES),
        2,
        figsize=(14.0, 14.0),
        constrained_layout=True,
        sharex=False,
    )
    style = {
        "NIRSpec": {"color": "black", "marker": "o", "lw": 1.1, "ms": 3.5, "alpha": 0.95},
        "MUSE raw": {"color": "tab:blue", "marker": "s", "lw": 1.0, "ms": 3.0, "alpha": 0.7},
        "MUSE calibrated": {"color": "tab:orange", "marker": "^", "lw": 1.2, "ms": 3.2, "alpha": 0.9},
    }
    for row_index, q in enumerate(QUANTITIES):
        fit = fits_by_quantity[q.name]
        ylabel = f"{q.label} [{q.unit}]" if q.unit else q.label
        for col_index, axis in enumerate(("major", "minor")):
            ax = axes[row_index, col_index]
            if q.name == "vlos" or q.name in {"h3", "h4"}:
                ax.axhline(0.0, color="0.25", lw=0.7, alpha=0.4)
            for dataset, kws in style.items():
                radius, med, p16, p84, _n = rows_to_arrays(rows, q.name, dataset, axis)
                if radius.size == 0:
                    continue
                ax.plot(radius, med, label=dataset, **kws)
                ax.fill_between(radius, p16, p84, color=kws["color"], alpha=0.08, lw=0)
            ax.grid(alpha=0.25)
            ax.set_xlabel(f"{axis} coordinate [arcsec]")
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{q.label} {axis}; PA={angles[q.name]:.1f} deg; "
                f"MUSE -> NIRSpec: {fit.multiplier:.3g}x + {fit.additive:.3g}",
                fontsize=10,
            )
            if row_index == 0 and col_index == 0:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"MUSE-to-NIRSpec principal-axis profiles | slit half-width {slit_half_width_arcsec:.2f} arcsec",
        fontsize=13,
    )
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fit_diagnostics(
    outpath: Path,
    nirspec: NirspecProduct,
    muse_at_nirspec: dict[str, np.ndarray],
    fits_by_quantity: dict[str, FitResult],
    used_masks: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), constrained_layout=True)
    for ax, q in zip(axes.ravel(), QUANTITIES):
        x = np.asarray(muse_at_nirspec[q.name], dtype=float)
        y = np.asarray(nirspec.values[q.name], dtype=float)
        used = used_masks[q.name]
        finite = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[finite & ~used], y[finite & ~used], s=18, color="0.75", alpha=0.7, label="clipped")
        ax.scatter(x[used], y[used], s=18, color="tab:blue", alpha=0.85, label="used")
        fit = fits_by_quantity[q.name]
        x_line = np.linspace(safe_percentile(x[finite], 1.0), safe_percentile(x[finite], 99.0), 100)
        y_line = apply_calibration(x_line, fit)
        ax.plot(x_line, y_line, color="tab:orange", lw=1.8, label="affine fit")
        lims = [
            min(safe_percentile(x[finite], 1.0), safe_percentile(y[finite], 1.0)),
            max(safe_percentile(x[finite], 99.0), safe_percentile(y[finite], 99.0)),
        ]
        if np.all(np.isfinite(lims)) and lims[0] < lims[1]:
            ax.plot(lims, lims, color="black", lw=0.8, alpha=0.35, label="1:1")
        xlabel = f"MUSE {q.label} at NIRSpec bins"
        ylabel = f"NIRSpec {q.label}"
        if q.unit:
            xlabel += f" [{q.unit}]"
            ylabel += f" [{q.unit}]"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{q.label}: {fit.multiplier:.4g}x + {fit.additive:.4g}; "
            f"N={fit.n_used}/{fit.n_initial}; rms={fit.rms_used:.3g}",
            fontsize=10,
        )
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("MUSE-to-NIRSpec affine calibration fits", fontsize=13)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def asinh_scaled(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    floor = float(np.nanpercentile(finite, 1.0))
    scale = float(np.nanpercentile(finite, 99.0) - floor)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.nanstd(finite))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return np.arcsinh(np.clip(values - floor, 0.0, None) / scale)


def axis_line(
    center_x: float,
    center_y: float,
    angle_deg: float,
    half_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.array([-half_length, half_length], dtype=float)
    theta = np.deg2rad(angle_deg)
    return center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)


def plot_spatial_alignment(
    outpath: Path,
    muse: MuseProduct,
    nirspec_signal: SignalMapData,
    muse_geometry: SignalGeometry,
    nirspec_geometry: SignalGeometry,
    spatial_alignment: SpatialAlignment,
) -> None:
    nirspec_x_aligned, nirspec_y_aligned = transform_xy(
        nirspec_signal.x,
        nirspec_signal.y,
        spatial_alignment.final_rotation_deg,
        source_center=(nirspec_geometry.center_x, nirspec_geometry.center_y),
        target_center=(muse_geometry.center_x, muse_geometry.center_y),
    )
    finite_muse = np.isfinite(muse.signal_map) & np.isfinite(muse.x_map) & np.isfinite(muse.y_map)
    finite_nirspec = np.isfinite(nirspec_signal.signal) & np.isfinite(nirspec_x_aligned) & np.isfinite(nirspec_y_aligned)
    nirspec_show = finite_nirspec & (
        nirspec_signal.signal >= np.nanpercentile(nirspec_signal.signal[finite_nirspec], 50.0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), constrained_layout=True)

    ax = axes[0]
    sc = ax.scatter(
        muse.x_map[finite_muse],
        muse.y_map[finite_muse],
        c=asinh_scaled(muse.signal_map[finite_muse]),
        s=2,
        marker="s",
        cmap="gray_r",
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        nirspec_x_aligned[nirspec_show],
        nirspec_y_aligned[nirspec_show],
        c=asinh_scaled(nirspec_signal.signal[nirspec_show]),
        s=18,
        marker="s",
        cmap="viridis",
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    half_length = max(
        1.0,
        0.45
        * max(
            float(np.nanmax(nirspec_x_aligned[finite_nirspec]) - np.nanmin(nirspec_x_aligned[finite_nirspec])),
            float(np.nanmax(nirspec_y_aligned[finite_nirspec]) - np.nanmin(nirspec_y_aligned[finite_nirspec])),
        ),
    )
    mx, my = axis_line(
        muse_geometry.center_x,
        muse_geometry.center_y,
        muse_geometry.principal_axis_deg,
        half_length,
    )
    nx, ny = axis_line(
        muse_geometry.center_x,
        muse_geometry.center_y,
        normalize_axis_angle(nirspec_geometry.principal_axis_deg + spatial_alignment.final_rotation_deg),
        half_length,
    )
    ax.plot(mx, my, color="tab:red", lw=1.5, label="MUSE signal PA")
    ax.plot(nx, ny, color="tab:green", lw=1.2, ls="--", label="NIRSpec transformed PA")
    ax.scatter(
        [muse_geometry.center_x],
        [muse_geometry.center_y],
        marker="+",
        s=150,
        color="tab:red",
        linewidths=2.0,
        label="MUSE center",
    )
    ax.scatter(
        [muse_geometry.center_x],
        [muse_geometry.center_y],
        marker="x",
        s=90,
        color="tab:green",
        linewidths=1.8,
        label="NIRSpec center after transform",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_title("Aligned Signal Footprints")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="MUSE asinh signal")

    ax = axes[1]
    native_good = np.isfinite(nirspec_signal.signal) & np.isfinite(nirspec_signal.x) & np.isfinite(nirspec_signal.y)
    ax.scatter(
        nirspec_signal.x[native_good],
        nirspec_signal.y[native_good],
        c=asinh_scaled(nirspec_signal.signal[native_good]),
        s=18,
        marker="s",
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    native_half_length = max(
        1.0,
        0.45
        * max(
            float(np.nanmax(nirspec_signal.x[native_good]) - np.nanmin(nirspec_signal.x[native_good])),
            float(np.nanmax(nirspec_signal.y[native_good]) - np.nanmin(nirspec_signal.y[native_good])),
        ),
    )
    npx, npy = axis_line(
        nirspec_geometry.center_x,
        nirspec_geometry.center_y,
        nirspec_geometry.principal_axis_deg,
        native_half_length,
    )
    ax.plot(npx, npy, color="tab:green", lw=1.5, label="NIRSpec signal PA")
    ax.scatter(
        [nirspec_geometry.center_x],
        [nirspec_geometry.center_y],
        marker="+",
        s=150,
        color="tab:green",
        linewidths=2.0,
        label="NIRSpec center",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Native NIRSpec X [arcsec]")
    ax.set_ylabel("Native NIRSpec Y [arcsec]")
    ax.set_title(
        "Native NIRSpec Signal Geometry\n"
        f"rotation={spatial_alignment.final_rotation_deg:.2f} deg, "
        f"translation=({spatial_alignment.center_shift_x_arcsec:.3f}, "
        f"{spatial_alignment.center_shift_y_arcsec:.3f}) arcsec"
    )
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Signal-Based Spatial Alignment Before Kinematic Scaling\n"
        f"MUSE PA={muse_geometry.principal_axis_deg:.2f} deg, "
        f"NIRSpec PA={nirspec_geometry.principal_axis_deg:.2f} deg",
        fontsize=13,
    )
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_profile_csv(path: Path, rows: list[ProfileRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)


def write_fit_samples_csv(
    path: Path,
    nirspec: NirspecProduct,
    muse_at_nirspec: dict[str, np.ndarray],
    muse_errors_at_nirspec: dict[str, np.ndarray],
    fits_by_quantity: dict[str, FitResult],
    used_masks: dict[str, np.ndarray],
) -> None:
    fieldnames = [
        "quantity",
        "bin_index",
        "x_arcsec",
        "y_arcsec",
        "nirspec_value",
        "muse_raw_at_nirspec",
        "muse_calibrated_at_nirspec",
        "nirspec_error",
        "muse_error_at_nirspec",
        "goodfit",
        "used_in_fit",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for q in QUANTITIES:
            fit = fits_by_quantity[q.name]
            calibrated = apply_calibration(muse_at_nirspec[q.name], fit)
            for idx in range(nirspec.x.size):
                writer.writerow(
                    {
                        "quantity": q.name,
                        "bin_index": idx,
                        "x_arcsec": float(nirspec.x[idx]),
                        "y_arcsec": float(nirspec.y[idx]),
                        "nirspec_value": float(nirspec.values[q.name][idx]),
                        "muse_raw_at_nirspec": float(muse_at_nirspec[q.name][idx]),
                        "muse_calibrated_at_nirspec": float(calibrated[idx]),
                        "nirspec_error": float(nirspec.errors[q.name][idx]),
                        "muse_error_at_nirspec": float(muse_errors_at_nirspec[q.name][idx]),
                        "goodfit": bool(nirspec.goodfit[idx]),
                        "used_in_fit": bool(used_masks[q.name][idx]),
                    }
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Affine-calibrate MUSE kinematics to NIRSpec")
    parser.add_argument("--muse-fits", type=Path, default=DEFAULT_MUSE_FITS)
    parser.add_argument("--nirspec-fits", type=Path, default=DEFAULT_NIRSPEC_FITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-fits", type=Path, default=None)
    parser.add_argument(
        "--nirspec-rotation-deg",
        type=float,
        default=-18.0,
        help=(
            "Manual base rotation applied to NIRSpec coordinates before matching to MUSE. "
            "Used when --spatial-align-mode=manual. In signal mode, the base rotation is "
            "fit from the MUSE and NIRSpec signal-map principal axes."
        ),
    )
    parser.add_argument(
        "--spatial-align-mode",
        choices=("signal", "manual"),
        default="signal",
        help=(
            "How to spatially align NIRSpec to MUSE before kinematic scaling. "
            "'signal' fits the brightest signal pixel and signal principal axis; "
            "'manual' uses --nirspec-rotation-deg with the chosen center mode."
        ),
    )
    parser.add_argument(
        "--alignment-center-mode",
        choices=("brightest", "centroid", "origin"),
        default="brightest",
        help="Center used for signal-map alignment. Default uses the brightest pixel.",
    )
    parser.add_argument(
        "--principal-axis-signal-percentile",
        type=float,
        default=90.0,
        help="Signal percentile used to fit the photometric principal axis. Default: top 10 percent.",
    )
    parser.add_argument(
        "--no-auto-vlos-180-align",
        dest="auto_vlos_180_align",
        action="store_false",
        help="Disable the pre-fit VLOS 180-degree orientation check.",
    )
    parser.set_defaults(auto_vlos_180_align=True)
    parser.add_argument("--clip-sigma", type=float, default=4.0)
    parser.add_argument("--error-mode", choices=("combined", "nirspec", "none"), default="combined")
    parser.add_argument("--include-badfits", action="store_true")
    parser.add_argument("--no-nearest-fill", action="store_true")
    parser.add_argument(
        "--axis-weight-mode",
        choices=("quantity", "signal", "footprint"),
        default="quantity",
        help="Weights used to infer the principal axis for each profile row.",
    )
    parser.add_argument(
        "--axis-angle-deg",
        type=float,
        default=None,
        help="Override the inferred profile axis angle with a fixed angle in the MUSE frame.",
    )
    parser.add_argument("--slit-half-width-arcsec", type=float, default=0.20)
    parser.add_argument("--radial-bin-width-arcsec", type=float, default=0.20)
    parser.add_argument("--min-per-bin", type=int, default=1)
    parser.add_argument("--profile-padding-arcsec", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    muse_path = args.muse_fits.expanduser().resolve()
    nirspec_path = args.nirspec_fits.expanduser().resolve()
    outdir = ensure_dir(args.output_dir.expanduser().resolve())
    output_fits = args.output_fits
    if output_fits is None:
        output_fits = outdir / f"{muse_path.stem}_nirspec_calibrated.fits"
    else:
        output_fits = output_fits.expanduser().resolve()

    muse = load_muse_product(muse_path)
    nirspec_signal = load_nirspec_signal_map(nirspec_path)
    principal_axis_percentile = float(args.principal_axis_signal_percentile)
    muse_signal = SignalMapData(
        label="MUSE",
        x=muse.x_map,
        y=muse.y_map,
        signal=muse.signal_map,
    )
    muse_geometry = measure_signal_geometry(
        muse_signal,
        principal_axis_percentile=principal_axis_percentile,
        center_mode=str(args.alignment_center_mode),
    )
    nirspec_geometry = measure_signal_geometry(
        nirspec_signal,
        principal_axis_percentile=principal_axis_percentile,
        center_mode=str(args.alignment_center_mode),
    )
    spatial_alignment = build_spatial_alignment(
        muse_geometry,
        nirspec_geometry,
        mode=str(args.spatial_align_mode),
        manual_rotation_deg=float(args.nirspec_rotation_deg),
    )
    base_rotation = float(spatial_alignment.base_rotation_deg)
    rotations_to_test = [base_rotation]
    if bool(args.auto_vlos_180_align):
        rotations_to_test.append(base_rotation + 180.0)
    orientation_trials = [
        evaluate_vlos_orientation(
            muse,
            nirspec_path,
            rotation_deg=rotation,
            goodfit_only=not bool(args.include_badfits),
            fill_nearest=not bool(args.no_nearest_fill),
            clip_sigma=float(args.clip_sigma),
            error_mode=str(args.error_mode),
            source_center=(nirspec_geometry.center_x, nirspec_geometry.center_y),
            target_center=(muse_geometry.center_x, muse_geometry.center_y),
        )
        for rotation in rotations_to_test
    ]
    nirspec, muse_at_nirspec, muse_errors_at_nirspec, orientation_candidates = choose_vlos_orientation(
        orientation_trials
    )
    selected_rotation = next(item.rotation_deg for item in orientation_candidates if item.selected)
    spatial_alignment = with_final_rotation(spatial_alignment, selected_rotation)

    goodfit = np.asarray(nirspec.goodfit, dtype=bool)
    nirspec_for_fit = NirspecProduct(
        path=nirspec.path,
        x_native=nirspec.x_native[goodfit],
        y_native=nirspec.y_native[goodfit],
        x=nirspec.x[goodfit],
        y=nirspec.y[goodfit],
        signal=nirspec.signal[goodfit],
        goodfit=nirspec.goodfit[goodfit],
        values={key: val[goodfit] for key, val in nirspec.values.items()},
        errors={key: val[goodfit] for key, val in nirspec.errors.items()},
    )
    muse_fit_values = {key: val[goodfit] for key, val in muse_at_nirspec.items()}
    muse_fit_errors = {key: val[goodfit] for key, val in muse_errors_at_nirspec.items()}

    fits_by_quantity, used_fit_masks_small = fit_calibrations(
        nirspec_for_fit,
        muse_fit_values,
        muse_fit_errors,
        clip_sigma=float(args.clip_sigma),
        error_mode=str(args.error_mode),
    )
    used_masks: dict[str, np.ndarray] = {}
    good_indices = np.flatnonzero(goodfit)
    for key, small_mask in used_fit_masks_small.items():
        full = np.zeros_like(goodfit, dtype=bool)
        full[good_indices] = small_mask
        used_masks[key] = full

    write_calibrated_muse_fits(
        muse_path,
        output_fits,
        fits_by_quantity,
        spatial_alignment=spatial_alignment,
        muse_geometry=muse_geometry,
        nirspec_geometry=nirspec_geometry,
        auto_vlos_180_align=bool(args.auto_vlos_180_align),
        overwrite=bool(args.overwrite),
    )

    profile_rows, angles = build_profile_rows(
        muse,
        nirspec,
        fits_by_quantity,
        used_masks,
        axis_weight_mode=str(args.axis_weight_mode),
        slit_half_width_arcsec=float(args.slit_half_width_arcsec),
        radial_bin_width_arcsec=float(args.radial_bin_width_arcsec),
        min_per_bin=max(1, int(args.min_per_bin)),
        padding_arcsec=float(args.profile_padding_arcsec),
        axis_angle_deg=args.axis_angle_deg,
    )

    profile_csv = outdir / "muse_to_nirspec_principal_axis_profiles.csv"
    fit_samples_csv = outdir / "muse_to_nirspec_fit_samples.csv"
    summary_json = outdir / "muse_to_nirspec_calibration_summary.json"
    axis_plot = outdir / "muse_to_nirspec_principal_axis_profiles.png"
    fit_plot = outdir / "muse_to_nirspec_fit_diagnostics.png"
    spatial_plot = outdir / "muse_to_nirspec_spatial_alignment.png"

    write_profile_csv(profile_csv, profile_rows)
    write_fit_samples_csv(
        fit_samples_csv,
        nirspec,
        muse_at_nirspec,
        muse_errors_at_nirspec,
        fits_by_quantity,
        used_masks,
    )
    plot_axis_profiles(
        axis_plot,
        profile_rows,
        angles,
        fits_by_quantity,
        slit_half_width_arcsec=float(args.slit_half_width_arcsec),
    )
    plot_fit_diagnostics(
        fit_plot,
        nirspec,
        muse_at_nirspec,
        fits_by_quantity,
        used_masks,
    )
    plot_spatial_alignment(
        spatial_plot,
        muse,
        nirspec_signal,
        muse_geometry,
        nirspec_geometry,
        spatial_alignment,
    )

    summary = {
        "muse_fits": str(muse_path),
        "nirspec_fits": str(nirspec_path),
        "output_fits": str(output_fits),
        "spatial_alignment": asdict(spatial_alignment),
        "muse_signal_geometry": asdict(muse_geometry),
        "nirspec_signal_geometry": asdict(nirspec_geometry),
        "base_nirspec_rotation_deg": spatial_alignment.base_rotation_deg,
        "final_nirspec_rotation_deg": spatial_alignment.final_rotation_deg,
        "nirspec_rotation_deg": spatial_alignment.final_rotation_deg,
        "auto_vlos_180_align": bool(args.auto_vlos_180_align),
        "orientation_candidates": [asdict(item) for item in orientation_candidates],
        "goodfit_only": not bool(args.include_badfits),
        "clip_sigma": float(args.clip_sigma),
        "error_mode": str(args.error_mode),
        "spatial_align_mode": str(args.spatial_align_mode),
        "alignment_center_mode": str(args.alignment_center_mode),
        "principal_axis_signal_percentile": principal_axis_percentile,
        "axis_weight_mode": str(args.axis_weight_mode),
        "axis_angle_override_deg": args.axis_angle_deg,
        "slit_half_width_arcsec": float(args.slit_half_width_arcsec),
        "radial_bin_width_arcsec": float(args.radial_bin_width_arcsec),
        "profile_principal_angles_deg": angles,
        "fits": {key: asdict(value) for key, value in fits_by_quantity.items()},
        "products": {
            "profile_csv": str(profile_csv),
            "fit_samples_csv": str(fit_samples_csv),
            "axis_profile_plot": str(axis_plot),
            "fit_diagnostics_plot": str(fit_plot),
            "spatial_alignment_plot": str(spatial_plot),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[muse-to-nirspec] Calibrated FITS : {output_fits}")
    print(f"[muse-to-nirspec] Summary         : {summary_json}")
    print(f"[muse-to-nirspec] Fit samples     : {fit_samples_csv}")
    print(f"[muse-to-nirspec] Axis profiles   : {profile_csv}")
    print(f"[muse-to-nirspec] Axis plot       : {axis_plot}")
    print(f"[muse-to-nirspec] Fit plot        : {fit_plot}")
    print(f"[muse-to-nirspec] Spatial plot    : {spatial_plot}")
    print(
        "[muse-to-nirspec] Signal alignment: "
        f"MUSE center=({muse_geometry.center_x:.4f}, {muse_geometry.center_y:.4f}), "
        f"PA={muse_geometry.principal_axis_deg:.3f} deg; "
        f"NIRSpec center=({nirspec_geometry.center_x:.4f}, {nirspec_geometry.center_y:.4f}), "
        f"PA={nirspec_geometry.principal_axis_deg:.3f} deg"
    )
    for candidate in orientation_candidates:
        marker = "selected" if candidate.selected else "rejected"
        print(
            f"[muse-to-nirspec] VLOS orientation {candidate.rotation_deg:.3f} deg "
            f"({marker}): slope={candidate.vlos_multiplier:.6g}, "
            f"r={candidate.pearson_r:.4f}, rms={candidate.rms_used:.4g}, "
            f"N={candidate.n_used}/{candidate.n_initial}"
        )
    for q in QUANTITIES:
        fit = fits_by_quantity[q.name]
        print(
            f"[muse-to-nirspec] {q.label:5s}: "
            f"NIRSpec = {fit.multiplier:.8g} * MUSE + {fit.additive:.8g} "
            f"(N={fit.n_used}/{fit.n_initial}, rms={fit.rms_used:.4g})"
        )


if __name__ == "__main__":
    main()
