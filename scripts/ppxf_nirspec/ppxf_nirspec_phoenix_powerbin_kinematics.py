#!/usr/bin/env python3
"""
Fit NIRSpec stellar kinematics with PHOENIX templates after PowerBin spatial binning.

This runner reuses the constant-R PHOENIX pPXF setup from
`ppxf_nirspec_phoenix_kinematics.py`, but fits spectra coadded in adaptive
PowerBin bins instead of fitting every spaxel independently.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_phoenix_powerbin"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from powerbin import PowerBin
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import ppxf_nirspec_phoenix_kinematics as base


DEFAULT_OUTPUT_DIR = base.ROOT / "Data/ppxf_nirspec/antoine_wicked_powerbin_sn120_constant_r2700"
DEFAULT_WICKED_CUBE = (
    base.ROOT
    / "Data/IFU/antoine/sombrero_nirspec_1150_p1293_g235_wicked.fits"
)
DEFAULT_MGE_SOLUTION_PATH = base.ROOT / "Data/mge_NAGN_0deg_pa_positive_gauss/mge_solution.csv"
DEFAULT_MGE_LUMINOSITY_PATH = base.ROOT / "Data/mge_NAGN_0deg_pa_positive_gauss/mge_luminosity_table.csv"
# MGE contours are drawn in the native IFU plotting frame by default. Use the
# CLI option below only when the photometric MGE and cube frames differ.
DEFAULT_MGE_CONTOUR_ROTATION_DEG = 0.0
DEFAULT_HIGH_SIGMA_CHECKPLOTS = 5
FIT_WORKER_STATE: dict[str, object] = {}


def parse_args() -> tuple[base.Config, float, int, Path, Path, float]:
    parser = argparse.ArgumentParser(
        description="Fit PHOENIX pPXF kinematics on PowerBin-binned NIRSpec spectra.",
    )
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_WICKED_CUBE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phoenix-dir", type=Path, default=base.DEFAULT_PHOENIX_DIR)
    parser.add_argument("--phoenix-wave-path", type=Path, default=base.DEFAULT_PHOENIX_WAVE)
    parser.add_argument("--template-list", type=str, default=None)
    parser.add_argument("--target-sn", type=float, default=120.0)
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
    parser.add_argument("--n-plot-bins", type=int, default=8)
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help="Number of worker processes for independent PowerBin pPXF fits. Use 1 for serial execution.",
    )
    parser.add_argument("--max-spaxels", type=int, default=None)
    parser.add_argument("--template-velocity-margin-kms", type=float, default=1500.0)
    parser.add_argument("--mge-solution-path", type=Path, default=DEFAULT_MGE_SOLUTION_PATH)
    parser.add_argument("--mge-luminosity-path", type=Path, default=DEFAULT_MGE_LUMINOSITY_PATH)
    parser.add_argument(
        "--mge-contour-rotation-deg",
        type=float,
        default=DEFAULT_MGE_CONTOUR_ROTATION_DEG,
        help=(
            "Rotation applied to photometric MGE coordinates before overlaying contours "
            "on the native IFU frame. Default is 0 deg."
        ),
    )
    parser.add_argument(
        "--wave-crpix-mode",
        choices=("fits", "first_pixel"),
        default="fits",
        help=(
            "How to interpret the spectral WCS. Use 'first_pixel' for SINFONI cubes "
            "whose CRVAL3 is the first wavelength sample."
        ),
    )
    args = parser.parse_args()

    cfg = base.Config(
        cube_path=args.cube_path.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        phoenix_dir=args.phoenix_dir.expanduser().resolve(),
        phoenix_wave_path=args.phoenix_wave_path.expanduser().resolve(),
        template_list_path=base.parse_optional_path(args.template_list),
        redshift=float(args.redshift),
        resolving_power=float(args.resolving_power),
        template_resolving_power=float(args.template_resolving_power),
        fit_windows_rest_um=base.parse_windows(args.fit_windows_rest_um),
        mask_windows_rest_um=base.parse_windows(args.mask_windows_rest_um),
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
        n_plot_spaxels=int(args.n_plot_bins),
        max_spaxels=args.max_spaxels,
        template_velocity_margin_kms=float(args.template_velocity_margin_kms),
        wave_crpix_mode=str(args.wave_crpix_mode),
    )
    return (
        cfg,
        float(args.target_sn),
        max(1, int(args.n_processes)),
        args.mge_solution_path.expanduser().resolve(),
        args.mge_luminosity_path.expanduser().resolve(),
        float(args.mge_contour_rotation_deg),
    )


def positive_signal_for_binning(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    good = np.isfinite(signal) & (signal > 0)
    if np.any(good):
        fill = float(np.nanmedian(signal[good]))
        return np.where(good, signal, fill)
    return np.ones_like(signal, dtype=float)


def make_power_bins(cube_data: base.CubeData, target_sn: float):
    signal = positive_signal_for_binning(cube_data.signal)
    noise = base.safe_positive(cube_data.noise_proxy)
    xy = np.column_stack([cube_data.x, cube_data.y])

    def capacity(index):
        return float(np.sum(signal[index]) / np.sqrt(np.sum(noise[index] ** 2)))

    powbin = PowerBin(
        xy,
        capacity,
        target_sn,
        pixelsize=cube_data.pixsize_arcsec,
        verbose=1,
    )
    bin_num = np.asarray(powbin.bin_num, dtype=int)
    bin_sn = np.full(np.max(bin_num) + 1, np.nan, dtype=float)
    for bid in np.unique(bin_num):
        idx = np.flatnonzero(bin_num == bid)
        bin_sn[bid] = capacity(idx)
    return powbin, bin_num, bin_sn


def stack_powerbin_spectrum(
    spectra_log: np.ndarray,
    valid_frac_log: np.ndarray,
    idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coadd all spaxels assigned to one PowerBin."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        raise ValueError("Cannot stack an empty PowerBin")
    galaxy = np.nanmean(spectra_log[:, idx], axis=1)
    valid_frac = np.nanmean(valid_frac_log[:, idx], axis=1)
    return galaxy, valid_frac


def init_fit_worker(
    templates: np.ndarray,
    spectra_log: np.ndarray,
    valid_frac_log: np.ndarray,
    fit_mask_log: np.ndarray,
    velscale: float,
    lam_gal_ang: np.ndarray,
    lam_temp: np.ndarray,
    cfg: base.Config,
    start_sigma: float,
) -> None:
    FIT_WORKER_STATE.clear()
    FIT_WORKER_STATE.update(
        {
            "templates": templates,
            "spectra_log": spectra_log,
            "valid_frac_log": valid_frac_log,
            "fit_mask_log": fit_mask_log,
            "velscale": float(velscale),
            "lam_gal_ang": lam_gal_ang,
            "lam_temp": lam_temp,
            "cfg": cfg,
            "start_sigma": float(start_sigma),
        }
    )


def fit_powerbin_worker(task: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, base.FitResult]:
    bid, idx = task
    spectra_log = FIT_WORKER_STATE["spectra_log"]
    valid_frac_log = FIT_WORKER_STATE["valid_frac_log"]
    fit_mask_log = FIT_WORKER_STATE["fit_mask_log"]
    cfg = FIT_WORKER_STATE["cfg"]
    galaxy, valid_frac = stack_powerbin_spectrum(spectra_log, valid_frac_log, idx)
    fit_mask = fit_mask_log & (valid_frac >= cfg.min_log_pixel_fraction)
    result = base.ppxf_fit_spectrum(
        FIT_WORKER_STATE["templates"],
        galaxy,
        FIT_WORKER_STATE["velscale"],
        [0.0, FIT_WORKER_STATE["start_sigma"]],
        fit_mask,
        FIT_WORKER_STATE["lam_gal_ang"],
        FIT_WORKER_STATE["lam_temp"],
        cfg,
    )
    return int(bid), idx, result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "BIN_ID",
        "X",
        "Y",
        "NSPAX",
        "SN_TARGET",
        "SN",
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
        "CHI2",
        "GOODPIX_FRAC",
        "GOODFIT",
        "MESSAGE",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def make_powerbin_maps(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    maps = base.make_maps(shape)
    maps["BIN_ID_MAP"] = np.full(shape, -1, dtype=int)
    maps["SN_TARGET_MAP"] = np.full(shape, np.nan, dtype=float)
    maps["NSPAX_MAP"] = np.full(shape, np.nan, dtype=float)
    return maps


def plot_powerbin_maps(outdir: Path, cube_data: base.CubeData, maps: dict[str, np.ndarray]) -> Path:
    extent = (
        cube_data.x.min() - 0.5 * cube_data.pixsize_arcsec,
        cube_data.x.max() + 0.5 * cube_data.pixsize_arcsec,
        cube_data.y.min() - 0.5 * cube_data.pixsize_arcsec,
        cube_data.y.max() + 0.5 * cube_data.pixsize_arcsec,
    )
    panels = [
        ("BIN_ID_MAP", "PowerBin ID", "tab20"),
        ("SN_TARGET_MAP", "PowerBin S/N", "viridis"),
        ("VREL_MAP", "V - systemic [km/s]", "RdBu_r"),
        ("SIGMA_MAP", "sigma [km/s]", "inferno"),
        ("VRMS_MAP", "Vrms [km/s]", "magma"),
        ("H3_MAP", "h3", "RdBu_r"),
        ("H4_MAP", "h4", "RdBu_r"),
        ("GOODFIT_MAP", "good fit", "gray"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, title, cmap) in zip(axes, panels):
        im = ax.imshow(maps[key], origin="lower", extent=extent, cmap=cmap, aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("X [arcsec]")
        ax.set_ylabel("Y [arcsec]")
    outpath = outdir / "phoenix_powerbin_kinematics_maps.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def read_named_csv(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if arr.size == 0:
        return None
    arr = np.atleast_1d(arr)
    return {name: np.asarray(arr[name], dtype=float) for name in arr.dtype.names or ()}


def load_mge_for_contours(
    solution_path: Path = DEFAULT_MGE_SOLUTION_PATH,
    luminosity_path: Path = DEFAULT_MGE_LUMINOSITY_PATH,
    pixel_scale_arcsec: float = 0.031,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    lum = read_named_csv(luminosity_path)
    if lum is not None and {"luminosity_Lsun", "sigma_arcsec", "q_obs"} <= set(lum):
        sigma = np.asarray(lum["sigma_arcsec"], dtype=float)
        q_obs = np.asarray(lum["q_obs"], dtype=float)
        total = np.asarray(lum["luminosity_Lsun"], dtype=float)
        surf = total / (2.0 * np.pi * sigma**2 * q_obs)
        return surf, sigma, q_obs

    sol = read_named_csv(solution_path)
    if sol is None or not {"total_counts", "sigma_pix", "q_obs"} <= set(sol):
        return None

    sigma_pix = np.asarray(sol["sigma_pix"], dtype=float)
    q_obs = np.asarray(sol["q_obs"], dtype=float)
    sigma = sigma_pix * float(pixel_scale_arcsec)
    total = np.asarray(sol["total_counts"], dtype=float)
    surf = total / (2.0 * np.pi * sigma**2 * q_obs)
    return surf, sigma, q_obs


def rotate_points(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    angle = np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def evaluate_mge_contours(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    mge_model: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    rotation_deg: float = DEFAULT_MGE_CONTOUR_ROTATION_DEG,
) -> np.ndarray | None:
    if mge_model is None:
        return None
    surf, sigma, q_obs = mge_model
    x_mge, y_mge = rotate_points(x_grid, y_grid, rotation_deg)
    image = np.zeros_like(x_grid, dtype=float)
    for amp, sig, q in zip(surf, sigma, q_obs):
        if not np.isfinite(amp) or not np.isfinite(sig) or not np.isfinite(q) or sig <= 0 or q <= 0:
            continue
        image += amp * np.exp(-0.5 * (x_mge**2 + (y_mge / q) ** 2) / sig**2)
    return image


def add_mge_contours(
    ax,
    extent: tuple[float, float, float, float],
    mge_model: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    rotation_deg: float = DEFAULT_MGE_CONTOUR_ROTATION_DEG,
) -> None:
    if mge_model is None:
        ax.text(
            0.03,
            0.97,
            "MGE contours unavailable",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.2",
        )
        return

    x_grid = np.linspace(extent[0], extent[1], 180)
    y_grid = np.linspace(extent[2], extent[3], 180)
    xx, yy = np.meshgrid(x_grid, y_grid)
    model = evaluate_mge_contours(xx, yy, mge_model, rotation_deg=rotation_deg)
    if model is None:
        return
    good = np.isfinite(model) & (model > 0)
    if np.count_nonzero(good) < 10:
        return
    log_model = np.full_like(model, np.nan, dtype=float)
    log_model[good] = np.log10(model[good])
    levels = np.linspace(
        float(np.nanpercentile(log_model[good], 15.0)),
        float(np.nanpercentile(log_model[good], 98.5)),
        8,
    )
    if np.unique(levels).size < 2:
        return
    ax.contour(xx, yy, log_model, levels=levels, colors="k", linewidths=0.6, alpha=0.75)


def format_fit_value(row: dict[str, object], key: str, err_key: str | None = None) -> str:
    value = float(row.get(key, np.nan))
    if err_key is None:
        return f"{value:.3f}" if np.isfinite(value) else "nan"
    err = float(row.get(err_key, np.nan))
    if np.isfinite(value) and np.isfinite(err):
        return f"{value:.1f}+/-{err:.1f}"
    return f"{value:.1f}" if np.isfinite(value) else "nan"


def write_sigma_bin_checkplot(
    outpath: Path,
    cube_data: base.CubeData,
    maps: dict[str, np.ndarray],
    bin_id: int,
    idx: np.ndarray,
    result: base.FitResult,
    row: dict[str, object],
    mge_model: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    mge_rotation_deg: float = DEFAULT_MGE_CONTOUR_ROTATION_DEG,
) -> list[Path]:
    extent = (
        cube_data.x.min() - 0.5 * cube_data.pixsize_arcsec,
        cube_data.x.max() + 0.5 * cube_data.pixsize_arcsec,
        cube_data.y.min() - 0.5 * cube_data.pixsize_arcsec,
        cube_data.y.max() + 0.5 * cube_data.pixsize_arcsec,
    )
    galaxy, valid_frac = stack_powerbin_spectrum(cube_data.spectra_log, cube_data.valid_frac_log, idx)
    fit_mask = result.clean_mask if result.clean_mask is not None else np.isfinite(galaxy)
    galaxy_norm, _ = base.normalize_galaxy(galaxy, fit_mask)
    residual = galaxy_norm - result.bestfit if result.bestfit is not None else np.full_like(galaxy_norm, np.nan)
    n_stack = int(idx.size)

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45], height_ratios=[2.4, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_resid = fig.add_subplot(gs[1, 1], sharex=ax_spec)

    signal = np.asarray(cube_data.signal_map, dtype=float)
    signal_good = np.isfinite(signal) & (signal > 0)
    background = np.full_like(signal, np.nan, dtype=float)
    background[signal_good] = np.log10(signal[signal_good])
    im = ax_map.imshow(background, origin="lower", extent=extent, cmap="Greys", aspect="equal")
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label="log signal")
    add_mge_contours(ax_map, extent, mge_model, rotation_deg=mge_rotation_deg)

    selected_mask = maps["BIN_ID_MAP"] == int(bin_id)
    if np.any(selected_mask):
        ax_map.contour(
            selected_mask.astype(float),
            levels=[0.5],
            origin="lower",
            extent=extent,
            colors="tab:red",
            linewidths=1.8,
        )
    ax_map.scatter(
        cube_data.x[idx],
        cube_data.y[idx],
        s=42,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.1,
        label=f"bin {int(bin_id)} ({n_stack} spaxels)",
    )
    ax_map.scatter([0.0], [0.0], marker="+", s=80, color="tab:blue", linewidths=1.5, label="center")
    ax_map.set_xlabel("X [arcsec]")
    ax_map.set_ylabel("Y [arcsec]")
    ax_map.set_title(
        f"High-sigma rank bin {int(bin_id)} location | "
        f"sigma={float(row.get('sigma', np.nan)):.1f} km/s"
    )
    ax_map.legend(loc="upper right", fontsize=8, frameon=True)

    bad = ~fit_mask
    ax_spec.plot(
        cube_data.lam_gal_ang,
        galaxy_norm,
        lw=0.9,
        color="0.2",
        label=f"Stacked galaxy (mean of {n_stack} spaxels)",
    )
    if result.bestfit is not None:
        ax_spec.plot(cube_data.lam_gal_ang, result.bestfit, lw=1.0, color="tab:blue", label="pPXF model")
    if np.any(bad):
        ax_spec.scatter(
            cube_data.lam_gal_ang[bad],
            galaxy_norm[bad],
            s=5,
            color="tab:orange",
            alpha=0.45,
            label="masked",
        )
    ax_spec.set_ylabel("Normalized flux")
    ax_spec.legend(loc="best", fontsize=8)

    goodfit = bool(row.get("GOODFIT", False))
    ax_spec.set_title(
        " | ".join(
            [
                f"ranked high-sigma bin {int(bin_id)}",
                f"GOODFIT={goodfit}",
                f"Nstack={n_stack}",
                f"PowerBin S/N={float(row.get('SN_TARGET', np.nan)):.1f}",
                f"pPXF S/N={float(row.get('SN', np.nan)):.1f}",
                f"chi2={float(row.get('CHI2', np.nan)):.3f}",
            ]
        )
    )
    stats_text = "\n".join(
        [
            f"Valid log-pixel frac = {float(np.nanmedian(valid_frac)):.2f}",
            f"VLOS = {format_fit_value(row, 'LOSV', 'LOSV_err')} km/s",
            f"Vrel = {format_fit_value(row, 'V_REL_KMS', 'V_REL_ERR_KMS')} km/s",
            f"sigma = {format_fit_value(row, 'sigma', 'sigma_err')} km/s",
            f"h3 = {format_fit_value(row, 'h3', 'h3_err')}",
            f"h4 = {format_fit_value(row, 'h4', 'h4_err')}",
        ]
    )
    ax_spec.text(
        0.015,
        0.05,
        stats_text,
        transform=ax_spec.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    ax_resid.axhline(0.0, color="0.3", lw=0.8)
    ax_resid.plot(cube_data.lam_gal_ang, residual, lw=0.75, color="tab:purple")
    if result.goodpixels is not None and result.goodpixels.size:
        good = np.asarray(result.goodpixels, dtype=int)
        ax_resid.scatter(cube_data.lam_gal_ang[good], residual[good], s=4, color="tab:purple", alpha=0.35)
    ax_resid.set_xlabel("Rest wavelength [Angstrom]")
    ax_resid.set_ylabel("Galaxy - model")

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [outpath]


def write_high_sigma_checkplots(
    outdir: Path,
    cube_data: base.CubeData,
    maps: dict[str, np.ndarray],
    bin_num: np.ndarray,
    rows: list[dict[str, object]],
    idx_by_bin: dict[int, np.ndarray],
    result_by_bin: dict[int, base.FitResult],
    *,
    prefix: str = "phoenix_powerbin",
    n_plots: int = DEFAULT_HIGH_SIGMA_CHECKPLOTS,
    mge_solution_path: Path = DEFAULT_MGE_SOLUTION_PATH,
    mge_luminosity_path: Path = DEFAULT_MGE_LUMINOSITY_PATH,
    mge_rotation_deg: float = DEFAULT_MGE_CONTOUR_ROTATION_DEG,
) -> list[Path]:
    candidates = [
        row
        for row in rows
        if int(row["BIN_ID"]) in idx_by_bin
        and int(row["BIN_ID"]) in result_by_bin
        and result_by_bin[int(row["BIN_ID"])].ok
        and np.isfinite(float(row.get("sigma", np.nan)))
    ]
    candidates.sort(key=lambda item: float(item["sigma"]), reverse=True)
    selected = candidates[: max(0, int(n_plots))]
    if not selected:
        return []

    check_dir = outdir / "high_sigma_checkplots"
    check_dir.mkdir(parents=True, exist_ok=True)
    mge_model = load_mge_for_contours(mge_solution_path, mge_luminosity_path)

    paths: list[Path] = []
    for rank, row in enumerate(selected, start=1):
        bid = int(row["BIN_ID"])
        outpath = check_dir / f"{prefix}_high_sigma_rank{rank:02d}_bin{bid:04d}_fit.png"
        paths.extend(write_sigma_bin_checkplot(
            outpath,
            cube_data,
            maps,
            bid,
            idx_by_bin[bid],
            result_by_bin[bid],
            row,
            mge_model,
            mge_rotation_deg=mge_rotation_deg,
        ))
    return paths


def save_fits(
    path: Path,
    cfg: base.Config,
    target_sn: float,
    cube_data: base.CubeData,
    library: base.TemplateLibrary,
    table_rows: list[dict[str, object]],
    maps: dict[str, np.ndarray],
) -> None:
    hdr = fits.Header()
    hdr["OBJECT"] = cfg.cube_path.stem[:68]
    hdr["BINNING"] = "POWERBIN"
    hdr["TARSN"] = float(target_sn)
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
    hdr["NBINS"] = len(table_rows)
    hdr["LAMMIN"] = float(np.nanmin(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["LAMMAX"] = float(np.nanmax(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["PIXSIZE"] = (float(cube_data.pixsize_arcsec), "arcsec")
    hdr["CENROW"] = int(cube_data.center_row + 1)
    hdr["CENCOL"] = int(cube_data.center_col + 1)

    table_keys = [
        "BIN_ID",
        "X",
        "Y",
        "NSPAX",
        "SN_TARGET",
        "SN",
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
        "CHI2",
        "GOODPIX_FRAC",
        "GOODFIT",
    ]
    cols = []
    for key in table_keys:
        values = np.array([row[key] for row in table_rows])
        if values.dtype.kind == "b":
            fmt = "L"
        elif np.issubdtype(values.dtype, np.integer):
            fmt = "J"
        else:
            fmt = "D"
        cols.append(fits.Column(name=key.upper(), format=fmt, array=values))
    msg = np.array([str(row["MESSAGE"])[:80] for row in table_rows])
    cols.append(fits.Column(name="MESSAGE", format="80A", array=msg))
    kin_hdu = fits.BinTableHDU.from_columns(cols, name="BIN_RESULTS")

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
        fits.ImageHDU(data=maps["BIN_ID_MAP"].astype(np.int16), name="BIN_ID_MAP"),
        fits.ImageHDU(data=maps["VREL_MAP"].astype(np.float32), name="VREL_MAP"),
        fits.ImageHDU(data=maps["LOSV_MAP"].astype(np.float32), name="LOSV_MAP"),
        fits.ImageHDU(data=maps["SIGMA_MAP"].astype(np.float32), name="SIGMA_MAP"),
        fits.ImageHDU(data=maps["VRMS_MAP"].astype(np.float32), name="VRMS_MAP"),
        fits.ImageHDU(data=maps["H3_MAP"].astype(np.float32), name="H3_MAP"),
        fits.ImageHDU(data=maps["H4_MAP"].astype(np.float32), name="H4_MAP"),
        fits.ImageHDU(data=maps["SN_MAP"].astype(np.float32), name="SN_MAP"),
        fits.ImageHDU(data=maps["SN_TARGET_MAP"].astype(np.float32), name="SN_TARGET_MAP"),
        fits.ImageHDU(data=maps["NSPAX_MAP"].astype(np.float32), name="NSPAX_MAP"),
        fits.ImageHDU(data=maps["CHI2_MAP"].astype(np.float32), name="CHI2_MAP"),
        fits.ImageHDU(data=maps["GOODFIT_MAP"].astype(np.int16), name="GOODFIT_MAP"),
        fits.ImageHDU(data=cube_data.signal_map.astype(np.float32), name="SIGNAL_MAP"),
        fits.ImageHDU(data=cube_data.noise_proxy_map.astype(np.float32), name="NOISE_MAP"),
        fits.ImageHDU(data=cube_data.lam_gal_ang.astype(np.float32), name="LAMBDA_REST"),
        fits.ImageHDU(data=cube_data.fit_mask_log.astype(np.int16), name="FIT_MASK"),
        fits.ImageHDU(data=library.lam_temp.astype(np.float32), name="TEMPLATE_LAMBDA"),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def main() -> None:
    (
        cfg,
        target_sn,
        n_processes,
        mge_solution_path,
        mge_luminosity_path,
        mge_contour_rotation_deg,
    ) = parse_args()
    outdir = base.ensure_dir(cfg.output_dir)

    print(f"Reading cube: {cfg.cube_path}")
    cube_data = base.read_cube_data(cfg)
    print(
        f"Log-rebinned cube: {cube_data.spectra_log.shape[0]} pixels, "
        f"{cube_data.spectra_log.shape[1]} spaxels, velscale={cube_data.velscale:.3f} km/s"
    )

    print(f"Building PowerBin bins with target S/N={target_sn:.1f}")
    powbin, bin_num, bin_sn = make_power_bins(cube_data, target_sn)
    unique_bins = np.unique(bin_num)
    print(f"PowerBin produced {unique_bins.size} bins from {bin_num.size} valid spaxels")
    try:
        powbin.plot(ylabel="S/N")
        plt.savefig(outdir / "phoenix_powerbin_sn_bins.png", dpi=180, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:
        print(f"WARNING: could not save PowerBin diagnostic plot: {exc}")

    print(f"Loading PHOENIX templates from: {cfg.phoenix_dir}")
    library = base.load_phoenix_templates(cfg, cube_data)
    print(
        f"Using {len(library.meta)} templates; convolved R={cfg.template_resolving_power:.0f} "
        f"to R={cfg.resolving_power:.0f} "
        f"(sigma_conv={library.sigma_conv_kms:.2f} km/s)"
    )

    n_global = max(1, int(np.ceil(cfg.top_template_frac * cube_data.spectra_log.shape[1])))
    global_sel = np.argsort(cube_data.signal)[-n_global:]
    global_spec = np.nanmean(cube_data.spectra_log[:, global_sel], axis=1)
    global_valid_frac = np.nanmean(cube_data.valid_frac_log[:, global_sel], axis=1)
    global_mask = cube_data.fit_mask_log & (global_valid_frac >= cfg.min_log_pixel_fraction)
    global_result = base.ppxf_fit_spectrum(
        library.templates,
        global_spec,
        cube_data.velscale,
        [0.0, cfg.start_sigma],
        global_mask,
        cube_data.lam_gal_ang,
        library.lam_temp,
        cfg,
    )
    base.plot_fit(
        outdir / "phoenix_powerbin_global_fit.png",
        cube_data.lam_gal_ang,
        global_spec,
        global_result,
        title=(
            f"Global PHOENIX PowerBin pPXF fit | "
            f"V={global_result.sol_rel[0]:.1f} km/s, "
            f"sigma={global_result.sigma:.1f} km/s"
            if global_result.sol_rel is not None
            else f"Global PHOENIX PowerBin pPXF fit failed: {global_result.message}"
        ),
    )
    start_sigma = global_result.sigma if np.isfinite(global_result.sigma) else cfg.start_sigma

    maps = make_powerbin_maps(cube_data.map_shape)
    rows: list[dict[str, object]] = []
    idx_by_bin: dict[int, np.ndarray] = {}
    result_by_bin: dict[int, base.FitResult] = {}
    preview_done = 0
    preview_ids = set(unique_bins[: max(cfg.n_plot_spaxels * 4, 20)])
    bin_tasks = [(int(bid), np.flatnonzero(bin_num == bid)) for bid in unique_bins]

    def record_result(bid: int, idx: np.ndarray, result: base.FitResult) -> None:
        nonlocal preview_done
        idx_by_bin[int(bid)] = np.asarray(idx, dtype=int)
        result_by_bin[int(bid)] = result
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

        row = {
            "BIN_ID": int(bid),
            "X": float(np.nanmean(cube_data.x[idx])),
            "Y": float(np.nanmean(cube_data.y[idx])),
            "NSPAX": int(idx.size),
            "SN_TARGET": float(bin_sn[bid]),
            "SN": float(result.sn),
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
            "CHI2": float(result.chi2),
            "GOODPIX_FRAC": goodpix_frac,
            "GOODFIT": goodfit,
            "MESSAGE": result.message,
        }
        rows.append(row)

        for j in idx:
            map_row = int(cube_data.row[j]) - 1
            map_col = int(cube_data.col[j]) - 1
            maps["BIN_ID_MAP"][map_row, map_col] = int(bid)
            maps["SN_TARGET_MAP"][map_row, map_col] = float(bin_sn[bid])
            maps["NSPAX_MAP"][map_row, map_col] = float(idx.size)
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

        if result.ok and preview_done < cfg.n_plot_spaxels and bid in preview_ids:
            galaxy, _ = stack_powerbin_spectrum(cube_data.spectra_log, cube_data.valid_frac_log, idx)
            base.plot_fit(
                outdir / f"phoenix_powerbin_fit_bin{int(bid):04d}.png",
                cube_data.lam_gal_ang,
                galaxy,
                result,
                title=(
                    f"PowerBin {int(bid)} stacked mean | N={idx.size}, "
                    f"PowerBin S/N={bin_sn[bid]:.1f}, pPXF S/N={result.sn:.1f}"
                ),
            )
            preview_done += 1

    if n_processes > 1:
        print(f"Fitting PowerBin spectra with {n_processes} worker processes")
        with ProcessPoolExecutor(
            max_workers=n_processes,
            initializer=init_fit_worker,
            initargs=(
                library.templates,
                cube_data.spectra_log,
                cube_data.valid_frac_log,
                cube_data.fit_mask_log,
                cube_data.velscale,
                cube_data.lam_gal_ang,
                library.lam_temp,
                cfg,
                start_sigma,
            ),
        ) as pool:
            for bid, idx, result in tqdm(
                pool.map(fit_powerbin_worker, bin_tasks, chunksize=1),
                total=len(bin_tasks),
                desc="Fitting PowerBin spectra",
            ):
                record_result(bid, idx, result)
    else:
        for bid, idx in tqdm(bin_tasks, desc="Fitting PowerBin spectra"):
            galaxy, valid_frac = stack_powerbin_spectrum(
                cube_data.spectra_log,
                cube_data.valid_frac_log,
                idx,
            )
            fit_mask = cube_data.fit_mask_log & (valid_frac >= cfg.min_log_pixel_fraction)
            result = base.ppxf_fit_spectrum(
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
            record_result(bid, idx, result)

    plot_powerbin_maps(outdir, cube_data, maps)
    high_sigma_check_paths = write_high_sigma_checkplots(
        outdir,
        cube_data,
        maps,
        bin_num,
        rows,
        idx_by_bin,
        result_by_bin,
        prefix="phoenix_powerbin",
        mge_solution_path=mge_solution_path,
        mge_luminosity_path=mge_luminosity_path,
        mge_rotation_deg=mge_contour_rotation_deg,
    )

    base_path = outdir / f"{cfg.cube_path.stem}_phoenix_powerbin_sn{int(round(target_sn))}_kinematics"
    csv_path = base_path.with_suffix(".csv")
    all_csv_path = base_path.with_name(base_path.name + "_all").with_suffix(".csv")
    fits_path = base_path.with_suffix(".fits")
    npz_path = base_path.with_suffix(".npz")
    json_path = outdir / "phoenix_powerbin_run_config.json"
    summary_path = outdir / "phoenix_powerbin_run_summary.txt"
    manifest_path = outdir / "selected_phoenix_templates.csv"

    good_rows = [row for row in rows if row["GOODFIT"]]
    write_csv(csv_path, good_rows)
    write_csv(all_csv_path, rows)
    base.write_template_manifest(manifest_path, library)
    save_fits(fits_path, cfg, target_sn, cube_data, library, rows, maps)
    np.savez_compressed(
        npz_path,
        lam_gal_ang=cube_data.lam_gal_ang,
        lam_temp=library.lam_temp,
        fit_mask_log=cube_data.fit_mask_log,
        x=cube_data.x,
        y=cube_data.y,
        row=cube_data.row,
        col=cube_data.col,
        bin_num=bin_num,
        bin_sn=bin_sn,
        table_rows=np.array(rows, dtype=object),
        bin_id_map=maps["BIN_ID_MAP"],
        vrel_map=maps["VREL_MAP"],
        losv_map=maps["LOSV_MAP"],
        sigma_map=maps["SIGMA_MAP"],
        vrms_map=maps["VRMS_MAP"],
        h3_map=maps["H3_MAP"],
        h4_map=maps["H4_MAP"],
        sn_map=maps["SN_MAP"],
        sn_target_map=maps["SN_TARGET_MAP"],
        nspax_map=maps["NSPAX_MAP"],
        chi2_map=maps["CHI2_MAP"],
        goodfit_map=maps["GOODFIT_MAP"],
        template_filenames=np.array([m.path.name for m in library.meta]),
        template_teff=np.array([m.teff for m in library.meta]),
        template_logg=np.array([m.logg for m in library.meta]),
        template_feh=np.array([m.feh for m in library.meta]),
    )

    config = base.config_to_json(cfg, library, cube_data)
    config.update(
        {
            "binning": "PowerBin",
            "target_sn": target_sn,
            "n_processes": n_processes,
            "pixel_size_arcsec": float(cube_data.pixsize_arcsec),
            "wave_obs_min_um": float(np.nanmin(cube_data.wave_obs_um)),
            "wave_obs_max_um": float(np.nanmax(cube_data.wave_obs_um)),
            "wave_rest_min_um": float(np.nanmin(cube_data.wave_rest_um)),
            "wave_rest_max_um": float(np.nanmax(cube_data.wave_rest_um)),
            "n_log_pixels": int(cube_data.spectra_log.shape[0]),
            "n_fit_log_pixels": int(np.count_nonzero(cube_data.fit_mask_log)),
            "n_bins": len(rows),
            "n_good_bins": len(good_rows),
            "high_sigma_checkplots": [str(path) for path in high_sigma_check_paths],
            "mge_solution_path": str(mge_solution_path),
            "mge_luminosity_path": str(mge_luminosity_path),
            "mge_contour_rotation_deg": float(mge_contour_rotation_deg),
        }
    )
    json_path.write_text(json.dumps(config, indent=2) + "\n")

    med_sn = float(np.nanmedian([row["SN"] for row in good_rows])) if good_rows else np.nan
    med_target_sn = float(np.nanmedian([row["SN_TARGET"] for row in rows])) if rows else np.nan
    med_sigma = float(np.nanmedian([row["sigma"] for row in good_rows])) if good_rows else np.nan
    summary_lines = [
        "PHOENIX pPXF stellar kinematics with PowerBin spatial binning",
        f"Cube: {cfg.cube_path}",
        f"Output dir: {cfg.output_dir}",
        f"Target PowerBin S/N: {target_sn:.1f}",
        f"Worker processes: {n_processes}",
        f"PowerBin bins: {len(rows)} from {bin_num.size} valid spaxels",
        f"Median PowerBin input S/N: {med_target_sn:.2f}",
        f"PHOENIX dir: {cfg.phoenix_dir}",
        f"PHOENIX wavelength file: {cfg.phoenix_wave_path}",
        f"Templates used: {len(library.meta)}",
        f"Resolving power: data R={cfg.resolving_power:.1f}, template R={cfg.template_resolving_power:.1f}",
        f"Template convolution sigma: {library.sigma_conv_kms:.3f} km/s ({library.sigma_conv_pix:.3f} pix)",
        f"pPXF setup: moments={cfg.moments}, degree={cfg.degree}, mdegree={cfg.mdegree}, bias={cfg.bias}, regul=0",
        f"Fit windows rest um: {cfg.fit_windows_rest_um}",
        f"Masked windows rest um: {cfg.mask_windows_rest_um}",
        f"MGE contour rotation on IFU frame: {mge_contour_rotation_deg:.2f} deg",
        f"Good-fit bins: {len(good_rows)}",
        f"Median good-fit pPXF S/N: {med_sn:.2f}",
        f"Median good-fit sigma: {med_sigma:.2f} km/s",
        f"High-sigma check plots: {len(high_sigma_check_paths)} in {outdir / 'high_sigma_checkplots'}",
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
    print(f"Good-fit bins: {len(good_rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
