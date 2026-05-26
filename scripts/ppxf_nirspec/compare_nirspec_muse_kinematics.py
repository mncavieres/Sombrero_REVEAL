#!/usr/bin/env python3
"""
Compare central kinematic maps across NIRSpec and MUSE products.

This script builds a 3x4 comparison figure for:
    - NIRSpec wavelength-dependent LSF fit
    - NIRSpec fixed R~2700 fit
    - Antoine's NIRSpec kinematic CSV
    - MUSE pPXF emiles product

Rows are:
    - VLOS
    - sigma
    - vrms

For comparison against MUSE, the NIRSpec-family datasets are rotated by -18 deg:

    x_rot = x cos(theta) - y sin(theta)
    y_rot = x sin(theta) + y cos(theta)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

C = 299792.458  # km/s

DEFAULT_WAVE_LSF_CSV = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/ppxf_nirspec/"
    "agn_substracted_david_wavelength_lsf/g235h_agn_sub_stellar_kinematics.csv"
)
DEFAULT_FIXED_LSF_CSV = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/ppxf_nirspec/"
    "agn_substracted_david/g235h_agn_sub_stellar_kinematics.csv"
)
DEFAULT_ANTOINE_CSV = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/antoine/M104_stellar_Kin.csv"
)
DEFAULT_MUSE_FITS = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Plots/ppxfppxf_c30_emiles_refactored_finer/"
    "c30_DATACUBE_normppxf_skycont_Part1_0000_ppxf_products_emiles.fits"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/ppxf_nirspec/agn_substracted_david_comparison"
)


@dataclass
class Dataset:
    name: str
    source_path: str
    x: np.ndarray
    y: np.ndarray
    vlos: np.ndarray
    sigma: np.ndarray
    vrms: np.ndarray
    systemic_for_vrms: float
    note: str
    marker_size: float


def rotate(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    angle_rad = np.radians(angle_deg)
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_delimited_rows(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, float]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if key is not None and value is not None})
    return list(rows[0].keys()) if rows else [], rows


def load_nirspec_csv(path: Path, name: str, rotate_deg: float) -> Dataset:
    _, rows = read_delimited_rows(path, delimiter=";")
    x = np.array([row["X"] for row in rows], dtype=float)
    y = np.array([row["Y"] for row in rows], dtype=float)
    x_rot, y_rot = rotate(x, y, rotate_deg)
    vlos = np.array([row["LOSV"] for row in rows], dtype=float)
    systemic = float(np.nanmedian(vlos))
    vlos = vlos - systemic
    sigma = np.array([row["sigma"] for row in rows], dtype=float)
    #vrms = np.array([row["Vrms"] for row in rows], dtype=float)
    # compute vrms as sqrt((vlos - systemic)**2 + sigma**2)
    
    vrms = np.sqrt((vlos)**2 + sigma**2)
    return Dataset(
        name=name,
        source_path=str(path),
        x=x_rot,
        y=y_rot,
        vlos=vlos,
        sigma=sigma,
        vrms=vrms,
        systemic_for_vrms=systemic,
        note=f"Rotated by {rotate_deg:.1f} deg; vrms from CSV",
        marker_size=28.0,
    )


def load_antoine_csv(path: Path, name: str, rotate_deg: float) -> Dataset:
    _, rows = read_delimited_rows(path, delimiter=";")
    x = np.array([row["X"] for row in rows], dtype=float)
    y = np.array([row["Y"] for row in rows], dtype=float)
    x_rot, y_rot = rotate(x, y, rotate_deg)
    vlos = np.array([row["LOSV"] for row in rows], dtype=float)
    vlos = vlos - np.nanmedian(vlos)
    sigma = np.array([row["sigma"] for row in rows], dtype=float)
    systemic = float(np.nanmedian(vlos))
    vrms = np.sqrt((vlos - systemic) ** 2 + sigma**2)
    return Dataset(
        name=name,
        source_path=str(path),
        x=x_rot,
        y=y_rot,
        vlos=vlos,
        sigma=sigma,
        vrms=vrms,
        systemic_for_vrms=systemic,
        note=f"Rotated by {rotate_deg:.1f} deg; vrms uses LOSV - median(LOSV)",
        marker_size=30.0,
    )


def load_muse_fits(path: Path, name: str) -> Dataset:
    with fits.open(path) as hdul:
        tab = hdul["SPAXELS"].data
        redshift = float(hdul[0].header.get("REDSHFT", 0.0))
        x = np.asarray(tab["X_ARCSEC"], dtype=float)
        y = np.asarray(tab["Y_ARCSEC"], dtype=float)
        vrel = np.asarray(tab["V_KMS"], dtype=float)
        sigma = np.asarray(tab["SIGMA_KMS"], dtype=float)
    vlos = vrel #C * ((1.0 + redshift) * np.exp(vrel / C) - 1.0)
    vrms = np.sqrt(vrel**2 + sigma**2)
    systemic = C * redshift
    return Dataset(
        name=name,
        source_path=str(path),
        x=x,
        y=y,
        vlos=vlos,
        sigma=sigma,
        vrms=vrms,
        systemic_for_vrms=systemic,
        note=f"MUSE V_KMS converted to absolute VLOS using z={redshift:.6f}; vrms uses V_KMS",
        marker_size=10.0,
    )


def crop_mask(dataset: Dataset, xlim: tuple[float, float], ylim: tuple[float, float]) -> np.ndarray:
    return (
        np.isfinite(dataset.x)
        & np.isfinite(dataset.y)
        & np.isfinite(dataset.vlos)
        & np.isfinite(dataset.sigma)
        & np.isfinite(dataset.vrms)
        & (dataset.x >= xlim[0])
        & (dataset.x <= xlim[1])
        & (dataset.y >= ylim[0])
        & (dataset.y <= ylim[1])
    )


def robust_minmax(values: np.ndarray, pct_lo: float = 2.0, pct_hi: float = 98.0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    return float(np.nanpercentile(arr, pct_lo)), float(np.nanpercentile(arr, pct_hi))


def write_summary(path: Path, datasets: list[Dataset], xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    summary = {
        "xlim": list(map(float, xlim)),
        "ylim": list(map(float, ylim)),
        "datasets": [
            {
                "name": ds.name,
                "source_path": ds.source_path,
                "n_points": int(ds.x.size),
                "systemic_for_vrms": float(ds.systemic_for_vrms),
                "note": ds.note,
                "median_vlos": float(np.nanmedian(ds.vlos)),
                "median_sigma": float(np.nanmedian(ds.sigma)),
                "median_vrms": float(np.nanmedian(ds.vrms)),
            }
            for ds in datasets
        ],
    }
    path.write_text(json.dumps(summary, indent=2))


def plot_comparison(datasets: list[Dataset], outpath: Path, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    cropped = {ds.name: crop_mask(ds, xlim, ylim) for ds in datasets}

    vlos_all = np.concatenate([ds.vlos[cropped[ds.name]] for ds in datasets])
    sigma_all = np.concatenate([ds.sigma[cropped[ds.name]] for ds in datasets])
    vrms_all = np.concatenate([ds.vrms[cropped[ds.name]] for ds in datasets])

    vlos_lo, vlos_hi = robust_minmax(vlos_all)
    sigma_lo, sigma_hi = robust_minmax(sigma_all)
    vrms_lo, vrms_hi = robust_minmax(vrms_all)
    vlos_center = float(np.nanmedian(vlos_all))

    norms = [
        colors.TwoSlopeNorm(vmin=vlos_lo, vcenter=vlos_center, vmax=vlos_hi),
        colors.Normalize(vmin=sigma_lo, vmax=sigma_hi),
        colors.Normalize(vmin=vrms_lo, vmax=vrms_hi),
    ]
    cmaps = ["RdBu_r", "inferno", "magma"]
    row_labels = ["VLOS [km/s]", "sigma [km/s]", "vrms [km/s]"]

    fig, axes = plt.subplots(3, len(datasets), figsize=(16, 11), constrained_layout=True)

    for col, ds in enumerate(datasets):
        mask = cropped[ds.name]
        x = ds.x[mask]
        y = ds.y[mask]
        vals = [ds.vlos[mask], ds.sigma[mask], ds.vrms[mask]]

        axes[0, col].set_title(ds.name)
        for row in range(3):
            ax = axes[row, col]
            sc = ax.scatter(
                x,
                y,
                c=vals[row],
                s=ds.marker_size,
                marker="s",
                cmap=cmaps[row],
                norm=norms[row],
                linewidths=0.0,
                rasterized=True,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\nY [arcsec]")
            else:
                ax.set_ylabel("Y [arcsec]")
            if row == 2:
                ax.set_xlabel("X [arcsec]")
            else:
                ax.set_xlabel("X [arcsec]")
            ax.text(
                0.02,
                0.03,
                f"N={mask.sum()}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
            )

    for row in range(3):
        sm = plt.cm.ScalarMappable(norm=norms[row], cmap=cmaps[row])
        sm.set_array([])
        fig.colorbar(sm, ax=axes[row, :], fraction=0.02, pad=0.01)

    fig.suptitle(
        "Sombrero Central Kinematic Comparison\n"
        "NIRSpec-family datasets rotated by -18 deg into the MUSE frame",
        fontsize=14,
    )
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_map(
    x,
    y,
    values,
    cmap="RdBu_r",
    cbar_label="Value",
    ax=None,
    show=False,
    vmin=None,
    vmax=None,
    cbar=True,
    norm=None,
    xlim=None,
    ylim=None,
    grid_n=150,
    method="linear",
    ):
    """
    Interpolate irregular x/y/value points onto a regular grid and plot with imshow.

    Parameters
    ----------
    method : {"linear", "nearest", "cubic"}
        Interpolation method passed to scipy.interpolate.griddata.
        For sparse or irregular IFU sampling, "linear" is often safer than "cubic".
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x = x[finite]
    y = y[finite]
    values = values[finite]

    if x.size < 3:
        raise ValueError("Need at least 3 finite points for 2D interpolation.")

    if xlim is None:
        xlim = (float(np.nanmin(x)), float(np.nanmax(x)))
    if ylim is None:
        ylim = (float(np.nanmin(y)), float(np.nanmax(y)))

    xi = np.linspace(xlim[0], xlim[1], grid_n)
    yi = np.linspace(ylim[0], ylim[1], grid_n)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), values, (xi, yi), method=method)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    imshow_kwargs = dict(
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )

    if norm is not None:
        imshow_kwargs["norm"] = norm
    else:
        imshow_kwargs["vmin"] = vmin
        imshow_kwargs["vmax"] = vmax

    im = ax.imshow(zi, **imshow_kwargs)

    if cbar:
        fig.colorbar(im, ax=ax, label=cbar_label, pad=0.01)

    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")

    if show:
        plt.show()

    return im

def plot_comparison_interpolated(
    datasets: list[Dataset],
    outpath: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    cropped = {ds.name: crop_mask(ds, xlim, ylim) for ds in datasets}

    vlos_all = np.concatenate([ds.vlos[cropped[ds.name]] for ds in datasets])
    sigma_all = np.concatenate([ds.sigma[cropped[ds.name]] for ds in datasets])
    vrms_all = np.concatenate([ds.vrms[cropped[ds.name]] for ds in datasets])

    vlos_lo, vlos_hi = robust_minmax(vlos_all)
    sigma_lo, sigma_hi = robust_minmax(sigma_all)
    vrms_lo, vrms_hi = robust_minmax(vrms_all)
    vlos_center = float(np.nanmedian(vlos_all))

    # Guard against TwoSlopeNorm errors if the robust limits are pathological.
    if not (vlos_lo < vlos_center < vlos_hi):
        vlos_lo = float(np.nanmin(vlos_all))
        vlos_hi = float(np.nanmax(vlos_all))
        vlos_center = 0.5 * (vlos_lo + vlos_hi)

    norms = [
        colors.TwoSlopeNorm(vmin=vlos_lo, vcenter=vlos_center, vmax=vlos_hi),
        colors.Normalize(vmin=sigma_lo, vmax=sigma_hi),
        colors.Normalize(vmin=vrms_lo, vmax=vrms_hi),
    ]

    cmaps = ["RdBu_r", "RdBu_r", "RdBu_r"]#["RdBu_r", "inferno", "magma"]
    row_labels = ["VLOS [km/s]", "sigma [km/s]", "vrms [km/s]"]

    fig, axes = plt.subplots(
        3,
        len(datasets),
        figsize=(16, 11),
        constrained_layout=True,
        squeeze=False,
    )

    for col, ds in enumerate(datasets):
        mask = cropped[ds.name]

        x = ds.x[mask]
        y = ds.y[mask]
        vals = [ds.vlos[mask], ds.sigma[mask], ds.vrms[mask]]

        axes[0, col].set_title(ds.name)

        for row in range(3):
            ax = axes[row, col]

            plot_map(
                x,
                y,
                vals[row],
                ax=ax,
                cmap=cmaps[row],
                norm=norms[row],
                cbar=False,
                cbar_label=row_labels[row],
                xlim=xlim,
                ylim=ylim,
                grid_n=200,
                method="linear",
            )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")

            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\nY [arcsec]")
            else:
                ax.set_ylabel("Y [arcsec]")

            ax.set_xlabel("X [arcsec]")

            ax.text(
                0.02,
                0.03,
                f"N={mask.sum()}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
            )

    for row in range(3):
        sm = plt.cm.ScalarMappable(norm=norms[row], cmap=cmaps[row])
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[row, :], fraction=0.02, pad=0.01)
        cbar.set_label(row_labels[row])

    fig.suptitle(
        "Sombrero Central Kinematic Comparison\n"
        "NIRSpec-family datasets rotated by -18 deg into the MUSE frame",
        fontsize=14,
    )

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NIRSpec and MUSE kinematic maps")
    parser.add_argument("--wave-lsf-csv", type=Path, default=DEFAULT_WAVE_LSF_CSV)
    parser.add_argument("--fixed-lsf-csv", type=Path, default=DEFAULT_FIXED_LSF_CSV)
    parser.add_argument("--antoine-csv", type=Path, default=DEFAULT_ANTOINE_CSV)
    parser.add_argument("--muse-fits", type=Path, default=DEFAULT_MUSE_FITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rotation-deg", type=float, default=-18.0)
    parser.add_argument("--padding-arcsec", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.output_dir.resolve())

    datasets = [
        load_nirspec_csv(args.wave_lsf_csv.resolve(), "NIRSpec LSF(lambda)", args.rotation_deg),
        load_nirspec_csv(args.fixed_lsf_csv.resolve(), "NIRSpec fixed R=2700", args.rotation_deg),
        load_antoine_csv(args.antoine_csv.resolve(), "Antoine CSV", args.rotation_deg),
        load_muse_fits(args.muse_fits.resolve(), "MUSE emiles"),
    ]

    x_all = np.concatenate([datasets[i].x for i in range(3)])
    y_all = np.concatenate([datasets[i].y for i in range(3)])
    xlim = (float(np.nanmin(x_all) - args.padding_arcsec), float(np.nanmax(x_all) + args.padding_arcsec))
    ylim = (float(np.nanmin(y_all) - args.padding_arcsec), float(np.nanmax(y_all) + args.padding_arcsec))

    plot_path = outdir / "nirspec_muse_kinematic_comparison_5.png"
    summary_path = outdir / "nirspec_muse_kinematic_comparison_summary.json"
    notes_path = outdir / "nirspec_muse_kinematic_comparison_notes.txt"

    plot_comparison_interpolated(datasets, plot_path, xlim, ylim)
    write_summary(summary_path, datasets, xlim, ylim)

    notes = [
        "Kinematic comparison notes",
        "",
        f"Rotation applied to NIRSpec-family datasets: {args.rotation_deg:.1f} deg",
        f"Common plotting limits: x={xlim[0]:.3f} to {xlim[1]:.3f} arcsec, y={ylim[0]:.3f} to {ylim[1]:.3f} arcsec",
        "",
        "Conventions:",
        "- VLOS for the NIRSpec CSV products uses the saved absolute LOSV column.",
        "- VLOS for the MUSE product is derived from the saved V_KMS values and the FITS header redshift.",
        "- vrms for the NIRSpec CSV products uses the saved Vrms column.",
        "- vrms for Antoine's CSV is computed as sqrt((LOSV - median(LOSV))^2 + sigma^2).",
        "- vrms for the MUSE product is computed as sqrt(V_KMS^2 + sigma^2).",
        "",
        "Input files:",
        *[f"- {ds.name}: {ds.source_path}" for ds in datasets],
    ]
    notes_path.write_text("\n".join(notes) + "\n")

    print(f"Saved plot   : {plot_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved notes  : {notes_path}")


if __name__ == "__main__":
    main()
