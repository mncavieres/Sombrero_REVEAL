#!/usr/bin/env python3
"""Plot BAYES-LOSVD spectral-fit checkplots along the IFU major/minor axes."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from astropy.io import fits

from reproducibility import write_reproduction_files


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MAP_ROTATION_DEG = -18.0
CUBE_FIT_RANGE_UM = (2.10, 2.398)


@dataclass(frozen=True)
class Selection:
    label: str
    bin_id: int
    target_xrot: float
    target_yrot: float
    x: float
    y: float
    xrot: float
    yrot: float
    distance: float
    bin_flux: float
    bin_snr: float
    rms: float


def rotate_coordinates(x, y, angle_deg=MAP_ROTATION_DEG):
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return x * c - y * s, x * s + y * c


def science_cube_hdu(hdul):
    for hdu in hdul:
        if hdu.data is not None and np.asarray(hdu.data).ndim == 3:
            return hdu
    raise ValueError("No 3D science cube HDU found.")


def orient_cube_nlambda_first(data, header):
    cube = np.asarray(data, dtype=float)
    naxis3 = int(header.get("NAXIS3", cube.shape[0]))
    if cube.shape[0] == naxis3:
        return cube
    if cube.shape[-1] == naxis3:
        return np.moveaxis(cube, -1, 0)
    spectral_axis = int(np.argmax(cube.shape))
    return np.moveaxis(cube, spectral_axis, 0) if spectral_axis != 0 else cube


def wavelength_axis_um(header, nlambda):
    crval = float(header.get("CRVAL3", 0.0))
    crpix = float(header.get("CRPIX3", 1.0))
    cdelt = float(header.get("CDELT3", header.get("CD3_3", 1.0)))
    wave = crval + (np.arange(nlambda, dtype=float) + 1.0 - crpix) * cdelt
    unit = str(header.get("CUNIT3", "")).strip().lower()
    if "ang" in unit:
        wave = wave / 1.0e4
    elif unit == "nm":
        wave = wave / 1.0e3
    elif unit in {"m", "meter", "metre"}:
        wave = wave * 1.0e6
    elif not unit and np.nanmedian(wave) > 100.0:
        wave = wave / 1.0e4
    return wave


def pixel_scale_arcsec(header):
    scale = header.get("CDELT1", header.get("CD1_1", None))
    if scale is None:
        scale = header.get("CDELT2", header.get("CD2_2", None))
    if scale is None:
        return 0.1
    scale = abs(float(scale))
    return scale * 3600.0 if scale < 0.01 else scale


def cube_peak_and_pixscale(cube_path: Path):
    with fits.open(cube_path, memmap=False) as hdul:
        hdu = science_cube_hdu(hdul)
        cube = orient_cube_nlambda_first(hdu.data, hdu.header)
        wave_um = wavelength_axis_um(hdu.header, cube.shape[0])
        pixscale = pixel_scale_arcsec(hdu.header)

    fit = (wave_um >= CUBE_FIT_RANGE_UM[0]) & (wave_um <= CUBE_FIT_RANGE_UM[1])
    if np.count_nonzero(fit) == 0:
        fit = np.isfinite(wave_um)
    image = np.nanmedian(cube[fit], axis=0)
    signal = np.where(np.isfinite(image), image, -np.inf)
    peak_row, peak_col = np.unravel_index(int(np.nanargmax(signal)), image.shape)
    return int(peak_row), int(peak_col), float(pixscale)


def parse_pixels(text: str):
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("At least one pixel offset is required.")
    return out


def safe_tag(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "fit"


def result_fit_tag(handle):
    return safe_tag(handle.attrs.get("fit_type", "fit"))


def nearest_unused_bin(xrot, yrot, tx, ty, used):
    distance = np.hypot(xrot - tx, yrot - ty)
    order = np.argsort(distance)
    for idx in order:
        idx = int(idx)
        if idx not in used:
            return idx, float(distance[idx])
    raise ValueError("No unused bin remains for axis selection.")


def residual_rms(handle, bin_id):
    mask = np.asarray(handle["in/mask"], dtype=int)
    obs = np.asarray(handle["in/spec_obs"][:, bin_id], dtype=float)
    best = np.asarray(handle[f"out/{bin_id}/bestfit"], dtype=float)
    best_med = best[2] if best.ndim == 2 and best.shape[0] > 2 else np.ravel(best)
    res = obs - best_med
    good = mask[np.isfinite(res[mask])]
    if good.size == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(res[good] ** 2)))


def build_selections(handle, cube_path: Path, major_pixels, minor_pixels):
    peak_row, peak_col, pixscale = cube_peak_and_pixscale(cube_path)
    x = np.asarray(handle["in/xbin"], dtype=float)
    y = np.asarray(handle["in/ybin"], dtype=float)
    xrot, yrot = rotate_coordinates(x, y)
    bin_flux = np.asarray(handle["in/bin_flux"], dtype=float)
    bin_snr = np.asarray(handle["in/bin_snr"], dtype=float)

    used = set()
    rows = []
    center_bin, center_distance = nearest_unused_bin(xrot, yrot, 0.0, 0.0, used)
    used.add(center_bin)
    rows.append(
        Selection(
            label=f"brightest_peak_r{peak_row}_c{peak_col}",
            bin_id=center_bin,
            target_xrot=0.0,
            target_yrot=0.0,
            x=float(x[center_bin]),
            y=float(y[center_bin]),
            xrot=float(xrot[center_bin]),
            yrot=float(yrot[center_bin]),
            distance=center_distance,
            bin_flux=float(bin_flux[center_bin]),
            bin_snr=float(bin_snr[center_bin]),
            rms=residual_rms(handle, center_bin),
        )
    )

    for pixel in major_pixels:
        tx = pixel * pixscale
        ty = 0.0
        bin_id, distance = nearest_unused_bin(xrot, yrot, tx, ty, used)
        used.add(bin_id)
        rows.append(
            Selection(
                label=f"major_plus_{pixel:g}pix",
                bin_id=bin_id,
                target_xrot=tx,
                target_yrot=ty,
                x=float(x[bin_id]),
                y=float(y[bin_id]),
                xrot=float(xrot[bin_id]),
                yrot=float(yrot[bin_id]),
                distance=distance,
                bin_flux=float(bin_flux[bin_id]),
                bin_snr=float(bin_snr[bin_id]),
                rms=residual_rms(handle, bin_id),
            )
        )

    for pixel in minor_pixels:
        tx = 0.0
        ty = pixel * pixscale
        bin_id, distance = nearest_unused_bin(xrot, yrot, tx, ty, used)
        used.add(bin_id)
        rows.append(
            Selection(
                label=f"minor_plus_{pixel:g}pix",
                bin_id=bin_id,
                target_xrot=tx,
                target_yrot=ty,
                x=float(x[bin_id]),
                y=float(y[bin_id]),
                xrot=float(xrot[bin_id]),
                yrot=float(yrot[bin_id]),
                distance=distance,
                bin_flux=float(bin_flux[bin_id]),
                bin_snr=float(bin_snr[bin_id]),
                rms=residual_rms(handle, bin_id),
            )
        )

    return rows, pixscale


def write_selection_csv(path: Path, selections):
    fields = [
        "label",
        "bin_id",
        "target_xrot_arcsec",
        "target_yrot_arcsec",
        "x_arcsec",
        "y_arcsec",
        "xrot_arcsec",
        "yrot_arcsec",
        "selection_distance_arcsec",
        "bin_flux",
        "bin_snr",
        "fit_rms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selections:
            writer.writerow(
                {
                    "label": row.label,
                    "bin_id": row.bin_id,
                    "target_xrot_arcsec": row.target_xrot,
                    "target_yrot_arcsec": row.target_yrot,
                    "x_arcsec": row.x,
                    "y_arcsec": row.y,
                    "xrot_arcsec": row.xrot,
                    "yrot_arcsec": row.yrot,
                    "selection_distance_arcsec": row.distance,
                    "bin_flux": row.bin_flux,
                    "bin_snr": row.bin_snr,
                    "fit_rms": row.rms,
                }
            )


def wavelength_um(handle):
    wave = np.exp(np.asarray(handle["in/wave_obs"], dtype=float))
    return wave / 1.0e4 if np.nanmedian(wave) > 100.0 else wave


def y_limits_for_spectrum(obs, best, residual):
    finite = np.r_[obs[np.isfinite(obs)], best[np.isfinite(best)]]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(finite, [1, 99])
    span = hi - lo if hi > lo else 1.0
    res_base = lo - 0.28 * span
    return res_base - 0.18 * span, hi + 0.12 * span


def plot_single_spectrum(ax, handle, selection: Selection):
    wave = wavelength_um(handle)
    mask = np.asarray(handle["in/mask"], dtype=int)
    obs = np.asarray(handle["in/spec_obs"][:, selection.bin_id], dtype=float)
    sigma = np.asarray(handle["in/sigma_obs"][:, selection.bin_id], dtype=float)
    bestfit = np.asarray(handle[f"out/{selection.bin_id}/bestfit"], dtype=float)
    poly = np.asarray(handle[f"out/{selection.bin_id}/poly"], dtype=float)
    best = bestfit[2] if bestfit.ndim == 2 and bestfit.shape[0] > 2 else np.ravel(bestfit)
    poly_med = poly[2] if poly.ndim == 2 and poly.shape[0] > 2 else np.ravel(poly)
    residual = obs - best
    lo, hi = y_limits_for_spectrum(obs, best, residual)
    span = hi - lo
    res_base = lo + 0.14 * span
    res = residual + res_base

    good_sigma = np.isfinite(sigma) & (sigma > 0)
    if np.any(good_sigma):
        ax.fill_between(wave, obs - sigma, obs + sigma, color="0.82", lw=0, alpha=0.55)
    if bestfit.ndim == 2 and bestfit.shape[0] > 3 and not np.allclose(bestfit[1], bestfit[3], equal_nan=True):
        ax.fill_between(wave, bestfit[1], bestfit[3], color="#f59e0b", lw=0, alpha=0.25)

    ax.plot(wave, obs, color="black", lw=0.9, label="Observed")
    ax.plot(wave, best, color="#d62728", lw=1.1, label="BAYES fit")
    ax.plot(wave, res, color="#2ca02c", lw=0.8, label="Residual")
    if poly_med.shape == obs.shape:
        ax.plot(wave, poly_med + 1.0, color="0.45", lw=0.8, ls="--", alpha=0.65, label="Polynomial")

    if mask.size:
        ax.axvline(wave[mask[0]], color="0.25", ls=":", lw=0.8)
        ax.axvline(wave[mask[-1]], color="0.25", ls=":", lw=0.8)
        gaps = np.flatnonzero(np.diff(mask) > 1)
        for gap in gaps:
            ax.axvspan(wave[mask[gap]], wave[mask[gap + 1]], color="0.5", alpha=0.12)
    ax.axhline(res_base, color="0.25", lw=0.7, ls=":")
    ax.set_ylim(lo, hi)
    ax.set_xlim(float(np.nanmin(wave)), float(np.nanmax(wave)))
    ax.set_title(
        f"{selection.label.replace('_', ' ')} | bin {selection.bin_id}\n"
        f"xrot={selection.xrot:+.2f}, yrot={selection.yrot:+.2f} arcsec; RMS={selection.rms:.3g}",
        fontsize=9,
    )
    ax.tick_params(labelsize=8)


def plot_locator(ax, handle, selections):
    x = np.asarray(handle["in/xbin"], dtype=float)
    y = np.asarray(handle["in/ybin"], dtype=float)
    xrot, yrot = rotate_coordinates(x, y)
    ax.scatter(xrot, yrot, s=10, color="0.82", edgecolors="none")
    colors = plt.cm.tab10(np.linspace(0, 1, len(selections)))
    for color, selection in zip(colors, selections):
        ax.scatter(selection.xrot, selection.yrot, s=70, color=color, edgecolor="black", zorder=3)
        ax.text(selection.xrot, selection.yrot, str(selection.bin_id), fontsize=7, ha="left", va="bottom")
    ax.axhline(0.0, color="0.35", lw=0.8, ls=":")
    ax.axvline(0.0, color="0.35", lw=0.8, ls=":")
    ax.set_aspect("equal")
    ax.set_xlabel("x_rot major axis (arcsec)")
    ax.set_ylabel("y_rot minor axis (arcsec)")
    ax.set_title("Selected fitted bins", fontsize=10)


def make_plot(handle, selections, output: Path):
    nrows = 4
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 13), constrained_layout=True)
    axes = axes.ravel()
    plot_locator(axes[0], handle, selections)
    for ax, selection in zip(axes[1:], selections):
        plot_single_spectrum(ax, handle, selection)
    for ax in axes[1 + len(selections):]:
        ax.axis("off")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fit_type = handle.attrs.get("fit_type", "BAYES-LOSVD")
    fig.suptitle(
        f"{fit_type} spectral-fit checks: brightest IFU peak and +1/+2/+3 pixel axis samples",
        fontsize=15,
    )
    fig.supxlabel("Rest wavelength (micron)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cube", type=Path, default=Path("Data/IFU/david_subs/g235h_agn_sub.fits"))
    parser.add_argument("--major-pixels", default="1,2,3")
    parser.add_argument("--minor-pixels", default="1,2,3")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    major_pixels = parse_pixels(args.major_pixels)
    minor_pixels = parse_pixels(args.minor_pixels)

    with h5py.File(args.results, "r") as handle:
        tag = result_fit_tag(handle)
        selections, pixscale = build_selections(handle, args.cube, major_pixels, minor_pixels)
        output = args.output_dir / f"bayeslosvd_{tag}_spectral_fit_axis_checkplots.png"
        csv_output = args.output_dir / f"bayeslosvd_{tag}_spectral_fit_axis_bins.csv"
        make_plot(handle, selections, output)
        write_selection_csv(csv_output, selections)

    run_file, manifest_file = write_reproduction_files(
        args.output_dir,
        run_name=f"{args.results.stem}_spectral_fit_axis_checkplots",
        input_paths=[args.results, args.cube],
        output_paths=[output, csv_output],
        extra={
            "runner": "plot_spectral_fit_axis_checkplots.py",
            "rotation_deg": MAP_ROTATION_DEG,
            "cube_fit_range_um": list(CUBE_FIT_RANGE_UM),
            "major_pixels": major_pixels,
            "minor_pixels": minor_pixels,
            "pixscale_arcsec": pixscale,
            "selected_bins": [row.bin_id for row in selections],
            "selected_labels": [row.label for row in selections],
        },
        run_file_name="reproduce_spectral_fit_axis_checkplots.sh",
        manifest_name="spectral_fit_axis_checkplots_run_manifest.json",
    )

    print(f"plot={output}")
    print(f"selection_csv={csv_output}")
    print(f"reproduce_script={run_file}")
    print(f"run_manifest={manifest_file}")
    print("selected_bins=" + ",".join(str(row.bin_id) for row in selections))


if __name__ == "__main__":
    main()
