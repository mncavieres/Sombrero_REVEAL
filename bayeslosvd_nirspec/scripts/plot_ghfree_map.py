#!/usr/bin/env python3
"""Plot GHfree MAP kinematics and top-dispersion LOSVDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from reproducibility import write_reproduction_files


LOSVD_XLIM = 1200.0
MAP_ROTATION_DEG = -18.0


def _rotate_coordinates(x, y, angle_deg=MAP_ROTATION_DEG):
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return x * c - y * s, x * s + y * c


def _normalized_losvd(result_file, bin_id, order, xplot):
    losvd = np.asarray(result_file[f"out/{bin_id}/losvd"])[2][order]
    norm = np.trapz(losvd, xplot)
    if np.isfinite(norm) and norm != 0:
        losvd = losvd / norm
    return losvd


def _losvd_moments(xplot, losvd):
    good = np.isfinite(xplot) & np.isfinite(losvd)
    if np.count_nonzero(good) < 3:
        return np.nan, np.nan, np.nan, np.nan
    x = xplot[good]
    y = losvd[good]
    area = np.sum(y)
    if not np.isfinite(area) or area <= 0:
        return np.nan, np.nan, np.nan, np.nan
    vel = np.sum(x * y) / area
    var = np.sum((x - vel) ** 2 * y) / area
    if not np.isfinite(var) or var <= 0:
        return vel, np.nan, np.nan, np.nan
    sigma = np.sqrt(var)
    w = (x - vel) / sigma
    h3 = np.sum((w**3) * y) / area
    h4 = np.sum((w**4) * y) / area - 3.0
    return vel, sigma, h3, h4


def _losvd_moment_values(result_file):
    bins = sorted((int(k) for k in result_file["out"].keys()))
    xvel = np.asarray(result_file["in/xvel"])
    order = np.argsort(xvel)
    xplot = xvel[order]
    values = {key: np.full(len(bins), np.nan) for key in ["vel", "sigma", "h3", "h4"]}
    for i, bin_id in enumerate(bins):
        losvd = _normalized_losvd(result_file, bin_id, order, xplot)
        vel, sigma, h3, h4 = _losvd_moments(xplot, losvd)
        values["vel"][i] = vel
        values["sigma"][i] = sigma
        values["h3"][i] = h3
        values["h4"][i] = h4
    return bins, values


def _top_bins(result_file, n=10, allowed_bins=None):
    bins, moments = _losvd_moment_values(result_file)
    sigma = moments["sigma"]
    allowed = set(allowed_bins) if allowed_bins is not None else None
    keep = np.asarray([allowed is None or bin_id in allowed for bin_id in bins], dtype=bool)
    order = np.argsort(np.where(keep, np.nan_to_num(sigma, nan=-np.inf), -np.inf))[::-1]
    order = order[np.isfinite(sigma[order]) & keep[order]]
    return [bins[i] for i in order[:n]], sigma[order[:n]]


def _scatter_map(ax, x, y, values, title, cmap, label, highlight=None):
    sc = ax.scatter(x, y, c=values, s=42, cmap=cmap, edgecolor="none")
    if highlight is not None:
        ax.scatter(x[highlight], y[highlight], s=110, facecolors="none", edgecolors="black", linewidths=1.2)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x_rot (arcsec)")
    ax.set_ylabel("y_rot (arcsec)")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label)


def make_kinematics_plot(result_file, output):
    x_raw = np.asarray(result_file["in/xbin"])
    y_raw = np.asarray(result_file["in/ybin"])
    x, y = _rotate_coordinates(x_raw, y_raw)
    _, moments = _losvd_moment_values(result_file)
    vel = moments["vel"]
    sigma = moments["sigma"]
    h3 = moments["h3"]
    h4 = moments["h4"]
    top, _ = _top_bins(result_file)
    highlight = np.asarray(top, dtype=int)
    fit_type = result_file.attrs.get("fit_type", "BAYES-LOSVD")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    _scatter_map(axes[0, 0], x, y, vel, "LOSVD mean velocity", "RdBu_r", "km/s")
    _scatter_map(axes[0, 1], x, y, sigma, "LOSVD dispersion", "magma", "km/s", highlight=highlight)
    _scatter_map(axes[1, 0], x, y, h3, "LOSVD skewness", "coolwarm", "")
    _scatter_map(axes[1, 1], x, y, h4, "LOSVD excess kurtosis", "coolwarm", "")
    fig.suptitle(f"BAYES-LOSVD {fit_type} MAP kinematics: NIRSpec AGN-subtracted cube, rotated -18 deg", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_losvd_plot(result_file, output, allowed_bins=None, title_suffix=""):
    xvel = np.asarray(result_file["in/xvel"])
    order = np.argsort(xvel)
    xplot = xvel[order]
    top, sigmas = _top_bins(result_file, allowed_bins=allowed_bins)
    fit_type = result_file.attrs.get("fit_type", "BAYES-LOSVD")
    if len(top) == 0:
        raise ValueError("No finite-dispersion bins found for the requested LOSVD selection.")

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, constrained_layout=True)
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    for ax, bin_id, sigma in zip(axes.ravel(), top, sigmas):
        losvd = _normalized_losvd(result_file, bin_id, order, xplot)
        ax.plot(xplot, losvd, color="black", linewidth=1.8, drawstyle="steps-mid")
        ax.axvline(0, color="0.5", linestyle=":", linewidth=1)
        ax.set_xlim(-LOSVD_XLIM, LOSVD_XLIM)
        ax.set_title(f"bin {bin_id}  sigma={sigma:.1f}")
        ax.set_yticks([])
        ax.set_xlabel("km/s")
    axes[0, 0].set_ylabel("LOSVD")
    axes[1, 0].set_ylabel("LOSVD")
    fig.suptitle(f"Top {len(top)} highest-dispersion BAYES-LOSVD {fit_type} bins{title_suffix}", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    return top, sigmas


def _bins_inside_radius(result_file, radius):
    x = np.asarray(result_file["in/xbin"])
    y = np.asarray(result_file["in/ybin"])
    bins = sorted((int(k) for k in result_file["out"].keys()))
    return [bin_id for bin_id in bins if bin_id < len(x) and np.hypot(x[bin_id], y[bin_id]) <= radius]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--central-radius", type=float, default=None, help="Also plot top-dispersion LOSVDs inside this radius in arcsec.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.results, "r") as result_file:
        fit_type = str(result_file.attrs.get("fit_type", "fit")).lower()
        kin = args.output_dir / f"bayeslosvd_{fit_type}_map_checkplots.png"
        losvd = args.output_dir / f"bayeslosvd_{fit_type}_top10_dispersion_losvds.png"
        make_kinematics_plot(result_file, kin)
        top, sigmas = make_losvd_plot(result_file, losvd)
        central = None
        central_top = None
        central_sigmas = None
        if args.central_radius is not None:
            central = _bins_inside_radius(result_file, args.central_radius)
            radius_label = f"r{args.central_radius:.2f}".replace(".", "p")
            central_losvd = args.output_dir / f"bayeslosvd_{fit_type}_top10_dispersion_losvds_{radius_label}arcsec.png"
            central_top, central_sigmas = make_losvd_plot(
                result_file,
                central_losvd,
                allowed_bins=central,
                title_suffix=f" within r <= {args.central_radius:.2f} arcsec",
            )

    print(f"kinematics_plot={kin}")
    print(f"losvd_plot={losvd}")
    print("top_bins=" + ",".join(str(i) for i in top))
    print("top_sigmas=" + ",".join(f"{s:.6g}" for s in sigmas))
    output_paths = [kin, losvd]
    if args.central_radius is not None:
        output_paths.append(central_losvd)
        print(f"central_radius_arcsec={args.central_radius:.6g}")
        print(f"central_bin_count={len(central)}")
        print(f"central_losvd_plot={central_losvd}")
        print("central_top_bins=" + ",".join(str(i) for i in central_top))
        print("central_top_sigmas=" + ",".join(f"{s:.6g}" for s in central_sigmas))
    write_reproduction_files(
        args.output_dir,
        run_name=f"{args.results.stem}_checkplots",
        input_paths=[args.results],
        output_paths=output_paths,
        extra={
            "runner": "plot_ghfree_map.py",
            "central_radius": args.central_radius,
            "top_bins": [int(i) for i in top],
            "top_sigmas": [float(s) for s in sigmas],
        },
        run_file_name="reproduce_checkplots.sh",
        manifest_name="checkplots_run_manifest.json",
    )


if __name__ == "__main__":
    main()
