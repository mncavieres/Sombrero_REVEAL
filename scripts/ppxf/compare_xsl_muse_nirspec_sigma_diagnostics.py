#!/usr/bin/env python3
"""
Compare MUSE and NIRSpec XSL pPXF velocity-dispersion products.

This is a lightweight diagnostic layer for the Sombrero REVEAL pPXF runs. It
does not refit the cubes. It reads the saved XSL products, writes central sigma
summary tables, makes side-by-side finding maps, and regenerates MUSE fit
checkplots for the brightest spaxel and central peak-sigma spaxel from the
stored binned spectra and best fits.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from ppxf.ppxf import robust_sigma


ROOT = Path("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL")
DEFAULT_MUSE_FITS = (
    ROOT
    / "Plots/ppxfppxf_c30_xsl/c30_DATACUBE_normppxf_skycont_Part1_0000_ppxf_products_xsl.fits"
)
DEFAULT_NIRSPEC_FITS = (
    ROOT
    / "Data/ppxf_nirspec/agn_sub_xsl_consistent_deg8_mom4/g235h_agn_sub_stellar_kinematics.fits"
)
DEFAULT_NIRSPEC_RUN_CONFIG = (
    ROOT / "Data/ppxf_nirspec/agn_sub_xsl_consistent_deg8_mom4/run_config.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "Data/ppxf_xsl_consistency_diagnostics"


@dataclass
class KinematicProduct:
    name: str
    source_path: Path
    x: np.ndarray
    y: np.ndarray
    signal: np.ndarray
    sigma: np.ndarray
    h3: np.ndarray
    h4: np.ndarray
    row: np.ndarray
    col: np.ndarray
    bin_id: np.ndarray | None
    signal_map: np.ndarray
    sigma_map: np.ndarray
    pixsize: float
    header: fits.Header


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rotate(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.radians(angle_deg)
    return (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta),
    )


def robust_minmax(*arrays: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    vals = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.nanpercentile(vals, lo)), float(np.nanpercentile(vals, hi))


def load_muse_product(path: Path) -> KinematicProduct:
    with fits.open(path, memmap=False) as hdul:
        spax = hdul["SPAXELS"].data
        header = hdul[0].header.copy()
        h3 = np.asarray(spax["H3"], dtype=float) if "H3" in spax.names else np.full(spax.size, np.nan)
        h4 = np.asarray(spax["H4"], dtype=float) if "H4" in spax.names else np.full(spax.size, np.nan)
        return KinematicProduct(
            name="MUSE XSL",
            source_path=path,
            x=np.asarray(spax["X_ARCSEC"], dtype=float),
            y=np.asarray(spax["Y_ARCSEC"], dtype=float),
            signal=np.asarray(spax["SIGNAL"], dtype=float),
            sigma=np.asarray(spax["SIGMA_KMS"], dtype=float),
            h3=h3,
            h4=h4,
            row=np.asarray(spax["ROW"], dtype=int),
            col=np.asarray(spax["COL"], dtype=int),
            bin_id=np.asarray(spax["BIN_ID"], dtype=int),
            signal_map=np.asarray(hdul["SIGNAL_MAP"].data, dtype=float),
            sigma_map=np.asarray(hdul["SIGMA_MAP"].data, dtype=float),
            pixsize=float(header.get("PIXSIZE", 0.2)),
            header=header,
        )


def load_nirspec_product(path: Path, rotation_deg: float) -> KinematicProduct:
    with fits.open(path, memmap=False) as hdul:
        tab = hdul["KIN_RESULTS"].data
        header = hdul[0].header.copy()
        x = np.asarray(tab["X"], dtype=float)
        y = np.asarray(tab["Y"], dtype=float)
        x_rot, y_rot = rotate(x, y, rotation_deg)
        h3 = np.asarray(tab["H3"], dtype=float) if "H3" in tab.names else np.full(tab.size, np.nan)
        h4 = np.asarray(tab["H4"], dtype=float) if "H4" in tab.names else np.full(tab.size, np.nan)
        return KinematicProduct(
            name=f"NIRSpec XSL rotated {rotation_deg:.1f} deg",
            source_path=path,
            x=x_rot,
            y=y_rot,
            signal=np.asarray(tab["SIGNAL"], dtype=float),
            sigma=np.asarray(tab["SIGMA"], dtype=float),
            h3=h3,
            h4=h4,
            row=np.asarray(tab["ROW"], dtype=int),
            col=np.asarray(tab["COL"], dtype=int),
            bin_id=None,
            signal_map=np.asarray(hdul["SIGNAL_MAP"].data, dtype=float),
            sigma_map=np.asarray(hdul["SIGMA_MAP"].data, dtype=float),
            pixsize=float(header.get("PIXSIZE", 0.1)),
            header=header,
        )


def central_peak(product: KinematicProduct, radius_arcsec: float) -> int | None:
    radius = np.hypot(product.x, product.y)
    good = np.isfinite(radius) & (radius <= radius_arcsec) & np.isfinite(product.sigma)
    if not np.any(good):
        return None
    idx = np.flatnonzero(good)
    return int(idx[np.nanargmax(product.sigma[idx])])


def brightest(product: KinematicProduct) -> int | None:
    good = np.isfinite(product.signal)
    if not np.any(good):
        return None
    idx = np.flatnonzero(good)
    return int(idx[np.nanargmax(product.signal[idx])])


def product_summary(product: KinematicProduct, radius_arcsec: float) -> dict[str, object]:
    good = np.isfinite(product.sigma)
    radius = np.hypot(product.x, product.y)
    central = good & np.isfinite(radius) & (radius <= radius_arcsec)
    bright_j = brightest(product)
    peak_j = central_peak(product, radius_arcsec)

    def target_payload(j: int | None) -> dict[str, float | int | None]:
        if j is None:
            return {"index": None}
        payload: dict[str, float | int | None] = {
            "index": int(j),
            "row": int(product.row[j]),
            "col": int(product.col[j]),
            "x_arcsec": float(product.x[j]),
            "y_arcsec": float(product.y[j]),
            "radius_arcsec": float(np.hypot(product.x[j], product.y[j])),
            "sigma_kms": float(product.sigma[j]),
            "h3": float(product.h3[j]),
            "h4": float(product.h4[j]),
            "signal": float(product.signal[j]),
        }
        if product.bin_id is not None:
            payload["bin_id"] = int(product.bin_id[j])
        return payload

    return {
        "name": product.name,
        "source_path": str(product.source_path),
        "n_finite_sigma": int(np.count_nonzero(good)),
        "median_sigma_kms": float(np.nanmedian(product.sigma[good])) if np.any(good) else np.nan,
        "p90_sigma_kms": float(np.nanpercentile(product.sigma[good], 90)) if np.any(good) else np.nan,
        "max_sigma_kms": float(np.nanmax(product.sigma[good])) if np.any(good) else np.nan,
        "central_radius_arcsec": float(radius_arcsec),
        "central_n_finite_sigma": int(np.count_nonzero(central)),
        "central_median_sigma_kms": float(np.nanmedian(product.sigma[central])) if np.any(central) else np.nan,
        "central_peak": target_payload(peak_j),
        "brightest": target_payload(bright_j),
        "header": {
            "sps_model": str(product.header.get("SPSMOD", "")),
            "velscale": float(product.header.get("VELSCAL", np.nan)),
            "moments": float(product.header.get("MOMENTS", np.nan)),
            "degree": float(product.header.get("DEGREE", product.header.get("KINDEG", np.nan))),
            "mdegree": float(product.header.get("MDEGREE", product.header.get("KINMDEG", np.nan))),
            "sigma_inst_kms": float(product.header.get("SIGINST", np.nan)),
            "sigma_template_eff_kms": float(product.header.get("SIGTEFF", np.nan)),
        },
    }


def image_extent(product: KinematicProduct) -> tuple[float, float, float, float]:
    return (
        float(np.nanmin(product.x) - 0.5 * product.pixsize),
        float(np.nanmax(product.x) + 0.5 * product.pixsize),
        float(np.nanmin(product.y) - 0.5 * product.pixsize),
        float(np.nanmax(product.y) + 0.5 * product.pixsize),
    )


def plot_sigma_maps(
    outpath: Path,
    muse: KinematicProduct,
    nirspec: KinematicProduct,
    radius_arcsec: float,
) -> None:
    vmin, vmax = robust_minmax(muse.sigma, nirspec.sigma)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    products = [muse, nirspec]
    for row, product in enumerate(products):
        bright_j = brightest(product)
        peak_j = central_peak(product, radius_arcsec)

        ax = axes[row, 0]
        if product.name.startswith("MUSE"):
            im = ax.imshow(product.signal_map, origin="lower", extent=image_extent(product), cmap="gray_r", aspect="equal")
        else:
            im = ax.scatter(product.x, product.y, c=product.signal, s=42, marker="s", cmap="gray_r", linewidths=0)
        if bright_j is not None:
            ax.scatter(product.x[bright_j], product.y[bright_j], marker="*", s=180, facecolor="none", edgecolor="tab:red", linewidth=1.5)
        if peak_j is not None:
            ax.scatter(product.x[peak_j], product.y[peak_j], marker="o", s=130, facecolor="none", edgecolor="cyan", linewidth=1.5)
        ax.set_title(f"{product.name}: signal")
        ax.set_xlabel("X [arcsec]")
        ax.set_ylabel("Y [arcsec]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[row, 1]
        if product.name.startswith("MUSE"):
            im = ax.imshow(product.sigma_map, origin="lower", extent=image_extent(product), cmap="inferno", norm=norm, aspect="equal")
        else:
            im = ax.scatter(product.x, product.y, c=product.sigma, s=42, marker="s", cmap="inferno", norm=norm, linewidths=0)
        if bright_j is not None:
            ax.scatter(product.x[bright_j], product.y[bright_j], marker="*", s=180, facecolor="none", edgecolor="tab:red", linewidth=1.5, label="brightest")
        if peak_j is not None:
            ax.scatter(product.x[peak_j], product.y[peak_j], marker="o", s=130, facecolor="none", edgecolor="cyan", linewidth=1.5, label="central peak sigma")
        ax.set_title(f"{product.name}: sigma")
        ax.set_xlabel("X [arcsec]")
        ax.set_ylabel("Y [arcsec]")
        ax.set_aspect("equal")
        ax.legend(loc="best", frameon=True, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="km/s")

    fig.suptitle(f"XSL pPXF kinematic diagnostics; central peak radius = {radius_arcsec:.2f} arcsec")
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_muse_checkplot_from_product(
    outpath: Path,
    muse_fits: Path,
    target_j: int,
    label: str,
) -> None:
    with fits.open(muse_fits, memmap=False) as hdul:
        spax = hdul["SPAXELS"].data
        lam = np.asarray(hdul["LAMBDA_GAL"].data, dtype=float)
        bin_spec = np.asarray(hdul["BIN_SPEC"].data, dtype=float)
        bin_bestfit = np.asarray(hdul["BIN_BESTFIT"].data, dtype=float)
        signal_map = np.asarray(hdul["SIGNAL_MAP"].data, dtype=float)
        sigma_map = np.asarray(hdul["SIGMA_MAP"].data, dtype=float)
        pixsize = float(hdul[0].header.get("PIXSIZE", 0.2))

        x = np.asarray(spax["X_ARCSEC"], dtype=float)
        y = np.asarray(spax["Y_ARCSEC"], dtype=float)
        row = np.asarray(spax["ROW"], dtype=int)
        col = np.asarray(spax["COL"], dtype=int)
        bin_id = np.asarray(spax["BIN_ID"], dtype=int)
        sigma = np.asarray(spax["SIGMA_KMS"], dtype=float)
        h3 = np.asarray(spax["H3"], dtype=float) if "H3" in spax.names else np.full(spax.size, np.nan)
        h4 = np.asarray(spax["H4"], dtype=float) if "H4" in spax.names else np.full(spax.size, np.nan)
        signal = np.asarray(spax["SIGNAL"], dtype=float)

    k = int(bin_id[target_j])
    galaxy = bin_spec[:, k]
    bestfit = bin_bestfit[:, k]
    resid = galaxy - bestfit
    scale = robust_sigma(resid[np.isfinite(resid)], zero=1)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    extent = (
        float(np.nanmin(x) - 0.5 * pixsize),
        float(np.nanmax(x) + 0.5 * pixsize),
        float(np.nanmin(y) - 0.5 * pixsize),
        float(np.nanmax(y) + 0.5 * pixsize),
    )

    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.0], width_ratios=[1.0, 1.0, 1.45])

    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(signal_map, origin="lower", extent=extent, cmap="gray_r", aspect="equal")
    ax.scatter(x[target_j], y[target_j], marker="*", s=220, facecolor="none", edgecolor="tab:red", linewidth=1.6)
    ax.set_title("MUSE finding map: signal")
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = fig.add_subplot(gs[:, 1])
    im = ax.imshow(sigma_map, origin="lower", extent=extent, cmap="inferno", aspect="equal")
    ax.scatter(x[target_j], y[target_j], marker="*", s=220, facecolor="none", edgecolor="cyan", linewidth=1.6)
    ax.set_title("MUSE finding map: sigma")
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="km/s")

    ax_fit = fig.add_subplot(gs[0, 2])
    ax_res = fig.add_subplot(gs[1, 2], sharex=ax_fit)
    ax_fit.plot(lam, galaxy, color="0.25", lw=0.8, label="Binned spectrum")
    ax_fit.plot(lam, bestfit, color="tab:blue", lw=1.0, label="pPXF/XSL best fit")
    ax_fit.set_title(
        f"{label} | row={row[target_j]}, col={col[target_j]}, bin={k}, "
        f"sigma={sigma[target_j]:.1f} km/s, h3={h3[target_j]:+.3f}, "
        f"h4={h4[target_j]:+.3f}, signal={signal[target_j]:.3g}"
    )
    ax_fit.set_ylabel("Flux")
    ax_fit.legend(loc="best", frameon=False)
    ax_res.plot(lam, resid / scale, color="0.25", lw=0.7)
    ax_res.axhline(0, color="k", lw=0.7)
    ax_res.axhline(3, color="0.5", ls="--", lw=0.6)
    ax_res.axhline(-3, color="0.5", ls="--", lw=0.6)
    ax_res.set_xlabel("Rest wavelength [Angstrom]")
    ax_res.set_ylabel("Res./MAD")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    rows = []
    for summary in summaries:
        for key in ("brightest", "central_peak"):
            target = summary[key]
            rows.append(
                {
                    "dataset": summary["name"],
                    "target": key,
                    "sigma_kms": target.get("sigma_kms"),
                    "h3": target.get("h3"),
                    "h4": target.get("h4"),
                    "x_arcsec": target.get("x_arcsec"),
                    "y_arcsec": target.get("y_arcsec"),
                    "radius_arcsec": target.get("radius_arcsec"),
                    "row": target.get("row"),
                    "col": target.get("col"),
                    "bin_id": target.get("bin_id"),
                    "median_sigma_kms": summary["median_sigma_kms"],
                    "central_median_sigma_kms": summary["central_median_sigma_kms"],
                    "p90_sigma_kms": summary["p90_sigma_kms"],
                    "max_sigma_kms": summary["max_sigma_kms"],
                }
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_notes(path: Path, muse_summary: dict[str, object], nirspec_summary: dict[str, object]) -> None:
    lines = [
        "XSL MUSE/NIRSpec velocity-dispersion diagnostic notes",
        "",
        "Controlled terms in this comparison:",
        "- Both products use the pPXF XSL SPS library.",
        "- The recommended consistency run uses degree=8, mdegree=0, moments=4, and bias=0 for both MUSE and NIRSpec.",
        "- Both workflows pass an instrumental FWHM/LSF to pPXF's SPS loader so the templates are broadened to the data resolution when XSL is narrower than the data.",
        "",
        "Terms that can still move sigma even with XSL held fixed:",
        "- LSF assumptions: MUSE uses a constant optical FWHM in the original script; NIRSpec should use the wavelength-dependent G235H table.",
        "- Wavelength coverage: optical MUSE and K-band NIRSpec constrain different absorption features and template mismatch can differ by band.",
        "- Continuum treatment: additive degree and multiplicative mdegree alter line depths and can trade against sigma.",
        "- Kinematic moments and bias: legacy moments=2 products can differ because h3/h4 are not available to absorb non-Gaussian line shape.",
        "- Masking: telluric, gas, dust, bad pixels, and DQ masks change which features drive the fit.",
        "- Noise/S/N and binning: PowerBin MUSE spectra and native-spaxel NIRSpec spectra weight residuals differently.",
        "- Spatial PSF/aperture: MUSE seeing mixes central gradients; NIRSpec native pixels are much sharper unless explicitly PSF matched.",
        "- The MUSE fit checkplots are especially useful for gas-line residuals; central H-alpha/[N II]/[S II] residual spikes should be masked or fit simultaneously before trusting sigma.",
        "",
        f"MUSE central peak sigma: {muse_summary['central_peak'].get('sigma_kms')} km/s",
        f"NIRSpec central peak sigma: {nirspec_summary['central_peak'].get('sigma_kms')} km/s",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muse-fits", type=Path, default=DEFAULT_MUSE_FITS)
    parser.add_argument("--nirspec-fits", type=Path, default=DEFAULT_NIRSPEC_FITS)
    parser.add_argument("--nirspec-run-config", type=Path, default=DEFAULT_NIRSPEC_RUN_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--central-radius-arcsec", type=float, default=0.5)
    parser.add_argument("--nirspec-rotation-deg", type=float, default=-18.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.output_dir.resolve())

    muse = load_muse_product(args.muse_fits.resolve())
    nirspec = load_nirspec_product(args.nirspec_fits.resolve(), args.nirspec_rotation_deg)

    muse_summary = product_summary(muse, args.central_radius_arcsec)
    nirspec_summary = product_summary(nirspec, args.central_radius_arcsec)

    map_path = outdir / "xsl_muse_nirspec_sigma_finding_maps.png"
    plot_sigma_maps(map_path, muse, nirspec, args.central_radius_arcsec)

    muse_bright = brightest(muse)
    muse_peak = central_peak(muse, args.central_radius_arcsec)
    muse_checkplots: dict[str, str] = {}
    if muse_bright is not None:
        path = outdir / "muse_xsl_check_brightest_spaxel.png"
        plot_muse_checkplot_from_product(path, args.muse_fits.resolve(), muse_bright, "MUSE brightest spaxel")
        muse_checkplots["brightest_spaxel"] = str(path)
    if muse_peak is not None:
        radius_tag = f"{args.central_radius_arcsec:.2f}".replace(".", "p")
        path = outdir / f"muse_xsl_check_highest_sigma_within_{radius_tag}arcsec.png"
        plot_muse_checkplot_from_product(
            path,
            args.muse_fits.resolve(),
            muse_peak,
            f"MUSE highest sigma within {args.central_radius_arcsec:.2f} arcsec",
        )
        muse_checkplots["central_peak_sigma"] = str(path)

    nirspec_checkplots = {}
    if args.nirspec_run_config.is_file():
        try:
            payload = json.loads(args.nirspec_run_config.read_text())
            nirspec_checkplots = payload.get("targeted_checkplots", {})
        except json.JSONDecodeError:
            nirspec_checkplots = {}

    summary = {
        "central_radius_arcsec": float(args.central_radius_arcsec),
        "nirspec_rotation_deg": float(args.nirspec_rotation_deg),
        "muse": muse_summary,
        "nirspec": nirspec_summary,
        "outputs": {
            "finding_maps": str(map_path),
            "muse_checkplots": muse_checkplots,
            "nirspec_checkplots_from_run_config": nirspec_checkplots,
        },
    }
    json_path = outdir / "xsl_muse_nirspec_sigma_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = outdir / "xsl_muse_nirspec_sigma_summary.csv"
    write_summary_csv(csv_path, [muse_summary, nirspec_summary])
    notes_path = outdir / "xsl_muse_nirspec_sigma_notes.txt"
    write_notes(notes_path, muse_summary, nirspec_summary)

    print(f"Saved finding maps: {map_path}")
    print(f"Saved summary JSON: {json_path}")
    print(f"Saved summary CSV : {csv_path}")
    print(f"Saved notes       : {notes_path}")
    for name, path in muse_checkplots.items():
        print(f"Saved MUSE checkplot {name}: {path}")
    for name, path in nirspec_checkplots.items():
        print(f"NIRSpec checkplot from run config {name}: {path}")


if __name__ == "__main__":
    main()
