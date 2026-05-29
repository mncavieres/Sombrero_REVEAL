#!/usr/bin/env python3
"""
Fast MUSE XSL moment-4 kinematic refit from an existing pPXF product.

The full MUSE workflow also does regularized stellar-population fitting, which
is slow for exploratory h3/h4 checks. This script reuses the saved XSL binned
spectra and XSL optimal templates, refits only [V, sigma, h3, h4], and writes a
FITS/NPZ product with the same map/table conventions used by the comparison
diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_muse_xsl_mom4"))

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from tqdm import tqdm


ROOT = Path("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL")
DEFAULT_INPUT = ROOT / "Plots/ppxfppxf_c30_xsl/c30_DATACUBE_normppxf_skycont_Part1_0000_ppxf_products_xsl.fits"
DEFAULT_OUTPUT_DIR = ROOT / "Data/ppxf_muse/xsl_mom4_refit_from_existing"

C = 299792.458
REDSHIFT = 0.003633
SPS_NORM_RANGE = [5070.0, 5950.0]
KIN_DEGREE = 8
KIN_MDEGREE = 0
KIN_MOMENTS = 4
KIN_BIAS = 0.0
CHECK_PLOT_RADIUS_ARCSEC = 0.5
KIN_MASK_WINDOWS = [
    (4856.0, 4868.0),
    (4953.0, 4965.0),
    (4998.0, 5018.0),
    (5190.0, 5205.0),
    (6295.0, 6308.0),
    (6540.0, 6590.0),
    (6710.0, 6740.0),
]


@dataclass(frozen=True)
class Config:
    input_fits: Path
    output_dir: Path
    redshift: float
    degree: int
    mdegree: int
    moments: int
    bias: float
    check_plot_radius_arcsec: float


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_wavelength_mask_windows(mask: np.ndarray, lam: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for lo, hi in windows:
        out &= ~((lam >= lo) & (lam <= hi))
    return out


def safe_noise(galaxy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(galaxy, dtype=float)[mask & np.isfinite(galaxy)]
    if x.size < 10:
        x = np.asarray(galaxy, dtype=float)[np.isfinite(galaxy)]
    noise = robust_sigma(np.diff(x), zero=1) / np.sqrt(2.0) if x.size > 2 else np.nan
    if not np.isfinite(noise) or noise <= 0:
        noise = robust_sigma(x, zero=1)
    if not np.isfinite(noise) or noise <= 0:
        noise = np.nanstd(x)
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.0
    return np.full_like(galaxy, float(noise), dtype=float)


def correct_ppxf_errors(pp) -> np.ndarray | None:
    if getattr(pp, "error", None) is None:
        return None
    return np.asarray(pp.error, dtype=float) * np.sqrt(float(pp.chi2))


def load_xsl_lam_temp(velscale: float, redshift: float) -> np.ndarray:
    filename = Path(resources.files("ppxf") / "sps_models" / "spectra_xsl_9.0.npz")
    sps = lib.sps_lib(filename, velscale, 2.62 / (1.0 + redshift), norm_range=SPS_NORM_RANGE)
    return np.asarray(sps.lam_temp, dtype=float)


def fit_bin(
    galaxy: np.ndarray,
    template: np.ndarray,
    velscale: float,
    lam_gal: np.ndarray,
    lam_temp: np.ndarray,
    mask0: np.ndarray,
    start: list[float],
    cfg: Config,
):
    template = np.asarray(template, dtype=float)
    scale = np.nanmedian(template[np.isfinite(template)])
    if np.isfinite(scale) and scale != 0:
        template = template / scale
    noise = safe_noise(galaxy, mask0)
    pp = ppxf(
        template,
        galaxy,
        noise,
        velscale,
        start,
        moments=cfg.moments,
        bias=cfg.bias,
        degree=cfg.degree,
        mdegree=cfg.mdegree,
        lam=lam_gal,
        lam_temp=lam_temp,
        mask=mask0,
        quiet=True,
    )
    noise *= np.sqrt(float(pp.chi2))
    pp = ppxf(
        template,
        galaxy,
        noise,
        velscale,
        pp.sol,
        moments=cfg.moments,
        bias=cfg.bias,
        degree=cfg.degree,
        mdegree=cfg.mdegree,
        lam=lam_gal,
        lam_temp=lam_temp,
        mask=mask0,
        quiet=True,
    )
    err = correct_ppxf_errors(pp)
    resid = (pp.galaxy - pp.bestfit)[pp.goodpixels]
    sn = np.nanmedian(pp.galaxy[pp.goodpixels]) / robust_sigma(resid, zero=1)
    return pp, err, float(sn)


def image_extent(x: np.ndarray, y: np.ndarray, pixsize: float) -> tuple[float, float, float, float]:
    return (
        float(np.nanmin(x) - 0.5 * pixsize),
        float(np.nanmax(x) + 0.5 * pixsize),
        float(np.nanmin(y) - 0.5 * pixsize),
        float(np.nanmax(y) + 0.5 * pixsize),
    )


def plot_checkplot(
    outpath: Path,
    lam_gal: np.ndarray,
    galaxy: np.ndarray,
    bestfit: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    signal_map: np.ndarray,
    sigma_map: np.ndarray,
    pixsize: float,
    target_x: float,
    target_y: float,
    title: str,
) -> None:
    resid = galaxy - bestfit
    scale = robust_sigma(resid[np.isfinite(resid)], zero=1)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    extent = image_extent(x, y, pixsize)
    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 1.45])
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(signal_map, origin="lower", extent=extent, cmap="gray_r", aspect="equal")
    ax.scatter([target_x], [target_y], marker="*", s=220, facecolor="none", edgecolor="tab:red", linewidth=1.6)
    ax.set_title("MUSE signal")
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = fig.add_subplot(gs[:, 1])
    im = ax.imshow(sigma_map, origin="lower", extent=extent, cmap="inferno", aspect="equal")
    ax.scatter([target_x], [target_y], marker="*", s=220, facecolor="none", edgecolor="cyan", linewidth=1.6)
    ax.set_title("MUSE sigma")
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="km/s")

    ax_fit = fig.add_subplot(gs[0, 2])
    ax_res = fig.add_subplot(gs[1, 2], sharex=ax_fit)
    ax_fit.plot(lam_gal, galaxy, color="0.25", lw=0.8, label="Binned spectrum")
    ax_fit.plot(lam_gal, bestfit, color="tab:blue", lw=1.0, label="pPXF/XSL mom4")
    ax_fit.set_title(title)
    ax_fit.set_ylabel("Flux")
    ax_fit.legend(loc="best", frameon=False)
    ax_res.plot(lam_gal, resid / scale, color="0.25", lw=0.7)
    ax_res.axhline(0, color="k", lw=0.7)
    ax_res.axhline(3, color="0.5", ls="--", lw=0.6)
    ax_res.axhline(-3, color="0.5", ls="--", lw=0.6)
    ax_res.set_xlabel("Rest wavelength [Angstrom]")
    ax_res.set_ylabel("Res./MAD")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    cfg: Config,
    header: fits.Header,
    spax,
    lam_gal: np.ndarray,
    bin_spec: np.ndarray,
    bin_bestfit: np.ndarray,
    signal_map: np.ndarray,
    noise_map: np.ndarray,
    bin_num: np.ndarray,
    results: dict[str, np.ndarray],
    checkplots: dict[str, str],
) -> tuple[Path, Path, Path]:
    outdir = ensure_dir(cfg.output_dir)
    nbins = results["v"].size
    ny, nx = signal_map.shape
    bin_map = bin_num.reshape(ny, nx)
    sigma_map = results["sigma"][bin_num].reshape(ny, nx)

    hdr = fits.Header()
    hdr["OBJECT"] = header.get("OBJECT", "MUSE_XSL")[:68]
    hdr["SPSMOD"] = "xsl"
    hdr["METHOD"] = "mom4_refit_from_existing_xsl_optimal_templates"[:68]
    hdr["MOMENTS"] = int(cfg.moments)
    hdr["BIAS"] = float(cfg.bias)
    hdr["DEGREE"] = int(cfg.degree)
    hdr["MDEGREE"] = int(cfg.mdegree)
    hdr["VELSCAL"] = float(header.get("VELSCAL", np.nan))
    hdr["PIXSIZE"] = float(header.get("PIXSIZE", 0.2))
    hdr["REDSHFT"] = float(cfg.redshift)
    hdr["CHKSRAD"] = float(cfg.check_plot_radius_arcsec)

    bin_id = np.arange(nbins, dtype=np.int32)
    bin_cols = [
        fits.Column(name="BIN_ID", format="J", array=bin_id),
        fits.Column(name="V_KMS", format="D", array=results["v"]),
        fits.Column(name="SIGMA_KMS", format="D", array=results["sigma"]),
        fits.Column(name="H3", format="D", array=results["h3"]),
        fits.Column(name="H4", format="D", array=results["h4"]),
        fits.Column(name="VERR_KMS", format="D", array=results["verr"]),
        fits.Column(name="SIGERR_KMS", format="D", array=results["sigmaerr"]),
        fits.Column(name="H3_ERR", format="D", array=results["h3err"]),
        fits.Column(name="H4_ERR", format="D", array=results["h4err"]),
        fits.Column(name="SN_BIN", format="D", array=results["sn"]),
        fits.Column(name="CHI2", format="D", array=results["chi2"]),
    ]

    spax_cols = [
        fits.Column(name="ROW", format="J", array=np.asarray(spax["ROW"], dtype=np.int32)),
        fits.Column(name="COL", format="J", array=np.asarray(spax["COL"], dtype=np.int32)),
        fits.Column(name="X_ARCSEC", format="D", array=np.asarray(spax["X_ARCSEC"], dtype=float)),
        fits.Column(name="Y_ARCSEC", format="D", array=np.asarray(spax["Y_ARCSEC"], dtype=float)),
        fits.Column(name="SIGNAL", format="D", array=np.asarray(spax["SIGNAL"], dtype=float)),
        fits.Column(name="NOISE", format="D", array=np.asarray(spax["NOISE"], dtype=float)),
        fits.Column(name="BIN_ID", format="J", array=bin_num.astype(np.int32)),
        fits.Column(name="V_KMS", format="D", array=results["v"][bin_num]),
        fits.Column(name="SIGMA_KMS", format="D", array=results["sigma"][bin_num]),
        fits.Column(name="H3", format="D", array=results["h3"][bin_num]),
        fits.Column(name="H4", format="D", array=results["h4"][bin_num]),
        fits.Column(name="VERR_KMS", format="D", array=results["verr"][bin_num]),
        fits.Column(name="SIGERR_KMS", format="D", array=results["sigmaerr"][bin_num]),
        fits.Column(name="H3_ERR", format="D", array=results["h3err"][bin_num]),
        fits.Column(name="H4_ERR", format="D", array=results["h4err"][bin_num]),
    ]

    fits_path = outdir / f"{cfg.input_fits.stem}_mom4_kinematics.fits"
    hdus = [
        fits.PrimaryHDU(header=hdr),
        fits.BinTableHDU.from_columns(bin_cols, name="BIN_RESULTS"),
        fits.BinTableHDU.from_columns(spax_cols, name="SPAXELS"),
        fits.ImageHDU(data=bin_map.astype(np.int32), name="BIN_MAP"),
        fits.ImageHDU(data=results["v"][bin_num].reshape(ny, nx).astype(np.float32), name="VEL_MAP"),
        fits.ImageHDU(data=sigma_map.astype(np.float32), name="SIGMA_MAP"),
        fits.ImageHDU(data=results["h3"][bin_num].reshape(ny, nx).astype(np.float32), name="H3_MAP"),
        fits.ImageHDU(data=results["h4"][bin_num].reshape(ny, nx).astype(np.float32), name="H4_MAP"),
        fits.ImageHDU(data=signal_map.astype(np.float32), name="SIGNAL_MAP"),
        fits.ImageHDU(data=noise_map.astype(np.float32), name="NOISE_MAP"),
        fits.ImageHDU(data=lam_gal.astype(np.float32), name="LAMBDA_GAL"),
        fits.ImageHDU(data=bin_spec.astype(np.float32), name="BIN_SPEC"),
        fits.ImageHDU(data=bin_bestfit.astype(np.float32), name="BIN_BESTFIT"),
    ]
    fits.HDUList(hdus).writeto(fits_path, overwrite=True)

    npz_path = fits_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        lam_gal=lam_gal,
        bin_num=bin_num,
        signal_map=signal_map,
        noise_map=noise_map,
        bin_spec=bin_spec,
        bin_bestfit=bin_bestfit,
        checkplots=np.array(checkplots, dtype=object),
        **results,
    )

    csv_path = fits_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["BIN_ID", "V_KMS", "SIGMA_KMS", "H3", "H4", "SN_BIN", "CHI2"],
        )
        writer.writeheader()
        for j in range(nbins):
            writer.writerow(
                {
                    "BIN_ID": j,
                    "V_KMS": results["v"][j],
                    "SIGMA_KMS": results["sigma"][j],
                    "H3": results["h3"][j],
                    "H4": results["h4"][j],
                    "SN_BIN": results["sn"][j],
                    "CHI2": results["chi2"][j],
                }
            )
    return fits_path, npz_path, csv_path


def run(cfg: Config) -> None:
    outdir = ensure_dir(cfg.output_dir)
    with fits.open(cfg.input_fits, memmap=False) as hdul:
        header = hdul[0].header.copy()
        spax = hdul["SPAXELS"].data.copy()
        lam_gal = np.asarray(hdul["LAMBDA_GAL"].data, dtype=float)
        bin_spec = np.asarray(hdul["BIN_SPEC"].data, dtype=float)
        opt_templ = np.asarray(hdul["OPT_TEMPL"].data, dtype=float)
        signal_map = np.asarray(hdul["SIGNAL_MAP"].data, dtype=float)
        noise_map = np.asarray(hdul["NOISE_MAP"].data, dtype=float)
        old_bin = hdul["BIN_RESULTS"].data.copy()

    velscale = float(header["VELSCAL"])
    lam_temp = load_xsl_lam_temp(velscale, cfg.redshift)
    if lam_temp.size != opt_templ.shape[0]:
        raise ValueError(f"Template wavelength length {lam_temp.size} != OPT_TEMPL length {opt_templ.shape[0]}")

    bin_num = np.asarray(spax["BIN_ID"], dtype=int)
    nbins = bin_spec.shape[1]
    mask0 = util.determine_mask(np.log(lam_gal), [float(lam_temp[0]), float(lam_temp[-1])], width=1000)
    mask0 = apply_wavelength_mask_windows(mask0, lam_gal, KIN_MASK_WINDOWS)

    results = {
        "v": np.full(nbins, np.nan),
        "sigma": np.full(nbins, np.nan),
        "h3": np.full(nbins, np.nan),
        "h4": np.full(nbins, np.nan),
        "verr": np.full(nbins, np.nan),
        "sigmaerr": np.full(nbins, np.nan),
        "h3err": np.full(nbins, np.nan),
        "h4err": np.full(nbins, np.nan),
        "sn": np.full(nbins, np.nan),
        "chi2": np.full(nbins, np.nan),
    }
    bin_bestfit = np.full_like(bin_spec, np.nan)

    for j in tqdm(range(nbins), desc="Refitting MUSE XSL bins to h4"):
        start = [
            float(old_bin["V_KMS"][j]) if "V_KMS" in old_bin.names else 0.0,
            float(old_bin["SIGMA_KMS"][j]) if "SIGMA_KMS" in old_bin.names else 250.0,
            0.0,
            0.0,
        ]
        try:
            pp, err, sn = fit_bin(
                bin_spec[:, j],
                opt_templ[:, j],
                velscale,
                lam_gal,
                lam_temp,
                mask0,
                start,
                cfg,
            )
        except Exception as exc:
            print(f"WARNING: bin {j} failed: {exc}")
            continue
        sol = np.asarray(pp.sol, dtype=float)
        results["v"][j], results["sigma"][j], results["h3"][j], results["h4"][j] = sol[:4]
        if err is not None and err.size >= 4:
            results["verr"][j], results["sigmaerr"][j], results["h3err"][j], results["h4err"][j] = err[:4]
        results["sn"][j] = sn
        results["chi2"][j] = float(pp.chi2)
        bin_bestfit[:, j] = np.asarray(pp.bestfit, dtype=float)

    x = np.asarray(spax["X_ARCSEC"], dtype=float)
    y = np.asarray(spax["Y_ARCSEC"], dtype=float)
    sig_spax = results["sigma"][bin_num]
    bright = int(np.nanargmax(np.asarray(spax["SIGNAL"], dtype=float)))
    radius = np.hypot(x, y)
    central = np.flatnonzero(np.isfinite(sig_spax) & (radius <= cfg.check_plot_radius_arcsec))
    peak = int(central[np.nanargmax(sig_spax[central])]) if central.size else bright
    checkplots = {}
    for label, spax_idx in {
        "brightest_spaxel": bright,
        f"highest_sigma_within_{cfg.check_plot_radius_arcsec:.2f}arcsec": peak,
    }.items():
        k = int(bin_num[spax_idx])
        outpath = outdir / f"muse_xsl_mom4_check_{label.replace('.', 'p')}.png"
        title = (
            f"{label.replace('_', ' ')} | bin={k}, sigma={results['sigma'][k]:.1f} km/s, "
            f"h3={results['h3'][k]:+.3f}, h4={results['h4'][k]:+.3f}"
        )
        plot_checkplot(
            outpath,
            lam_gal,
            bin_spec[:, k],
            bin_bestfit[:, k],
            x,
            y,
            signal_map,
            results["sigma"][bin_num].reshape(signal_map.shape),
            float(header.get("PIXSIZE", 0.2)),
            float(x[spax_idx]),
            float(y[spax_idx]),
            title,
        )
        checkplots[label] = str(outpath)

    fits_path, npz_path, csv_path = save_outputs(
        cfg,
        header,
        spax,
        lam_gal,
        bin_spec,
        bin_bestfit,
        signal_map,
        noise_map,
        bin_num,
        results,
        checkplots,
    )
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "n_bins": int(nbins),
        "median_sigma": float(np.nanmedian(results["sigma"])),
        "central_peak_sigma": float(sig_spax[peak]),
        "central_peak_h3": float(results["h3"][bin_num[peak]]),
        "central_peak_h4": float(results["h4"][bin_num[peak]]),
        "outputs": {
            "fits": str(fits_path),
            "npz": str(npz_path),
            "csv": str(csv_path),
            "checkplots": checkplots,
        },
    }
    summary_path = outdir / "muse_xsl_mom4_refit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved FITS   : {fits_path}")
    print(f"Saved NPZ    : {npz_path}")
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved summary: {summary_path}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fits", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--redshift", type=float, default=REDSHIFT)
    parser.add_argument("--degree", type=int, default=KIN_DEGREE)
    parser.add_argument("--mdegree", type=int, default=KIN_MDEGREE)
    parser.add_argument("--moments", type=int, default=KIN_MOMENTS)
    parser.add_argument("--bias", type=float, default=KIN_BIAS)
    parser.add_argument("--check-plot-radius-arcsec", type=float, default=CHECK_PLOT_RADIUS_ARCSEC)
    args = parser.parse_args()
    return Config(
        input_fits=args.input_fits.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        redshift=float(args.redshift),
        degree=int(args.degree),
        mdegree=int(args.mdegree),
        moments=int(args.moments),
        bias=float(args.bias),
        check_plot_radius_arcsec=float(args.check_plot_radius_arcsec),
    )


if __name__ == "__main__":
    run(parse_args())
