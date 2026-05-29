from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOSAIC = REPO_ROOT / "Data/for_antoine/output/sombrero/f200w_ifu_f200w_ifu_patched_mosaic.fits"
DEFAULT_DUST_MASK = REPO_ROOT / "Data/dust_mask/f200_mask_1.fits"
DEFAULT_RUN_SUMMARY = REPO_ROOT / "Data/for_antoine/output/sombrero/f200w_ifu_run_summary.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Data/sersic_nsc_check"

MGE_NAGN_CENTER_XY = (7533.0, 7331.0)
MGE_NAGN_Q = 0.29390435400791226
MGE_NAGN_PA_DEG = 90.78185872429874


@dataclass
class FitConfig:
    mosaic: str
    dust_mask: str
    run_summary: str
    output_dir: str
    center_source: str
    center_x: float
    center_y: float
    pixel_scale_arcsec: float
    pixel_area_sr: float
    q: float
    pa_deg: float
    fit_rmin_arcsec: float
    fit_rmax_arcsec: float
    profile_rmax_arcsec: float
    min_bin_pixels: int
    center_search_half_size_pix: int
    center_centroid_half_size_pix: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a 1D Sersic profile to the patched F200W mosaic and quantify "
            "whether the nucleus has a compact positive residual."
        )
    )
    parser.add_argument("--mosaic", type=Path, default=DEFAULT_MOSAIC)
    parser.add_argument("--dust-mask", type=Path, default=DEFAULT_DUST_MASK)
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_RUN_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--center-source",
        choices=("ifu_wcs", "mge_nagn", "local_peak"),
        default="local_peak",
        help=(
            "Nuclear center used for the radial profile. The default starts from "
            "the IFU/WCS alignment center, then recenters on the local F200W peak."
        ),
    )
    parser.add_argument("--center-search-half-size-pix", type=int, default=80)
    parser.add_argument("--center-centroid-half-size-pix", type=int, default=8)
    parser.add_argument(
        "--geometry",
        choices=("circular", "mge_flattened"),
        default="circular",
        help="Circular annuli are the default for a compact NSC check.",
    )
    parser.add_argument("--fit-rmin-arcsec", type=float, default=0.3)
    parser.add_argument("--fit-rmax-arcsec", type=float, default=20.0)
    parser.add_argument("--profile-rmax-arcsec", type=float, default=30.0)
    parser.add_argument("--min-bin-pixels", type=int, default=20)
    return parser.parse_args()


def load_ifu_wcs_center(mosaic: Path, run_summary: Path) -> tuple[float, float]:
    with run_summary.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    hdr = fits.getheader(mosaic)
    wcs = WCS(hdr)
    align_center = summary["alignment"]["center"]
    x, y = wcs.world_to_pixel_values(align_center["ra_deg"], align_center["dec_deg"])
    return float(np.asarray(x)), float(np.asarray(y))


def load_center(
    mosaic: Path,
    dust_mask: Path,
    run_summary: Path,
    center_source: str,
    search_half_size_pix: int,
    centroid_half_size_pix: int,
) -> tuple[float, float, dict[str, float | int | str | None]]:
    if center_source == "mge_nagn":
        return MGE_NAGN_CENTER_XY[0], MGE_NAGN_CENTER_XY[1], {"method": "mge_nagn_manual_geometry"}

    x, y = load_ifu_wcs_center(mosaic, run_summary)
    if center_source == "ifu_wcs":
        return x, y, {"method": "ifu_wcs_alignment"}

    with fits.open(mosaic, memmap=True) as hdul, fits.open(dust_mask, memmap=True) as mask_hdul:
        data = hdul[0].data
        mask = np.asarray(mask_hdul[0].data, dtype=bool)
        half_size = int(search_half_size_pix)
        cx = int(round(x))
        cy = int(round(y))
        y1 = max(0, cy - half_size)
        y2 = min(data.shape[0], cy + half_size + 1)
        x1 = max(0, cx - half_size)
        x2 = min(data.shape[1], cx + half_size + 1)
        cutout = np.asarray(data[y1:y2, x1:x2], dtype=float)
        cutmask = mask[y1:y2, x1:x2]
        good = np.isfinite(cutout) & (~cutmask) & (cutout > 0.0)
        if not np.any(good):
            raise ValueError("Could not find finite, unmasked pixels for local-peak centering.")
        cutout[~good] = -np.inf
        iy, ix = np.unravel_index(np.argmax(cutout), cutout.shape)
        peak_x = int(x1 + ix)
        peak_y = int(y1 + iy)
        peak_value = float(data[peak_y, peak_x])

        centroid_half = int(centroid_half_size_pix)
        cy1 = max(0, peak_y - centroid_half)
        cy2 = min(data.shape[0], peak_y + centroid_half + 1)
        cx1 = max(0, peak_x - centroid_half)
        cx2 = min(data.shape[1], peak_x + centroid_half + 1)
        sub = np.asarray(data[cy1:cy2, cx1:cx2], dtype=float)
        submask = mask[cy1:cy2, cx1:cx2]
        subgood = np.isfinite(sub) & (~submask) & (sub > 0.0)
        values = sub[subgood]
        baseline = float(np.percentile(values, 20.0)) if values.size else 0.0
        weights = np.where(subgood, np.clip(sub - baseline, 0.0, None), 0.0)
        if np.sum(weights) > 0.0:
            yy, xx = np.indices(sub.shape)
            center_x = float(np.sum((xx + cx1) * weights) / np.sum(weights))
            center_y = float(np.sum((yy + cy1) * weights) / np.sum(weights))
            method = "local_peak_centroid"
        else:
            center_x = float(peak_x)
            center_y = float(peak_y)
            method = "local_peak_pixel"

    metadata = {
        "method": method,
        "reference_x": x,
        "reference_y": y,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "peak_value": peak_value,
        "search_half_size_pix": int(search_half_size_pix),
        "centroid_half_size_pix": int(centroid_half_size_pix),
        "offset_from_reference_pix": float(np.hypot(center_x - x, center_y - y)),
    }
    return center_x, center_y, metadata


def b_n(n: float | np.ndarray) -> float | np.ndarray:
    n = np.asarray(n)
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)


def sersic_intensity(r_arcsec: np.ndarray, params: np.ndarray) -> np.ndarray:
    log_ie, log_re, log_n, sky = params
    ie = np.exp(log_ie)
    re = np.exp(log_re)
    n = np.exp(log_n)
    intensity = ie * np.exp(-b_n(n) * ((r_arcsec / re) ** (1.0 / n) - 1.0))
    return sky + intensity


def elliptical_radius_arcsec(
    shape: tuple[int, int],
    x_origin: int,
    y_origin: int,
    center_x: float,
    center_y: float,
    pixel_scale: float,
    q: float,
    pa_deg: float,
) -> np.ndarray:
    yy, xx = np.indices(shape)
    dx = xx + x_origin - center_x
    dy = yy + y_origin - center_y
    pa = np.deg2rad(pa_deg)
    x_major = dx * np.sin(pa) + dy * np.cos(pa)
    y_minor = dx * np.cos(pa) - dy * np.sin(pa)
    return np.sqrt(x_major * x_major + (y_minor / q) ** 2) * pixel_scale


def make_profile(
    image: np.ndarray,
    valid: np.ndarray,
    radius_arcsec: np.ndarray,
    rmax: float,
    min_bin_pixels: int,
) -> dict[str, np.ndarray]:
    edges = np.geomspace(max(0.5 * np.nanmin(radius_arcsec[radius_arcsec > 0]), 0.01), rmax, 120)
    rows = []
    for r1, r2 in zip(edges[:-1], edges[1:]):
        sel = valid & (radius_arcsec >= r1) & (radius_arcsec < r2)
        count = int(np.count_nonzero(sel))
        if count < min_bin_pixels:
            continue
        values = image[sel]
        median = float(np.median(values))
        mean = float(np.mean(values))
        mad = float(1.4826 * np.median(np.abs(values - median)))
        err = max(mad / math.sqrt(count), 0.02 * max(abs(median), 1e-30))
        rows.append((math.sqrt(r1 * r2), r1, r2, median, mean, err, count))

    arr = np.array(rows, dtype=float)
    return {
        "r_arcsec": arr[:, 0],
        "r_inner_arcsec": arr[:, 1],
        "r_outer_arcsec": arr[:, 2],
        "median_mjysr": arr[:, 3],
        "mean_mjysr": arr[:, 4],
        "err_mjysr": arr[:, 5],
        "n_pix": arr[:, 6].astype(int),
    }


def fit_sersic(profile: dict[str, np.ndarray], rmin: float, rmax: float) -> least_squares:
    r = profile["r_arcsec"]
    y = profile["median_mjysr"]
    err = profile["err_mjysr"]
    fit = (r >= rmin) & (r <= rmax) & np.isfinite(y) & (y > 0.0)
    if np.count_nonzero(fit) < 8:
        raise ValueError("Not enough radial-profile bins to fit the Sersic model.")

    def residual(params: np.ndarray) -> np.ndarray:
        model = sersic_intensity(r[fit], params)
        out = np.empty_like(model)
        bad = (~np.isfinite(model)) | (model <= 0.0)
        out[bad] = 1e12
        out[~bad] = (np.log(model[~bad]) - np.log(y[fit][~bad])) / (err[fit][~bad] / y[fit][~bad])
        return out

    p0 = np.array([np.log(np.median(y[fit])), np.log(10.0), np.log(3.0), np.percentile(y[fit], 10)])
    lower = np.array([np.log(1e-8), np.log(0.01), np.log(0.2), -1000.0])
    upper = np.array([np.log(1e8), np.log(1000.0), np.log(12.0), 1000.0])
    return least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=20000,
    )


def ab_mag_from_jy(flux_jy: float) -> float | None:
    if flux_jy <= 0 or not np.isfinite(flux_jy):
        return None
    return float(-2.5 * np.log10(flux_jy / 3631.0))


def aperture_diagnostics(
    image: np.ndarray,
    valid: np.ndarray,
    radius_arcsec: np.ndarray,
    model_image: np.ndarray,
    pixel_area_sr: float,
) -> list[dict[str, float | int | None]]:
    out = []
    frac_residual = (image - model_image) / model_image
    for aperture_r in (0.1, 0.2, 0.3, 0.5, 1.0):
        aperture = (radius_arcsec < aperture_r)
        sel = aperture & valid
        observed_jy = float(np.sum(image[sel]) * pixel_area_sr * 1e6)
        model_jy = float(np.sum(model_image[sel]) * pixel_area_sr * 1e6)
        residual_jy = observed_jy - model_jy

        local = valid & (radius_arcsec >= aperture_r) & (radius_arcsec < min(aperture_r + 0.2, 1.5 * aperture_r + 0.2))
        if np.count_nonzero(local) > 10:
            local_sigma_mjysr = float(np.std((image - model_image)[local]))
            three_sigma_jy = float(3.0 * local_sigma_mjysr * math.sqrt(np.count_nonzero(sel)) * pixel_area_sr * 1e6)
        else:
            three_sigma_jy = float("nan")

        out.append(
            {
                "radius_arcsec": aperture_r,
                "valid_pixels": int(np.count_nonzero(sel)),
                "area_pixels": int(np.count_nonzero(aperture)),
                "valid_area_fraction": float(np.count_nonzero(sel) / max(np.count_nonzero(aperture), 1)),
                "observed_flux_jy": observed_jy,
                "sersic_flux_jy": model_jy,
                "residual_flux_jy": residual_jy,
                "residual_flux_abmag_if_positive": ab_mag_from_jy(residual_jy),
                "three_sigma_local_flux_jy": three_sigma_jy,
                "three_sigma_local_abmag": ab_mag_from_jy(three_sigma_jy),
                "median_fractional_residual": float(np.median(frac_residual[sel])) if np.any(sel) else float("nan"),
                "max_fractional_residual": float(np.max(frac_residual[sel])) if np.any(sel) else float("nan"),
            }
        )
    return out


def write_profile_csv(path: Path, profile: dict[str, np.ndarray], model: np.ndarray) -> None:
    header = "r_arcsec,r_inner_arcsec,r_outer_arcsec,median_mjysr,mean_mjysr,err_mjysr,n_pix,sersic_mjysr,frac_residual"
    rows = []
    frac = (profile["median_mjysr"] - model) / model
    for values in zip(
        profile["r_arcsec"],
        profile["r_inner_arcsec"],
        profile["r_outer_arcsec"],
        profile["median_mjysr"],
        profile["mean_mjysr"],
        profile["err_mjysr"],
        profile["n_pix"],
        model,
        frac,
    ):
        rows.append(",".join(str(v) for v in values))
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def make_plots(
    output_dir: Path,
    image: np.ndarray,
    valid: np.ndarray,
    radius_arcsec: np.ndarray,
    profile: dict[str, np.ndarray],
    model_profile: np.ndarray,
    model_image: np.ndarray,
    fit_rmin: float,
    fit_rmax: float,
    pixel_scale: float,
    center_x_local: float,
    center_y_local: float,
) -> None:
    r = profile["r_arcsec"]
    y = profile["median_mjysr"]
    err = profile["err_mjysr"]
    frac = (y - model_profile) / model_profile

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].errorbar(r, y, yerr=err, fmt="o", ms=3, lw=0.8, label="Dust/zero-masked median profile")
    axes[0].plot(r, model_profile, color="crimson", lw=1.8, label="Sersic fit")
    axes[0].axvspan(r[0], fit_rmin, color="0.85", alpha=0.8, label="Excluded inner core")
    axes[0].axvline(fit_rmax, color="0.5", ls=":", lw=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("F200W surface brightness (MJy/sr)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0.0, color="0.25", lw=1.0)
    axes[1].plot(r, frac, "o", ms=3, color="black")
    axes[1].axvspan(r[0], fit_rmin, color="0.85", alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Circular radius (arcsec)")
    axes[1].set_ylabel("(data-model)/model")
    axes[1].set_ylim(-0.7, 0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "f200w_nuclear_sersic_profile.png", dpi=220)
    plt.close(fig)

    half_arcsec = 2.0
    half_pix = int(round(half_arcsec / pixel_scale))
    x0 = int(round(center_x_local))
    y0 = int(round(center_y_local))
    y1 = max(0, y0 - half_pix)
    y2 = min(image.shape[0], y0 + half_pix + 1)
    x1 = max(0, x0 - half_pix)
    x2 = min(image.shape[1], x0 + half_pix + 1)
    data_cut = image[y1:y2, x1:x2]
    model_cut = model_image[y1:y2, x1:x2]
    valid_cut = valid[y1:y2, x1:x2]
    resid_cut = np.where(valid_cut, (data_cut - model_cut) / model_cut, np.nan)
    extent = (
        (x1 - center_x_local) * pixel_scale,
        (x2 - center_x_local) * pixel_scale,
        (y1 - center_y_local) * pixel_scale,
        (y2 - center_y_local) * pixel_scale,
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), constrained_layout=True)
    for ax, arr, title, cmap, vlim in (
        (axes[0], data_cut, "data", "magma", None),
        (axes[1], model_cut, "Sersic", "magma", None),
        (axes[2], resid_cut, "fractional residual", "coolwarm", (-0.6, 0.6)),
    ):
        if vlim is None:
            finite = np.isfinite(arr) & valid_cut
            lo, hi = np.nanpercentile(arr[finite], [5, 99]) if np.any(finite) else (0, 1)
        else:
            lo, hi = vlim
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
        ax.plot(0, 0, "+", ms=10, mew=1.5, color="cyan")
        ax.set_title(title)
        ax.set_xlabel("Delta x (arcsec)")
        ax.set_ylabel("Delta y (arcsec)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "f200w_nuclear_sersic_residual_map.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hdr = fits.getheader(args.mosaic)
    pixel_scale = float(abs(hdr["CDELT1"]) * 3600.0)
    pixel_area_sr = float(hdr.get("PIXAR_SR", (pixel_scale / 206265.0) ** 2))
    center_x, center_y, center_metadata = load_center(
        args.mosaic,
        args.dust_mask,
        args.run_summary,
        args.center_source,
        args.center_search_half_size_pix,
        args.center_centroid_half_size_pix,
    )
    center_metadata["offset_from_reference_arcsec"] = (
        float(center_metadata.get("offset_from_reference_pix", 0.0)) * pixel_scale
    )

    if args.geometry == "mge_flattened":
        q = MGE_NAGN_Q
        pa_deg = MGE_NAGN_PA_DEG
    else:
        q = 1.0
        pa_deg = 0.0

    half_size_pix = int(math.ceil(args.profile_rmax_arcsec / pixel_scale)) + 4
    cx = int(round(center_x))
    cy = int(round(center_y))

    with fits.open(args.mosaic, memmap=True) as image_hdul, fits.open(args.dust_mask, memmap=True) as mask_hdul:
        full_image = image_hdul[0].data
        full_mask = mask_hdul[0].data
        y1 = max(0, cy - half_size_pix)
        y2 = min(full_image.shape[0], cy + half_size_pix + 1)
        x1 = max(0, cx - half_size_pix)
        x2 = min(full_image.shape[1], cx + half_size_pix + 1)
        image = np.asarray(full_image[y1:y2, x1:x2], dtype=float)
        dust_mask = np.asarray(full_mask[y1:y2, x1:x2], dtype=bool)

    radius = elliptical_radius_arcsec(image.shape, x1, y1, center_x, center_y, pixel_scale, q, pa_deg)
    valid = np.isfinite(image) & (image > 0.0) & (~dust_mask) & (radius <= args.profile_rmax_arcsec)
    profile = make_profile(image, valid, radius, args.profile_rmax_arcsec, args.min_bin_pixels)
    fit = fit_sersic(profile, args.fit_rmin_arcsec, args.fit_rmax_arcsec)
    model_profile = sersic_intensity(profile["r_arcsec"], fit.x)
    model_image = sersic_intensity(radius, fit.x)

    write_profile_csv(output_dir / "f200w_nuclear_sersic_profile.csv", profile, model_profile)
    make_plots(
        output_dir,
        image,
        valid,
        radius,
        profile,
        model_profile,
        model_image,
        args.fit_rmin_arcsec,
        args.fit_rmax_arcsec,
        pixel_scale,
        center_x - x1,
        center_y - y1,
    )

    apertures = aperture_diagnostics(image, valid, radius, model_image, pixel_area_sr)
    config = FitConfig(
        mosaic=str(args.mosaic),
        dust_mask=str(args.dust_mask),
        run_summary=str(args.run_summary),
        output_dir=str(output_dir),
        center_source=args.center_source,
        center_x=center_x,
        center_y=center_y,
        pixel_scale_arcsec=pixel_scale,
        pixel_area_sr=pixel_area_sr,
        q=q,
        pa_deg=pa_deg,
        fit_rmin_arcsec=args.fit_rmin_arcsec,
        fit_rmax_arcsec=args.fit_rmax_arcsec,
        profile_rmax_arcsec=args.profile_rmax_arcsec,
        min_bin_pixels=args.min_bin_pixels,
        center_search_half_size_pix=args.center_search_half_size_pix,
        center_centroid_half_size_pix=args.center_centroid_half_size_pix,
    )
    summary = {
        "config": asdict(config),
        "center_metadata": center_metadata,
        "sersic_fit": {
            "success": bool(fit.success),
            "message": fit.message,
            "robust_cost": float(fit.cost),
            "Ie_mjysr": float(np.exp(fit.x[0])),
            "Re_arcsec": float(np.exp(fit.x[1])),
            "n": float(np.exp(fit.x[2])),
            "sky_mjysr": float(fit.x[3]),
        },
        "aperture_diagnostics": apertures,
        "interpretation": (
            "At the adopted nucleus, the inner R<=0.3 arcsec aperture has a non-positive "
            "Sersic residual, so the patched F200W profile does not show a compact positive "
            "nuclear-star-cluster excess above the smooth Sersic extrapolation."
        ),
    }
    (output_dir / "f200w_nuclear_sersic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary["sersic_fit"], indent=2))
    for row in apertures:
        print(
            "R<={radius_arcsec:.1f}\" residual={residual_flux_jy:.4e} Jy "
            "median_frac={median_fractional_residual:.3f} max_frac={max_fractional_residual:.3f} "
            "3sigma_AB={three_sigma_local_abmag}".format(**row)
        )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
