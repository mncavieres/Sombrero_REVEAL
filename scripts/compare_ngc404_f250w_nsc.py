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
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Data/nsc_positive_control_ngc404"
DEFAULT_SOMBRERO_PROJECTED = REPO_ROOT / "Data/sersic_nsc_check/f200w_nuclear_sersic_summary.json"
DEFAULT_SOMBRERO_DEPROJECTED = REPO_ROOT / "Data/sersic_nsc_deproj_check/f200w_deprojected_sersic_summary.json"
DEFAULT_FITS = (
    DEFAULT_OUTPUT_DIR
    / "astroquery_downloads/mastDownload/HST/j8ff01010/j8ff01011_drz.fits"
)


@dataclass
class MastSelection:
    host_name: str = "NGC 404"
    known_nsc_host: bool = True
    ra_deg: float = 17.362595
    dec_deg: float = 35.718047
    radius_arcsec: float = 30.0
    obs_collection: str = "HST"
    instrument_name: str = "ACS/HRC"
    filter_name: str = "F250W"
    selected_obs_id: str = "j8ff01010"
    selected_product: str = "j8ff01011_drz.fits"


@dataclass
class FitConfig:
    output_dir: str
    fits_path: str
    science_hdu: str
    fit_rmin_arcsec: float
    fit_rmax_arcsec: float
    profile_rmax_arcsec: float
    min_bin_pixels: int
    distance_mpc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use astroquery/MAST to download NGC 404 ACS/HRC F250W imaging and "
            "compare its nuclear Sersic residual against the Sombrero checks."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fits-path", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--science-hdu", default="SCI")
    parser.add_argument("--fit-rmin-arcsec", type=float, default=0.3)
    parser.add_argument("--fit-rmax-arcsec", type=float, default=5.0)
    parser.add_argument("--profile-rmax-arcsec", type=float, default=8.0)
    parser.add_argument("--min-bin-pixels", type=int, default=20)
    parser.add_argument("--distance-mpc", type=float, default=3.06)
    parser.add_argument("--sombrero-projected-summary", type=Path, default=DEFAULT_SOMBRERO_PROJECTED)
    parser.add_argument("--sombrero-deprojected-summary", type=Path, default=DEFAULT_SOMBRERO_DEPROJECTED)
    return parser.parse_args()


def download_ngc404_product(output_dir: Path, fits_path: Path, force: bool) -> Path:
    if fits_path.exists() and not force:
        return fits_path

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.mast import Observations

    selection = MastSelection()
    coord = SkyCoord(selection.ra_deg, selection.dec_deg, unit="deg")
    observations = Observations.query_criteria(
        coordinates=coord,
        radius=selection.radius_arcsec * u.arcsec,
        obs_collection=selection.obs_collection,
        instrument_name=selection.instrument_name,
        filters=selection.filter_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    observations.write(output_dir / "ngc404_acs_hrc_f250w_observations.ecsv", format="ascii.ecsv", overwrite=True)

    chosen_obs = observations[observations["obs_id"] == selection.selected_obs_id]
    if len(chosen_obs) != 1:
        raise RuntimeError(f"Expected one selected observation {selection.selected_obs_id}, found {len(chosen_obs)}.")

    products = Observations.get_product_list(chosen_obs)
    products.write(output_dir / "ngc404_acs_hrc_f250w_products.ecsv", format="ascii.ecsv", overwrite=True)
    chosen_product = products[products["productFilename"] == selection.selected_product]
    if len(chosen_product) != 1:
        raise RuntimeError(f"Expected one selected product {selection.selected_product}, found {len(chosen_product)}.")

    manifest = Observations.download_products(
        chosen_product,
        download_dir=str(output_dir / "astroquery_downloads"),
    )
    manifest.write(output_dir / "ngc404_acs_hrc_f250w_download_manifest.ecsv", format="ascii.ecsv", overwrite=True)
    local_path = Path(str(manifest["Local Path"][0]))
    if not local_path.is_absolute():
        local_path = REPO_ROOT / local_path
    return local_path


def pixel_scale_arcsec(header: fits.Header) -> float:
    cd = np.array(
        [
            [header["CD1_1"], header["CD1_2"]],
            [header["CD2_1"], header["CD2_2"]],
        ],
        dtype=float,
    )
    return float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)


def find_bright_center(image: np.ndarray, valid: np.ndarray, margin: int = 50) -> tuple[float, float]:
    search = valid.copy()
    search[:margin, :] = False
    search[-margin:, :] = False
    search[:, :margin] = False
    search[:, -margin:] = False
    work = np.array(image, dtype=float, copy=True)
    work[~search] = -np.inf
    y_peak, x_peak = np.unravel_index(np.nanargmax(work), work.shape)

    half = 4
    y1 = max(0, y_peak - half)
    y2 = min(image.shape[0], y_peak + half + 1)
    x1 = max(0, x_peak - half)
    x2 = min(image.shape[1], x_peak + half + 1)
    cut = np.asarray(image[y1:y2, x1:x2], dtype=float)
    good = np.isfinite(cut)
    baseline = np.nanpercentile(cut[good], 20) if np.any(good) else 0.0
    weights = np.clip(cut - baseline, 0.0, None)
    if np.sum(weights) <= 0:
        return float(x_peak), float(y_peak)
    yy, xx = np.indices(cut.shape)
    x_cent = float(np.sum((xx + x1) * weights) / np.sum(weights))
    y_cent = float(np.sum((yy + y1) * weights) / np.sum(weights))
    return x_cent, y_cent


def b_n(n: float | np.ndarray) -> float | np.ndarray:
    n = np.asarray(n)
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)


def sersic_profile(r_arcsec: np.ndarray, params: np.ndarray) -> np.ndarray:
    log_ie, log_re, log_n, sky = params
    ie = np.exp(log_ie)
    re = np.exp(log_re)
    n = np.exp(log_n)
    return sky + ie * np.exp(-b_n(n) * ((r_arcsec / re) ** (1.0 / n) - 1.0))


def radial_profile(
    image: np.ndarray,
    valid: np.ndarray,
    radius: np.ndarray,
    rmax: float,
    min_bin_pixels: int,
) -> dict[str, np.ndarray]:
    edges = np.geomspace(0.03, rmax, 130)
    rows = []
    for r1, r2 in zip(edges[:-1], edges[1:]):
        sel = valid & (radius >= r1) & (radius < r2)
        count = int(np.count_nonzero(sel))
        if count < min_bin_pixels:
            continue
        values = image[sel]
        median = float(np.median(values))
        mean = float(np.mean(values))
        mad = float(1.4826 * np.median(np.abs(values - median)))
        err = max(mad / math.sqrt(count), 0.02 * abs(median), 1e-5)
        rows.append((math.sqrt(r1 * r2), r1, r2, median, mean, err, count))
    table = np.array(rows, dtype=float)
    return {
        "r_arcsec": table[:, 0],
        "r_inner_arcsec": table[:, 1],
        "r_outer_arcsec": table[:, 2],
        "median_electrons_per_s": table[:, 3],
        "mean_electrons_per_s": table[:, 4],
        "err_electrons_per_s": table[:, 5],
        "n_pix": table[:, 6].astype(int),
    }


def fit_sersic(profile: dict[str, np.ndarray], rmin: float, rmax: float) -> least_squares:
    r = profile["r_arcsec"]
    y = profile["median_electrons_per_s"]
    err = profile["err_electrons_per_s"]
    fit = (r >= rmin) & (r <= rmax) & np.isfinite(y) & (y > 0.0)
    if np.count_nonzero(fit) < 8:
        raise ValueError("Not enough positive profile bins to fit.")

    def residual(params: np.ndarray) -> np.ndarray:
        model = sersic_profile(r[fit], params)
        out = np.empty_like(model)
        bad = (~np.isfinite(model)) | (model <= 0.0)
        out[bad] = 1e9
        out[~bad] = (np.log(model[~bad]) - np.log(y[fit][~bad])) / (err[fit][~bad] / y[fit][~bad])
        return out

    p0 = np.array([np.log(np.median(y[fit])), np.log(1.0), np.log(1.0), np.percentile(y[fit], 10)])
    lower = np.array([np.log(1e-9), np.log(0.01), np.log(0.2), -1.0])
    upper = np.array([np.log(1e3), np.log(100.0), np.log(12.0), 1.0])
    return least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=20000,
    )


def aperture_diagnostics(
    image: np.ndarray,
    valid: np.ndarray,
    radius: np.ndarray,
    model: np.ndarray,
    distance_mpc: float,
) -> list[dict[str, float | int]]:
    pc_per_arcsec = distance_mpc * 1.0e6 / 206265.0
    rows = []
    frac_image = (image - model) / model
    for aperture_r in (0.1, 0.2, 0.3, 0.5, 1.0):
        sel = valid & (radius < aperture_r)
        observed = float(np.sum(image[sel]))
        predicted = float(np.sum(model[sel]))
        residual = observed - predicted
        rows.append(
            {
                "radius_arcsec": aperture_r,
                "radius_pc": aperture_r * pc_per_arcsec,
                "valid_pixels": int(np.count_nonzero(sel)),
                "observed_electrons_per_s": observed,
                "sersic_electrons_per_s": predicted,
                "residual_electrons_per_s": residual,
                "residual_fraction_of_sersic": residual / predicted,
                "median_fractional_residual": float(np.median(frac_image[sel])),
            }
        )
    return rows


def write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    lines = [",".join(names)]
    for values in zip(*(columns[name] for name in names)):
        lines.append(",".join(str(v) for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sombrero_metric(summary: dict | None, radius_arcsec: float, kind: str) -> dict | None:
    if summary is None:
        return None
    key = "aperture_diagnostics" if kind == "projected" else "enclosed_luminosity_diagnostics"
    for row in summary.get(key, []):
        if abs(float(row["radius_arcsec"]) - radius_arcsec) < 1e-8:
            if kind == "projected":
                frac = row["residual_flux_jy"] / row["sersic_flux_jy"]
                return {
                    "radius_arcsec": radius_arcsec,
                    "residual_fraction": frac,
                    "residual": row["residual_flux_jy"],
                    "unit": "Jy",
                }
            return {
                "radius_arcsec": radius_arcsec,
                "residual_fraction": row["enclosed_fractional_residual"],
                "residual": row["enclosed_residual_lsun"],
                "unit": "Lsun",
            }
    return None


def make_plots(
    output_dir: Path,
    image: np.ndarray,
    valid: np.ndarray,
    radius: np.ndarray,
    profile: dict[str, np.ndarray],
    model_profile: np.ndarray,
    model_image: np.ndarray,
    center_xy: tuple[float, float],
    pixel_scale: float,
    fit_rmin: float,
    fit_rmax: float,
    comparison: dict,
) -> None:
    r = profile["r_arcsec"]
    y = profile["median_electrons_per_s"]
    err = profile["err_electrons_per_s"]
    frac = (y - model_profile) / model_profile

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].errorbar(r, y, yerr=err, fmt="o", ms=3, lw=0.8, label="NGC 404 ACS/HRC F250W")
    axes[0].plot(r, model_profile, color="crimson", lw=1.8, label="Host Sersic fit")
    axes[0].axvspan(r[0], fit_rmin, color="0.85", alpha=0.8, label="Excluded NSC core")
    axes[0].axvline(fit_rmax, color="0.5", ls=":", lw=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Surface brightness (electrons/s)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0.0, color="0.25", lw=1.0)
    axes[1].plot(r, frac, "o", color="black", ms=3)
    axes[1].axvspan(r[0], fit_rmin, color="0.85", alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Radius (arcsec)")
    axes[1].set_ylabel("(data-model)/model")
    fig.tight_layout()
    fig.savefig(output_dir / "ngc404_acs_hrc_f250w_sersic_profile.png", dpi=220)
    plt.close(fig)

    x0, y0 = center_xy
    half_pix = int(round(2.0 / pixel_scale))
    cx = int(round(x0))
    cy = int(round(y0))
    y1 = max(0, cy - half_pix)
    y2 = min(image.shape[0], cy + half_pix + 1)
    x1 = max(0, cx - half_pix)
    x2 = min(image.shape[1], cx + half_pix + 1)
    data_cut = image[y1:y2, x1:x2]
    model_cut = model_image[y1:y2, x1:x2]
    valid_cut = valid[y1:y2, x1:x2]
    resid_cut = np.where(valid_cut, (data_cut - model_cut) / model_cut, np.nan)
    extent = ((x1 - x0) * pixel_scale, (x2 - x0) * pixel_scale, (y1 - y0) * pixel_scale, (y2 - y0) * pixel_scale)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), constrained_layout=True)
    for ax, arr, title, cmap, lim in (
        (axes[0], data_cut, "NGC 404 F250W", "magma", None),
        (axes[1], model_cut, "Sersic host", "magma", None),
        (axes[2], resid_cut, "fractional residual", "coolwarm", (-1.0, 5.0)),
    ):
        if lim is None:
            good = np.isfinite(arr) & valid_cut
            vmin, vmax = np.nanpercentile(arr[good], [5, 99]) if np.any(good) else (0, 1)
        else:
            vmin, vmax = lim
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.plot(0, 0, "+", color="cyan", ms=10, mew=1.5)
        ax.set_title(title)
        ax.set_xlabel("Delta x (arcsec)")
        ax.set_ylabel("Delta y (arcsec)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "ngc404_acs_hrc_f250w_residual_map.png", dpi=220)
    plt.close(fig)

    labels = []
    values = []
    colors = []
    ngc = comparison.get("ngc404_projected_r0p3")
    if ngc is not None:
        labels.append("NGC 404\nF250W projected")
        values.append(ngc["residual_fraction"])
        colors.append("tab:green")
    for name, label, color in (
        ("sombrero_projected_r0p3", "Sombrero\nF200W projected", "tab:blue"),
        ("sombrero_deprojected_r0p3", "Sombrero\nF200W deproj.", "tab:orange"),
    ):
        row = comparison.get(name)
        if row is not None:
            labels.append(label)
            values.append(row["residual_fraction"])
            colors.append(color)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0.0, color="0.2", lw=1.0)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel('Central residual fraction at r <= 0.3"')
    ax.set_title("Positive-control NSC versus Sombrero")
    fig.tight_layout()
    fig.savefig(output_dir / "ngc404_vs_sombrero_central_residual_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = download_ngc404_product(args.output_dir, args.fits_path, args.force_download)

    with fits.open(fits_path, memmap=True) as hdul:
        primary_header = hdul[0].header.copy()
        sci = hdul[args.science_hdu]
        image = np.asarray(sci.data, dtype=float)
        header = sci.header.copy()
        weight = np.asarray(hdul["WHT"].data, dtype=float) if "WHT" in hdul else np.ones_like(image)

    scale = pixel_scale_arcsec(header)
    valid = np.isfinite(image) & np.isfinite(weight) & (weight > 0.0)
    center_x, center_y = find_bright_center(image, valid)
    yy, xx = np.indices(image.shape)
    radius = np.hypot(xx - center_x, yy - center_y) * scale
    valid &= radius <= args.profile_rmax_arcsec

    profile = radial_profile(image, valid, radius, args.profile_rmax_arcsec, args.min_bin_pixels)
    fit = fit_sersic(profile, args.fit_rmin_arcsec, args.fit_rmax_arcsec)
    model_profile = sersic_profile(profile["r_arcsec"], fit.x)
    model_image = sersic_profile(radius, fit.x)
    aperture_rows = aperture_diagnostics(image, valid, radius, model_image, args.distance_mpc)

    profile_columns = {
        **profile,
        "sersic_electrons_per_s": model_profile,
        "fractional_residual": (profile["median_electrons_per_s"] - model_profile) / model_profile,
    }
    write_csv(args.output_dir / "ngc404_acs_hrc_f250w_sersic_profile.csv", profile_columns)

    selected_aperture = next(row for row in aperture_rows if abs(row["radius_arcsec"] - 0.3) < 1e-8)
    sombrero_projected = sombrero_metric(load_json(args.sombrero_projected_summary), 0.3, "projected")
    sombrero_deprojected = sombrero_metric(load_json(args.sombrero_deprojected_summary), 0.3, "deprojected")
    comparison = {
        "ngc404_projected_r0p3": {
            "radius_arcsec": 0.3,
            "residual_fraction": selected_aperture["residual_fraction_of_sersic"],
            "residual": selected_aperture["residual_electrons_per_s"],
            "unit": "electrons/s",
        },
        "sombrero_projected_r0p3": sombrero_projected,
        "sombrero_deprojected_r0p3": sombrero_deprojected,
    }
    make_plots(
        args.output_dir,
        image,
        valid,
        radius,
        profile,
        model_profile,
        model_image,
        (center_x, center_y),
        scale,
        args.fit_rmin_arcsec,
        args.fit_rmax_arcsec,
        comparison,
    )

    selection = MastSelection()
    fit_summary = {
        "success": bool(fit.success),
        "message": fit.message,
        "robust_cost": float(fit.cost),
        "Ie_electrons_per_s": float(np.exp(fit.x[0])),
        "Re_arcsec": float(np.exp(fit.x[1])),
        "n": float(np.exp(fit.x[2])),
        "sky_electrons_per_s": float(fit.x[3]),
    }
    summary = {
        "mast_selection": asdict(selection),
        "config": asdict(
            FitConfig(
                output_dir=str(args.output_dir),
                fits_path=str(fits_path),
                science_hdu=args.science_hdu,
                fit_rmin_arcsec=args.fit_rmin_arcsec,
                fit_rmax_arcsec=args.fit_rmax_arcsec,
                profile_rmax_arcsec=args.profile_rmax_arcsec,
                min_bin_pixels=args.min_bin_pixels,
                distance_mpc=args.distance_mpc,
            )
        ),
        "image": {
            "pixel_scale_arcsec": scale,
            "center_x": center_x,
            "center_y": center_y,
            "bunit": header.get("BUNIT"),
            "instrument": header.get("INSTRUME", primary_header.get("INSTRUME", "ACS")),
            "detector": header.get("DETECTOR", primary_header.get("DETECTOR", "HRC")),
            "filter1": header.get("FILTER1", primary_header.get("FILTER1")),
            "filter2": header.get("FILTER2", primary_header.get("FILTER2")),
            "photflam": header.get("PHOTFLAM"),
            "photplam": header.get("PHOTPLAM"),
        },
        "sersic_fit": fit_summary,
        "aperture_diagnostics": aperture_rows,
        "comparison": comparison,
        "interpretation": (
            "NGC 404, a known nuclear-star-cluster host, shows a strong positive "
            "central residual above the host Sersic fit in ACS/HRC F250W, while "
            "the Sombrero projected and deprojected checks are negative at the "
            "same 0.3 arcsec aperture."
        ),
    }
    (args.output_dir / "ngc404_acs_hrc_f250w_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(fit_summary, indent=2))
    print("central comparison")
    print(json.dumps(comparison, indent=2))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
