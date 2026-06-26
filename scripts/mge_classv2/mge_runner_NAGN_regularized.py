import argparse
import os
import sys

sys.path.append('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/scripts/mge_classv2')

from astropy.io import fits
from astropy.table import Table
import numpy as np

from mge_gen3 import MGEFitter


PROJECT_ROOT = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL"
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "Data/mge_NAGN_0deg_pa_positive_gauss_regularized",
)
DEFAULT_DISTANCE_MPC = 9.55
DEFAULT_M_SUN_AB_F200W = 4.93


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _float_tag(value):
    return f"{float(value):.2f}".replace(".", "p")


def brightest_pixel_near(img, x0, y0, halfsize=50, goodmask=None):
    """
    Return brightest pixel near an initial guess.

    Public convention:
        input/output are (x, y) = (col, row)

    NumPy indexing:
        img[y, x]
    """
    ny, nx = img.shape

    x1 = max(0, int(round(x0 - halfsize)))
    x2 = min(nx, int(round(x0 + halfsize + 1)))
    y1 = max(0, int(round(y0 - halfsize)))
    y2 = min(ny, int(round(y0 + halfsize + 1)))

    cut = np.array(img[y1:y2, x1:x2], copy=True)

    if goodmask is not None:
        gm = goodmask[y1:y2, x1:x2]
        cut[~gm] = -np.inf

    iy, ix = np.unravel_index(np.nanargmax(cut), cut.shape)
    return x1 + ix, y1 + iy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the NAGN F200W MGE fit with Cappellari's regularized "
            "mge_fit_sectors wrapper."
        )
    )
    parser.add_argument(
        "--qbounds",
        type=float,
        nargs=2,
        default=(0.05, 1.0),
        metavar=("QMIN", "QMAX"),
        help=(
            "Initial observed-axis-ratio interval explored by the regularized "
            "wrapper. The wrapper tightens this range until ABSDEV rises too much."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for regularized checkplots, cache files, and summaries.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any matching cached regularized fit and recompute it.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable MGE diagnostic plotting during the fit.",
    )
    parser.add_argument(
        "--skip-mass-profile",
        action="store_true",
        help="Skip the enclosed-mass profile after fitting.",
    )
    parser.add_argument(
        "--distance-mpc",
        type=float,
        default=DEFAULT_DISTANCE_MPC,
        help="Distance used to convert the F200W MGE to DYNAMITE luminosity surface density.",
    )
    parser.add_argument(
        "--m-sun-ab-f200w",
        type=float,
        default=DEFAULT_M_SUN_AB_F200W,
        help="Adopted solar absolute AB magnitude in JWST/NIRCam F200W.",
    )
    parser.add_argument(
        "--pa-twist",
        type=float,
        default=0.0,
        help="PA_twist value written to the DYNAMITE MGE ECSV.",
    )
    parser.add_argument(
        "--dynamite-output-name",
        default=None,
        help=(
            "Filename for the DYNAMITE-ready MGE ECSV. "
            "Default: <prefix>_dynamite_mge.ecsv."
        ),
    )
    parser.set_defaults(plot=True)
    return parser.parse_args()


def summarize_fit(runner, path):
    total_counts, sigma_pix, q_obs = runner.fit_result.sol
    q_obs = np.asarray(q_obs, dtype=float)
    sigma_pix = np.asarray(sigma_pix, dtype=float)
    total_counts = np.asarray(total_counts, dtype=float)
    surf = total_counts / (2.0 * np.pi * q_obs * sigma_pix**2)
    sigma_arcsec = sigma_pix * float(runner.pixel_scale)

    table = np.vstack([surf, sigma_pix, sigma_arcsec, q_obs, total_counts]).T
    np.savetxt(
        path.replace(".txt", "_table.txt"),
        table,
        header="surf_counts_per_pix sigma_pix sigma_arcsec q_obs total_counts",
    )

    final_qbounds = getattr(runner.fit_result, "qbounds", None)
    absdev = getattr(runner.fit_result, "absdev", None)
    best_absdev = getattr(runner.fit_result, "best_absdev", None)

    lines = [
        "Regularized NAGN MGE fit",
        f"prefix: {runner.prefix}",
        f"checkplot_dir: {runner.checkplot_dir}",
        f"requested_qbounds: {runner.qbounds}",
        f"final_regularized_qbounds: {final_qbounds}",
        f"absdev: {absdev}",
        f"best_absdev_reference: {best_absdev}",
        f"n_gauss: {q_obs.size}",
        f"q_min: {np.min(q_obs):.8g}",
        f"q_p05: {np.percentile(q_obs, 5):.8g}",
        f"q_median: {np.median(q_obs):.8g}",
        f"q_max: {np.max(q_obs):.8g}",
        "q_sorted:",
        np.array2string(np.sort(q_obs), precision=6, separator=", "),
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _f200w_lsun_per_mjy(distance_mpc, m_sun_ab_f200w):
    return (
        (1e6 / 3631.0)
        * 10.0 ** (0.4 * float(m_sun_ab_f200w))
        * (1e5) ** 2
        * float(distance_mpc) ** 2
    )


def build_dynamite_mge_table(
    runner,
    *,
    distance_mpc=DEFAULT_DISTANCE_MPC,
    m_sun_ab_f200w=DEFAULT_M_SUN_AB_F200W,
    pa_twist=0.0,
):
    """
    Build DYNAMITE's mge.ecsv table: I, sigma, q, PA_twist.

    DYNAMITE's I is the Gaussian surface-brightness amplitude in Lsun/pc^2,
    while sigma is kept in arcsec.
    """
    total_counts, sigma_pix, q_obs = runner.fit_result.sol
    total_counts = np.asarray(total_counts, dtype=float)
    sigma_arcsec = np.asarray(sigma_pix, dtype=float) * float(runner.pixel_scale)
    q_obs = np.asarray(q_obs, dtype=float)

    pixel_area_sr = (float(runner.pixel_scale) / 206265.0) ** 2
    flux_mjy = total_counts * pixel_area_sr
    luminosity_lsun = flux_mjy * _f200w_lsun_per_mjy(
        distance_mpc=distance_mpc,
        m_sun_ab_f200w=m_sun_ab_f200w,
    )

    pc_per_arcsec = float(distance_mpc) * 1e6 / 206265.0
    intensity = luminosity_lsun / (
        2.0 * np.pi * sigma_arcsec**2 * q_obs * pc_per_arcsec**2
    )

    table = Table()
    table["I"] = intensity
    table["sigma"] = sigma_arcsec
    table["q"] = q_obs
    table["PA_twist"] = np.full(len(table), float(pa_twist))
    return table


def save_dynamite_mge(runner, path, **kwargs):
    table = build_dynamite_mge_table(runner, **kwargs)
    table.write(path, format="ascii.ecsv", overwrite=True)
    return path


if __name__ == "__main__":
    args = parse_args()
    qmin, qmax = args.qbounds
    if not (0.0 < qmin < qmax <= 1.0):
        raise ValueError("--qbounds must satisfy 0 < QMIN < QMAX <= 1")

    img_f200 = fits.open(
        os.path.join(PROJECT_ROOT, "Data/IFU/photometry/f200w_ifu_coadd_NOAGN_aligned.fits")
    )[0].data

    dust_mask = fits.open(
        os.path.join(PROJECT_ROOT, "Data/dust_mask/f200_mask_1.fits")
    )[0].data

    nan_mask = np.isnan(img_f200)
    if np.any(nan_mask):
        print(f"Found {np.sum(nan_mask)} NaN pixels in the image. Replacing with 0 and adding to dust mask.")
        img_f200[nan_mask] = 0.0
        dust_mask = dust_mask | nan_mask

    checkplot_dir = _ensure_dir(args.output_dir)
    prefix = f"sombrero_f200_regularized_q{_float_tag(qmin)}_{_float_tag(qmax)}"

    runner = MGEFitter(
        img_f200,
        dust_mask,
        pixel_scale=0.031,
        subtract_sky=False,
        linear=False,
        ngauss=30,
        qbounds=(qmin, qmax),
        regularized=True,
        plot=args.plot,
        checkplot_dir=checkplot_dir,
        cache_dir=checkplot_dir,
        prefix=prefix,
        contour_half_size_arcsec=20,
        contour_oversample=1,
        n_sectors=19,
        allow_negative=False,
        bulge_disk=False,
    )

    x_peak, y_peak = brightest_pixel_near(
        img_f200, 7538, 7333, halfsize=40, goodmask=runner.goodmask
    )

    print(f"Initial guess for galaxy center (x, y) [pix]: ({x_peak:.2f}, {y_peak:.2f})")
    print(
        f"Pixel value at center img[y, x] = img[{int(round(y_peak))}, {int(round(x_peak))}] = "
        f"{img_f200[int(round(y_peak)), int(round(x_peak))]}"
    )

    runner.set_manual_geometry(
        center=(x_peak, y_peak),
        pa_deg=0,
        eps=0.7060956459920877,
        theta_deg=90.78185872429874,
    )

    print(f"Stored manual center in runner: (x, y) = ({runner.xc:.2f}, {runner.yc:.2f})")
    print(f"Running regularized MGE with initial qbounds=({qmin:.3f}, {qmax:.3f})")

    runner.run_fit(force=args.force, load=not args.force)
    runner.save_final_results()

    summary_path = os.path.join(checkplot_dir, f"{prefix}_summary.txt")
    summarize_fit(runner, summary_path)
    print(f"Wrote fit summary to {summary_path}")

    dynamite_name = args.dynamite_output_name
    if dynamite_name is None:
        dynamite_name = f"{prefix}_dynamite_mge.ecsv"
    dynamite_mge_path = os.path.join(checkplot_dir, dynamite_name)
    save_dynamite_mge(
        runner,
        dynamite_mge_path,
        distance_mpc=args.distance_mpc,
        m_sun_ab_f200w=args.m_sun_ab_f200w,
        pa_twist=args.pa_twist,
    )
    print(f"Wrote DYNAMITE-ready MGE ECSV to {dynamite_mge_path}")

    if not args.skip_mass_profile:
        runner.compute_enclosed_mass_profile(
            checkplots_path=checkplot_dir,
            prefix=prefix,
            show=False,
        )
