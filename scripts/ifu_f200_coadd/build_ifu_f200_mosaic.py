from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from alignment import align_ifu_to_nircam
from diagnostics import (
    build_profile_table,
    software_versions,
    write_json,
    write_patch_diagnostic_fits,
    write_profile_table,
    write_scale_pixel_table,
    write_scaling_checkplot,
)
from fits_utils import write_primary_image
from filter_catalog import (
    compatible_filters_for_ifu,
    infer_nirspec_band,
    normalize_filter_name,
    throughput_path_for_filter,
)
from mast_query import query_and_download_matching_jwst_imaging
from mosaic import coadd_f200_images, patch_mosaic_with_ifu, write_patched_mosaic
from photometry import build_ifu_f200_image, write_ifu_photometry


DEFAULT_THROUGHPUT_DIR = REPO_ROOT / "Data/for_antoine/nircam_throughputs/mean_throughputs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Data/IFU/f200_coadd_tool"
DEFAULT_PREFIX = "f200w_ifu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    def parse_bool(value: str) -> bool:
        value = str(value).strip().lower()
        if value in {"true", "t", "yes", "y", "1"}:
            return True
        if value in {"false", "f", "no", "n", "0"}:
            return False
        raise argparse.ArgumentTypeError("Expected True or False.")

    parser = argparse.ArgumentParser(
        description="Build a NIRCam mosaic patched with a synthetic NIRCam-filter NIRSpec IFU image."
    )
    parser.add_argument("--ifu", required=True, type=Path, help="Path to the NIRSpec IFU cube FITS file.")
    parser.add_argument(
        "--f200",
        required=False,
        nargs="+",
        type=Path,
        help="One or more NIRCam FITS images to coadd before patching. Optional when --mast-query is used.",
    )
    parser.add_argument(
        "--throughput",
        type=Path,
        default=None,
        help="Throughput table with Microns and Throughput columns. Defaults to the active filter in --throughput-dir.",
    )
    parser.add_argument(
        "--throughput-dir",
        type=Path,
        default=DEFAULT_THROUGHPUT_DIR,
        help="Directory of mean NIRCam throughput tables used for MAST filter compatibility and default throughput lookup.",
    )
    parser.add_argument(
        "--imaging-filter",
        default="F200W",
        help="NIRCam filter label for explicitly supplied --f200 images. MAST mode overrides this with the selected filter.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output FITS files.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Output filename prefix.")
    parser.add_argument("--ifu-hdu", type=int, default=None, help="IFU cube HDU index. Defaults to first 3D image.")
    parser.add_argument("--f200-hdu", type=int, default=None, help="NIRCam image HDU index. Defaults to first 2D image.")
    parser.add_argument("--erode-pixels", type=int, default=1, help="Pixels by which to erode the finite IFU footprint.")
    parser.add_argument(
        "--no-photon-weighted",
        action="store_true",
        help="Use throughput only instead of throughput times wavelength.",
    )

    center = parser.add_mutually_exclusive_group()
    center.add_argument(
        "--align-center-sky",
        nargs=2,
        type=float,
        metavar=("RA_DEG", "DEC_DEG"),
        help="Alignment cutout center in ICRS degrees.",
    )
    center.add_argument(
        "--align-center-pixel",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Alignment cutout center in NIRCam mosaic pixels.",
    )
    parser.add_argument("--align-size", nargs=2, type=int, default=(200, 200), metavar=("NY", "NX"))
    parser.add_argument("--max-shift-arcsec", type=float, default=1.0)
    parser.add_argument("--coarse-step-pix", type=float, default=0.5)
    parser.add_argument("--min-valid-pixels", type=int, default=100)
    parser.add_argument(
        "--align-mask-reference-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude exact-zero NIRCam reference pixels from the chi2 alignment fit.",
    )
    parser.add_argument(
        "--align-reference-zero-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for masking zero-valued reference pixels during alignment.",
    )

    parser.add_argument(
        "--scale-mode",
        choices=("median_ratio", "linear_fit", "none"),
        default="median_ratio",
        help="How to fit the multiplicative IFU flux scale on finite NIRCam overlap.",
    )
    parser.add_argument(
        "--scaling",
        type=parse_bool,
        default=True,
        metavar="True|False",
        help="Fit and apply the IFU flux scale. Use False to patch without rescaling the IFU.",
    )
    parser.add_argument("--scale-factor", type=float, default=None, help="Manual IFU scale factor; overrides --scale-mode.")
    parser.add_argument("--min-scale-pixels", type=int, default=25)
    parser.add_argument(
        "--scale-mask-reference-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude exact-zero NIRCam reference pixels from the IFU flux-scale fit.",
    )
    parser.add_argument(
        "--scale-reference-zero-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for masking zero-valued reference pixels during flux scaling.",
    )
    parser.add_argument("--write-footprints", action="store_true", help="Write mask/footprint extensions in output FITS.")
    parser.add_argument(
        "--profile-pa-deg",
        type=float,
        default=90.0,
        help="Major-axis position angle for profile diagnostics, in degrees east of north.",
    )
    parser.add_argument(
        "--profile-half-length-arcsec",
        type=float,
        default=None,
        help="Half-length of major/minor profile cuts. Defaults to the IFU patch size plus padding.",
    )
    parser.add_argument("--profile-step-pix", type=float, default=1.0, help="Profile sampling step in mosaic pixels.")
    parser.add_argument("--profile-width-pix", type=int, default=3, help="Cross-axis width averaged in profile cuts.")
    parser.add_argument("--diagnostic-pad-pixels", type=int, default=20, help="Padding around IFU patch diagnostic cutout.")
    parser.add_argument("--no-checkplot", action="store_true", help="Skip the scaling/profile PNG checkplot.")
    parser.add_argument("--no-diagnostic-fits", action="store_true", help="Skip the compact diagnostic FITS cutout.")

    mast = parser.add_argument_group("MAST discovery")
    mast.add_argument(
        "--mast-query",
        action="store_true",
        help="Query MAST for matching JWST/NIRCam imaging instead of passing --f200 paths.",
    )
    mast.add_argument("--mast-filter", default="F200W", help="MAST/NIRCam filter to query and download.")
    mast.add_argument(
        "--mast-filter-fallback",
        choices=("prompt", "auto", "off"),
        default="prompt",
        help="What to do if the requested MAST filter is not found but compatible filters are available.",
    )
    mast.add_argument(
        "--mast-fallback-filter",
        default=None,
        help="Explicit compatible filter to use if --mast-filter is unavailable.",
    )
    mast.add_argument(
        "--mast-fallback-min-response-fraction",
        type=float,
        default=0.75,
        help="Minimum throughput response fraction inside the IFU wavelength range for fallback-compatible filters.",
    )
    mast.add_argument(
        "--mast-radius-arcsec",
        type=float,
        default=None,
        help="MAST cone-search radius. Overrides the automatic radius.",
    )
    mast.add_argument(
        "--mast-min-radius-arcsec",
        type=float,
        default=180.0,
        help="Minimum automatic MAST cone-search radius. Default catches the three Sombrero NIRCam tiles.",
    )
    mast.add_argument("--mast-padding-arcsec", type=float, default=5.0, help="Padding added to the IFU footprint radius.")
    mast.add_argument("--mast-instrument", default="NIRCAM/IMAGE", help="MAST instrument_name criterion.")
    mast.add_argument("--mast-product-subgroup", default="I2D", help="Product subgroup to mosaic, usually I2D.")
    mast.add_argument("--mast-calib-level", type=int, default=3, help="MAST calibration level criterion.")
    mast.add_argument("--mast-proposal-id", default=None, help="Optional JWST proposal ID filter.")
    mast.add_argument("--mast-max-products", type=int, default=None, help="Optional cap on selected imaging products.")
    mast.add_argument(
        "--mast-no-connectivity",
        action="store_true",
        help="Disable footprint graph selection and use all matching MAST products in the query radius.",
    )
    mast.add_argument(
        "--mast-connect-tolerance-arcsec",
        type=float,
        default=1.0,
        help="Tolerance for deciding whether MAST image footprints touch/overlap.",
    )
    mast.add_argument(
        "--mast-download-dir",
        type=Path,
        default=None,
        help="Directory for MAST query tables and downloads. Defaults to OUTPUT_DIR/mast_downloads.",
    )
    mast.add_argument("--mast-dry-run", action="store_true", help="Query MAST and write tables, but do not download or mosaic.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args(argv)
    if args.mast_query and args.f200:
        parser.error("Use either --mast-query or --f200 paths, not both.")
    if not args.mast_query and not args.f200:
        parser.error("Pass --f200 paths, or use --mast-query to discover matching imaging from MAST.")
    return args


def default_alignment_center(ifu_hdu) -> SkyCoord:
    """Use the center of the synthetic IFU image as the default sky center."""
    wcs = WCS(ifu_hdu.header).celestial
    ny, nx = ifu_hdu.data.shape
    return wcs.pixel_to_world(0.5 * (nx - 1), 0.5 * (ny - 1))


def alignment_center_summary(center) -> dict:
    if isinstance(center, SkyCoord):
        icrs = center.icrs
        return {"kind": "sky", "ra_deg": float(icrs.ra.deg), "dec_deg": float(icrs.dec.deg)}
    return {"kind": "pixel", "x": float(center[0]), "y": float(center[1])}


def compact_alignment_summary(alignment: dict) -> dict:
    keys = (
        "dx_pix",
        "dy_pix",
        "dRA_arcsec",
        "dDec_arcsec",
        "chi2",
        "scale",
        "background",
        "n_fit_pixels",
        "n_base_pixels",
        "n_reference_zero_masked",
        "max_shift_pix",
        "coarse_best_dx_pix",
        "coarse_best_dy_pix",
        "coarse_best_chi2",
        "boundary_fraction_x",
        "boundary_fraction_y",
        "near_search_boundary",
        "mask_reference_zero",
        "reference_zero_atol",
    )
    summary = {key: alignment[key] for key in keys if key in alignment}
    res = alignment.get("optimizer_result")
    if res is not None:
        summary["optimizer"] = {
            "success": bool(getattr(res, "success", False)),
            "message": str(getattr(res, "message", "")),
            "nfev": int(getattr(res, "nfev", -1)),
            "nit": int(getattr(res, "nit", -1)),
        }
    return summary


def mast_result_summary(mast_result) -> dict | None:
    if mast_result is None:
        return None
    payload = asdict(mast_result)
    payload["image_paths"] = [str(path) for path in mast_result.image_paths]
    return payload


def filter_tag(filter_name: str) -> str:
    return normalize_filter_name(filter_name).lower()


def build_output_paths(output_dir: Path, prefix: str, filter_name: str) -> dict[str, Path]:
    tag = filter_tag(filter_name)
    return {
        "mosaic": output_dir / f"{prefix}_{tag}_mosaic.fits",
        "ifu_raw": output_dir / f"{prefix}_nirspec_{tag}_raw.fits",
        "ifu_aligned": output_dir / f"{prefix}_nirspec_{tag}_aligned.fits",
        "patched": output_dir / f"{prefix}_{tag}_ifu_patched_mosaic.fits",
        "scale_pixels": output_dir / f"{prefix}_scale_pixels.ecsv",
        "profiles": output_dir / f"{prefix}_axis_profiles.ecsv",
        "checkplot": output_dir / f"{prefix}_scaling_checkplot.png",
        "diagnostic": output_dir / f"{prefix}_diagnostic_cutout.fits",
        "summary": output_dir / f"{prefix}_run_summary.json",
    }


def format_filter_rows(table, *, response_column: str = "ifu_response_fraction") -> str:
    if len(table) == 0:
        return "none"
    rows = []
    for row in table:
        filter_name = normalize_filter_name(row["filter"])
        if response_column in table.colnames:
            rows.append(f"{filter_name} ({float(row[response_column]):.3f})")
        else:
            rows.append(filter_name)
    return ", ".join(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_scale_mode = args.scale_mode if args.scaling else "none"
    effective_scale_factor = args.scale_factor if args.scaling else None

    mast_result = None
    compatible_filters = None
    ifu_wave_range = None
    active_filter_name = normalize_filter_name(args.imaging_filter)
    if args.mast_query:
        mast_dir = args.mast_download_dir or (args.output_dir / "mast_downloads")
        compatible_filters, ifu_wave_range = compatible_filters_for_ifu(
            args.ifu,
            args.throughput_dir,
            ifu_hdu_index=args.ifu_hdu,
            min_response_fraction=args.mast_fallback_min_response_fraction,
        )
        ifu_band = infer_nirspec_band(*ifu_wave_range)
        print(f"IFU band: {ifu_band} ({ifu_wave_range[0]:.4f}-{ifu_wave_range[1]:.4f} micron)")
        print(
            "Compatible JWST/NIRCam filters from this IFU range "
            f"(response fraction >= {args.mast_fallback_min_response_fraction:.2f}): "
            f"{format_filter_rows(compatible_filters)}"
        )
        print("Querying MAST for matching JWST/NIRCam imaging...")
        mast_result = query_and_download_matching_jwst_imaging(
            ifu_path=args.ifu,
            output_dir=mast_dir,
            ifu_hdu_index=args.ifu_hdu,
            filter_name=args.mast_filter,
            radius_arcsec=args.mast_radius_arcsec,
            min_radius_arcsec=args.mast_min_radius_arcsec,
            padding_arcsec=args.mast_padding_arcsec,
            instrument_name=args.mast_instrument,
            product_subgroup=args.mast_product_subgroup,
            calib_level=args.mast_calib_level,
            proposal_id=args.mast_proposal_id,
            max_products=args.mast_max_products,
            connectivity=not args.mast_no_connectivity,
            connect_tolerance_arcsec=args.mast_connect_tolerance_arcsec,
            throughput_dir=args.throughput_dir,
            filter_fallback_mode=args.mast_filter_fallback,
            fallback_filter=args.mast_fallback_filter,
            fallback_min_response_fraction=args.mast_fallback_min_response_fraction,
            dry_run=args.mast_dry_run,
            overwrite=args.overwrite,
        )
        active_filter_name = mast_result.filter_name
        found = ", ".join(mast_result.filters_found) if mast_result.filters_found else "none"
        print(f"MAST filters found in the query footprint: {found}")
        if mast_result.fallback_used:
            print(f"Using fallback filter {mast_result.filter_name} instead of requested {mast_result.requested_filter_name}.")
        print(
            f"Selected {mast_result.n_filtered_products} {mast_result.filter_name} product(s) "
            f"from {mast_result.n_candidate_products} candidate product(s)."
        )

        output_prefix = args.prefix
        if output_prefix == DEFAULT_PREFIX and normalize_filter_name(active_filter_name) != "F200W":
            output_prefix = f"{filter_tag(active_filter_name)}_ifu"
        paths = build_output_paths(args.output_dir, output_prefix, active_filter_name)

        if args.mast_dry_run:
            write_json(
                paths["summary"],
                {
                    "software": software_versions(),
                    "inputs": {
                        "ifu": args.ifu,
                        "throughput": args.throughput,
                        "throughput_dir": args.throughput_dir,
                        "active_filter": active_filter_name,
                    },
                    "ifu_band": {
                        "name": infer_nirspec_band(*ifu_wave_range),
                        "wavelength_min_micron": ifu_wave_range[0],
                        "wavelength_max_micron": ifu_wave_range[1],
                        "compatible_filters": [
                            {
                                "filter": normalize_filter_name(row["filter"]),
                                "ifu_response_fraction": float(row["ifu_response_fraction"]),
                            }
                            for row in compatible_filters
                        ],
                    },
                    "mast": mast_result_summary(mast_result),
                    "outputs": {"run_summary": paths["summary"]},
                    "dry_run": True,
                },
                overwrite=args.overwrite,
            )
            print("MAST dry run complete.")
            print(f"Observations table: {mast_result.observations_path}")
            print(f"All products table: {mast_result.products_path}")
            print(f"Candidate products table: {mast_result.candidate_products_path}")
            print(f"Selected products table: {mast_result.filtered_products_path}")
            print(f"Run summary: {paths['summary']}")
            return 0
        if not mast_result.image_paths:
            raise RuntimeError("MAST query did not download any FITS images suitable for mosaicking.")
        f200_paths = mast_result.image_paths
    else:
        f200_paths = args.f200
        output_prefix = args.prefix
        paths = build_output_paths(args.output_dir, output_prefix, active_filter_name)

    throughput_path = args.throughput or throughput_path_for_filter(active_filter_name, args.throughput_dir)

    f200_mosaic_path = paths["mosaic"]
    ifu_raw_path = paths["ifu_raw"]
    ifu_aligned_path = paths["ifu_aligned"]
    patched_path = paths["patched"]
    scale_pixels_path = paths["scale_pixels"]
    profile_table_path = paths["profiles"]
    checkplot_path = paths["checkplot"]
    diagnostic_fits_path = paths["diagnostic"]
    summary_path = paths["summary"]

    print(f"Using {active_filter_name} throughput table: {throughput_path}")
    print(f"Coadding {len(f200_paths)} {active_filter_name} image(s) with reproject_interp...")
    coadd_f200_images(
        f200_paths,
        f200_mosaic_path,
        hdu_index=args.f200_hdu,
        overwrite=args.overwrite,
        write_footprint=args.write_footprints,
    )

    print(f"Building synthetic {active_filter_name} image from the IFU cube...")
    ifu_result = build_ifu_f200_image(
        args.ifu,
        throughput_path,
        ifu_hdu_index=args.ifu_hdu,
        erode_pixels=args.erode_pixels,
        photon_weighted=not args.no_photon_weighted,
    )
    write_ifu_photometry(ifu_result, ifu_raw_path, overwrite=args.overwrite)

    with fits.open(f200_mosaic_path, memmap=True) as f200_hdul, fits.open(ifu_raw_path, memmap=True) as ifu_hdul:
        f200_hdu = f200_hdul[0]
        ifu_hdu = ifu_hdul[0]

        if args.align_center_sky is not None:
            center = SkyCoord(args.align_center_sky[0] * u.deg, args.align_center_sky[1] * u.deg, frame="icrs")
        elif args.align_center_pixel is not None:
            center = tuple(args.align_center_pixel)
        else:
            center = default_alignment_center(ifu_hdu)
        center_summary = alignment_center_summary(center)

        print(f"Running chi2 IFU-to-{active_filter_name} alignment...")
        alignment = align_ifu_to_nircam(
            nircam_hdu=f200_hdu,
            ifu_hdu=ifu_hdu,
            center=center,
            size=tuple(args.align_size),
            max_shift_arcsec=args.max_shift_arcsec,
            coarse_step_pix=args.coarse_step_pix,
            min_valid_pixels=args.min_valid_pixels,
            mask_reference_zero=args.align_mask_reference_zero,
            reference_zero_atol=args.align_reference_zero_atol,
        )
        if alignment.get("near_search_boundary"):
            print(
                "Warning: best alignment is near the search boundary "
                f"(x={alignment['boundary_fraction_x']:.2f}, y={alignment['boundary_fraction_y']:.2f}). "
                "Consider increasing --max-shift-arcsec or checking the alignment diagnostics."
            )

        aligned_header = alignment["aligned_header"]
        aligned_header["SRCFILE"] = (ifu_raw_path.name, "Synthetic IFU image before alignment")
        write_primary_image(
            ifu_aligned_path,
            ifu_hdu.data,
            aligned_header,
            overwrite=args.overwrite,
        )

    with fits.open(f200_mosaic_path, memmap=True) as f200_hdul, fits.open(ifu_aligned_path, memmap=True) as ifu_hdul:
        print("Reprojecting aligned IFU image, fitting flux scale, and patching mosaic...")
        patch = patch_mosaic_with_ifu(
            f200_hdul[0],
            ifu_hdul[0],
            scale_mode=effective_scale_mode,
            scale_factor=effective_scale_factor,
            min_scale_pixels=args.min_scale_pixels,
            mask_reference_zero=args.scale_mask_reference_zero,
            reference_zero_atol=args.scale_reference_zero_atol,
        )
        write_patched_mosaic(
            patch,
            f200_hdul[0].header,
            patched_path,
            overwrite=args.overwrite,
            write_footprint=args.write_footprints,
        )

        print("Writing scale diagnostics, profile table, and checkplot...")
        write_scale_pixel_table(scale_pixels_path, f200_hdul[0].data, patch, overwrite=args.overwrite)
        profile_table = build_profile_table(
            f200_hdul[0],
            patch,
            pa_deg=args.profile_pa_deg,
            half_length_arcsec=args.profile_half_length_arcsec,
            step_pix=args.profile_step_pix,
            width_pix=args.profile_width_pix,
            pad_pix=float(args.diagnostic_pad_pixels),
        )
        write_profile_table(profile_table_path, profile_table, overwrite=args.overwrite)
        if not args.no_diagnostic_fits:
            write_patch_diagnostic_fits(
                diagnostic_fits_path,
                f200_hdul[0],
                patch,
                pad_pixels=args.diagnostic_pad_pixels,
                overwrite=args.overwrite,
            )
        if not args.no_checkplot:
            write_scaling_checkplot(
                checkplot_path,
                f200_hdul[0],
                patch,
                profile_table,
                scale_pixels_path,
                pad_pixels=args.diagnostic_pad_pixels,
                overwrite=args.overwrite,
            )

    summary = {
        "software": software_versions(),
        "inputs": {
            "ifu": args.ifu,
            "f200": f200_paths,
            "imaging_filter": active_filter_name,
            "throughput": throughput_path,
            "throughput_dir": args.throughput_dir,
            "ifu_hdu": args.ifu_hdu,
            "f200_hdu": args.f200_hdu,
        },
        "ifu_band": None
        if ifu_wave_range is None
        else {
            "name": infer_nirspec_band(*ifu_wave_range),
            "wavelength_min_micron": ifu_wave_range[0],
            "wavelength_max_micron": ifu_wave_range[1],
            "compatible_filters": [
                {
                    "filter": normalize_filter_name(row["filter"]),
                    "ifu_response_fraction": float(row["ifu_response_fraction"]),
                }
                for row in compatible_filters
            ],
        },
        "mast": mast_result_summary(mast_result),
        "outputs": {
            "f200_mosaic": f200_mosaic_path,
            "ifu_raw": ifu_raw_path,
            "ifu_aligned": ifu_aligned_path,
            "patched_mosaic": patched_path,
            "scale_pixels": scale_pixels_path,
            "axis_profiles": profile_table_path,
            "diagnostic_cutout": None if args.no_diagnostic_fits else diagnostic_fits_path,
            "scaling_checkplot": None if args.no_checkplot else checkplot_path,
            "run_summary": summary_path,
        },
        "ifu_photometry": {
            "source_hdu_index": ifu_result.source_hdu_index,
            "image_shape": ifu_result.image.shape,
            "uneroded_finite_pixels": int(np.count_nonzero(np.isfinite(ifu_result.uneroded_image))),
            "eroded_finite_pixels": int(np.count_nonzero(np.isfinite(ifu_result.image))),
            "eroded_pixels_removed": int(
                np.count_nonzero(np.isfinite(ifu_result.uneroded_image))
                - np.count_nonzero(np.isfinite(ifu_result.image))
            ),
            "erode_pixels": args.erode_pixels,
            "photon_weighted": not args.no_photon_weighted,
            "wavelength_min_micron": float(np.nanmin(ifu_result.wavelength_microns)),
            "wavelength_max_micron": float(np.nanmax(ifu_result.wavelength_microns)),
            "throughput_denominator": ifu_result.denominator,
        },
        "alignment": {
            "center": center_summary,
            "size_pixels": tuple(args.align_size),
            "max_shift_arcsec": args.max_shift_arcsec,
            "coarse_step_pix": args.coarse_step_pix,
            "min_valid_pixels": args.min_valid_pixels,
            "mask_reference_zero": args.align_mask_reference_zero,
            "reference_zero_atol": args.align_reference_zero_atol,
            "result": compact_alignment_summary(alignment),
        },
        "scaling": {
            **asdict(patch.scale_fit),
            "enabled": args.scaling,
            "requested_scale_mode": args.scale_mode,
            "min_scale_pixels": args.min_scale_pixels,
            "manual_scale_factor_argument": args.scale_factor,
            "patch_pixels": int(np.count_nonzero(patch.patch_mask)),
            "mask_reference_zero": args.scale_mask_reference_zero,
            "reference_zero_atol": args.scale_reference_zero_atol,
            "reference_zero_masked_pixels": int(patch.n_reference_zero_masked),
        },
        "profiles": {
            "major_axis_pa_deg": args.profile_pa_deg,
            "profile_half_length_arcsec": args.profile_half_length_arcsec,
            "profile_step_pix": args.profile_step_pix,
            "profile_width_pix": args.profile_width_pix,
            "diagnostic_pad_pixels": args.diagnostic_pad_pixels,
        },
    }
    write_json(summary_path, summary, overwrite=args.overwrite)

    print("Done.")
    print(f"{active_filter_name} mosaic: {f200_mosaic_path}")
    print(f"Raw IFU {active_filter_name}: {ifu_raw_path}")
    print(f"Aligned IFU {active_filter_name}: {ifu_aligned_path}")
    print(f"Patched mosaic: {patched_path}")
    print(f"Run summary: {summary_path}")
    print(f"Scale pixels: {scale_pixels_path}")
    print(f"Axis profiles: {profile_table_path}")
    if not args.no_diagnostic_fits:
        print(f"Diagnostic cutout: {diagnostic_fits_path}")
    if not args.no_checkplot:
        print(f"Scaling checkplot: {checkplot_path}")
    print(
        "Alignment: "
        f"dx={alignment['dx_pix']:.3f} pix, dy={alignment['dy_pix']:.3f} pix, "
        f"dRA={alignment['dRA_arcsec']:.4f} arcsec, dDec={alignment['dDec_arcsec']:.4f} arcsec"
    )
    print(f"Patch IFU scale factor: {patch.scale_factor:.8g} from {patch.n_scale_pixels} overlap pixels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
