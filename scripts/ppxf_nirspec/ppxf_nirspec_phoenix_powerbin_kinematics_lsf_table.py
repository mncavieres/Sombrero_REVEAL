#!/usr/bin/env python3
"""
Fit NIRSpec stellar kinematics with PHOENIX templates, PowerBin bins, and a wavelength-dependent LSF.

This combines the PowerBin spatial binning runner with the PHOENIX LSF-table
template preparation. It is intended for high-S/N binned stellar kinematics on
the AGN-subtracted NIRSpec cube.
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
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_phoenix_powerbin_lsf"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import ppxf_nirspec_phoenix_kinematics as base
import ppxf_nirspec_phoenix_kinematics_lsf_table as lsf
import ppxf_nirspec_phoenix_powerbin_kinematics as powerbin_base


DEFAULT_OUTPUT_DIR = base.ROOT / "Data/ppxf_nirspec/agn_sub_powerbin_sn120_wavelength_lsf"
FIT_WORKER_STATE: dict[str, object] = {}


def parse_args() -> tuple[lsf.TableConfig, float, int, float, bool]:
    parser = argparse.ArgumentParser(
        description="Fit PHOENIX pPXF kinematics on PowerBin-binned NIRSpec spectra with wavelength-dependent LSF.",
    )
    parser.add_argument("--cube-path", type=Path, default=base.DEFAULT_CUBE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phoenix-dir", type=Path, default=base.DEFAULT_PHOENIX_DIR)
    parser.add_argument("--phoenix-wave-path", type=Path, default=base.DEFAULT_PHOENIX_WAVE)
    parser.add_argument("--template-list", type=str, default=None)
    parser.add_argument("--lsf-table-path", type=Path, default=lsf.DEFAULT_LSF_TABLE_PATH)
    parser.add_argument("--lsf-wave-column", type=str, default="WAVELENGTH")
    parser.add_argument("--lsf-r-column", type=str, default="R")
    parser.add_argument("--target-sn", type=float, default=120.0)
    parser.add_argument("--redshift", type=float, default=0.003633)
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
    parser.add_argument(
        "--check-plot-radius-arcsec",
        type=float,
        default=0.5,
        help="Radius used to select the highest-sigma central check plot.",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Ignore existing completed products in the output directory and run the full fit again.",
    )
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help="Alias for --force-refit; rerun the full fit and overwrite products in the output directory.",
    )
    parser.add_argument("--max-spaxels", type=int, default=None)
    parser.add_argument("--template-velocity-margin-kms", type=float, default=1500.0)
    args = parser.parse_args()

    cfg = lsf.TableConfig(
        cube_path=args.cube_path.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        phoenix_dir=args.phoenix_dir.expanduser().resolve(),
        phoenix_wave_path=args.phoenix_wave_path.expanduser().resolve(),
        template_list_path=base.parse_optional_path(args.template_list),
        lsf_table_path=args.lsf_table_path.expanduser().resolve(),
        lsf_wave_column=str(args.lsf_wave_column),
        lsf_r_column=str(args.lsf_r_column),
        redshift=float(args.redshift),
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
    )
    return (
        cfg,
        float(args.target_sn),
        max(1, int(args.n_processes)),
        float(args.check_plot_radius_arcsec),
        bool(args.force_refit or args.overwrite_run),
    )


def init_fit_worker(
    templates: np.ndarray,
    spectra_log: np.ndarray,
    valid_frac_log: np.ndarray,
    fit_mask_log: np.ndarray,
    velscale: float,
    lam_gal_ang: np.ndarray,
    lam_temp: np.ndarray,
    cfg: lsf.TableConfig,
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
    galaxy = np.nanmean(spectra_log[:, idx], axis=1)
    valid_frac = np.nanmean(valid_frac_log[:, idx], axis=1)
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


def output_paths(cfg: lsf.TableConfig, target_sn: float) -> dict[str, Path]:
    base_path = cfg.output_dir / f"{cfg.cube_path.stem}_phoenix_powerbin_lsf_sn{int(round(target_sn))}_kinematics"
    return {
        "csv": base_path.with_suffix(".csv"),
        "all_csv": base_path.with_name(base_path.name + "_all").with_suffix(".csv"),
        "fits": base_path.with_suffix(".fits"),
        "npz": base_path.with_suffix(".npz"),
        "json": cfg.output_dir / "phoenix_powerbin_lsf_run_config.json",
        "summary": cfg.output_dir / "phoenix_powerbin_lsf_run_summary.txt",
        "manifest": cfg.output_dir / "selected_phoenix_templates.csv",
    }


def completed_run_exists(paths: dict[str, Path]) -> bool:
    return paths["all_csv"].is_file() and paths["npz"].is_file()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_existing_rows(path: Path) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for raw in reader:
            if not raw.get("BIN_ID"):
                continue
            bid = int(raw["BIN_ID"])
            row: dict[str, object] = dict(raw)
            for key in (
                "X",
                "Y",
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
            ):
                row[key] = parse_float(row.get(key))
            for key in ("BIN_ID", "NSPAX"):
                row[key] = int(float(row[key]))
            row["GOODFIT"] = parse_bool(row.get("GOODFIT"))
            rows[bid] = row
    return rows


def save_fits_lsf_powerbin(
    path: Path,
    cfg: lsf.TableConfig,
    target_sn: float,
    cube_data: base.CubeData,
    library: lsf.TableTemplateLibrary,
    table_rows: list[dict[str, object]],
    maps: dict[str, np.ndarray],
) -> None:
    hdr = fits.Header()
    hdr["OBJECT"] = cfg.cube_path.stem[:68]
    hdr["BINNING"] = "POWERBIN"
    hdr["TARSN"] = float(target_sn)
    hdr["LSFMODE"] = "TABLE"
    hdr["LSFSRC"] = str(cfg.lsf_table_path)[:68]
    hdr["REDSHFT"] = float(cfg.redshift)
    hdr["RPOWMIN"] = float(library.resolving_power_min)
    hdr["RPOWER"] = float(library.resolving_power_med)
    hdr["RPOWMAX"] = float(library.resolving_power_max)
    hdr["RTEMPL"] = float(cfg.template_resolving_power)
    hdr["SIGINSMN"] = (float(library.sigma_inst_min_kms), "km/s")
    hdr["SIGINST"] = (float(library.sigma_inst_kms), "km/s")
    hdr["SIGINSMX"] = (float(library.sigma_inst_max_kms), "km/s")
    hdr["SIGTEMP"] = (float(library.sigma_template_kms), "km/s")
    hdr["SIGCNVMN"] = (float(library.sigma_conv_min_kms), "km/s")
    hdr["SIGCONV"] = (float(library.sigma_conv_kms), "km/s")
    hdr["SIGCNVMX"] = (float(library.sigma_conv_max_kms), "km/s")
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

    lsf_cols = [
        fits.Column(name="LAMBDA_OBS_UM", format="D", array=library.lsf_lam_obs_um),
        fits.Column(name="R", format="D", array=library.lsf_resolving_power),
    ]
    lsf_hdu = fits.BinTableHDU.from_columns(lsf_cols, name="LSF_ON_TEMPLATE")

    hdus = [
        fits.PrimaryHDU(header=hdr),
        kin_hdu,
        tmpl_hdu,
        lsf_hdu,
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


def write_check_fit_plot(
    outdir: Path,
    cube_data: base.CubeData,
    bin_id: int,
    idx: np.ndarray,
    result: base.FitResult,
    bin_sn: np.ndarray,
    filename: str,
    label: str,
) -> Path:
    galaxy = np.nanmean(cube_data.spectra_log[:, idx], axis=1)
    outpath = outdir / filename
    base.plot_fit(
        outpath,
        cube_data.lam_gal_ang,
        galaxy,
        result,
        title=(
            f"{label} | PowerBin {bin_id} | N={idx.size}, "
            f"S/N target={bin_sn[bin_id]:.1f}, sigma={result.sigma:.1f} km/s"
        ),
    )
    return outpath


def write_targeted_check_plots(
    outdir: Path,
    cube_data: base.CubeData,
    bin_num: np.ndarray,
    bin_sn: np.ndarray,
    idx_by_bin: dict[int, np.ndarray],
    result_by_bin: dict[int, base.FitResult],
    central_radius_arcsec: float,
) -> tuple[Path | None, Path | None, int | None, int | None]:
    radius = np.hypot(cube_data.x, cube_data.y)
    central_spaxel = int(np.nanargmin(radius))
    central_bin = int(bin_num[central_spaxel])
    central_path = write_check_fit_plot(
        outdir,
        cube_data,
        central_bin,
        idx_by_bin[central_bin],
        result_by_bin[central_bin],
        bin_sn,
        "phoenix_powerbin_lsf_check_central_spaxel_fit.png",
        "Central spaxel fit",
    )

    in_central_radius = np.flatnonzero(radius <= central_radius_arcsec)
    best_bin: int | None = None
    best_sigma = -np.inf
    for j in in_central_radius:
        bid = int(bin_num[j])
        result = result_by_bin.get(bid)
        if result is None or not result.ok or not np.isfinite(result.sigma):
            continue
        if result.sigma > best_sigma:
            best_sigma = float(result.sigma)
            best_bin = bid

    high_sigma_path: Path | None = None
    if best_bin is not None:
        radius_tag = f"{central_radius_arcsec:.2f}".replace(".", "p")
        high_sigma_path = write_check_fit_plot(
            outdir,
            cube_data,
            best_bin,
            idx_by_bin[best_bin],
            result_by_bin[best_bin],
            bin_sn,
            f"phoenix_powerbin_lsf_check_highest_sigma_within_{radius_tag}arcsec_fit.png",
            f"Highest sigma fit within r<={central_radius_arcsec:.2f} arcsec",
        )
    else:
        print(
            "WARNING: could not make central high-sigma check plot; "
            f"no good fitted bin within r <= {central_radius_arcsec:.2f} arcsec"
        )

    return central_path, high_sigma_path, central_bin, best_bin


def select_existing_check_bins(
    cube_data: base.CubeData,
    bin_num: np.ndarray,
    row_by_bin: dict[int, dict[str, object]],
    central_radius_arcsec: float,
) -> tuple[int, int | None]:
    radius = np.hypot(cube_data.x, cube_data.y)
    central_spaxel = int(np.nanargmin(radius))
    central_bin = int(bin_num[central_spaxel])

    best_bin: int | None = None
    best_sigma = -np.inf
    for j in np.flatnonzero(radius <= central_radius_arcsec):
        bid = int(bin_num[j])
        row = row_by_bin.get(bid)
        if row is None:
            continue
        sigma = parse_float(row.get("sigma"))
        if not np.isfinite(sigma):
            continue
        if not parse_bool(row.get("GOODFIT")):
            continue
        if sigma > best_sigma:
            best_sigma = float(sigma)
            best_bin = bid

    if best_bin is None:
        for j in np.flatnonzero(radius <= central_radius_arcsec):
            bid = int(bin_num[j])
            row = row_by_bin.get(bid)
            if row is None:
                continue
            sigma = parse_float(row.get("sigma"))
            if np.isfinite(sigma) and sigma > best_sigma:
                best_sigma = float(sigma)
                best_bin = bid
    return central_bin, best_bin


def refit_existing_bin_for_plot(
    cfg: lsf.TableConfig,
    cube_data: base.CubeData,
    library: lsf.TableTemplateLibrary,
    bin_id: int,
    idx: np.ndarray,
    existing_row: dict[str, object] | None,
) -> base.FitResult:
    galaxy = np.nanmean(cube_data.spectra_log[:, idx], axis=1)
    valid_frac = np.nanmean(cube_data.valid_frac_log[:, idx], axis=1)
    fit_mask = cube_data.fit_mask_log & (valid_frac >= cfg.min_log_pixel_fraction)
    start_v = parse_float(existing_row.get("V_REL_KMS")) if existing_row else np.nan
    start_sigma = parse_float(existing_row.get("sigma")) if existing_row else np.nan
    if not np.isfinite(start_v):
        start_v = 0.0
    if not np.isfinite(start_sigma):
        start_sigma = cfg.start_sigma
    return base.ppxf_fit_spectrum(
        library.templates,
        galaxy,
        cube_data.velscale,
        [float(start_v), float(start_sigma)],
        fit_mask,
        cube_data.lam_gal_ang,
        library.lam_temp,
        cfg,
    )


def append_or_update_existing_summary(
    summary_path: Path,
    central_check_path: Path | None,
    high_sigma_check_path: Path | None,
    central_bin: int | None,
    high_sigma_bin: int | None,
    central_radius_arcsec: float,
) -> None:
    lines = [
        "",
        "Check plots generated from existing run",
        f"Check plot central-radius cut: {central_radius_arcsec:.2f} arcsec",
        f"Central-spaxel check bin: {central_bin}",
        f"Central high-sigma check bin: {high_sigma_bin}",
        f"Central-spaxel check plot: {central_check_path}",
        f"Central high-sigma check plot: {high_sigma_check_path}",
    ]
    existing = summary_path.read_text() if summary_path.is_file() else ""
    marker = "Check plots generated from existing run"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n"
    summary_path.write_text(existing.rstrip() + "\n" + "\n".join(lines).lstrip() + "\n")


def generate_checkplots_from_existing_run(
    cfg: lsf.TableConfig,
    target_sn: float,
    check_plot_radius_arcsec: float,
    paths: dict[str, Path],
) -> bool:
    if not completed_run_exists(paths):
        return False

    print(f"Existing completed run detected in: {cfg.output_dir}")
    print("Skipping full PowerBin pPXF loop and regenerating targeted check plots only.")

    cube_data = base.read_cube_data(cfg)
    with np.load(paths["npz"], allow_pickle=True) as data:
        bin_num = np.asarray(data["bin_num"], dtype=int)
        bin_sn = np.asarray(data["bin_sn"], dtype=float)

    if bin_num.size != cube_data.spectra_log.shape[1]:
        raise ValueError(
            "Existing bin assignment does not match the current cube/spaxel selection: "
            f"bin_num has {bin_num.size} entries, cube selection has {cube_data.spectra_log.shape[1]}. "
            "Use the same selection arguments as the original run or pass --force-refit."
        )

    row_by_bin = read_existing_rows(paths["all_csv"])
    idx_by_bin = {int(bid): np.flatnonzero(bin_num == bid) for bid in np.unique(bin_num)}
    central_bin, high_sigma_bin = select_existing_check_bins(
        cube_data,
        bin_num,
        row_by_bin,
        check_plot_radius_arcsec,
    )
    target_bins = [central_bin]
    if high_sigma_bin is not None and high_sigma_bin not in target_bins:
        target_bins.append(high_sigma_bin)

    print(f"Loading PHOENIX templates from: {cfg.phoenix_dir}")
    print(f"Using wavelength-dependent LSF table: {cfg.lsf_table_path}")
    library = lsf.load_phoenix_templates_with_lsf_table(cfg, cube_data)

    result_by_bin: dict[int, base.FitResult] = {}
    for bid in tqdm(target_bins, desc="Refitting check-plot bins"):
        result_by_bin[bid] = refit_existing_bin_for_plot(
            cfg,
            cube_data,
            library,
            bid,
            idx_by_bin[bid],
            row_by_bin.get(bid),
        )

    central_check_path = write_check_fit_plot(
        cfg.output_dir,
        cube_data,
        central_bin,
        idx_by_bin[central_bin],
        result_by_bin[central_bin],
        bin_sn,
        "phoenix_powerbin_lsf_check_central_spaxel_fit.png",
        "Central spaxel fit",
    )

    high_sigma_check_path: Path | None = None
    if high_sigma_bin is not None:
        radius_tag = f"{check_plot_radius_arcsec:.2f}".replace(".", "p")
        high_sigma_check_path = write_check_fit_plot(
            cfg.output_dir,
            cube_data,
            high_sigma_bin,
            idx_by_bin[high_sigma_bin],
            result_by_bin[high_sigma_bin],
            bin_sn,
            f"phoenix_powerbin_lsf_check_highest_sigma_within_{radius_tag}arcsec_fit.png",
            f"Highest sigma fit within r<={check_plot_radius_arcsec:.2f} arcsec",
        )

    if paths["json"].is_file():
        config = json.loads(paths["json"].read_text())
    else:
        config = {}
    config.update(
        {
            "check_plot_radius_arcsec": check_plot_radius_arcsec,
            "central_check_bin": central_bin,
            "high_sigma_check_bin": high_sigma_bin,
            "central_check_plot": str(central_check_path),
            "high_sigma_check_plot": str(high_sigma_check_path) if high_sigma_check_path is not None else None,
            "check_plots_generated_from_existing_run": True,
        }
    )
    paths["json"].write_text(json.dumps(config, indent=2) + "\n")
    append_or_update_existing_summary(
        paths["summary"],
        central_check_path,
        high_sigma_check_path,
        central_bin,
        high_sigma_bin,
        check_plot_radius_arcsec,
    )
    print(f"Saved central check plot    : {central_check_path}")
    print(f"Saved high-sigma check plot : {high_sigma_check_path}")
    print(f"Updated JSON                : {paths['json']}")
    print(f"Updated summary             : {paths['summary']}")
    return True


def main() -> None:
    cfg, target_sn, n_processes, check_plot_radius_arcsec, force_refit = parse_args()
    outdir = base.ensure_dir(cfg.output_dir)
    paths = output_paths(cfg, target_sn)

    if not force_refit and generate_checkplots_from_existing_run(
        cfg,
        target_sn,
        check_plot_radius_arcsec,
        paths,
    ):
        return

    print(f"Reading cube: {cfg.cube_path}")
    cube_data = base.read_cube_data(cfg)
    print(
        f"Log-rebinned cube: {cube_data.spectra_log.shape[0]} pixels, "
        f"{cube_data.spectra_log.shape[1]} spaxels, velscale={cube_data.velscale:.3f} km/s"
    )

    print(f"Building PowerBin bins with target S/N={target_sn:.1f}")
    powbin, bin_num, bin_sn = powerbin_base.make_power_bins(cube_data, target_sn)
    unique_bins = np.unique(bin_num)
    print(f"PowerBin produced {unique_bins.size} bins from {bin_num.size} valid spaxels")
    try:
        powbin.plot(ylabel="S/N")
        plt.savefig(outdir / "phoenix_powerbin_lsf_sn_bins.png", dpi=180, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:
        print(f"WARNING: could not save PowerBin diagnostic plot: {exc}")

    print(f"Loading PHOENIX templates from: {cfg.phoenix_dir}")
    print(f"Using wavelength-dependent LSF table: {cfg.lsf_table_path}")
    library = lsf.load_phoenix_templates_with_lsf_table(cfg, cube_data)
    print(
        f"Using {len(library.meta)} templates; R(lambda)={library.resolving_power_min:.1f}-"
        f"{library.resolving_power_max:.1f}, median={library.resolving_power_med:.1f}; "
        f"sigma_conv median={library.sigma_conv_kms:.2f} km/s"
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
        outdir / "phoenix_powerbin_lsf_global_fit.png",
        cube_data.lam_gal_ang,
        global_spec,
        global_result,
        title=(
            f"Global PHOENIX PowerBin LSF pPXF fit | "
            f"V={global_result.sol_rel[0]:.1f} km/s, "
            f"sigma={global_result.sigma:.1f} km/s"
            if global_result.sol_rel is not None
            else f"Global PHOENIX PowerBin LSF pPXF fit failed: {global_result.message}"
        ),
    )
    start_sigma = global_result.sigma if np.isfinite(global_result.sigma) else cfg.start_sigma

    maps = powerbin_base.make_powerbin_maps(cube_data.map_shape)
    rows: list[dict[str, object]] = []
    preview_done = 0
    preview_ids = set(unique_bins[: max(cfg.n_plot_spaxels * 4, 20)])
    bin_tasks = [(int(bid), np.flatnonzero(bin_num == bid)) for bid in unique_bins]
    idx_by_bin = {int(bid): idx for bid, idx in bin_tasks}
    result_by_bin: dict[int, base.FitResult] = {}

    if n_processes > 1:
        print(
            f"Fitting PowerBin LSF spectra with {n_processes} worker processes "
            f"(fixed start sigma={start_sigma:.2f} km/s)"
        )
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
            fit_outputs = list(
                tqdm(
                    pool.map(fit_powerbin_worker, bin_tasks, chunksize=1),
                    total=len(bin_tasks),
                    desc="Fitting PowerBin LSF spectra",
                )
            )
    else:
        fit_outputs = []
        for bid, idx in tqdm(bin_tasks, desc="Fitting PowerBin LSF spectra"):
            galaxy = np.nanmean(cube_data.spectra_log[:, idx], axis=1)
            valid_frac = np.nanmean(cube_data.valid_frac_log[:, idx], axis=1)
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
            fit_outputs.append((int(bid), idx, result))

    for bid, idx, result in fit_outputs:
        result_by_bin[int(bid)] = result
        galaxy = np.nanmean(cube_data.spectra_log[:, idx], axis=1)

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
            base.plot_fit(
                outdir / f"phoenix_powerbin_lsf_fit_bin{int(bid):04d}.png",
                cube_data.lam_gal_ang,
                galaxy,
                result,
                title=f"PowerBin LSF {int(bid)} | N={idx.size}, S/N target={bin_sn[bid]:.1f}",
            )
            preview_done += 1

    central_check_path, high_sigma_check_path, central_check_bin, high_sigma_check_bin = write_targeted_check_plots(
        outdir,
        cube_data,
        bin_num,
        bin_sn,
        idx_by_bin,
        result_by_bin,
        check_plot_radius_arcsec,
    )
    powerbin_base.plot_powerbin_maps(outdir, cube_data, maps)

    base_path = outdir / f"{cfg.cube_path.stem}_phoenix_powerbin_lsf_sn{int(round(target_sn))}_kinematics"
    csv_path = base_path.with_suffix(".csv")
    all_csv_path = base_path.with_name(base_path.name + "_all").with_suffix(".csv")
    fits_path = base_path.with_suffix(".fits")
    npz_path = base_path.with_suffix(".npz")
    json_path = outdir / "phoenix_powerbin_lsf_run_config.json"
    summary_path = outdir / "phoenix_powerbin_lsf_run_summary.txt"
    manifest_path = outdir / "selected_phoenix_templates.csv"

    good_rows = [row for row in rows if row["GOODFIT"]]
    write_csv(csv_path, good_rows)
    write_csv(all_csv_path, rows)
    base.write_template_manifest(manifest_path, library)
    save_fits_lsf_powerbin(fits_path, cfg, target_sn, cube_data, library, rows, maps)
    np.savez_compressed(
        npz_path,
        lam_gal_ang=cube_data.lam_gal_ang,
        lam_temp=library.lam_temp,
        lsf_lam_obs_um=library.lsf_lam_obs_um,
        lsf_resolving_power=library.lsf_resolving_power,
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

    config = lsf.config_to_json(cfg, library, cube_data)
    config.update(
        {
            "binning": "PowerBin",
            "target_sn": target_sn,
            "n_processes": n_processes,
            "check_plot_radius_arcsec": check_plot_radius_arcsec,
            "central_check_bin": central_check_bin,
            "high_sigma_check_bin": high_sigma_check_bin,
            "central_check_plot": str(central_check_path) if central_check_path is not None else None,
            "high_sigma_check_plot": str(high_sigma_check_path) if high_sigma_check_path is not None else None,
            "n_bins": len(rows),
            "n_good_bins": len(good_rows),
        }
    )
    json_path.write_text(json.dumps(config, indent=2) + "\n")

    med_sn = float(np.nanmedian([row["SN"] for row in good_rows])) if good_rows else np.nan
    med_target_sn = float(np.nanmedian([row["SN_TARGET"] for row in rows])) if rows else np.nan
    med_sigma = float(np.nanmedian([row["sigma"] for row in good_rows])) if good_rows else np.nan
    summary_lines = [
        "PHOENIX pPXF NIRSpec stellar kinematics with PowerBin spatial binning and wavelength-dependent LSF",
        f"Cube: {cfg.cube_path}",
        f"Output dir: {cfg.output_dir}",
        f"Target PowerBin S/N: {target_sn:.1f}",
        f"Worker processes: {n_processes}",
        f"Check plot central-radius cut: {check_plot_radius_arcsec:.2f} arcsec",
        f"PowerBin bins: {len(rows)} from {bin_num.size} valid spaxels",
        f"Median PowerBin input S/N: {med_target_sn:.2f}",
        f"PHOENIX dir: {cfg.phoenix_dir}",
        f"PHOENIX wavelength file: {cfg.phoenix_wave_path}",
        f"LSF table: {cfg.lsf_table_path}",
        f"Templates used: {len(library.meta)}",
        f"Resolving power range: {library.resolving_power_min:.1f} to {library.resolving_power_max:.1f}",
        f"Resolving power median: {library.resolving_power_med:.1f}",
        f"Template native resolving power: {cfg.template_resolving_power:.1f}",
        f"Template convolution sigma range: {library.sigma_conv_min_kms:.3f} to {library.sigma_conv_max_kms:.3f} km/s",
        f"Template convolution sigma median: {library.sigma_conv_kms:.3f} km/s ({library.sigma_conv_pix:.3f} pix)",
        f"pPXF setup: moments={cfg.moments}, degree={cfg.degree}, mdegree={cfg.mdegree}, bias={cfg.bias}, regul=0",
        f"Fit windows rest um: {cfg.fit_windows_rest_um}",
        f"Masked windows rest um: {cfg.mask_windows_rest_um}",
        f"Good-fit bins: {len(good_rows)}",
        f"Median good-fit pPXF S/N: {med_sn:.2f}",
        f"Median good-fit sigma: {med_sigma:.2f} km/s",
        f"Good-fit CSV: {csv_path}",
        f"All-fits CSV: {all_csv_path}",
        f"FITS: {fits_path}",
        f"NPZ: {npz_path}",
        f"Template manifest: {manifest_path}",
        f"Central-spaxel check plot: {central_check_path}",
        f"Central high-sigma check plot: {high_sigma_check_path}",
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
