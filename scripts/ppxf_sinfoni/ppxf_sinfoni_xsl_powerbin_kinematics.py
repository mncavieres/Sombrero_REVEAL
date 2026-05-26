#!/usr/bin/env python3
"""
Fit SINFONI stellar kinematics with pPXF, XSL SPS templates, and PowerBin bins.

This is a SINFONI/XSL analogue of the PHOENIX PowerBin runner. It keeps the
same fitting machinery used for the NIRSpec/SINFONI PHOENIX tests, but replaces
the PHOENIX template grid with the pPXF XSL SPS library, which covers the K band.
The SINFONI LSF is represented as a fixed resolving power R, i.e. FWHM=lambda/R.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
import json
import os
import sys
import tempfile
from importlib import resources
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_sinfoni_xsl"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from ppxf import sps_util
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
NIRSPEC_DIR = THIS_DIR.parent / "ppxf_nirspec"
for path in (THIS_DIR, NIRSPEC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppxf_nirspec_phoenix_kinematics as base
import ppxf_nirspec_phoenix_powerbin_kinematics as powerbin_base


C = 299792.458
FWHM_PER_SIGMA = 2.35482004503
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUBE_PATH = ROOT / "Data/IFU/sinfoni/M104_coadded_SINFONI(1).fits"
DEFAULT_OUTPUT_DIR = ROOT / "Data/ppxf_sinfoni/M104_coadded_SINFONI1_xsl_powerbin_sn400_fixed_r4000"

FIT_WORKER_STATE: dict[str, object] = {}


@dataclass(frozen=True)
class XslConfig:
    cube_path: Path
    output_dir: Path
    xsl_template_path: Path
    mge_solution_path: Path
    mge_luminosity_path: Path
    mge_contour_rotation_deg: float
    redshift: float
    resolving_power: float
    fit_windows_rest_um: tuple[tuple[float, float], ...]
    mask_windows_rest_um: tuple[tuple[float, float], ...]
    xsl_norm_range_rest_um: tuple[float, float] | None
    xsl_age_range_gyr: tuple[float, float] | None
    xsl_metal_range: tuple[float, float] | None
    degree: int
    mdegree: int
    moments: int
    bias: float
    start_sigma: float
    max_abs_velocity: float
    min_sigma: float
    max_sigma: float
    wave_crpix_mode: str
    min_wave_finite_frac: float
    min_spaxel_finite_frac: float
    min_log_pixel_fraction: float
    min_goodpixels: int
    top_template_frac: float
    csv_min_sn: float
    n_plot_bins: int
    max_spaxels: int | None
    template_velocity_margin_kms: float


@dataclass
class XslLibrary:
    templates: np.ndarray
    lam_temp: np.ndarray
    age_grid: np.ndarray
    metal_grid: np.ndarray
    reg_dim: tuple[int, int]
    template_path: Path
    template_count: int
    xsl_native_fwhm_min_ang: float
    xsl_native_fwhm_med_ang: float
    xsl_native_fwhm_max_ang: float
    data_fwhm_min_ang: float
    data_fwhm_med_ang: float
    data_fwhm_max_ang: float
    sigma_inst_kms: float
    lam_min_ang: float
    lam_max_ang: float


def default_xsl_path() -> Path:
    return Path(resources.files("ppxf") / "sps_models" / "spectra_xsl_9.0.npz")


def parse_range(text: str | None, units: str) -> tuple[float, float] | None:
    if text is None:
        return None
    text = text.strip()
    if not text or text.lower() == "none":
        return None
    parts = text.replace(",", "-").split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected {units} range like lo-hi, got {text!r}")
    lo, hi = (float(parts[0]), float(parts[1]))
    if not lo < hi:
        raise ValueError(f"Expected increasing {units} range, got {text!r}")
    return lo, hi


def parse_args() -> tuple[XslConfig, float, int]:
    parser = argparse.ArgumentParser(
        description="Fit SINFONI stellar kinematics with pPXF, PowerBin bins, and XSL templates.",
    )
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xsl-template-path", type=Path, default=default_xsl_path())
    parser.add_argument("--mge-solution-path", type=Path, default=powerbin_base.DEFAULT_MGE_SOLUTION_PATH)
    parser.add_argument("--mge-luminosity-path", type=Path, default=powerbin_base.DEFAULT_MGE_LUMINOSITY_PATH)
    parser.add_argument(
        "--mge-contour-rotation-deg",
        type=float,
        default=powerbin_base.DEFAULT_MGE_CONTOUR_ROTATION_DEG,
        help=(
            "Rotation applied to photometric MGE coordinates before overlaying contours "
            "on the native SINFONI/IFU frame. Default is 0 deg."
        ),
    )
    parser.add_argument("--target-sn", type=float, default=400.0)
    parser.add_argument("--redshift", type=float, default=0.003633)
    parser.add_argument("--resolving-power", type=float, default=4000.0)
    parser.add_argument("--fit-windows-rest-um", type=str, default="1.9-2.15")
    parser.add_argument("--mask-windows-rest-um", type=str, default="2.0077-2.0157")
    parser.add_argument(
        "--xsl-norm-range-rest-um",
        type=str,
        default="1.9-2.15",
        help="Template normalization range in rest microns. Use 'none' for global normalization.",
    )
    parser.add_argument("--xsl-age-range-gyr", type=str, default=None)
    parser.add_argument("--xsl-metal-range", type=str, default=None)
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--mdegree", type=int, default=6)
    parser.add_argument("--moments", type=int, default=4)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--start-sigma", type=float, default=220.0)
    parser.add_argument("--max-abs-velocity", type=float, default=700.0)
    parser.add_argument("--min-sigma", type=float, default=10.0)
    parser.add_argument("--max-sigma", type=float, default=700.0)
    parser.add_argument(
        "--wave-crpix-mode",
        choices=("fits", "first_pixel"),
        default="first_pixel",
        help=(
            "How to interpret the spectral WCS. SINFONI coadds here use CRVAL3 as the "
            "first wavelength sample, so the default is 'first_pixel'."
        ),
    )
    parser.add_argument("--min-wave-finite-frac", type=float, default=0.35)
    parser.add_argument("--min-spaxel-finite-frac", type=float, default=0.60)
    parser.add_argument("--min-log-pixel-fraction", type=float, default=0.70)
    parser.add_argument("--min-goodpixels", type=int, default=180)
    parser.add_argument("--top-template-frac", type=float, default=0.10)
    parser.add_argument("--csv-min-sn", type=float, default=10.0)
    parser.add_argument("--n-plot-bins", type=int, default=8)
    parser.add_argument("--n-processes", type=int, default=6)
    parser.add_argument("--max-spaxels", type=int, default=None)
    parser.add_argument("--template-velocity-margin-kms", type=float, default=1500.0)
    args = parser.parse_args()

    cfg = XslConfig(
        cube_path=args.cube_path.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        xsl_template_path=args.xsl_template_path.expanduser().resolve(),
        mge_solution_path=args.mge_solution_path.expanduser().resolve(),
        mge_luminosity_path=args.mge_luminosity_path.expanduser().resolve(),
        mge_contour_rotation_deg=float(args.mge_contour_rotation_deg),
        redshift=float(args.redshift),
        resolving_power=float(args.resolving_power),
        fit_windows_rest_um=base.parse_windows(args.fit_windows_rest_um),
        mask_windows_rest_um=base.parse_windows(args.mask_windows_rest_um),
        xsl_norm_range_rest_um=parse_range(args.xsl_norm_range_rest_um, "micron"),
        xsl_age_range_gyr=parse_range(args.xsl_age_range_gyr, "Gyr"),
        xsl_metal_range=parse_range(args.xsl_metal_range, "[M/H]"),
        degree=int(args.degree),
        mdegree=int(args.mdegree),
        moments=int(args.moments),
        bias=float(args.bias),
        start_sigma=float(args.start_sigma),
        max_abs_velocity=float(args.max_abs_velocity),
        min_sigma=float(args.min_sigma),
        max_sigma=float(args.max_sigma),
        wave_crpix_mode=str(args.wave_crpix_mode),
        min_wave_finite_frac=float(args.min_wave_finite_frac),
        min_spaxel_finite_frac=float(args.min_spaxel_finite_frac),
        min_log_pixel_fraction=float(args.min_log_pixel_fraction),
        min_goodpixels=int(args.min_goodpixels),
        top_template_frac=float(args.top_template_frac),
        csv_min_sn=float(args.csv_min_sn),
        n_plot_bins=int(args.n_plot_bins),
        max_spaxels=args.max_spaxels,
        template_velocity_margin_kms=float(args.template_velocity_margin_kms),
    )
    return cfg, float(args.target_sn), max(1, int(args.n_processes))


def to_base_config(cfg: XslConfig) -> base.Config:
    return base.Config(
        cube_path=cfg.cube_path,
        output_dir=cfg.output_dir,
        phoenix_dir=cfg.xsl_template_path.parent,
        phoenix_wave_path=cfg.xsl_template_path,
        template_list_path=None,
        redshift=cfg.redshift,
        resolving_power=cfg.resolving_power,
        template_resolving_power=np.nan,
        fit_windows_rest_um=cfg.fit_windows_rest_um,
        mask_windows_rest_um=cfg.mask_windows_rest_um,
        teff_min=np.nan,
        teff_max=np.nan,
        feh_min=np.nan,
        feh_max=np.nan,
        logg_min=np.nan,
        logg_max=np.nan,
        expected_template_count=0,
        strict_template_count=False,
        degree=cfg.degree,
        mdegree=cfg.mdegree,
        moments=cfg.moments,
        bias=cfg.bias,
        start_sigma=cfg.start_sigma,
        max_abs_velocity=cfg.max_abs_velocity,
        min_sigma=cfg.min_sigma,
        max_sigma=cfg.max_sigma,
        wave_crpix_mode=cfg.wave_crpix_mode,
        min_wave_finite_frac=cfg.min_wave_finite_frac,
        min_spaxel_finite_frac=cfg.min_spaxel_finite_frac,
        min_log_pixel_fraction=cfg.min_log_pixel_fraction,
        min_goodpixels=cfg.min_goodpixels,
        top_template_frac=cfg.top_template_frac,
        csv_min_sn=cfg.csv_min_sn,
        n_plot_spaxels=cfg.n_plot_bins,
        max_spaxels=cfg.max_spaxels,
        template_velocity_margin_kms=cfg.template_velocity_margin_kms,
    )


def tuple_um_to_ang(value: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    return (value[0] * 1e4, value[1] * 1e4)


def load_xsl_templates(cfg: XslConfig, fit_cfg: base.Config, cube_data: base.CubeData) -> XslLibrary:
    if not cfg.xsl_template_path.is_file():
        raise FileNotFoundError(
            f"XSL SPS template file not found: {cfg.xsl_template_path}. "
            "The pPXF package normally ships spectra_xsl_9.0.npz."
        )

    vel_pad = fit_cfg.max_abs_velocity + fit_cfg.template_velocity_margin_kms
    lam_min = float(cube_data.lam_gal_ang[0] * np.exp(-vel_pad / C))
    lam_max = float(cube_data.lam_gal_ang[-1] * np.exp(vel_pad / C))
    lam_range = (lam_min, lam_max)

    fwhm_gal = {
        "lam": cube_data.lam_gal_ang,
        "fwhm": cube_data.lam_gal_ang / cfg.resolving_power,
    }

    sps = sps_util.sps_lib(
        cfg.xsl_template_path,
        cube_data.velscale,
        fwhm_gal=fwhm_gal,
        age_range=cfg.xsl_age_range_gyr,
        lam_range=lam_range,
        metal_range=cfg.xsl_metal_range,
        norm_range=tuple_um_to_ang(cfg.xsl_norm_range_rest_um),
    )
    npix, *reg_dim = sps.templates.shape
    templates = np.asarray(sps.templates.reshape(npix, -1), dtype=float)
    templates = base.normalize_columns(templates)

    with np.load(cfg.xsl_template_path) as data:
        native_lam = np.asarray(data["lam"], dtype=float)
        native_fwhm = np.asarray(data["fwhm"], dtype=float)
    native_sel = (native_lam >= lam_min) & (native_lam <= lam_max)
    native_fwhm_sel = native_fwhm[native_sel]
    data_fwhm = cube_data.lam_gal_ang / cfg.resolving_power

    sigma_inst = C / cfg.resolving_power / FWHM_PER_SIGMA
    return XslLibrary(
        templates=templates,
        lam_temp=np.asarray(sps.lam_temp, dtype=float),
        age_grid=np.asarray(sps.age_grid, dtype=float),
        metal_grid=np.asarray(sps.metal_grid, dtype=float),
        reg_dim=(int(reg_dim[0]), int(reg_dim[1])),
        template_path=cfg.xsl_template_path,
        template_count=int(templates.shape[1]),
        xsl_native_fwhm_min_ang=float(np.nanmin(native_fwhm_sel)),
        xsl_native_fwhm_med_ang=float(np.nanmedian(native_fwhm_sel)),
        xsl_native_fwhm_max_ang=float(np.nanmax(native_fwhm_sel)),
        data_fwhm_min_ang=float(np.nanmin(data_fwhm)),
        data_fwhm_med_ang=float(np.nanmedian(data_fwhm)),
        data_fwhm_max_ang=float(np.nanmax(data_fwhm)),
        sigma_inst_kms=float(sigma_inst),
        lam_min_ang=float(sps.lam_temp[0]),
        lam_max_ang=float(sps.lam_temp[-1]),
    )


def init_fit_worker(
    templates: np.ndarray,
    spectra_log: np.ndarray,
    valid_frac_log: np.ndarray,
    fit_mask_log: np.ndarray,
    velscale: float,
    lam_gal_ang: np.ndarray,
    lam_temp: np.ndarray,
    fit_cfg: base.Config,
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
            "fit_cfg": fit_cfg,
            "start_sigma": float(start_sigma),
        }
    )


def fit_powerbin_worker(task: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, base.FitResult]:
    bid, idx = task
    spectra_log = FIT_WORKER_STATE["spectra_log"]
    valid_frac_log = FIT_WORKER_STATE["valid_frac_log"]
    fit_mask_log = FIT_WORKER_STATE["fit_mask_log"]
    fit_cfg = FIT_WORKER_STATE["fit_cfg"]
    galaxy, valid_frac = powerbin_base.stack_powerbin_spectrum(spectra_log, valid_frac_log, idx)
    fit_mask = fit_mask_log & (valid_frac >= fit_cfg.min_log_pixel_fraction)
    result = base.ppxf_fit_spectrum(
        FIT_WORKER_STATE["templates"],
        galaxy,
        FIT_WORKER_STATE["velscale"],
        [0.0, FIT_WORKER_STATE["start_sigma"]],
        fit_mask,
        FIT_WORKER_STATE["lam_gal_ang"],
        FIT_WORKER_STATE["lam_temp"],
        fit_cfg,
    )
    return int(bid), idx, result


def output_paths(cfg: XslConfig, target_sn: float) -> dict[str, Path]:
    base_path = cfg.output_dir / f"{cfg.cube_path.stem}_xsl_powerbin_sn{int(round(target_sn))}_kinematics"
    return {
        "csv": base_path.with_suffix(".csv"),
        "all_csv": base_path.with_name(base_path.name + "_all").with_suffix(".csv"),
        "fits": base_path.with_suffix(".fits"),
        "npz": base_path.with_suffix(".npz"),
        "json": cfg.output_dir / "xsl_powerbin_run_config.json",
        "summary": cfg.output_dir / "xsl_powerbin_run_summary.txt",
    }


def write_xsl_grid_manifest(path: Path, library: XslLibrary) -> None:
    rows = []
    flat_age = library.age_grid.ravel()
    flat_metal = library.metal_grid.ravel()
    for j, (age, metal) in enumerate(zip(flat_age, flat_metal)):
        rows.append({"TEMPLATE_ID": j, "AGE_GYR": float(age), "METAL": float(metal)})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["TEMPLATE_ID", "AGE_GYR", "METAL"])
        writer.writeheader()
        writer.writerows(rows)


def write_xsl_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
        "LOGAGE_YR",
        "METAL",
        "CHI2",
        "GOODPIX_FRAC",
        "GOODFIT",
        "MESSAGE",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def xsl_mean_population(weights: np.ndarray | None, library: XslLibrary) -> tuple[float, float]:
    if weights is None:
        return np.nan, np.nan
    weights = np.asarray(weights, dtype=float)
    if weights.size != library.template_count or not np.any(np.isfinite(weights)):
        return np.nan, np.nan
    weights = weights.reshape(library.reg_dim)
    denom = np.nansum(weights)
    if not np.isfinite(denom) or denom == 0:
        return np.nan, np.nan
    lg_age_grid = np.log10(library.age_grid) + 9.0
    mean_lg_age = np.nansum(weights * lg_age_grid) / denom
    mean_metal = np.nansum(weights * library.metal_grid) / denom
    return float(mean_lg_age), float(mean_metal)


def plot_xsl_powerbin_maps(outdir: Path, cube_data: base.CubeData, maps: dict[str, np.ndarray]) -> Path:
    outpath = powerbin_base.plot_powerbin_maps(outdir, cube_data, maps)
    xsl_outpath = outdir / "xsl_powerbin_kinematics_maps.png"
    if outpath != xsl_outpath:
        outpath.replace(xsl_outpath)
    return xsl_outpath


def save_fits_xsl(
    path: Path,
    cfg: XslConfig,
    fit_cfg: base.Config,
    target_sn: float,
    cube_data: base.CubeData,
    library: XslLibrary,
    rows: list[dict[str, object]],
    maps: dict[str, np.ndarray],
) -> None:
    hdr = fits.Header()
    hdr["OBJECT"] = cfg.cube_path.stem[:68]
    hdr["TPLLIB"] = "XSL"
    hdr["BINNING"] = "POWERBIN"
    hdr["TARSN"] = float(target_sn)
    hdr["REDSHFT"] = float(cfg.redshift)
    hdr["RPOWER"] = float(cfg.resolving_power)
    hdr["SIGINST"] = (float(library.sigma_inst_kms), "km/s")
    hdr["DEGREE"] = int(cfg.degree)
    hdr["MDEGREE"] = int(cfg.mdegree)
    hdr["MOMENTS"] = int(cfg.moments)
    hdr["BIAS"] = float(cfg.bias)
    hdr["NTMPL"] = int(library.template_count)
    hdr["NBINS"] = len(rows)
    hdr["VELSCAL"] = (float(cube_data.velscale), "km/s")
    hdr["LAMMIN"] = float(np.nanmin(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["LAMMAX"] = float(np.nanmax(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["PIXSIZE"] = (float(cube_data.pixsize_arcsec), "arcsec")
    hdr["XSLLAM0"] = float(library.lam_min_ang)
    hdr["XSLLAM1"] = float(library.lam_max_ang)
    hdr["XSLFWHM"] = (float(library.xsl_native_fwhm_med_ang), "median native template FWHM [A]")
    hdr["DATFWHM"] = (float(library.data_fwhm_med_ang), "median data FWHM [A]")

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
        "LOGAGE_YR",
        "METAL",
        "CHI2",
        "GOODPIX_FRAC",
        "GOODFIT",
    ]
    cols = []
    for key in table_keys:
        values = np.array([row[key] for row in rows])
        if values.dtype.kind == "b":
            fmt = "L"
        elif np.issubdtype(values.dtype, np.integer):
            fmt = "J"
        else:
            fmt = "D"
        cols.append(fits.Column(name=key.upper(), format=fmt, array=values))
    cols.append(fits.Column(name="MESSAGE", format="80A", array=np.array([str(row["MESSAGE"])[:80] for row in rows])))
    bin_hdu = fits.BinTableHDU.from_columns(cols, name="BIN_RESULTS")

    grid_cols = [
        fits.Column(name="TEMPLATE_ID", format="J", array=np.arange(library.template_count, dtype=np.int32)),
        fits.Column(name="AGE_GYR", format="D", array=library.age_grid.ravel()),
        fits.Column(name="METAL", format="D", array=library.metal_grid.ravel()),
    ]
    grid_hdu = fits.BinTableHDU.from_columns(grid_cols, name="XSL_GRID")

    hdus = [
        fits.PrimaryHDU(header=hdr),
        bin_hdu,
        grid_hdu,
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


def config_to_json(
    cfg: XslConfig,
    target_sn: float,
    n_processes: int,
    cube_data: base.CubeData,
    library: XslLibrary,
    rows: list[dict[str, object]],
    good_rows: list[dict[str, object]],
) -> dict[str, object]:
    data = asdict(cfg)
    for key, val in list(data.items()):
        if isinstance(val, Path):
            data[key] = str(val)
        elif isinstance(val, tuple):
            data[key] = list(val)
    data.update(
        {
            "template_library": "xsl",
            "target_sn": float(target_sn),
            "n_processes": int(n_processes),
            "n_templates": int(library.template_count),
            "age_grid_shape": list(library.reg_dim),
            "xsl_lam_min_ang": float(library.lam_min_ang),
            "xsl_lam_max_ang": float(library.lam_max_ang),
            "xsl_native_fwhm_min_ang": float(library.xsl_native_fwhm_min_ang),
            "xsl_native_fwhm_med_ang": float(library.xsl_native_fwhm_med_ang),
            "xsl_native_fwhm_max_ang": float(library.xsl_native_fwhm_max_ang),
            "data_fwhm_min_ang": float(library.data_fwhm_min_ang),
            "data_fwhm_med_ang": float(library.data_fwhm_med_ang),
            "data_fwhm_max_ang": float(library.data_fwhm_max_ang),
            "sigma_inst_kms": float(library.sigma_inst_kms),
            "velscale_kms": float(cube_data.velscale),
            "pixel_size_arcsec": float(cube_data.pixsize_arcsec),
            "wave_obs_min_um": float(np.nanmin(cube_data.wave_obs_um)),
            "wave_obs_max_um": float(np.nanmax(cube_data.wave_obs_um)),
            "wave_rest_min_um": float(np.nanmin(cube_data.wave_rest_um)),
            "wave_rest_max_um": float(np.nanmax(cube_data.wave_rest_um)),
            "n_log_pixels": int(cube_data.spectra_log.shape[0]),
            "n_fit_log_pixels": int(np.count_nonzero(cube_data.fit_mask_log)),
            "cube_shape": list(cube_data.cube_shape),
            "map_shape": list(cube_data.map_shape),
            "binning": "PowerBin",
            "n_bins": len(rows),
            "n_good_bins": len(good_rows),
        }
    )
    return data


def main() -> None:
    cfg, target_sn, n_processes = parse_args()
    fit_cfg = to_base_config(cfg)
    outdir = base.ensure_dir(cfg.output_dir)
    paths = output_paths(cfg, target_sn)

    print(f"Reading cube: {cfg.cube_path}")
    cube_data = base.read_cube_data(fit_cfg)
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
        plt.savefig(outdir / "xsl_powerbin_sn_bins.png", dpi=180, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:
        print(f"WARNING: could not save PowerBin diagnostic plot: {exc}")

    print(f"Loading XSL templates from: {cfg.xsl_template_path}")
    library = load_xsl_templates(cfg, fit_cfg, cube_data)
    print(
        f"Using {library.template_count} XSL templates; "
        f"native FWHM={library.xsl_native_fwhm_med_ang:.2f} A median, "
        f"data fixed R={cfg.resolving_power:.0f} "
        f"(FWHM={library.data_fwhm_med_ang:.2f} A median)"
    )

    n_global = max(1, int(np.ceil(fit_cfg.top_template_frac * cube_data.spectra_log.shape[1])))
    global_sel = np.argsort(cube_data.signal)[-n_global:]
    global_spec = np.nanmean(cube_data.spectra_log[:, global_sel], axis=1)
    global_valid_frac = np.nanmean(cube_data.valid_frac_log[:, global_sel], axis=1)
    global_mask = cube_data.fit_mask_log & (global_valid_frac >= fit_cfg.min_log_pixel_fraction)
    global_result = base.ppxf_fit_spectrum(
        library.templates,
        global_spec,
        cube_data.velscale,
        [0.0, fit_cfg.start_sigma],
        global_mask,
        cube_data.lam_gal_ang,
        library.lam_temp,
        fit_cfg,
    )
    base.plot_fit(
        outdir / "xsl_powerbin_global_fit.png",
        cube_data.lam_gal_ang,
        global_spec,
        global_result,
        title=(
            f"Global XSL PowerBin pPXF fit | V={global_result.sol_rel[0]:.1f} km/s, "
            f"sigma={global_result.sigma:.1f} km/s"
            if global_result.sol_rel is not None
            else f"Global XSL PowerBin pPXF fit failed: {global_result.message}"
        ),
    )
    start_sigma = global_result.sigma if np.isfinite(global_result.sigma) else fit_cfg.start_sigma

    maps = powerbin_base.make_powerbin_maps(cube_data.map_shape)
    rows: list[dict[str, object]] = []
    idx_by_bin: dict[int, np.ndarray] = {}
    result_by_bin: dict[int, base.FitResult] = {}
    preview_done = 0
    preview_ids = set(unique_bins[: max(fit_cfg.n_plot_spaxels * 4, 20)])
    bin_tasks = [(int(bid), np.flatnonzero(bin_num == bid)) for bid in unique_bins]

    def record_result(bid: int, idx: np.ndarray, result: base.FitResult) -> None:
        nonlocal preview_done
        idx_by_bin[int(bid)] = np.asarray(idx, dtype=int)
        result_by_bin[int(bid)] = result
        goodpix_frac = (
            float(result.goodpixels.size / np.count_nonzero(result.clean_mask))
            if result.goodpixels is not None
            and result.clean_mask is not None
            and np.count_nonzero(result.clean_mask) > 0
            else np.nan
        )
        vrel = float(result.sol_rel[0]) if result.sol_rel is not None else np.nan
        vrel_err = float(result.err_rel[0]) if result.err_rel is not None else np.nan
        logage, metal = xsl_mean_population(result.weights, library)
        goodfit = bool(
            result.ok
            and np.isfinite(vrel)
            and np.isfinite(result.sigma)
            and np.isfinite(result.h3)
            and np.isfinite(result.h4)
            and np.isfinite(result.sn)
            and result.sn >= fit_cfg.csv_min_sn
            and fit_cfg.min_sigma <= result.sigma <= fit_cfg.max_sigma
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
            "LOGAGE_YR": logage,
            "METAL": metal,
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

        if result.ok and preview_done < fit_cfg.n_plot_spaxels and bid in preview_ids:
            galaxy, _ = powerbin_base.stack_powerbin_spectrum(
                cube_data.spectra_log,
                cube_data.valid_frac_log,
                idx,
            )
            base.plot_fit(
                outdir / f"xsl_powerbin_fit_bin{int(bid):04d}.png",
                cube_data.lam_gal_ang,
                galaxy,
                result,
                title=(
                    f"XSL PowerBin {int(bid)} stacked mean | N={idx.size}, "
                    f"PowerBin S/N={bin_sn[bid]:.1f}, pPXF S/N={result.sn:.1f}"
                ),
            )
            preview_done += 1

    if n_processes > 1:
        print(f"Fitting PowerBin spectra with {n_processes} worker processes")
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
                fit_cfg,
                start_sigma,
            ),
        ) as pool:
            for bid, idx, result in tqdm(
                pool.map(fit_powerbin_worker, bin_tasks, chunksize=1),
                total=len(bin_tasks),
                desc="Fitting XSL PowerBin spectra",
            ):
                record_result(bid, idx, result)
    else:
        for bid, idx in tqdm(bin_tasks, desc="Fitting XSL PowerBin spectra"):
            galaxy, valid_frac = powerbin_base.stack_powerbin_spectrum(
                cube_data.spectra_log,
                cube_data.valid_frac_log,
                idx,
            )
            fit_mask = cube_data.fit_mask_log & (valid_frac >= fit_cfg.min_log_pixel_fraction)
            result = base.ppxf_fit_spectrum(
                library.templates,
                galaxy,
                cube_data.velscale,
                [0.0, start_sigma],
                fit_mask,
                cube_data.lam_gal_ang,
                library.lam_temp,
                fit_cfg,
            )
            if result.ok and np.isfinite(result.sigma):
                start_sigma = result.sigma
            record_result(bid, idx, result)

    plot_xsl_powerbin_maps(outdir, cube_data, maps)
    high_sigma_check_paths = powerbin_base.write_high_sigma_checkplots(
        outdir,
        cube_data,
        maps,
        bin_num,
        rows,
        idx_by_bin,
        result_by_bin,
        prefix="xsl_powerbin",
        mge_solution_path=cfg.mge_solution_path,
        mge_luminosity_path=cfg.mge_luminosity_path,
        mge_rotation_deg=cfg.mge_contour_rotation_deg,
    )

    good_rows = [row for row in rows if row["GOODFIT"]]
    write_xsl_csv(paths["csv"], good_rows)
    write_xsl_csv(paths["all_csv"], rows)
    manifest_path = outdir / "xsl_template_grid.csv"
    write_xsl_grid_manifest(manifest_path, library)
    save_fits_xsl(paths["fits"], cfg, fit_cfg, target_sn, cube_data, library, rows, maps)
    np.savez_compressed(
        paths["npz"],
        lam_gal_ang=cube_data.lam_gal_ang,
        lam_temp=library.lam_temp,
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
        age_grid=library.age_grid,
        metal_grid=library.metal_grid,
        xsl_template_path=str(library.template_path),
    )

    paths["json"].write_text(
        json.dumps(
            {
                **config_to_json(cfg, target_sn, n_processes, cube_data, library, rows, good_rows),
                "high_sigma_checkplots": [str(path) for path in high_sigma_check_paths],
            },
            indent=2,
        )
        + "\n"
    )

    med_sn = float(np.nanmedian([row["SN"] for row in good_rows])) if good_rows else np.nan
    med_target_sn = float(np.nanmedian([row["SN_TARGET"] for row in rows])) if rows else np.nan
    med_sigma = float(np.nanmedian([row["sigma"] for row in good_rows])) if good_rows else np.nan
    summary_lines = [
        "XSL pPXF SINFONI stellar kinematics with PowerBin spatial binning",
        f"Cube: {cfg.cube_path}",
        f"Output dir: {cfg.output_dir}",
        f"Target PowerBin S/N: {target_sn:.1f}",
        f"Worker processes: {n_processes}",
        f"PowerBin bins: {len(rows)} from {bin_num.size} valid spaxels",
        f"Median PowerBin input S/N: {med_target_sn:.2f}",
        f"XSL template file: {cfg.xsl_template_path}",
        f"XSL templates used: {library.template_count} ({library.reg_dim[0]} ages x {library.reg_dim[1]} metallicities)",
        f"Resolving power: data fixed R={cfg.resolving_power:.1f}",
        f"Median XSL native FWHM: {library.xsl_native_fwhm_med_ang:.3f} A",
        f"Median data FWHM: {library.data_fwhm_med_ang:.3f} A",
        f"pPXF setup: moments={cfg.moments}, degree={cfg.degree}, mdegree={cfg.mdegree}, bias={cfg.bias}, regul=0",
        f"Fit windows rest um: {cfg.fit_windows_rest_um}",
        f"Masked windows rest um: {cfg.mask_windows_rest_um}",
        f"MGE contour rotation on SINFONI frame: {cfg.mge_contour_rotation_deg:.2f} deg",
        f"Good-fit bins: {len(good_rows)}",
        f"Median good-fit pPXF S/N: {med_sn:.2f}",
        f"Median good-fit sigma: {med_sigma:.2f} km/s",
        f"High-sigma check plots: {len(high_sigma_check_paths)} in {outdir / 'high_sigma_checkplots'}",
        f"Good-fit CSV: {paths['csv']}",
        f"All-fits CSV: {paths['all_csv']}",
        f"FITS: {paths['fits']}",
        f"NPZ: {paths['npz']}",
        f"Template grid manifest: {manifest_path}",
    ]
    paths["summary"].write_text("\n".join(summary_lines) + "\n")

    print(f"Saved CSV  : {paths['csv']}")
    print(f"Saved all  : {paths['all_csv']}")
    print(f"Saved FITS : {paths['fits']}")
    print(f"Saved NPZ  : {paths['npz']}")
    print(f"Saved JSON : {paths['json']}")
    print(f"Saved text : {paths['summary']}")
    print(f"Good-fit bins: {len(good_rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
