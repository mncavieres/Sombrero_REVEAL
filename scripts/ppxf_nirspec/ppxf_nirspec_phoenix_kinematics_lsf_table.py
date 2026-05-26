#!/usr/bin/env python3
"""
Fit JWST/NIRSpec stellar kinematics with PHOENIX templates and a wavelength-dependent LSF.

This is the PHOENIX analogue of the previous wavelength-dependent NIRSpec pPXF
run. It uses the same fitting setup as `ppxf_nirspec_phoenix_kinematics.py`,
but instead of convolving the templates to a constant R=2700, it reads the
NIRSpec dispersion/LSF FITS table used before:

    scripts/ppxf_nirspec/jwst_nirspec_g235h_disp.fits

The default table columns are:

    WAVELENGTH  observed wavelength in microns
    R           resolving power

Templates are convolved on the logarithmic wavelength grid with a local Gaussian
whose width follows R(lambda), after subtracting the native PHOENIX R~100000 in
quadrature.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_phoenix_lsf_table"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import numpy as np
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import ppxf_nirspec_phoenix_kinematics as base


C = base.C
FWHM_PER_SIGMA = base.FWHM_PER_SIGMA
ROOT = base.ROOT
DEFAULT_OUTPUT_DIR = ROOT / "Data/ppxf_nirspec/phoenix_g235h_wavelength_lsf"
DEFAULT_LSF_TABLE_PATH = THIS_DIR / "jwst_nirspec_g235h_disp.fits"


@dataclass(frozen=True)
class TableConfig:
    cube_path: Path
    output_dir: Path
    phoenix_dir: Path
    phoenix_wave_path: Path
    template_list_path: Path | None
    lsf_table_path: Path
    lsf_wave_column: str
    lsf_r_column: str
    redshift: float
    template_resolving_power: float
    fit_windows_rest_um: tuple[tuple[float, float], ...]
    mask_windows_rest_um: tuple[tuple[float, float], ...]
    teff_min: float
    teff_max: float
    feh_min: float
    feh_max: float
    logg_min: float
    logg_max: float
    expected_template_count: int
    strict_template_count: bool
    degree: int
    mdegree: int
    moments: int
    bias: float
    start_sigma: float
    max_abs_velocity: float
    min_sigma: float
    max_sigma: float
    min_wave_finite_frac: float
    min_spaxel_finite_frac: float
    min_log_pixel_fraction: float
    min_goodpixels: int
    top_template_frac: float
    csv_min_sn: float
    n_plot_spaxels: int
    max_spaxels: int | None
    template_velocity_margin_kms: float


@dataclass
class TableTemplateLibrary:
    templates: np.ndarray
    lam_temp: np.ndarray
    meta: list[base.TemplateMeta]
    lsf_source: str
    resolving_power_min: float
    resolving_power_med: float
    resolving_power_max: float
    sigma_inst_min_kms: float
    sigma_inst_kms: float
    sigma_inst_max_kms: float
    sigma_template_kms: float
    sigma_conv_min_kms: float
    sigma_conv_kms: float
    sigma_conv_max_kms: float
    sigma_conv_min_pix: float
    sigma_conv_pix: float
    sigma_conv_max_pix: float
    lam_min_ang: float
    lam_max_ang: float
    lsf_lam_obs_um: np.ndarray
    lsf_resolving_power: np.ndarray


def load_lsf_table(path: Path, wave_column: str, r_column: str) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        table_hdu = next(
            (hdu for hdu in hdul if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "columns")),
            None,
        )
        if table_hdu is None:
            raise ValueError(f"Could not find a binary table in LSF FITS file: {path}")
        columns = {name.upper(): name for name in table_hdu.data.columns.names}
        wave_key = columns.get(wave_column.upper())
        r_key = columns.get(r_column.upper())
        if wave_key is None or r_key is None:
            raise KeyError(
                f"LSF table must contain columns {wave_column!r} and {r_column!r}; "
                f"available columns are {table_hdu.data.columns.names}"
            )
        wave = np.asarray(table_hdu.data[wave_key], dtype=float)
        r = np.asarray(table_hdu.data[r_key], dtype=float)
        unit = table_hdu.data.columns[wave_key].unit
        wave_um = base.wavelength_to_um(wave, unit)

    good = np.isfinite(wave_um) & np.isfinite(r) & (wave_um > 0) & (r > 0)
    wave_um = wave_um[good]
    r = r[good]
    if wave_um.size < 2:
        raise ValueError(f"LSF table has too few valid samples: {path}")
    order = np.argsort(wave_um)
    return wave_um[order], r[order]


def variable_gaussian_filter1d(
    values: np.ndarray,
    sigma_pix: np.ndarray,
    truncate: float = 4.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sigma_pix = np.asarray(sigma_pix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected values with shape (npix, ntemplates)")
    if sigma_pix.shape != (values.shape[0],):
        raise ValueError("sigma_pix must have one value per wavelength pixel")

    out = np.empty_like(values)
    npix = values.shape[0]
    for i, sig in enumerate(sigma_pix):
        if not np.isfinite(sig) or sig <= 1e-6:
            out[i] = values[i]
            continue
        radius = max(1, int(np.ceil(truncate * sig)))
        lo = max(0, i - radius)
        hi = min(npix, i + radius + 1)
        dx = np.arange(lo, hi, dtype=float) - i
        weights = np.exp(-0.5 * (dx / sig) ** 2)
        weights /= np.sum(weights)
        out[i] = weights @ values[lo:hi]
    return out


def load_phoenix_templates_with_lsf_table(
    cfg: TableConfig,
    cube_data: base.CubeData,
) -> TableTemplateLibrary:
    meta = base.discover_templates(cfg)
    lsf_wave_um, lsf_r = load_lsf_table(cfg.lsf_table_path, cfg.lsf_wave_column, cfg.lsf_r_column)

    with fits.open(cfg.phoenix_wave_path, memmap=False) as hdul:
        wave_all = np.asarray(hdul[0].data, dtype=float).ravel()
    if wave_all.size == 0:
        raise ValueError(f"Empty PHOENIX wavelength array: {cfg.phoenix_wave_path}")

    fit_lam = cube_data.lam_gal_ang[cube_data.fit_mask_log]
    vel_pad = cfg.max_abs_velocity + cfg.template_velocity_margin_kms
    lam_min = float(np.nanmin(fit_lam) * np.exp(-vel_pad / C))
    lam_max = float(np.nanmax(fit_lam) * np.exp(vel_pad / C))
    wave_sel = (wave_all >= lam_min) & (wave_all <= lam_max)
    if np.count_nonzero(wave_sel) < 10:
        raise ValueError("PHOENIX wavelength file does not overlap the requested fit range")
    wave = wave_all[wave_sel]

    flux = np.empty((wave.size, len(meta)), dtype=np.float32)
    for j, item in enumerate(tqdm(meta, desc="Reading PHOENIX templates")):
        with fits.open(item.path, memmap=True) as hdul:
            data = np.asarray(hdul[0].data[wave_sel], dtype=np.float32)
        flux[:, j] = base.fill_invalid_spectrum(data)

    templates_log, ln_lam_temp, velscale_temp = base.util.log_rebin(
        wave,
        flux,
        velscale=cube_data.velscale,
        flux=False,
    )
    if not np.isclose(velscale_temp, cube_data.velscale, rtol=0, atol=1e-6):
        raise RuntimeError("Template and galaxy velocity scales do not match")

    lam_temp = np.exp(ln_lam_temp)
    lam_temp_obs_um = lam_temp / 1e4 * (1.0 + cfg.redshift)
    if lam_temp_obs_um.min() < lsf_wave_um.min() or lam_temp_obs_um.max() > lsf_wave_um.max():
        raise ValueError(
            "Requested PHOENIX/template wavelength range falls outside the LSF table coverage: "
            f"template observed {lam_temp_obs_um.min():.4f}-{lam_temp_obs_um.max():.4f} um, "
            f"LSF table {lsf_wave_um.min():.4f}-{lsf_wave_um.max():.4f} um"
        )
    resolving_power = np.interp(lam_temp_obs_um, lsf_wave_um, lsf_r)
    sigma_inst = C / resolving_power / FWHM_PER_SIGMA
    sigma_template = C / cfg.template_resolving_power / FWHM_PER_SIGMA
    sigma_conv = np.sqrt(np.maximum(sigma_inst**2 - sigma_template**2, 0.0))
    sigma_conv_pix = sigma_conv / cube_data.velscale

    templates_log = variable_gaussian_filter1d(templates_log, sigma_conv_pix)
    templates_log = base.normalize_columns(templates_log)

    return TableTemplateLibrary(
        templates=np.asarray(templates_log, dtype=float),
        lam_temp=lam_temp,
        meta=meta,
        lsf_source=str(cfg.lsf_table_path),
        resolving_power_min=float(np.nanmin(resolving_power)),
        resolving_power_med=float(np.nanmedian(resolving_power)),
        resolving_power_max=float(np.nanmax(resolving_power)),
        sigma_inst_min_kms=float(np.nanmin(sigma_inst)),
        sigma_inst_kms=float(np.nanmedian(sigma_inst)),
        sigma_inst_max_kms=float(np.nanmax(sigma_inst)),
        sigma_template_kms=float(sigma_template),
        sigma_conv_min_kms=float(np.nanmin(sigma_conv)),
        sigma_conv_kms=float(np.nanmedian(sigma_conv)),
        sigma_conv_max_kms=float(np.nanmax(sigma_conv)),
        sigma_conv_min_pix=float(np.nanmin(sigma_conv_pix)),
        sigma_conv_pix=float(np.nanmedian(sigma_conv_pix)),
        sigma_conv_max_pix=float(np.nanmax(sigma_conv_pix)),
        lam_min_ang=float(wave[0]),
        lam_max_ang=float(wave[-1]),
        lsf_lam_obs_um=lam_temp_obs_um,
        lsf_resolving_power=resolving_power,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "X",
        "Y",
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
        "ROW",
        "COL",
        "SN",
        "CHI2",
        "GOODPIX_FRAC",
        "SPAXEL_FINITE_FRAC",
        "GOODFIT",
        "MESSAGE",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def save_fits_table_lsf(
    path: Path,
    cfg: TableConfig,
    cube_data: base.CubeData,
    library: TableTemplateLibrary,
    table_rows: list[dict[str, object]],
    maps: dict[str, np.ndarray],
) -> None:
    hdr = fits.Header()
    hdr["OBJECT"] = cfg.cube_path.stem[:68]
    hdr["REDSHFT"] = float(cfg.redshift)
    hdr["LSFMODE"] = "TABLE"
    hdr["LSFSRC"] = str(cfg.lsf_table_path)[:68]
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
    hdr["LAMMIN"] = float(np.nanmin(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["LAMMAX"] = float(np.nanmax(cube_data.lam_gal_ang[cube_data.fit_mask_log]))
    hdr["PIXSIZE"] = (float(cube_data.pixsize_arcsec), "arcsec")
    hdr["CENROW"] = int(cube_data.center_row + 1)
    hdr["CENCOL"] = int(cube_data.center_col + 1)

    cols = []
    table_keys = [
        "ROW",
        "COL",
        "X",
        "Y",
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
        "SN",
        "CHI2",
        "GOODPIX_FRAC",
        "SPAXEL_FINITE_FRAC",
        "GOODFIT",
    ]
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
    kin_hdu = fits.BinTableHDU.from_columns(cols, name="KIN_RESULTS")

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
        fits.ImageHDU(data=maps["VREL_MAP"].astype(np.float32), name="VREL_MAP"),
        fits.ImageHDU(data=maps["LOSV_MAP"].astype(np.float32), name="LOSV_MAP"),
        fits.ImageHDU(data=maps["SIGMA_MAP"].astype(np.float32), name="SIGMA_MAP"),
        fits.ImageHDU(data=maps["VRMS_MAP"].astype(np.float32), name="VRMS_MAP"),
        fits.ImageHDU(data=maps["H3_MAP"].astype(np.float32), name="H3_MAP"),
        fits.ImageHDU(data=maps["H4_MAP"].astype(np.float32), name="H4_MAP"),
        fits.ImageHDU(data=maps["SN_MAP"].astype(np.float32), name="SN_MAP"),
        fits.ImageHDU(data=maps["CHI2_MAP"].astype(np.float32), name="CHI2_MAP"),
        fits.ImageHDU(data=maps["GOODFIT_MAP"].astype(np.int16), name="GOODFIT_MAP"),
        fits.ImageHDU(data=cube_data.signal_map.astype(np.float32), name="SIGNAL_MAP"),
        fits.ImageHDU(data=cube_data.noise_proxy_map.astype(np.float32), name="NOISE_MAP"),
        fits.ImageHDU(data=cube_data.lam_gal_ang.astype(np.float32), name="LAMBDA_REST"),
        fits.ImageHDU(data=cube_data.fit_mask_log.astype(np.int16), name="FIT_MASK"),
        fits.ImageHDU(data=library.lam_temp.astype(np.float32), name="TEMPLATE_LAMBDA"),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def config_to_json(cfg: TableConfig, library: TableTemplateLibrary, cube_data: base.CubeData) -> dict[str, object]:
    data = asdict(cfg)
    for key, val in list(data.items()):
        if isinstance(val, Path):
            data[key] = str(val)
        elif isinstance(val, tuple):
            data[key] = list(val)
    data.update(
        {
            "n_templates": len(library.meta),
            "lsf_source": library.lsf_source,
            "resolving_power_min": library.resolving_power_min,
            "resolving_power_med": library.resolving_power_med,
            "resolving_power_max": library.resolving_power_max,
            "sigma_inst_min_kms": library.sigma_inst_min_kms,
            "sigma_inst_kms": library.sigma_inst_kms,
            "sigma_inst_max_kms": library.sigma_inst_max_kms,
            "sigma_template_kms": library.sigma_template_kms,
            "sigma_conv_min_kms": library.sigma_conv_min_kms,
            "sigma_conv_kms": library.sigma_conv_kms,
            "sigma_conv_max_kms": library.sigma_conv_max_kms,
            "sigma_conv_min_pix": library.sigma_conv_min_pix,
            "sigma_conv_pix": library.sigma_conv_pix,
            "sigma_conv_max_pix": library.sigma_conv_max_pix,
            "velscale_kms": cube_data.velscale,
            "cube_shape": cube_data.cube_shape,
            "map_shape": cube_data.map_shape,
        }
    )
    return data


def parse_args() -> TableConfig:
    parser = argparse.ArgumentParser(
        description="Fit NIRSpec stellar kinematics with PHOENIX templates and wavelength-dependent LSF table.",
    )
    parser.add_argument("--cube-path", type=Path, default=base.DEFAULT_CUBE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phoenix-dir", type=Path, default=base.DEFAULT_PHOENIX_DIR)
    parser.add_argument("--phoenix-wave-path", type=Path, default=base.DEFAULT_PHOENIX_WAVE)
    parser.add_argument("--template-list", type=str, default=None)
    parser.add_argument("--lsf-table-path", type=Path, default=DEFAULT_LSF_TABLE_PATH)
    parser.add_argument("--lsf-wave-column", type=str, default="WAVELENGTH")
    parser.add_argument("--lsf-r-column", type=str, default="R")
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
    parser.add_argument("--n-plot-spaxels", type=int, default=8)
    parser.add_argument("--max-spaxels", type=int, default=None)
    parser.add_argument("--template-velocity-margin-kms", type=float, default=1500.0)

    args = parser.parse_args()
    return TableConfig(
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
        n_plot_spaxels=int(args.n_plot_spaxels),
        max_spaxels=args.max_spaxels,
        template_velocity_margin_kms=float(args.template_velocity_margin_kms),
    )


def main() -> None:
    cfg = parse_args()
    outdir = base.ensure_dir(cfg.output_dir)

    print(f"Reading cube: {cfg.cube_path}")
    cube_data = base.read_cube_data(cfg)
    print(
        f"Log-rebinned cube: {cube_data.spectra_log.shape[0]} pixels, "
        f"{cube_data.spectra_log.shape[1]} spectra, velscale={cube_data.velscale:.3f} km/s"
    )

    print(f"Loading PHOENIX templates from: {cfg.phoenix_dir}")
    print(f"Using wavelength-dependent LSF table: {cfg.lsf_table_path}")
    library = load_phoenix_templates_with_lsf_table(cfg, cube_data)
    print(
        f"Using {len(library.meta)} templates; R(lambda)={library.resolving_power_min:.1f}-"
        f"{library.resolving_power_max:.1f}, median={library.resolving_power_med:.1f}; "
        f"sigma_conv median={library.sigma_conv_kms:.2f} km/s"
    )

    n_valid = cube_data.valid_spaxel_indices.size
    n_global = max(1, int(np.ceil(cfg.top_template_frac * n_valid)))
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
        outdir / "phoenix_lsf_table_global_fit.png",
        cube_data.lam_gal_ang,
        global_spec,
        global_result,
        title=(
            f"Global PHOENIX LSF-table pPXF fit | "
            f"V={global_result.sol_rel[0]:.1f} km/s, "
            f"sigma={global_result.sigma:.1f} km/s"
            if global_result.sol_rel is not None
            else f"Global PHOENIX LSF-table pPXF fit failed: {global_result.message}"
        ),
    )
    start_sigma = global_result.sigma if np.isfinite(global_result.sigma) else cfg.start_sigma

    maps = base.make_maps(cube_data.map_shape)
    rows: list[dict[str, object]] = []
    preview_order = np.argsort(cube_data.signal)[::-1]
    preview_candidates = set(preview_order[: max(cfg.n_plot_spaxels * 4, 20)])
    preview_done = 0

    for j in tqdm(range(n_valid), desc="Fitting NIRSpec spaxels"):
        galaxy = cube_data.spectra_log[:, j]
        fit_mask = cube_data.fit_mask_log & (cube_data.valid_frac_log[:, j] >= cfg.min_log_pixel_fraction)
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

        row = int(cube_data.row[j])
        col = int(cube_data.col[j])
        map_row = row - 1
        map_col = col - 1
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

        row_dict = {
            "ROW": row,
            "COL": col,
            "X": float(cube_data.x[j]),
            "Y": float(cube_data.y[j]),
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
            "SN": float(result.sn),
            "CHI2": float(result.chi2),
            "GOODPIX_FRAC": goodpix_frac,
            "SPAXEL_FINITE_FRAC": float(cube_data.spaxel_finite_frac[j]),
            "GOODFIT": goodfit,
            "MESSAGE": result.message,
        }
        rows.append(row_dict)

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

            if preview_done < cfg.n_plot_spaxels and j in preview_candidates:
                base.plot_fit(
                    outdir / f"phoenix_lsf_table_spaxel_fit_r{row:02d}_c{col:02d}.png",
                    cube_data.lam_gal_ang,
                    galaxy,
                    result,
                    title=f"Spaxel row={row}, col={col}",
                )
                preview_done += 1

    base.plot_maps(outdir, cube_data, maps)

    base_name = outdir / f"{cfg.cube_path.stem}_phoenix_lsf_table_kinematics"
    csv_path = base_name.with_suffix(".csv")
    all_csv_path = base_name.with_name(base_name.name + "_all").with_suffix(".csv")
    fits_path = base_name.with_suffix(".fits")
    npz_path = base_name.with_suffix(".npz")
    json_path = outdir / "phoenix_lsf_table_run_config.json"
    summary_path = outdir / "phoenix_lsf_table_run_summary.txt"
    manifest_path = outdir / "selected_phoenix_templates.csv"

    good_rows = [row for row in rows if row["GOODFIT"]]
    write_csv(csv_path, good_rows)
    write_csv(all_csv_path, rows)
    base.write_template_manifest(manifest_path, library)
    save_fits_table_lsf(fits_path, cfg, cube_data, library, rows, maps)
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
        table_rows=np.array(rows, dtype=object),
        vrel_map=maps["VREL_MAP"],
        losv_map=maps["LOSV_MAP"],
        sigma_map=maps["SIGMA_MAP"],
        vrms_map=maps["VRMS_MAP"],
        h3_map=maps["H3_MAP"],
        h4_map=maps["H4_MAP"],
        sn_map=maps["SN_MAP"],
        chi2_map=maps["CHI2_MAP"],
        goodfit_map=maps["GOODFIT_MAP"],
        template_filenames=np.array([m.path.name for m in library.meta]),
        template_teff=np.array([m.teff for m in library.meta]),
        template_logg=np.array([m.logg for m in library.meta]),
        template_feh=np.array([m.feh for m in library.meta]),
    )
    json_path.write_text(json.dumps(config_to_json(cfg, library, cube_data), indent=2) + "\n")

    n_good = len(good_rows)
    med_sn = float(np.nanmedian([row["SN"] for row in good_rows])) if good_rows else np.nan
    med_sigma = float(np.nanmedian([row["sigma"] for row in good_rows])) if good_rows else np.nan
    summary_lines = [
        "PHOENIX pPXF NIRSpec stellar kinematics with wavelength-dependent LSF",
        f"Cube: {cfg.cube_path}",
        f"Output dir: {cfg.output_dir}",
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
        f"Total fitted spectra: {len(rows)}",
        f"Good-fit spectra: {n_good}",
        f"Median good-fit S/N: {med_sn:.2f}",
        f"Median good-fit sigma: {med_sigma:.2f} km/s",
        f"Good-fit CSV: {csv_path}",
        f"All-fits CSV: {all_csv_path}",
        f"FITS: {fits_path}",
        f"NPZ: {npz_path}",
        f"Template manifest: {manifest_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved CSV  : {csv_path}")
    print(f"Saved all  : {all_csv_path}")
    print(f"Saved FITS : {fits_path}")
    print(f"Saved NPZ  : {npz_path}")
    print(f"Saved JSON : {json_path}")
    print(f"Saved text : {summary_path}")
    print(f"Good-fit rows: {n_good}/{len(rows)}")


if __name__ == "__main__":
    main()
