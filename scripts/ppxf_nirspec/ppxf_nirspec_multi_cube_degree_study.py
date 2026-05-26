#!/usr/bin/env python3
"""
Run and compare PHOENIX pPXF degree studies across multiple NIRSpec cubes.

This wrapper launches the existing single-cube degree/mdegree study, runs the
rotated major/minor-axis profile diagnostic, and then combines the outputs into
cross-cube comparison tables and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_multi_cube_degree"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_OUTPUT_DIR = ROOT / "Data/ppxf_nirspec/cube_comparison_powerbin_sn300_lsf_degree_study"
DEFAULT_DEGREE_SCRIPT = ROOT / "scripts/ppxf_nirspec/ppxf_nirspec_phoenix_powerbin_degree_study.py"
DEFAULT_AXIS_SCRIPT = ROOT / "scripts/ppxf_nirspec/ppxf_nirspec_degree_axis_profiles.py"
DEFAULT_LSF_TABLE = ROOT / "scripts/ppxf_nirspec/jwst_nirspec_g235h_disp.fits"
DEFAULT_CASE_PAIRS = "0:0,0:5,0:10,5:0,5:5,5:10,10:0,10:5,10:6,10:10"
DEFAULT_CUBES = (
    (
        "agn_sub",
        ROOT / "Data/IFU/david_subs/g235h_agn_sub.fits",
    ),
    (
        "antoine_wicked",
        ROOT / "Data/IFU/antoine/sombrero_nirspec_1150_p1293_g235_wicked.fits",
    ),
    (
        "kam_adaptive_trace",
        ROOT / "Data/IFU/kam_adaptive_trace_step/f170lp_g235h-f170lp_s3d.fits",
    ),
)
THREAD_LIMIT_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class CubeSpec:
    label: str
    path: Path
    study_dir: Path


@dataclass(frozen=True)
class BatchConfig:
    cubes: tuple[CubeSpec, ...]
    base_output_dir: Path
    degree_script: Path
    axis_script: Path
    python_executable: str
    lsf_table_path: Path
    target_sn: float
    fit_windows_rest_um: str
    expected_template_count: int
    n_processes: int
    n_plot_bins: int
    check_plot_radius_arcsec: float
    case_pairs: str
    reference_case: tuple[int, int]
    rotation_deg: float
    slit_half_width_arcsec: float
    radial_bin_width_arcsec: float
    min_per_bin: int
    velocity_column: str
    overwrite_run: bool
    skip_fits: bool
    skip_profiles: bool
    dry_run: bool


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_case_pair(text: str) -> tuple[int, int]:
    text = text.strip()
    if ":" in text:
        d_text, m_text = text.split(":", maxsplit=1)
    elif "/" in text:
        d_text, m_text = text.split("/", maxsplit=1)
    else:
        raise ValueError(f"Could not parse case '{text}'. Use degree:mdegree.")
    return int(d_text), int(m_text)


def parse_cube_specs(values: list[str] | None, base_output_dir: Path) -> tuple[CubeSpec, ...]:
    raw_specs = values or [f"{label}:{path}" for label, path in DEFAULT_CUBES]
    specs: list[CubeSpec] = []
    for raw in raw_specs:
        if ":" not in raw:
            raise ValueError(f"Cube spec '{raw}' must be label:/path/to/cube.fits")
        label, path_text = raw.split(":", maxsplit=1)
        label = label.strip()
        if not label:
            raise ValueError(f"Cube spec '{raw}' has an empty label")
        safe_label = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label)
        path = Path(path_text).expanduser().resolve()
        specs.append(CubeSpec(label=safe_label, path=path, study_dir=base_output_dir / safe_label))
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("Cube labels must be unique")
    return tuple(specs)


def expected_case_fits(cube: CubeSpec, target_sn: float, degree: int, mdegree: int) -> Path:
    case_label = f"deg{degree:02d}_mdeg{mdegree:02d}"
    base = cube.study_dir / case_label / f"{cube.path.stem}_phoenix_powerbin_lsf_sn{int(round(target_sn))}_kinematics"
    return base.with_suffix(".fits")


def build_degree_command(cfg: BatchConfig, cube: CubeSpec, forward_args: list[str]) -> list[str]:
    cmd = [
        cfg.python_executable,
        "-u",
        str(cfg.degree_script),
        "--cube-path",
        str(cube.path),
        "--study-dir",
        str(cube.study_dir),
        "--lsf-table-path",
        str(cfg.lsf_table_path),
        "--target-sn",
        f"{cfg.target_sn:.8g}",
        "--fit-windows-rest-um",
        cfg.fit_windows_rest_um,
        "--expected-template-count",
        str(cfg.expected_template_count),
        "--n-processes",
        str(cfg.n_processes),
        "--n-plot-bins",
        str(cfg.n_plot_bins),
        "--check-plot-radius-arcsec",
        f"{cfg.check_plot_radius_arcsec:.8g}",
        "--case-pairs",
        cfg.case_pairs,
    ]
    if cfg.overwrite_run:
        cmd.append("--overwrite-run")
    cmd.extend(forward_args)
    return cmd


def build_axis_command(cfg: BatchConfig, cube: CubeSpec) -> list[str]:
    return [
        cfg.python_executable,
        "-u",
        str(cfg.axis_script),
        "--cube-path",
        str(cube.path),
        "--study-dir",
        str(cube.study_dir),
        "--output-dir",
        str(cube.study_dir),
        "--target-sn",
        f"{cfg.target_sn:.8g}",
        "--case-pairs",
        cfg.case_pairs,
        "--rotation-deg",
        f"{cfg.rotation_deg:.8g}",
        "--slit-half-width-arcsec",
        f"{cfg.slit_half_width_arcsec:.8g}",
        "--radial-bin-width-arcsec",
        f"{cfg.radial_bin_width_arcsec:.8g}",
        "--min-per-bin",
        str(cfg.min_per_bin),
        "--velocity-column",
        cfg.velocity_column,
    ]


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_LIMIT_ENV:
        env.setdefault(key, "1")
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_multi_cube_degree"))
    return env


def shell_join(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_cube_columns(rows: list[dict[str, str]], cube: CubeSpec) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        enriched: dict[str, object] = {
            "cube_label": cube.label,
            "cube_path": str(cube.path),
        }
        enriched.update(row)
        out.append(enriched)
    return out


def load_combined_tables(cfg: BatchConfig) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    brightness_rows: list[dict[str, object]] = []
    for cube in cfg.cubes:
        summary_rows.extend(add_cube_columns(read_csv(cube.study_dir / "degree_study_summary.csv"), cube))
        profile_rows.extend(
            add_cube_columns(
                read_csv(cube.study_dir / "degree_study_major_minor_velocity_sigma_profiles.csv"),
                cube,
            )
        )
        brightness_rows.extend(
            add_cube_columns(
                read_csv(cube.study_dir / "degree_study_brightness_axis_profiles.csv"),
                cube,
            )
        )
    return summary_rows, profile_rows, brightness_rows


def to_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def to_int(row: dict[str, object], key: str) -> int:
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return -999999


def ordered_cases(summary_rows: list[dict[str, object]]) -> list[tuple[int, int]]:
    cases = sorted({(to_int(row, "degree"), to_int(row, "mdegree")) for row in summary_rows})
    return [(degree, mdegree) for degree, mdegree in cases if degree > -999999 and mdegree > -999999]


def plot_metric_comparison(
    outdir: Path,
    cfg: BatchConfig,
    summary_rows: list[dict[str, object]],
) -> Path | None:
    if not summary_rows:
        return None
    cases = ordered_cases(summary_rows)
    labels = [f"{degree}/{mdegree}" for degree, mdegree in cases]
    x = np.arange(len(cases))
    metrics = [
        ("median_sigma", "Median sigma [km/s]"),
        ("central_peak_sigma", "Central peak sigma [km/s]"),
        ("peak_sigma", "Global peak sigma [km/s]"),
        ("median_chi2", "Median chi2"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(cfg.cubes), 1)))

    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for color, cube in zip(colors, cfg.cubes):
            cube_rows = [row for row in summary_rows if row.get("cube_label") == cube.label]
            values = []
            for degree, mdegree in cases:
                match = next(
                    (
                        row
                        for row in cube_rows
                        if to_int(row, "degree") == degree and to_int(row, "mdegree") == mdegree
                    ),
                    None,
                )
                values.append(to_float(match, metric) if match is not None else np.nan)
            ax.plot(x, values, marker="o", lw=1.4, color=color, label=cube.label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("degree/mdegree")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Cube Comparison Degree Study | PowerBin S/N={cfg.target_sn:.0f}", fontsize=13)
    outpath = outdir / "cube_comparison_degree_metrics.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def profile_arrays(
    rows: list[dict[str, object]],
    cube_label: str,
    degree: int,
    mdegree: int,
    axis: str,
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row.get("cube_label") == cube_label
        and to_int(row, "degree") == degree
        and to_int(row, "mdegree") == mdegree
        and row.get("axis") == axis
        and str(row.get("quantity", "")).upper() == quantity.upper()
    ]
    selected.sort(key=lambda row: to_float(row, "radius_arcsec"))
    return (
        np.array([to_float(row, "radius_arcsec") for row in selected], dtype=float),
        np.array([to_float(row, "median") for row in selected], dtype=float),
        np.array([to_float(row, "p16") for row in selected], dtype=float),
        np.array([to_float(row, "p84") for row in selected], dtype=float),
    )


def plot_reference_axis_comparison(
    outdir: Path,
    cfg: BatchConfig,
    profile_rows: list[dict[str, object]],
) -> Path | None:
    if not profile_rows:
        return None
    degree, mdegree = cfg.reference_case
    quantity_specs = [
        ("major", cfg.velocity_column, "Major-Axis Velocity", "V - systemic [km/s]"),
        ("minor", cfg.velocity_column, "Minor-Axis Velocity", "V - systemic [km/s]"),
        ("major", "SIGMA", "Major-Axis Dispersion", "sigma [km/s]"),
        ("minor", "SIGMA", "Minor-Axis Dispersion", "sigma [km/s]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(cfg.cubes), 1)))
    plotted_any = False
    for ax, (axis, quantity, title, ylabel) in zip(axes.ravel(), quantity_specs):
        if quantity.upper() == cfg.velocity_column.upper():
            ax.axhline(0.0, color="black", lw=0.7, alpha=0.35)
        for color, cube in zip(colors, cfg.cubes):
            radius, median, p16, p84 = profile_arrays(
                profile_rows,
                cube.label,
                degree,
                mdegree,
                axis,
                quantity,
            )
            if radius.size == 0:
                continue
            plotted_any = True
            ax.plot(radius, median, marker="o", ms=3.0, lw=1.4, color=color, label=cube.label)
            ax.fill_between(radius, p16, p84, color=color, alpha=0.10, lw=0)
        ax.set_title(title)
        ax.set_xlabel(f"{axis} coordinate after {cfg.rotation_deg:.1f} deg rotation [arcsec]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Cube Comparison Axis Profiles | degree={degree}, mdegree={mdegree}",
        fontsize=13,
    )
    if not plotted_any:
        plt.close(fig)
        return None
    outpath = outdir / f"cube_comparison_axis_profiles_deg{degree:02d}_mdeg{mdegree:02d}.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def brightness_arrays(
    rows: list[dict[str, object]],
    cube_label: str,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row.get("cube_label") == cube_label and row.get("axis") == axis
    ]
    selected.sort(key=lambda row: to_float(row, "radius_arcsec"))
    return (
        np.array([to_float(row, "radius_arcsec") for row in selected], dtype=float),
        np.array([to_float(row, "normalized_median_brightness") for row in selected], dtype=float),
    )


def plot_brightness_comparison(
    outdir: Path,
    cfg: BatchConfig,
    brightness_rows: list[dict[str, object]],
) -> Path | None:
    if not brightness_rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(cfg.cubes), 1)))
    plotted_any = False
    for ax, axis in zip(axes, ("major", "minor")):
        ax.axvline(0.0, color="black", lw=0.7, alpha=0.35)
        for color, cube in zip(colors, cfg.cubes):
            radius, brightness = brightness_arrays(brightness_rows, cube.label, axis)
            if radius.size == 0:
                continue
            plotted_any = True
            ax.plot(radius, brightness, marker="o", ms=3.0, lw=1.4, color=color, label=cube.label)
        ax.set_title(f"{axis.capitalize()}-Axis Brightness")
        ax.set_xlabel(f"{axis} coordinate after {cfg.rotation_deg:.1f} deg rotation [arcsec]")
        ax.set_ylabel("normalized median brightness")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Cube Comparison Brightness Profiles", fontsize=13)
    if not plotted_any:
        plt.close(fig)
        return None
    outpath = outdir / "cube_comparison_brightness_profiles.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def map_extent(header: fits.Header, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    ny, nx = shape
    pixsize = float(header.get("PIXSIZE", 1.0))
    center_row = float(header.get("CENROW", (ny + 1) / 2.0)) - 1.0
    center_col = float(header.get("CENCOL", (nx + 1) / 2.0)) - 1.0
    return (
        float((-0.5 - center_col) * pixsize),
        float((nx - 0.5 - center_col) * pixsize),
        float((-0.5 - center_row) * pixsize),
        float((ny - 0.5 - center_row) * pixsize),
    )


def robust_limits(arrays: list[np.ndarray], symmetric: bool = False) -> tuple[float, float]:
    values = []
    for arr in arrays:
        finite = np.asarray(arr, dtype=float).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(finite)
    if not values:
        return -1.0, 1.0
    merged = np.concatenate(values)
    lo = float(np.nanpercentile(merged, 2.0))
    hi = float(np.nanpercentile(merged, 98.0))
    if symmetric:
        lim = max(abs(lo), abs(hi))
        lo, hi = -lim, lim
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmedian(merged))
        lo, hi = center - 1.0, center + 1.0
    return lo, hi


def load_reference_maps(
    cfg: BatchConfig,
) -> dict[str, dict[str, tuple[np.ndarray, fits.Header]]]:
    degree, mdegree = cfg.reference_case
    maps: dict[str, dict[str, tuple[np.ndarray, fits.Header]]] = {}
    for cube in cfg.cubes:
        path = expected_case_fits(cube, cfg.target_sn, degree, mdegree)
        if not path.exists():
            continue
        with fits.open(path) as hdul:
            header = hdul[0].header.copy()
            good = np.asarray(hdul["GOODFIT_MAP"].data, dtype=bool)
            signal = np.asarray(hdul["SIGNAL_MAP"].data, dtype=float)
            signal_norm = signal / np.nanmax(signal[np.isfinite(signal)])
            maps[cube.label] = {
                "VREL_MAP": (np.where(good, np.asarray(hdul["VREL_MAP"].data, dtype=float), np.nan), header),
                "SIGMA_MAP": (np.where(good, np.asarray(hdul["SIGMA_MAP"].data, dtype=float), np.nan), header),
                "SIGNAL_MAP": (signal_norm, header),
            }
    return maps


def plot_reference_maps(outdir: Path, cfg: BatchConfig) -> Path | None:
    maps = load_reference_maps(cfg)
    if not maps:
        return None
    cube_labels = [cube.label for cube in cfg.cubes if cube.label in maps]
    map_specs = [
        ("VREL_MAP", "Vrel [km/s]", "RdBu_r", True),
        ("SIGMA_MAP", "sigma [km/s]", "magma", False),
        ("SIGNAL_MAP", "normalized brightness", "gray_r", False),
    ]
    fig, axes = plt.subplots(
        len(map_specs),
        len(cube_labels),
        figsize=(4.2 * len(cube_labels), 3.7 * len(map_specs)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (map_name, label, cmap, symmetric) in enumerate(map_specs):
        arrays = [maps[cube_label][map_name][0] for cube_label in cube_labels]
        vmin, vmax = robust_limits(arrays, symmetric=symmetric)
        for col_index, cube_label in enumerate(cube_labels):
            data, header = maps[cube_label][map_name]
            ax = axes[row_index, col_index]
            im = ax.imshow(
                data,
                origin="lower",
                extent=map_extent(header, data.shape),
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.axhline(0.0, color="black", lw=0.4, alpha=0.35)
            ax.axvline(0.0, color="black", lw=0.4, alpha=0.35)
            if row_index == 0:
                ax.set_title(cube_label)
            ax.set_xlabel("arcsec")
            ax.set_ylabel("arcsec")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    degree, mdegree = cfg.reference_case
    fig.suptitle(f"Reference Case Maps | degree={degree}, mdegree={mdegree}", fontsize=13)
    outpath = outdir / f"cube_comparison_reference_maps_deg{degree:02d}_mdeg{mdegree:02d}.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def make_combined_outputs(cfg: BatchConfig) -> list[Path]:
    summary_rows, profile_rows, brightness_rows = load_combined_tables(cfg)
    outputs: list[Path] = []
    summary_csv = cfg.base_output_dir / "cube_comparison_degree_study_summary.csv"
    profile_csv = cfg.base_output_dir / "cube_comparison_axis_profiles.csv"
    brightness_csv = cfg.base_output_dir / "cube_comparison_brightness_profiles.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(profile_csv, profile_rows)
    write_csv(brightness_csv, brightness_rows)
    outputs.extend([summary_csv, profile_csv, brightness_csv])

    for path in (
        plot_metric_comparison(cfg.base_output_dir, cfg, summary_rows),
        plot_reference_axis_comparison(cfg.base_output_dir, cfg, profile_rows),
        plot_brightness_comparison(cfg.base_output_dir, cfg, brightness_rows),
        plot_reference_maps(cfg.base_output_dir, cfg),
    ):
        if path is not None:
            outputs.append(path)
    return outputs


def parse_args(argv: list[str] | None = None) -> tuple[BatchConfig, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run degree/mdegree pPXF studies and comparison plots for multiple cubes.",
    )
    parser.add_argument("--cube", action="append", default=None, help="label:/absolute/path/to/cube.fits")
    parser.add_argument("--base-output-dir", type=Path, default=DEFAULT_BASE_OUTPUT_DIR)
    parser.add_argument("--degree-script", type=Path, default=DEFAULT_DEGREE_SCRIPT)
    parser.add_argument("--axis-script", type=Path, default=DEFAULT_AXIS_SCRIPT)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--lsf-table-path", type=Path, default=DEFAULT_LSF_TABLE)
    parser.add_argument("--target-sn", type=float, default=300.0)
    parser.add_argument("--fit-windows-rest-um", type=str, default="2.1-2.4")
    parser.add_argument("--expected-template-count", type=int, default=0)
    parser.add_argument("--n-processes", type=int, default=6)
    parser.add_argument("--n-plot-bins", type=int, default=0)
    parser.add_argument("--check-plot-radius-arcsec", type=float, default=0.5)
    parser.add_argument("--case-pairs", type=str, default=DEFAULT_CASE_PAIRS)
    parser.add_argument("--reference-case", type=str, default="10:6")
    parser.add_argument("--rotation-deg", type=float, default=-18.0)
    parser.add_argument("--slit-half-width-arcsec", type=float, default=0.15)
    parser.add_argument("--radial-bin-width-arcsec", type=float, default=0.10)
    parser.add_argument("--min-per-bin", type=int, default=1)
    parser.add_argument("--velocity-column", type=str, default="V_REL_KMS")
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--skip-fits", action="store_true", help="Only run profiles/comparison from existing FITS.")
    parser.add_argument("--skip-profiles", action="store_true", help="Run fits only; do not build profile/comparison plots.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running anything.")
    args, forward_args = parser.parse_known_args(argv)

    base_output_dir = args.base_output_dir.expanduser().resolve()
    cfg = BatchConfig(
        cubes=parse_cube_specs(args.cube, base_output_dir),
        base_output_dir=base_output_dir,
        degree_script=args.degree_script.expanduser().resolve(),
        axis_script=args.axis_script.expanduser().resolve(),
        python_executable=str(args.python_executable),
        lsf_table_path=args.lsf_table_path.expanduser().resolve(),
        target_sn=float(args.target_sn),
        fit_windows_rest_um=str(args.fit_windows_rest_um),
        expected_template_count=int(args.expected_template_count),
        n_processes=max(1, int(args.n_processes)),
        n_plot_bins=max(0, int(args.n_plot_bins)),
        check_plot_radius_arcsec=float(args.check_plot_radius_arcsec),
        case_pairs=str(args.case_pairs),
        reference_case=parse_case_pair(args.reference_case),
        rotation_deg=float(args.rotation_deg),
        slit_half_width_arcsec=float(args.slit_half_width_arcsec),
        radial_bin_width_arcsec=float(args.radial_bin_width_arcsec),
        min_per_bin=max(1, int(args.min_per_bin)),
        velocity_column=str(args.velocity_column).upper(),
        overwrite_run=bool(args.overwrite_run),
        skip_fits=bool(args.skip_fits),
        skip_profiles=bool(args.skip_profiles),
        dry_run=bool(args.dry_run),
    )
    return cfg, forward_args


def main(argv: list[str] | None = None) -> None:
    cfg, forward_args = parse_args(argv)

    if cfg.dry_run:
        for cube in cfg.cubes:
            if not cfg.skip_fits:
                print(shell_join(build_degree_command(cfg, cube, forward_args)))
            if not cfg.skip_profiles:
                print(shell_join(build_axis_command(cfg, cube)))
        print("[multi-cube-degree] Dry run complete.")
        return

    ensure_dir(cfg.base_output_dir)
    for cube in cfg.cubes:
        if not cube.path.exists():
            raise FileNotFoundError(f"Cube does not exist: {cube.path}")
        ensure_dir(cube.study_dir)

    config_payload = asdict(cfg)
    config_payload["cubes"] = [
        {"label": cube.label, "path": str(cube.path), "study_dir": str(cube.study_dir)}
        for cube in cfg.cubes
    ]
    config_payload["base_output_dir"] = str(cfg.base_output_dir)
    config_payload["degree_script"] = str(cfg.degree_script)
    config_payload["axis_script"] = str(cfg.axis_script)
    config_payload["lsf_table_path"] = str(cfg.lsf_table_path)
    config_payload["reference_case"] = list(cfg.reference_case)
    config_payload["forward_args"] = forward_args
    (cfg.base_output_dir / "cube_comparison_degree_study_config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n"
    )

    env = subprocess_env()
    for cube in cfg.cubes:
        print(f"[multi-cube-degree] Cube {cube.label}: {cube.path}")
        if not cfg.skip_fits:
            degree_cmd = build_degree_command(cfg, cube, forward_args)
            print(f"[multi-cube-degree] Running degree study in {cube.study_dir}")
            subprocess.run(degree_cmd, check=True, env=env)
        if not cfg.skip_profiles:
            axis_cmd = build_axis_command(cfg, cube)
            print(f"[multi-cube-degree] Building axis profiles in {cube.study_dir}")
            subprocess.run(axis_cmd, check=True, env=env)

    outputs: list[Path] = []
    if not cfg.skip_profiles:
        outputs = make_combined_outputs(cfg)

    print(f"[multi-cube-degree] Base output dir: {cfg.base_output_dir}")
    for path in outputs:
        print(f"[multi-cube-degree] Output: {path}")


if __name__ == "__main__":
    main()
