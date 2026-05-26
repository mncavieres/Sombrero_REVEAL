#!/usr/bin/env python3
"""
Run a sparse pPXF polynomial-degree study for the PHOENIX PowerBin LSF workflow.

The default study runs ten fits spanning additive polynomial degree and
multiplicative polynomial mdegree from 0 to 10. It includes the two endpoints,
corners/midpoints of the 0-10 plane, and the paper-like (degree=10, mdegree=6)
case used in the current PHOENIX pipeline.
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
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_degree_study"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUBE_PATH = ROOT / "Data/IFU/antoine/sombrero_nirspec_1150_p1293_g235_wicked.fits"
DEFAULT_STUDY_DIR = ROOT / "Data/ppxf_nirspec/antoine_wicked_powerbin_sn120_lsf_degree_study"
DEFAULT_CORE_SCRIPT = ROOT / "scripts/ppxf_nirspec/ppxf_nirspec_phoenix_powerbin_kinematics_lsf_table.py"
DEFAULT_LSF_TABLE = ROOT / "scripts/ppxf_nirspec/jwst_nirspec_g235h_disp.fits"
DEFAULT_CASE_PAIRS = "0:0,0:5,0:10,5:0,5:5,5:10,10:0,10:5,10:6,10:10"
THREAD_LIMIT_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class DegreeCase:
    degree: int
    mdegree: int

    @property
    def label(self) -> str:
        return f"deg{self.degree:02d}_mdeg{self.mdegree:02d}"

    @property
    def short_label(self) -> str:
        return f"d={self.degree}, m={self.mdegree}"


@dataclass(frozen=True)
class StudyConfig:
    cube_path: Path
    study_dir: Path
    core_script: Path
    python_executable: str
    lsf_table_path: Path
    target_sn: float
    fit_windows_rest_um: str
    expected_template_count: int
    n_processes: int
    n_plot_bins: int
    check_plot_radius_arcsec: float
    cases: tuple[DegreeCase, ...]
    overwrite_run: bool
    summarize_only: bool
    dry_run: bool


@dataclass(frozen=True)
class CaseSummary:
    label: str
    degree: int
    mdegree: int
    output_dir: str
    fits_path: str
    n_total: int
    n_goodfit: int
    goodfit_fraction: float
    median_sigma: float
    p95_sigma: float
    p99_sigma: float
    peak_sigma: float
    peak_sigma_x_arcsec: float
    peak_sigma_y_arcsec: float
    peak_sigma_bin_id: int
    central_radius_arcsec: float
    central_peak_sigma: float
    central_peak_sigma_x_arcsec: float
    central_peak_sigma_y_arcsec: float
    central_peak_sigma_bin_id: int
    median_vrms: float
    p99_vrms: float
    peak_vrms: float
    median_vrel: float
    p95_abs_vrel: float
    peak_abs_vrel: float
    median_h3: float
    median_h4: float
    median_sn: float
    min_sn: float
    median_chi2: float
    p95_chi2: float
    median_nspax: float
    max_nspax: float


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_case_pairs(text: str) -> tuple[DegreeCase, ...]:
    cases: list[DegreeCase] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            d_text, m_text = chunk.split(":", maxsplit=1)
        elif "/" in chunk:
            d_text, m_text = chunk.split("/", maxsplit=1)
        else:
            raise ValueError(
                f"Could not parse case '{chunk}'. Use degree:mdegree, for example 10:6."
            )
        degree = int(d_text)
        mdegree = int(m_text)
        if degree < 0 or mdegree < 0:
            raise ValueError(f"Degrees must be non-negative for this study, got {chunk}.")
        cases.append(DegreeCase(degree=degree, mdegree=mdegree))
    if not cases:
        raise ValueError("Need at least one degree:mdegree case.")
    labels = [case.label for case in cases]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate degree:mdegree cases are not allowed.")
    return tuple(cases)


def expected_products(outdir: Path, cube_stem: str, target_sn: float) -> dict[str, Path]:
    base_path = outdir / f"{cube_stem}_phoenix_powerbin_lsf_sn{int(round(target_sn))}_kinematics"
    return {
        "fits": base_path.with_suffix(".fits"),
        "good_csv": base_path.with_suffix(".csv"),
        "all_csv": base_path.with_name(base_path.name + "_all").with_suffix(".csv"),
        "npz": base_path.with_suffix(".npz"),
    }


def completed_case(outdir: Path, cube_stem: str, target_sn: float) -> bool:
    products = expected_products(outdir, cube_stem, target_sn)
    return products["fits"].exists() and products["all_csv"].exists() and products["npz"].exists()


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def finite_selected(tab: np.recarray, key: str, sel: np.ndarray) -> np.ndarray:
    if key not in tab.names:
        return np.array([], dtype=float)
    arr = np.asarray(tab[key], dtype=float)
    return arr[sel & np.isfinite(arr)]


def percentile(values: np.ndarray, q: float) -> float:
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.nanpercentile(values, q))


def median(values: np.ndarray) -> float:
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.nanmedian(values))


def maximum(values: np.ndarray) -> float:
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.nanmax(values))


def map_extent(header: fits.Header, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    ny, nx = shape
    pixsize = float(header.get("PIXSIZE", 1.0))
    cen_row = float(header.get("CENROW", (ny + 1) / 2.0)) - 1.0
    cen_col = float(header.get("CENCOL", (nx + 1) / 2.0)) - 1.0
    return (
        float((-0.5 - cen_col) * pixsize),
        float((nx - 0.5 - cen_col) * pixsize),
        float((-0.5 - cen_row) * pixsize),
        float((ny - 0.5 - cen_row) * pixsize),
    )


def pixel_coordinates(header: fits.Header, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    pixsize = float(header.get("PIXSIZE", np.nan))
    cen_row = float(header.get("CENROW", (ny + 1) / 2.0)) - 1.0
    cen_col = float(header.get("CENCOL", (nx + 1) / 2.0)) - 1.0
    row, col = np.indices(shape, dtype=float)
    return (col - cen_col) * pixsize, (row - cen_row) * pixsize


def central_peak_from_maps(
    hdul: fits.HDUList,
    radius_arcsec: float,
) -> tuple[float, float, float, int]:
    sigma_map = np.asarray(hdul["SIGMA_MAP"].data, dtype=float)
    good_map = np.asarray(hdul["GOODFIT_MAP"].data, dtype=bool)
    bin_map = np.asarray(hdul["BIN_ID_MAP"].data, dtype=int)
    x, y = pixel_coordinates(hdul[0].header, sigma_map.shape)
    radius = np.hypot(x, y)
    mask = (radius <= radius_arcsec) & good_map & np.isfinite(sigma_map)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, -1
    masked_sigma = np.where(mask, sigma_map, np.nan)
    row, col = np.unravel_index(np.nanargmax(masked_sigma), sigma_map.shape)
    return (
        float(sigma_map[row, col]),
        float(x[row, col]),
        float(y[row, col]),
        int(bin_map[row, col]),
    )


def summarize_case(outdir: Path, cfg: StudyConfig, case: DegreeCase) -> CaseSummary:
    products = expected_products(outdir, cfg.cube_path.stem, cfg.target_sn)
    fits_path = products["fits"]
    if not fits_path.exists():
        raise FileNotFoundError(f"Missing FITS product for {case.label}: {fits_path}")

    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        tab = hdul["BIN_RESULTS"].data
        total = int(len(tab))
        good = np.asarray(tab["GOODFIT"], dtype=bool) if total else np.array([], dtype=bool)
        sigma_all = np.asarray(tab["SIGMA"], dtype=float) if total else np.array([], dtype=float)
        if np.any(good):
            sel = good
        else:
            sel = np.isfinite(sigma_all)

        sigma = finite_selected(tab, "SIGMA", sel)
        vrel = finite_selected(tab, "V_REL_KMS", sel)
        vrms = finite_selected(tab, "VRMS", sel)
        h3 = finite_selected(tab, "H3", sel)
        h4 = finite_selected(tab, "H4", sel)
        sn = finite_selected(tab, "SN", sel)
        chi2 = finite_selected(tab, "CHI2", sel)
        nspax = finite_selected(tab, "NSPAX", sel)

        peak_sigma = np.nan
        peak_sigma_x = np.nan
        peak_sigma_y = np.nan
        peak_sigma_bin = -1
        if sigma.size:
            sigma_tab = np.asarray(tab["SIGMA"], dtype=float)
            peak_mask = sel & np.isfinite(sigma_tab)
            peak_indices = np.flatnonzero(peak_mask)
            if peak_indices.size:
                peak_index = int(peak_indices[np.nanargmax(sigma_tab[peak_mask])])
                peak_sigma = float(sigma_tab[peak_index])
                peak_sigma_x = float(tab["X"][peak_index])
                peak_sigma_y = float(tab["Y"][peak_index])
                peak_sigma_bin = int(tab["BIN_ID"][peak_index])

        central_peak_sigma, central_peak_x, central_peak_y, central_peak_bin = central_peak_from_maps(
            hdul,
            cfg.check_plot_radius_arcsec,
        )

        degree = int(hdr.get("DEGREE", case.degree))
        mdegree = int(hdr.get("MDEGREE", case.mdegree))

    return CaseSummary(
        label=case.label,
        degree=degree,
        mdegree=mdegree,
        output_dir=str(outdir),
        fits_path=str(fits_path),
        n_total=total,
        n_goodfit=int(np.count_nonzero(good)),
        goodfit_fraction=float(np.count_nonzero(good) / total) if total else np.nan,
        median_sigma=median(sigma),
        p95_sigma=percentile(sigma, 95.0),
        p99_sigma=percentile(sigma, 99.0),
        peak_sigma=peak_sigma,
        peak_sigma_x_arcsec=peak_sigma_x,
        peak_sigma_y_arcsec=peak_sigma_y,
        peak_sigma_bin_id=peak_sigma_bin,
        central_radius_arcsec=float(cfg.check_plot_radius_arcsec),
        central_peak_sigma=central_peak_sigma,
        central_peak_sigma_x_arcsec=central_peak_x,
        central_peak_sigma_y_arcsec=central_peak_y,
        central_peak_sigma_bin_id=central_peak_bin,
        median_vrms=median(vrms),
        p99_vrms=percentile(vrms, 99.0),
        peak_vrms=maximum(vrms),
        median_vrel=median(vrel),
        p95_abs_vrel=percentile(np.abs(vrel), 95.0),
        peak_abs_vrel=maximum(np.abs(vrel)),
        median_h3=median(h3),
        median_h4=median(h4),
        median_sn=median(sn),
        min_sn=float(np.nanmin(sn)) if sn.size else np.nan,
        median_chi2=median(chi2),
        p95_chi2=percentile(chi2, 95.0),
        median_nspax=median(nspax),
        max_nspax=maximum(nspax),
    )


def write_summary_csv(path: Path, cases: list[CaseSummary]) -> None:
    rows = [asdict(case) for case in cases]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, cfg: StudyConfig, cases: list[CaseSummary], forward_args: list[str]) -> None:
    payload = {
        "cube_path": str(cfg.cube_path),
        "study_dir": str(cfg.study_dir),
        "core_script": str(cfg.core_script),
        "python_executable": cfg.python_executable,
        "lsf_table_path": str(cfg.lsf_table_path),
        "target_sn": cfg.target_sn,
        "fit_windows_rest_um": cfg.fit_windows_rest_um,
        "expected_template_count": cfg.expected_template_count,
        "n_processes": cfg.n_processes,
        "n_plot_bins": cfg.n_plot_bins,
        "check_plot_radius_arcsec": cfg.check_plot_radius_arcsec,
        "cases": [asdict(case) for case in cfg.cases],
        "forward_args": forward_args,
        "summaries": [asdict(case) for case in cases],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_notes(path: Path, cfg: StudyConfig, forward_args: list[str]) -> None:
    lines = [
        "PHOENIX PowerBin pPXF polynomial-degree study",
        "",
        "Purpose:",
        "- Compare how additive polynomial degree and multiplicative polynomial mdegree affect the kinematic maps.",
        "- Track median, high-percentile, global peak, and central-0.5-arcsec peak velocity dispersion.",
        "",
        "Default 10-case sparse grid:",
        "- "
        + ", ".join(f"({case.degree}, {case.mdegree})" for case in cfg.cases),
        "",
        "Configuration:",
        f"- Cube: {cfg.cube_path}",
        f"- Core runner: {cfg.core_script}",
        f"- LSF table: {cfg.lsf_table_path}",
        f"- Target PowerBin S/N: {cfg.target_sn:.1f}",
        f"- Fit windows rest um: {cfg.fit_windows_rest_um}",
        f"- Worker processes per fit: {cfg.n_processes}",
        f"- Check/central peak radius: {cfg.check_plot_radius_arcsec:.2f} arcsec",
        f"- Extra forwarded core arguments: {' '.join(forward_args) if forward_args else '(none)'}",
        "",
        "Outputs:",
        "- degree_study_summary.csv",
        "- degree_study_summary.json",
        "- degree_study_metrics.png",
        "- degree_study_sigma_maps.png",
        "- degree_study_vrel_maps.png",
        "- degree_study_vrms_maps.png",
        "- degree_study_h4_maps.png",
        "- degree_study_chi2_maps.png",
    ]
    path.write_text("\n".join(lines) + "\n")


def robust_limits(arrays: list[np.ndarray], symmetric: bool = False) -> tuple[float, float]:
    values = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=float).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size:
            values.append(flat)
    if not values:
        return -1.0, 1.0
    all_values = np.concatenate(values)
    if all_values.size < 4:
        lo = float(np.nanmin(all_values))
        hi = float(np.nanmax(all_values))
    else:
        lo = float(np.nanpercentile(all_values, 2.0))
        hi = float(np.nanpercentile(all_values, 98.0))
    if symmetric:
        lim = max(abs(lo), abs(hi))
        lo, hi = -lim, lim
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmedian(all_values)) if all_values.size else 0.0
        lo, hi = center - 1.0, center + 1.0
    return lo, hi


def load_masked_map(fits_path: Path, map_name: str) -> tuple[np.ndarray, fits.Header]:
    with fits.open(fits_path) as hdul:
        data = np.asarray(hdul[map_name].data, dtype=float)
        if "GOODFIT_MAP" in hdul:
            good = np.asarray(hdul["GOODFIT_MAP"].data, dtype=bool)
            data = np.where(good, data, np.nan)
        header = hdul[0].header.copy()
    return data, header


def plot_map_grid(
    study_dir: Path,
    cfg: StudyConfig,
    summaries: list[CaseSummary],
    map_name: str,
    output_name: str,
    title: str,
    cmap: str,
    colorbar_label: str,
    symmetric: bool = False,
) -> Path:
    maps: list[np.ndarray] = []
    headers: list[fits.Header] = []
    for summary in summaries:
        data, header = load_masked_map(Path(summary.fits_path), map_name)
        maps.append(data)
        headers.append(header)

    vmin, vmax = robust_limits(maps, symmetric=symmetric)
    n = len(summaries)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.45 * ncols, 3.35 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    im = None
    for ax, summary, data, header in zip(axes.ravel(), summaries, maps, headers):
        extent = map_extent(header, data.shape)
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.axhline(0.0, color="black", lw=0.4, alpha=0.35)
        ax.axvline(0.0, color="black", lw=0.4, alpha=0.35)
        if map_name == "SIGMA_MAP":
            ax.set_title(
                f"d={summary.degree}, m={summary.mdegree}\n"
                f"peak={summary.peak_sigma:.0f}, c={summary.central_peak_sigma:.0f}",
                fontsize=9,
            )
        else:
            ax.set_title(f"d={summary.degree}, m={summary.mdegree}", fontsize=9)
        ax.set_xlabel("arcsec")
        ax.set_ylabel("arcsec")

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, label=colorbar_label)
    fig.suptitle(title, fontsize=13)
    outpath = study_dir / output_name
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_metric_summary(study_dir: Path, summaries: list[CaseSummary]) -> Path:
    labels = [f"{s.degree}/{s.mdegree}" for s in summaries]
    x = np.arange(len(summaries))
    median_sigma = np.array([s.median_sigma for s in summaries], dtype=float)
    p95_sigma = np.array([s.p95_sigma for s in summaries], dtype=float)
    p99_sigma = np.array([s.p99_sigma for s in summaries], dtype=float)
    peak_sigma = np.array([s.peak_sigma for s in summaries], dtype=float)
    central_peak_sigma = np.array([s.central_peak_sigma for s in summaries], dtype=float)
    median_chi2 = np.array([s.median_chi2 for s in summaries], dtype=float)
    p95_chi2 = np.array([s.p95_chi2 for s in summaries], dtype=float)
    good_fraction = np.array([s.goodfit_fraction for s in summaries], dtype=float)
    median_sn = np.array([s.median_sn for s in summaries], dtype=float)
    peak_abs_vrel = np.array([s.peak_abs_vrel for s in summaries], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, median_sigma, marker="o", lw=1.5, label="median")
    ax.plot(x, p95_sigma, marker="s", lw=1.5, label="p95")
    ax.plot(x, p99_sigma, marker="^", lw=1.5, label="p99")
    ax.plot(x, peak_sigma, marker="D", lw=1.5, label="global peak")
    ax.plot(x, central_peak_sigma, marker="*", lw=1.5, label="peak within central radius")
    ax.set_ylabel("sigma [km/s]")
    ax.set_title("Velocity Dispersion")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, median_chi2, marker="o", lw=1.5, label="median chi2")
    ax.plot(x, p95_chi2, marker="s", lw=1.5, label="p95 chi2")
    ax.set_ylabel("chi2")
    ax.set_title("Fit Quality")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, good_fraction, marker="o", lw=1.5, color="tab:green", label="good-fit fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("fraction")
    ax2 = ax.twinx()
    ax2.plot(x, median_sn, marker="s", lw=1.5, color="tab:purple", label="median S/N")
    ax2.set_ylabel("median pPXF S/N")
    ax.set_title("Good Fits and S/N")
    ax.grid(alpha=0.25)
    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, line_labels + line_labels2, fontsize=8, loc="best")

    ax = axes[1, 1]
    ax.plot(x, peak_abs_vrel, marker="o", lw=1.5, color="tab:red")
    ax.set_ylabel("peak |Vrel| [km/s]")
    ax.set_title("Velocity Outliers")
    ax.grid(alpha=0.25)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("degree/mdegree")

    outpath = study_dir / "degree_study_metrics.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def write_text_summary(path: Path, summaries: list[CaseSummary]) -> None:
    if not summaries:
        path.write_text("No completed cases.\n")
        return
    peak = max(summaries, key=lambda item: np.nan_to_num(item.peak_sigma, nan=-np.inf))
    central_peak = max(
        summaries,
        key=lambda item: np.nan_to_num(item.central_peak_sigma, nan=-np.inf),
    )
    lowest_chi2 = min(summaries, key=lambda item: np.nan_to_num(item.median_chi2, nan=np.inf))
    sigma_vals = np.array([s.median_sigma for s in summaries], dtype=float)
    peak_vals = np.array([s.peak_sigma for s in summaries], dtype=float)
    central_vals = np.array([s.central_peak_sigma for s in summaries], dtype=float)
    lines = [
        "pPXF degree/mdegree study summary",
        "",
        f"Completed cases: {len(summaries)}",
        f"Median sigma span: {np.nanmin(sigma_vals):.2f} to {np.nanmax(sigma_vals):.2f} km/s",
        f"Global peak sigma span: {np.nanmin(peak_vals):.2f} to {np.nanmax(peak_vals):.2f} km/s",
        f"Central peak sigma span: {np.nanmin(central_vals):.2f} to {np.nanmax(central_vals):.2f} km/s",
        "",
        (
            "Highest global peak sigma: "
            f"{peak.peak_sigma:.2f} km/s for degree={peak.degree}, mdegree={peak.mdegree} "
            f"at x={peak.peak_sigma_x_arcsec:.3f}, y={peak.peak_sigma_y_arcsec:.3f} arcsec"
        ),
        (
            "Highest central peak sigma: "
            f"{central_peak.central_peak_sigma:.2f} km/s for degree={central_peak.degree}, "
            f"mdegree={central_peak.mdegree}"
        ),
        (
            "Lowest median chi2: "
            f"{lowest_chi2.median_chi2:.4g} for degree={lowest_chi2.degree}, "
            f"mdegree={lowest_chi2.mdegree}"
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def make_plots(study_dir: Path, cfg: StudyConfig, summaries: list[CaseSummary]) -> list[Path]:
    paths: list[Path] = []
    if not summaries:
        return paths
    paths.append(plot_metric_summary(study_dir, summaries))
    paths.append(
        plot_map_grid(
            study_dir,
            cfg,
            summaries,
            "SIGMA_MAP",
            "degree_study_sigma_maps.png",
            "Velocity Dispersion Maps",
            "magma",
            "sigma [km/s]",
            symmetric=False,
        )
    )
    paths.append(
        plot_map_grid(
            study_dir,
            cfg,
            summaries,
            "VREL_MAP",
            "degree_study_vrel_maps.png",
            "Relative Velocity Maps",
            "RdBu_r",
            "Vrel [km/s]",
            symmetric=True,
        )
    )
    paths.append(
        plot_map_grid(
            study_dir,
            cfg,
            summaries,
            "VRMS_MAP",
            "degree_study_vrms_maps.png",
            "Vrms Maps",
            "viridis",
            "Vrms [km/s]",
            symmetric=False,
        )
    )
    paths.append(
        plot_map_grid(
            study_dir,
            cfg,
            summaries,
            "H4_MAP",
            "degree_study_h4_maps.png",
            "h4 Maps",
            "RdBu_r",
            "h4",
            symmetric=True,
        )
    )
    paths.append(
        plot_map_grid(
            study_dir,
            cfg,
            summaries,
            "CHI2_MAP",
            "degree_study_chi2_maps.png",
            "Chi2 Maps",
            "plasma",
            "chi2",
            symmetric=False,
        )
    )
    return paths


def parse_args(argv: list[str] | None = None) -> tuple[StudyConfig, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a 10-case degree/mdegree study for PHOENIX PowerBin LSF pPXF fits.",
    )
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE_PATH)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--core-script", type=Path, default=DEFAULT_CORE_SCRIPT)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--lsf-table-path", type=Path, default=DEFAULT_LSF_TABLE)
    parser.add_argument("--target-sn", type=float, default=120.0)
    parser.add_argument("--fit-windows-rest-um", type=str, default="2.1-2.4")
    parser.add_argument("--expected-template-count", type=int, default=0)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--n-plot-bins", type=int, default=0)
    parser.add_argument("--check-plot-radius-arcsec", type=float, default=0.5)
    parser.add_argument("--case-pairs", type=str, default=DEFAULT_CASE_PAIRS)
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help="Pass --overwrite-run to each core pPXF fit and regenerate existing case folders.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not launch pPXF. Only collate already completed case folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pPXF commands without running them.",
    )
    args, forward_args = parser.parse_known_args(argv)

    cfg = StudyConfig(
        cube_path=args.cube_path.expanduser().resolve(),
        study_dir=args.study_dir.expanduser().resolve(),
        core_script=args.core_script.expanduser().resolve(),
        python_executable=str(args.python_executable),
        lsf_table_path=args.lsf_table_path.expanduser().resolve(),
        target_sn=float(args.target_sn),
        fit_windows_rest_um=str(args.fit_windows_rest_um),
        expected_template_count=int(args.expected_template_count),
        n_processes=max(1, int(args.n_processes)),
        n_plot_bins=max(0, int(args.n_plot_bins)),
        check_plot_radius_arcsec=float(args.check_plot_radius_arcsec),
        cases=parse_case_pairs(args.case_pairs),
        overwrite_run=bool(args.overwrite_run),
        summarize_only=bool(args.summarize_only),
        dry_run=bool(args.dry_run),
    )
    return cfg, forward_args


def build_case_command(
    cfg: StudyConfig,
    case: DegreeCase,
    outdir: Path,
    forward_args: list[str],
) -> list[str]:
    cmd = [
        cfg.python_executable,
        "-u",
        str(cfg.core_script),
        "--cube-path",
        str(cfg.cube_path),
        "--output-dir",
        str(outdir),
        "--lsf-table-path",
        str(cfg.lsf_table_path),
        "--target-sn",
        f"{cfg.target_sn:.8g}",
        "--fit-windows-rest-um",
        cfg.fit_windows_rest_um,
        "--expected-template-count",
        str(cfg.expected_template_count),
        "--n-plot-bins",
        str(cfg.n_plot_bins),
        "--check-plot-radius-arcsec",
        f"{cfg.check_plot_radius_arcsec:.8g}",
        "--n-processes",
        str(cfg.n_processes),
        "--degree",
        str(case.degree),
        "--mdegree",
        str(case.mdegree),
    ]
    if cfg.overwrite_run:
        cmd.append("--overwrite-run")
    cmd.extend(forward_args)
    return cmd


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_LIMIT_ENV:
        env.setdefault(key, "1")
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_degree_study"))
    return env


def shell_join(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def main(argv: list[str] | None = None) -> None:
    cfg, forward_args = parse_args(argv)
    ensure_dir(cfg.study_dir)
    write_notes(cfg.study_dir / "degree_study_notes.txt", cfg, forward_args)

    study_config = {
        "cube_path": str(cfg.cube_path),
        "study_dir": str(cfg.study_dir),
        "core_script": str(cfg.core_script),
        "python_executable": cfg.python_executable,
        "lsf_table_path": str(cfg.lsf_table_path),
        "target_sn": cfg.target_sn,
        "fit_windows_rest_um": cfg.fit_windows_rest_um,
        "expected_template_count": cfg.expected_template_count,
        "n_processes": cfg.n_processes,
        "n_plot_bins": cfg.n_plot_bins,
        "check_plot_radius_arcsec": cfg.check_plot_radius_arcsec,
        "case_pairs": [asdict(case) for case in cfg.cases],
        "overwrite_run": cfg.overwrite_run,
        "summarize_only": cfg.summarize_only,
        "dry_run": cfg.dry_run,
        "forward_args": forward_args,
    }
    (cfg.study_dir / "degree_study_config.json").write_text(json.dumps(study_config, indent=2) + "\n")

    summaries: list[CaseSummary] = []
    env = subprocess_env()

    for index, case in enumerate(cfg.cases, start=1):
        outdir = ensure_dir(cfg.study_dir / case.label)
        is_done = completed_case(outdir, cfg.cube_path.stem, cfg.target_sn)
        cmd = build_case_command(cfg, case, outdir, forward_args)
        print(f"[degree-study] Case {index}/{len(cfg.cases)}: {case.short_label} -> {outdir}")

        if cfg.dry_run:
            print(shell_join(cmd))
            continue

        if cfg.summarize_only:
            if not is_done:
                raise FileNotFoundError(
                    f"Cannot summarize missing/incomplete case {case.label}. Expected products in {outdir}"
                )
        elif is_done and not cfg.overwrite_run:
            print(f"[degree-study] Reusing completed case {case.label}")
        else:
            subprocess.run(cmd, check=True, env=env)

        if completed_case(outdir, cfg.cube_path.stem, cfg.target_sn):
            summaries.append(summarize_case(outdir, cfg, case))
        else:
            raise FileNotFoundError(f"Core fit finished but expected products are missing for {case.label}")

    if cfg.dry_run:
        print("[degree-study] Dry run complete; no pPXF fits were launched.")
        return

    write_summary_csv(cfg.study_dir / "degree_study_summary.csv", summaries)
    write_summary_json(cfg.study_dir / "degree_study_summary.json", cfg, summaries, forward_args)
    write_text_summary(cfg.study_dir / "degree_study_summary.txt", summaries)
    plot_paths = make_plots(cfg.study_dir, cfg, summaries)

    print(f"[degree-study] Summary CSV : {cfg.study_dir / 'degree_study_summary.csv'}")
    print(f"[degree-study] Summary text: {cfg.study_dir / 'degree_study_summary.txt'}")
    for path in plot_paths:
        print(f"[degree-study] Plot        : {path}")


if __name__ == "__main__":
    main()
