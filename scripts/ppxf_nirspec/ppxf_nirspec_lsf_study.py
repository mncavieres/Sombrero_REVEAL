#!/usr/bin/env python3
"""
Run a fixed-LSF sensitivity study for the NIRSpec stellar-kinematics pPXF workflow.

This wrapper launches the main NIRSpec pPXF script multiple times with different
fixed resolving powers, then collates the outputs into a compact summary table
and comparison plots.

Default assumption for the first-pass study:
    - JWST/NIRSpec G235H/F170LP covers about 1.7-3.1 um
    - published high-resolution range is about R ~ 1900-3600
    - we sample five fixed resolving powers across that bracket:
      1900, 2300, 2700, 3150, 3600

This is still a simplified approximation to the true wavelength-dependent LSF.
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
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_lsf_study"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


C = 299792.458  # km/s
GAUSS_FWHM_PER_SIGMA = 2.35482004503

DEFAULT_CUBE_PATH = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/david_subs/g235h_agn_sub.fits"
)
DEFAULT_STUDY_DIR = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/ppxf_nirspec/agn_substracted_david_lsf_study"
)
DEFAULT_CORE_SCRIPT = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/scripts/ppxf_nirspec/ppxf_nirspec_stellar_kinematics.py"
)


@dataclass(frozen=True)
class StudyConfig:
    cube_path: Path
    study_dir: Path
    core_script: Path
    python_executable: str
    resolving_powers: tuple[float, ...]
    n_plot_spaxels: int


@dataclass(frozen=True)
class CaseSummary:
    label: str
    output_dir: str
    resolving_power: float
    sigma_inst_kms: float
    sigma_template_kms: float
    sigma_template_eff_kms: float
    template_r_eff: float
    template_broader_than_data: bool
    n_total: int
    n_goodfit: int
    median_losv: float
    median_vrel_kms: float
    median_sigma_ppxf: float
    median_sigma: float
    median_vrms: float
    median_h3: float
    median_h4: float
    median_sn: float
    median_chi2: float
    systemic_losv_fit: float
    fitted_systemic_redshift: float


def parse_resolving_powers(text: str) -> tuple[float, ...]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        val = float(chunk)
        if val <= 0:
            raise ValueError(f"Resolving power must be positive, got {val}")
        values.append(val)
    if not values:
        raise ValueError("Need at least one resolving power")
    return tuple(values)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sigma_inst_from_r(resolving_power: float) -> float:
    return float(C / resolving_power / GAUSS_FWHM_PER_SIGMA)


def case_label(resolving_power: float) -> str:
    sigma_inst = sigma_inst_from_r(resolving_power)
    return f"R{int(round(resolving_power)):04d}_siginst_{sigma_inst:05.2f}"


def summarize_case(outdir: Path, cube_stem: str) -> CaseSummary:
    fits_path = outdir / f"{cube_stem}_stellar_kinematics.fits"
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        tab = hdul["KIN_RESULTS"].data
        total = int(len(tab))
        good = np.asarray(tab["GOODFIT"], dtype=bool)
        if np.any(good):
            sel = good
        else:
            sel = np.ones(total, dtype=bool)

        def med(key: str) -> float:
            arr = np.asarray(tab[key], dtype=float)
            return float(np.nanmedian(arr[sel])) if arr.size else np.nan

        return CaseSummary(
            label=outdir.name,
            output_dir=str(outdir),
            resolving_power=float(hdr["RPOWER"]),
            sigma_inst_kms=float(hdr["SIGINST"]),
            sigma_template_kms=float(hdr["SIGTEMP"]),
            sigma_template_eff_kms=float(hdr["SIGTEFF"]),
            template_r_eff=float(hdr["RTEMP"]),
            template_broader_than_data=bool(hdr.get("TBROADD", hdr.get("TBROAD", False))),
            n_total=total,
            n_goodfit=int(np.count_nonzero(good)),
            median_losv=med("LOSV"),
            median_vrel_kms=med("V_REL_KMS"),
            median_sigma_ppxf=med("SIGMA_PPXF"),
            median_sigma=med("SIGMA"),
            median_vrms=med("VRMS"),
            median_h3=med("H3"),
            median_h4=med("H4"),
            median_sn=med("SN"),
            median_chi2=med("CHI2"),
            systemic_losv_fit=float(parse_summary_value(outdir / "run_summary.txt", "Fitted systemic LOSV")),
            fitted_systemic_redshift=float(parse_summary_value(outdir / "run_summary.txt", "Fitted systemic redshift")),
        )


def parse_summary_value(path: Path, prefix: str) -> str:
    for line in path.read_text().splitlines():
        if line.startswith(prefix + ":"):
            return line.split(":", maxsplit=1)[1].strip().split()[0]
    raise KeyError(f"Could not find '{prefix}' in {path}")


def plot_summary(study_dir: Path, cases: list[CaseSummary]) -> Path:
    r = np.array([c.resolving_power for c in cases], dtype=float)
    sig_inst = np.array([c.sigma_inst_kms for c in cases], dtype=float)
    sig_raw = np.array([c.median_sigma_ppxf for c in cases], dtype=float)
    sig_corr = np.array([c.median_sigma for c in cases], dtype=float)
    vrms = np.array([c.median_vrms for c in cases], dtype=float)
    sn = np.array([c.median_sn for c in cases], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(r, sig_raw, marker="o", lw=1.7, label="median sigma_ppxf")
    ax.plot(r, sig_corr, marker="s", lw=1.7, label="median sigma")
    ax.set_xlabel("Assumed resolving power R")
    ax.set_ylabel("Median sigma [km/s]")
    ax.set_title("Sigma Sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(r, vrms, marker="o", lw=1.7, color="tab:purple")
    ax.set_xlabel("Assumed resolving power R")
    ax.set_ylabel("Median vrms [km/s]")
    ax.set_title("Vrms Sensitivity")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(sig_inst, sig_corr, marker="o", lw=1.7, color="tab:red")
    for x, y, rr in zip(sig_inst, sig_corr, r):
        ax.annotate(f"R={int(round(rr))}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Assumed instrumental sigma [km/s]")
    ax.set_ylabel("Median sigma [km/s]")
    ax.set_title("Median Sigma vs Instrumental Sigma")
    ax.grid(alpha=0.25)

    outpath = study_dir / "lsf_study_summary.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ax.plot(r, sn, marker="o", lw=1.7, color="tab:green")
    ax.set_xlabel("Assumed resolving power R")
    ax.set_ylabel("Median S/N")
    ax.set_title("Median S/N Stability")
    ax.grid(alpha=0.25)
    outpath2 = study_dir / "lsf_study_sn_check.png"
    fig.savefig(outpath2, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return outpath


def write_summary_csv(path: Path, cases: list[CaseSummary]) -> None:
    rows = [asdict(case) for case in cases]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_notes(path: Path, cfg: StudyConfig) -> None:
    lines = [
        "Fixed-LSF NIRSpec sensitivity study",
        "",
        "Assumptions:",
        "- This is a first-order approximation using constant resolving power in each run.",
        "- The true NIRSpec G235H LSF is wavelength-dependent and more complicated.",
        "- Five fixed resolving powers were chosen to span the published G235H/F170LP high-resolution range.",
        "",
        "Reference notes:",
        "- STScI JWST User Documentation lists G235H/F170LP as a high-resolution mode with nominal R ~ 2700 over about 1.66-3.05 um.",
        "- ESA/COSMOS NIRSpec documentation summarizes the G235H/F170LP high-resolution range as about R = 1900-3600 over about 1.7-3.1 um.",
        "",
        "Study configuration:",
        f"- Cube: {cfg.cube_path}",
        f"- Core script: {cfg.core_script}",
        f"- Python executable: {cfg.python_executable}",
        f"- Resolving powers: {', '.join(f'{r:.1f}' for r in cfg.resolving_powers)}",
        "",
        "Source URLs used when setting the study bracket:",
        "- https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters",
        "- https://www.cosmos.esa.int/web/jwst-nirspec/exoplanets",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_summary_text(path: Path, cases: list[CaseSummary]) -> None:
    ref = next((case for case in cases if int(round(case.resolving_power)) == 2700), cases[len(cases) // 2])
    sigma_vals = np.array([case.median_sigma for case in cases], dtype=float)
    vrms_vals = np.array([case.median_vrms for case in cases], dtype=float)
    lines = [
        "LSF sensitivity summary",
        "",
        f"Runs completed: {len(cases)}",
        f"Reference run: {ref.label}",
        f"Median sigma span: {float(np.nanmin(sigma_vals)):.2f} to {float(np.nanmax(sigma_vals)):.2f} km/s",
        f"Median sigma peak-to-peak: {float(np.nanmax(sigma_vals) - np.nanmin(sigma_vals)):.2f} km/s",
        f"Median vrms span: {float(np.nanmin(vrms_vals)):.2f} to {float(np.nanmax(vrms_vals)):.2f} km/s",
        f"Median vrms peak-to-peak: {float(np.nanmax(vrms_vals) - np.nanmin(vrms_vals)):.2f} km/s",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> tuple[StudyConfig, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a 5-case fixed-LSF sensitivity study for NIRSpec pPXF fits",
    )
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE_PATH)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--core-script", type=Path, default=DEFAULT_CORE_SCRIPT)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--resolving-powers", type=str, default="1900,2300,2700,3150,3600")
    parser.add_argument("--n-plot-spaxels", type=int, default=0)
    args, forward_args = parser.parse_known_args(argv)
    cfg = StudyConfig(
        cube_path=args.cube_path.resolve(),
        study_dir=args.study_dir.resolve(),
        core_script=args.core_script.resolve(),
        python_executable=args.python_executable,
        resolving_powers=parse_resolving_powers(args.resolving_powers),
        n_plot_spaxels=int(args.n_plot_spaxels),
    )
    return cfg, forward_args


def main(argv: list[str] | None = None) -> None:
    cfg, forward_args = parse_args(argv)
    ensure_dir(cfg.study_dir)

    study_config = {
        "cube_path": str(cfg.cube_path),
        "study_dir": str(cfg.study_dir),
        "core_script": str(cfg.core_script),
        "python_executable": cfg.python_executable,
        "resolving_powers": list(cfg.resolving_powers),
        "n_plot_spaxels": cfg.n_plot_spaxels,
        "forward_args": forward_args,
    }
    (cfg.study_dir / "study_config.json").write_text(json.dumps(study_config, indent=2))
    write_notes(cfg.study_dir / "study_notes.txt", cfg)

    cases: list[CaseSummary] = []
    for resolving_power in cfg.resolving_powers:
        label = case_label(resolving_power)
        outdir = ensure_dir(cfg.study_dir / label)
        cmd = [
            cfg.python_executable,
            str(cfg.core_script),
            "--cube-path",
            str(cfg.cube_path),
            "--output-dir",
            str(outdir),
            "--resolving-power",
            f"{resolving_power:.8f}",
            "--n-plot-spaxels",
            str(cfg.n_plot_spaxels),
        ] + forward_args
        print(f"[lsf-study] Running {label}")
        subprocess.run(cmd, check=True)
        cases.append(summarize_case(outdir, cfg.cube_path.stem))

    write_summary_csv(cfg.study_dir / "lsf_study_summary.csv", cases)
    write_summary_text(cfg.study_dir / "lsf_study_summary.txt", cases)
    plot_summary(cfg.study_dir, cases)
    print(f"[lsf-study] Summary CSV : {cfg.study_dir / 'lsf_study_summary.csv'}")
    print(f"[lsf-study] Summary plot: {cfg.study_dir / 'lsf_study_summary.png'}")


if __name__ == "__main__":
    main()
