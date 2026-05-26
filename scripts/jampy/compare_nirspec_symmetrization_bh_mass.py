#!/usr/bin/env python3
"""
Compare NIRSpec black-hole mass constraints with and without kinematic symmetrization.

The comparison is done for the same three anisotropy prescriptions:
    - free beta
    - logistic beta
    - constant beta

For each run, the script uses the final NPZ result if present and otherwise
falls back to the current dynesty checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import logging
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from dynesty import DynamicNestedSampler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

NIRSPEC_JAM_DIR = Path(__file__).resolve().parents[1] / "nirspec_jam"
if str(NIRSPEC_JAM_DIR) not in sys.path:
    sys.path.insert(0, str(NIRSPEC_JAM_DIR))

import nested_free as base
import nested_free_nirspec as nirspec


@dataclass
class ModelSpec:
    name: str
    label: str
    color: str


@dataclass
class SymState:
    name: str
    label: str
    marker: str
    offset: float
    output_dirs: dict[str, Path]


@dataclass
class BHMassSummary:
    state: SymState
    model: ModelSpec
    source_kind: str
    source_path: Path
    best_bh_mass: float
    bh_mass_p16: float
    bh_mass_p50: float
    bh_mass_p84: float


MODELS = [
    ModelSpec(name="free", label="Free beta", color="tab:blue"),
    ModelSpec(name="logistic", label="Logistic beta", color="tab:orange"),
    ModelSpec(name="constant", label="Constant beta", color="tab:green"),
]

SYMMETRIZED = SymState(
    name="symmetrized",
    label="With symmetrization",
    marker="o",
    offset=-0.12,
    output_dirs={
        "free": Path("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/free_beta_2"),
        "logistic": Path("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/logistic_beta"),
        "constant": Path("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/free_ml_beta_bh2"),
    },
)

UNSYMMETRIZED = SymState(
    name="unsymmetrized",
    label="Without symmetrization",
    marker="s",
    offset=0.12,
    output_dirs={
        "free": nirspec.FREE_OUTPUT_DIR,
        "logistic": nirspec.LOGISTIC_OUTPUT_DIR,
        "constant": nirspec.CONSTANT_OUTPUT_DIR,
    },
)

OUT_DIR = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/nirspec_symmetrization_comparison"
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def restore_checkpoint_safely(checkpoint_path: Path, work_dir: Path):
    with tempfile.NamedTemporaryFile(
        prefix="dynesty_compare_",
        suffix=".save",
        dir=work_dir,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        shutil.copy2(checkpoint_path, tmp_path)
        sampler = DynamicNestedSampler.restore(str(tmp_path))
        return sampler.results
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def weighted_quantile(values: np.ndarray, quantiles: list[float], weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.full_like(quantiles, np.nan, dtype=float)

    values = values[good]
    weights = weights[good]
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, values)


def posterior_bh_quantiles(samples: np.ndarray, logwt: np.ndarray, logz: np.ndarray) -> tuple[float, float, float]:
    samples = np.asarray(samples, dtype=float)
    logwt = np.asarray(logwt, dtype=float)
    logz = np.asarray(logz, dtype=float)

    log_weights = logwt - float(logz[-1])
    log_weights -= np.nanmax(log_weights)
    weights = np.exp(log_weights)

    q16, q50, q84 = weighted_quantile(samples[:, 0], [0.15865, 0.5, 0.84135], weights)
    return float(q16), float(q50), float(q84)


def load_model_state_from_npz(results_path: Path):
    with np.load(results_path, allow_pickle=True) as data:
        best_params = np.asarray(data["best_params"], dtype=float)
        samples = np.asarray(data["samples"], dtype=float)
        logwt = np.asarray(data["logwt"], dtype=float)
        logz = np.asarray(data["logz"], dtype=float)
    return best_params, samples, logwt, logz


def load_model_state_from_checkpoint(checkpoint_path: Path, work_dir: Path):
    results = restore_checkpoint_safely(checkpoint_path, work_dir)
    if len(results.samples) == 0:
        raise RuntimeError(f"Checkpoint contains zero samples: {checkpoint_path}")
    best_params = np.asarray(base.get_best_fit_parameters(results), dtype=float)
    samples = np.asarray(results.samples, dtype=float)
    logwt = np.asarray(results.logwt, dtype=float)
    logz = np.asarray(results.logz, dtype=float)
    return best_params, samples, logwt, logz


def resolve_model_state(output_dir: Path, work_dir: Path):
    results_npz = output_dir / "nested_bh_beta_ml_results.npz"
    if results_npz.exists():
        best_params, samples, logwt, logz = load_model_state_from_npz(results_npz)
        return best_params, samples, logwt, logz, "results_npz", results_npz

    checkpoint = output_dir / base.Config().checkpoint_filename
    if checkpoint.exists():
        best_params, samples, logwt, logz = load_model_state_from_checkpoint(checkpoint, work_dir)
        return best_params, samples, logwt, logz, "checkpoint", checkpoint

    raise FileNotFoundError(f"No final results NPZ or checkpoint found in {output_dir}")


def load_bh_mass_summary(state: SymState, model: ModelSpec, work_dir: Path) -> BHMassSummary:
    best_params, samples, logwt, logz, source_kind, source_path = resolve_model_state(
        state.output_dirs[model.name],
        work_dir,
    )
    q16, q50, q84 = posterior_bh_quantiles(samples, logwt, logz)

    return BHMassSummary(
        state=state,
        model=model,
        source_kind=source_kind,
        source_path=source_path,
        best_bh_mass=float(best_params[0]),
        bh_mass_p16=q16,
        bh_mass_p50=q50,
        bh_mass_p84=q84,
    )


def save_summary_csv(rows: list[BHMassSummary], out_dir: Path) -> None:
    outpath = out_dir / "nirspec_symmetrization_bh_mass_summary.csv"

    by_key = {(row.state.name, row.model.name): row for row in rows}

    with outpath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_name",
                "model_label",
                "sym_source_kind",
                "sym_source_path",
                "sym_best_bh_mass",
                "sym_bh_mass_p16",
                "sym_bh_mass_p50",
                "sym_bh_mass_p84",
                "no_sym_source_kind",
                "no_sym_source_path",
                "no_sym_best_bh_mass",
                "no_sym_bh_mass_p16",
                "no_sym_bh_mass_p50",
                "no_sym_bh_mass_p84",
                "best_mass_ratio_no_sym_over_sym",
                "median_mass_ratio_no_sym_over_sym",
            ]
        )

        for model in MODELS:
            sym = by_key[(SYMMETRIZED.name, model.name)]
            no_sym = by_key[(UNSYMMETRIZED.name, model.name)]
            writer.writerow(
                [
                    model.name,
                    model.label,
                    sym.source_kind,
                    str(sym.source_path),
                    f"{sym.best_bh_mass:.12e}",
                    f"{sym.bh_mass_p16:.12e}",
                    f"{sym.bh_mass_p50:.12e}",
                    f"{sym.bh_mass_p84:.12e}",
                    no_sym.source_kind,
                    str(no_sym.source_path),
                    f"{no_sym.best_bh_mass:.12e}",
                    f"{no_sym.bh_mass_p16:.12e}",
                    f"{no_sym.bh_mass_p50:.12e}",
                    f"{no_sym.bh_mass_p84:.12e}",
                    f"{(no_sym.best_bh_mass / sym.best_bh_mass):.12e}",
                    f"{(no_sym.bh_mass_p50 / sym.bh_mass_p50):.12e}",
                ]
            )


def save_plot(rows: list[BHMassSummary], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    xpos = np.arange(len(MODELS), dtype=float)
    by_key = {(row.state.name, row.model.name): row for row in rows}

    for model_idx, model in enumerate(MODELS):
        sym = by_key[(SYMMETRIZED.name, model.name)]
        no_sym = by_key[(UNSYMMETRIZED.name, model.name)]

        ax.plot(
            [xpos[model_idx] + SYMMETRIZED.offset, xpos[model_idx] + UNSYMMETRIZED.offset],
            [sym.bh_mass_p50, no_sym.bh_mass_p50],
            color=model.color,
            alpha=0.4,
            linewidth=1.4,
            zorder=1,
        )

    for row in rows:
        model_idx = next(idx for idx, model in enumerate(MODELS) if model.name == row.model.name)
        x = xpos[model_idx] + row.state.offset
        yerr = np.array(
            [
                [row.bh_mass_p50 - row.bh_mass_p16],
                [row.bh_mass_p84 - row.bh_mass_p50],
            ],
            dtype=float,
        )
        ax.errorbar(
            x,
            row.bh_mass_p50,
            yerr=yerr,
            fmt=row.state.marker,
            ms=8.5,
            lw=1.8,
            capsize=4.5,
            color=row.model.color,
            markerfacecolor=row.model.color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.scatter(
            x,
            row.best_bh_mass,
            marker="x",
            s=52,
            color="black",
            linewidths=1.5,
            zorder=4,
        )

    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_xticks(xpos, [model.label for model in MODELS], rotation=18)
    ax.set_ylabel(r"$M_{\rm BH}$ ($M_\odot$)")
    ax.set_title("NIRSpec black-hole mass: with vs without symmetrization")
    ax.grid(axis="y", which="both", alpha=0.22)

    state_handles = [
        Line2D(
            [0],
            [0],
            marker=SYMMETRIZED.marker,
            color="black",
            markerfacecolor="white",
            markersize=8,
            linewidth=0,
            label=f"{SYMMETRIZED.label} median + 16-84%",
        ),
        Line2D(
            [0],
            [0],
            marker=UNSYMMETRIZED.marker,
            color="black",
            markerfacecolor="white",
            markersize=8,
            linewidth=0,
            label=f"{UNSYMMETRIZED.label} median + 16-84%",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            markersize=8,
            linewidth=0,
            label="Best fit",
        ),
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=model.color,
            markerfacecolor=model.color,
            markersize=8,
            linewidth=0,
            label=model.label,
        )
        for model in MODELS
    ]

    legend_left = ax.legend(handles=state_handles, loc="upper left", frameon=False, title="Marker")
    ax.add_artist(legend_left)
    ax.legend(handles=model_handles, loc="upper right", frameon=False, title="Color")

    fig.savefig(out_dir / "nirspec_bh_mass_symmetrization_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[BHMassSummary] = []
    for state in (SYMMETRIZED, UNSYMMETRIZED):
        for model in MODELS:
            logging.info("Loading %s / %s", state.label, model.label)
            rows.append(load_bh_mass_summary(state, model, OUT_DIR))

    save_summary_csv(rows, OUT_DIR)
    save_plot(rows, OUT_DIR)
    logging.info("Wrote NIRSpec symmetrization comparison to %s", OUT_DIR)


if __name__ == "__main__":
    main()
