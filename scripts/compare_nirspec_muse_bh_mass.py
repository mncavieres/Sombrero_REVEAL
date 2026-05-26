#!/usr/bin/env python3
"""
Plot the NIRSpec and MUSE black-hole mass estimates on a single figure.

The script reads the summary CSVs produced by:
    - scripts/jampy/compare_nirspec_anisotropy_models.py
    - scripts/muse_jam/compare_muse_anisotropy_models.py

It then makes a combined black-hole mass comparison where:
    - color identifies the anisotropy prescription,
    - marker shape identifies the instrument,
    - the Jardel et al. (2011) value is shown as a horizontal reference band.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_NIRSPEC_SUMMARY = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/"
    "nirspec_anisotropy_comparison/nirspec_anisotropy_comparison_summary.csv"
)
DEFAULT_MUSE_SUMMARY = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/"
    "muse_anisotropy_comparison/muse_anisotropy_comparison_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/"
    "combined_bh_mass_comparison"
)
DEFAULT_PLOT_FILENAME = "nirspec_muse_bh_mass_comparison.png"
DEFAULT_SUMMARY_FILENAME = "nirspec_muse_bh_mass_summary.csv"
DEFAULT_TITLE = "Sombrero black-hole mass comparison: NIRSpec vs MUSE"

# Jardel et al. (2011), ApJ 739, 21, abstract:
# M_BH = (6.6 +/- 0.4) x 10^8 Msun
JARDEL_BH_MASS = 6.6e8
JARDEL_BH_MASS_ERR = 0.4e8
JARDEL_SOURCE = "Jardel et al. 2011, ApJ, 739, 21"

MODEL_ORDER = ["free", "logistic", "constant"]
MODEL_LABELS = {
    "free": "Free beta",
    "logistic": "Logistic beta",
    "constant": "Constant beta",
}
MODEL_COLORS = {
    "free": "tab:blue",
    "logistic": "tab:orange",
    "constant": "tab:green",
}
INSTRUMENT_STYLES = {
    "NIRSpec": {"marker": "o", "offset": -0.13},
    "MUSE": {"marker": "s", "offset": 0.13},
}


@dataclass
class SummaryRow:
    instrument: str
    model_name: str
    model_label: str
    best_bh_mass: float
    bh_mass_p16: float
    bh_mass_p50: float
    bh_mass_p84: float
    reduced_chi2: float


def load_summary(path: Path, instrument: str) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            rows.append(
                SummaryRow(
                    instrument=instrument,
                    model_name=item["model_name"],
                    model_label=item["model_label"],
                    best_bh_mass=float(item["best_bh_mass"]),
                    bh_mass_p16=float(item["bh_mass_p16"]),
                    bh_mass_p50=float(item["bh_mass_p50"]),
                    bh_mass_p84=float(item["bh_mass_p84"]),
                    reduced_chi2=float(item["reduced_chi2"]),
                )
            )
    return rows


def save_combined_summary(rows: list[SummaryRow], out_dir: Path, summary_filename: str) -> None:
    outpath = out_dir / summary_filename
    with outpath.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instrument",
                "model_name",
                "model_label",
                "best_bh_mass",
                "bh_mass_p16",
                "bh_mass_p50",
                "bh_mass_p84",
                "reduced_chi2",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.instrument,
                    row.model_name,
                    row.model_label,
                    f"{row.best_bh_mass:.12e}",
                    f"{row.bh_mass_p16:.12e}",
                    f"{row.bh_mass_p50:.12e}",
                    f"{row.bh_mass_p84:.12e}",
                    f"{row.reduced_chi2:.12e}",
                ]
            )
        writer.writerow([])
        writer.writerow(["literature_source", JARDEL_SOURCE])
        writer.writerow(["literature_bh_mass", f"{JARDEL_BH_MASS:.12e}"])
        writer.writerow(["literature_bh_mass_err", f"{JARDEL_BH_MASS_ERR:.12e}"])


def make_plot(rows: list[SummaryRow], out_dir: Path, *, title: str, plot_filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)

    xpos_base = np.arange(len(MODEL_ORDER), dtype=float)

    ax.axhspan(
        JARDEL_BH_MASS - JARDEL_BH_MASS_ERR,
        JARDEL_BH_MASS + JARDEL_BH_MASS_ERR,
        color="0.88",
        alpha=1.0,
        zorder=0,
    )
    ax.axhline(
        JARDEL_BH_MASS,
        color="black",
        linestyle="--",
        linewidth=1.6,
        zorder=1,
    )
    ax.text(
        0.99,
        JARDEL_BH_MASS * 1.03,
        "Jardel+2011",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color="black",
    )

    for row in rows:
        if row.model_name not in MODEL_ORDER:
            continue
        base_x = float(MODEL_ORDER.index(row.model_name))
        style = INSTRUMENT_STYLES[row.instrument]
        x = base_x + style["offset"]
        color = MODEL_COLORS[row.model_name]
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
            fmt=style["marker"],
            ms=8.5,
            lw=1.8,
            capsize=4.5,
            color=color,
            markerfacecolor=color,
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
    ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
    ax.set_xticks(xpos_base, [MODEL_LABELS[name] for name in MODEL_ORDER], rotation=18)
    ax.set_ylabel(r"$M_{\rm BH}$ ($M_\odot$)")
    ax.set_title(title)
    ax.grid(axis="y", which="both", alpha=0.22)

    instrument_handles = [
        Line2D(
            [0],
            [0],
            marker=INSTRUMENT_STYLES["NIRSpec"]["marker"],
            color="black",
            markerfacecolor="white",
            markersize=8,
            linewidth=0,
            label="NIRSpec median + 16-84%",
        ),
        Line2D(
            [0],
            [0],
            marker=INSTRUMENT_STYLES["MUSE"]["marker"],
            color="black",
            markerfacecolor="white",
            markersize=8,
            linewidth=0,
            label="MUSE median + 16-84%",
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
            color=MODEL_COLORS[name],
            markerfacecolor=MODEL_COLORS[name],
            markersize=8,
            linewidth=0,
            label=MODEL_LABELS[name],
        )
        for name in MODEL_ORDER
    ]
    jardel_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=r"Jardel+2011: $6.6 \pm 0.4 \times 10^8\,M_\odot$",
    )

    legend_left = ax.legend(handles=instrument_handles, loc="upper left", frameon=False, title="Marker")
    ax.add_artist(legend_left)
    ax.legend(handles=model_handles + [jardel_handle], loc="upper right", frameon=False, title="Color / reference")

    fig.savefig(out_dir / plot_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_comparison(
    *,
    nirspec_summary: Path = DEFAULT_NIRSPEC_SUMMARY,
    muse_summary: Path = DEFAULT_MUSE_SUMMARY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    title: str = DEFAULT_TITLE,
    plot_filename: str = DEFAULT_PLOT_FILENAME,
    summary_filename: str = DEFAULT_SUMMARY_FILENAME,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(load_summary(nirspec_summary, "NIRSpec"))
    rows.extend(load_summary(muse_summary, "MUSE"))

    save_combined_summary(rows, output_dir, summary_filename)
    make_plot(rows, output_dir, title=title, plot_filename=plot_filename)


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
