#!/usr/bin/env python3
"""
Plot the non-symmetrized NIRSpec and MUSE black-hole mass estimates together.
"""

from __future__ import annotations

from pathlib import Path

from compare_nirspec_muse_bh_mass import run_comparison


NIRSPEC_SUMMARY = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/"
    "no_symmetrization/nirspec_anisotropy_comparison/"
    "nirspec_anisotropy_comparison_summary.csv"
)
MUSE_SUMMARY = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/"
    "muse_anisotropy_comparison/muse_anisotropy_comparison_summary.csv"
)
OUTPUT_DIR = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/"
    "combined_bh_mass_comparison_no_symmetrization"
)


def main() -> None:
    run_comparison(
        nirspec_summary=NIRSPEC_SUMMARY,
        muse_summary=MUSE_SUMMARY,
        output_dir=OUTPUT_DIR,
        title="Sombrero black-hole mass comparison: non-symmetrized NIRSpec vs MUSE",
        plot_filename="nirspec_muse_bh_mass_comparison_no_sym.png",
        summary_filename="nirspec_muse_bh_mass_summary_no_sym.csv",
    )


if __name__ == "__main__":
    main()
