#!/usr/bin/env python3
"""
Run the NIRSpec JAM dynesty fit with free anisotropy.

This is a thin wrapper around the common NIRSpec pipeline in
`scripts/jampy/nested_free.py`, analogous to the MUSE wrapper layout in
`scripts/muse_jam/`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys


JAMPY_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "jampy"
if str(JAMPY_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(JAMPY_SCRIPT_DIR))

import nested_free as base


Config = base.Config

NO_SYM_OUTPUT_ROOT = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/no_symmetrization"
)
FREE_OUTPUT_DIR = Path(
    NO_SYM_OUTPUT_ROOT / "nirspec_free_beta"
)
LOGISTIC_OUTPUT_DIR = Path(
    NO_SYM_OUTPUT_ROOT / "nirspec_logistic_beta"
)
CONSTANT_OUTPUT_DIR = Path(
    NO_SYM_OUTPUT_ROOT / "nirspec_constant_beta"
)


def make_free_config() -> Config:
    return replace(
        base.Config(),
        output_dir=FREE_OUTPUT_DIR,
        beta_prescription="free",
        sample_method="rwalk",
        bound_method="multi",
        walks=32,
        bootstrap=20,
    )


def make_logistic_config() -> Config:
    return replace(
        base.Config(),
        output_dir=LOGISTIC_OUTPUT_DIR,
        beta_prescription="logistic",
        sample_method="rwalk",
        bound_method="multi",
        walks=32,
        bootstrap=20,
    )


def make_constant_config() -> Config:
    return replace(
        base.Config(),
        output_dir=CONSTANT_OUTPUT_DIR,
        beta_prescription="constant",
        sample_method="rwalk",
        bound_method="multi",
        walks=32,
        bootstrap=20,
    )


def run_with_config(cfg: Config) -> None:
    base.run_with_config(cfg)


def main() -> None:
    base.setup_logging()
    run_with_config(make_free_config())


if __name__ == "__main__":
    main()
