#!/usr/bin/env python3
"""
Checkpoint watcher for the MUSE JAM logistic-anisotropy run.

This is a thin wrapper around checkplot_muse.py with the logistic-run
configuration and output directory baked in.
"""

from __future__ import annotations

from pathlib import Path

import checkplot_muse as checkplot
import nested_free_muse as muse


def main() -> None:
    run_cfg = muse.Config()
    run_cfg.beta_prescription = "logistic"
    run_cfg.output_dir = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/muse_logistic_beta_rwalk"
    )
    run_cfg.sample_method = "rwalk"
    run_cfg.bound_method = "multi"
    run_cfg.walks = 32
    run_cfg.bootstrap = 20
    run_cfg.beta_min = -4.0
    run_cfg.beta_max = 0.99
    run_cfg.beta_ra_min = 0.2
    run_cfg.beta_ra_max = 30.0
    run_cfg.beta_alpha_min = 0.5
    run_cfg.beta_alpha_max = 5.0
    checkplot.run_cli(run_cfg)


if __name__ == "__main__":
    main()
