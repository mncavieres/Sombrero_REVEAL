#!/usr/bin/env python3
"""
Checkpoint watcher for the MUSE JAM constant-anisotropy run.

This is a thin wrapper around checkplot_muse.py with the constant-beta
configuration and output directory baked in.
"""

from __future__ import annotations

from pathlib import Path

import checkplot_muse as checkplot
import nested_free_muse as muse


def main() -> None:
    run_cfg = muse.Config()
    run_cfg.beta_prescription = "constant"
    run_cfg.output_dir = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/muse_constant_beta"
    )
    checkplot.run_cli(run_cfg)


if __name__ == "__main__":
    main()
