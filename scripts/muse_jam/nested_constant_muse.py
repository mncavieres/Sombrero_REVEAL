"""
Run the MUSE JAM dynesty fit with a single constant anisotropy beta.

This reuses the common MUSE pipeline in nested_free_muse.py, changing only:
    - beta_prescription = "constant"
    - output directory for the constant-beta run products
"""

from __future__ import annotations

from pathlib import Path

import nested_free_muse as muse


def main() -> None:
    muse.base.setup_logging()

    cfg = muse.Config()
    cfg.beta_prescription = "constant"
    cfg.output_dir = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/muse_constant_beta"
    )

    muse.run_with_config(cfg)


if __name__ == "__main__":
    main()
