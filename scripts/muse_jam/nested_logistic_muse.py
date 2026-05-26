"""
Run the MUSE JAM dynesty fit with the logistic anisotropy prescription.

This reuses the common MUSE pipeline in nested_free_muse.py, changing only:
    - beta_prescription = "logistic"
    - output directory for the logistic run products
"""

from __future__ import annotations

from pathlib import Path

import nested_free_muse as muse


def main() -> None:
    muse.base.setup_logging()

    cfg = muse.Config()
    cfg.beta_prescription = "logistic"
    cfg.output_dir = Path(
        "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/muse_logistic_beta_rwalk"
    )
    cfg.sample_method = "rwalk"
    cfg.bound_method = "multi"
    cfg.walks = 32
    cfg.bootstrap = 20
    cfg.beta_min = -4.0
    cfg.beta_max = 0.99
    cfg.beta_ra_min = 0.2
    cfg.beta_ra_max = 30.0
    cfg.beta_alpha_min = 0.5
    cfg.beta_alpha_max = 5.0

    muse.run_with_config(cfg)


if __name__ == "__main__":
    main()
