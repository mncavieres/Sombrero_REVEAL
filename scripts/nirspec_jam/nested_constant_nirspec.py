#!/usr/bin/env python3
"""
Run the NIRSpec JAM dynesty fit with a single constant anisotropy beta.

This reuses the common NIRSpec pipeline in nested_free_nirspec.py, changing
only the anisotropy prescription and output directory.
"""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nested_free_nirspec as nirspec


def main() -> None:
    nirspec.base.setup_logging()
    nirspec.run_with_config(nirspec.make_constant_config())


if __name__ == "__main__":
    main()
