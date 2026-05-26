#!/usr/bin/env python3
"""
Checkpoint watcher for the NIRSpec JAM logistic-anisotropy run.

This is a thin wrapper around checkplot_nirspec.py with the logistic-run
configuration and output directory baked in.
"""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import checkplot_nirspec as checkplot
import nested_free_nirspec as nirspec


def main() -> None:
    checkplot.run_cli(nirspec.make_logistic_config())


if __name__ == "__main__":
    main()
