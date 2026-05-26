#!/usr/bin/env python3
"""
Fixed-R PHOENIX pPXF NIRSpec fit.

This is the paper-reproduction version: it keeps the PHOENIX templates at the
same fitting setup as `ppxf_nirspec_phoenix_kinematics.py`, but makes the
constant NIRSpec resolving power explicit in the script name and defaults:

    PHOENIX R~100000 -> constant NIRSpec R=2700
    pPXF moments = [V, sigma, h3, h4]
    degree = 10, mdegree = 6, bias = 0, regul = 0
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import ppxf_nirspec_phoenix_kinematics as fixed_r


def main() -> None:
    if "--resolving-power" not in sys.argv:
        sys.argv.extend(["--resolving-power", "2700"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(
            [
                "--output-dir",
                str(fixed_r.ROOT / "Data/ppxf_nirspec/phoenix_g235h_constant_r2700"),
            ]
        )
    fixed_r.main()


if __name__ == "__main__":
    main()
