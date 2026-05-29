#!/usr/bin/env python3
"""Stage the local NIRSpec overlay into the BAYES-LOSVD clone."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
WORKDIR = WORKSPACE / "bayeslosvd_nirspec"
DEFAULT_BAYES_ROOT = WORKDIR / "BAYES-LOSVD"
DEFAULT_CUBE = WORKSPACE / "Data/IFU/david_subs/g235h_agn_sub.fits"
CONFIG_DIR = WORKDIR / "config"

INSTRUMENT_BLOCK = """
[NIRSPEC-G235H]
read_file = 'NIRSPEC_G235H.py'
lsf_file  = 'NIRSPEC_G235H.lsf'
""".strip()


def copy_file(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        print(f"keep existing {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"staged {dst}")


def link_or_copy_cube(src: Path, dst: Path, copy_cube: bool, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            print(f"keep existing {dst}")
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_cube:
        shutil.copy2(src, dst)
        print(f"copied cube to {dst}")
    else:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
        print(f"linked cube {dst} -> {rel}")


def ensure_instrument_block(path: Path, force: bool) -> None:
    text = path.read_text()
    if "[NIRSPEC-G235H]" in text:
        print(f"instrument block already present in {path}")
        return
    backup = path.with_suffix(path.suffix + ".before_nirspec")
    if not backup.exists() or force:
        shutil.copy2(path, backup)
        print(f"backed up {path} to {backup}")
    path.write_text(text.rstrip() + "\n\n" + INSTRUMENT_BLOCK + "\n")
    print(f"added NIRSPEC-G235H block to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayes-root", type=Path, default=DEFAULT_BAYES_ROOT)
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--copy-cube", action="store_true", help="Copy the cube instead of symlinking it.")
    parser.add_argument("--force", action="store_true", help="Overwrite staged overlay files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bayes_root = args.bayes_root.expanduser().resolve()
    cube_path = args.cube_path.expanduser().resolve()

    if not bayes_root.exists():
        raise FileNotFoundError(f"BAYES-LOSVD clone not found: {bayes_root}")
    if not cube_path.exists():
        raise FileNotFoundError(f"AGN-subtracted cube not found: {cube_path}")

    link_or_copy_cube(
        cube_path,
        bayes_root / "data/g235h_agn_sub.fits",
        copy_cube=bool(args.copy_cube),
        force=bool(args.force),
    )

    copy_file(CONFIG_DIR / "NIRSPEC_G235H.py", bayes_root / "config_files/instruments/NIRSPEC_G235H.py", args.force)
    copy_file(CONFIG_DIR / "NIRSPEC_G235H.lsf", bayes_root / "config_files/instruments/NIRSPEC_G235H.lsf", args.force)
    copy_file(
        CONFIG_DIR / "PHOENIX_NIRSPEC_G235H.lsf",
        bayes_root / "config_files/instruments/PHOENIX_NIRSPEC_G235H.lsf",
        args.force,
    )
    copy_file(
        CONFIG_DIR / "nirspec_g235h_stellar.mask",
        bayes_root / "config_files/nirspec_g235h_stellar.mask",
        args.force,
    )
    copy_file(
        CONFIG_DIR / "nirspec_g235h_agn_sub_preproc.properties",
        bayes_root / "config_files/nirspec_g235h_agn_sub_preproc.properties",
        args.force,
    )
    ensure_instrument_block(bayes_root / "config_files/instruments.properties", args.force)

    print("")
    print("Ready. Next commands:")
    print(f"  cd {bayes_root / 'scripts'}")
    print("  python bayes_losvd_preproc_data.py -c ../config_files/nirspec_g235h_agn_sub_preproc.properties")
    print("  python bayes_losvd_run.py -f ../preproc_data/sombrero_nirspec_g235h_agn_sub.hdf5 -b 0 -t SP")


if __name__ == "__main__":
    main()
