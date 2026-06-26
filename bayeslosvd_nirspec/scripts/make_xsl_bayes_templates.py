#!/usr/bin/env python3
"""Convert pPXF XSL SPS templates into BAYES-LOSVD template FITS files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from reproducibility import write_reproduction_files


WORKSPACE = Path(__file__).resolve().parents[2]
WORKDIR = WORKSPACE / "bayeslosvd_nirspec"
DEFAULT_BAYES_ROOT = WORKDIR / "BAYES-LOSVD"
DEFAULT_XSL_PATH = Path("/opt/miniconda3/envs/ppxf/lib/python3.14/site-packages/ppxf/sps_models/spectra_xsl_9.0.npz")


@dataclass(frozen=True)
class XslTemplate:
    index: int
    age_gyr: float
    metal: float
    age_index: int
    metal_index: int


def output_grid(lmin: float, lmax: float, step: float) -> np.ndarray:
    if lmax <= lmin:
        raise ValueError("lmax must be greater than lmin")
    n = int(np.floor((lmax - lmin) / step)) + 1
    return lmin + step * np.arange(n, dtype=float)


def write_template(outpath: Path, wave_out: np.ndarray, flux: np.ndarray, meta: XslTemplate) -> None:
    hdr = fits.Header()
    hdr["CRVAL1"] = float(wave_out[0])
    hdr["CDELT1"] = float(wave_out[1] - wave_out[0])
    hdr["CRPIX1"] = 1.0
    hdr["CTYPE1"] = "WAVE"
    hdr["CUNIT1"] = "Angstrom"
    hdr["AGE_GYR"] = float(meta.age_gyr)
    hdr["METAL"] = float(meta.metal)
    hdr["AGEIDX"] = int(meta.age_index)
    hdr["METIDX"] = int(meta.metal_index)
    fits.PrimaryHDU(np.asarray(flux, dtype=np.float32), header=hdr).writeto(outpath, overwrite=True)


def write_lsf(lsf_path: Path, wave: np.ndarray, fwhm: np.ndarray, lmin: float, lmax: float) -> None:
    use = (wave >= lmin) & (wave <= lmax) & np.isfinite(fwhm)
    if np.count_nonzero(use) < 2:
        raise ValueError("XSL FWHM grid does not overlap requested wavelength range.")
    # Keep the LSF table compact but still wavelength dependent.
    sample = np.linspace(np.flatnonzero(use)[0], np.flatnonzero(use)[-1], 200, dtype=int)
    rows = zip(wave[sample], fwhm[sample])
    lsf_path.parent.mkdir(parents=True, exist_ok=True)
    with lsf_path.open("w", encoding="utf-8") as handle:
        handle.write("# Lambda  FWHM\n")
        handle.write("# XSL SPS native FWHM in Angstrom, sampled from spectra_xsl_9.0.npz.\n")
        for lam, width in rows:
            handle.write(f"{lam:.6f}  {width:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayes-root", type=Path, default=DEFAULT_BAYES_ROOT)
    parser.add_argument("--xsl-path", type=Path, default=DEFAULT_XSL_PATH)
    parser.add_argument("--template-lib", type=str, default="XSL_NIRSPEC_G235H")
    parser.add_argument("--lmin", type=float, default=20500.0)
    parser.add_argument("--lmax", type=float, default=24500.0)
    parser.add_argument("--linear-step", type=float, default=0.5)
    parser.add_argument("--age-min", type=float, default=None)
    parser.add_argument("--age-max", type=float, default=None)
    parser.add_argument("--metal-min", type=float, default=None)
    parser.add_argument("--metal-max", type=float, default=None)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.bayes_root = args.bayes_root.expanduser().resolve()
    args.xsl_path = args.xsl_path.expanduser().resolve()
    if not args.xsl_path.exists():
        raise FileNotFoundError(args.xsl_path)

    outdir = args.bayes_root / "templates" / args.template_lib
    manifest = args.bayes_root / "templates" / f"{args.template_lib}_manifest.csv"
    lsf_paths = [
        args.bayes_root / "config_files" / "instruments" / f"{args.template_lib}.lsf",
        WORKDIR / "config" / f"{args.template_lib}.lsf",
    ]
    if outdir.exists() and any(outdir.glob("*.fits")) and not args.force:
        raise FileExistsError(f"{outdir} already contains FITS files. Pass --force to overwrite.")
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.fits"):
        old.unlink()

    with np.load(args.xsl_path) as data:
        templates = np.asarray(data["templates"], dtype=float)
        wave_in = np.asarray(data["lam"], dtype=float)
        ages = np.asarray(data["ages"], dtype=float)
        metals = np.asarray(data["metals"], dtype=float)
        fwhm = np.asarray(data["fwhm"], dtype=float)

    wave_out = output_grid(args.lmin, args.lmax, args.linear_step)
    use_wave = (wave_in >= args.lmin - 5.0 * args.linear_step) & (wave_in <= args.lmax + 5.0 * args.linear_step)
    if np.count_nonzero(use_wave) < 10:
        raise ValueError("XSL wavelength grid does not overlap requested template range.")
    wave_trim = wave_in[use_wave]

    pairs = []
    for age_idx, age in enumerate(ages):
        if args.age_min is not None and age < args.age_min:
            continue
        if args.age_max is not None and age > args.age_max:
            continue
        for metal_idx, metal in enumerate(metals):
            if args.metal_min is not None and metal < args.metal_min:
                continue
            if args.metal_max is not None and metal > args.metal_max:
                continue
            pairs.append((age_idx, metal_idx))

    if args.max_templates is not None and len(pairs) > args.max_templates:
        indices = np.linspace(0, len(pairs) - 1, int(args.max_templates), dtype=int)
        pairs = [pairs[i] for i in indices]
    if not pairs:
        raise ValueError("No XSL templates matched the requested selection.")

    rows = []
    for out_idx, (age_idx, metal_idx) in enumerate(pairs, start=1):
        flux_in = templates[use_wave, age_idx, metal_idx]
        good = np.isfinite(flux_in) & np.isfinite(wave_trim)
        if np.count_nonzero(good) < 2:
            print(f"skip age_idx={age_idx} metal_idx={metal_idx}: too few finite pixels")
            continue
        flux = np.interp(wave_out, wave_trim[good], flux_in[good])
        scale = np.nanmedian(flux)
        if np.isfinite(scale) and scale != 0:
            flux = flux / scale
        flux = np.nan_to_num(flux, nan=1.0, posinf=1.0, neginf=1.0)
        meta = XslTemplate(
            index=out_idx,
            age_gyr=float(ages[age_idx]),
            metal=float(metals[metal_idx]),
            age_index=int(age_idx),
            metal_index=int(metal_idx),
        )
        outname = f"xsl_g235h_{out_idx:04d}_age{meta.age_gyr:06.3f}_metal{meta.metal:+.2f}.fits"
        write_template(outdir / outname, wave_out, flux, meta)
        rows.append(
            {
                "output": outname,
                "source": str(args.xsl_path),
                "age_gyr": meta.age_gyr,
                "metal": meta.metal,
                "age_index": meta.age_index,
                "metal_index": meta.metal_index,
            }
        )
        print(f"[{out_idx:04d}/{len(pairs):04d}] {outname}")

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output", "source", "age_gyr", "metal", "age_index", "metal_index"])
        writer.writeheader()
        writer.writerows(rows)

    for lsf_path in lsf_paths:
        write_lsf(lsf_path, wave_in, fwhm, args.lmin, args.lmax)

    run_file, manifest_json = write_reproduction_files(
        outdir,
        run_name=f"{args.template_lib}_template_export",
        input_paths=[args.xsl_path],
        output_paths=[manifest, *lsf_paths, *sorted(outdir.glob("*.fits"))],
        extra={
            "runner": "make_xsl_bayes_templates.py",
            "template_lib": args.template_lib,
            "source": str(args.xsl_path),
            "n_templates": len(rows),
            "lmin": args.lmin,
            "lmax": args.lmax,
            "linear_step": args.linear_step,
        },
        run_file_name="reproduce_template_export.sh",
        manifest_name="template_export_manifest.json",
    )

    print(f"wrote {len(rows)} templates to {outdir}")
    print(f"wrote manifest to {manifest}")
    print(f"wrote LSF files to {', '.join(str(path) for path in lsf_paths)}")
    print(f"wrote reproduction files: {run_file}, {manifest_json}")


if __name__ == "__main__":
    main()
