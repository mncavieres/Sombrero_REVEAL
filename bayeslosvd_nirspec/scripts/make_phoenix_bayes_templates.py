#!/usr/bin/env python3
"""Convert local PHOENIX spectra into BAYES-LOSVD template FITS files."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


WORKSPACE = Path(__file__).resolve().parents[2]
WORKDIR = WORKSPACE / "bayeslosvd_nirspec"
DEFAULT_BAYES_ROOT = WORKDIR / "BAYES-LOSVD"
DEFAULT_PHOENIX_DIR = WORKSPACE / "Data/phoenix_high_res"
DEFAULT_WAVE_PATH = DEFAULT_PHOENIX_DIR / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
PHOENIX_RE = re.compile(r"lte(?P<teff>\d{5})-(?P<logg>\d\.\d{2})(?P<feh>[+-]\d\.\d)")


@dataclass(frozen=True)
class TemplateMeta:
    path: Path
    teff: float
    logg: float
    feh: float


def parse_meta(path: Path) -> TemplateMeta | None:
    match = PHOENIX_RE.search(path.name)
    if match is None:
        return None
    return TemplateMeta(
        path=path,
        teff=float(match.group("teff")),
        logg=float(match.group("logg")),
        feh=float(match.group("feh")),
    )


def discover_templates(args: argparse.Namespace) -> list[TemplateMeta]:
    if args.template_list is not None:
        names = [
            line.strip()
            for line in args.template_list.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        paths = [(Path(name) if Path(name).is_absolute() else args.phoenix_dir / name) for name in names]
    else:
        paths = sorted(
            p
            for p in args.phoenix_dir.rglob("*")
            if p.name.startswith("lte") and (p.name.endswith(".fits") or p.name.endswith(".fits.gz"))
        )

    meta = []
    for path in paths:
        item = parse_meta(path)
        if item is None:
            continue
        if (
            args.teff_min <= item.teff <= args.teff_max
            and args.logg_min <= item.logg <= args.logg_max
            and args.feh_min <= item.feh <= args.feh_max
        ):
            meta.append(item)

    meta.sort(key=lambda m: (m.feh, m.logg, m.teff, m.path.name))
    if args.max_templates is not None and len(meta) > args.max_templates:
        indices = np.linspace(0, len(meta) - 1, int(args.max_templates), dtype=int)
        meta = [meta[i] for i in indices]
    if not meta:
        raise ValueError("No PHOENIX templates matched the requested selection.")
    return meta


def output_grid(lmin: float, lmax: float, step: float) -> np.ndarray:
    if lmax <= lmin:
        raise ValueError("lmax must be greater than lmin")
    n = int(np.floor((lmax - lmin) / step)) + 1
    return lmin + step * np.arange(n, dtype=float)


def write_template(outpath: Path, wave_out: np.ndarray, flux: np.ndarray, meta: TemplateMeta) -> None:
    hdr = fits.Header()
    hdr["CRVAL1"] = float(wave_out[0])
    hdr["CDELT1"] = float(wave_out[1] - wave_out[0])
    hdr["CRPIX1"] = 1.0
    hdr["CTYPE1"] = "WAVE"
    hdr["CUNIT1"] = "Angstrom"
    hdr["TEFF"] = float(meta.teff)
    hdr["LOGG"] = float(meta.logg)
    hdr["FEH"] = float(meta.feh)
    fits.PrimaryHDU(np.asarray(flux, dtype=np.float32), header=hdr).writeto(outpath, overwrite=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayes-root", type=Path, default=DEFAULT_BAYES_ROOT)
    parser.add_argument("--phoenix-dir", type=Path, default=DEFAULT_PHOENIX_DIR)
    parser.add_argument("--phoenix-wave-path", type=Path, default=DEFAULT_WAVE_PATH)
    parser.add_argument("--template-list", type=Path, default=None)
    parser.add_argument("--template-lib", type=str, default="PHOENIX_NIRSPEC_G235H")
    parser.add_argument("--lmin", type=float, default=20500.0)
    parser.add_argument("--lmax", type=float, default=24500.0)
    parser.add_argument("--linear-step", type=float, default=0.5)
    parser.add_argument("--teff-min", type=float, default=3000.0)
    parser.add_argument("--teff-max", type=float, default=6700.0)
    parser.add_argument("--feh-min", type=float, default=-2.0)
    parser.add_argument("--feh-max", type=float, default=1.0)
    parser.add_argument("--logg-min", type=float, default=0.0)
    parser.add_argument("--logg-max", type=float, default=4.0)
    parser.add_argument("--max-templates", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.bayes_root = args.bayes_root.expanduser().resolve()
    args.phoenix_dir = args.phoenix_dir.expanduser().resolve()
    args.phoenix_wave_path = args.phoenix_wave_path.expanduser().resolve()
    if args.template_list is not None:
        args.template_list = args.template_list.expanduser().resolve()

    outdir = args.bayes_root / "templates" / args.template_lib
    if outdir.exists() and any(outdir.glob("*.fits")) and not args.force:
        raise FileExistsError(f"{outdir} already contains FITS files. Pass --force to overwrite.")
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.fits"):
        old.unlink()

    meta = discover_templates(args)
    wave_out = output_grid(args.lmin, args.lmax, args.linear_step)
    with fits.open(args.phoenix_wave_path, memmap=False) as hdul:
        wave_all = np.asarray(hdul[0].data, dtype=float).ravel()
    use = (wave_all >= args.lmin - 5.0 * args.linear_step) & (wave_all <= args.lmax + 5.0 * args.linear_step)
    if np.count_nonzero(use) < 10:
        raise ValueError("PHOENIX wavelength grid does not overlap requested template range.")
    wave_in = wave_all[use]

    rows = []
    for i, item in enumerate(meta, start=1):
        with fits.open(item.path, memmap=True) as hdul:
            flux_in = np.asarray(hdul[0].data[use], dtype=float)
        good = np.isfinite(flux_in)
        if np.count_nonzero(good) < 2:
            print(f"skip {item.path.name}: too few finite pixels")
            continue
        flux = np.interp(wave_out, wave_in[good], flux_in[good])
        scale = np.nanmedian(flux)
        if np.isfinite(scale) and scale != 0:
            flux = flux / scale
        outname = f"phoenix_g235h_{i:04d}_teff{item.teff:05.0f}_logg{item.logg:.2f}_feh{item.feh:+.1f}.fits"
        write_template(outdir / outname, wave_out, flux, item)
        rows.append(
            {
                "output": outname,
                "source": str(item.path),
                "teff": item.teff,
                "logg": item.logg,
                "feh": item.feh,
            }
        )
        print(f"[{i:04d}/{len(meta):04d}] {outname}")

    manifest = args.bayes_root / "templates" / f"{args.template_lib}_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output", "source", "teff", "logg", "feh"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} templates to {outdir}")
    print(f"wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
