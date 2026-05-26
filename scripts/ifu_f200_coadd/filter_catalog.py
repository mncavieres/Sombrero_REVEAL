from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from fits_utils import selected_hdu
from photometry import wavelength_microns_from_header


NIRSPEC_BANDS = {
    "g235": (1.66, 3.17),
    "g395": (2.87, 5.27),
}

NON_IMAGING_FILTER_NAMES = {"WLP4"}


def normalize_filter_name(filter_name: str) -> str:
    return str(filter_name).strip().upper()


def filter_name_from_throughput_path(path: str | Path) -> str:
    match = re.match(r"([A-Za-z0-9]+)", Path(path).name)
    if not match:
        raise ValueError(f"Cannot infer filter name from {path}")
    return normalize_filter_name(match.group(1))


def throughput_path_for_filter(filter_name: str, throughput_dir: str | Path) -> Path:
    filter_name = normalize_filter_name(filter_name)
    throughput_dir = Path(throughput_dir)
    candidates = sorted(throughput_dir.glob(f"{filter_name}_*mean_system_throughput.txt"))
    if filter_name == "WLP4":
        candidates += sorted(throughput_dir.glob("WLP4_mean_system_throughput.txt"))
    if not candidates:
        raise FileNotFoundError(f"No mean throughput table found for {filter_name} in {throughput_dir}")
    return candidates[0]


def ifu_wavelength_range_microns(ifu_path: str | Path, ifu_hdu_index: int | None = None) -> tuple[float, float]:
    with fits.open(ifu_path, memmap=True) as hdul:
        ifu_hdu = selected_hdu(hdul, ifu_hdu_index, ndim=3)
        wave = wavelength_microns_from_header(ifu_hdu.header, ifu_hdu.data.shape[0])
    return float(np.nanmin(wave)), float(np.nanmax(wave))


def infer_nirspec_band(wave_min_micron: float, wave_max_micron: float) -> str:
    """Return the closest nominal NIRSpec band label for an IFU wavelength range."""
    span = max(float(wave_max_micron) - float(wave_min_micron), 1.0e-12)
    best_name = None
    best_overlap = 0.0
    for name, (band_min, band_max) in NIRSPEC_BANDS.items():
        overlap = max(0.0, min(float(wave_max_micron), band_max) - max(float(wave_min_micron), band_min))
        overlap_fraction = overlap / span
        if overlap_fraction > best_overlap:
            best_name = name
            best_overlap = overlap_fraction

    if best_name is not None and best_overlap >= 0.5:
        return {"g235": "G235H", "g395": "G395H"}.get(best_name, best_name.upper())
    return "custom"


def throughput_stats(path: str | Path, *, significant_fraction: float = 0.01) -> dict:
    table = Table.read(path, format="ascii")
    wave = np.asarray(table["Microns"], dtype=float)
    throughput = np.asarray(table["Throughput"], dtype=float)
    peak = float(np.nanmax(throughput))
    significant = throughput >= max(significant_fraction * peak, 1.0e-4)
    if not np.any(significant):
        raise ValueError(f"No significant throughput found in {path}")
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(integrate(throughput, wave))
    weighted_mean = float(np.average(wave, weights=throughput)) if np.nansum(throughput) > 0 else np.nan
    return {
        "filter": filter_name_from_throughput_path(path),
        "throughput_path": str(path),
        "wave_min_micron": float(wave[significant].min()),
        "wave_max_micron": float(wave[significant].max()),
        "weighted_mean_micron": weighted_mean,
        "peak_throughput": peak,
        "response_integral": area,
    }


def response_fraction_inside(path: str | Path, wave_min_micron: float, wave_max_micron: float) -> float:
    table = Table.read(path, format="ascii")
    wave = np.asarray(table["Microns"], dtype=float)
    throughput = np.asarray(table["Throughput"], dtype=float)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    total = float(integrate(throughput, wave))
    if total <= 0 or not np.isfinite(total):
        return 0.0
    inside = (wave >= wave_min_micron) & (wave <= wave_max_micron)
    if not np.any(inside):
        return 0.0
    return float(integrate(throughput[inside], wave[inside]) / total)


def build_filter_compatibility_table(
    throughput_dir: str | Path,
    *,
    ifu_wave_min_micron: float | None = None,
    ifu_wave_max_micron: float | None = None,
    min_response_fraction: float = 0.75,
    include_non_imaging: bool = False,
) -> Table:
    """Build a NIRCam filter table with nominal G235/G395 and optional IFU coverage."""
    rows = []
    for path in sorted(Path(throughput_dir).glob("*_mean_system_throughput.txt")):
        stats = throughput_stats(path)
        if not include_non_imaging and stats["filter"] in NON_IMAGING_FILTER_NAMES:
            continue
        row = dict(stats)
        for band_name, (band_min, band_max) in NIRSPEC_BANDS.items():
            frac = response_fraction_inside(path, band_min, band_max)
            row[f"{band_name}_response_fraction"] = frac
            row[f"{band_name}_compatible"] = frac >= min_response_fraction
        if ifu_wave_min_micron is not None and ifu_wave_max_micron is not None:
            frac = response_fraction_inside(path, ifu_wave_min_micron, ifu_wave_max_micron)
            row["ifu_response_fraction"] = frac
            row["ifu_compatible"] = frac >= min_response_fraction
        rows.append(row)

    table = Table(rows=rows)
    if len(table):
        table.sort("weighted_mean_micron")
    return table


def compatible_filters_for_ifu(
    ifu_path: str | Path,
    throughput_dir: str | Path,
    *,
    ifu_hdu_index: int | None = None,
    min_response_fraction: float = 0.75,
    include_non_imaging: bool = False,
) -> tuple[Table, tuple[float, float]]:
    wave_min, wave_max = ifu_wavelength_range_microns(ifu_path, ifu_hdu_index)
    table = build_filter_compatibility_table(
        throughput_dir,
        ifu_wave_min_micron=wave_min,
        ifu_wave_max_micron=wave_max,
        min_response_fraction=min_response_fraction,
        include_non_imaging=include_non_imaging,
    )
    keep = np.asarray(table["ifu_compatible"], dtype=bool) if len(table) else []
    return table[keep], (wave_min, wave_max)
