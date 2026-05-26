from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits


METADATA_KEYS = (
    "BUNIT",
    "PHOTMJSR",
    "PIXAR_SR",
    "FILTER",
    "PUPIL",
    "INSTRUME",
    "TELESCOP",
    "DETECTOR",
    "DATE-OBS",
    "TIME-OBS",
)


def first_image_hdu(hdul: fits.HDUList, ndim: int | None = None) -> int:
    """Return the first HDU index with image data, optionally matching ndim."""
    for idx, hdu in enumerate(hdul):
        data = hdu.data
        if data is None:
            continue
        if ndim is not None and getattr(data, "ndim", None) != ndim:
            continue
        return idx
    detail = f" with {ndim} dimensions" if ndim is not None else ""
    raise ValueError(f"No image HDU{detail} found in FITS file.")


def selected_hdu(hdul: fits.HDUList, hdu_index: int | None, ndim: int | None = None):
    """Return the requested HDU, or auto-select the first image HDU."""
    idx = first_image_hdu(hdul, ndim=ndim) if hdu_index is None else hdu_index
    hdu = hdul[idx]
    if hdu.data is None:
        raise ValueError(f"HDU {idx} has no image data.")
    if ndim is not None and hdu.data.ndim != ndim:
        raise ValueError(f"HDU {idx} has ndim={hdu.data.ndim}; expected {ndim}.")
    return hdu


def copy_metadata(
    target_header: fits.Header,
    source_headers: Iterable[fits.Header],
    keys: Iterable[str] = METADATA_KEYS,
) -> fits.Header:
    """Copy useful calibration metadata without overwriting WCS keywords."""
    for key in keys:
        if key in target_header:
            continue
        for header in source_headers:
            if key in header:
                target_header[key] = header[key]
                break
    return target_header


def write_primary_image(
    path: str | Path,
    data,
    header: fits.Header,
    *,
    overwrite: bool = True,
    extra_hdus: list[fits.ImageHDU] | None = None,
) -> Path:
    """Write a primary image and optional image extensions as float32 where useful."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.asarray(data)
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32, copy=False)

    hdus = [fits.PrimaryHDU(data=image, header=header)]
    if extra_hdus:
        hdus.extend(extra_hdus)
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    return path

