from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import binary_erosion

from fits_utils import copy_metadata, selected_hdu, write_primary_image


@dataclass
class IfuPhotometryResult:
    image: np.ndarray
    uneroded_image: np.ndarray
    header: fits.Header
    wavelength_microns: np.ndarray
    response: np.ndarray
    denominator: float
    source_hdu_index: int


def wavelength_microns_from_header(header: fits.Header, n_wave: int) -> np.ndarray:
    """Build the cube wavelength grid in microns."""
    if "WAVSTART" in header and "WAVEND" in header:
        wave = np.linspace(float(header["WAVSTART"]), float(header["WAVEND"]), n_wave)
    elif "CRVAL3" in header and "CDELT3" in header:
        crpix = float(header.get("CRPIX3", 1.0))
        pixels = np.arange(n_wave, dtype=float) + 1.0
        wave = float(header["CRVAL3"]) + (pixels - crpix) * float(header["CDELT3"])
    else:
        raise ValueError("Could not infer wavelength grid: need WAVSTART/WAVEND or CRVAL3/CDELT3.")

    unit = str(header.get("CUNIT3", "")).strip().lower()
    median_wave = float(np.nanmedian(wave))
    if unit in {"m", "meter", "meters"} or median_wave < 1.0e-3:
        return wave * 1.0e6
    if unit in {"nm", "nanometer", "nanometers"}:
        return wave * 1.0e-3
    if unit in {"angstrom", "angstroms", "aa"}:
        return wave * 1.0e-4
    return wave


def throughput_response(
    wavelength_microns: np.ndarray,
    throughput_table: Table,
    *,
    photon_weighted: bool = True,
) -> np.ndarray:
    """Interpolate the NIRCam throughput curve onto the IFU wavelength grid."""
    throughput = np.interp(
        wavelength_microns,
        np.asarray(throughput_table["Microns"], dtype=float),
        np.asarray(throughput_table["Throughput"], dtype=float),
        left=0.0,
        right=0.0,
    )
    if photon_weighted:
        return throughput * wavelength_microns
    return throughput


def bandpass_average_image(
    cube: np.ndarray,
    wavelength_microns: np.ndarray,
    response: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Collapse a cube into a throughput-weighted bandpass-average image."""
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    denominator = float(integrate(response, wavelength_microns))
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError("Throughput response has zero overlap with the IFU wavelength grid.")

    cube0 = np.nan_to_num(cube.astype(float, copy=False), nan=0.0)
    numerator = integrate(cube0 * response[:, None, None], wavelength_microns, axis=0)
    image = numerator / denominator
    image[image == 0.0] = np.nan
    return image, denominator


def erode_valid_region_to_nan(image: np.ndarray, n_pixels: int) -> np.ndarray:
    """Erode the finite IFU footprint and set edge pixels to NaN."""
    if n_pixels <= 0:
        return image.copy()
    valid = np.isfinite(image)
    eroded = valid.copy()
    for _ in range(n_pixels):
        eroded = binary_erosion(eroded, border_value=0)
    out = image.copy()
    out[~eroded] = np.nan
    return out


def two_dimensional_ifu_header(ifu_header: fits.Header) -> fits.Header:
    """Build a 2D celestial WCS header from an IFU cube header."""
    header = WCS(ifu_header).celestial.to_header(relax=True)
    copy_metadata(header, [ifu_header], keys=("BUNIT", "PHOTMJSR", "PIXAR_SR", "TELESCOP", "INSTRUME"))
    return header


def build_ifu_f200_image(
    ifu_path: str | Path,
    throughput_path: str | Path,
    *,
    ifu_hdu_index: int | None = None,
    erode_pixels: int = 1,
    photon_weighted: bool = True,
) -> IfuPhotometryResult:
    """Build the synthetic F200W-equivalent IFU image used for patching."""
    throughput_table = Table.read(throughput_path, format="ascii")

    with fits.open(ifu_path, memmap=True) as hdul:
        idx = ifu_hdu_index
        if idx is None:
            idx = next(i for i, hdu in enumerate(hdul) if hdu.data is not None and hdu.data.ndim == 3)
        ifu_hdu = selected_hdu(hdul, idx, ndim=3)
        cube = ifu_hdu.data
        wavelength_microns = wavelength_microns_from_header(ifu_hdu.header, cube.shape[0])
        response = throughput_response(wavelength_microns, throughput_table, photon_weighted=photon_weighted)
        uneroded, denominator = bandpass_average_image(cube, wavelength_microns, response)
        image = erode_valid_region_to_nan(uneroded, erode_pixels)
        header = two_dimensional_ifu_header(ifu_hdu.header)
        header["HDUIN"] = (idx, "Source IFU cube HDU index")
        header["ERODEPIX"] = (erode_pixels, "Finite IFU footprint erosion in pixels")
        header["PHOTWT"] = (bool(photon_weighted), "Photon-weighted throughput response")
        header["THRFILE"] = (Path(throughput_path).name, "Throughput table basename")
        header["THRDEN"] = (denominator, "Bandpass response integral")

    return IfuPhotometryResult(
        image=image,
        uneroded_image=uneroded,
        header=header,
        wavelength_microns=wavelength_microns,
        response=response,
        denominator=denominator,
        source_hdu_index=idx,
    )


def write_ifu_photometry(
    result: IfuPhotometryResult,
    path: str | Path,
    *,
    overwrite: bool = True,
    include_uneroded: bool = True,
) -> Path:
    """Save the raw synthetic IFU image before chi2 alignment."""
    extra_hdus = None
    if include_uneroded:
        extra_hdus = [
            fits.ImageHDU(
                data=np.asarray(result.uneroded_image, dtype=np.float32),
                name="UNERODED",
            )
        ]
    return write_primary_image(path, result.image, result.header, overwrite=overwrite, extra_hdus=extra_hdus)
