from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from fits_utils import copy_metadata, selected_hdu, write_primary_image


@dataclass
class ScaleFitResult:
    mode: str
    factor: float
    n_pixels: int
    ratio_median: float | None = None
    ratio_mean: float | None = None
    ratio_std: float | None = None
    ratio_p05: float | None = None
    ratio_p16: float | None = None
    ratio_p84: float | None = None
    ratio_p95: float | None = None
    manual: bool = False


@dataclass
class PatchResult:
    mosaic: np.ndarray
    ifu_on_mosaic: np.ndarray
    footprint: np.ndarray
    patch_mask: np.ndarray
    scale_mask: np.ndarray
    scale_fit: ScaleFitResult
    n_reference_zero_masked: int = 0
    mask_reference_zero: bool = True
    reference_zero_atol: float = 0.0

    @property
    def scale_factor(self) -> float:
        return self.scale_fit.factor

    @property
    def n_scale_pixels(self) -> int:
        return self.scale_fit.n_pixels


def _require_reproject():
    try:
        from reproject import reproject_interp
        from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
    except ImportError as exc:
        raise RuntimeError("This tool requires the reproject package.") from exc
    return reproject_interp, find_optimal_celestial_wcs, reproject_and_coadd


def coadd_f200_images(
    f200_paths: list[str | Path],
    output_path: str | Path,
    *,
    hdu_index: int | None = None,
    overwrite: bool = True,
    write_footprint: bool = False,
) -> Path:
    """Coadd F200W images with reproject_interp and save the F200-only mosaic."""
    reproject_interp, find_optimal_celestial_wcs, reproject_and_coadd = _require_reproject()

    handles = []
    try:
        for path in f200_paths:
            handles.append(fits.open(path, memmap=True))
        hdus = [selected_hdu(handle, hdu_index, ndim=2) for handle in handles]

        wcs, shape = find_optimal_celestial_wcs(hdus)
        coadd, footprint = reproject_and_coadd(
            hdus,
            output_projection=wcs,
            shape_out=shape,
            reproject_function=reproject_interp,
        )

        header = wcs.to_header(relax=True)
        copy_metadata(header, [hdus[0].header, handles[0][0].header])
        header["NINPUT"] = (len(f200_paths), "Number of F200W images coadded")
        header["RPROJ"] = ("interp", "Reprojection function used for coadd")

        extra = None
        if write_footprint:
            extra = [fits.ImageHDU(data=np.asarray(footprint, dtype=np.float32), name="FOOTPRINT")]
        return write_primary_image(output_path, coadd, header, overwrite=overwrite, extra_hdus=extra)
    finally:
        for handle in handles:
            handle.close()


def robust_scale_factor(
    reference: np.ndarray,
    model: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str = "median_ratio",
    min_pixels: int = 25,
) -> ScaleFitResult:
    """Fit a multiplicative model scale so model matches reference on finite overlap."""
    use = mask & np.isfinite(reference) & np.isfinite(model) & (model != 0.0)
    n_pixels = int(np.count_nonzero(use))
    if n_pixels == 0:
        if mode == "none":
            return ScaleFitResult(mode=mode, factor=1.0, n_pixels=0)
        raise RuntimeError("No overlap pixels available for IFU/F200W scaling.")

    ref = reference[use].astype(float, copy=False)
    mod = model[use].astype(float, copy=False)
    ratio = ref / mod
    ratio = ratio[np.isfinite(ratio)]

    stats = {}
    if ratio.size:
        p05, p16, p84, p95 = np.nanpercentile(ratio, [5.0, 16.0, 84.0, 95.0])
        stats = {
            "ratio_median": float(np.nanmedian(ratio)),
            "ratio_mean": float(np.nanmean(ratio)),
            "ratio_std": float(np.nanstd(ratio)),
            "ratio_p05": float(p05),
            "ratio_p16": float(p16),
            "ratio_p84": float(p84),
            "ratio_p95": float(p95),
        }

    if mode == "none":
        return ScaleFitResult(mode=mode, factor=1.0, n_pixels=n_pixels, **stats)

    if n_pixels < min_pixels:
        raise RuntimeError(f"Only {n_pixels} overlap pixels available for IFU/F200W scaling.")

    if mode == "median_ratio":
        if ratio.size < min_pixels:
            raise RuntimeError("Too few finite ratio pixels available for IFU/F200W scaling.")
        clipped = ratio[(ratio >= stats["ratio_p05"]) & (ratio <= stats["ratio_p95"])]
        return ScaleFitResult(
            mode=mode,
            factor=float(np.nanmedian(clipped)),
            n_pixels=int(clipped.size),
            **stats,
        )

    if mode == "linear_fit":
        denom = float(np.sum(mod * mod))
        if denom == 0.0:
            raise RuntimeError("Cannot fit IFU/F200W scale: zero model norm.")
        return ScaleFitResult(
            mode=mode,
            factor=float(np.sum(ref * mod) / denom),
            n_pixels=n_pixels,
            **stats,
        )

    raise ValueError(f"Unknown scale mode: {mode}")


def patch_mosaic_with_ifu(
    f200_hdu,
    ifu_aligned_hdu,
    *,
    scale_mode: str = "median_ratio",
    scale_factor: float | None = None,
    footprint_min: float = 0.0,
    min_scale_pixels: int = 25,
    mask_reference_zero: bool = True,
    reference_zero_atol: float = 0.0,
) -> PatchResult:
    """Reproject the aligned IFU onto the F200W mosaic and patch valid IFU pixels."""
    try:
        from reproject import reproject_interp
    except ImportError as exc:
        raise RuntimeError("The patching step requires the reproject package.") from exc

    f200_data = np.asarray(f200_hdu.data, dtype=np.float32)
    f200_wcs = WCS(f200_hdu.header).celestial
    ifu_wcs = WCS(ifu_aligned_hdu.header).celestial

    ifu_on_mosaic, footprint = reproject_interp(
        (ifu_aligned_hdu.data.astype(float, copy=False), ifu_wcs),
        f200_wcs,
        shape_out=f200_data.shape,
    )
    ifu_on_mosaic = np.asarray(ifu_on_mosaic, dtype=np.float32)
    footprint = np.asarray(footprint, dtype=np.float32)
    patch_mask = np.isfinite(ifu_on_mosaic) & (footprint > footprint_min)
    scale_mask = patch_mask & np.isfinite(f200_data)
    reference_zero_mask = np.zeros(f200_data.shape, dtype=bool)
    if mask_reference_zero:
        reference_zero_mask = np.isfinite(f200_data) & np.isclose(f200_data, 0.0, atol=reference_zero_atol)
        scale_mask &= ~reference_zero_mask
    n_reference_zero_masked = int(np.count_nonzero(reference_zero_mask & patch_mask))

    if scale_factor is None:
        scale_fit = robust_scale_factor(
            f200_data,
            ifu_on_mosaic,
            scale_mask,
            mode=scale_mode,
            min_pixels=min_scale_pixels,
        )
    else:
        fitted_scale = float(scale_factor)
        scale_fit = robust_scale_factor(
            f200_data,
            ifu_on_mosaic,
            scale_mask,
            mode="none",
            min_pixels=min_scale_pixels,
        )
        scale_fit.mode = scale_mode
        scale_fit.factor = fitted_scale
        scale_fit.manual = True

    mosaic = f200_data.copy()
    mosaic[patch_mask] = (ifu_on_mosaic[patch_mask] * scale_fit.factor).astype(np.float32)

    return PatchResult(
        mosaic=mosaic,
        ifu_on_mosaic=ifu_on_mosaic,
        footprint=footprint,
        patch_mask=patch_mask,
        scale_mask=scale_mask,
        scale_fit=scale_fit,
        n_reference_zero_masked=n_reference_zero_masked,
        mask_reference_zero=bool(mask_reference_zero),
        reference_zero_atol=float(reference_zero_atol),
    )


def write_patched_mosaic(
    result: PatchResult,
    f200_header: fits.Header,
    output_path: str | Path,
    *,
    overwrite: bool = True,
    write_footprint: bool = False,
) -> Path:
    """Save the final F200W mosaic with the aligned, scaled IFU central patch."""
    header = f200_header.copy()
    header["IFUPATCH"] = (True, "Central region patched with synthetic IFU F200W")
    header["IFUSCALE"] = (float(result.scale_factor), "Multiplicative scale applied to IFU patch")
    header["SCALMODE"] = (result.scale_fit.mode, "Method used to determine IFU scale")
    header["SCALMAN"] = (bool(result.scale_fit.manual), "IFU scale was manually supplied")
    header["IFUSNPIX"] = (int(result.n_scale_pixels), "Pixels used to fit IFU scale")
    header["IFUPIX"] = (int(np.count_nonzero(result.patch_mask)), "Pixels replaced by IFU patch")
    header["RPROJIFU"] = ("interp", "IFU reprojection function used for patch")
    header["SCLZMSK"] = (bool(result.mask_reference_zero), "Masked zero reference pixels in scale fit")
    header["SCLZATL"] = (float(result.reference_zero_atol), "Zero-reference absolute tolerance")
    header["SCLNZRO"] = (int(result.n_reference_zero_masked), "Reference zero pixels masked in scale fit")
    for key, value in (
        ("SCLRMED", result.scale_fit.ratio_median),
        ("SCLRMEAN", result.scale_fit.ratio_mean),
        ("SCLRSTD", result.scale_fit.ratio_std),
        ("SCLRP05", result.scale_fit.ratio_p05),
        ("SCLRP16", result.scale_fit.ratio_p16),
        ("SCLRP84", result.scale_fit.ratio_p84),
        ("SCLRP95", result.scale_fit.ratio_p95),
    ):
        if value is not None:
            header[key] = (float(value), "F200W / IFU overlap ratio diagnostic")

    extra = None
    if write_footprint:
        extra = [
            fits.ImageHDU(data=result.patch_mask.astype(np.uint8), name="PATCHMASK"),
            fits.ImageHDU(data=result.scale_mask.astype(np.uint8), name="SCALEMASK"),
        ]
    return write_primary_image(output_path, result.mosaic, header, overwrite=overwrite, extra_hdus=extra)
