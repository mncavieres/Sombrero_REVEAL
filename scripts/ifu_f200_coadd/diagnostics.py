from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

from fits_utils import write_primary_image
from mosaic import PatchResult


def json_safe(value: Any):
    """Convert common scientific Python objects into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict, *, overwrite: bool = True) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def patch_bounds(mask: np.ndarray, pad: int = 20) -> tuple[slice, slice]:
    y, x = np.nonzero(mask)
    if y.size == 0:
        raise RuntimeError("Cannot make diagnostics: IFU patch mask is empty.")
    ymin = max(int(y.min()) - pad, 0)
    ymax = min(int(y.max()) + pad + 1, mask.shape[0])
    xmin = max(int(x.min()) - pad, 0)
    xmax = min(int(x.max()) + pad + 1, mask.shape[1])
    return slice(ymin, ymax), slice(xmin, xmax)


def patch_center(mask: np.ndarray) -> tuple[float, float]:
    y, x = np.nonzero(mask)
    if y.size == 0:
        raise RuntimeError("Cannot determine patch center: IFU patch mask is empty.")
    return float(np.mean(x)), float(np.mean(y))


def write_patch_diagnostic_fits(
    path: str | Path,
    f200_hdu,
    patch: PatchResult,
    *,
    pad_pixels: int = 20,
    overwrite: bool = True,
) -> Path:
    """Write a compact FITS cutout with patch arrays needed for later QA."""
    yslice, xslice = patch_bounds(patch.patch_mask, pad=pad_pixels)
    header = WCS(f200_hdu.header).celestial.slice((yslice, xslice)).to_header(relax=True)
    for key in ("BUNIT", "FILTER", "INSTRUME", "TELESCOP", "PIXAR_SR", "PHOTMJSR"):
        if key in f200_hdu.header:
            header[key] = f200_hdu.header[key]
    header["XMIN0"] = (xslice.start, "0-indexed x origin in full F200W mosaic")
    header["YMIN0"] = (yslice.start, "0-indexed y origin in full F200W mosaic")
    header["IFUSCALE"] = (patch.scale_factor, "Multiplicative scale applied to IFU patch")

    extra = [
        fits.ImageHDU(
            data=np.asarray(patch.ifu_on_mosaic[yslice, xslice], dtype=np.float32),
            header=header.copy(),
            name="IFU_UNSCALED",
        ),
        fits.ImageHDU(
            data=np.asarray(patch.ifu_on_mosaic[yslice, xslice] * patch.scale_factor, dtype=np.float32),
            header=header.copy(),
            name="IFU_SCALED",
        ),
        fits.ImageHDU(
            data=np.asarray(patch.mosaic[yslice, xslice], dtype=np.float32),
            header=header.copy(),
            name="PATCHED",
        ),
        fits.ImageHDU(
            data=np.asarray(patch.footprint[yslice, xslice], dtype=np.float32),
            header=header.copy(),
            name="IFU_FOOT",
        ),
        fits.ImageHDU(
            data=patch.patch_mask[yslice, xslice].astype(np.uint8),
            header=header.copy(),
            name="PATCHMASK",
        ),
        fits.ImageHDU(
            data=patch.scale_mask[yslice, xslice].astype(np.uint8),
            header=header.copy(),
            name="SCALEMASK",
        ),
    ]
    return write_primary_image(
        path,
        np.asarray(f200_hdu.data[yslice, xslice], dtype=np.float32),
        header,
        overwrite=overwrite,
        extra_hdus=extra,
    )


def write_scale_pixel_table(
    path: str | Path,
    f200_data: np.ndarray,
    patch: PatchResult,
    *,
    overwrite: bool = True,
) -> Path:
    """Save the finite overlap pixels used for flux scaling."""
    y, x = np.nonzero(patch.scale_mask)
    f200 = f200_data[y, x].astype(float)
    ifu = patch.ifu_on_mosaic[y, x].astype(float)
    ifu_scaled = ifu * patch.scale_factor
    ratio = np.full_like(f200, np.nan, dtype=float)
    good = np.isfinite(f200) & np.isfinite(ifu) & (ifu != 0.0)
    ratio[good] = f200[good] / ifu[good]

    table = Table()
    table["x_pix"] = x
    table["y_pix"] = y
    table["f200_mosaic"] = f200
    table["ifu_unscaled"] = ifu
    table["ifu_scaled"] = ifu_scaled
    table["f200_over_ifu"] = ratio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", overwrite=overwrite)
    return path


def _pixel_vector_for_pa(wcs: WCS, x0: float, y0: float, pa_deg: float) -> tuple[np.ndarray, float]:
    sky0 = wcs.pixel_to_world(x0, y0)
    sky1 = sky0.directional_offset_by(pa_deg * u.deg, 1.0 * u.arcsec)
    x1, y1 = wcs.world_to_pixel(sky1)
    vec = np.array([x1 - x0, y1 - y0], dtype=float)
    norm = float(np.hypot(vec[0], vec[1]))
    if not np.isfinite(norm) or norm == 0.0:
        raise RuntimeError("Could not convert sky PA to a finite mosaic pixel vector.")
    return vec / norm, 1.0 / norm


def _sample_image_along_line(
    image: np.ndarray,
    x0: float,
    y0: float,
    direction: np.ndarray,
    distances_pix: np.ndarray,
    *,
    width_pix: int = 3,
) -> np.ndarray:
    perp = np.array([-direction[1], direction[0]])
    half_width = max(int(width_pix) // 2, 0)
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    values = []
    for dist in distances_pix:
        samples = []
        for offset in offsets:
            x = x0 + dist * direction[0] + offset * perp[0]
            y = y0 + dist * direction[1] + offset * perp[1]
            samples.append(
                map_coordinates(
                    image,
                    np.array([[y], [x]]),
                    order=1,
                    mode="constant",
                    cval=np.nan,
                    prefilter=False,
                )[0]
            )
        samples = np.asarray(samples, dtype=float)
        if np.any(np.isfinite(samples)):
            values.append(float(np.nanmean(samples)))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _axis_half_length_pix(mask: np.ndarray, x0: float, y0: float, direction: np.ndarray, pad_pix: float) -> float:
    y, x = np.nonzero(mask)
    offsets = (x.astype(float) - x0) * direction[0] + (y.astype(float) - y0) * direction[1]
    return float(np.nanmax(np.abs(offsets)) + pad_pix)


def build_profile_table(
    f200_hdu,
    patch: PatchResult,
    *,
    pa_deg: float,
    half_length_arcsec: float | None = None,
    step_pix: float = 1.0,
    width_pix: int = 3,
    pad_pix: float = 10.0,
) -> Table:
    """Sample major/minor-axis profiles through the IFU patch center."""
    wcs = WCS(f200_hdu.header).celestial
    x0, y0 = patch_center(patch.patch_mask)
    axes = (("major", pa_deg), ("minor", pa_deg + 90.0))
    rows = []

    for axis_name, axis_pa in axes:
        direction, arcsec_per_pix = _pixel_vector_for_pa(wcs, x0, y0, axis_pa)
        if half_length_arcsec is None:
            half_pix = _axis_half_length_pix(patch.patch_mask, x0, y0, direction, pad_pix)
        else:
            half_pix = float(half_length_arcsec) / arcsec_per_pix
        distances_pix = np.arange(-half_pix, half_pix + step_pix, float(step_pix))
        distances_arcsec = distances_pix * arcsec_per_pix

        f200_profile = _sample_image_along_line(
            f200_hdu.data.astype(float, copy=False), x0, y0, direction, distances_pix, width_pix=width_pix
        )
        ifu_profile = _sample_image_along_line(
            patch.ifu_on_mosaic, x0, y0, direction, distances_pix, width_pix=width_pix
        )
        ifu_scaled_profile = ifu_profile * patch.scale_factor
        patched_profile = _sample_image_along_line(
            patch.mosaic, x0, y0, direction, distances_pix, width_pix=width_pix
        )

        for i, distance_pix in enumerate(distances_pix):
            rows.append(
                (
                    axis_name,
                    float(axis_pa % 180.0),
                    float(distances_arcsec[i]),
                    float(x0 + distance_pix * direction[0]),
                    float(y0 + distance_pix * direction[1]),
                    float(f200_profile[i]),
                    float(ifu_profile[i]),
                    float(ifu_scaled_profile[i]),
                    float(patched_profile[i]),
                )
            )

    table = Table(
        rows=rows,
        names=(
            "axis",
            "pa_deg",
            "distance_arcsec",
            "x_pix",
            "y_pix",
            "f200_mosaic",
            "ifu_unscaled",
            "ifu_scaled",
            "patched_mosaic",
        ),
    )
    table.meta["profile_center_x_pix"] = x0
    table.meta["profile_center_y_pix"] = y0
    table.meta["major_axis_pa_deg"] = float(pa_deg)
    table.meta["profile_width_pix"] = int(width_pix)
    table.meta["profile_step_pix"] = float(step_pix)
    table.meta["scale_factor"] = float(patch.scale_factor)
    return table


def write_profile_table(path: str | Path, table: Table, *, overwrite: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", overwrite=overwrite)
    return path


def write_scaling_checkplot(
    path: str | Path,
    f200_hdu,
    patch: PatchResult,
    profile_table: Table,
    scale_table_path: str | Path,
    *,
    pad_pixels: int = 20,
    overwrite: bool = True,
) -> Path:
    """Write a PNG checkplot for scale-factor and profile QA."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mpl_cache = Path(tempfile.gettempdir()) / "ifu_f200_coadd_mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yslice, xslice = patch_bounds(patch.patch_mask, pad=pad_pixels)
    cut = np.asarray(f200_hdu.data[yslice, xslice], dtype=float)
    finite = cut[np.isfinite(cut)]
    vmin, vmax = (np.nanpercentile(finite, [5.0, 99.0]) if finite.size else (0.0, 1.0))
    filter_label = str(f200_hdu.header.get("FILTER", "NIRCam"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax_img, ax_hist, ax_major, ax_minor = axes.ravel()

    ax_img.imshow(cut, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax_img.contour(patch.patch_mask[yslice, xslice].astype(float), levels=[0.5], colors="tab:red", linewidths=1.0)
    ax_img.set_title(f"{filter_label} cutout with IFU patch mask")
    ax_img.set_xlabel("x cutout pixel")
    ax_img.set_ylabel("y cutout pixel")

    for axis_name, color in (("major", "tab:orange"), ("minor", "tab:cyan")):
        rows = profile_table[profile_table["axis"] == axis_name]
        if len(rows) == 0:
            continue
        xline = np.asarray(rows["x_pix"], dtype=float) - xslice.start
        yline = np.asarray(rows["y_pix"], dtype=float) - yslice.start
        ax_img.plot(xline, yline, color=color, lw=1.3, label=axis_name)
    ax_img.legend(loc="best", fontsize=8)

    scale_table = Table.read(scale_table_path, format="ascii.ecsv")
    ratios = np.asarray(scale_table["f200_over_ifu"], dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size:
        lo, hi = np.nanpercentile(ratios, [2.0, 98.0])
        ax_hist.hist(ratios[(ratios >= lo) & (ratios <= hi)], bins=40, color="0.35", alpha=0.8)
        ax_hist.axvline(patch.scale_factor, color="tab:red", lw=2, label=f"scale = {patch.scale_factor:.4g}")
    ax_hist.set_title(f"{filter_label} / IFU overlap ratios")
    ax_hist.set_xlabel("ratio")
    ax_hist.set_ylabel("pixels")
    ax_hist.legend(loc="best", fontsize=8)

    for axis_name, ax in (("major", ax_major), ("minor", ax_minor)):
        rows = profile_table[profile_table["axis"] == axis_name]
        x = np.asarray(rows["distance_arcsec"], dtype=float)
        ax.plot(x, rows["f200_mosaic"], color="black", lw=1.6, label=f"{filter_label} mosaic")
        ax.plot(x, rows["ifu_unscaled"], color="tab:blue", ls=":", lw=1.3, label="IFU unscaled")
        ax.plot(x, rows["ifu_scaled"], color="tab:red", lw=1.3, label="IFU scaled")
        ax.plot(x, rows["patched_mosaic"], color="tab:green", lw=1.0, alpha=0.8, label="patched")
        ax.set_title(f"{axis_name.capitalize()} axis profile")
        ax.set_xlabel("offset from IFU patch center [arcsec]")
        ax.set_ylabel(str(f200_hdu.header.get("BUNIT", "flux")))
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"IFU scale diagnostics: factor={patch.scale_factor:.6g}, "
        f"pixels={patch.n_scale_pixels}, mode={patch.scale_fit.mode}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def software_versions() -> dict[str, str]:
    import astropy
    import reproject
    import scipy

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "astropy": astropy.__version__,
        "reproject": reproject.__version__,
    }
