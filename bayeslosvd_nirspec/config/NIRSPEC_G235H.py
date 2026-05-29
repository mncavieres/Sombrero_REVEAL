import numpy as np
from astropy.io import fits


CUBE_FIT_RANGE_UM = (2.10, 2.398)


def _orient_nlam_first(arr, header):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {arr.shape}")
    nlam = header.get("NAXIS3")
    if nlam is None:
        return arr
    if arr.shape[0] == nlam:
        return arr
    if arr.shape[-1] == nlam:
        return np.moveaxis(arr, -1, 0)
    return arr


def _find_science_hdu(hdul):
    for name in ("SCI", "DATA"):
        if name in hdul and getattr(hdul[name], "data", None) is not None:
            return hdul[name]
    for hdu in hdul:
        if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) == 3:
            return hdu
    raise ValueError("Could not find a 3D science cube")


def _read_error_cube(hdul):
    for extname in ("ERR", "ERROR", "SIGMA", "VAR", "IVAR"):
        if extname not in hdul:
            continue
        hdu = hdul[extname]
        arr = _orient_nlam_first(np.asarray(hdu.data, dtype=float), hdu.header)
        if extname in ("ERR", "ERROR", "SIGMA"):
            return np.clip(arr, 0.0, None)
        if extname == "VAR":
            return np.sqrt(np.clip(arr, 0.0, None))
        if extname == "IVAR":
            out = np.full_like(arr, np.nan, dtype=float)
            good = arr > 0
            out[good] = 1.0 / np.sqrt(arr[good])
            return out
    return None


def _wavelength_to_um(wave, unit):
    unit = (unit or "um").strip().lower()
    if unit in ("um", "micron", "microns", "micrometer", "micrometers"):
        return wave
    if unit in ("angstrom", "angstroms", "aa", "a"):
        return wave / 1.0e4
    if unit in ("nm", "nanometer", "nanometers"):
        return wave / 1.0e3
    if unit in ("m", "meter", "meters"):
        return wave * 1.0e6
    return wave


def _wave_axis_angstrom(header, nlam):
    cd3 = header.get("CD3_3", header.get("CDELT3"))
    crval3 = header.get("CRVAL3")
    crpix3 = float(header.get("CRPIX3", 1.0))
    if cd3 is None or crval3 is None:
        raise KeyError("Could not determine wavelength solution from FITS header")
    pix = np.arange(nlam, dtype=float) + 1.0
    wave = float(crval3) + (pix - crpix3) * float(cd3)
    return _wavelength_to_um(wave, header.get("CUNIT3")) * 1.0e4


def _pixsize_arcsec(header):
    if "CD1_1" in header:
        return abs(float(header["CD1_1"])) * 3600.0
    if "CDELT1" in header:
        return abs(float(header["CDELT1"])) * 3600.0
    if "PIXAR_A2" in header:
        return float(np.sqrt(header["PIXAR_A2"]))
    return 0.1


def _mad_std(arr, axis=None):
    arr = np.asarray(arr, dtype=float)
    med = np.nanmedian(arr, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(arr - med), axis=axis)
    return 1.4826 * mad


def _estimate_error_cube(cube):
    noise = _mad_std(np.diff(cube, axis=0), axis=0) / np.sqrt(2.0)
    good = np.isfinite(noise) & (noise > 0)
    fill = float(np.nanmedian(noise[good])) if np.any(good) else 1.0
    noise = np.where(good, noise, fill)
    return np.broadcast_to(noise, cube.shape).copy()


def _fill_invalid_vector(values, fallback=0.0):
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if np.all(good):
        return values
    out = values.copy()
    if np.count_nonzero(good) >= 2:
        idx = np.flatnonzero(good)
        out[~good] = np.interp(np.flatnonzero(~good), idx, values[good])
        return out
    out[~good] = fallback
    out[good] = values[good]
    return out


def _sanitize_flux_and_error(spec, espec):
    spec2 = spec.reshape(spec.shape[0], -1).astype(float, copy=True)
    err2 = espec.reshape(espec.shape[0], -1).astype(float, copy=True)

    finite_err = np.isfinite(err2) & (err2 > 0)
    err_fill = float(np.nanmedian(err2[finite_err])) if np.any(finite_err) else 1.0
    large_err = err_fill * 1.0e6

    for j in range(spec2.shape[1]):
        good_flux = np.isfinite(spec2[:, j])
        fallback = float(np.nanmedian(spec2[good_flux, j])) if np.any(good_flux) else 0.0
        spec2[:, j] = _fill_invalid_vector(spec2[:, j], fallback=fallback)

        bad_err = ~np.isfinite(err2[:, j]) | (err2[:, j] <= 0)
        if np.any(bad_err):
            err2[bad_err, j] = large_err

    return spec2, err2


def read_data(filename):
    with fits.open(filename, memmap=False) as hdul:
        sci_hdu = _find_science_hdu(hdul)
        header = sci_hdu.header.copy()
        cube = _orient_nlam_first(np.asarray(sci_hdu.data, dtype=float), header)
        err_cube = _read_error_cube(hdul)

    if err_cube is None:
        err_cube = _estimate_error_cube(cube)

    nlam, ny, nx = cube.shape
    wave = _wave_axis_angstrom(header, nlam)
    wave_um = wave / 1.0e4
    pixsize = _pixsize_arcsec(header)

    fit = (wave_um >= CUBE_FIT_RANGE_UM[0]) & (wave_um <= CUBE_FIT_RANGE_UM[1])
    signal_map = np.nanmedian(cube[fit], axis=0) if np.any(fit) else np.nanmedian(cube, axis=0)
    signal_for_center = np.where(np.isfinite(signal_map), signal_map, -np.inf)
    center_row, center_col = np.unravel_index(int(np.nanargmax(signal_for_center)), signal_map.shape)

    row2d, col2d = np.indices((ny, nx))
    x = (col2d - center_col) * pixsize
    y = (row2d - center_row) * pixsize

    spec, espec = _sanitize_flux_and_error(cube, err_cube)

    return {
        "wave": wave,
        "spec": spec,
        "espec": espec,
        "x": x.ravel(),
        "y": y.ravel(),
        "npix": nlam,
        "nspax": ny * nx,
        "psize": pixsize,
        "ndim": 2,
    }
