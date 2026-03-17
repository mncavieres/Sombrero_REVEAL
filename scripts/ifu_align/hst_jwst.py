import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from scipy.ndimage import shift as ndi_shift, gaussian_filter
from scipy.optimize import minimize
from astropy.io import fits


def _fit_scale_and_background(ref, model, w=None):
    """
    Fit ref ~ a * model + b and return chi2, a, b.
    """
    if w is None:
        w = np.ones_like(ref)

    A = np.column_stack([model, np.ones_like(model)])
    sw = np.sqrt(w)
    Aw = A * sw[:, None]
    bw = ref * sw

    pars, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    a, b = pars

    resid = ref - (a * model + b)
    chi2 = np.sum(w * resid**2)
    return chi2, a, b


def _shift_image_and_valid(img, valid, dx, dy, order=1):
    """
    Shift image by (dx, dy) in pixels on the reference grid.
    Positive dx -> right, positive dy -> up.
    """
    img0 = np.nan_to_num(img, nan=0.0)

    shifted = ndi_shift(
        img0,
        shift=(dy, dx),
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    valid_shifted = ndi_shift(
        valid.astype(float),
        shift=(dy, dx),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ) > 0.5

    return shifted, valid_shifted


def _normalize_size(size):
    if np.isscalar(size):
        return (int(size), int(size))
    return (int(size[0]), int(size[1]))


def _compute_moving_cutout_size(ref_wcs, moving_wcs, ref_size, max_shift_arcsec, pad_arcsec=5.0):
    """
    Choose a moving-image cutout size big enough to cover the reference cutout
    plus the allowed astrometric search margin.
    """
    ref_ny, ref_nx = _normalize_size(ref_size)

    ref_pixscale = np.mean(proj_plane_pixel_scales(ref_wcs)) * 3600.0   # arcsec/pix
    mov_pixscale = np.mean(proj_plane_pixel_scales(moving_wcs)) * 3600.0 # arcsec/pix

    ref_width_arcsec = ref_nx * ref_pixscale
    ref_height_arcsec = ref_ny * ref_pixscale

    mov_width_arcsec = ref_width_arcsec + 2 * max_shift_arcsec + pad_arcsec
    mov_height_arcsec = ref_height_arcsec + 2 * max_shift_arcsec + pad_arcsec

    mov_nx = int(np.ceil(mov_width_arcsec / mov_pixscale))
    mov_ny = int(np.ceil(mov_height_arcsec / mov_pixscale))

    return (mov_ny, mov_nx)


def align_image_to_reference(
    ref_hdu,
    moving_hdu,
    center,
    size,
    ref_good_pixel_mask=None,
    moving_good_pixel_mask=None,
    weights=None,
    max_shift_arcsec=5.0,
    coarse_step_pix=1.0,
    min_valid_pixels=100,
    highpass_sigma_pix=None,
):
    """
    Align moving_hdu to ref_hdu by solving for a translational WCS shift.

    Parameters
    ----------
    ref_hdu : FITS HDU
        Reference image.
    moving_hdu : FITS HDU
        Image to be aligned to the reference.
    center : tuple or SkyCoord
        Center of the reference cutout. If tuple, interpreted as (x, y) in reference pixels.
    size : int or tuple
        Size of the reference cutout in reference-image pixels.
    ref_good_pixel_mask : 2D bool array, optional
        Valid-pixel mask for the reference image.
    moving_good_pixel_mask : 2D bool array, optional
        Valid-pixel mask for the moving image.
    weights : 2D array, optional
        Optional weights on the reference grid.
    max_shift_arcsec : float
        Search half-range in arcsec.
    coarse_step_pix : float
        Coarse search step in reference-grid pixels.
    min_valid_pixels : int
        Minimum number of valid overlap pixels required.
    highpass_sigma_pix : float, optional
        If set, subtract a Gaussian-smoothed version from both images before fitting.
        Useful for HST vs JWST where color gradients can bias the fit.
    """

    ref_wcs = WCS(ref_hdu.header).celestial
    moving_wcs = WCS(moving_hdu.header).celestial

    ref_data = np.asarray(ref_hdu.data, dtype=float)
    moving_data = np.asarray(moving_hdu.data, dtype=float)

    # 1) Reference cutout
    ref_cut = Cutout2D(
        data=ref_data,
        position=center,
        size=size,
        wcs=ref_wcs,
        mode="partial",
        fill_value=np.nan,
    )
    ref_img = ref_cut.data

    # 2) Moving cutout: make it larger to allow for large shifts
    moving_cutout_size = _compute_moving_cutout_size(
        ref_wcs=ref_cut.wcs,
        moving_wcs=moving_wcs,
        ref_size=size,
        max_shift_arcsec=max_shift_arcsec,
        pad_arcsec=5.0,
    )

    moving_cut = Cutout2D(
        data=moving_data,
        position=center,
        size=moving_cutout_size,
        wcs=moving_wcs,
        mode="partial",
        fill_value=np.nan,
    )
    moving_img = moving_cut.data

    # 3) Moving-image valid mask in native frame
    if moving_good_pixel_mask is None:
        moving_valid_native = np.isfinite(moving_img)
    else:
        if moving_good_pixel_mask.shape == moving_hdu.data.shape:
            moving_mask_cut = Cutout2D(
                data=moving_good_pixel_mask.astype(bool),
                position=center,
                size=moving_cutout_size,
                wcs=moving_wcs,
                mode="partial",
                fill_value=False,
            ).data.astype(bool)
        else:
            moving_mask_cut = moving_good_pixel_mask.astype(bool)

        moving_valid_native = np.isfinite(moving_img) & moving_mask_cut

    moving_img_filled = np.where(moving_valid_native, moving_img, 0.0)

    # 4) Reproject moving image and moving valid mask once
    moving_reproj, footprint = reproject_interp(
        (moving_img_filled, moving_cut.wcs),
        ref_cut.wcs,
        shape_out=ref_img.shape,
    )

    moving_valid_reproj, _ = reproject_interp(
        (moving_valid_native.astype(float), moving_cut.wcs),
        ref_cut.wcs,
        shape_out=ref_img.shape,
    )

    valid_ref = np.isfinite(ref_img)
    valid_moving = np.isfinite(moving_reproj) & (footprint > 0) & (moving_valid_reproj > 0.999)

    # 5) Reference valid mask
    if ref_good_pixel_mask is None:
        fit_mask = np.ones(ref_img.shape, dtype=bool)
    else:
        if ref_good_pixel_mask.shape == ref_hdu.data.shape:
            fit_mask = Cutout2D(
                data=ref_good_pixel_mask.astype(bool),
                position=center,
                size=size,
                wcs=ref_wcs,
                mode="partial",
                fill_value=False,
            ).data.astype(bool)
        else:
            fit_mask = ref_good_pixel_mask.astype(bool)

    base_mask = valid_ref & fit_mask

    # 6) Optional weights
    if weights is None:
        weight_cut = None
    else:
        if weights.shape == ref_hdu.data.shape:
            weight_cut = Cutout2D(
                data=weights.astype(float),
                position=center,
                size=size,
                wcs=ref_wcs,
                mode="partial",
                fill_value=0.0,
            ).data.astype(float)
        else:
            weight_cut = weights.astype(float)

    # 7) Optional high-pass filtering
    if highpass_sigma_pix is not None:
        ref_work = ref_img - gaussian_filter(np.nan_to_num(ref_img, nan=0.0), highpass_sigma_pix)
        moving_work = moving_reproj - gaussian_filter(np.nan_to_num(moving_reproj, nan=0.0), highpass_sigma_pix)
    else:
        ref_work = ref_img
        moving_work = moving_reproj

    # 8) Search range in reference-grid pixels
    pixscale = np.mean(proj_plane_pixel_scales(ref_cut.wcs)) * u.deg
    max_shift_pix = (max_shift_arcsec * u.arcsec / pixscale).decompose().value

    def objective(p):
        dx, dy = p

        shifted, valid_shifted = _shift_image_and_valid(
            moving_work, valid_moving, dx, dy, order=1
        )

        use = base_mask & valid_shifted
        if np.count_nonzero(use) < min_valid_pixels:
            return np.inf

        ref_use = ref_work[use]
        model_use = shifted[use]

        if weight_cut is None:
            w = None
        else:
            w = weight_cut[use]

        chi2, _, _ = _fit_scale_and_background(ref_use, model_use, w=w)
        return chi2

    # 9) Coarse grid search
    trial = np.arange(-max_shift_pix, max_shift_pix + coarse_step_pix, coarse_step_pix)
    best0 = (0.0, 0.0)
    bestv = np.inf

    for dx in trial:
        for dy in trial:
            val = objective((dx, dy))
            if val < bestv:
                bestv = val
                best0 = (dx, dy)

    # 10) Local refinement
    res = minimize(
        objective,
        x0=np.array(best0),
        method="Powell",
        bounds=[(-max_shift_pix, max_shift_pix),
                (-max_shift_pix, max_shift_pix)],
        options={"xtol": 1e-2, "ftol": 1e-4},
    )

    dx_best, dy_best = res.x

    shifted_best, valid_shifted = _shift_image_and_valid(
        moving_reproj, valid_moving, dx_best, dy_best, order=1
    )
    use = base_mask & valid_shifted

    if weight_cut is None:
        w = None
    else:
        w = weight_cut[use]

    chi2_best, scale_best, bkg_best = _fit_scale_and_background(
        ref_img[use], shifted_best[use], w=w
    )

    # 11) Convert best pixel shift on the reference grid to a sky offset
    ny, nx = ref_img.shape
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)

    sky0 = ref_cut.wcs.pixel_to_world(cx, cy)
    sky1 = ref_cut.wcs.pixel_to_world(cx + dx_best, cy + dy_best)

    dlon, dlat = sky0.spherical_offsets_to(sky1)

    # 12) Apply shift to moving image header
    crval0 = SkyCoord(
        moving_hdu.header["CRVAL1"] * u.deg,
        moving_hdu.header["CRVAL2"] * u.deg,
        frame=sky0.frame
    )
    crval1 = crval0.spherical_offsets_by(dlon, dlat)

    aligned_header = moving_hdu.header.copy()
    aligned_header["CRVAL1"] = crval1.ra.deg
    aligned_header["CRVAL2"] = crval1.dec.deg

    return {
        "dx_pix": dx_best,
        "dy_pix": dy_best,
        "dRA_arcsec": dlon.to_value(u.arcsec),
        "dDec_arcsec": dlat.to_value(u.arcsec),
        "chi2": chi2_best,
        "scale": scale_best,
        "background": bkg_best,
        "ref_cutout": ref_img,
        "moving_reprojected": moving_reproj,
        "moving_aligned_on_ref_grid": scale_best * shifted_best + bkg_best,
        "fit_mask": use,
        "aligned_header": aligned_header,
        "optimizer_result": res,
    }


if __name__ == "__main__":
    # Example: HST F814W as reference, JWST NIRCam as moving image
    hst_hdu = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/HST/MAST_2026-03-10T1856/HST/u2j20e07t/u2j20e07t_drw.fits')[1]

    # CHANGE THIS TO THE ACTUAL JWST NIRCam MOSAIC, not the IFU file
    nircam_hdu = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/photometry/f200w_ifu_coadd_NOAGN_aligned.fits')[0]

    initial_center_sky = SkyCoord(
        ra=189.9976182 * u.deg,
        dec=-11.6230435 * u.deg,
        frame='icrs'
    )

    hst_data = np.asarray(hst_hdu.data, dtype=float)
    good_mask_hst = np.isfinite(hst_data)

    nircam_data = np.asarray(nircam_hdu.data, dtype=float)
    good_mask_nircam = np.isfinite(nircam_data)

    result = align_image_to_reference(
        ref_hdu=hst_hdu,
        moving_hdu=nircam_hdu,
        center=initial_center_sky,
        size=(600, 600),              # use a larger patch for HST↔JWST
        ref_good_pixel_mask=good_mask_hst,
        moving_good_pixel_mask=good_mask_nircam,
        max_shift_arcsec=20.0,
        coarse_step_pix=2.0,
        min_valid_pixels=500,
        highpass_sigma_pix=8.0,       # helpful for F814W vs F200W morphology differences
    )

    print("dRA  =", result["dRA_arcsec"], "arcsec")
    print("dDec =", result["dDec_arcsec"], "arcsec")

    aligned_header = result["aligned_header"]
    aligned_hdu = fits.PrimaryHDU(data=nircam_hdu.data, header=aligned_header)
    aligned_hdu.writeto(
        '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/JWST_to_hst_alignment.fits',
        overwrite=True
    )