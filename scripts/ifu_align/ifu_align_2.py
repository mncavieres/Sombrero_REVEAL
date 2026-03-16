import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from scipy.ndimage import shift as ndi_shift
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


def align_ifu_to_nircam(
    nircam_hdu,
    ifu_hdu,
    center,
    size,
    good_pixel_mask=None,
    weights=None,
    max_shift_arcsec=1.0,
    coarse_step_pix=0.5,
    min_valid_pixels=100,
):
    """
    Align IFU synthetic image to a NIRCam image.

    Parameters
    ----------
    nircam_hdu : FITS HDU
        Reference image (e.g. NIRCam F200W).
    ifu_hdu : FITS HDU
        IFU-derived synthetic F200W image.
    center : tuple or SkyCoord
        Center of the NIRCam cutout. If tuple, interpreted as (x, y) in NIRCam pixels.
    size : int, tuple, or Quantity
        Cutout size passed to Cutout2D.
    good_pixel_mask : 2D bool array, optional
        True where pixels are allowed in the fit.
        Use this to exclude saturation, bad pixels, AGN core, etc.
        Can be the full NIRCam image shape or already the cutout shape.
    weights : 2D array, optional
        Optional weights on the NIRCam grid. Same shape convention as good_pixel_mask.
    max_shift_arcsec : float
        Search half-range in arcsec.
    coarse_step_pix : float
        Coarse grid step in pixels before local refinement.
    min_valid_pixels : int
        Minimum number of valid overlap pixels required in a trial.
    """

    ref_wcs = WCS(nircam_hdu.header).celestial
    ifu_wcs = WCS(ifu_hdu.header).celestial

    # 1) Make a cutout on the reference image only
    ref_cut = Cutout2D(
        data=nircam_hdu.data.astype(float),
        position=center,
        size=size,
        wcs=ref_wcs
    )

    ref_img = ref_cut.data

    # 2) Reproject the IFU image ONCE onto the exact cutout WCS
    ifu_reproj, footprint = reproject_interp(
        (ifu_hdu.data.astype(float), ifu_wcs),
        ref_cut.wcs,
        shape_out=ref_img.shape,
    )

    valid_ref = np.isfinite(ref_img)
    valid_ifu = np.isfinite(ifu_reproj) & (footprint > 0)

    # 3) Build fit mask
    if good_pixel_mask is None:
        fit_mask = np.ones(ref_img.shape, dtype=bool)
    else:
        if good_pixel_mask.shape == nircam_hdu.data.shape:
            fit_mask = Cutout2D(
                data=good_pixel_mask.astype(bool),
                position=center,
                size=size,
                wcs=ref_wcs
            ).data.astype(bool)
        else:
            fit_mask = good_pixel_mask.astype(bool)

    base_mask = valid_ref & fit_mask

    # 4) Optional weights
    if weights is None:
        weight_cut = None
    else:
        if weights.shape == nircam_hdu.data.shape:
            weight_cut = Cutout2D(
                data=weights.astype(float),
                position=center,
                size=size,
                wcs=ref_wcs
            ).data.astype(float)
        else:
            weight_cut = weights.astype(float)

    # 5) Convert search range from arcsec to reference-grid pixels
    pixscale = np.mean(proj_plane_pixel_scales(ref_cut.wcs)) * u.deg
    max_shift_pix = (max_shift_arcsec * u.arcsec / pixscale).decompose().value

    def objective(p):
        dx, dy = p

        shifted, valid_shifted = _shift_image_and_valid(
            ifu_reproj, valid_ifu, dx, dy, order=1
        )

        use = base_mask & valid_shifted
        if np.count_nonzero(use) < min_valid_pixels:
            return np.inf

        ref_use = ref_img[use]
        model_use = shifted[use]

        if weight_cut is None:
            w = None
        else:
            w = weight_cut[use]

        chi2, _, _ = _fit_scale_and_background(ref_use, model_use, w=w)
        return chi2

    # 6) Coarse grid search in pixel space (cheap)
    trial = np.arange(-max_shift_pix, max_shift_pix + coarse_step_pix, coarse_step_pix)
    best0 = (0.0, 0.0)
    bestv = np.inf

    for dx in trial:
        for dy in trial:
            val = objective((dx, dy))
            if val < bestv:
                bestv = val
                best0 = (dx, dy)

    # 7) Local refinement
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
        ifu_reproj, valid_ifu, dx_best, dy_best, order=1
    )
    use = base_mask & valid_shifted

    if weight_cut is None:
        w = None
    else:
        w = weight_cut[use]

    chi2_best, scale_best, bkg_best = _fit_scale_and_background(
        ref_img[use], shifted_best[use], w=w
    )

    # 8) Convert the best pixel shift on the NIRCam grid to a sky offset
    ny, nx = ref_img.shape
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)

    sky0 = ref_cut.wcs.pixel_to_world(cx, cy)
    sky1 = ref_cut.wcs.pixel_to_world(cx + dx_best, cy + dy_best)

    dlon, dlat = sky0.spherical_offsets_to(sky1)

    # 9) Apply that sky offset to the IFU header via CRVAL
    crval0 = SkyCoord(
        ifu_hdu.header["CRVAL1"] * u.deg,
        ifu_hdu.header["CRVAL2"] * u.deg,
        frame=sky0.frame
    )
    crval1 = crval0.spherical_offsets_by(dlon, dlat)

    aligned_header = ifu_hdu.header.copy()
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
        "nircam_cutout": ref_img,
        "ifu_reprojected": ifu_reproj,
        "ifu_aligned_on_nircam_grid": scale_best * shifted_best + bkg_best,
        "fit_mask": use,
        "aligned_header": aligned_header,
        "optimizer_result": res,
    }

if __name__ == "__main__":
    # load the F200W mosaic and the IFU-derived photometric image
    f200w_hdu = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mosaics/f200_mosaic.fits')[0]
    ifu_f200w_hdu = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/photometry/f200_from_ifu_NOAGN.fits')[1]
    # initial guess for the center (x0, y0) in NIRCam sky coordinates, from visual inspection
    initial_center_sky = SkyCoord(ra=189.9976182 * u.deg, dec=-11.6230435 * u.deg, frame='icrs')

    # make a good pixel mask that excludes NaNs from the F200 mosaic
    f200w_data = f200w_hdu.data
    nan_mask = np.isnan(f200w_data)
    if np.any(nan_mask):
        print(f"Found {np.sum(nan_mask)} NaN pixels in the F200W mosaic. These will be masked out in the fit.")
    good_mask = ~nan_mask

    # same for the IFU image
    ifu_data = ifu_f200w_hdu.data
    nan_mask_ifu = np.isnan(ifu_data)
    if np.any(nan_mask_ifu):
        print(f"Found {np.sum(nan_mask_ifu)} NaN pixels in the IFU photometric image. These will be masked out in the fit.")
    good_mask_ifu = ~nan_mask_ifu


    result = align_ifu_to_nircam(
    nircam_hdu=f200w_hdu,
    ifu_hdu=ifu_f200w_hdu,
    center=initial_center_sky,     # in NIRCam pixels, or SkyCoord
    size=(200, 200),
    good_pixel_mask=good_mask,
    max_shift_arcsec=1.0,
    coarse_step_pix=0.5,
    )

    print(result["dRA_arcsec"], result["dDec_arcsec"])
    aligned_header = result["aligned_header"]

    # save the aligned IFU image as a new FITS file
    aligned_hdu = fits.PrimaryHDU(data=ifu_f200w_hdu.data, header=aligned_header)
    aligned_hdu.writeto('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/photometry/f200_from_ifu_NOAGN_aligned.fits', overwrite=True)