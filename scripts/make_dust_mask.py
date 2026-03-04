import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from reproject import reproject_interp, reproject_exact
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_fill_holes, label

def read_sci_dq(path, sci_ext="SCI", dq_ext="DQ"):
    with fits.open(path) as hdul:
        if sci_ext not in hdul:
            sci = hdul[0].data
            wcs = WCS(hdul[0].header)
            dq = None
            return sci, dq, wcs
        sci = np.squeeze(hdul[sci_ext].data).astype(float)
        wcs = WCS(hdul[sci_ext].header)
        dq = None
        if dq_ext in hdul:
            dq = np.squeeze(hdul[dq_ext].data)
    sci[~np.isfinite(sci)] = np.nan
    return sci, dq, wcs

def robust_background(img, border=50):
    # quick robust bg estimate from the border pixels
    ny, nx = img.shape
    rim = np.zeros_like(img, dtype=bool)
    rim[:border, :] = True
    rim[-border:, :] = True
    rim[:, :border] = True
    rim[:, -border:] = True
    vals = img[rim & np.isfinite(img)]
    if vals.size == 0:
        return 0.0
    return np.nanmedian(vals)

def mad_std(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))

def remove_small_regions(mask, min_pixels=200):
    lab, n = label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[counts >= min_pixels] = True
    keep[0] = False
    return keep[lab]

def make_dust_mask(f090_path, f200_path,
                   sci_ext="SCI", dq_ext="DQ",
                   psf_sigma_pix=1.2,
                   smooth_large_pix=30,
                   thresh_sigma=3.0,
                   min_region_pix=300,
                   threads =1):
    # Read
    i200, dq200, w200 = read_sci_dq(f200_path, sci_ext=sci_ext, dq_ext=dq_ext)
    i090, dq090, w090 = read_sci_dq(f090_path, sci_ext=sci_ext, dq_ext=dq_ext)

    # Background subtract (important before ratios)
    i200 = i200 - robust_background(i200)
    i090 = i090 - robust_background(i090)

    # Reproject F090W onto F200W grid
    i090_r, _ = reproject_interp((i090, w090), w200, shape_out=i200.shape,
                                 parallel=threads)

    # (Optional) PSF match: convolve both to a common PSF.
    # Here: lightly smooth both so noise/cosmic rays don’t dominate the ratio.
    i200_s = gaussian_filter(i200, psf_sigma_pix)
    i090_s = gaussian_filter(i090_r, psf_sigma_pix)

    # Ratio -> color (dust makes ratio smaller -> color larger)
    eps = np.nanpercentile(i200_s[np.isfinite(i200_s)], 1) * 1e-3
    eps = max(eps, 1e-12)
    ratio = (i090_s + eps) / (i200_s + eps)
    dmag = -2.5 * np.log10(ratio)

    # Remove large-scale trends (bulge/disk gradient)
    dmag_smooth = gaussian_filter(dmag, smooth_large_pix)
    dmag_resid = dmag - dmag_smooth

    # Threshold using robust scatter
    sig = mad_std(dmag_resid)
    dust = dmag_resid > (thresh_sigma * sig)

    # Add DQ bad pixels (conservative: dq != 0)
    bad = np.zeros_like(dust, dtype=bool)
    if dq200 is not None:
        bad |= (dq200 != 0)
    if dq090 is not None:
        # reprojecting dq is messy; simplest is to just include dq090 on its own grid if you want,
        # but typically dq200 already captures most issues on the fitting image.
        pass

    mask = dust | bad

    # Morphology cleanup
    mask = binary_opening(mask, iterations=1)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)

    # Remove tiny islands
    mask = remove_small_regions(mask, min_pixels=min_region_pix)

    return mask, dmag, dmag_resid, i200

def show_with_zscale_log(img, ax=None, title=""):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    good = img[np.isfinite(img)]
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(good)

    # "Sombrero log": show log-like scaling safely (clip to positive)
    img_pos = img.copy()
    img_pos[img_pos <= 0] = np.nan
    # if mostly positive, use log10 for display
    disp = np.log10(img_pos)

    im = ax.imshow(disp, origin="lower", vmin=np.nanpercentile(disp, 5), vmax=np.nanpercentile(disp, 99.5))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax

if __name__ == "__main__":
    f090_path = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mosaics/f090_mosaic.fits"
    f200_path = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mosaics/f200_mosaic.fits"

    

    threads = 1
    mask, dmag, dmag_resid, ref = make_dust_mask(f090_path, f200_path, threads=threads)

    # fig, ax = plt.subplots(1, 3, figsize=(18,6))
    # show_with_zscale_log(i200, ax=ax[0], title="F200W")
    # show_with_zscale_log(dmag, ax=ax[1], title="Raw dmag")
    # show_with_zscale_log(dmag_resid, ax=ax[2], title="Residual dmag")
    # plt.show()

    # here we will set every NaN to the number 3 so that its included for later
    dmag_no_nan = np.where(np.isfinite(dmag), dmag, 3)
    mask_new = dmag_no_nan > 1
    # make the regions in the mask with NaN be true

    masked_ref = np.copy(ref)
    masked_ref[mask_new] = np.nan  # mask out dusty regions

    # save the mask, the dmag, and the masked reference image to a new FITS file
    hdu_mask = fits.PrimaryHDU(mask.astype(np.uint8), header=WCS(fits.open(f200_path)[0].header).to_header())
    hdu_dmag = fits.ImageHDU(dmag, header=WCS(fits.open(f200_path)[0].header).to_header())
    hdu_masked_ref = fits.ImageHDU(masked_ref, header=WCS(fits.open(f200_path)[0].header).to_header())
    hdu_mask.writeto("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/f090_f200/dust_mask.fits", overwrite=True)
    hdu_dmag.writeto("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/f090_f200/dmag.fits", overwrite=True)
    hdu_masked_ref.writeto("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/f090_f200/masked_ref.fits", overwrite=True)


    fig, ax = plt.subplots(figsize=(10, 10))
    cb = ax.imshow(dmag_no_nan, origin="lower", cmap="inferno",
                vmin=np.nanpercentile(dmag_no_nan, 5), vmax=np.nanpercentile(dmag_no_nan, 99.5))
    plt.colorbar(cb, ax=ax)
    ax.set_title("F090W/F200W color")
    plt.show()


    # mask out the image regions identified as dusty
    fig, ax = plt.subplots(figsize=(10, 10))
   
    ax.imshow(masked_ref, origin="lower", cmap="gray",
            vmin=np.nanpercentile(masked_ref, 5), vmax=np.nanpercentile(masked_ref, 99.5))
    ax.set_title("Reference with dusty regions masked")
    plt.show()