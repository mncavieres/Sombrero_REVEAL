import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/scripts/mge_classv2')
from mge_gen3 import MGEFitter
from astropy.io import fits
import numpy as np
import astropy.units as u   

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def brightest_pixel_near(img, x0, y0, halfsize=50, goodmask=None):
    """
    Return brightest pixel near an initial guess.

    Public convention:
        input/output are (x, y) = (col, row)

    NumPy indexing:
        img[y, x]
    """
    ny, nx = img.shape

    x1 = max(0, int(round(x0 - halfsize)))
    x2 = min(nx, int(round(x0 + halfsize + 1)))
    y1 = max(0, int(round(y0 - halfsize)))
    y2 = min(ny, int(round(y0 + halfsize + 1)))

    cut = np.array(img[y1:y2, x1:x2], copy=True)

    if goodmask is not None:
        gm = goodmask[y1:y2, x1:x2]
        cut[~gm] = -np.inf

    iy, ix = np.unravel_index(np.nanargmax(cut), cut.shape)
    return x1 + ix, y1 + iy


if __name__ == "__main__":

    img_f200 = fits.open(
        '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/HST/MAST_2026-03-10T1856/HST/u2j20e07t/u2j20e07t_drw.fits'
    )[1].data

    dust_mask = fits.open(
        '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/dust_mask/hst_dust_mask_f814w.fits'
    )[0].data

    nan_mask = np.isnan(img_f200)
    if np.any(nan_mask):
        print(f"Found {np.sum(nan_mask)} NaN pixels in the image. Replacing with 0 and adding to dust mask.")
        img_f200[nan_mask] = 0.0
        #dust_mask = dust_mask | nan_mask
        # make the NaN pixels be 0 in the dust mask (i.e. masked out)
        dust_mask[nan_mask] = 0

    # save the dust mask as a fits file
    hdu = fits.PrimaryHDU(dust_mask.astype(np.uint8), header=fits.open(
        '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/dust_mask/hst_dust_mask_f814w.fits')[0].header)
    hdu.writeto("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/dust_mask/hst_dust_mask_f814w_binary.fits", overwrite=True)

    # make the dustmask boolean
    #dust_mask = dust_mask.astype(bool)

    checkplot_dir = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/MGE_HST_f814w"
    _ensure_dir(checkplot_dir)

    runner = MGEFitter(
        img_f200,
        dust_mask,
        dust_mask_is_bad=False,  # in this case, 1 means dust (bad), 0 means good
        pixel_scale= 0.1, # pixel scale for HST/WFPC2 in arcsec/pixel is 0.0996  
        subtract_sky=True,
        linear=False,
        ngauss=30,
        plot=True,
        checkplot_dir=checkplot_dir,
        cache_dir=checkplot_dir,
        prefix="sombrero_f814w",
        contour_half_size_arcsec=10,
        contour_oversample=1,
        n_sectors=19,
        allow_negative=False,
        bulge_disk=False,
    )

    #runner.run_all(force_find=True, force_sectors=True, force_fit=True)
    x_peak, y_peak = brightest_pixel_near(
        img_f200, 2720.5, 2392.85, halfsize=10, goodmask=runner.goodmask
    )

    print(f"Initial guess for galaxy center (x, y) [pix]: ({x_peak:.2f}, {y_peak:.2f})")
    print(f"Pixel value at center img[y, x] = img[{int(round(y_peak))}, {int(round(x_peak))}] = "
          f"{img_f200[int(round(y_peak)), int(round(x_peak))]}")

    runner.set_manual_geometry(
        center=(2720.5, 2392.85),   # correct public convention: (x, y)
        pa_deg=0,#90.78185872429874,#87.2,
        eps=0.7060956459920877, #0.35,
        theta_deg=90.78185872429874,#87.2,
    )

    # print(f"Stored manual center in runner: (x, y) = ({runner.xc:.2f}, {runner.yc:.2f})")

    # # remember to set things to force=True if you want to re-run steps that have already been cached
    #runner.run_find(force=True)  # this will use the new geometry and overwrite any previous sectors
    runner.run_sectors()  # this will use the new geometry and overwrite any previous sectors
    runner.run_fit() # 

    # save the mge parameters to a text file
    
