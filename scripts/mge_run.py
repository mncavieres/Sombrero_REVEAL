import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/scripts')
from mge_general import MGEFitter
from astropy.io import fits
import numpy as np


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def brightest_pixel_near(img, x0, y0, halfsize=50, goodmask=None):
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

    img_f200 = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/photometry/f200w_ifu_coadd_masked.fits')[0].data
    dust_mask = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/dust_mask/f200_mask_1.fits')[0].data

    nan_mask = np.isnan(img_f200)
    if np.any(nan_mask):
        print(f"Found {np.sum(nan_mask)} NaN pixels in the image. Replacing with 0 and adding to dust mask.")
        img_f200[nan_mask] = 0.0
        dust_mask = dust_mask | nan_mask

    checkplot_dir = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_test"
    _ensure_dir(checkplot_dir)


    runner = MGEFitter(
    img_f200,
    dust_mask,
    pixel_scale=0.031,
    subtract_sky=False,
    linear=False,
    ngauss=20,
    plot=True,
    checkplot_dir=checkplot_dir,
    cache_dir=checkplot_dir,
    prefix="sombrero_f200",
    contour_half_size_arcsec=80,
    contour_oversample=1,
    n_sectors=24,
    #center=(7537.219884279368, 7332.548985227105),   # (x, y)
    #pa_deg=87.2,
    #eps=0.35,
    #theta_deg=87.2,            # optional
    )

    x_peak, y_peak = brightest_pixel_near(img_f200, 7538, 7333, halfsize=40, goodmask=runner.goodmask)

    print(f"Initial guess for galaxy center (x, y) [pix]: ({x_peak:.2f}, {y_peak:.2f})")

    runner.set_manual_geometry(center=(y_peak, x_peak), pa_deg=87.2, eps=0.35, theta_deg=87.2)
    runner.run_sectors(force=True)
    runner.run_fit(force=True)

    # res = runner.run_all(
    #                      force_sectors=True,
    #                        force_fit=True,
    #                        )