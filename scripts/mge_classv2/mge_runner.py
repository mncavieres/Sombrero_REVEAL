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
        '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/photometry/f200w_ifu_coadd_masked.fits'
    )[0].data

    dust_mask = fits.open(
        '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/dust_mask/f200_mask_1.fits'
    )[0].data

    nan_mask = np.isnan(img_f200)
    if np.any(nan_mask):
        print(f"Found {np.sum(nan_mask)} NaN pixels in the image. Replacing with 0 and adding to dust mask.")
        img_f200[nan_mask] = 0.0
        dust_mask = dust_mask | nan_mask

    checkplot_dir = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_test_nosky_0deg_pa_positive_gauss"
    _ensure_dir(checkplot_dir)

    runner = MGEFitter(
        img_f200,
        dust_mask,
        pixel_scale=0.031,
        subtract_sky=True,
        linear=False,
        ngauss=30,
        plot=True,
        checkplot_dir=checkplot_dir,
        cache_dir=checkplot_dir,
        prefix="sombrero_f200",
        contour_half_size_arcsec=80,
        contour_oversample=1,
        n_sectors=19,
        allow_negative=False,
        bulge_disk=False,
    )

    x_peak, y_peak = brightest_pixel_near(
        img_f200, 7538, 7333, halfsize=40, goodmask=runner.goodmask
    )

    print(f"Initial guess for galaxy center (x, y) [pix]: ({x_peak:.2f}, {y_peak:.2f})")
    print(f"Pixel value at center img[y, x] = img[{int(round(y_peak))}, {int(round(x_peak))}] = "
          f"{img_f200[int(round(y_peak)), int(round(x_peak))]}")

    runner.set_manual_geometry(
        center=(x_peak, y_peak),   # correct public convention: (x, y)
        pa_deg=0,#90.78185872429874,#87.2,
        eps=0.7060956459920877, #0.35,
        theta_deg=90.78185872429874,#87.2,
    )

    print(f"Stored manual center in runner: (x, y) = ({runner.xc:.2f}, {runner.yc:.2f})")

    # remember to set things to force=True if you want to re-run steps that have already been cached
    #runner.run_sectors()  # this will use the new geometry and overwrite any previous sectors
    runner.run_fit() # 
    runner.run_fit()

    runner.set_deprojection(
        inclination_deg=86.12, # inclination in degrees (0 = face-on, 90 = edge-on)
        distance=9.55,      # Mpc if passed as a plain float
        sb_unit="MJy/sr",
    )


    dmge = runner.deprojected_mge

    # Angular LOS profile in arcsec
    res = dmge.central_los_aperture(
        aperture=5.5, # arcsec
        frac=0.99, 
        physical=False, 
        checkplot_dir=runner.checkplot_dir,
        prefix=f"{runner.prefix}_central_ap5p5as_99frac",
        dpi=runner.dpi,
        save_plots=True,
    )

    print(res)

    # Physical LOS profile in pc
    res_phys = dmge.central_los_aperture(
        aperture=10.5 * 9.55e6 * (u.arcsec).to(u.rad),   # convert 10.5" to pc
        frac=0.99,
        physical=True,
    )

    print(res_phys)