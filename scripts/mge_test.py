import numpy as np
import mgefit as mge
from astropy.io import fits

def fit_mge_from_f200(
    img_f200,
    dust_mask,
    *,
    pixel_scale=1.0,                 # arcsec/pix (only affects plot axes)
    dust_mask_is_bad=True,           # True if your dust_mask marks dusty/bad pixels
    subtract_sky=False,              # only use if your image is mostly sky (see note below)
    find_fraction=0.03,              # fraction of brightest pixels used by find_galaxy
    find_binning=5,
    n_sectors=19,
    minlevel=0,                      # stop profile when sector-mean <= minlevel
    # PSF (optional): circular MGE in *pixels* (sigmapsf) with weights summing to 1 (normpsf)
    sigmapsf=0.0,
    normpsf=1.0,
    # MGE fit controls
    linear=True,                     # robust; recommended by mgefit docs
    ngauss=400,                      # max allowed Gaussians (linear=True wants many)
    qbounds=(0.05, 1.0),
    outer_slope=4,
    plot=True,
    quiet=False,
):
    """
    Fit an MGE model to a 2D F200 image, excluding dusty pixels via a mask.

    Returns a dict with:
      - geometry (center, pa, eps)
      - s: sectors_photometry object (radius, angle, counts)
      - m: fit_sectors object (m.sol = total_counts, sigma_pix, q_obs)
      - table: (surf, sigma_pix, sigma_arcsec, q_obs, total_counts)
    """
    img = np.asarray(img_f200, dtype=float)
    if img.ndim != 2:
        raise ValueError("img_f200 must be a 2D array")

    mask = np.asarray(dust_mask)
    if mask.shape != img.shape:
        raise ValueError("dust_mask must have the same shape as img_f200")
    if mask.dtype != bool:
        mask = mask.astype(bool)

    # mge.sectors_photometry expects mask=True for *good* pixels.
    goodmask = (~mask) if dust_mask_is_bad else mask

    # Basic sanity: no NaNs in the input image (mgefit routines assert this).
    # For find_galaxy, replace masked (dusty) pixels with the median of unmasked pixels.
    img_for_find = img.copy()
    if np.any(goodmask):
        fill = np.nanmedian(img[goodmask])
    else:
        fill = np.nanmedian(img)
    img_for_find[~goodmask] = fill
    if not np.all(np.isfinite(img_for_find)):
        raise ValueError("Image contains NaN/Inf even after filling masked pixels")

    # Optional sky subtraction (only valid if most pixels are sky background)
    sky_mean = None
    sky_sigma = None
    img_work = img.copy()
    if subtract_sky:
        sky_mean, sky_sigma = mge.sky_level(img_for_find[goodmask], plot=plot, quiet=quiet)
        img_work = img_work - sky_mean
        img_for_find = img_for_find - sky_mean  # keep geometry consistent

    # 1) Find galaxy geometry (center, PA, ellipticity)
    f = mge.find_galaxy(
        img_for_find,
        fraction=find_fraction,
        binning=find_binning,
        plot=plot,
        quiet=quiet
    )
    # Note: xpeak/ypeak are numpy indices (row/col) in mgefit.
    xc, yc = f.xpeak, f.ypeak
    eps = f.eps
    pa = f.pa  # astronomical PA, if your image is N-up


    # 2) Extract sector photometry using your dust mask
    s = mge.sectors_photometry(
        img_work,
        eps=eps,
        ang=pa,
        xc=xc,
        yc=yc,
        mask=goodmask,
        n_sectors=n_sectors,
        minlevel=minlevel,
        plot=plot
    )

    # 3) Fit the MGE to the sector photometry
    m = mge.fit_sectors(
        s.radius, s.angle, s.counts, eps,
        linear=linear,
        ngauss=ngauss,
        qbounds=qbounds,
        outer_slope=outer_slope,
        sigmapsf=sigmapsf,
        normpsf=normpsf,
        scale=pixel_scale,
        plot=plot,
        quiet=quiet
    )

    total_counts, sigma_pix, q_obs = m.sol  # mgefit convention

    # Peak surface brightness (counts/pix) for each Gaussian:
    # surf = total_counts / (2*pi*q_obs*sigma^2)
    surf = total_counts / (2.0 * np.pi * q_obs * sigma_pix**2)
    sigma_arcsec = sigma_pix * float(pixel_scale)

    table = np.vstack([surf, sigma_pix, sigma_arcsec, q_obs, total_counts]).T

    return {
        "center_pix": (xc, yc),
        "pa_deg": pa,
        "eps": eps,
        "sky_mean": sky_mean,
        "sky_sigma": sky_sigma,
        "sectors": s,
        "mgefit": m,
        "table_cols": ["surf_counts_per_pix", "sigma_pix", "sigma_arcsec", "q_obs", "total_counts"],
        "table": table,
    }


# Load image and dust mask

img_f200 = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mosaics/f200_mosaic.fits')[0].data
dust_mask = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/dust_mask/f200_mask_1.fits')[0].data

# find NaNs in the image and replace them with 0 while also adding those pixels to the dust mask
nan_mask = np.isnan(img_f200)
if np.any(nan_mask):
    print(f"Found {np.sum(nan_mask)} NaN pixels in the image. Replacing with 0 and adding to dust mask.")
    img_f200[nan_mask] = 0.0
    dust_mask = dust_mask | nan_mask  # Mark NaN pixels as dusty/bad in the mask


res = fit_mge_from_f200(
    img_f200, dust_mask,
    pixel_scale=0.031,        # e.g., NIRCam SW ~0.031"/pix, adjust to your mosaic
    subtract_sky=False,       # set True only if the image is mostly sky
    # sigmapsf=[1.2, 2.8],      # PSF sigma in pixels (optional)
    # normpsf=[0.7, 0.3],       # weights sum to 1 (optional)
    plot=True
)

print("Center (pix):", res["center_pix"])
print("PA (deg):", res["pa_deg"], "eps:", res["eps"])
print("absdev:", res["mgefit"].absdev)
print(res["table_cols"])
print(res["table"])

# save res to a file for later analysis (e.g., with pandas)
import pickle
with open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_fit_results.pkl', 'wb') as f:
    pickle.dump(res, f)