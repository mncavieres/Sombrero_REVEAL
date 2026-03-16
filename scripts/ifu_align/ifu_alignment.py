import matplotlib.pyplot as plt
import os
import sys
from astropy.io import fits
import numpy as np
import astropy.units as u   
# reproject functions
from reproject import reproject_interp
from astropy.wcs import WCS

def chi2(img1, img2, mask=None):
    if mask is not None:
        img1 = img1[mask]
        img2 = img2[mask]
    return np.sum((img1 - img2)**2)

def extract_cutout(img, x0, y0, halfsize=50):
    ny, nx = img.shape

    x1 = max(0, int(round(x0 - halfsize)))
    x2 = min(nx, int(round(x0 + halfsize + 1)))
    y1 = max(0, int(round(y0 - halfsize)))
    y2 = min(ny, int(round(y0 + halfsize + 1)))

    return img[y1:y2, x1:x2]

def plot_with_wcs(img1, img2, wcs1, wcs2, title1="Image 1", title2="Image 2"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': wcs1})
    axes[0].imshow(img1, origin='lower', cmap='viridis')
    axes[0].set_title(title1)
    axes[0].set_xlabel('RA')
    axes[0].set_ylabel('Dec')

    axes[1] = plt.subplot(122, projection=wcs2)
    axes[1].imshow(img2, origin='lower', cmap='viridis')
    axes[1].set_title(title2)
    axes[1].set_xlabel('RA')
    axes[1].set_ylabel('Dec')

    plt.tight_layout()
    plt.show()

def chi2_images(img1, img2, wcs1, wcs2, x0, y0, x1, y1,
     halfsize=50, mask=None, use_full_im1=False, use_full_im2=False):
    # geta cutout around the initial guess position from both images
    if use_full_im1:
        cutout1 = img1
    else:
        cutout1 = extract_cutout(img1, x0, y0, halfsize)
    if use_full_im2:
        cutout2 = img2
    else:
        cutout2 = extract_cutout(img2, x1, y1, halfsize)

    # Reproject cutout2 to the WCS of cutout1
    cutout2_reproj, _ = reproject_interp((cutout2, wcs2), wcs1, shape_out=cutout1.shape)

    # Compute chi-squared between the two cutouts
    chi2_value = chi2(cutout1, cutout2_reproj, mask=mask)

    return chi2_value



def align_images_by_chi2(hdu1, hdu2, x0, y0, halfsize=50,
  mask=None, shift_range=np.arange(-1, 1, 0.1), use_full_im1=False, use_full_im2=False):
    """Align img2 to img1 by minimizing chi-squared in a cutout around (x0, y0) in img1.
    
    This will adjust the headers of img2 to align with img1, by changing the CRVAL1
    and CRVAL2 values to shift the WCS. It will not change the pixel data of img2,
    just the WCS header keywords.

    It will search for the best alignment by trying small shifts in the WCS of img2 and
    reprojecting to the WCS of img1, then computing chi-squared in the cutout region. 
    The shift that minimizes chi-squared will be applied to the WCS of img2. And we 
    will return the best chi-squared value and the corresponding shift in arcseconds,
    along with the updated header for img2.

    """
    # Get the WCS objects for both images
    wcs1 = WCS(hdu1.header)
    wcs2 = WCS(hdu2.header)

    # Convert the initial guess pixel coordinates (x0, y0) in img1 to world coordinates (RA, Dec)
    ra_dec = wcs1.pixel_to_world(x0, y0)
    ra0, dec0 = ra_dec.ra.deg, ra_dec.dec.deg
    print(f"Initial guess in img1: pixel ({x0}, {y0}) -> world (RA={ra0:.6f}, Dec={dec0:.6f})")

    # compute the initial chi-squared value at the initial guess (no shift)
    chi2_initial = chi2_images(hdu1.data, hdu2.data, wcs1, wcs2, x0, y0, x0, y0, halfsize=halfsize, mask=mask, use_full_im1=use_full_im1, use_full_im2=use_full_im2)
    print(f"Initial chi-squared at initial guess: {chi2_initial:.2f}")

    # compute the chi2-squared values for a grid of shifts in the WCS of img2
    best_chi2 = chi2_initial
    best_shift_ra = 0.0
    best_shift_dec = 0.0
    for shift_ra in shift_range:
        for shift_dec in shift_range:
            # Create a copy of the header of img2 to modify
            new_header = hdu2.header.copy()

            # Apply the shift to the CRVAL1 and CRVAL2 keywords in the header of img2
            new_header['CRVAL1'] += shift_ra / 3600.0  # convert arcsec to degrees
            new_header['CRVAL2'] += shift_dec / 3600.0  # convert arcsec to degrees

            # Create a new WCS object with the modified header
            wcs2_shifted = WCS(new_header)

            # Compute chi-squared with the shifted WCS
            chi2_value = chi2_images(hdu1.data, hdu2.data, wcs1, wcs2_shifted, x0, y0, x0, y0, halfsize=halfsize, mask=mask, use_full_im1=use_full_im1, use_full_im2=use_full_im2)

            # Check if this is the best chi-squared so far
            if chi2_value < best_chi2:
                best_chi2 = chi2_value
                best_shift_ra = shift_ra
                best_shift_dec = shift_dec

    print(f"Best chi-squared: {best_chi2:.2f} at shift (RA={best_shift_ra:.2f} arcsec, Dec={best_shift_dec:.2f} arcsec)")

    # now do a second pass around the best shift with a finer grid to refine the alignment
    fine_shift_range = np.arange(-0.5, 0.5, 0.05)  # finer grid around the best shift
    for shift_ra in fine_shift_range:
        for shift_dec in fine_shift_range:
            # Create a copy of the header of img2 to modify
            new_header = hdu2.header.copy()

            # Apply the shift to the CRVAL1 and CRVAL2 keywords in the header of img2
            new_header['CRVAL1'] += (best_shift_ra + shift_ra) / 3600.0  # convert arcsec to degrees
            new_header['CRVAL2'] += (best_shift_dec + shift_dec) / 3600.0  # convert arcsec to degrees

            # Create a new WCS object with the modified header
            wcs2_shifted = WCS(new_header)

            # Compute chi-squared with the shifted WCS
            chi2_value = chi2_images(hdu1.data, hdu2.data, wcs1, wcs2_shifted, x0, y0, x0, y0, halfsize=halfsize, mask=mask, use_full_im1=use_full_im1, use_full_im2=use_full_im2)

            # Check if this is the best chi-squared so far
            if chi2_value < best_chi2:
                best_chi2 = chi2_value
                best_shift_ra = best_shift_ra + shift_ra
                best_shift_dec = best_shift_dec + shift_dec
            
    print(f"Refined best chi-squared: {best_chi2:.2f} at shift (RA={best_shift_ra:.2f} arcsec, Dec={best_shift_dec:.2f} arcsec)")

    # Apply the best shift to the header of img2
    aligned_header = hdu2.header.copy()
    aligned_header['CRVAL1'] += best_shift_ra / 3600.0  # convert arcsec to degrees
    aligned_header['CRVAL2'] += best_shift_dec / 3600.0  # convert arcsec to degrees

    return best_chi2, best_shift_ra, best_shift_dec, aligned_header