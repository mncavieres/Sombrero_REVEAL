from reproject.mosaicking import reproject_and_coadd
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_exact
import pandas
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# threads = 8
# f090_path_1 = "/f090_f200/jw06565-o002_t001_nircam_clear-f090w/jw06565-o002_t001_nircam_clear-f090w_i2d.fits"
# f090_path_2 = '/f090_f200/jw06565-o003_t001_nircam_clear-f090w/jw06565-o003_t001_nircam_clear-f090w_i2d.fits'
# f090_path_3 = "/f090_f200/jw06565-o005_t002_nircam_clear-f090w/jw06565-o005_t002_nircam_clear-f090w_i2d.fits"

main_data_path= '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data'
output_path = '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mosaics'
os.makedirs(output_path, exist_ok=True)

# f090_path_1 = os.path.join(main_data_path, f090_path_1)
# f090_path_2 = os.path.join(main_data_path, f090_path_2)
# f090_path_3 = os.path.join(main_data_path, f090_path_3)

# with fits.open(f090_path_1), fits.open(f090_path_2), fits.open(f090_path_3) as (f090_1, f090_2, f090_3):
#     print("All files opened successfully")  
#     # f090_1 = fits.open(f090_path_1)
#     # f090_2 = fits.open(f090_path_2)
#     # f090_3 = fits.open(f090_path_3)

#     # find the optimal WCS and shape for our mosaic
#     wcs, shape = find_optimal_celestial_wcs(
#         [f090_1[1], f090_2[1], f090_3[1]]
#     )

#     # reproject and coadd the images
#     coadd, footprint = reproject_and_coadd([f090_path_1, f090_path_2, f090_path_3],
#                                             output_projection=wcs, shape_out=shape,
#                                             reproject_function=reproject_exact, hdu_in=1,
#                                             parallel=threads
#                                             )

#     # save the mosaic to a new FITS file
#     hdu = fits.PrimaryHDU(coadd, header=wcs.to_header())
#     hdu.writeto(os.path.join(output_path, 'f090_mosaic.fits'), overwrite=True)

# now the f0200

f200w_path_1 = "f090_f200/jw06565-o002_t001_nircam_clear-f200w/jw06565-o002_t001_nircam_clear-f200w_i2d.fits"
f200w_path_2 = "f090_f200/jw06565-o003_t001_nircam_clear-f200w/jw06565-o003_t001_nircam_clear-f200w_i2d.fits"
f200w_path_3 = "f090_f200/jw06565-o005_t002_nircam_clear-f200w/jw06565-o005_t002_nircam_clear-f200w_i2d.fits"

f200w_path_1 = os.path.join(main_data_path, f200w_path_1)
f200w_path_2 = os.path.join(main_data_path, f200w_path_2)
f200w_path_3 = os.path.join(main_data_path, f200w_path_3)




f200w_1 = fits.open(f200w_path_1)
f200w_2 = fits.open(f200w_path_2)
f200w_3 = fits.open(f200w_path_3)
print("All files opened successfully")  

# find the optimal WCS and shape for our mosaic
wcs, shape = find_optimal_celestial_wcs(
    [f200w_1[1], f200w_2[1], f200w_3[1]]
)
print("Optimal WCS and shape found")

print('Reprojecting and coadding images...')

# reproject and coadd the images
coadd, footprint = reproject_and_coadd([f200w_path_1, f200w_path_2, f200w_path_3],
                                        output_projection=wcs, shape_out=shape,
                                        reproject_function=reproject_exact, hdu_in=1)

# save the mosaic to a new FITS file
hdu = fits.PrimaryHDU(coadd, header=wcs.to_header())
hdu.writeto(os.path.join(output_path, 'f200_mosaic.fits'), overwrite=True)

print("Mosaic created and saved to", os.path.join(output_path, 'f200_mosaic.fits'))