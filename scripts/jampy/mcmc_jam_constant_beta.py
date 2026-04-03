"""
Use MCMC to fit JAM models to the kinematic data of the Sombrero galaxy.
This script will fit for the black hole mass, anisotropy, using a fixed M/L from stellar population
modelling. 

It uses as input the MGE and kinematic data by Antoine.
"""
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import os

# settings
output_dir = '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/constant_beta_bh'

# make sure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load the kin_table
# the file is a csv with the format:
# X;Y;LOSV;LOSV_err;sigma;sigma_err;h3;h4
kin_table = Table.read("/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/antoine/M104_stellar_Kin.csv", format="csv", delimiter=";")

# helper functions
def plot_map(x, y, values, cmap='RdBu_r', cbar_label='Value', ax=None, show=True):
    """
    Utility function to plot a 2D map of values at given x, y coordinates.
    Uses interpolation to create a smooth image from scattered data points.
    """
    # takes the xy and values and plots as an image with a colorbar
    # create a grid of x and y values
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    # interpolate the values onto the grid
    from scipy.interpolate import griddata
    zi = griddata((x, y), values, (xi, yi), method='cubic')
    # plot the image
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(zi, extent=(np.min(x), np.max(x), np.min(y), np.max(y)), origin='lower', cmap=cmap)
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel('X (arcsec)')
    ax.set_ylabel('Y (arcsec)')
    #ax.set_title('Interpolated Map')
    if show:
        plt.show()

# load the MGE best fit
ARCSEC2_PER_SR = (180.0 / np.pi * 3600.0) ** 2
MAG_ARCSEC2_TO_LSUN_PC2 = 21.572
M_SUN_AB_F200W = 4.93             # adopted solar absolute AB magnitude
def gaussian_peak_from_total_counts(total_counts, sigma_pix, q_obs):
    """
    Convert the integrated MGE quantity produced by mgefit into the Gaussian peak.

    For a 2D elliptical Gaussian:
        total_counts = 2*pi*peak*sigma_pix^2*q_obs
    """
    return total_counts / (2.0 * np.pi * sigma_pix**2 * q_obs)



def mjysr_to_lsun_pc2(mu_mjysr, m_sun_ab=M_SUN_AB_F200W):
    """
    Convert surface brightness in MJy/sr into band luminosity surface density Lsun/pc^2.

    Uses:
        mu_AB[mag/arcsec^2] = -2.5*log10(I_nu[Jy/arcsec^2] / 3631)
        Sigma[Lsun/pc^2] = 10^[-0.4 * (mu_AB - M_sun_AB - 21.572)]
    """
    jy_arcsec2 = mu_mjysr * 1e6 / ARCSEC2_PER_SR
    mu_ab = -2.5 * np.log10(jy_arcsec2 / 3631.0)
    return 10.0 ** (-0.4 * (mu_ab - m_sun_ab - MAG_ARCSEC2_TO_LSUN_PC2))



def make_jam_mge_from_table(
    mge_tab,
    total_col="total_counts",
    sigma_pix_col="sigma_pix",
    q_col="q_obs",
    pixel_scale_arcsec=0.031,
    m_sun_ab= 4.93
):
    """Build the three JAM luminosity arrays from your MGE table."""
    sigma_pix = np.asarray(mge_tab[sigma_pix_col], dtype=float)
    q_obs = np.asarray(mge_tab[q_col], dtype=float)
    total_counts = np.asarray(mge_tab[total_col], dtype=float)

    peak_mjysr = gaussian_peak_from_total_counts(total_counts, sigma_pix, q_obs)
    surf_lum = mjysr_to_lsun_pc2(peak_mjysr, m_sun_ab=m_sun_ab)
    sigma_arcsec = sigma_pix * pixel_scale_arcsec

    return surf_lum, sigma_arcsec, q_obs

# this is not aligned with the photometry, it is in IFU alignment. We will need to rotate and shift it to match the photometry. The center is at (0, 0) in this table, which seems to match the center  of the galaxy.
# rotation to go from IFU alignment to photometry alignment is about 20 degrees.
def rotate(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot

# rotate the x and y coordinates in the kinematics table to align the kinematics
kin_table["X_rot"], kin_table["Y_rot"] = rotate(kin_table["X"],
                                                 kin_table["Y"], -18)

# plot the rotated LOSV map, we need this rotation for JAM to work
plot_map(kin_table["X_rot"], kin_table["Y_rot"], kin_table["LOSV"], cbar_label='LOSV (km/s)', show=False)
plt.savefig(f"{output_dir}/kinematic_maps_LOSV.png", dpi=600) 
plt.clf() # clear the figure to save memory

# plot the sigma
plot_map(kin_table["X_rot"], kin_table["Y_rot"], kin_table["sigma"], cbar_label='sigma (km/s)', show=False)
plt.savefig(f"{output_dir}/kinematic_maps_sigma.png", dpi=600) 
plt.clf() # clear the figure to save memory

# get the xbin and ybin from the kinematics table
xbin = kin_table["X_rot"].data
ybin = kin_table["Y_rot"].data
# get the observed LOSV and sigma from the kinematics table
v_obs_lum = kin_table["LOSV"].data
sigma_obs_lum = kin_table["sigma"].data


# compute line of sight velocity compensated by redshift
def compute_vlos_compensated(kin_table, z):
    c = 299792.458  # speed of light in km/s
    v_los_compensated = kin_table["LOSV"] - z * c
    return v_los_compensated

vlos_rf = compute_vlos_compensated(kin_table, z=0.003633) # redshift of the Sombrero galaxy from NED
plot_map(xbin, ybin, vlos_rf, cbar_label='Vlos Compensated (km/s)', show=False)
plt.savefig(f"{output_dir}/kinematic_maps_vlos_compensated.png", dpi=600) 
plt.clf() # clear the figure to save memory
# make the Vrms from the observed LOSV and sigma
vrms= np.sqrt(vlos_rf**2 + sigma_obs_lum**2)
vrms_err = np.sqrt((vlos_rf * kin_table["LOSV_err"].data)**2 + (sigma_obs_lum * kin_table["sigma_err"].data)**2) / vrms
# plot the vrms map
plot_map(xbin, ybin, vrms, cbar_label='Vrms (km/s)', show=False)
plt.savefig(f"{output_dir}/kinematic_maps_vrms.png", dpi=600) 
plt.clf() # clear the figure to save memory


# load the MGE table

mge_tab = Table.read('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_solution.csv')
surf_lum, sigma_lum, q_obs_lum = make_jam_mge_from_table(mge_tab) 

# load the MGE best fit
mge_table = Table.read('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_luminosity_table.csv')
total_lum= mge_table['luminosity_Lsun']
# convert 
sigma_lum = mge_table['sigma_arcsec'] # for the JAM input we need the sigma in arcsec, not in pixels we already computed it before
q_obs_lum = mge_table['q_obs']

# now get the JAM
# import jampy and just evaluate the likelihood at some test parameters to make sure it works with the new kinematics file
import jampy as jam

# # test parameters
# bh_mass = 2.5e8 # from Jardel+2011
ml =  0.86# from stellar populations with MUSE

# 
def log_likelihood(params):
    bh_mass, beta = params
    # bh mass in Msun
    # beta is the radial anisotropy of the individual kinematic-tracer MGE Gaussians (Default: beta=np.zeros(n)):
    beta = np.full_like(surf_lum, beta)
    # gamma is the tangential anisotropy of the individual kinematic-tracer MGE Gaussians
    # gamma is a list of length equal to the number of Gaussians in the MGE,
    #  with one gamma value per Gaussian. This allows for more flexibility in the anisotropy profile,
    #  as each Gaussian can have its own tangential anisotropy.
    out = jam.axi.proj(
            surf_lum=surf_lum, sigma_lum=sigma_lum, qobs_lum=q_obs_lum, surf_pot=surf_lum*ml, sigma_pot=sigma_lum, qobs_pot=q_obs_lum,
            inc=87.0, mbh=bh_mass, distance=9.55, xbin=xbin, ybin=ybin, align='sph', analytic_los=False,
            beta=beta, data=vrms, epsrel=1e-2, errors=vrms_err, flux_obs=None,
            gamma=None, interp=True, kappa=None,
            sigmapsf=0.1,
            normpsf= np.array([1.0]),
            pixsize=0.1, # NIRSpec pixel size is 0.1 arcsec, so we use that as the pixsize for the PSF convolution
            pixang=-18, # Rotation angle to align with the photometry 
            logistic=False, ml=1, moment='zz',
            quiet=True,  # Suppress output during chi2 evaluation
             )
    chi2 = out.chi2
    loglike = -0.5 * chi2
    return loglike

# we will now use dynesty to sample the posterior distribution of the parameters
from dynesty import DynamicNestedSampler
# define the prior transform function for dynesty


def prior_transform(utheta):
    # utheta is a unit cube in [0, 1]^D where D is the number of parameters
    # we need to transform it to the actual parameter space
    # we will use uniform priors for both parameters in the following ranges:
    bh_mass_min = 1e7
    bh_mass_max = 1e10

    bh_mass = bh_mass_min + utheta[0] * (bh_mass_max - bh_mass_min)

    beta_min = 0.0
    beta_max = 0.9

    beta = beta_min + utheta[1] * (beta_max - beta_min)
    return np.array([bh_mass, beta])



sampler = DynamicNestedSampler(log_likelihood, prior_transform, ndim=2,
                               nlive=100,
                               #dlogz=0.5,  # stopping criterion for nested sampling
)
sampler.run_nested(checkpoint_every=100,
                   dlogz_init=1.0,
                    checkpoint_file=f'{output_dir}/checkpoint.save')
results = sampler.results
# save the results to a npz file
np.savez(f"{output_dir}/nested_beta_bh_results.npz", results=results)

# make a corner plot of the posterior distribution using dynesty's built in corner plot function
from dynesty import plotting as dyplot
fig, axes = dyplot.cornerplot(results, show_titles=True, title_kwargs={'x': 0.65})
plt.savefig(f"{output_dir}/corner_plot.png", dpi=600)
plt.show()
   # vrms_model = out.model  # with moment='zz' the output is the LOS Vrms
    #vlos_model = out.model  # with moment='z' the output is the LOS velocity

    #out.plot()   # Generate data/model comparison when data is given