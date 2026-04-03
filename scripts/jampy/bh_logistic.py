"""
Nested-sampling JAM fit for the Sombrero galaxy (M104).

This version follows the constant-anisotropy JAMcyl prescription:
- cylindrical alignment: align='cyl'
- constant free anisotropy parameter
- fixed stellar M/L
- free black hole mass
- fit to Vrms = sqrt(V^2 + sigma^2)

Notes:
- The free anisotropy parameter is sampled as qz = sigma_z / sigma_R
  and converted to beta_z = 1 - qz^2 for JAM.
- The fixed stellar M/L is absorbed directly into surf_pot, and ml=1.0
  is passed to JAM so the sampled mbh remains the physical BH mass.
- If you do not provide a non-zero PSF sigma and pixel size, JAM will
  not perform PSF convolution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.interpolate import griddata

import jampy as jam
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot


# ============================================================
# User settings
# ============================================================

output_dir = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/jam_models/constant_anisotropy"

kinematics_file = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/antoine/M104_stellar_Kin.csv"
mge_solution_file = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_solution.csv"
mge_luminosity_file = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_luminosity_table.csv"

# geometric / astrophysical inputs
rotation_angle_deg = -18.0
distance_mpc = 9.55
inclination_deg = 87.0
redshift = 0.003633
ml_fixed = 0.86

# central masking radius if no PSF convolution is used
rmin_arcsec = 0.10

# PSF / pixel inputs
# Replace these with the correct values for your data if available.
# If sigmapsf or pixsize are zero, JAM will not convolve with the PSF.
sigmapsf = 0.#np.array([0.0], dtype=float)   # arcsec
normpsf = 0. #np.array([1.0], dtype=float)    # sums to 1
pixsize = 0.0                             # arcsec

# optional error inflation outside a chosen sphere of influence radius
# set to None to disable
r_soi_arcsec = None

# dynesty settings
nlive = 100
dlogz_init = 1.0
checkpoint_every = 100

# prior bounds
log10_mbh_min = 7.0
log10_mbh_max = 10.0

# sample qz = sigma_z / sigma_R directly
qz_min = 0.2
qz_max = 2.0

# plotting
dpi = 300


# ============================================================
# Helper functions
# ============================================================

ARCSEC2_PER_SR = (180.0 / np.pi * 3600.0) ** 2
MAG_ARCSEC2_TO_LSUN_PC2 = 21.572
M_SUN_AB_F200W = 4.93


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def rotate(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot


def plot_map(x, y, values, cmap="RdBu_r", cbar_label="Value", ax=None, show=True):
    """
    Plot a smooth interpolated map from scattered x, y, value points.
    """
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), values, (xi, yi), method="cubic")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        zi,
        extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")

    if show:
        plt.show()

    return ax


def save_map(x, y, values, filename, cmap="RdBu_r", cbar_label="Value"):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_map(x, y, values, cmap=cmap, cbar_label=cbar_label, ax=ax, show=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)


def gaussian_peak_from_total_counts(total_counts, sigma_pix, q_obs):
    """
    Convert total_counts from mgefit into the Gaussian peak intensity.

    For a 2D elliptical Gaussian:
        total_counts = 2*pi*peak*sigma_pix^2*q_obs
    """
    return total_counts / (2.0 * np.pi * sigma_pix**2 * q_obs)


def mjysr_to_lsun_pc2(mu_mjysr, m_sun_ab=M_SUN_AB_F200W):
    """
    Convert surface brightness in MJy/sr into luminosity surface density Lsun/pc^2.
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
    m_sun_ab=M_SUN_AB_F200W,
):
    """
    Build the JAM luminosity arrays from an MGE solution table.
    """
    sigma_pix = np.asarray(mge_tab[sigma_pix_col], dtype=float)
    q_obs = np.asarray(mge_tab[q_col], dtype=float)
    total_counts = np.asarray(mge_tab[total_col], dtype=float)

    peak_mjysr = gaussian_peak_from_total_counts(total_counts, sigma_pix, q_obs)
    surf_lum = mjysr_to_lsun_pc2(peak_mjysr, m_sun_ab=m_sun_ab)
    sigma_arcsec = sigma_pix * pixel_scale_arcsec

    return surf_lum, sigma_arcsec, q_obs


def compute_vlos_compensated(kin_table, z):
    c_kms = 299792.458
    return np.asarray(kin_table["LOSV"], dtype=float) - z * c_kms


def compute_vrms_and_error(vlos, vlos_err, sigma, sigma_err):
    vrms = np.sqrt(vlos**2 + sigma**2)

    vrms_err = np.sqrt((vlos * vlos_err) ** 2 + (sigma * sigma_err) ** 2) / vrms

    # guard against division by zero or invalid values
    bad = ~np.isfinite(vrms_err)
    if np.any(bad):
        vrms_err[bad] = np.nanmedian(vrms_err[np.isfinite(vrms_err)])

    return vrms, vrms_err


def inflate_errors_outside_soi(errors, x, y, r_soi=None):
    """
    Inflate errors outside a chosen sphere-of-influence radius.
    If r_soi is None, return the input errors unchanged.
    """
    errors = np.asarray(errors, dtype=float).copy()

    if r_soi is None:
        return errors

    r = np.hypot(x, y)
    outside = r > r_soi
    n_out = np.count_nonzero(outside)

    if n_out > 0:
        errors[outside] *= (2.0 * n_out) ** 0.25

    return errors


def weighted_quantile(x, q, weights):
    """
    Weighted quantiles for posterior summaries.
    """
    x = np.asarray(x)
    q = np.asarray(q)
    weights = np.asarray(weights)

    sorter = np.argsort(x)
    x_sorted = x[sorter]
    w_sorted = weights[sorter]

    cdf = np.cumsum(w_sorted)
    cdf /= cdf[-1]

    return np.interp(q, cdf, x_sorted)


# ============================================================
# Load data
# ============================================================

ensure_dir(output_dir)

kin_table = Table.read(kinematics_file, format="csv", delimiter=";")

# rotate kinematics into photometric frame
x_rot, y_rot = rotate(
    np.asarray(kin_table["X"], dtype=float),
    np.asarray(kin_table["Y"], dtype=float),
    rotation_angle_deg,
)
kin_table["X_rot"] = x_rot
kin_table["Y_rot"] = y_rot

# quick diagnostic maps
save_map(
    kin_table["X_rot"],
    kin_table["Y_rot"],
    kin_table["LOSV"],
    f"{output_dir}/kinematic_maps_LOSV.png",
    cbar_label="LOSV (km/s)",
)

save_map(
    kin_table["X_rot"],
    kin_table["Y_rot"],
    kin_table["sigma"],
    f"{output_dir}/kinematic_maps_sigma.png",
    cbar_label="sigma (km/s)",
)

# line-of-sight velocity in the galaxy rest frame
vlos_rf = compute_vlos_compensated(kin_table, z=redshift)

save_map(
    kin_table["X_rot"],
    kin_table["Y_rot"],
    vlos_rf,
    f"{output_dir}/kinematic_maps_vlos_compensated.png",
    cbar_label="Vlos compensated (km/s)",
)

# observed kinematics
xbin_all = np.asarray(kin_table["X_rot"], dtype=float)
ybin_all = np.asarray(kin_table["Y_rot"], dtype=float)
sigma_obs_all = np.asarray(kin_table["sigma"], dtype=float)
sigma_err_all = np.asarray(kin_table["sigma_err"], dtype=float)
vlos_err_all = np.asarray(kin_table["LOSV_err"], dtype=float)

vrms_all, vrms_err_all = compute_vrms_and_error(
    vlos=vlos_rf,
    vlos_err=vlos_err_all,
    sigma=sigma_obs_all,
    sigma_err=sigma_err_all,
)

save_map(
    xbin_all,
    ybin_all,
    vrms_all,
    f"{output_dir}/kinematic_maps_vrms.png",
    cbar_label="Vrms (km/s)",
)

# ============================================================
# Load MGE
# ============================================================

# Option 1: build from the original MGE solution table
mge_solution = Table.read(mge_solution_file)
surf_lum_from_solution, sigma_lum_from_solution, q_obs_from_solution = make_jam_mge_from_table(
    mge_solution
)

# Option 2: use your already prepared JAM luminosity table
# This is what we adopt below because it directly provides luminosity in Lsun,
# sigma in arcsec, and observed axial ratio q_obs.
mge_table = Table.read(mge_luminosity_file)

surf_lum = np.asarray(mge_table["luminosity_Lsun"], dtype=float)
sigma_lum = np.asarray(mge_table["sigma_arcsec"], dtype=float)
q_obs_lum = np.asarray(mge_table["q_obs"], dtype=float)

# fixed stellar mass model
surf_pot = surf_lum * ml_fixed
sigma_pot = sigma_lum.copy()
q_obs_pot = q_obs_lum.copy()

# ============================================================
# Select bins to fit
# ============================================================

rbin_all = np.hypot(xbin_all, ybin_all)

fit_mask = np.isfinite(vrms_all) & np.isfinite(vrms_err_all)
fit_mask &= np.isfinite(xbin_all) & np.isfinite(ybin_all)
fit_mask &= vrms_err_all > 0.0

# if no PSF/aperture convolution, avoid the central singularity
if (np.all(sigmapsf == 0.0)) or (pixsize == 0.0):
    fit_mask &= rbin_all >= rmin_arcsec

xfit = xbin_all[fit_mask]
yfit = ybin_all[fit_mask]
vrms_fit = vrms_all[fit_mask]
vrms_err_fit = vrms_err_all[fit_mask]

vrms_err_fit = inflate_errors_outside_soi(
    vrms_err_fit,
    xfit,
    yfit,
    r_soi=r_soi_arcsec,
)

# goodbins is defined for the arrays actually passed to JAM
goodbins = np.ones_like(vrms_fit, dtype=bool)

# ============================================================
# Likelihood and priors
# ============================================================

def log_likelihood(theta):
    """
    theta = [log10(MBH/Msun), qz]
    where qz = sigma_z / sigma_R
    and beta_z = 1 - qz^2
    """
    log10_mbh, qz = theta

    mbh = 10.0 ** log10_mbh
    beta_z = 1.0 - qz**2

    # constant anisotropy across all MGE Gaussians
    beta = np.full_like(sigma_lum, beta_z, dtype=float)

    out = jam.axi.proj(
        surf_lum=surf_lum,
        sigma_lum=sigma_lum,
        qobs_lum=q_obs_lum,
        surf_pot=surf_pot,
        sigma_pot=sigma_pot,
        qobs_pot=q_obs_pot,
        inc=inclination_deg,
        mbh=mbh,
        distance=distance_mpc,
        xbin=xfit,
        ybin=yfit,
        align="cyl",
        analytic_los=True,
        beta=beta,
        gamma=None,
        logistic=False,
        data=vrms_fit,
        errors=vrms_err_fit,
        flux_obs=None,
        goodbins=goodbins,
        interp=False,
        kappa=None,
        ml=1.0,
        moment="zz",
        quiet=True,
        epsrel=1e-2
    )

    chi2 = out.chi2
    return -0.5 * chi2


def prior_transform(utheta):
    """
    Transform unit-cube parameters to physical parameters.
    """
    log10_mbh = log10_mbh_min + utheta[0] * (log10_mbh_max - log10_mbh_min)
    qz = qz_min + utheta[1] * (qz_max - qz_min)

    return np.array([log10_mbh, qz], dtype=float)


# ============================================================
# Run nested sampling
# ============================================================

sampler = DynamicNestedSampler(
    log_likelihood,
    prior_transform,
    ndim=2,
    nlive=nlive,
)

sampler.run_nested(
    checkpoint_every=checkpoint_every,
    dlogz_init=dlogz_init,
    checkpoint_file=f"{output_dir}/checkpoint_constant_anisotropy.save",
)

results = sampler.results

# ============================================================
# Save results
# ============================================================

weights = np.exp(results.logwt - results.logz[-1])

np.savez(
    f"{output_dir}/nested_constant_anisotropy_results.npz",
    samples=results.samples,
    logl=results.logl,
    logwt=results.logwt,
    logz=results.logz,
    logzerr=results.logzerr,
    weights=weights,
)

# posterior summaries
log10_mbh_samples = results.samples[:, 0]
qz_samples = results.samples[:, 1]
beta_z_samples = 1.0 - qz_samples**2
mbh_samples = 10.0 ** log10_mbh_samples

q16, q50, q84 = 0.16, 0.50, 0.84

log10_mbh_q = weighted_quantile(log10_mbh_samples, [q16, q50, q84], weights)
mbh_q = weighted_quantile(mbh_samples, [q16, q50, q84], weights)
qz_q = weighted_quantile(qz_samples, [q16, q50, q84], weights)
beta_z_q = weighted_quantile(beta_z_samples, [q16, q50, q84], weights)

summary_file = f"{output_dir}/posterior_summary.txt"
with open(summary_file, "w") as f:
    f.write("Posterior summaries (16th, 50th, 84th percentiles)\n")
    f.write("\n")
    f.write(
        f"log10(MBH/Msun): {log10_mbh_q[1]:.5f} "
        f"-{log10_mbh_q[1] - log10_mbh_q[0]:.5f} "
        f"+{log10_mbh_q[2] - log10_mbh_q[1]:.5f}\n"
    )
    f.write(
        f"MBH [Msun]: {mbh_q[1]:.5e} "
        f"-{mbh_q[1] - mbh_q[0]:.5e} "
        f"+{mbh_q[2] - mbh_q[1]:.5e}\n"
    )
    f.write(
        f"qz = sigma_z/sigma_R: {qz_q[1]:.5f} "
        f"-{qz_q[1] - qz_q[0]:.5f} "
        f"+{qz_q[2] - qz_q[1]:.5f}\n"
    )
    f.write(
        f"beta_z = 1 - qz^2: {beta_z_q[1]:.5f} "
        f"-{beta_z_q[1] - beta_z_q[0]:.5f} "
        f"+{beta_z_q[2] - beta_z_q[1]:.5f}\n"
    )

print(open(summary_file).read())

# ============================================================
# Corner plot
# ============================================================

fig, axes = dyplot.cornerplot(
    results,
    labels=[r"$\log_{10}(M_{\rm BH}/M_\odot)$", r"$q_z=\sigma_z/\sigma_R$"],
    show_titles=True,
    title_kwargs={"x": 0.65},
)
plt.savefig(f"{output_dir}/corner_plot_constant_anisotropy.png", dpi=dpi)
plt.close(fig)

# ============================================================
# Best-fit model evaluation
# ============================================================

imax = np.argmax(results.logl)
best_log10_mbh, best_qz = results.samples[imax]
best_mbh = 10.0 ** best_log10_mbh
best_beta_z = 1.0 - best_qz**2
best_beta = np.full_like(sigma_lum, best_beta_z, dtype=float)

best_out = jam.axi.proj(
    surf_lum=surf_lum,
    sigma_lum=sigma_lum,
    qobs_lum=q_obs_lum,
    surf_pot=surf_pot,
    sigma_pot=sigma_pot,
    qobs_pot=q_obs_pot,
    inc=inclination_deg,
    mbh=best_mbh,
    distance=distance_mpc,
    xbin=xfit,
    ybin=yfit,
    align="cyl",
    analytic_los=True,
    beta=best_beta,
    gamma=None,
    logistic=False,
    data=vrms_fit,
    errors=vrms_err_fit,
    flux_obs=None,
    goodbins=goodbins,
    interp=True,
    kappa=None,
    ml=1.0,
    moment="zz",
    quiet=True,
    epsrel=1e-2,
    sigmapsf=sigmapsf,
    normpsf=normpsf,
    pixsize=pixsize,
)

vrms_model_best = best_out.model

# scatter plot: data vs model
fig, ax = plt.subplots(figsize=(6, 6))
ax.errorbar(vrms_fit, vrms_model_best, xerr=vrms_err_fit, fmt="o", ms=4, alpha=0.7)
vmin = min(np.min(vrms_fit), np.min(vrms_model_best))
vmax = max(np.max(vrms_fit), np.max(vrms_model_best))
ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
ax.set_xlabel("Observed Vrms (km/s)")
ax.set_ylabel("Model Vrms (km/s)")
ax.set_title("Best-fit JAMcyl constant-anisotropy model")
fig.tight_layout()
fig.savefig(f"{output_dir}/vrms_data_vs_model.png", dpi=dpi)
plt.close(fig)

# model map
save_map(
    xfit,
    yfit,
    vrms_model_best,
    f"{output_dir}/vrms_model_bestfit.png",
    cbar_label="Best-fit model Vrms (km/s)",
)

# residual map
save_map(
    xfit,
    yfit,
    vrms_fit - vrms_model_best,
    f"{output_dir}/vrms_residual_bestfit.png",
    cbar_label="Vrms residual (data - model) [km/s]",
)

print(f"Best-fit log10(MBH/Msun) = {best_log10_mbh:.5f}")
print(f"Best-fit MBH [Msun]      = {best_mbh:.5e}")
print(f"Best-fit qz             = {best_qz:.5f}")
print(f"Best-fit beta_z         = {best_beta_z:.5f}")
print(f"Reduced chi2            = {best_out.chi2:.5f}")