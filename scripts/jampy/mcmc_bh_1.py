import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table

# External packages from Michele Cappellari
# pip install jampy adamet plotbin capfit
import jampy as jam
import capfit
from adamet.adamet import adamet
from adamet.corner_plot import corner_plot
from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield
import os


# -----------------------------------------------------------------------------
# USER INPUTS
# -----------------------------------------------------------------------------
MGE_PATH = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_solution.csv"
KIN_PATH = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/antoine/M104_stellar_Kin_rotated.csv"
KIN_SEP = ";"
OUTPUT_PATH = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Plots/mcmc_bh_2"

DISTANCE_MPC = 9.55
PIXEL_SCALE_ARCSEC = 0.031          # JWST image scale used for the MGE fit
PIXEL_SIZE_KIN_ARCSEC = 0.103       # NIRSpec spaxel/bin size; replace if different
M_SUN_AB_F200W = 4.93               # your adopted solar absolute AB magnitude

# If your kinematics are not already aligned with the galaxy major axis,
# rotate them by this angle (degrees) BEFORE fitting.
# Positive angles are counter-clockwise.
PA_OFFSET_DEG = 0.0

# PSF model for the kinematics: replace with your actual PSF model.
SIGMAPSF = np.array([0.05, 0.15])   # Gaussian sigmas in arcsec
NORMPSF = np.array([0.7, 0.3])

# Central stellar-population prior on M/L in F200W units.
ML_PRIOR = 0.86
ML_PRIOR_ERR = 0.10                 # replace by your measured 1-sigma uncertainty

# I recommend fixing inclination for Sombrero unless you explicitly want to fit it.
FIT_INCLINATION = False
INC_FIXED_DEG = 87.0                # replace by your preferred photometric value

# MCMC controls
NSTEP = 2000
SEED = 7

# Optional: exclude innermost bins if AGN or template mismatch contaminates the stellar kinematics.
EXCLUDE_R_ARCSEC = None             # e.g. 0.05, or None to keep all bins


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
ARCSEC2_PER_SR = (180.0 / np.pi * 3600.0) ** 2
MAG_ARCSEC2_TO_LSUN_PC2 = 21.572

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def rotate_points(x, y, theta_deg):
    """Rotate sky coordinates counter-clockwise by theta_deg."""
    theta = np.deg2rad(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * x - st * y
    yr = st * x + ct * y
    return xr, yr



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
        mu_AB[mag/arcsec^2] = -2.5*log10( I_nu[Jy/arcsec^2] / 3631 )
        Sigma[Lsun/pc^2] = 10^[-0.4 * (mu_AB - M_sun_AB - 21.572)]
    """
    jy_arcsec2 = mu_mjysr * 1e6 / ARCSEC2_PER_SR
    mu_ab = -2.5 * np.log10(jy_arcsec2 / 3631.0)
    return 10.0 ** (-0.4 * (mu_ab - m_sun_ab - MAG_ARCSEC2_TO_LSUN_PC2))



def make_jam_mge_from_table(mge_tab,
                            total_col="total_counts",
                            sigma_pix_col="sigma_pix",
                            q_col="q_obs",
                            pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
                            m_sun_ab=M_SUN_AB_F200W):
    """
    Build the three JAM luminosity arrays from your MGE table.

    Returns
    -------
    surf_lum : peak surface brightness of each Gaussian in Lsun/pc^2
    sigma    : Gaussian sigma in arcsec
    q_obs    : observed axial ratio
    """
    sigma_pix = np.asarray(mge_tab[sigma_pix_col], dtype=float)
    q_obs = np.asarray(mge_tab[q_col], dtype=float)
    total_counts = np.asarray(mge_tab[total_col], dtype=float)

    peak_mjysr = gaussian_peak_from_total_counts(total_counts, sigma_pix, q_obs)
    surf_lum = mjysr_to_lsun_pc2(peak_mjysr, m_sun_ab=m_sun_ab)
    sigma_arcsec = sigma_pix * pixel_scale_arcsec

    return surf_lum, sigma_arcsec, q_obs



def eval_mge_surface_brightness(x, y, surf, sigma, q_obs):
    """Evaluate the observed 2D MGE surface brightness at sky positions (x, y)."""
    x = np.asarray(x)
    y = np.asarray(y)
    surf_bin = np.zeros_like(x, dtype=float)
    for s0, sig, q in zip(surf, sigma, q_obs):
        arg = -0.5 * (x**2 + (y / q)**2) / sig**2
        surf_bin += s0 * np.exp(arg)
    return surf_bin



def load_kinematics_table(path, sep=KIN_SEP):
    """Read the Antoine-format CSV and compute Vrms and its uncertainty."""
    kin = Table.read(path, format="ascii.csv", delimiter=sep)

    x = np.asarray(kin["X"], dtype=float)
    y = np.asarray(kin["Y"], dtype=float)
    vel = np.asarray(kin["LOSV"], dtype=float)
    evel = np.asarray(kin["LOSV_err"], dtype=float)
    sig = np.asarray(kin["sigma"], dtype=float)
    esig = np.asarray(kin["sigma_err"], dtype=float)

    x, y = rotate_points(x, y, PA_OFFSET_DEG)

    vrms = np.sqrt(vel**2 + sig**2)
    evrms = np.sqrt((vel * evel)**2 + (sig * esig)**2) / vrms

    # Protect against pathological bins with vrms ~ 0 or unrealistically tiny errors.
    med_evrms = np.nanmedian(evrms[np.isfinite(evrms)])
    floor = 0.02 * np.nanmedian(vrms[np.isfinite(vrms)])
    evrms = np.where(np.isfinite(evrms) & (evrms > 0), evrms, med_evrms)
    evrms = np.maximum(evrms, floor)

    return x, y, vrms, evrms, vel, sig


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
mge = Table.read(MGE_PATH)
if "total_counts" not in mge.colnames and "total_flux" in mge.colnames:
    mge["total_counts"] = mge["total_flux"]

surf_lum, sigma_lum, q_obs_lum = make_jam_mge_from_table(
    mge,
    total_col="total_counts",
    sigma_pix_col="sigma_pix",
    q_col="q_obs",
    pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
    m_sun_ab=M_SUN_AB_F200W,
)

# Self-consistent light-traces-mass parameterization.
# The free parameter lg_ml scales this luminosity MGE into the stellar mass MGE.
surf_pot = surf_lum.copy()
sigma_pot = sigma_lum.copy()
q_obs_pot = q_obs_lum.copy()

xbin, ybin, rms, erms, vel, sig = load_kinematics_table(KIN_PATH, sep=KIN_SEP)
flux_obs = eval_mge_surface_brightness(xbin, ybin, surf_lum, sigma_lum, q_obs_lum)

radius = np.hypot(xbin, ybin)
goodbins = np.isfinite(xbin) & np.isfinite(ybin) & np.isfinite(rms) & np.isfinite(erms) & (erms > 0)
if EXCLUDE_R_ARCSEC is not None:
    goodbins &= radius >= EXCLUDE_R_ARCSEC

if not np.any(goodbins):
    raise RuntimeError("No valid kinematic bins survive the quality cuts.")


# -----------------------------------------------------------------------------
# JAM + MCMC SETUP
# -----------------------------------------------------------------------------
lg_ml_prior = np.log10(ML_PRIOR)
lg_ml_prior_err = ML_PRIOR_ERR / (ML_PRIOR * np.log(10.0))

# Sombrero is not an M32-scale BH problem; use a wide prior centered near 1e9 Msun.
lg_mbh0 = 9.0
ratio0 = 0.90  # sigma_z/sigma_R
lg_ml0 = lg_ml_prior

if FIT_INCLINATION:
    qmin_obs = np.min(q_obs_lum)
    q0 = min(0.3, 0.95 * qmin_obs)
    p0 = np.array([q0, ratio0, lg_mbh0, lg_ml0])
    bounds = np.array([
        [0.05, 0.50, 7.0, lg_ml_prior - 0.5],
        [qmin_obs - 1e-4, 1.20, 10.5, lg_ml_prior + 0.5],
    ])
    labels = [r"$q_{\rm intr,min}$", r"$\sigma_z/\sigma_R$", r"$\log_{10} M_{\rm BH}$", r"$\log_{10}(M/L)$"]
    sigpar = np.array([0.03, 0.05, 0.15, 0.05])
else:
    p0 = np.array([ratio0, lg_mbh0, lg_ml0])
    bounds = np.array([
        [0.50, 7.0, lg_ml_prior - 0.5],
        [1.20, 10.5, lg_ml_prior + 0.5],
    ])
    labels = [r"$\sigma_z/\sigma_R$", r"$\log_{10} M_{\rm BH}$", r"$\log_{10}(M/L)$"]
    sigpar = np.array([0.05, 0.15, 0.05])



def unpack_parameters(pars):
    if FIT_INCLINATION:
        q_intr_min, ratio, lg_mbh, lg_ml = pars
        qmin_obs = np.min(q_obs_lum)
        inc = np.degrees(
            np.arctan2(np.sqrt(1.0 - qmin_obs**2), np.sqrt(qmin_obs**2 - q_intr_min**2))
        )
    else:
        ratio, lg_mbh, lg_ml = pars
        inc = INC_FIXED_DEG
        q_intr_min = None

    return q_intr_min, ratio, lg_mbh, lg_ml, inc



def jam_lnprob(pars):
    # Hard bounds for safety; AdaMet also enforces them.
    if np.any(pars < bounds[0]) or np.any(pars > bounds[1]):
        return -np.inf

    q_intr_min, ratio, lg_mbh, lg_ml, inc = unpack_parameters(pars)

    # Constant anisotropy across MGE components.
    beta = np.full_like(q_obs_lum, 1.0 - ratio**2)
    mbh = 10.0**lg_mbh
    ml = 10.0**lg_ml

    try:
        out = jam.axi.proj(
            surf_lum,
            sigma_lum,
            q_obs_lum,
            surf_pot * ml,
            sigma_pot,
            q_obs_pot,
            inc,
            mbh,
            DISTANCE_MPC,
            xbin,
            ybin,
            plot=False,
            #pixsize=PIXEL_SIZE_KIN_ARCSEC,
            quiet=1,
            #sigmapsf=SIGMAPSF,
            #normpsf=NORMPSF,
            goodbins=goodbins,
            align="cyl",
            beta=beta,
            data=rms,
            errors=erms,
            flux_obs=flux_obs,
            ml=1,
            moment="zz",
        )
    except Exception:
        return -np.inf

    jam_lnprob.out = out

    resid = (rms[goodbins] - out.model[goodbins]) / erms[goodbins]
    chi2 = resid @ resid

    # Gaussian prior on stellar-population M/L.
    lnprior_ml = -0.5 * ((lg_ml - lg_ml_prior) / lg_ml_prior_err) ** 2

    return -0.5 * chi2 + lnprior_ml


# -----------------------------------------------------------------------------
# RUN ADAMET
# -----------------------------------------------------------------------------
print("Fitting Sombrero JAM model...")
print(f"Number of fitted bins: {goodbins.sum()} / {goodbins.size}")
print(f"M/L prior: {ML_PRIOR:.3f} +/- {ML_PRIOR_ERR:.3f}")
print(f"Inclination mode: {'fit' if FIT_INCLINATION else f'fixed at {INC_FIXED_DEG:.1f} deg'}")
print(f"Saving plots to: {ensure_dir(OUTPUT_PATH)}")
pars, lnprob = adamet(
    jam_lnprob,
    p0,
    sigpar,
    bounds,
    NSTEP,
    plot=False,
    nprint=max(1, NSTEP // 10),
    labels=labels,
    seed=SEED,
)

bestfit = pars[np.argmax(lnprob)]
perc = np.percentile(pars, [15.86, 84.14], axis=0)
sig_bestfit = np.squeeze(np.diff(perc, axis=0) / 2.0)
txt = capfit.format_values_with_errors(bestfit, sig_bestfit, labels)
print("\nBest-fit parameters:\n")
print(txt.plain)

# Recover the best-fit JAM model and derived physical quantities.
_ = jam_lnprob(bestfit)
out = jam_lnprob.out

if FIT_INCLINATION:
    _, ratio_best, lg_mbh_best, lg_ml_best, inc_best = unpack_parameters(bestfit)
else:
    _, ratio_best, lg_mbh_best, lg_ml_best, inc_best = unpack_parameters(bestfit)

mbh_best = 10.0**lg_mbh_best
ml_best = 10.0**lg_ml_best

print(f"\nDerived quantities:\n")
print(f"inclination = {inc_best:.2f} deg")
print(f"sigma_z/sigma_R = {ratio_best:.3f}")
print(f"M_BH = {mbh_best:.3e} Msun")
print(f"M/L = {ml_best:.3f}")
print(f"reduced chi2 reported by JAM = {out.chi2:.3f}")


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
fig = corner_plot(pars, lnprob, labels=labels, extents=bounds)
fig.text(0.34, 0.99, txt.latex, ha="left", va="top")

# Symmetrized data for prettier display only.
rms_sym = rms.copy()
rms_sym[goodbins] = symmetrize_velfield(xbin[goodbins], ybin[goodbins], rms[goodbins])
vmin, vmax = np.percentile(rms_sym[goodbins], [0.5, 99.5])

dx = 0.24
yfac = 0.87

fig.add_axes([0.69, 0.99 - dx * yfac, dx, dx * yfac])
plot_velfield(
    xbin,
    ybin,
    rms_sym,
    vmin=vmin,
    vmax=vmax,
    linescolor="w",
    colorbar=1,
    label=r"Data $V_{\rm rms}$ (km/s)",
    flux=flux_obs,
)
plt.tick_params(labelbottom=False)
plt.ylabel("arcsec")

fig.add_axes([0.69, 0.98 - 2 * dx * yfac, dx, dx * yfac])
plot_velfield(
    xbin,
    ybin,
    out.model,
    vmin=vmin,
    vmax=vmax,
    linescolor="w",
    colorbar=1,
    label=r"Model $V_{\rm rms}$ (km/s)",
    flux=flux_obs,
)
plt.tick_params(labelbottom=False)
plt.ylabel("arcsec")
plt.savefig(f"{OUTPUT_PATH}/sombrero_jam_mcmc_corner.png", dpi=600, bbox_inches="tight")
plt.show()
