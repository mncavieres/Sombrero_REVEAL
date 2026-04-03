import os
import numpy as np
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


# -----------------------------------------------------------------------------
# USER INPUTS
# -----------------------------------------------------------------------------
MGE_PATH = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_solution.csv"
KIN_PATH = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/antoine/M104_stellar_Kin.csv"
KIN_SEP = ";"
OUTPUT_PATH = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Plots/mcmc_bh"

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

# Fixed stellar-population M/L in F200W units.
FIXED_ML = 0.86

# I recommend fixing inclination for Sombrero unless you explicitly want to fit it.
FIT_INCLINATION = False
INC_FIXED_DEG = 87.0                # replace by your preferred photometric value

# AdaMet controls
# AdaMet is a single adaptive-Metropolis chain, so there is no 'nwalkers' keyword.
# To get more posterior samples, increase NSTEP and/or run multiple independent chains.
NSTEP = 2000
BURN_FRAC = 0.50
NCHAIN = 1
SEED = 7

# Initial proposal widths for AdaMet.
SIGPAR_RATIO = 0.04
SIGPAR_LGMBH = 0.10
SIGPAR_QINTR = 0.03

# Physically motivated BH search range for Sombrero.
LG_MBH_MIN = 7.0
LG_MBH_MAX = 10.2
LG_MBH0 = 9.0

# Optional: exclude bins if AGN/template mismatch contaminates the stellar kinematics.
EXCLUDE_R_ARCSEC = None             # e.g. 0.05, or None to keep all bins
EXCLUDE_HIGHSIGMA_KMS = None        # e.g. 420.0, or None to keep all bins

SAVE_FULL_CHAIN_TXT = True
SAVE_POST_BURN_TXT = True


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
ARCSEC2_PER_SR = (180.0 / np.pi * 3600.0) ** 2
MAG_ARCSEC2_TO_LSUN_PC2 = 21.572


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
    """Read the CSV and compute Vrms and its uncertainty."""
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

    med_evrms = np.nanmedian(evrms[np.isfinite(evrms)])
    floor = 0.02 * np.nanmedian(vrms[np.isfinite(vrms)])
    evrms = np.where(np.isfinite(evrms) & (evrms > 0), evrms, med_evrms)
    evrms = np.maximum(evrms, floor)

    return x, y, vrms, evrms, vel, sig



def summarize_chain(pars):
    bestfit = pars.mean(axis=0)
    med = np.percentile(pars, 50.0, axis=0)
    p16 = np.percentile(pars, 15.86, axis=0)
    p84 = np.percentile(pars, 84.14, axis=0)
    sig = 0.5 * (p84 - p16)
    return bestfit, med, p16, p84, sig



def save_chain_txt(path, pars, lnprob, labels):
    header = " ".join([lab.replace("$", "").replace("\\", "") for lab in labels]) + " lnprob"
    arr = np.column_stack([pars, lnprob])
    np.savetxt(path, arr, header=header)


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
os.makedirs(OUTPUT_PATH, exist_ok=True)

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

# Fixed stellar M/L: scale the potential MGE once and keep ml=1 inside JAM.
surf_pot = surf_lum * FIXED_ML
sigma_pot = sigma_lum.copy()
q_obs_pot = q_obs_lum.copy()

xbin, ybin, rms, erms, vel, sig = load_kinematics_table(KIN_PATH, sep=KIN_SEP)
flux_obs = eval_mge_surface_brightness(xbin, ybin, surf_lum, sigma_lum, q_obs_lum)

radius = np.hypot(xbin, ybin)
goodbins = np.isfinite(xbin) & np.isfinite(ybin) & np.isfinite(rms) & np.isfinite(erms) & (erms > 0)
if EXCLUDE_R_ARCSEC is not None:
    goodbins &= radius >= EXCLUDE_R_ARCSEC
if EXCLUDE_HIGHSIGMA_KMS is not None:
    goodbins &= sig <= EXCLUDE_HIGHSIGMA_KMS

if not np.any(goodbins):
    raise RuntimeError("No valid kinematic bins survive the quality cuts.")


# -----------------------------------------------------------------------------
# JAM + MCMC SETUP
# -----------------------------------------------------------------------------
ratio0 = 0.90  # sigma_z/sigma_R

if FIT_INCLINATION:
    qmin_obs = np.min(q_obs_lum)
    q0 = min(0.3, 0.95 * qmin_obs)
    p0 = np.array([q0, ratio0, LG_MBH0])
    bounds = np.array([
        [0.05, 0.50, LG_MBH_MIN],
        [qmin_obs - 1e-4, 1.20, LG_MBH_MAX],
    ])
    labels = [r"$q_{\rm intr,min}$", r"$\sigma_z/\sigma_R$", r"$\log_{10} M_{\rm BH}$"]
    sigpar = np.array([SIGPAR_QINTR, SIGPAR_RATIO, SIGPAR_LGMBH])
else:
    p0 = np.array([ratio0, LG_MBH0])
    bounds = np.array([
        [0.50, LG_MBH_MIN],
        [1.20, LG_MBH_MAX],
    ])
    labels = [r"$\sigma_z/\sigma_R$", r"$\log_{10} M_{\rm BH}$"]
    sigpar = np.array([SIGPAR_RATIO, SIGPAR_LGMBH])



def unpack_parameters(pars):
    if FIT_INCLINATION:
        q_intr_min, ratio, lg_mbh = pars
        qmin_obs = np.min(q_obs_lum)
        inc = np.degrees(
            np.arctan2(np.sqrt(1.0 - qmin_obs**2), np.sqrt(qmin_obs**2 - q_intr_min**2))
        )
    else:
        ratio, lg_mbh = pars
        inc = INC_FIXED_DEG
        q_intr_min = None

    return q_intr_min, ratio, lg_mbh, inc



def jam_lnprob(pars):
    if np.any(pars < bounds[0]) or np.any(pars > bounds[1]):
        return -np.inf

    q_intr_min, ratio, lg_mbh, inc = unpack_parameters(pars)

    beta = np.full_like(q_obs_lum, 1.0 - ratio**2)
    mbh = 10.0**lg_mbh

    try:
        out = jam.axi.proj(
            surf_lum,
            sigma_lum,
            q_obs_lum,
            surf_pot,
            sigma_pot,
            q_obs_pot,
            inc,
            mbh,
            DISTANCE_MPC,
            xbin,
            ybin,
            plot=False,
            pixsize=PIXEL_SIZE_KIN_ARCSEC,
            quiet=1,
            sigmapsf=SIGMAPSF,
            normpsf=NORMPSF,
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
    return -0.5 * chi2



def run_one_chain(seed):
    pars, lnprob = adamet(
        jam_lnprob,
        p0,
        sigpar,
        bounds,
        NSTEP,
        plot=False,
        nprint=max(1, NSTEP // 10),
        labels=labels,
        seed=seed,
    )
    return pars, lnprob


# -----------------------------------------------------------------------------
# RUN ADAMET
# -----------------------------------------------------------------------------
print("Fitting Sombrero JAM model with fixed stellar M/L...")
print(f"Number of fitted bins: {goodbins.sum()} / {goodbins.size}")
print(f"Fixed M/L = {FIXED_ML:.3f}")
print(f"Inclination mode: {'fit' if FIT_INCLINATION else f'fixed at {INC_FIXED_DEG:.1f} deg'}")
print(f"AdaMet steps per chain: {NSTEP}")
print(f"Number of independent chains: {NCHAIN}")
print(f"Total attempted moves: {NSTEP * NCHAIN}")

all_pars = []
all_lnprob = []
for j in range(NCHAIN):
    chain_seed = SEED + j
    print(f"Running chain {j + 1}/{NCHAIN} with seed={chain_seed}")
    pars_j, lnprob_j = run_one_chain(chain_seed)
    all_pars.append(pars_j)
    all_lnprob.append(lnprob_j)

pars_all = np.vstack(all_pars)
lnprob_all = np.concatenate(all_lnprob)

nburn = int(BURN_FRAC * NSTEP)
pars_post = np.vstack([p[nburn:] for p in all_pars])
lnprob_post = np.concatenate([lp[nburn:] for lp in all_lnprob])

imax = np.argmax(lnprob_post)
bestfit = pars_post[imax]
_, med, p16, p84, sig_bestfit = summarize_chain(pars_post)
txt = capfit.format_values_with_errors(med, sig_bestfit, labels)
print("\nPosterior summary from post-burn chain:\n")
print(txt.plain)

_ = jam_lnprob(bestfit)
out = jam_lnprob.out

if FIT_INCLINATION:
    _, ratio_best, lg_mbh_best, inc_best = unpack_parameters(bestfit)
    _, ratio_med, lg_mbh_med, inc_med = unpack_parameters(med)
else:
    _, ratio_best, lg_mbh_best, inc_best = unpack_parameters(bestfit)
    _, ratio_med, lg_mbh_med, inc_med = unpack_parameters(med)

mbh_best = 10.0**lg_mbh_best
mbh_med = 10.0**lg_mbh_med
mbh_p16 = 10.0**p16[-1]
mbh_p84 = 10.0**p84[-1]

print("\nDerived quantities (best posterior sample):\n")
print(f"inclination = {inc_best:.2f} deg")
print(f"sigma_z/sigma_R = {ratio_best:.3f}")
print(f"M_BH = {mbh_best:.3e} Msun")
print(f"fixed M/L = {FIXED_ML:.3f}")
print(f"reduced chi2 reported by JAM = {out.chi2:.3f}")

# Save numerical outputs
np.savez(
    os.path.join(OUTPUT_PATH, "sombrero_jam_adamet_fixedml_results.npz"),
    pars_all=pars_all,
    lnprob_all=lnprob_all,
    pars_post=pars_post,
    lnprob_post=lnprob_post,
    bestfit=bestfit,
    median=med,
    p16=p16,
    p84=p84,
    goodbins=goodbins,
    model_vrms=out.model,
    data_vrms=rms,
    err_vrms=erms,
    xbin=xbin,
    ybin=ybin,
)

summary_path = os.path.join(OUTPUT_PATH, "sombrero_jam_adamet_fixedml_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Sombrero JAM + AdaMet (fixed M/L) summary\n")
    f.write("=====================================\n\n")
    f.write(f"MGE_PATH = {MGE_PATH}\n")
    f.write(f"KIN_PATH = {KIN_PATH}\n")
    f.write(f"OUTPUT_PATH = {OUTPUT_PATH}\n\n")
    f.write(f"DISTANCE_MPC = {DISTANCE_MPC}\n")
    f.write(f"PIXEL_SCALE_ARCSEC = {PIXEL_SCALE_ARCSEC}\n")
    f.write(f"PIXEL_SIZE_KIN_ARCSEC = {PIXEL_SIZE_KIN_ARCSEC}\n")
    f.write(f"PA_OFFSET_DEG = {PA_OFFSET_DEG}\n")
    f.write(f"FIXED_ML = {FIXED_ML}\n")
    f.write(f"FIT_INCLINATION = {FIT_INCLINATION}\n")
    f.write(f"INC_FIXED_DEG = {INC_FIXED_DEG}\n")
    f.write(f"LG_MBH_MIN = {LG_MBH_MIN}\n")
    f.write(f"LG_MBH_MAX = {LG_MBH_MAX}\n")
    f.write(f"NSTEP = {NSTEP}\n")
    f.write(f"NCHAIN = {NCHAIN}\n")
    f.write(f"BURN_FRAC = {BURN_FRAC}\n")
    f.write(f"goodbins_used = {goodbins.sum()} / {goodbins.size}\n\n")
    f.write("Posterior median +/- 1 sigma:\n")
    f.write(txt.plain)
    f.write("\n\n")
    f.write(f"Best-fit inclination [deg] = {inc_best:.6f}\n")
    f.write(f"Median inclination [deg] = {inc_med:.6f}\n")
    f.write(f"Best-fit sigma_z/sigma_R = {ratio_best:.6f}\n")
    f.write(f"Median sigma_z/sigma_R = {ratio_med:.6f}\n")
    f.write(f"Best-fit log10(M_BH/Msun) = {lg_mbh_best:.6f}\n")
    f.write(f"Median log10(M_BH/Msun) = {lg_mbh_med:.6f}\n")
    f.write(f"Median M_BH [Msun] = {mbh_med:.6e}\n")
    f.write(f"16th percentile M_BH [Msun] = {mbh_p16:.6e}\n")
    f.write(f"84th percentile M_BH [Msun] = {mbh_p84:.6e}\n")
    f.write(f"Fixed M/L = {FIXED_ML:.6f}\n")
    f.write(f"JAM reduced chi2 (best sample) = {out.chi2:.6f}\n")

if SAVE_FULL_CHAIN_TXT:
    save_chain_txt(
        os.path.join(OUTPUT_PATH, "sombrero_jam_adamet_fixedml_chain_all.txt"),
        pars_all,
        lnprob_all,
        labels,
    )

if SAVE_POST_BURN_TXT:
    save_chain_txt(
        os.path.join(OUTPUT_PATH, "sombrero_jam_adamet_fixedml_chain_postburn.txt"),
        pars_post,
        lnprob_post,
        labels,
    )


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
fig = corner_plot(pars_post, lnprob_post, labels=labels, extents=bounds)
fig.text(0.34, 0.99, txt.latex, ha="left", va="top")

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
plt.savefig(os.path.join(OUTPUT_PATH, "sombrero_jam_adamet_fixedml_corner.png"), dpi=600, bbox_inches="tight")
plt.show()
