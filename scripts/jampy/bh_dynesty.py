import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# External packages from Michele Cappellari + dynesty
# pip install jampy dynesty plotbin scipy
import jampy as jam
from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc


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
M_SUN_AB_F200W = 4.93               # adopted solar absolute AB magnitude

# If your kinematics are not already aligned with the galaxy major axis,
# rotate them by this angle (degrees) BEFORE fitting.
PA_OFFSET_DEG = 0.0

# PSF model for the kinematics: replace with your actual PSF model.
SIGMAPSF = np.array([0.05, 0.15])   # Gaussian sigmas in arcsec
NORMPSF = np.array([0.7, 0.3])

# Fixed stellar-population M/L in F200W units.
FIXED_ML = 0.86

# Fix inclination unless you explicitly want to fit it.
FIT_INCLINATION = False
INC_FIXED_DEG = 87.0

# dynesty controls
NLIVE_INIT = 500
NLIVE_BATCH = 500
DLOGZ_INIT = 0.02
SAMPLE = "rwalk"
BOUND = "multi"
WALKS = 32
SEED = 7

# Optional quality cuts
EXCLUDE_R_ARCSEC = None             # e.g. 0.05, or None
EXCLUDE_HIGHSIGMA_KMS = None        # e.g. 420.0, or None

# Physically motivated BH search range for Sombrero.
LG_MBH_MIN = 7.0
LG_MBH_MAX = 10.2
RATIO_MIN = 0.50
RATIO_MAX = 1.20
QINTR_MIN = 0.05

SAVE_WEIGHTED_TXT = True
SAVE_EQUAL_WEIGHT_TXT = True
SAVE_TRACEPLOT = True
SAVE_RUNPLOT = True
SAVE_CORNERPLOT = True
SAVE_VRMS_COMPARISON = True
SAVE_KIN_CHECKPLOT = True


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
    pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
    m_sun_ab=M_SUN_AB_F200W,
):
    """Build the three JAM luminosity arrays from your MGE table."""
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
        arg = -0.5 * (x**2 + (y / q) ** 2) / sig**2
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
    evrms = np.sqrt((vel * evel) ** 2 + (sig * esig) ** 2) / vrms

    med_evrms = np.nanmedian(evrms[np.isfinite(evrms)])
    floor = 0.02 * np.nanmedian(vrms[np.isfinite(vrms)])
    evrms = np.where(np.isfinite(evrms) & (evrms > 0), evrms, med_evrms)
    evrms = np.maximum(evrms, floor)

    return x, y, vrms, evrms, vel, evel, sig, esig



def weighted_quantile(values, quantiles, sample_weight=None):
    """Weighted quantiles of a 1D array."""
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(values)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    m = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    values = values[m]
    sample_weight = sample_weight[m]

    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    cdf = np.cumsum(sample_weight)
    cdf /= cdf[-1]
    return np.interp(quantiles, cdf, values)



def summarize_weighted_samples(samples, weights):
    """Posterior median and central 68% interval for each parameter."""
    med = []
    p16 = []
    p84 = []
    for j in range(samples.shape[1]):
        q16, q50, q84 = weighted_quantile(samples[:, j], [0.15865, 0.5, 0.84135], weights)
        p16.append(q16)
        med.append(q50)
        p84.append(q84)
    med = np.array(med)
    p16 = np.array(p16)
    p84 = np.array(p84)
    sig = 0.5 * (p84 - p16)
    return med, p16, p84, sig



def format_summary_lines(labels, med, p16, p84):
    lines = []
    for lab, q50, q16, q84 in zip(labels, med, p16, p84):
        lines.append(f"{lab:>24s} = {q50:.6f}  -{q50 - q16:.6f}  +{q84 - q50:.6f}")
    return lines



def save_chain_txt(path, samples, weights=None, logl=None, labels=None):
    cols = [samples]
    header_parts = []
    if labels is None:
        labels = [f"par{i}" for i in range(samples.shape[1])]
    header_parts.extend([lab.replace("$", "").replace("\\", "") for lab in labels])
    if weights is not None:
        cols.append(np.asarray(weights)[:, None])
        header_parts.append("weight")
    if logl is not None:
        cols.append(np.asarray(logl)[:, None])
        header_parts.append("logl")
    arr = np.hstack(cols)
    np.savetxt(path, arr, header=" ".join(header_parts))



def get_posterior_weights(results):
    """Posterior importance weights, normalized to sum to one."""
    if hasattr(results, "importance_weights"):
        w = np.asarray(results.importance_weights(), dtype=float)
    else:
        w = np.exp(np.asarray(results.logwt) - results.logz[-1])
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    w /= np.sum(w)
    return w



def get_equal_weight_samples(results, rng):
    """Equal-weight posterior samples with backward-compatible fallbacks."""
    if hasattr(results, "samples_equal"):
        try:
            return np.asarray(results.samples_equal(rstate=rng), dtype=float)
        except TypeError:
            return np.asarray(results.samples_equal(), dtype=float)
    return np.asarray(dyfunc.resample_equal(results.samples, get_posterior_weights(results), rstate=rng), dtype=float)


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

xbin, ybin, rms, erms, vel, evel, sig, esig = load_kinematics_table(KIN_PATH, sep=KIN_SEP)
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
# PRIOR SETUP
# -----------------------------------------------------------------------------
if FIT_INCLINATION:
    qmin_obs = np.min(q_obs_lum)
    qintr_lo = QINTR_MIN
    qintr_hi = qmin_obs - 1e-4
    if qintr_hi <= qintr_lo:
        raise ValueError("Inclination prior is invalid because q_intr upper bound <= lower bound.")
    labels = [r"$q_{\\rm intr,min}$", r"$\\sigma_z/\\sigma_R$", r"$\\log_{10} M_{\\rm BH}$"]
    ndim = 3
else:
    labels = [r"$\\sigma_z/\\sigma_R$", r"$\\log_{10} M_{\\rm BH}$"]
    ndim = 2



def unpack_parameters(pars):
    if FIT_INCLINATION:
        q_intr_min, ratio, lg_mbh = pars
        qmin_obs_local = np.min(q_obs_lum)
        if not (0.0 < q_intr_min < qmin_obs_local):
            return None, None, None, None
        inc = np.degrees(
            np.arctan2(np.sqrt(1.0 - qmin_obs_local**2), np.sqrt(qmin_obs_local**2 - q_intr_min**2))
        )
    else:
        ratio, lg_mbh = pars
        inc = INC_FIXED_DEG
        q_intr_min = None
    return q_intr_min, ratio, lg_mbh, inc



def prior_transform(u):
    """Map a unit-cube sample to the physical parameter space."""
    u = np.asarray(u, dtype=float)
    if FIT_INCLINATION:
        q_intr_min = qintr_lo + u[0] * (qintr_hi - qintr_lo)
        ratio = RATIO_MIN + u[1] * (RATIO_MAX - RATIO_MIN)
        lg_mbh = LG_MBH_MIN + u[2] * (LG_MBH_MAX - LG_MBH_MIN)
        return np.array([q_intr_min, ratio, lg_mbh])
    ratio = RATIO_MIN + u[0] * (RATIO_MAX - RATIO_MIN)
    lg_mbh = LG_MBH_MIN + u[1] * (LG_MBH_MAX - LG_MBH_MIN)
    return np.array([ratio, lg_mbh])



def jam_loglike(pars):
    q_intr_min, ratio, lg_mbh, inc = unpack_parameters(pars)
    if inc is None:
        return -np.inf
    if not (RATIO_MIN <= ratio <= RATIO_MAX):
        return -np.inf
    if not (LG_MBH_MIN <= lg_mbh <= LG_MBH_MAX):
        return -np.inf

    beta = np.full_like(q_obs_lum, 1.0 - ratio**2)
    mbh = 10.0 ** lg_mbh

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

    jam_loglike.out = out
    resid = (rms[goodbins] - out.model[goodbins]) / erms[goodbins]
    chi2 = resid @ resid
    return -0.5 * chi2



def compute_bestfit_jam_outputs(pars):
    """
    Compute best-fit JAM outputs for the fitted second moment, plus a
    post-processed first-moment velocity field and corresponding dispersion map.

    The fit itself uses only Vrms. The velocity map generated here is therefore
    a qualitative check plot, obtained from a separate JAM first-moment call
    with kappa determined from the observed LOS velocity field.
    """
    q_intr_min, ratio, lg_mbh, inc = unpack_parameters(pars)
    beta = np.full_like(q_obs_lum, 1.0 - ratio**2)
    mbh = 10.0 ** lg_mbh

    out_vrms = jam.axi.proj(
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

    out_vel = jam.axi.proj(
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
        gamma=np.zeros_like(q_obs_lum),
        data=vel,
        errors=evel,
        flux_obs=flux_obs,
        ml=1,
        moment="z",
        analytic_los=False,
        kappa=None,
    )

    sig2_model = np.maximum(out_vrms.model**2 - out_vel.model**2, 0.0)
    sigma_model = np.sqrt(sig2_model)
    return out_vrms, out_vel, sigma_model


# -----------------------------------------------------------------------------
# RUN DYNESTY
# -----------------------------------------------------------------------------
print("Fitting Sombrero JAM model with dynesty and fixed stellar M/L...")
print(f"Number of fitted bins: {goodbins.sum()} / {goodbins.size}")
print(f"Fixed M/L = {FIXED_ML:.3f}")
print(f"Inclination mode: {'fit' if FIT_INCLINATION else f'fixed at {INC_FIXED_DEG:.1f} deg'}")
print(f"nlive_init = {NLIVE_INIT}")
print(f"nlive_batch = {NLIVE_BATCH}")
print(f"Sampler = {SAMPLE}, bound = {BOUND}")

rng = np.random.default_rng(SEED)
sampler = DynamicNestedSampler(
    jam_loglike,
    prior_transform,
    ndim,
    bound=BOUND,
    sample=SAMPLE,
    walks=WALKS,
    rstate=rng,
)
sampler.run_nested(
    nlive_init=NLIVE_INIT,
    nlive_batch=NLIVE_BATCH,
    dlogz_init=DLOGZ_INIT,
    print_progress=True,
)
results = sampler.results

weights = get_posterior_weights(results)
med, p16, p84, sig_post = summarize_weighted_samples(results.samples, weights)
equal_samples = get_equal_weight_samples(results, rng)

imax = np.argmax(results.logl)
bestfit = results.samples[imax]
out_vrms, out_vel, sigma_model = compute_bestfit_jam_outputs(bestfit)

if FIT_INCLINATION:
    _, ratio_best, lg_mbh_best, inc_best = unpack_parameters(bestfit)
    _, ratio_med, lg_mbh_med, inc_med = unpack_parameters(med)
else:
    _, ratio_best, lg_mbh_best, inc_best = unpack_parameters(bestfit)
    _, ratio_med, lg_mbh_med, inc_med = unpack_parameters(med)

mbh_best = 10.0 ** lg_mbh_best
mbh_med = 10.0 ** lg_mbh_med
mbh_p16 = 10.0 ** p16[-1]
mbh_p84 = 10.0 ** p84[-1]

summary_lines = format_summary_lines(labels, med, p16, p84)
print("\nPosterior summary (weighted median +/- 68% interval):\n")
for line in summary_lines:
    print(line)
print("\nDerived quantities (maximum-likelihood sample):\n")
print(f"inclination = {inc_best:.2f} deg")
print(f"sigma_z/sigma_R = {ratio_best:.3f}")
print(f"M_BH = {mbh_best:.3e} Msun")
print(f"fixed M/L = {FIXED_ML:.3f}")
print(f"reduced chi2 reported by JAM = {out_vrms.chi2:.3f}")
print(f"best-fit JAM kappa for velocity check plot = {out_vel.kappa:.3f}")
print(f"logZ = {results.logz[-1]:.6f} +/- {results.logzerr[-1]:.6f}")

# Save numerical outputs
np.savez(
    os.path.join(OUTPUT_PATH, "sombrero_jam_dynesty_fixedml_results.npz"),
    samples=results.samples,
    logl=results.logl,
    logwt=results.logwt,
    logz=results.logz,
    logzerr=results.logzerr,
    weights=weights,
    equal_samples=equal_samples,
    bestfit=bestfit,
    median=med,
    p16=p16,
    p84=p84,
    goodbins=goodbins,
    model_vrms=out_vrms.model,
    data_vrms=rms,
    err_vrms=erms,
    model_vel=out_vel.model,
    data_vel=vel,
    err_vel=evel,
    model_sigma=sigma_model,
    data_sigma=sig,
    err_sigma=esig,
    jam_kappa=out_vel.kappa,
    xbin=xbin,
    ybin=ybin,
)

summary_path = os.path.join(OUTPUT_PATH, "sombrero_jam_dynesty_fixedml_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Sombrero JAM + dynesty (fixed M/L) summary\n")
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
    f.write(f"RATIO_MIN = {RATIO_MIN}\n")
    f.write(f"RATIO_MAX = {RATIO_MAX}\n")
    f.write(f"NLIVE_INIT = {NLIVE_INIT}\n")
    f.write(f"NLIVE_BATCH = {NLIVE_BATCH}\n")
    f.write(f"DLOGZ_INIT = {DLOGZ_INIT}\n")
    f.write(f"SAMPLE = {SAMPLE}\n")
    f.write(f"BOUND = {BOUND}\n")
    f.write(f"WALKS = {WALKS}\n")
    f.write(f"goodbins_used = {goodbins.sum()} / {goodbins.size}\n\n")
    f.write("Posterior median +/- 1 sigma:\n")
    for line in summary_lines:
        f.write(line + "\n")
    f.write("\n")
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
    f.write(f"JAM reduced chi2 (best sample) = {out_vrms.chi2:.6f}\n")
    f.write(f"JAM kappa for velocity check plot = {out_vel.kappa:.6f}\n")
    f.write(f"logZ = {results.logz[-1]:.6f}\n")
    f.write(f"logZerr = {results.logzerr[-1]:.6f}\n")

if SAVE_WEIGHTED_TXT:
    save_chain_txt(
        os.path.join(OUTPUT_PATH, "sombrero_jam_dynesty_fixedml_weighted_samples.txt"),
        results.samples,
        weights=weights,
        logl=results.logl,
        labels=labels,
    )

if SAVE_EQUAL_WEIGHT_TXT:
    save_chain_txt(
        os.path.join(OUTPUT_PATH, "sombrero_jam_dynesty_fixedml_equal_weight_samples.txt"),
        equal_samples,
        weights=None,
        logl=None,
        labels=labels,
    )


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
if SAVE_RUNPLOT:
    fig, axes = dyplot.runplot(results)
    fig.savefig(
        os.path.join(OUTPUT_PATH, "sombrero_jam_dynesty_fixedml_runplot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

if SAVE_TRACEPLOT:
    fig, axes = dyplot.traceplot(results, labels=labels)
    fig.savefig(
        os.path.join(OUTPUT_PATH, "sombrero_jam_dynesty_fixedml_traceplot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

if SAVE_CORNERPLOT:
    fig, axes = dyplot.cornerplot(results, labels=labels, show_titles=True, title_fmt=".3f")
    fig.savefig(
        os.path.join(OUTPUT_PATH, "sombrero_jam_dynesty_fixedml_corner.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

if SAVE_VRMS_COMPARISON:
    fig = plt.figure(figsize=(9, 9))
    rms_sym = rms.copy()
    rms_sym[goodbins] = symmetrize_velfield(xbin[goodbins], ybin[goodbins], rms[goodbins])
    vmin_rms, vmax_rms = np.percentile(rms_sym[goodbins], [0.5, 99.5])

    ax1 = fig.add_subplot(2, 1, 1)
    plot_velfield(
        xbin,
        ybin,
        rms_sym,
        vmin=vmin_rms,
        vmax=vmax_rms,
        linescolor="w",
        colorbar=1,
        label=r"Data $V_{\rm rms}$ (km/s)",
        flux=flux_obs,
    )
    ax1.set_ylabel("arcsec")
    ax1.tick_params(labelbottom=False)

    ax2 = fig.add_subplot(2, 1, 2)
    plot_velfield(
        xbin,
        ybin,
        out_vrms.model,
        vmin=vmin_rms,
        vmax=vmax_rms,
        linescolor="w",
        colorbar=1,
        label=r"Model $V_{\rm rms}$ (km/s)",
        flux=flux_obs,
    )
    ax2.set_ylabel("arcsec")
    ax2.set_xlabel("arcsec")

    fig.savefig(
        os.path.join(OUTPUT_PATH, "sombrero_jam_dynesty_fixedml_vrms_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

if SAVE_KIN_CHECKPLOT:
    fig = plt.figure(figsize=(12, 10))

    vel_scale = np.nanpercentile(np.abs(vel[goodbins]), 99.0)
    if not np.isfinite(vel_scale) or vel_scale <= 0:
        vel_scale = np.nanmax(np.abs(vel[goodbins]))
    if not np.isfinite(vel_scale) or vel_scale <= 0:
        vel_scale = 1.0
    vmin_vel, vmax_vel = -vel_scale, vel_scale

    sig_min, sig_max = np.nanpercentile(sig[goodbins], [1.0, 99.0])
    if not np.isfinite(sig_min):
        sig_min = np.nanmin(sig[goodbins])
    if not np.isfinite(sig_max):
        sig_max = np.nanmax(sig[goodbins])

    ax = fig.add_subplot(2, 2, 1)
    plot_velfield(
        xbin,
        ybin,
        vel,
        vmin=vmin_vel,
        vmax=vmax_vel,
        linescolor="w",
        colorbar=1,
        label=r"Data $V_{\rm los}$ (km/s)",
        flux=flux_obs,
    )
    ax.set_ylabel("arcsec")
    ax.tick_params(labelbottom=False)

    ax = fig.add_subplot(2, 2, 2)
    plot_velfield(
        xbin,
        ybin,
        out_vel.model,
        vmin=vmin_vel,
        vmax=vmax_vel,
        linescolor="w",
        colorbar=1,
        label=r"Model $V_{\rm los}$ (km/s)",
        flux=flux_obs,
    )
    ax.tick_params(labelbottom=False, labelleft=False)

    ax = fig.add_subplot(2, 2, 3)
    plot_velfield(
        xbin,
        ybin,
        sig,
        vmin=sig_min,
        vmax=sig_max,
        linescolor="w",
        colorbar=1,
        label=r"Data $\sigma$ (km/s)",
        flux=flux_obs,
    )
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")

    ax = fig.add_subplot(2, 2, 4)
    plot_velfield(
        xbin,
        ybin,
        sigma_model,
        vmin=sig_min,
        vmax=sig_max,
        linescolor="w",
        colorbar=1,
        label=r"Model $\sigma$ (km/s)",
        flux=flux_obs,
    )
    ax.set_xlabel("arcsec")
    ax.tick_params(labelleft=False)

    fig.suptitle("Sombrero JAM best-fit kinematic checkplots", y=0.98)
    fig.savefig(
        os.path.join(OUTPUT_PATH, "sombrero_jam_dynesty_fixedml_kinematic_checkplots.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

print("\nDone. Output files written to:")
print(OUTPUT_PATH)
