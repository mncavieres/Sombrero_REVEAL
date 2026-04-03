import os
import time
import shutil
import threading
import traceback
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from dataclasses import dataclass

# External packages from Michele Cappellari + dynesty
# pip install jampy dynesty plotbin scipy
import jampy as jam
from plotbin.plot_velfield import plot_velfield
from plotbin.symmetrize_velfield import symmetrize_velfield
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty.pool import Pool as DynestyPool


# -----------------------------------------------------------------------------
# USER INPUTS
# -----------------------------------------------------------------------------
MGE_PATH = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_NAGN_0deg_pa_positive_gauss/mge_solution.csv"
KIN_PATH = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/antoine/M104_stellar_Kin_rotated.csv"
KIN_SEP = ";"
OUTPUT_PATH = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Plots/nested_sampling_bh_fixedML_rot"

DISTANCE_MPC = 9.55
PIXEL_SCALE_ARCSEC = 0.031          # JWST image scale used for the MGE fit
PIXEL_SIZE_KIN_ARCSEC = 1#0.103       # NIRSpec spaxel/bin size; replace if different
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
# This script uses dynesty.DynamicNestedSampler explicitly.
NLIVE_INIT = 500
NLIVE_BATCH = 500
DLOGZ_INIT = 0.5
N_EFFECTIVE_INIT = 3000
N_EFFECTIVE_TARGET = 15000
MAXBATCH = None
SAMPLE = "rwalk"
BOUND = "multi"
WALKS = 32
SEED = 7

# Parallel dynesty controls
USE_MULTIPROCESSING = True
NCPU = min(8, os.cpu_count() or 1)

# Partial-save / checkpoint controls
CHECKPOINT_FILENAME = "sombrero_jam_dynamic_fixedml_checkpoint.save"
CHECKPOINT_EVERY_SEC = 600.0          # dynesty sampler state written this often
PARTIAL_SAVE_EVERY_SEC = 600.0        # write latest txt/npz/plots this often from checkpoint
RESUME_IF_CHECKPOINT_EXISTS = True
SAVE_PARTIAL_PLOTS = True
KEEP_CHECKPOINT_AFTER_SUCCESS = True

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


@dataclass
class JAMModelData:
    surf_lum: np.ndarray
    sigma_lum: np.ndarray
    q_obs_lum: np.ndarray
    surf_pot: np.ndarray
    sigma_pot: np.ndarray
    q_obs_pot: np.ndarray
    distance_mpc: float
    xbin: np.ndarray
    ybin: np.ndarray
    rms: np.ndarray
    erms: np.ndarray
    vel: np.ndarray
    evel: np.ndarray
    sig: np.ndarray
    esig: np.ndarray
    flux_obs: np.ndarray
    goodbins: np.ndarray
    pixsize: float
    sigmapsf: np.ndarray
    normpsf: np.ndarray
    inc_fixed_deg: float


@dataclass
class PriorConfig:
    fit_inclination: bool
    q_intr_min_lo: float | None
    q_intr_min_hi: float | None
    ratio_min: float
    ratio_max: float
    lg_mbh_min: float
    lg_mbh_max: float

    @property
    def ndim(self):
        return 3 if self.fit_inclination else 2

    # @property
    # def labels(self):
    #     if self.fit_inclination:
    #         return [r"$q_{\\rm intr,min}$", r"$\\sigma_z/\\sigma_R$", r"$\\log_{10} M_{\\rm BH}$"]
    #     return [r"$\\sigma_z/\\sigma_R$", r"$\\log_{10} M_{\\rm BH}$"]

    @property
    def labels(self):
        if self.fit_inclination:
            return [r"$q_{\rm intr,min}$", r"$\sigma_z/\sigma_R$", r"$\log_{10} M_{\rm BH}$"]
        return [r"$\sigma_z/\sigma_R$", r"$\log_{10} M_{\rm BH}$"]

class PeriodicSnapshotter(threading.Thread):
    """Periodically restore the dynesty checkpoint and write partial products."""

    def __init__(self, checkpoint_file, output_path, model, prior_cfg, seed, interval_sec=600.0):
        super().__init__(daemon=True)
        self.checkpoint_file = checkpoint_file
        self.output_path = output_path
        self.model = model
        self.prior_cfg = prior_cfg
        self.seed = int(seed)
        self.interval_sec = float(interval_sec)
        self.stop_event = threading.Event()
        self.last_checkpoint_mtime = None
        self.snapshot_counter = 0

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.wait(self.interval_sec):
            self.try_snapshot()

    def try_snapshot(self, force=False):
        if not os.path.exists(self.checkpoint_file):
            return

        try:
            mtime = os.path.getmtime(self.checkpoint_file)
        except OSError:
            return

        if (not force) and (self.last_checkpoint_mtime is not None) and (mtime <= self.last_checkpoint_mtime):
            return

        tmp_path = os.path.join(self.output_path, ".partial_restore_checkpoint.tmp")
        try:
            shutil.copy2(self.checkpoint_file, tmp_path)
            sampler = DynamicNestedSampler.restore(tmp_path)
            results = sampler.results
            if len(results.samples) == 0:
                return
            self.snapshot_counter += 1
            write_all_outputs(
                results,
                self.model,
                self.prior_cfg,
                self.output_path,
                np.random.default_rng(self.seed + 1000 + self.snapshot_counter),
                prefix="sombrero_jam_dynamic_fixedml_partial",
                partial=True,
                save_plots=SAVE_PARTIAL_PLOTS,
            )
            self.last_checkpoint_mtime = mtime
            print(
                f"\n[partial-save] Wrote checkpoint-derived outputs #{self.snapshot_counter} "
                f"from {self.checkpoint_file}"
            )
        except Exception as exc:
            print(f"\n[partial-save] Warning: could not restore checkpoint yet: {exc}")
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass



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



def clean_label(lab):
    return lab.replace("$", "").replace("\\", "")



def save_chain_txt(path, samples, weights=None, logl=None, labels=None):
    cols = [samples]
    header_parts = []
    if labels is None:
        labels = [f"par{i}" for i in range(samples.shape[1])]
    header_parts.extend([clean_label(lab) for lab in labels])
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



def unpack_parameters(pars, prior_cfg, qmin_obs, inc_fixed_deg):
    if prior_cfg.fit_inclination:
        q_intr_min, ratio, lg_mbh = np.asarray(pars, dtype=float)
        if not (0.0 < q_intr_min < qmin_obs):
            return None, None, None, None
        inc = np.degrees(
            np.arctan2(np.sqrt(1.0 - qmin_obs**2), np.sqrt(qmin_obs**2 - q_intr_min**2))
        )
    else:
        ratio, lg_mbh = np.asarray(pars, dtype=float)
        inc = inc_fixed_deg
        q_intr_min = None
    return q_intr_min, ratio, lg_mbh, inc



def prior_transform(u, prior_cfg):
    """Map a unit-cube sample to the physical parameter space."""
    u = np.asarray(u, dtype=float)
    if prior_cfg.fit_inclination:
        q_intr_min = prior_cfg.q_intr_min_lo + u[0] * (prior_cfg.q_intr_min_hi - prior_cfg.q_intr_min_lo)
        ratio = prior_cfg.ratio_min + u[1] * (prior_cfg.ratio_max - prior_cfg.ratio_min)
        lg_mbh = prior_cfg.lg_mbh_min + u[2] * (prior_cfg.lg_mbh_max - prior_cfg.lg_mbh_min)
        return np.array([q_intr_min, ratio, lg_mbh])
    ratio = prior_cfg.ratio_min + u[0] * (prior_cfg.ratio_max - prior_cfg.ratio_min)
    lg_mbh = prior_cfg.lg_mbh_min + u[1] * (prior_cfg.lg_mbh_max - prior_cfg.lg_mbh_min)
    return np.array([ratio, lg_mbh])



def jam_loglike(pars, model, prior_cfg):
    qmin_obs = np.min(model.q_obs_lum)
    q_intr_min, ratio, lg_mbh, inc = unpack_parameters(pars, prior_cfg, qmin_obs, model.inc_fixed_deg)
    if inc is None:
        return -np.inf
    if not (prior_cfg.ratio_min <= ratio <= prior_cfg.ratio_max):
        return -np.inf
    if not (prior_cfg.lg_mbh_min <= lg_mbh <= prior_cfg.lg_mbh_max):
        return -np.inf

    beta = np.full_like(model.q_obs_lum, 1.0 - ratio**2)
    mbh = 10.0 ** lg_mbh

    try:
        out = jam.axi.proj(
            model.surf_lum,
            model.sigma_lum,
            model.q_obs_lum,
            model.surf_pot,
            model.sigma_pot,
            model.q_obs_pot,
            inc,
            mbh,
            model.distance_mpc,
            model.xbin,
            model.ybin,
            plot=False,
            pixsize=model.pixsize,
            quiet=1,
            sigmapsf=model.sigmapsf,
            normpsf=model.normpsf,
            goodbins=model.goodbins,
            align="cyl",
            beta=beta,
            data=model.rms,
            errors=model.erms,
            flux_obs=model.flux_obs,
            ml=1,
            moment="zz",
        )
    except Exception:
        return -np.inf

    resid = (model.rms[model.goodbins] - out.model[model.goodbins]) / model.erms[model.goodbins]
    chi2 = resid @ resid
    return -0.5 * chi2



def compute_bestfit_jam_outputs(pars, model, prior_cfg):
    """
    Compute best-fit JAM outputs for the fitted second moment, plus a
    post-processed first-moment velocity field and corresponding dispersion map.

    The fit itself uses only Vrms. The velocity map generated here is therefore
    a qualitative check plot, obtained from a separate JAM first-moment call
    with kappa determined from the observed LOS velocity field.
    """
    qmin_obs = np.min(model.q_obs_lum)
    _, ratio, lg_mbh, inc = unpack_parameters(pars, prior_cfg, qmin_obs, model.inc_fixed_deg)
    beta = np.full_like(model.q_obs_lum, 1.0 - ratio**2)
    mbh = 10.0 ** lg_mbh

    out_vrms = jam.axi.proj(
        model.surf_lum,
        model.sigma_lum,
        model.q_obs_lum,
        model.surf_pot,
        model.sigma_pot,
        model.q_obs_pot,
        inc,
        mbh,
        model.distance_mpc,
        model.xbin,
        model.ybin,
        plot=False,
        pixsize=model.pixsize,
        quiet=1,
        sigmapsf=model.sigmapsf,
        normpsf=model.normpsf,
        goodbins=model.goodbins,
        align="cyl",
        beta=beta,
        data=model.rms,
        errors=model.erms,
        flux_obs=model.flux_obs,
        ml=1,
        moment="zz",
    )

    out_vel = jam.axi.proj(
        model.surf_lum,
        model.sigma_lum,
        model.q_obs_lum,
        model.surf_pot,
        model.sigma_pot,
        model.q_obs_pot,
        inc,
        mbh,
        model.distance_mpc,
        model.xbin,
        model.ybin,
        plot=False,
        pixsize=model.pixsize,
        quiet=1,
        sigmapsf=model.sigmapsf,
        normpsf=model.normpsf,
        goodbins=model.goodbins,
        align="cyl",
        beta=beta,
        gamma=np.zeros_like(model.q_obs_lum),
        data=model.vel,
        errors=model.evel,
        flux_obs=model.flux_obs,
        ml=1,
        moment="z",
        analytic_los=False,
        kappa=None,
    )

    sig2_model = np.maximum(out_vrms.model**2 - out_vel.model**2, 0.0)
    sigma_model = np.sqrt(sig2_model)
    return out_vrms, out_vel, sigma_model



def make_summary_bundle(results, model, prior_cfg, rng):
    if len(results.samples) == 0:
        raise RuntimeError("No samples are available yet.")

    labels = prior_cfg.labels
    weights = get_posterior_weights(results)
    med, p16, p84, _ = summarize_weighted_samples(results.samples, weights)
    equal_samples = get_equal_weight_samples(results, rng)

    imax = np.argmax(results.logl)
    bestfit = results.samples[imax]
    out_vrms, out_vel, sigma_model = compute_bestfit_jam_outputs(bestfit, model, prior_cfg)

    _, ratio_best, lg_mbh_best, inc_best = unpack_parameters(bestfit, prior_cfg, np.min(model.q_obs_lum), model.inc_fixed_deg)
    _, ratio_med, lg_mbh_med, inc_med = unpack_parameters(med, prior_cfg, np.min(model.q_obs_lum), model.inc_fixed_deg)

    mbh_best = 10.0 ** lg_mbh_best
    mbh_med = 10.0 ** lg_mbh_med
    mbh_p16 = 10.0 ** p16[-1]
    mbh_p84 = 10.0 ** p84[-1]

    summary_lines = format_summary_lines(labels, med, p16, p84)

    return {
        "labels": labels,
        "weights": weights,
        "med": med,
        "p16": p16,
        "p84": p84,
        "equal_samples": equal_samples,
        "bestfit": bestfit,
        "out_vrms": out_vrms,
        "out_vel": out_vel,
        "sigma_model": sigma_model,
        "ratio_best": ratio_best,
        "lg_mbh_best": lg_mbh_best,
        "inc_best": inc_best,
        "ratio_med": ratio_med,
        "lg_mbh_med": lg_mbh_med,
        "inc_med": inc_med,
        "mbh_best": mbh_best,
        "mbh_med": mbh_med,
        "mbh_p16": mbh_p16,
        "mbh_p84": mbh_p84,
        "summary_lines": summary_lines,
    }



def save_vrms_comparison(model, out_vrms, output_file):
    xbin, ybin = model.xbin, model.ybin
    rms, goodbins, flux_obs = model.rms, model.goodbins, model.flux_obs

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

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)



def save_kinematic_checkplot(model, out_vel, sigma_model, output_file):
    xbin, ybin = model.xbin, model.ybin
    vel, sig, goodbins, flux_obs = model.vel, model.sig, model.goodbins, model.flux_obs

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
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)



def write_all_outputs(results, model, prior_cfg, output_path, rng, prefix, partial=False, save_plots=True):
    bundle = make_summary_bundle(results, model, prior_cfg, rng)

    labels = bundle["labels"]
    weights = bundle["weights"]
    med = bundle["med"]
    p16 = bundle["p16"]
    p84 = bundle["p84"]
    equal_samples = bundle["equal_samples"]
    bestfit = bundle["bestfit"]
    out_vrms = bundle["out_vrms"]
    out_vel = bundle["out_vel"]
    sigma_model = bundle["sigma_model"]
    summary_lines = bundle["summary_lines"]

    summary_path = os.path.join(output_path, f"{prefix}_summary.txt")
    weighted_path = os.path.join(output_path, f"{prefix}_weighted_samples.txt")
    equal_path = os.path.join(output_path, f"{prefix}_equal_weight_samples.txt")
    npz_path = os.path.join(output_path, f"{prefix}_results.npz")

    np.savez(
        npz_path,
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
        goodbins=model.goodbins,
        model_vrms=out_vrms.model,
        data_vrms=model.rms,
        err_vrms=model.erms,
        model_vel=out_vel.model,
        data_vel=model.vel,
        err_vel=model.evel,
        model_sigma=sigma_model,
        data_sigma=model.sig,
        err_sigma=model.esig,
        jam_kappa=out_vel.kappa,
        xbin=model.xbin,
        ybin=model.ybin,
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Sombrero JAM + dynesty summary [{prefix}]\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"partial_snapshot = {partial}\n")
        f.write(f"MGE_PATH = {MGE_PATH}\n")
        f.write(f"KIN_PATH = {KIN_PATH}\n")
        f.write(f"OUTPUT_PATH = {output_path}\n")
        f.write(f"CHECKPOINT_FILE = {os.path.join(output_path, CHECKPOINT_FILENAME)}\n\n")
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
        f.write("SAMPLER = DynamicNestedSampler\n")
        f.write(f"NLIVE_INIT = {NLIVE_INIT}\n")
        f.write(f"NLIVE_BATCH = {NLIVE_BATCH}\n")
        f.write(f"DLOGZ_INIT = {DLOGZ_INIT}\n")
        f.write(f"N_EFFECTIVE_INIT = {N_EFFECTIVE_INIT}\n")
        f.write(f"N_EFFECTIVE_TARGET = {N_EFFECTIVE_TARGET}\n")
        f.write(f"MAXBATCH = {MAXBATCH}\n")
        f.write(f"SAMPLE = {SAMPLE}\n")
        f.write(f"BOUND = {BOUND}\n")
        f.write(f"WALKS = {WALKS}\n")
        f.write(f"USE_MULTIPROCESSING = {USE_MULTIPROCESSING}\n")
        f.write(f"NCPU = {NCPU}\n")
        f.write(f"goodbins_used = {model.goodbins.sum()} / {model.goodbins.size}\n")
        f.write(f"nsamples = {len(results.samples)}\n")
        f.write(f"logZ = {results.logz[-1]:.6f}\n")
        f.write(f"logZerr = {results.logzerr[-1]:.6f}\n\n")
        f.write("Posterior median +/- 1 sigma:\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n")
        f.write(f"Best-fit inclination [deg] = {bundle['inc_best']:.6f}\n")
        f.write(f"Median inclination [deg] = {bundle['inc_med']:.6f}\n")
        f.write(f"Best-fit sigma_z/sigma_R = {bundle['ratio_best']:.6f}\n")
        f.write(f"Median sigma_z/sigma_R = {bundle['ratio_med']:.6f}\n")
        f.write(f"Best-fit log10(M_BH/Msun) = {bundle['lg_mbh_best']:.6f}\n")
        f.write(f"Median log10(M_BH/Msun) = {bundle['lg_mbh_med']:.6f}\n")
        f.write(f"Median M_BH [Msun] = {bundle['mbh_med']:.6e}\n")
        f.write(f"16th percentile M_BH [Msun] = {bundle['mbh_p16']:.6e}\n")
        f.write(f"84th percentile M_BH [Msun] = {bundle['mbh_p84']:.6e}\n")
        f.write(f"Fixed M/L = {FIXED_ML:.6f}\n")
        f.write(f"JAM reduced chi2 (best sample) = {out_vrms.chi2:.6f}\n")
        f.write(f"JAM kappa for velocity check plot = {out_vel.kappa:.6f}\n")

    if SAVE_WEIGHTED_TXT:
        save_chain_txt(weighted_path, results.samples, weights=weights, logl=results.logl, labels=labels)

    if SAVE_EQUAL_WEIGHT_TXT:
        save_chain_txt(equal_path, equal_samples, weights=None, logl=None, labels=labels)

    if save_plots and SAVE_RUNPLOT:
        fig, axes = dyplot.runplot(results)
        fig.savefig(os.path.join(output_path, f"{prefix}_runplot.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    if save_plots and SAVE_TRACEPLOT:
        fig, axes = dyplot.traceplot(results, labels=labels)
        fig.savefig(os.path.join(output_path, f"{prefix}_traceplot.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    if save_plots and SAVE_CORNERPLOT:
        fig, axes = dyplot.cornerplot(results, labels=labels, show_titles=True, title_fmt=".3f")
        fig.savefig(os.path.join(output_path, f"{prefix}_corner.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    if save_plots and SAVE_VRMS_COMPARISON:
        save_vrms_comparison(model, out_vrms, os.path.join(output_path, f"{prefix}_vrms_comparison.png"))

    if save_plots and SAVE_KIN_CHECKPLOT:
        save_kinematic_checkplot(model, out_vel, sigma_model, os.path.join(output_path, f"{prefix}_kinematic_checkplots.png"))

    return bundle



def make_run_nested_kwargs(checkpoint_file, resume=False):
    kwargs = dict(
        checkpoint_file=checkpoint_file,
        checkpoint_every=CHECKPOINT_EVERY_SEC,
        print_progress=True,
    )
    if resume:
        kwargs["resume"] = True
    else:
        kwargs.update(
            nlive_init=NLIVE_INIT,
            nlive_batch=NLIVE_BATCH,
            dlogz_init=DLOGZ_INIT,
            #n_effective_init=N_EFFECTIVE_INIT,
            n_effective=N_EFFECTIVE_TARGET,
        )
        if MAXBATCH is not None:
            kwargs["maxbatch"] = MAXBATCH
    return kwargs


def build_sampler(model, prior_cfg, rng, pool=None):
    common_kwargs = dict(
        ndim=prior_cfg.ndim,
        bound=BOUND,
        sample=SAMPLE,
        walks=WALKS,
        rstate=rng,
    )
    if pool is not None:
        return DynamicNestedSampler(pool.loglike, pool.prior_transform, pool=pool, queue_size=NCPU, **common_kwargs)
    return DynamicNestedSampler(
        jam_loglike,
        prior_transform,
        logl_args=(model, prior_cfg),
        ptform_args=(prior_cfg,),
        **common_kwargs,
    )



def run_sampler_with_optional_resume(model, prior_cfg, rng, checkpoint_file):
    resume = RESUME_IF_CHECKPOINT_EXISTS and os.path.exists(checkpoint_file)

    if USE_MULTIPROCESSING and NCPU > 1:
        print(f"Using dynesty multiprocessing pool with {NCPU} processes")
        with DynestyPool(
            NCPU,
            jam_loglike,
            prior_transform,
            logl_args=(model, prior_cfg),
            ptform_args=(prior_cfg,),
        ) as pool:
            if resume:
                print(f"Resuming from checkpoint: {checkpoint_file}")
                sampler = DynamicNestedSampler.restore(checkpoint_file, pool=pool)
                sampler.run_nested(**make_run_nested_kwargs(checkpoint_file, resume=True))
            else:
                sampler = build_sampler(model, prior_cfg, rng, pool=pool)
                sampler.run_nested(**make_run_nested_kwargs(checkpoint_file, resume=False))
            return sampler.results

    print("Using dynesty in serial mode")
    if resume:
        print(f"Resuming from checkpoint: {checkpoint_file}")
        sampler = DynamicNestedSampler.restore(checkpoint_file)
        sampler.run_nested(**make_run_nested_kwargs(checkpoint_file, resume=True))
    else:
        sampler = build_sampler(model, prior_cfg, rng, pool=None)
        sampler.run_nested(**make_run_nested_kwargs(checkpoint_file, resume=False))
    return sampler.results



def build_model_and_prior():
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

    model = JAMModelData(
        surf_lum=surf_lum,
        sigma_lum=sigma_lum,
        q_obs_lum=q_obs_lum,
        surf_pot=surf_pot,
        sigma_pot=sigma_pot,
        q_obs_pot=q_obs_pot,
        distance_mpc=DISTANCE_MPC,
        xbin=xbin,
        ybin=ybin,
        rms=rms,
        erms=erms,
        vel=vel,
        evel=evel,
        sig=sig,
        esig=esig,
        flux_obs=flux_obs,
        goodbins=goodbins,
        pixsize=PIXEL_SIZE_KIN_ARCSEC,
        sigmapsf=np.asarray(SIGMAPSF, dtype=float),
        normpsf=np.asarray(NORMPSF, dtype=float),
        inc_fixed_deg=INC_FIXED_DEG,
    )

    if FIT_INCLINATION:
        qmin_obs = np.min(q_obs_lum)
        qintr_lo = QINTR_MIN
        qintr_hi = qmin_obs - 1e-4
        if qintr_hi <= qintr_lo:
            raise ValueError("Inclination prior is invalid because q_intr upper bound <= lower bound.")
    else:
        qintr_lo = None
        qintr_hi = None

    prior_cfg = PriorConfig(
        fit_inclination=FIT_INCLINATION,
        q_intr_min_lo=qintr_lo,
        q_intr_min_hi=qintr_hi,
        ratio_min=RATIO_MIN,
        ratio_max=RATIO_MAX,
        lg_mbh_min=LG_MBH_MIN,
        lg_mbh_max=LG_MBH_MAX,
    )
    return model, prior_cfg



def main():
    checkpoint_file = os.path.join(OUTPUT_PATH, CHECKPOINT_FILENAME)
    model, prior_cfg = build_model_and_prior()

    print("Fitting Sombrero JAM model with dynesty and fixed stellar M/L...")
    print(f"Number of fitted bins: {model.goodbins.sum()} / {model.goodbins.size}")
    print(f"Fixed M/L = {FIXED_ML:.3f}")
    print(f"Inclination mode: {'fit' if FIT_INCLINATION else f'fixed at {INC_FIXED_DEG:.1f} deg'}")
    print("sampler = DynamicNestedSampler")
    print(f"nlive_init = {NLIVE_INIT}")
    print(f"nlive_batch = {NLIVE_BATCH}")
    print(f"n_effective_init = {N_EFFECTIVE_INIT}")
    print(f"n_effective_target = {N_EFFECTIVE_TARGET}")
    print(f"Sampler = {SAMPLE}, bound = {BOUND}")
    print(f"Multiprocessing = {USE_MULTIPROCESSING}")
    print(f"NCPU = {NCPU}")
    print(f"Checkpoint file = {checkpoint_file}")
    print(f"Checkpoint every = {CHECKPOINT_EVERY_SEC:.1f} s")
    print(f"Partial-save every = {PARTIAL_SAVE_EVERY_SEC:.1f} s")

    rng = np.random.default_rng(SEED)
    snapshotter = None

    if PARTIAL_SAVE_EVERY_SEC is not None and PARTIAL_SAVE_EVERY_SEC > 0:
        snapshotter = PeriodicSnapshotter(
            checkpoint_file=checkpoint_file,
            output_path=OUTPUT_PATH,
            model=model,
            prior_cfg=prior_cfg,
            seed=SEED,
            interval_sec=PARTIAL_SAVE_EVERY_SEC,
        )
        snapshotter.start()

    try:
        results = run_sampler_with_optional_resume(model, prior_cfg, rng, checkpoint_file)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Attempting one last partial save from the latest checkpoint...")
        if snapshotter is not None:
            snapshotter.try_snapshot(force=True)
            snapshotter.stop()
            snapshotter.join(timeout=2.0)
        raise
    except Exception:
        print("\nSampling failed. Attempting one last partial save from the latest checkpoint...")
        if snapshotter is not None:
            snapshotter.try_snapshot(force=True)
            snapshotter.stop()
            snapshotter.join(timeout=2.0)
        traceback.print_exc()
        raise
    else:
        if snapshotter is not None:
            snapshotter.try_snapshot(force=True)
            snapshotter.stop()
            snapshotter.join(timeout=2.0)

    bundle = write_all_outputs(
        results,
        model,
        prior_cfg,
        OUTPUT_PATH,
        rng,
        prefix="sombrero_jam_dynamic_fixedml",
        partial=False,
        save_plots=True,
    )

    print("\nPosterior summary (weighted median +/- 68% interval):\n")
    for line in bundle["summary_lines"]:
        print(line)
    print("\nDerived quantities (maximum-likelihood sample):\n")
    print(f"inclination = {bundle['inc_best']:.2f} deg")
    print(f"sigma_z/sigma_R = {bundle['ratio_best']:.3f}")
    print(f"M_BH = {bundle['mbh_best']:.3e} Msun")
    print(f"fixed M/L = {FIXED_ML:.3f}")
    print(f"reduced chi2 reported by JAM = {bundle['out_vrms'].chi2:.3f}")
    print(f"best-fit JAM kappa for velocity check plot = {bundle['out_vel'].kappa:.3f}")
    print(f"logZ = {results.logz[-1]:.6f} +/- {results.logzerr[-1]:.6f}")

    if (not KEEP_CHECKPOINT_AFTER_SUCCESS) and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("\nDone. Output files written to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
