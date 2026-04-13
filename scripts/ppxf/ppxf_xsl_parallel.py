
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# # pPXF: refactored integral-field fitting workflow
#
# Main changes relative to the original example:
#
# 1. Stellar kinematics are fit first with iterative clipping and an empirical
#    noise estimate derived from the residuals.
# 2. Stellar populations are then fit with the kinematics fixed and with
#    regularization chosen from the standard Δχ² criterion.
# 3. Corrected formal errors are stored for the nonlinear parameters:
#       error_corr = pp.error * sqrt(pp.chi2)
# 4. Optional Monte Carlo uncertainties on logAge and [M/H] can be computed
#    without changing the main workflow.
# 5. The code avoids using a flat fake noise vector for the stellar-population
#    fit.
#
# The gas fit is kept as a separate example on one spaxel to avoid complicating
# the stellar-population workflow.

# %%
from pathlib import Path
from importlib import resources
from urllib import request
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from powerbin import PowerBin
from plotbin.display_bins import display_bins


# # NumPy >= 2 compatibility patch for pPXF log_rebin
# def log_rebin_compat(lamRange, spec, oversample=False, velscale=None, flux=False):
#     """
#     Drop-in replacement for ppxf.ppxf_util.log_rebin that is compatible with
#     NumPy >= 2, where int(np.array([x])) is no longer allowed. This version
#     also supports spectra with shape (n_pix, n_spec), rebinned along axis 0.
#     """
#     lamRange = np.asarray(lamRange, dtype=float).ravel()
#     if lamRange.size != 2:
#         raise ValueError('lamRange must contain two elements')
#     if not lamRange[0] < lamRange[1]:
#         raise ValueError('It must be lamRange[0] < lamRange[1]')

#     spec = np.asarray(spec, dtype=float)
#     if spec.ndim < 1:
#         raise ValueError('input spectrum must have at least one dimension')

#     n = spec.shape[0]
#     m = int(n * oversample) if oversample else int(n)

#     dLam = (lamRange[1] - lamRange[0]) / (n - 1.0)
#     lim = lamRange / dLam + np.array([-0.5, 0.5], dtype=float)
#     borders = np.linspace(lim[0], lim[1], num=n + 1)
#     logLim = np.log(lim)

#     c = 299792.458
#     if velscale is None:
#         velscale = (np.diff(logLim)[0] / m) * c
#     else:
#         logScale = float(velscale) / c
#         m = int(np.diff(logLim)[0] / logScale)
#         logLim[1] = logLim[0] + m * logScale
#         velscale = float(velscale)

#     newBorders = np.exp(np.linspace(logLim[0], logLim[1], num=m + 1))
#     k = (newBorders - lim[0]).clip(0, n - 1).astype(int)

#     specNew = np.add.reduceat(spec, k, axis=0)[:-1]

#     shape = (-1,) + (1,) * (spec.ndim - 1)
#     valid = (np.diff(k) > 0).reshape(shape)
#     specNew *= valid

#     delta = (newBorders - borders[k]).reshape((m + 1,) + (1,) * (spec.ndim - 1))
#     specNew += np.diff(delta * spec[k], axis=0)

#     if not flux:
#         specNew /= np.diff(newBorders).reshape(shape)

#     logLam = np.log(np.sqrt(newBorders[1:] * newBorders[:-1]) * dLam)
#     return specNew, logLam, velscale


# # Monkey-patch the installed pPXF utility so the rest of the code can stay the same
# util.log_rebin = log_rebin_compat

# %% [markdown]
# ## Configuration

# %%
C = 299792.458  # km/s

LAM_RANGE = [4750.0, 7409.0]   # rest-frame Angstrom
REDSHIFT = 0.003633
TARGET_SN = 100
SN_MIN = 0

SPS_NAME = "xsl"            # "fsps", "galaxev", "emiles", "xsl"
SPS_NORM_RANGE = [5070, 5950]

KIN_DEGREE = 8                 # additive polynomial degree for kinematics
KIN_MDEGREE = 0                # multiplicative polynomial degree for kinematics
POP_MDEGREE = 8                # multiplicative polynomial degree for populations

REGUL_START = 100.0
REGUL_MAX = 5.0e4
REGUL_BRACKET_STEPS = 10
REGUL_BISECT_STEPS = 10

N_PLOTS_BINS = 3
SAVE_ALL_BIN_FITS = True       # save one fit+bestfit plot per bin
N_WORKERS = None               # None -> use os.cpu_count() - 1
MP_START_METHOD = "spawn"      # robust on macOS; set to "fork" if you prefer

RUN_POP_MC = False             # optional; expensive
POP_MC_N = 20
RNG_SEED = 42

FIT_GAS_EXAMPLE = True
GAS_SPIXEL_INDEX = 682         # example spaxel
NGAS_COMP = 3

OBJFILE = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/MUSE/"
    "c30_cubes/c30_DATACUBE_normppxf_skycont_Part1_0000.fits"
)
PLOTS_PATH = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/plots/"
    "ppxf_c30_xsl_refactored"
)

# %% [markdown]
# ## Small utilities

# %%
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mad_std(x, axis=None):
    """
    Robust standard deviation using the MAD.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis)
    return 1.4826 * mad


def estimate_noise_from_differences(arr, axis=0):
    """
    Estimate noise from first differences along a spectral axis.
    For white noise, std(diff)/sqrt(2) ~= std(signal).
    """
    arr = np.asarray(arr, dtype=float)
    diff = np.diff(arr, axis=axis)
    return mad_std(diff, axis=axis) / np.sqrt(2.0)


def safe_positive(a, fill_value=None):
    """
    Replace non-finite or non-positive values by a sensible fallback.
    """
    a = np.asarray(a, dtype=float).copy()
    bad = ~np.isfinite(a) | (a <= 0)
    if np.any(bad):
        if fill_value is None:
            good = a[~bad]
            fill_value = np.nanmedian(good) if good.size else 1.0
        a[bad] = fill_value
    return a


def estimate_spectrum_noise(galaxy, mask=None):
    """
    Scalar noise estimate from the spectrum itself.
    """
    galaxy = np.asarray(galaxy, dtype=float)
    if mask is None:
        x = galaxy[np.isfinite(galaxy)]
    else:
        x = galaxy[np.isfinite(galaxy) & mask]
    if x.size < 10:
        x = galaxy[np.isfinite(galaxy)]
    noise = estimate_noise_from_differences(x, axis=0)
    if not np.isfinite(noise) or noise <= 0:
        noise = robust_sigma(x, zero=1)
    if not np.isfinite(noise) or noise <= 0:
        noise = np.nanstd(x)
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.0
    return float(noise)


def correct_ppxf_errors(pp):
    """
    Correct formal pPXF errors by sqrt(reduced chi^2).
    """
    if getattr(pp, "error", None) is None:
        return None
    return np.asarray(pp.error, dtype=float) * np.sqrt(pp.chi2)


# %% [markdown]
# ## Read the MUSE cube

# %%
class read_data_cube:
    def __init__(self, filename, lam_range, redshift):
        """
        Read the MUSE data cube, de-redshift, log-rebin spectra, and compute
        coordinates plus a spaxel-level S/N estimate for PowerBin.

        This version is specialized to the Sombrero REVEAL cube layout:
          - DATA : science cube
          - STAT : variance cube
          - DQ   : data-quality cube
        """
        self.read_fits_file(filename)

        wave_rest = self.wave / (1 + redshift)
        w = (wave_rest > lam_range[0]) & (wave_rest < lam_range[1])

        wave_rest = wave_rest[w]
        cube = self.cube[w, ...]

        # Build 2D signal/noise maps for PowerBin from the science and variance
        # cubes. We keep the fitting spectra untouched and use DQ only to flag
        # obviously unusable spaxels in the binning maps.
        signal_2d = np.nanmedian(cube, axis=0)

        if self.cubevar is not None:
            cubevar = np.clip(self.cubevar[w, ...], 0, None)
            noise_2d = np.sqrt(np.nanmedian(cubevar, axis=0))
        else:
            cubevar = None
            noise_2d = estimate_noise_from_differences(cube, axis=0)

        if self.dq is not None:
            dq_slice = self.dq[w, ...]
            badfrac = np.mean(dq_slice != 0, axis=0)
            unusable = badfrac >= 0.5
            signal_2d[unusable] = np.nan
            noise_2d[unusable] = np.nan

        signal_2d = np.asarray(signal_2d, dtype=np.float32)
        noise_2d = safe_positive(np.asarray(noise_2d, dtype=np.float32))

        jm = np.nanargmax(signal_2d)
        row, col = map(np.ravel, np.indices(cube.shape[-2:]))
        x = (col - col[jm]) * self.pixsize
        y = (row - row[jm]) * self.pixsize

        npix = cube.shape[0]
        spectra_lin = cube.reshape(npix, -1)

        velscale0 = np.min(C * np.diff(np.log(wave_rest)))
        lam_range_temp = [float(np.min(wave_rest)), float(np.max(wave_rest))]
        print(f"Velocity scale of the data: {velscale0:.2f} km/s per pixel")

        spectra_log, ln_lam_gal, velscale = util.log_rebin(
            lam_range_temp, spectra_lin, velscale=velscale0
        )

        self.spectra = spectra_log
        self.spectra_linear = spectra_lin
        self.x = x
        self.y = y
        self.signal = signal_2d.ravel()
        self.noise = noise_2d.ravel()
        self.row = row + 1
        self.col = col + 1
        self.velscale = velscale
        self.ln_lam_gal = ln_lam_gal
        self.wave_rest = wave_rest
        self.fwhm_gal = self.fwhm_gal / (1 + redshift)

    @staticmethod
    def _orient_cube_nlam_first(arr, header):
        """Ensure the array shape is (n_lambda, ny, nx)."""
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(f"Expected a 3D cube, got shape {arr.shape}")

        nlam = header.get("NAXIS3")
        if nlam is None:
            return arr

        if arr.shape[0] == nlam:
            return arr
        if arr.shape[-1] == nlam:
            return np.moveaxis(arr, -1, 0)
        return arr

    def read_fits_file(self, filename):
        """
        Read the Sombrero REVEAL MUSE cube.

        The file structure provided by the user is:
          HDU 1: DATA  (science)
          HDU 2: STAT  (variance)
          HDU 3: DQ    (data quality)

        Wavelength and spatial WCS are taken from the DATA header when
        available, with a fallback to the primary header for pixel scale.
        """
        with fits.open(filename, memmap=True) as hdul:
            primary_head = hdul[0].header

            if "DATA" in hdul:
                data_hdu = hdul["DATA"]
            else:
                data_hdu = next(
                    (hdu for hdu in hdul if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) == 3),
                    None,
                )
            if data_hdu is None:
                raise ValueError(f"Could not find a 3D science cube in {filename}")

            data_head = data_hdu.header
            cube = self._orient_cube_nlam_first(np.asarray(data_hdu.data, dtype=np.float32), data_head)

            cubevar = None
            if "STAT" in hdul:
                cubevar = self._orient_cube_nlam_first(np.asarray(hdul["STAT"].data, dtype=np.float32), hdul["STAT"].header)
            else:
                for extname in ["VAR", "IVAR", "ERROR", "SIGMA"]:
                    if extname in hdul:
                        arr = self._orient_cube_nlam_first(np.asarray(hdul[extname].data, dtype=np.float32), hdul[extname].header)
                        cubevar = arr**2 if extname in ["ERROR", "SIGMA"] else arr
                        break

            dq = None
            if "DQ" in hdul:
                dq = self._orient_cube_nlam_first(np.asarray(hdul["DQ"].data, dtype=np.uint8), hdul["DQ"].header)

        cd3 = data_head.get("CD3_3", data_head.get("CDELT3"))
        crval3 = data_head.get("CRVAL3")
        if cd3 is None or crval3 is None:
            raise KeyError("Could not determine wavelength solution from the DATA header")
        wave = crval3 + cd3 * np.arange(cube.shape[0], dtype=np.float32)

        if "CD1_1" in data_head:
            pixsize = abs(data_head["CD1_1"]) * 3600
        elif "CDELT1" in data_head:
            pixsize = abs(data_head["CDELT1"]) * 3600
        elif "HIERARCH ESO OCS IPS PIXSCALE" in primary_head:
            pixsize = float(primary_head["HIERARCH ESO OCS IPS PIXSCALE"])
        else:
            raise KeyError("Could not determine spatial pixel size from FITS headers")

        self.cube = cube
        self.cubevar = cubevar
        self.dq = dq
        self.wave = wave
        self.fwhm_gal = 2.62
        self.pixsize = float(pixsize)


# %% [markdown]
# ## Outlier clipping

# %%
def clip_outliers(galaxy, bestfit, mask):
    """
    Iteratively clip pixels deviating by more than 3 sigma in relative error.
    Only pixels currently included by `mask` are considered.
    """
    good = mask.copy()
    while True:
        scale = galaxy[good] @ bestfit[good] / np.sum(bestfit[good] ** 2)
        resid = galaxy[good] - scale * bestfit[good]
        err = robust_sigma(resid, zero=1)

        new_good = good.copy()
        new_good[good] = np.abs(resid) < 3 * err

        if np.array_equal(new_good, good):
            break
        good = new_good

    return good


# %% [markdown]
# ## Stellar fitting helpers

# %%
def fit_stellar_kinematics(
    templates,
    galaxy,
    velscale,
    start,
    mask0,
    lam,
    lam_temp,
    degree=KIN_DEGREE,
    mdegree=KIN_MDEGREE,
    plot=False,
    quiet=False,
):
    """
    Fit stellar kinematics with:
      1) initial fit
      2) outlier clipping
      3) empirical noise estimate from the residuals
      4) final fit with corrected formal errors
    """
    mask = mask0.copy()
    noise0 = np.full_like(galaxy, estimate_spectrum_noise(galaxy, mask=mask))

    pp = ppxf(
        templates, galaxy, noise0, velscale, start,
        moments=2, degree=degree, mdegree=mdegree,
        lam=lam, lam_temp=lam_temp, mask=mask, quiet=quiet
    )

    if plot:
        plt.figure(figsize=(18, 4))
        plt.subplot(121)
        pp.plot()
        plt.title("Initial stellar-kinematics fit")

    mask = clip_outliers(galaxy, pp.bestfit, mask) & mask0

    resid = galaxy[mask] - pp.bestfit[mask]
    noise = np.full_like(galaxy, robust_sigma(resid, zero=1))
    noise = safe_positive(noise)

    pp = ppxf(
        templates, galaxy, noise, velscale, pp.sol,
        moments=2, degree=degree, mdegree=mdegree,
        lam=lam, lam_temp=lam_temp, mask=mask, quiet=quiet
    )

    noise *= np.sqrt(pp.chi2)

    pp = ppxf(
        templates, galaxy, noise, velscale, pp.sol,
        moments=2, degree=degree, mdegree=mdegree,
        lam=lam, lam_temp=lam_temp, mask=mask, quiet=quiet
    )

    pp.clean_mask = mask
    pp.noise_vector = noise
    pp.noise_scalar = float(np.nanmedian(noise))
    pp.error_corr = correct_ppxf_errors(pp)
    pp.optimal_template = templates @ pp.weights

    resid = (pp.galaxy - pp.bestfit)[pp.goodpixels]
    pp.sn = np.nanmedian(pp.galaxy[pp.goodpixels]) / robust_sigma(resid, zero=1)

    if plot:
        plt.subplot(122)
        pp.plot()
        plt.title("Final stellar-kinematics fit")
        plt.tight_layout()

    return pp


# %%
def _population_fit_inner(
    templates,
    galaxy,
    noise,
    velscale,
    kin_sol,
    mask,
    lam,
    lam_temp,
    reg_dim,
    regul=0,
    mdegree=POP_MDEGREE,
    quiet=True,
):
    """
    One population fit with fixed stellar kinematics.
    """
    return ppxf(
        templates, galaxy, noise, velscale, kin_sol,
        moments=-2,                  # keep [V, sigma] fixed
        degree=-1, mdegree=mdegree,  # no additive poly for populations
        lam=lam, lam_temp=lam_temp, mask=mask,
        regul=regul, reg_dim=reg_dim,
        quiet=quiet
    )


# %%
def fit_population_regularized(
    templates,
    galaxy,
    noise,
    velscale,
    kin_sol,
    mask,
    lam,
    lam_temp,
    reg_dim,
    regul_start=REGUL_START,
    regul_max=REGUL_MAX,
    bracket_steps=REGUL_BRACKET_STEPS,
    bisect_steps=REGUL_BISECT_STEPS,
    mdegree=POP_MDEGREE,
    quiet=True,
):
    """
    Fit regularized stellar populations with fixed kinematics.

    Workflow:
      1) unregularized fit
      2) rescale noise so chi2/dof ~ 1
      3) increase regul until Δχ² ~ sqrt(2N)
    """
    scale = np.nanmedian(galaxy[mask])
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanmedian(galaxy[np.isfinite(galaxy)])
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0

    galaxy1 = galaxy / scale
    noise1 = noise / scale

    pp0 = _population_fit_inner(
        templates, galaxy1, noise1, velscale, kin_sol, mask,
        lam, lam_temp, reg_dim, regul=0, mdegree=mdegree, quiet=True
    )

    noise1 *= np.sqrt(pp0.chi2)

    pp0 = _population_fit_inner(
        templates, galaxy1, noise1, velscale, kin_sol, mask,
        lam, lam_temp, reg_dim, regul=0, mdegree=mdegree, quiet=True
    )

    target_dchi2 = np.sqrt(2.0 * pp0.goodpixels.size)

    def total_dchi2(pp):
        return (pp.chi2 - pp0.chi2) * pp0.goodpixels.size

    best_pp = pp0
    best_regul = 0.0

    lo_reg = 0.0
    hi_reg = float(regul_start)
    hi_pp = None

    for _ in range(bracket_steps):
        hi_pp = _population_fit_inner(
            templates, galaxy1, noise1, velscale, kin_sol, mask,
            lam, lam_temp, reg_dim, regul=hi_reg, mdegree=mdegree, quiet=True
        )
        dchi2_hi = total_dchi2(hi_pp)

        if dchi2_hi >= target_dchi2:
            break

        best_pp = hi_pp
        best_regul = hi_reg
        lo_reg = hi_reg
        hi_reg *= 2.0

        if hi_reg > regul_max:
            hi_reg = regul_max
            hi_pp = _population_fit_inner(
                templates, galaxy1, noise1, velscale, kin_sol, mask,
                lam, lam_temp, reg_dim, regul=hi_reg, mdegree=mdegree, quiet=True
            )
            break

    if hi_pp is not None and total_dchi2(hi_pp) >= target_dchi2 and hi_reg > 0:
        for _ in range(bisect_steps):
            if lo_reg <= 0:
                mid_reg = 0.5 * hi_reg
            else:
                mid_reg = np.sqrt(lo_reg * hi_reg)

            mid_pp = _population_fit_inner(
                templates, galaxy1, noise1, velscale, kin_sol, mask,
                lam, lam_temp, reg_dim, regul=mid_reg, mdegree=mdegree, quiet=True
            )

            if total_dchi2(mid_pp) <= target_dchi2:
                best_pp = mid_pp
                best_regul = mid_reg
                lo_reg = mid_reg
            else:
                hi_reg = mid_reg

    if quiet is False:
        print(
            f"Population regularization: regul={best_regul:.3g}, "
            f"target Δχ²={target_dchi2:.2f}, achieved Δχ²={total_dchi2(best_pp):.2f}"
        )

    best_pp.galaxy *= scale
    best_pp.bestfit *= scale
    best_pp.noise_vector = noise1 * scale
    best_pp.noise_scalar = float(np.nanmedian(best_pp.noise_vector))
    best_pp.regul_used = float(best_regul)
    best_pp.target_dchi2 = float(target_dchi2)
    best_pp.error_corr = correct_ppxf_errors(best_pp)

    pp0.galaxy *= scale
    pp0.bestfit *= scale
    pp0.noise_vector = noise1 * scale
    pp0.noise_scalar = float(np.nanmedian(pp0.noise_vector))
    pp0.regul_used = 0.0
    pp0.target_dchi2 = float(target_dchi2)
    pp0.error_corr = correct_ppxf_errors(pp0)

    return best_pp, pp0


# %%
def monte_carlo_population_errors(
    templates,
    model_spectrum,
    noise_vector,
    velscale,
    kin_sol,
    mask,
    lam,
    lam_temp,
    reg_dim,
    sps,
    regul,
    n_mc=20,
    mdegree=POP_MDEGREE,
    seed=RNG_SEED,
):
    """
    Optional MC uncertainties on logAge and [M/H].
    This uses the best-fitting model spectrum plus Gaussian noise and repeats
    the same regularized population fit with fixed kinematics and fixed regul.
    """
    if n_mc <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    ages = []
    metals = []

    scale = np.nanmedian(model_spectrum[mask])
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0

    model1 = model_spectrum / scale
    noise1 = noise_vector / scale

    for _ in range(n_mc):
        gal_mc = model1 + rng.normal(0.0, noise1, size=model1.size)
        pp_mc = _population_fit_inner(
            templates, gal_mc, noise1, velscale, kin_sol, mask,
            lam, lam_temp, reg_dim, regul=regul, mdegree=mdegree, quiet=True
        )
        weights_mc = pp_mc.weights.reshape(reg_dim)
        age_mc, metal_mc = sps.mean_age_metal(weights_mc, quiet=True)
        ages.append(age_mc)
        metals.append(metal_mc)

    return np.nanstd(ages, ddof=1), np.nanstd(metals, ddof=1)


# %% [markdown]
# ## Plotting helpers and multiprocessing

# %%
def compute_spaxel_sn(signal, noise):
    sn = np.asarray(signal, dtype=float) / safe_positive(noise)
    sn[~np.isfinite(sn)] = np.nan
    return sn


def compute_bin_centroids(x, y, bin_num, nbins):
    xbin = np.full(nbins, np.nan)
    ybin = np.full(nbins, np.nan)
    for j in range(nbins):
        w = bin_num == j
        if np.any(w):
            xbin[j] = np.nanmean(x[w])
            ybin[j] = np.nanmean(y[w])
    return xbin, ybin


def compute_bin_sn(signal, noise, bin_num, nbins):
    sn = np.full(nbins, np.nan)
    for j in range(nbins):
        w = bin_num == j
        if np.any(w):
            ssum = np.nansum(signal[w])
            nsum = np.sqrt(np.nansum(np.square(noise[w])))
            sn[j] = ssum / nsum if np.isfinite(nsum) and nsum > 0 else np.nan
    return sn


def add_isophotes(s, levels=None):
    signal = np.asarray(s.signal, dtype=float)
    good = np.isfinite(signal) & (signal > 0)
    if np.count_nonzero(good) < 10:
        return
    if levels is None:
        levels = np.arange(20)
    with np.errstate(invalid='ignore'):
        mu = -2.5 * np.log10(signal[good] / np.nanmax(signal[good]))
    plt.tricontour(s.x[good], s.y[good], mu, levels=levels, colors='k', linewidths=0.5)


def plot_binning_diagnostics(s, bin_num, target_sn, plots_path):
    nbins = np.unique(bin_num).size
    bin_values = np.arange(nbins, dtype=float)
    bin_sn = compute_bin_sn(s.signal, s.noise, bin_num, nbins)
    xbin, ybin = compute_bin_centroids(s.x, s.y, bin_num, nbins)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.sca(axes[0])
    display_bins(s.x, s.y, bin_num, bin_values, colorbar=0, label='Bin ID')
    add_isophotes(s)
    axes[0].set_title('Power bins and isophotes')
    axes[0].set_xlabel('x (arcsec)')
    axes[0].set_ylabel('y (arcsec)')

    plt.sca(axes[1])
    display_bins(s.x, s.y, bin_num, bin_sn, colorbar=1, label='Bin S/N')
    add_isophotes(s)
    axes[1].set_title(f'Achieved bin S/N (target={target_sn:.0f})')
    axes[1].set_xlabel('x (arcsec)')
    axes[1].set_ylabel('y (arcsec)')

    plt.tight_layout()
    plt.savefig(plots_path / 'binning_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    spax_sn = compute_spaxel_sn(s.signal, s.noise)
    rspax = np.hypot(s.x, s.y)
    rbin = np.hypot(xbin, ybin)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(rspax, spax_sn, s=4, alpha=0.15, label='Input spaxels')
    ax.scatter(rbin, bin_sn, s=40, label='Power bins')
    ax.axhline(target_sn, ls='--', color='k', lw=1, label=f'Target S/N={target_sn:.0f}')
    ax.set_xlabel('R (arcsec)')
    ax.set_ylabel('S/N')
    ax.set_title('S/N as a function of projected radius')
    ax.legend(loc='best', frameon=False)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plots_path / 'sn_vs_radius.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return bin_sn, xbin, ybin


def plot_bin_fit_summary(lam_gal, galaxy, bestfit, mask, noise, outpath, title=None):
    resid = galaxy - bestfit
    with np.errstate(invalid='ignore', divide='ignore'):
        resid_n = resid / safe_positive(noise)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    ax.plot(lam_gal, galaxy, lw=0.8, label='Galaxy')
    ax.plot(lam_gal, bestfit, lw=0.8, label='Best fit')
    bad = ~mask
    if np.any(bad):
        ax.scatter(lam_gal[bad], galaxy[bad], s=4, alpha=0.3, label='Masked')
    ax.set_ylabel('Flux')
    if title:
        ax.set_title(title)
    ax.legend(loc='best', frameon=False)

    axes[1].plot(lam_gal, resid_n, lw=0.7)
    axes[1].axhline(0, color='k', lw=0.7)
    axes[1].axhline(3, color='k', ls='--', lw=0.5)
    axes[1].axhline(-3, color='k', ls='--', lw=0.5)
    axes[1].set_ylabel('Res./noise')
    axes[1].set_xlabel('Wavelength (Angstrom)')

    plt.tight_layout()
    plt.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close(fig)


def plot_fit_maps(s, bin_num, velbin, sigbin, lg_age_bin, metalbin, plots_path):
    vrms = np.sqrt(velbin**2 + sigbin**2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    entries = [
        ('VLOS (km/s)', velbin, None),
        ('Sigma (km/s)', sigbin, None),
        ('Vrms (km/s)', vrms, None),
        ('log Age (yr)', lg_age_bin, 'inferno'),
        ('[M/H]', metalbin, 'inferno'),
    ]
    for ax, (label, values, cmap) in zip(axes.flat, entries):
        plt.sca(ax)
        kwargs = {'colorbar': 1, 'label': label}
        if cmap is not None:
            kwargs['cmap'] = cmap
        display_bins(s.x, s.y, bin_num, values, **kwargs)
        add_isophotes(s)
        ax.set_xlabel('x (arcsec)')
        ax.set_ylabel('y (arcsec)')
        ax.set_title(label)

    axes.flat[-1].axis('off')
    plt.tight_layout()
    plt.savefig(plots_path / 'stellar_fit_maps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Individual maps for convenience
    for name, values, cmap in [
        ('vlos_map.png', velbin, None),
        ('sigma_map.png', sigbin, None),
        ('vrms_map.png', vrms, None),
        ('age_map.png', lg_age_bin, 'inferno'),
        ('metallicity_map.png', metalbin, 'inferno'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.sca(ax)
        kwargs = {'colorbar': 1}
        if cmap is not None:
            kwargs['cmap'] = cmap
        display_bins(s.x, s.y, bin_num, values, **kwargs)
        add_isophotes(s)
        plt.tight_layout()
        plt.savefig(plots_path / name, dpi=300, bbox_inches='tight')
        plt.close(fig)


_WORKER_CFG = {}


def _worker_init(cfg):
    global _WORKER_CFG
    _WORKER_CFG = cfg


def fit_one_bin_worker(task):
    jbin, galaxy = task
    cfg = _WORKER_CFG

    pp_kin = fit_stellar_kinematics(
        cfg['templates'], galaxy, cfg['velscale'], cfg['start'], cfg['mask0'],
        cfg['lam_gal'], cfg['lam_temp'],
        degree=cfg['kin_degree'], mdegree=cfg['kin_mdegree'],
        plot=False, quiet=True
    )

    pp_pop, pp_pop0 = fit_population_regularized(
        cfg['templates'], galaxy, pp_kin.noise_vector, cfg['velscale'], pp_kin.sol,
        pp_kin.clean_mask, cfg['lam_gal'], cfg['lam_temp'], cfg['reg_dim'],
        regul_start=cfg['regul_start'], regul_max=cfg['regul_max'],
        bracket_steps=cfg['reg_bracket_steps'], bisect_steps=cfg['reg_bisect_steps'],
        mdegree=cfg['pop_mdegree'], quiet=True
    )

    return {
        'jbin': jbin,
        'vel': float(pp_kin.sol[0]),
        'sig': float(pp_kin.sol[1]),
        'velerr': float(pp_kin.error_corr[0]) if pp_kin.error_corr is not None else np.nan,
        'sigerr': float(pp_kin.error_corr[1]) if pp_kin.error_corr is not None else np.nan,
        'sn': float(pp_kin.sn),
        'chi2': float(pp_pop.chi2),
        'regul': float(pp_pop.regul_used),
        'optimal_template': np.asarray(pp_kin.optimal_template, dtype=np.float32),
        'weights': np.asarray(pp_pop.weights, dtype=np.float32),
        'bestfit': np.asarray(pp_pop.bestfit, dtype=np.float32),
        'noise_vector': np.asarray(pp_pop.noise_vector, dtype=np.float32),
        'clean_mask': np.asarray(pp_kin.clean_mask, dtype=bool),
    }


def fit_bins_parallel(galaxy_bins, cfg, n_workers):
    tasks = [(j, galaxy_bins[:, j]) for j in range(galaxy_bins.shape[1])]

    if n_workers == 1:
        _worker_init(cfg)
        return [fit_one_bin_worker(task) for task in tasks]

    ctx = mp.get_context(MP_START_METHOD)
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_worker_init, initargs=(cfg,)) as ex:
        futures = {ex.submit(fit_one_bin_worker, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results[res['jbin']] = res
            print(f"Finished bin {res['jbin'] + 1}/{len(tasks)}")
    return results


# %% [markdown]
# ## Saving products

# %%
def save_ppxf_products(
    outdir,
    objfile,
    s,
    sps,
    sps_name,
    redshift,
    target_sn,
    bin_num,
    velbin,
    sigbin,
    velerr_bin,
    sigerr_bin,
    lg_age_bin,
    metalbin,
    lg_age_err_bin,
    metal_err_bin,
    regul_bin,
    optimal_templates,
    weights_bin,
    binned_spectra,
    binned_bestfit,
    sn_bin,
    chi2_bin,
    lam_gal,
    gas_output=None,
):
    """
    Save all useful pPXF products to FITS and NPZ.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = outdir / f"{objfile.stem}_ppxf_products_{sps_name}"

    ny = int(s.row.max())
    nx = int(s.col.max())

    bin_map = bin_num.reshape(ny, nx)
    vel_map = velbin[bin_num].reshape(ny, nx)
    sig_map = sigbin[bin_num].reshape(ny, nx)
    velerr_map = velerr_bin[bin_num].reshape(ny, nx)
    sigerr_map = sigerr_bin[bin_num].reshape(ny, nx)
    lg_age_map = lg_age_bin[bin_num].reshape(ny, nx)
    metal_map = metalbin[bin_num].reshape(ny, nx)
    vrmsbin = np.sqrt(velbin**2 + sigbin**2)
    vrms_map = vrmsbin[bin_num].reshape(ny, nx)
    signal_map = s.signal.reshape(ny, nx)
    noise_map = s.noise.reshape(ny, nx)

    nbins = len(velbin)
    bin_id = np.arange(nbins, dtype=np.int32)
    nspax_bin = np.bincount(bin_num, minlength=nbins).astype(np.int32)
    xbin = np.array([np.nanmean(s.x[bin_num == j]) for j in range(nbins)], dtype=float)
    ybin = np.array([np.nanmean(s.y[bin_num == j]) for j in range(nbins)], dtype=float)

    hdr = fits.Header()
    hdr["OBJECT"] = objfile.stem[:68]
    hdr["SPSMOD"] = sps_name
    hdr["NBINS"] = nbins
    hdr["VELSCAL"] = (float(s.velscale), "km/s per pixel")
    hdr["REDSHFT"] = float(redshift)
    hdr["TARSN"] = float(target_sn)
    hdr["LAMMIN"] = (float(np.min(lam_gal)), "Angstrom")
    hdr["LAMMAX"] = (float(np.max(lam_gal)), "Angstrom")
    hdr["PIXSIZE"] = (float(s.pixsize), "arcsec")
    hdr["COMMENT"] = "Refactored stellar population results from pPXF Power-bin fit"

    bin_cols = [
        fits.Column(name="BIN_ID", format="J", array=bin_id),
        fits.Column(name="X_ARCSEC", format="D", array=xbin),
        fits.Column(name="Y_ARCSEC", format="D", array=ybin),
        fits.Column(name="V_KMS", format="D", array=velbin),
        fits.Column(name="SIGMA_KMS", format="D", array=sigbin),
        fits.Column(name="VERR_KMS", format="D", array=velerr_bin),
        fits.Column(name="SIGERR_KMS", format="D", array=sigerr_bin),
        fits.Column(name="LOGAGE_YR", format="D", array=lg_age_bin),
        fits.Column(name="MEAN_METAL", format="D", array=metalbin),
        fits.Column(name="LOGAGE_ERR", format="D", array=lg_age_err_bin),
        fits.Column(name="METAL_ERR", format="D", array=metal_err_bin),
        fits.Column(name="REGUL", format="D", array=regul_bin),
        fits.Column(name="N_SPAX", format="J", array=nspax_bin),
        fits.Column(name="SN_BIN", format="D", array=sn_bin),
        fits.Column(name="CHI2", format="D", array=chi2_bin),
    ]
    bin_hdu = fits.BinTableHDU.from_columns(bin_cols, name="BIN_RESULTS")

    spax_cols = [
        fits.Column(name="ROW", format="J", array=s.row.astype(np.int32)),
        fits.Column(name="COL", format="J", array=s.col.astype(np.int32)),
        fits.Column(name="X_ARCSEC", format="D", array=s.x.astype(float)),
        fits.Column(name="Y_ARCSEC", format="D", array=s.y.astype(float)),
        fits.Column(name="SIGNAL", format="D", array=s.signal.astype(float)),
        fits.Column(name="NOISE", format="D", array=s.noise.astype(float)),
        fits.Column(name="BIN_ID", format="J", array=bin_num.astype(np.int32)),
        fits.Column(name="V_KMS", format="D", array=velbin[bin_num].astype(float)),
        fits.Column(name="SIGMA_KMS", format="D", array=sigbin[bin_num].astype(float)),
        fits.Column(name="VERR_KMS", format="D", array=velerr_bin[bin_num].astype(float)),
        fits.Column(name="SIGERR_KMS", format="D", array=sigerr_bin[bin_num].astype(float)),
        fits.Column(name="LOGAGE_YR", format="D", array=lg_age_bin[bin_num].astype(float)),
        fits.Column(name="MEAN_METAL", format="D", array=metalbin[bin_num].astype(float)),
    ]
    spax_hdu = fits.BinTableHDU.from_columns(spax_cols, name="SPAXELS")

    hdus = [
        fits.PrimaryHDU(header=hdr),
        bin_hdu,
        spax_hdu,
        fits.ImageHDU(data=bin_map.astype(np.int32), name="BIN_MAP"),
        fits.ImageHDU(data=vel_map.astype(np.float32), name="VEL_MAP"),
        fits.ImageHDU(data=sig_map.astype(np.float32), name="SIGMA_MAP"),
        fits.ImageHDU(data=velerr_map.astype(np.float32), name="VELERR_MAP"),
        fits.ImageHDU(data=sigerr_map.astype(np.float32), name="SIGERR_MAP"),
        fits.ImageHDU(data=lg_age_map.astype(np.float32), name="LOGAGE_MAP"),
        fits.ImageHDU(data=metal_map.astype(np.float32), name="METAL_MAP"),
        fits.ImageHDU(data=vrms_map.astype(np.float32), name="VRMS_MAP"),
        fits.ImageHDU(data=signal_map.astype(np.float32), name="SIGNAL_MAP"),
        fits.ImageHDU(data=noise_map.astype(np.float32), name="NOISE_MAP"),
        fits.ImageHDU(data=lam_gal.astype(np.float32), name="LAMBDA_GAL"),
        fits.ImageHDU(data=optimal_templates.astype(np.float32), name="OPT_TEMPL"),
        fits.ImageHDU(data=weights_bin.astype(np.float32), name="LIGHT_WGT"),
        fits.ImageHDU(data=binned_spectra.astype(np.float32), name="BIN_SPEC"),
        fits.ImageHDU(data=binned_bestfit.astype(np.float32), name="BIN_BESTFIT"),
        fits.ImageHDU(data=np.asarray(weights_bin.shape, dtype=np.int32), name="WGT_SHAPE"),
    ]

    if hasattr(sps, "age_grid"):
        hdus.append(fits.ImageHDU(data=np.asarray(sps.age_grid, dtype=np.float32), name="AGE_GRID"))
    if hasattr(sps, "metal_grid"):
        hdus.append(fits.ImageHDU(data=np.asarray(sps.metal_grid, dtype=np.float32), name="METAL_GRID"))
    if hasattr(sps, "alpha_grid"):
        hdus.append(fits.ImageHDU(data=np.asarray(sps.alpha_grid, dtype=np.float32), name="ALPHA_GRID"))

    save_dict = {
        "bin_id": bin_id,
        "xbin": xbin,
        "ybin": ybin,
        "velbin": velbin,
        "sigbin": sigbin,
        "velerr_bin": velerr_bin,
        "sigerr_bin": sigerr_bin,
        "lg_age_bin": lg_age_bin,
        "metalbin": metalbin,
        "lg_age_err_bin": lg_age_err_bin,
        "metal_err_bin": metal_err_bin,
        "regul_bin": regul_bin,
        "nspax_bin": nspax_bin,
        "sn_bin": sn_bin,
        "chi2_bin": chi2_bin,
        "bin_num": bin_num,
        "bin_map": bin_map,
        "vel_map": vel_map,
        "sig_map": sig_map,
        "velerr_map": velerr_map,
        "sigerr_map": sigerr_map,
        "lg_age_map": lg_age_map,
        "metal_map": metal_map,
        "vrmsbin": vrmsbin,
        "vrms_map": vrms_map,
        "signal_map": signal_map,
        "noise_map": noise_map,
        "x": s.x,
        "y": s.y,
        "row": s.row,
        "col": s.col,
        "signal": s.signal,
        "noise": s.noise,
        "lam_gal": lam_gal,
        "optimal_templates": optimal_templates,
        "weights_bin": weights_bin,
        "weights_shape": np.asarray(weights_bin.shape, dtype=np.int32),
        "binned_spectra": binned_spectra,
        "binned_bestfit": binned_bestfit,
    }

    if hasattr(sps, "age_grid"):
        save_dict["age_grid"] = np.asarray(sps.age_grid)
    if hasattr(sps, "metal_grid"):
        save_dict["metal_grid"] = np.asarray(sps.metal_grid)
    if hasattr(sps, "alpha_grid"):
        save_dict["alpha_grid"] = np.asarray(sps.alpha_grid)

    if gas_output is not None:
        pp_gas = gas_output["pp"]
        gas_names = np.asarray(gas_output["gas_names"])
        line_wave = np.asarray(gas_output["line_wave"])
        component = np.asarray(gas_output["component"])
        gas_component = np.asarray(gas_output["gas_component"])
        j_spax = int(gas_output["spaxel_index"])
        k_bin = int(gas_output["bin_index"])

        dlam = line_wave * s.velscale / C
        integrated_flux = pp_gas.gas_flux * dlam

        maxlen = max(len(name) for name in gas_names)
        gas_cols = [
            fits.Column(name="GAS_NAME", format=f"{maxlen}A", array=gas_names),
            fits.Column(name="LINE_WAVE", format="D", array=line_wave),
            fits.Column(name="COMPONENT", format="J", array=component.astype(np.int32)),
            fits.Column(name="IS_GAS", format="L", array=gas_component.astype(bool)),
            fits.Column(name="DLAM", format="D", array=dlam),
            fits.Column(name="GAS_FLUX_RAW", format="D", array=np.asarray(pp_gas.gas_flux, dtype=float)),
            fits.Column(name="GAS_FLUX_INT", format="D", array=np.asarray(integrated_flux, dtype=float)),
        ]

        if hasattr(pp_gas, "gas_flux_error_corr"):
            gas_cols.append(
                fits.Column(
                    name="GAS_FLUX_ERR",
                    format="D",
                    array=np.asarray(pp_gas.gas_flux_error_corr, dtype=float),
                )
            )

        hdus.append(fits.BinTableHDU.from_columns(gas_cols, name="GAS_LINES"))
        hdus.append(fits.ImageHDU(data=np.asarray(pp_gas.galaxy, dtype=np.float32), name="GAS_GALAXY"))
        hdus.append(fits.ImageHDU(data=np.asarray(pp_gas.bestfit, dtype=np.float32), name="GAS_BESTFIT"))

        if hasattr(pp_gas, "gas_bestfit"):
            hdus.append(fits.ImageHDU(data=np.asarray(pp_gas.gas_bestfit, dtype=np.float32), name="GAS_ONLYFIT"))
        if hasattr(pp_gas, "gas_bestfit_templates"):
            hdus.append(fits.ImageHDU(data=np.asarray(pp_gas.gas_bestfit_templates, dtype=np.float32), name="GAS_TMPLFIT"))
        if hasattr(pp_gas, "sol"):
            hdus.append(fits.ImageHDU(data=np.asarray(pp_gas.sol, dtype=np.float32), name="GAS_SOL"))
        if hasattr(pp_gas, "error") and pp_gas.error is not None:
            hdus.append(fits.ImageHDU(data=np.asarray(pp_gas.error, dtype=np.float32), name="GAS_ERR"))

        save_dict["gas_spaxel_index"] = j_spax
        save_dict["gas_bin_index"] = k_bin
        save_dict["gas_names"] = gas_names
        save_dict["gas_line_wave"] = line_wave
        save_dict["gas_component_labels"] = component
        save_dict["gas_component_mask"] = gas_component
        save_dict["gas_dlam"] = dlam
        save_dict["gas_flux_raw"] = np.asarray(pp_gas.gas_flux)
        save_dict["gas_flux_integrated"] = np.asarray(integrated_flux)
        if hasattr(pp_gas, "gas_flux_error_corr"):
            save_dict["gas_flux_error_corr"] = np.asarray(pp_gas.gas_flux_error_corr)
        save_dict["gas_galaxy"] = np.asarray(pp_gas.galaxy)
        save_dict["gas_bestfit"] = np.asarray(pp_gas.bestfit)
        if hasattr(pp_gas, "gas_bestfit"):
            save_dict["gas_onlyfit"] = np.asarray(pp_gas.gas_bestfit)
        if hasattr(pp_gas, "gas_bestfit_templates"):
            save_dict["gas_template_fits"] = np.asarray(pp_gas.gas_bestfit_templates)
        if hasattr(pp_gas, "sol"):
            save_dict["gas_sol"] = np.asarray(pp_gas.sol)
        if hasattr(pp_gas, "error") and pp_gas.error is not None:
            save_dict["gas_error"] = np.asarray(pp_gas.error)

    fits_path = base.with_suffix(".fits")
    npz_path = base.with_suffix(".npz")

    fits.HDUList(hdus).writeto(fits_path, overwrite=True)
    np.savez_compressed(npz_path, **save_dict)

    print(f"Saved FITS products to: {fits_path}")
    print(f"Saved NPZ products  to: {npz_path}")

    return fits_path, npz_path


# %% [markdown]
# ## Example gas fit on one spaxel

# %%
def fit_example_gas_spaxel(
    s,
    sps,
    optimal_templates,
    bin_num,
    velbin,
    sigbin,
    j,
    lam_gal,
    plots_path,
    ngas_comp=NGAS_COMP,
):
    """
    Example gas fit on one spaxel, keeping the stellar kinematics fixed to the
    value measured in the parent Power bin.
    """
    lam_range_gal = [np.min(lam_gal), np.max(lam_gal)]
    gas_templates, gas_names0, line_wave0 = util.emission_lines(
        sps.ln_lam_temp, lam_range_gal, s.fwhm_gal
    )

    gas_templates = np.tile(gas_templates, ngas_comp)
    gas_names = np.asarray([f"{a}_({p+1})" for p in range(ngas_comp) for a in gas_names0])
    line_wave = np.tile(line_wave0, ngas_comp)

    galaxy = s.spectra[:, j]
    noise = np.full_like(galaxy, estimate_spectrum_noise(galaxy))

    k = bin_num[j]
    template = optimal_templates[:, k]
    stars_gas_templates = np.column_stack([template, gas_templates])

    n_lines = len(gas_names0)
    component = [0] + [1] * n_lines + [2] * n_lines + [3] * n_lines
    gas_component = np.array(component) > 0
    moments = [-2, 2, 2, 2]

    tied = [["", ""] for _ in range(len(moments))]
    tied[2][1] = "p[3]"
    tied[3][0] = "(p[2] + p[4])/2"

    sig_diff = 200
    A_ineq = np.array([
        [0, 0, 0, 1, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 1, 0, -1],
    ])
    b_ineq = np.array([-sig_diff, -sig_diff]) / s.velscale
    constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

    start = [
        [velbin[k], sigbin[k]],
        [velbin[k], 50],
        [velbin[k], 50],
        [velbin[k], 500],
    ]

    vlim = lambda x: velbin[k] + x * np.array([-100, 100])
    bounds = [
        [vlim(2), [20, 300]],
        [vlim(2), [20, 100]],
        [vlim(6), [20, 100]],
        [vlim(2), [20, 1000]],
    ]

    plt.figure(figsize=(15, 8))
    pp = ppxf(
        stars_gas_templates, galaxy, noise, s.velscale, start,
        plot=1, moments=moments, degree=8, mdegree=-1,
        component=component, gas_component=gas_component, gas_names=gas_names,
        constr_kinem=constr_kinem, lam=lam_gal, lam_temp=sps.lam_temp,
        tied=tied, bounds=bounds, global_search=True
    )

    plt.savefig(plots_path / "ppxf_fit_gas_initial.png", dpi=300, bbox_inches="tight")

    noise *= np.sqrt(pp.chi2)

    plt.figure(figsize=(15, 8))
    pp = ppxf(
        stars_gas_templates, galaxy, noise, s.velscale, pp.sol,
        plot=1, moments=moments, degree=8, mdegree=-1,
        component=component, gas_component=gas_component, gas_names=gas_names,
        constr_kinem=constr_kinem, lam=lam_gal, lam_temp=sps.lam_temp,
        tied=tied, bounds=bounds, global_search=False
    )

    pp.error_corr = correct_ppxf_errors(pp)
    if hasattr(pp, "gas_flux_error"):
        pp.gas_flux_error_corr = pp.gas_flux_error * np.sqrt(pp.chi2)

    plt.savefig(plots_path / "ppxf_fit_gas_final.png", dpi=300, bbox_inches="tight")

    rms = robust_sigma(galaxy - pp.bestfit, zero=1)

    names = ["Halpha", "Hbeta", "[NII]6583_d", "[OIII]5007_d"]
    for name in names:
        kk = gas_names == name + "_(1)"
        if np.any(kk):
            dlam = line_wave[kk] * s.velscale / C
            flux = (pp.gas_flux[kk] * dlam)[0]
            an = np.max(pp.gas_bestfit_templates[:, kk]) / rms
            print(f"{name:12s} - Amplitude/Noise: {an:7.3f}; gas flux: {flux:10.3g}")

    w = np.array(["(1)" in name for name in gas_names])
    fig = plt.figure(figsize=(5, 7))
    lam_ranges = ([0.652, 0.662], [0.493, 0.5045])
    vel_stars = pp.sol[0][0]

    for jplot, lam_range in enumerate(lam_ranges):
        plt.subplot(len(lam_ranges), 1, jplot + 1)
        pp.plot()
        plt.xlim(lam_range)
        kk = (1e4 * lam_range[0] < pp.lam) & (pp.lam < 1e4 * lam_range[1])
        ymax = np.max(pp.bestfit[kk])
        plt.ylim(None, ymax * 1.1)
        for lam0, nam in zip(line_wave[w], gas_names[w]):
            lamz = lam0 * np.exp(vel_stars / C) / 1e4
            if lam_range[0] < lamz < lam_range[1]:
                plt.axvline(lamz, ls=":")
                plt.text(lamz + 1e-4, ymax, nam.split("_")[0])

    plt.tight_layout()
    plt.savefig(plots_path / "emission_lines.png", dpi=300, bbox_inches="tight")

    return {
        "pp": pp,
        "gas_names": gas_names,
        "line_wave": line_wave,
        "component": np.asarray(component),
        "gas_component": gas_component,
        "spaxel_index": j,
        "bin_index": k,
    }


# %% [markdown]
# ## Main script

# %%
def main():
    plots_path = ensure_dir(PLOTS_PATH)
    fit_plots_path = ensure_dir(plots_path / 'fit_checks')

    s = read_data_cube(OBJFILE, LAM_RANGE, REDSHIFT)

    def fun_capacity(index):
        return np.sum(s.signal[index]) / np.sqrt(np.sum(s.noise[index] ** 2))

    xy = np.column_stack([s.x, s.y])
    pow = PowerBin(xy, fun_capacity, TARGET_SN)
    bin_num = pow.bin_num

    pow.plot(ylabel='S/N')
    plt.savefig(plots_path / 'sn_bins.png', dpi=300, bbox_inches='tight')
    plt.close()

    bin_sn_capacity, xbin, ybin = plot_binning_diagnostics(s, bin_num, TARGET_SN, plots_path)

    ppxf_dir = resources.files('ppxf')
    basename = f"spectra_{SPS_NAME}_9.0.npz"
    filename = ppxf_dir / 'sps_models' / basename
    if not filename.is_file():
        url = 'https://raw.githubusercontent.com/micappe/ppxf_data/main/' + basename
        request.urlretrieve(url, filename)

    FWHM_gal = s.fwhm_gal
    sps = lib.sps_lib(filename, s.velscale, FWHM_gal, norm_range=SPS_NORM_RANGE)

    npix_temp, *reg_dim = sps.templates.shape
    sps.templates /= np.median(sps.templates)
    sps.templates = sps.templates.reshape(npix_temp, -1)

    lam_range_temp = np.exp(sps.ln_lam_temp[[0, -1]])
    mask0 = util.determine_mask(s.ln_lam_gal, lam_range_temp, width=1000)

    vel0 = 0.0
    start_guess = [vel0, 200.0]

    nbins = np.unique(bin_num).size
    lam_gal = np.exp(s.ln_lam_gal)
    ngalpix = len(lam_gal)

    galaxy_bins = np.empty((ngalpix, nbins), dtype=np.float32)
    for j in range(nbins):
        galaxy_bins[:, j] = np.nanmean(s.spectra[:, bin_num == j], axis=1)

    worker_cfg = {
        'templates': sps.templates,
        'velscale': s.velscale,
        'start': start_guess,
        'mask0': mask0,
        'lam_gal': lam_gal,
        'lam_temp': sps.lam_temp,
        'reg_dim': tuple(reg_dim),
        'kin_degree': KIN_DEGREE,
        'kin_mdegree': KIN_MDEGREE,
        'pop_mdegree': POP_MDEGREE,
        'regul_start': REGUL_START,
        'regul_max': REGUL_MAX,
        'reg_bracket_steps': REGUL_BRACKET_STEPS,
        'reg_bisect_steps': REGUL_BISECT_STEPS,
    }

    n_workers = N_WORKERS
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    n_workers = min(n_workers, nbins)
    print(f'Fitting {nbins} bins using {n_workers} worker(s)')

    results = fit_bins_parallel(galaxy_bins, worker_cfg, n_workers=n_workers)

    velbin = np.zeros(nbins)
    sigbin = np.zeros(nbins)
    velerr_bin = np.full(nbins, np.nan)
    sigerr_bin = np.full(nbins, np.nan)
    lg_age_bin = np.full(nbins, np.nan)
    metalbin = np.full(nbins, np.nan)
    lg_age_err_bin = np.full(nbins, np.nan)
    metal_err_bin = np.full(nbins, np.nan)
    sn_bin = np.full(nbins, np.nan)
    chi2_bin = np.full(nbins, np.nan)
    regul_bin = np.full(nbins, np.nan)
    optimal_templates = np.empty((npix_temp, nbins), dtype=np.float32)
    weights_bin = np.empty((*reg_dim, nbins), dtype=np.float32)
    binned_spectra = galaxy_bins.copy()
    binned_bestfit = np.empty((ngalpix, nbins), dtype=np.float32)

    for res in results:
        jbin = res['jbin']
        velbin[jbin] = res['vel']
        sigbin[jbin] = res['sig']
        velerr_bin[jbin] = res['velerr']
        sigerr_bin[jbin] = res['sigerr']
        sn_bin[jbin] = res['sn']
        chi2_bin[jbin] = res['chi2']
        regul_bin[jbin] = res['regul']
        optimal_templates[:, jbin] = res['optimal_template']
        binned_bestfit[:, jbin] = res['bestfit']

        light_weights = res['weights'].reshape(reg_dim)
        weights_bin[..., jbin] = light_weights
        lg_age_bin[jbin], metalbin[jbin] = sps.mean_age_metal(light_weights, quiet=True)

        if RUN_POP_MC:
            age_err, metal_err = monte_carlo_population_errors(
                sps.templates,
                model_spectrum=res['bestfit'],
                noise_vector=res['noise_vector'],
                velscale=s.velscale,
                kin_sol=[res['vel'], res['sig']],
                mask=res['clean_mask'],
                lam=lam_gal,
                lam_temp=sps.lam_temp,
                reg_dim=reg_dim,
                sps=sps,
                regul=res['regul'],
                n_mc=POP_MC_N,
                mdegree=POP_MDEGREE,
                seed=RNG_SEED + jbin,
            )
            lg_age_err_bin[jbin] = age_err
            metal_err_bin[jbin] = metal_err

        if SAVE_ALL_BIN_FITS or (jbin < N_PLOTS_BINS):
            title = (
                f'Bin {jbin + 1}/{nbins} | V={velbin[jbin]:.1f} km/s | '
                f'sigma={sigbin[jbin]:.1f} km/s | S/N={sn_bin[jbin]:.1f} | regul={regul_bin[jbin]:.1f}'
            )
            plot_bin_fit_summary(
                lam_gal, galaxy_bins[:, jbin], res['bestfit'], res['clean_mask'],
                res['noise_vector'], fit_plots_path / f'fit_bin_{jbin:03d}.png', title=title
            )

    plot_fit_maps(s, bin_num, velbin, sigbin, lg_age_bin, metalbin, plots_path)

    # Keep the original two-panel legacy plots too
    plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(121)
    display_bins(s.x, s.y, bin_num, velbin, colorbar=1, label='V (km/s)')
    add_isophotes(s)
    plt.subplot(122)
    display_bins(s.x, s.y, bin_num, lg_age_bin, colorbar=1, cmap='inferno', label='lg Age (yr)')
    add_isophotes(s)
    plt.savefig(plots_path / 'vel_age_maps.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    display_bins(s.x, s.y, bin_num, metalbin, colorbar=1, cmap='inferno', label='[M/H]')
    add_isophotes(s)
    plt.savefig(plots_path / 'metal_maps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    gas_output = None
    if FIT_GAS_EXAMPLE:
        gas_output = fit_example_gas_spaxel(
            s=s,
            sps=sps,
            optimal_templates=optimal_templates,
            bin_num=bin_num,
            velbin=velbin,
            sigbin=sigbin,
            j=GAS_SPIXEL_INDEX,
            lam_gal=lam_gal,
            plots_path=plots_path,
            ngas_comp=NGAS_COMP,
        )

    fits_path, npz_path = save_ppxf_products(
        outdir=plots_path,
        objfile=OBJFILE,
        s=s,
        sps=sps,
        sps_name=SPS_NAME,
        redshift=REDSHIFT,
        target_sn=TARGET_SN,
        bin_num=bin_num,
        velbin=velbin,
        sigbin=sigbin,
        velerr_bin=velerr_bin,
        sigerr_bin=sigerr_bin,
        lg_age_bin=lg_age_bin,
        metalbin=metalbin,
        lg_age_err_bin=lg_age_err_bin,
        metal_err_bin=metal_err_bin,
        regul_bin=regul_bin,
        optimal_templates=optimal_templates,
        weights_bin=weights_bin,
        binned_spectra=binned_spectra,
        binned_bestfit=binned_bestfit,
        sn_bin=sn_bin,
        chi2_bin=chi2_bin,
        lam_gal=lam_gal,
        gas_output=gas_output,
    )

    print(f'Saved outputs to {fits_path} and {npz_path}')


if __name__ == '__main__':
    main()
