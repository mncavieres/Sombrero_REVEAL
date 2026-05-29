
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

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from powerbin import PowerBin
from plotbin.display_bins import display_bins
from tqdm import tqdm

# %% [markdown]
# ## Configuration

# %%
C = 299792.458  # km/s

LAM_RANGE = [4750.0, 7409.0]   # rest-frame Angstrom
REDSHIFT = 0.003633
TARGET_SN = 100
SN_MIN = 0

SPS_NAME = "fsps"            # "fsps", "galaxev", "emiles", "xsl"
SPS_NORM_RANGE = [5070, 5950]

KIN_DEGREE = 8                 # additive polynomial degree for kinematics
KIN_MDEGREE = 0                # multiplicative polynomial degree for kinematics
POP_MDEGREE = 8                # multiplicative polynomial degree for populations

REGUL_START = 100.0
REGUL_MAX = 5.0e4
REGUL_BRACKET_STEPS = 10
REGUL_BISECT_STEPS = 10

N_PLOTS_BINS = 3

RUN_POP_MC = False             # optional; expensive
POP_MC_N = 20
RNG_SEED = 42

FIT_GAS_EXAMPLE = True
GAS_SPIXEL_INDEX = 682         # example spaxel
NGAS_COMP = 3

OBJFILE = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/MUSE/c30_cubes/c30_DATACUBE_normppxf_skycont_Part1_0000.fits"
)
PLOTS_PATH = Path(
    "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Plots/ppxf"
    "ppxf_c30_fsps"
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
plots_path = ensure_dir(PLOTS_PATH)
s = read_data_cube(OBJFILE, LAM_RANGE, REDSHIFT)

# %% [markdown]
# ### Power binning

# %%
def fun_capacity(index):
    return np.sum(s.signal[index]) / np.sqrt(np.sum(s.noise[index] ** 2))

xy = np.column_stack([s.x, s.y])
pow = PowerBin(xy, fun_capacity, TARGET_SN)
bin_num = pow.bin_num

pow.plot(ylabel="S/N")
plt.savefig(plots_path / "sn_bins.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### SPS templates

# %%
ppxf_dir = resources.files("ppxf")
basename = f"spectra_{SPS_NAME}_9.0.npz"
filename = ppxf_dir / "sps_models" / basename
if not filename.is_file():
    url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
    request.urlretrieve(url, filename)

# Match the template resolution to the galaxy when possible.
FWHM_gal = s.fwhm_gal
sps = lib.sps_lib(filename, s.velscale, FWHM_gal, norm_range=SPS_NORM_RANGE)

npix_temp, *reg_dim = sps.templates.shape
sps.templates /= np.median(sps.templates)
sps.templates = sps.templates.reshape(npix_temp, -1)

# %% [markdown]
# ### Stellar fits

# %%
lam_range_temp = np.exp(sps.ln_lam_temp[[0, -1]])
mask0 = util.determine_mask(s.ln_lam_gal, lam_range_temp, width=1000)

vel0 = 0.0
start = [vel0, 200.0]

nbins = np.unique(bin_num).size
lam_gal = np.exp(s.ln_lam_gal)
ngalpix = len(lam_gal)

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

optimal_templates = np.empty((npix_temp, nbins))
weights_bin = np.empty((*reg_dim, nbins))
binned_spectra = np.empty((ngalpix, nbins))
binned_bestfit = np.empty((ngalpix, nbins))

for jbin in tqdm(range(nbins)):
    plot = jbin < N_PLOTS_BINS
    w = bin_num == jbin
    galaxy = np.nanmean(s.spectra[:, w], axis=1)

    pp_kin = fit_stellar_kinematics(
        sps.templates, galaxy, s.velscale, start, mask0,
        lam_gal, sps.lam_temp,
        degree=KIN_DEGREE, mdegree=KIN_MDEGREE,
        plot=plot, quiet=not plot
    )

    velbin[jbin], sigbin[jbin] = pp_kin.sol[:2]
    if pp_kin.error_corr is not None:
        velerr_bin[jbin], sigerr_bin[jbin] = pp_kin.error_corr[:2]

    sn_bin[jbin] = pp_kin.sn
    optimal_templates[:, jbin] = pp_kin.optimal_template

    pp_pop, pp_pop0 = fit_population_regularized(
        sps.templates, galaxy, pp_kin.noise_vector, s.velscale, pp_kin.sol,
        pp_kin.clean_mask, lam_gal, sps.lam_temp, reg_dim,
        regul_start=REGUL_START, regul_max=REGUL_MAX,
        bracket_steps=REGUL_BRACKET_STEPS, bisect_steps=REGUL_BISECT_STEPS,
        mdegree=POP_MDEGREE, quiet=not plot
    )

    light_weights = pp_pop.weights.reshape(reg_dim)
    weights_bin[..., jbin] = light_weights
    lg_age_bin[jbin], metalbin[jbin] = sps.mean_age_metal(light_weights, quiet=not plot)

    if RUN_POP_MC:
        age_err, metal_err = monte_carlo_population_errors(
            sps.templates,
            model_spectrum=pp_pop.bestfit,
            noise_vector=pp_pop.noise_vector,
            velscale=s.velscale,
            kin_sol=pp_kin.sol,
            mask=pp_kin.clean_mask,
            lam=lam_gal,
            lam_temp=sps.lam_temp,
            reg_dim=reg_dim,
            sps=sps,
            regul=pp_pop.regul_used,
            n_mc=POP_MC_N,
            mdegree=POP_MDEGREE,
            seed=RNG_SEED + jbin,
        )
        lg_age_err_bin[jbin] = age_err
        metal_err_bin[jbin] = metal_err

    chi2_bin[jbin] = pp_pop.chi2
    regul_bin[jbin] = pp_pop.regul_used
    binned_spectra[:, jbin] = galaxy
    binned_bestfit[:, jbin] = pp_pop.bestfit

    if plot:
        plt.figure(figsize=(16, 4))
        pp_pop.plot()
        plt.title(
            f"Power bin {jbin + 1}/{nbins} | "
            f"sigma={sigbin[jbin]:.1f}±{sigerr_bin[jbin]:.1f} km/s | "
            f"S/N={sn_bin[jbin]:.1f} | regul={regul_bin[jbin]:.2f}"
        )
        plt.tight_layout()
        plt.savefig(plots_path / f"population_fit_bin_{jbin:03d}.png", dpi=300, bbox_inches="tight")

        print(
            f"bin {jbin + 1:3d}/{nbins:3d}  "
            f"V={velbin[jbin]:8.2f}±{velerr_bin[jbin]:6.2f}  "
            f"sigma={sigbin[jbin]:7.2f}±{sigerr_bin[jbin]:6.2f}  "
            f"logAge={lg_age_bin[jbin]:7.3f}  [M/H]={metalbin[jbin]:7.3f}  "
            f"regul={regul_bin[jbin]:8.2f}"
        )

# %% [markdown]
# ### Maps

# %%
plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(wspace=0.5)

plt.subplot(121)
display_bins(s.x, s.y, bin_num, velbin, colorbar=1, label="V (km/s)")
plt.tricontour(
    s.x, s.y,
    -2.5 * np.log10(s.signal / np.nanmax(s.signal)),
    levels=np.arange(20)
)

plt.subplot(122)
display_bins(s.x, s.y, bin_num, lg_age_bin, colorbar=1, cmap="inferno", label="lg Age (yr)")
plt.tricontour(
    s.x, s.y,
    -2.5 * np.log10(s.signal / np.nanmax(s.signal)),
    levels=np.arange(20)
)

plt.savefig(plots_path / "vel_age_maps.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 7))
display_bins(s.x, s.y, bin_num, metalbin, colorbar=1, cmap="inferno", label="[M/H]")
plt.tricontour(
    s.x, s.y,
    -2.5 * np.log10(s.signal / np.nanmax(s.signal)),
    levels=np.arange(20)
)
plt.savefig(plots_path / "metal_maps.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### Optional example gas fit

# %%
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

# %% [markdown]
# ### Save products

# %%
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
