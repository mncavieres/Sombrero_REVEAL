import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mgefit as mge
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter



ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)
PC_TO_M = 3.085677581491367e16
JY_TO_W_M2_HZ = 1e-26


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _capture_new_figures(fn):
    """Run fn() and return (fn_return_value, list_of_new_matplotlib_figure_numbers)."""
    before = set(plt.get_fignums())
    out = fn()
    after = set(plt.get_fignums())
    new = sorted(list(after - before))
    return out, new


def _save_and_close_figs(fig_nums, outpaths, dpi=200):
    for fignum, path in zip(fig_nums, outpaths):
        fig = plt.figure(fignum)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)


def _stretch_for_display(img, goodmask=None):
    """Robust asinh stretch for display."""
    x = np.asarray(img, float)
    if goodmask is None:
        goodmask = np.isfinite(x)
    v = x[goodmask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return x
    lo, hi = np.percentile(v, [5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return x
    z = np.clip(x, lo, hi)
    z = np.arcsinh((z - lo) / (hi - lo + 1e-30))
    return z

def _native_like_mge_levels(image, goodmask, magstep=0.5, minlevel=None, nlevels=None):
    """
    Build contour levels similar in spirit to mge_print_contours:
    logarithmic spacing in intensity, corresponding to equal steps in magnitudes.

    Parameters
    ----------
    image : 2D array
    goodmask : 2D bool array
    magstep : float
        Step in magnitudes between contours.
    minlevel : float or None
        Absolute minimum intensity level to include.
    nlevels : int or None
        Optional maximum number of contour levels.
    """
    valid = goodmask & np.isfinite(image) & (image > 0)
    v = image[valid]
    if v.size == 0:
        return None

    peak = np.nanmax(v)
    if not np.isfinite(peak) or peak <= 0:
        return None

    if minlevel is None:
        # sensible fallback: stop before the noisy floor
        minlevel = np.nanpercentile(v, 70)

    if (not np.isfinite(minlevel)) or (minlevel <= 0) or (minlevel >= peak):
        minlevel = np.nanpercentile(v, 70)

    # number of equal-mag contours from peak down to minlevel
    max_decades = np.log10(peak / minlevel)
    max_mag = max_decades / 0.4
    n = int(np.floor(max_mag / magstep)) + 1
    n = max(n, 3)

    if nlevels is not None:
        n = min(n, int(nlevels))

    k = np.arange(n)
    levels = peak * 10.0**(-0.4 * magstep * k)

    levels = levels[(levels >= minlevel) & np.isfinite(levels)]
    levels = np.unique(np.sort(levels))

    if levels.size < 3:
        return None

    return levels

def mge_model_counts_at_polar_points(radius_pix, angle_deg, sol):
    """
    Evaluate the surface-brightness model (counts/pix) at polar points
    (radius, angle), where angle is measured from the major axis.
    """
    total_counts, sigma, q = sol
    total_counts = np.asarray(total_counts, float)
    sigma = np.asarray(sigma, float)
    q = np.asarray(q, float)

    surf = total_counts / (2.0 * np.pi * sigma**2 * q)

    th = np.deg2rad(np.asarray(angle_deg, float))
    r = np.asarray(radius_pix, float)
    xp = r * np.cos(th)
    yp = r * np.sin(th)

    xx = xp[:, None]
    yy = yp[:, None]
    sig2 = sigma[None, :] ** 2
    q2 = q[None, :] ** 2

    expo = -0.5 * (xx**2 + (yy**2) / q2) / sig2
    return np.sum(surf[None, :] * np.exp(expo), axis=1)


def polar_points_to_image_xy(radius_pix, angle_deg, pa_deg, center_pix_xy):
    """
    Convert (radius, angle from major axis) -> image (x,y) pixels for overlay.

    Public convention:
        center_pix_xy = (xc, yc) = (x, y) = (col, row)

    Returns x=col, y=row for imshow(origin='lower').
    """
    xc, yc = center_pix_xy
    xc = float(xc)
    yc = float(yc)

    r = np.asarray(radius_pix, float)
    th = np.deg2rad(np.asarray(angle_deg, float))
    xp = r * np.cos(th)
    yp = r * np.sin(th)

    pa = np.deg2rad(pa_deg)

    x_img = xc + xp * np.sin(pa) - yp * np.cos(pa)
    y_img = yc + xp * np.cos(pa) + yp * np.sin(pa)
    return x_img, y_img


def build_mge_model_image_cutout(img_shape, sol, pa_deg, center_pix_xy,
                                 half_size_pix=400, oversample=1):
    """
    Build a model image cutout (counts/pix) from the MGE solution.

    Public convention:
        center_pix_xy = (xc, yc) = (x, y) = (col, row)
    """
    ny, nx = img_shape
    xc, yc = center_pix_xy
    xc = float(xc)
    yc = float(yc)

    x1 = int(max(0, np.floor(xc - half_size_pix)))
    x2 = int(min(nx, np.ceil(xc + half_size_pix)))
    y1 = int(max(0, np.floor(yc - half_size_pix)))
    y2 = int(min(ny, np.ceil(yc + half_size_pix)))

    xs = np.arange(x1, x2, 1 / oversample)
    ys = np.arange(y1, y2, 1 / oversample)
    X, Y = np.meshgrid(xs, ys)

    dx = X - xc
    dy = Y - yc
    pa = np.deg2rad(pa_deg)

    xp = dx * np.sin(pa) + dy * np.cos(pa)
    yp = -dx * np.cos(pa) + dy * np.sin(pa)

    total_counts, sigma, q = sol
    total_counts = np.asarray(total_counts, float)
    sigma = np.asarray(sigma, float)
    q = np.asarray(q, float)
    surf = total_counts / (2.0 * np.pi * sigma**2 * q)

    model = np.zeros_like(X, dtype=float)
    for s0, sj, qj in zip(surf, sigma, q):
        expo = -0.5 * (xp**2 + (yp**2) / (qj * qj)) / (sj * sj)
        model += s0 * np.exp(expo)

    if oversample > 1:
        oy = (y2 - y1) * oversample
        ox = (x2 - x1) * oversample
        model = model[:oy, :ox]
        model = model.reshape((y2 - y1), oversample, (x2 - x1), oversample).mean(axis=(1, 3))

    return (x1, x2, y1, y2), model
@dataclass
class AxisymmetricDeprojectedMGE:
    inclination_deg: float
    qobs: np.ndarray
    qintr: np.ndarray
    sigma_pix: np.ndarray
    sigma_arcsec: np.ndarray
    sigma_pc: Optional[np.ndarray]
    total_native_pix2: np.ndarray
    total_native_arcsec2: np.ndarray
    total_flux_jy: Optional[np.ndarray]
    total_lnu_W_hz: Optional[np.ndarray]
    rho0_native_per_arcsec: np.ndarray
    rho0_W_hz_pc3: Optional[np.ndarray]
    native_density_unit: str
    physical_density_unit: Optional[str]
    sb_unit: Optional[str]

    def density(self, R, z, physical=False):
        """
        Evaluate the intrinsic axisymmetric density.

        Parameters
        ----------
        R, z : float or array_like
            Cylindrical intrinsic coordinates.
            - physical=False  -> R,z in arcsec
            - physical=True   -> R,z in pc

        physical : bool
            If True, return physical emissivity in W Hz^-1 pc^-3.
            Otherwise return angular density in native_SB_unit / arcsec.
        """
        R_arr, z_arr = np.broadcast_arrays(np.asarray(R, dtype=float), np.asarray(z, dtype=float))

        if physical:
            if self.sigma_pc is None or self.rho0_W_hz_pc3 is None:
                raise ValueError(
                    "Physical density requested, but this object was not built "
                    "with both distance and sb_unit."
                )
            sig = self.sigma_pc
            rho0 = self.rho0_W_hz_pc3
        else:
            sig = self.sigma_arcsec
            rho0 = self.rho0_native_per_arcsec

        expo = -0.5 * (
            (R_arr[..., None] / sig) ** 2 +
            (z_arr[..., None] / (sig * self.qintr)) ** 2
        )
        return np.sum(rho0 * np.exp(expo), axis=-1)

    __call__ = density

    def grid(self, half_size, npix=400, physical=False):
        x = np.linspace(-half_size, half_size, int(npix))
        z = np.linspace(-half_size, half_size, int(npix))
        xx, zz = np.meshgrid(x, z)
        rr = np.abs(xx)
        rho = self.density(rr, zz, physical=physical)
        return x, z, rho

    def _plot_central_los_aperture(self, result, checkplot_dir, prefix="mge", dpi=300):
        if checkplot_dir is None:
            raise ValueError("checkplot_dir must be provided to save LOS aperture plots.")

        _ensure_dir(checkplot_dir)

        length_unit = result["length_unit"]
        profile_unit = result["profile_unit"]
        frac = result["frac"]

        s_full = result["s"]
        dMds_full = result["dMds"]
        S = result["S"]
        cdf = result["cdf"]
        sfrac = result["s_frac"]
        s50 = result["s50"]
        s90 = result["s90"]
        s95 = result["s95"]
        aperture = result["aperture"]

        # LOS profile
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(s_full, dMds_full, linewidth=1.5)
        ax.axvline(-sfrac, linestyle="--", linewidth=1.0, alpha=0.9)
        ax.axvline(+sfrac, linestyle="--", linewidth=1.0, alpha=0.9)
        ax.set_xlabel(f"LOS distance from center [{length_unit}]")
        ax.set_ylabel(f"dM/ds [{profile_unit}]")
        ax.set_title(
            f"Central LOS profile in circular aperture = {aperture:.3f} {length_unit}\n"
            f"{100.0*frac:.1f}% enclosed half-length = ±{sfrac:.3f} {length_unit}"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            os.path.join(checkplot_dir, f"{prefix}_los_aperture_profile.png"),
            dpi=dpi
        )
        plt.close(fig)

        # CDF
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(S, cdf, linewidth=1.5)
        ax.axhline(frac, linestyle="--", linewidth=1.0, alpha=0.9)
        ax.axvline(sfrac, linestyle="--", linewidth=1.0, alpha=0.9)
        ax.axvline(s50, linestyle=":", linewidth=1.0, alpha=0.8)
        ax.axvline(s90, linestyle=":", linewidth=1.0, alpha=0.8)
        ax.axvline(s95, linestyle=":", linewidth=1.0, alpha=0.8)
        ax.set_xlabel(f"Enclosed LOS half-length |s| [{length_unit}]")
        ax.set_ylabel("Enclosed LOS mass fraction")
        ax.set_title(f"Central LOS enclosed fraction in aperture = {aperture:.3f} {length_unit}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            os.path.join(checkplot_dir, f"{prefix}_los_aperture_cdf.png"),
            dpi=dpi
        )
        plt.close(fig)

    def central_los_aperture(
        self,
        aperture,
        frac=0.90,
        n_s=2000,
        n_y=401,
        max_s=None,
        physical=False,
        ml=1.0,
        checkplot_dir=None,
        prefix="mge",
        dpi=300,
        save_plots=False,
    ):
        """
        Compute the LOS profile through the galaxy center, integrated over a
        circular projected aperture.

        Parameters
        ----------
        aperture : float
            Aperture radius.
            - physical=False -> arcsec
            - physical=True  -> pc

        frac : float, optional
            Enclosed LOS fraction to report. Default 0.90.

        n_s : int, optional
            Number of LOS grid points on s >= 0.

        n_y : int, optional
            Number of integration points along projected y inside the aperture.

        max_s : float, optional
            Maximum LOS half-range. If None, chosen automatically.

        physical : bool, optional
            If True, compute in physical units using sigma_pc and total_lnu_W_hz.
            Otherwise compute in angular/native units using sigma_arcsec and
            total_native_arcsec2.

        ml : float, optional
            Multiplicative mass-to-light scaling. This changes normalization,
            not the enclosed fractions or the percentile radius.

        checkplot_dir : str, optional
            Directory where the LOS profile and CDF plots are saved.

        prefix : str, optional
            Prefix for saved plot filenames.

        dpi : int, optional
            DPI for saved plots.

        save_plots : bool, optional
            If True, save the LOS profile and CDF plots.

        Returns
        -------
        result : dict
            Contains the sampled LOS coordinates, LOS profile, cumulative fraction,
            and the percentile radii.

            Generic keys:
                s, dMds, mean_density_in_aperture, ds, mass_per_bin
                S, dMds_pos, mean_density_in_aperture_pos, cdf
                s_frac, s50, s90, s95
                aperture, qintr, sigma, inclination_deg, frac
                length_unit, profile_unit, mean_density_unit

            Also includes unit-specific aliases:
                if physical=False:
                    s_arcsec, S_arcsec, ds_arcsec,
                    s_frac_arcsec, s50_arcsec, s90_arcsec, s95_arcsec
                if physical=True:
                    s_pc, S_pc, ds_pc,
                    s_frac_pc, s50_pc, s90_pc, s95_pc
        """
        try:
            from scipy.special import erf
        except Exception:
            from math import erf as _erf
            erf = np.vectorize(_erf)

        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        aperture = float(aperture)
        frac = float(frac)

        if aperture <= 0:
            raise ValueError("aperture must be > 0")
        if not (0.0 < frac < 1.0):
            raise ValueError("frac must satisfy 0 < frac < 1")
        if int(n_s) < 2:
            raise ValueError("n_s must be >= 2")
        if int(n_y) < 2:
            raise ValueError("n_y must be >= 2")

        inc = np.deg2rad(float(self.inclination_deg))
        sini = np.sin(inc)
        cosi = np.cos(inc)

        if np.isclose(sini, 0.0):
            raise ValueError("Face-on deprojection is non-unique; inclination must be > 0.")

        qintr = np.asarray(self.qintr, dtype=float)
        qobs = np.asarray(self.qobs, dtype=float)

        if np.any(qintr <= 0):
            raise ValueError(
                "At least one Gaussian has non-positive intrinsic flattening."
            )

        if physical:
            if self.sigma_pc is None or self.total_lnu_W_hz is None:
                raise ValueError(
                    "physical=True requires a deprojection built with both distance and sb_unit."
                )
            sigma = np.asarray(self.sigma_pc, dtype=float)
            L = np.asarray(self.total_lnu_W_hz, dtype=float) * float(ml)
            length_unit = "pc"
            profile_unit = "W Hz^-1 pc^-1"
            mean_density_unit = "W Hz^-1 pc^-3"
        else:
            sigma = np.asarray(self.sigma_arcsec, dtype=float)
            L = np.asarray(self.total_native_arcsec2, dtype=float) * float(ml)
            length_unit = "arcsec"
            profile_unit = "native_integrated_unit_per_length"
            mean_density_unit = self.native_density_unit

        a_los = sigma * qintr / qobs

        if max_s is None:
            max_s = 8.0 * np.max(a_los) + aperture
        max_s = float(max_s)

        if max_s <= 0:
            raise ValueError("max_s must be > 0")

        S = np.linspace(0.0, max_s, int(n_s))
        y = np.linspace(-aperture, aperture, int(n_y))

        y2 = y**2
        xmax = np.sqrt(np.clip(aperture**2 - y2, 0.0, None))

        dMds_pos = np.zeros_like(S)

        for Lj, sigj, qj, qoj in zip(L, sigma, qintr, qobs):
            amp = Lj / (((2.0 * np.pi) ** 1.5) * sigj**3 * qj)

            A = cosi**2 + (sini**2) / (qj**2)
            B = sini * cosi * (1.0 / (qj**2) - 1.0)
            C = sini**2 + (cosi**2) / (qj**2)

            xint = np.sqrt(2.0 * np.pi) * sigj * erf(xmax / (np.sqrt(2.0) * sigj))

            expo = -(
                A * y[:, None] ** 2
                + 2.0 * B * y[:, None] * S[None, :]
                + C * S[None, :] ** 2
            ) / (2.0 * sigj**2)

            integrand_y = amp * np.exp(expo) * xint[:, None]
            dMds_pos += trapz(integrand_y, y, axis=0)

        s_full = np.concatenate((-S[:0:-1], S))
        dMds_full = np.concatenate((dMds_pos[:0:-1], dMds_pos))

        ds = S[1] - S[0]
        area_ap = np.pi * aperture**2
        mean_density_in_aperture_full = dMds_full / area_ap
        mean_density_in_aperture_pos = dMds_pos / area_ap
        mass_per_bin = dMds_full * ds

        cum_pos = np.zeros_like(S)
        if len(S) > 1:
            shell_mass = 0.5 * (dMds_pos[:-1] + dMds_pos[1:]) * np.diff(S)
            cum_pos[1:] = 2.0 * np.cumsum(shell_mass)

        total_mass_los = cum_pos[-1]
        if not np.isfinite(total_mass_los) or total_mass_los <= 0:
            raise RuntimeError("Computed non-positive total LOS mass/profile integral.")

        cdf = cum_pos / total_mass_los

        def radius_at_fraction(f):
            return np.interp(float(f), cdf, S)

        s50 = radius_at_fraction(0.50)
        s90 = radius_at_fraction(0.90)
        s95 = radius_at_fraction(0.95)
        sfrac = radius_at_fraction(frac)

        result = {
            "s": s_full,
            "dMds": dMds_full,
            "mean_density_in_aperture": mean_density_in_aperture_full,
            "ds": ds,
            "mass_per_bin": mass_per_bin,
            "S": S,
            "dMds_pos": dMds_pos,
            "mean_density_in_aperture_pos": mean_density_in_aperture_pos,
            "cdf": cdf,
            "s_frac": sfrac,
            "s50": s50,
            "s90": s90,
            "s95": s95,
            "aperture": aperture,
            "qintr": qintr,
            "qobs": qobs,
            "sigma": sigma,
            "inclination_deg": float(self.inclination_deg),
            "frac": frac,
            "length_unit": length_unit,
            "profile_unit": profile_unit,
            "mean_density_unit": mean_density_unit,
            "normalization_scale_ml": float(ml),
        }

        if physical:
            result.update({
                "s_pc": s_full,
                "S_pc": S,
                "ds_pc": ds,
                "s_frac_pc": sfrac,
                "s50_pc": s50,
                "s90_pc": s90,
                "s95_pc": s95,
                "aperture_pc": aperture,
                "sigma_pc": sigma,
            })
        else:
            result.update({
                "s_arcsec": s_full,
                "S_arcsec": S,
                "ds_arcsec": ds,
                "s_frac_arcsec": sfrac,
                "s50_arcsec": s50,
                "s90_arcsec": s90,
                "s95_arcsec": s95,
                "aperture_arcsec": aperture,
                "sigma_arcsec": sigma,
            })

        if save_plots:
            self._plot_central_los_aperture(
                result=result,
                checkplot_dir=checkplot_dir,
                prefix=prefix,
                dpi=dpi,
            )

        return result

#########################################################
class MGEFitter:
    """
    Public convention used everywhere in this wrapper:
        center = (x, y) = (col, row)

    NumPy indexing:
        img[y, x]

    mgefit convention is adapted internally at the call boundary.
    """

    def __init__(
        self,
        img_f200,
        dust_mask,
        *,
        pixel_scale=1.0,
        dust_mask_is_bad=True,
        subtract_sky=False,
        find_fraction=0.03,
        find_binning=5,
        n_sectors=19,
        minlevel=0,
        sigmapsf=0.0,
        normpsf=1.0,
        linear=True,
        ngauss=900,
        qbounds=None,
        outer_slope=4,
        plot=True,
        quiet=False,
        checkplot_dir=None,
        cache_dir=None,
        prefix="f200",
        dpi=300,
        max_points_overlay=200000,
        contour_half_size_arcsec=80,
        contour_oversample=1,
        center=None,          # public convention: (x, y)
        pa_deg=None,
        eps=None,
        theta_deg=None,
        allow_negative=False,
        bulge_disk=False,
    ):
        self.img = np.asarray(img_f200, dtype=float)
        if self.img.ndim != 2:
            raise ValueError("img_f200 must be a 2D array")

        self.mask = np.asarray(dust_mask)
        if self.mask.shape != self.img.shape:
            raise ValueError("dust_mask must have same shape as img_f200")
        if self.mask.dtype != bool:
            self.mask = self.mask.astype(bool)

        self.pixel_scale = pixel_scale
        self.dust_mask_is_bad = dust_mask_is_bad
        self.subtract_sky = subtract_sky
        self.find_fraction = find_fraction
        self.find_binning = find_binning
        self.n_sectors = n_sectors
        self.minlevel = minlevel
        self.sigmapsf = sigmapsf
        self.normpsf = normpsf
        self.linear = linear
        self.ngauss = ngauss
        self.qbounds = qbounds
        self.outer_slope = outer_slope
        self.plot = plot
        self.quiet = quiet
        self.checkplot_dir = checkplot_dir
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.dpi = dpi
        self.max_points_overlay = max_points_overlay
        self.contour_half_size_arcsec = contour_half_size_arcsec
        self.contour_oversample = contour_oversample
        self.allow_negative = allow_negative
        self.bulge_disk = bulge_disk

        if self.checkplot_dir is not None:
            _ensure_dir(self.checkplot_dir)
        if self.cache_dir is not None:
            _ensure_dir(self.cache_dir)

        self.goodmask = (~self.mask) if self.dust_mask_is_bad else self.mask

        self.img_for_find = self.img.copy()
        fill = np.nanmedian(self.img[self.goodmask]) if np.any(self.goodmask) else np.nanmedian(self.img)
        self.img_for_find[~self.goodmask] = fill
        if not np.all(np.isfinite(self.img_for_find)):
            raise ValueError("Image contains NaN/Inf even after filling masked pixels")

        self.sky_mean = None
        self.sky_sigma = None
        self.img_work = self.img.copy()

        if self.subtract_sky:
            self.sky_mean, self.sky_sigma = mge.sky_level(
                self.img_for_find[self.goodmask], plot=self.plot, quiet=self.quiet
            )
            self.img_work = self.img_work - self.sky_mean
            self.img_for_find = self.img_for_find - self.sky_mean

            # Use half the sky RMS as the minimum photometry flux threshold.
            self.minlevel = self.sky_sigma / 2
            print(f"Sky mean: {self.sky_mean:.3f}  Sky sigma: {self.sky_sigma:.3f}  Setting minlevel to {self.minlevel:.3f}")

        self.find_result = None
        self.sectors_result = None
        self.fit_result = None

        self._deprojection_config = {
            "inclination_deg": None,
            "distance": None,
            "sb_unit": None,
        }
        self._deprojected_mge_cache = None

        self._manual_center = None    # always public (x, y)
        self._manual_pa = None
        self._manual_eps = None
        self._manual_theta = None

        if center is not None or pa_deg is not None or eps is not None or theta_deg is not None:
            self.set_manual_geometry(center=center, pa_deg=pa_deg, eps=eps, theta_deg=theta_deg)

    # helpers for geometry conventions because mge is a nightmare
    @staticmethod
    def _find_result_to_public_xy(find_result):
        """
        Convert find_galaxy output to public convention (x, y)=(col, row).
        """
        return float(find_result.ypeak), float(find_result.xpeak)

    @staticmethod
    def _public_xy_to_mge_xy(center_xy):
        """
        Convert public (x, y)=(col, row) to the convention expected by
        mge.sectors_photometry, which follows mgefit's internal indexing.
        """
        x_public, y_public = center_xy
        mge_xc = float(y_public)   # first index / row
        mge_yc = float(x_public)   # second index / column
        return mge_xc, mge_yc

    # set manual geometry, much better than the mgefind galaxy shit
    def set_manual_geometry(self, *, center=None, pa_deg=None, eps=None, theta_deg=None):
        if center is not None:
            if len(center) != 2:
                raise ValueError("center must be a 2-element tuple/list (x, y)")
            self._manual_center = (float(center[0]), float(center[1]))

        if pa_deg is not None:
            self._manual_pa = float(pa_deg)

        if eps is not None:
            self._manual_eps = float(eps)

        if theta_deg is not None:
            self._manual_theta = float(theta_deg)

    def clear_manual_geometry(self):
        self._manual_center = None
        self._manual_pa = None
        self._manual_eps = None
        self._manual_theta = None

    def has_manual_geometry(self):
        return (
            self._manual_center is not None and
            self._manual_pa is not None and
            self._manual_eps is not None
        )

    def geometry_is_available(self):
        return self.has_manual_geometry() or (self.find_result is not None)

    # -------------------------
    # cache helpers
    # -------------------------
    def _cache_path(self, stage):
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"{self.prefix}_{stage}.pkl")

    def _save_cache(self, stage, payload):
        path = self._cache_path(stage)
        if path is None:
            return
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def _load_cache(self, stage):
        path = self._cache_path(stage)
        if path is None or (not os.path.exists(path)):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    # -------------------------
    # geometry properties
    # -------------------------
    @property
    def xc(self):
        if self._manual_center is not None:
            return self._manual_center[0]
        if self.find_result is None:
            return None
        return self._find_result_to_public_xy(self.find_result)[0]

    @property
    def yc(self):
        if self._manual_center is not None:
            return self._manual_center[1]
        if self.find_result is None:
            return None
        return self._find_result_to_public_xy(self.find_result)[1]

    @property
    def pa(self):
        if self._manual_pa is not None:
            return self._manual_pa
        if self.find_result is None:
            return None
        return float(self.find_result.pa)

    @property
    def eps(self):
        if self._manual_eps is not None:
            return self._manual_eps
        if self.find_result is None:
            return None
        return float(self.find_result.eps)

    @property
    def theta(self):
        if self._manual_theta is not None:
            return self._manual_theta
        if self._manual_pa is not None:
            return self._manual_pa
        if self.find_result is None:
            return None
        return float(self.find_result.theta)

    # deprojection helpers for the future
    def set_deprojection(self, inclination_deg, distance=None, sb_unit=None):
        """
        Store the axisymmetric deprojection configuration.

        Parameters
        ----------
        inclination_deg : float
            Galaxy inclination in degrees. 90 = edge-on.
        distance : float or astropy quantity, optional
            If float, assumed to be in Mpc.
            If astropy quantity, converted to pc.
        sb_unit : str, optional
            Surface-brightness unit of the input image.
            Supported examples:
                'MJy/sr', 'Jy/sr', 'mJy/sr', 'uJy/sr',
                'MJy/arcsec^2', 'Jy/arcsec^2', 'mJy/arcsec^2', 'uJy/arcsec^2'
        """
        self._deprojection_config = {
            "inclination_deg": float(inclination_deg),
            "distance": distance,
            "sb_unit": sb_unit,
        }
        self._deprojected_mge_cache = None

    def clear_deprojection(self):
        self._deprojection_config = {
            "inclination_deg": None,
            "distance": None,
            "sb_unit": None,
        }
        self._deprojected_mge_cache = None

    def deprojection_is_configured(self):
        return self._deprojection_config.get("inclination_deg") is not None

    @property
    def deprojected_mge(self):
        """
        Cached axisymmetric deprojected MGE model.

        Requires:
          1) fit_result to exist
          2) set_deprojection(...) to have been called
        """
        if not self.deprojection_is_configured():
            raise ValueError(
                "No deprojection is configured. Call "
                "set_deprojection(inclination_deg, distance=None, sb_unit=None) first."
            )

        if self._deprojected_mge_cache is None:
            self._deprojected_mge_cache = self.get_deprojected_mge(
                inclination_deg=self._deprojection_config["inclination_deg"],
                distance=self._deprojection_config["distance"],
                sb_unit=self._deprojection_config["sb_unit"],
            )
        return self._deprojected_mge_cache

    @staticmethod
    def _distance_to_pc(distance):
        """
        Convert distance to pc.
        - plain float -> assumed Mpc
        - astropy quantity -> converted to pc
        """
        try:
            import astropy.units as u
            if hasattr(distance, "to"):
                return float(distance.to(u.pc).value)
        except Exception:
            pass

        return float(distance) * 1.0e6

    @staticmethod
    def _normalize_sb_unit(sb_unit):
        s = str(sb_unit).replace(" ", "")
        s = s.replace("μ", "u").replace("µ", "u")
        s = s.replace("²", "^2").replace("arcsec2", "arcsec^2")
        return s

    def _native_total_to_flux_jy(self, total_native_pix2, sb_unit):
        """
        Convert the fitted Gaussian total from native image units * pix^2
        into integrated flux density in Jy.
        """
        unit = self._normalize_sb_unit(sb_unit)

        pix_solid_angle_sr = (float(self.pixel_scale) * ARCSEC_TO_RAD) ** 2
        pix_area_arcsec2 = float(self.pixel_scale) ** 2

        factors = {
            "MJy/sr":        1.0e6 * pix_solid_angle_sr,
            "Jy/sr":         1.0    * pix_solid_angle_sr,
            "mJy/sr":        1.0e-3 * pix_solid_angle_sr,
            "uJy/sr":        1.0e-6 * pix_solid_angle_sr,

            "MJy/arcsec^2":  1.0e6 * pix_area_arcsec2,
            "Jy/arcsec^2":   1.0    * pix_area_arcsec2,
            "mJy/arcsec^2":  1.0e-3 * pix_area_arcsec2,
            "uJy/arcsec^2":  1.0e-6 * pix_area_arcsec2,
        }

        if unit not in factors:
            raise ValueError(
                f"Unsupported sb_unit='{sb_unit}'. Supported units are: "
                + ", ".join(factors.keys())
            )

        return np.asarray(total_native_pix2, dtype=float) * factors[unit]

    def get_deprojected_mge(self, inclination_deg=None, distance=None, sb_unit=None):
        """
        Build and return the axisymmetric oblate deprojected MGE model.

        Notes
        -----
        Uses the axisymmetric oblate relation
            q_intr^2 = (q_obs^2 - cos(i)^2) / sin(i)^2
        and keeps the same Gaussian total in projection and in 3D.

        Returns
        -------
        AxisymmetricDeprojectedMGE
        """
        if self.fit_result is None or not hasattr(self.fit_result, "sol"):
            raise RuntimeError("No fit_result found. Run run_fit() first.")

        if inclination_deg is None:
            inclination_deg = self._deprojection_config.get("inclination_deg", None)

        if inclination_deg is None:
            raise ValueError(
                "inclination_deg must be provided or configured with set_deprojection()."
            )

        total_native_pix2, sigma_pix, qobs = self.fit_result.sol
        total_native_pix2 = np.asarray(total_native_pix2, dtype=float)
        sigma_pix = np.asarray(sigma_pix, dtype=float)
        qobs = np.asarray(qobs, dtype=float)

        inc_deg = float(inclination_deg)
        if not (0.0 < inc_deg <= 90.0):
            raise ValueError("inclination_deg must satisfy 0 < inclination_deg <= 90.")

        inc = np.deg2rad(inc_deg)
        sini = np.sin(inc)
        cosi = np.cos(inc)

        # Oblate axisymmetric deprojection
        qintr2 = (qobs**2 - cosi**2) / (sini**2)

        if np.any(qintr2 <= 0):
            qmin = np.min(qobs)
            inc_min = np.degrees(np.arccos(np.clip(qmin, 0.0, 1.0)))
            raise ValueError(
                f"Chosen inclination {inc_deg:.2f} deg is too low for this MGE. "
                f"Need inclination > arccos(min(q_obs)) ≈ {inc_min:.2f} deg."
            )

        qintr = np.sqrt(qintr2)

        # Here the current wrapper treats fit_result.sol[1] as sigma in pixels
        sigma_arcsec = sigma_pix * float(self.pixel_scale)

        # total_native_pix2 has units [native SB unit * pix^2]
        total_native_arcsec2 = total_native_pix2 * float(self.pixel_scale) ** 2

        # Intrinsic density in angular units: [native SB unit / arcsec]
        rho0_native_per_arcsec = (
            total_native_arcsec2 /
            (((2.0 * np.pi) ** 1.5) * sigma_arcsec**3 * qintr)
        )

        sigma_pc = None
        total_flux_jy = None
        total_lnu_W_hz = None
        rho0_W_hz_pc3 = None
        physical_density_unit = None

        if (distance is None) ^ (sb_unit is None):
            raise ValueError(
                "To get physical units you must provide both distance and sb_unit."
            )

        if distance is not None and sb_unit is not None:
            D_pc = self._distance_to_pc(distance)
            sigma_pc = D_pc * sigma_arcsec * ARCSEC_TO_RAD

            total_flux_jy = self._native_total_to_flux_jy(total_native_pix2, sb_unit)
            D_m = D_pc * PC_TO_M
            total_lnu_W_hz = 4.0 * np.pi * D_m**2 * total_flux_jy * JY_TO_W_M2_HZ

            # Physical emissivity: [W Hz^-1 pc^-3]
            rho0_W_hz_pc3 = (
                total_lnu_W_hz /
                (((2.0 * np.pi) ** 1.5) * sigma_pc**3 * qintr)
            )
            physical_density_unit = "W Hz^-1 pc^-3"

        native_density_unit = (
            f"{sb_unit}/arcsec" if sb_unit is not None else "native_SB_unit/arcsec"
        )

        return AxisymmetricDeprojectedMGE(
            inclination_deg=inc_deg,
            qobs=qobs,
            qintr=qintr,
            sigma_pix=sigma_pix,
            sigma_arcsec=sigma_arcsec,
            sigma_pc=sigma_pc,
            total_native_pix2=total_native_pix2,
            total_native_arcsec2=total_native_arcsec2,
            total_flux_jy=total_flux_jy,
            total_lnu_W_hz=total_lnu_W_hz,
            rho0_native_per_arcsec=rho0_native_per_arcsec,
            rho0_W_hz_pc3=rho0_W_hz_pc3,
            native_density_unit=native_density_unit,
            physical_density_unit=physical_density_unit,
            sb_unit=sb_unit,
        )

    def evaluate_deprojected_density(
        self,
        R,
        z,
        physical=False,
        inclination_deg=None,
        distance=None,
        sb_unit=None,
    ):
        """
        Convenience wrapper around the deprojected MGE density evaluator.
        """
        if (
            inclination_deg is None
            and distance is None
            and sb_unit is None
            and self.deprojection_is_configured()
        ):
            return self.deprojected_mge.density(R, z, physical=physical)

        dmge = self.get_deprojected_mge(
            inclination_deg=inclination_deg,
            distance=distance,
            sb_unit=sb_unit,
        )
        return dmge.density(R, z, physical=physical)

    # -------------------------
    # stage 1: find_galaxy
    # -------------------------
    def run_find(self, force=False, save=True, load=True):
        if self.has_manual_geometry() and not force:
            print("Manual geometry is set; skipping run_find().")
            if self.checkplot_dir is not None:
                self._plot_find_geometry_overlay()
            return None

        if (not force) and load:
            cached = self._load_cache("find")
            if cached is not None:
                self.find_result = cached["find_result"]
                return self.find_result

        def _do_find():
            return mge.find_galaxy(
                self.img_for_find,
                fraction=self.find_fraction,
                binning=self.find_binning,
                plot=self.plot,
                quiet=self.quiet
            )

        f, new_figs = _capture_new_figures(_do_find)
        self.find_result = f

        if self.checkplot_dir is not None:
            if self.plot and len(new_figs) > 0:
                paths = [
                    os.path.join(self.checkplot_dir, f"{self.prefix}_00_find_galaxy_diagnostic_{i+1:02d}.png")
                    for i in range(len(new_figs))
                ]
                _save_and_close_figs(new_figs, paths, dpi=self.dpi)

            self._plot_find_geometry_overlay()

        if save:
            self._save_cache("find", {"find_result": self.find_result})

        print(
            f"Found galaxy center at public (x, y) = ({self.xc:.2f}, {self.yc:.2f}), "
            f"PA={self.pa:.2f} deg, eps={self.eps:.3f}, theta={self.theta:.2f} deg"
        )
        print(
            f"Sky mean: {self.sky_mean:.3f}  Sky sigma: {self.sky_sigma:.3f}"
            if self.subtract_sky else "No sky subtraction applied."
        )

        return self.find_result

    def _plot_find_geometry_overlay(self):
        if not self.geometry_is_available():
            raise RuntimeError("No geometry available. Use set_manual_geometry() or run_find().")

        qbar = 1.0 - self.eps

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(_stretch_for_display(self.img, goodmask=self.goodmask), origin="lower")
        try:
            ax.contour((~self.goodmask).astype(float), levels=[0.5], linewidths=0.8, alpha=0.8, origin="lower")
        except Exception:
            pass

        ax.plot([self.xc], [self.yc], marker="+", markersize=14)

        L_arcsec = 20.0
        L_pix = L_arcsec / self.pixel_scale
        pa_rad = np.deg2rad(self.pa)
        dx = L_pix * np.sin(pa_rad)
        dy = L_pix * np.cos(pa_rad)
        ax.plot([self.xc - dx, self.xc + dx], [self.yc - dy, self.yc + dy], linewidth=1.2)

        for a_arc in [5.0, 15.0, 30.0]:
            a_pix = a_arc / self.pixel_scale
            b_pix = a_pix * qbar
            e = Ellipse(
                (self.xc, self.yc),
                width=2 * a_pix,
                height=2 * b_pix,
                angle=(90.0 - self.pa),
                fill=False,
                linewidth=1.0
            )
            ax.add_patch(e)

        ax.set_title(f"{self.prefix}: geometry overlay  (PA={self.pa:.2f} deg, eps={self.eps:.3f})")
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_01_geometry_overlay.png"), dpi=self.dpi)
        plt.close(fig)


    def run_sectors(self, force=False, save=True, load=True, ensure_find=True):
        if ensure_find and (not self.geometry_is_available()):
            self.run_find(force=False, save=save, load=load)

        if not self.geometry_is_available():
            raise RuntimeError("Geometry is not available. Use set_manual_geometry() or run_find().")

        if (not force) and load:
            cached = self._load_cache("sectors")
            if cached is not None:
                self.find_result = cached.get("find_result", self.find_result)
                self.sectors_result = cached["sectors_result"]

                manual = cached.get("manual_geometry", None)
                if manual is not None:
                    self._manual_center = manual["center"]
                    self._manual_pa = manual["pa_deg"]
                    self._manual_eps = manual["eps"]
                    self._manual_theta = manual["theta_deg"]

                return self.sectors_result

        # Public center is (x, y)=(col, row).
        # Translate at the library boundary.
        mge_xc, mge_yc = self._public_xy_to_mge_xy((self.xc, self.yc))

        def _do_sectors():
            return mge.sectors_photometry(
                self.img_work,
                eps=self.eps,
                ang=self.pa,
                xc=mge_xc,
                yc=mge_yc,
                mask=self.goodmask,
                n_sectors=self.n_sectors,
                minlevel=self.minlevel,
                plot=self.plot
            )

        s, new_figs = _capture_new_figures(_do_sectors)
        self.sectors_result = s

        if self.checkplot_dir is not None:
            if self.plot and len(new_figs) > 0:
                paths = [
                    os.path.join(self.checkplot_dir, f"{self.prefix}_10_sectors_photometry_{i+1:02d}.png")
                    for i in range(len(new_figs))
                ]
                _save_and_close_figs(new_figs, paths, dpi=self.dpi)

            self._plot_sectors_overlay()
            self._plot_sector_profiles()

        if save:
            self._save_cache(
                "sectors",
                {
                    "find_result": self.find_result,
                    "manual_geometry": {
                        "center": self._manual_center,
                        "pa_deg": self._manual_pa,
                        "eps": self._manual_eps,
                        "theta_deg": self._manual_theta,
                    },
                    "sectors_result": self.sectors_result,
                }
            )

        print("Proceeding to sectors_photometry with n_sectors =", self.n_sectors)
        print(f"Public center used by wrapper: (x, y)=({self.xc:.2f}, {self.yc:.2f})")
        print(f"Center passed to mgefit: xc={mge_xc:.2f}, yc={mge_yc:.2f}")
        return self.sectors_result

    def _plot_sectors_overlay(self):
        s = self.sectors_result
        x_img, y_img = polar_points_to_image_xy(s.radius, s.angle, self.pa, (self.xc, self.yc))

        npts = x_img.size
        if npts > self.max_points_overlay:
            rng = np.random.default_rng(12345)
            idx = rng.choice(npts, size=self.max_points_overlay, replace=False)
        else:
            idx = slice(None)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(_stretch_for_display(self.img, goodmask=self.goodmask), origin="lower")

        ax.scatter(x_img[idx], y_img[idx], s=1, alpha=0.4, color="orange")
        ax.plot([self.xc], [self.yc], marker="+", markersize=14)
        ax.set_title(f"{self.prefix}: sampled photometry points (sectors)")
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_11_sectors_sampled_points.png"), dpi=self.dpi)
        plt.close(fig)

    def _plot_sector_profiles(self):
        s = self.sectors_result
        r_arc = np.asarray(s.radius) * self.pixel_scale
        y_dat = np.asarray(s.counts)
        ang = np.asarray(s.angle)
        uniq = np.unique(np.round(ang, 6))

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for a in uniq:
            sel = np.isclose(ang, a)
            rr = r_arc[sel]
            yy = y_dat[sel]
            o = np.argsort(rr)
            rr, yy = rr[o], yy[o]
            ax.plot(rr, yy, linewidth=0.8, alpha=0.5)

        pos = y_dat[np.isfinite(y_dat) & (y_dat > 0)]
        if pos.size > 0 and np.nanmin(pos) > 0:
            ax.set_yscale("log")
            ax.set_xscale("log")

        ax.set_xlabel("R (arcsec)")
        ax.set_ylabel("counts / pix")
        ax.set_title(f"{self.prefix}: sector radial profiles (data)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_12_sector_profiles.pdf"), dpi=self.dpi)
        plt.close(fig)

    # -------------------------
    # stage 3: fit_sectors
    # -------------------------
    def run_fit(self, force=False, save=True, load=True, ensure_sectors=True):
        if ensure_sectors and self.sectors_result is None:
            self.run_sectors(force=False, save=save, load=load)

        if self.sectors_result is None:
            raise RuntimeError("sectors_result is not available. Run run_sectors() first.")

        if (not force) and load:
            cached = self._load_cache("fit")
            if cached is not None:
                self.find_result = cached.get("find_result", self.find_result)
                self.sectors_result = cached["sectors_result"]
                self.fit_result = cached["fit_result"]
                self._deprojected_mge_cache = None

                manual = cached.get("manual_geometry", None)
                if manual is not None:
                    self._manual_center = manual["center"]
                    self._manual_pa = manual["pa_deg"]
                    self._manual_eps = manual["eps"]
                    self._manual_theta = manual["theta_deg"]

                deproj = cached.get("deprojection_config", None)
                if deproj is not None:
                    self._deprojection_config = deproj

                return self.fit_result

        s = self.sectors_result

        def _do_fit():
            return mge.fit_sectors(
                s.radius, s.angle, s.counts, self.eps,
                linear=self.linear,
                ngauss=self.ngauss,
                qbounds=self.qbounds,
                outer_slope=self.outer_slope,
                sigmapsf=self.sigmapsf,
                normpsf=self.normpsf,
                scale=self.pixel_scale,
                plot=self.plot,
                quiet=self.quiet,
                bulge_disk=self.bulge_disk,
                negative=self.allow_negative
            )

        m, new_figs = _capture_new_figures(_do_fit)
        self.fit_result = m
        self._deprojected_mge_cache = None

        if self.checkplot_dir is not None:
            if self.plot and len(new_figs) > 0:
                paths = [
                    os.path.join(self.checkplot_dir, f"{self.prefix}_20_fit_sectors_{i+1:02d}.png")
                    for i in range(len(new_figs))
                ]
                _save_and_close_figs(new_figs, paths, dpi=self.dpi)

            self._plot_fit_profiles_and_residuals()
            self._plot_fit_contours_and_ratio()

        if save:
            self._save_cache(
                "fit",
                {
                    "find_result": self.find_result,
                    "manual_geometry": {
                        "center": self._manual_center,
                        "pa_deg": self._manual_pa,
                        "eps": self._manual_eps,
                        "theta_deg": self._manual_theta,
                    },
                    "deprojection_config": self._deprojection_config,
                    "sectors_result": self.sectors_result,
                    "fit_result": self.fit_result,
                }
            )

        return self.fit_result

    def _plot_fit_profiles_and_residuals(self):
        s = self.sectors_result
        m = self.fit_result

        r_pix = np.asarray(s.radius, dtype=float)
        ang_deg_all = np.asarray(s.angle, dtype=float)
        y_dat = np.asarray(s.counts, dtype=float)

        # Model evaluated exactly at the sector photometry points
        y_fit_pts = mge_model_counts_at_polar_points(r_pix, ang_deg_all, m.sol)

        # Build a common radial grid
        rpos = r_pix[np.isfinite(r_pix) & (r_pix > 0)]
        if rpos.size == 0:
            return

        rmin = max(np.nanmin(rpos), 0.5)
        rmax = np.nanmax(rpos)
        rgrid = np.geomspace(rmin, rmax, 250)
        rgrid_arc = rgrid * self.pixel_scale
        r_arc = r_pix * self.pixel_scale

        # Unique sector angles
        angle_round_decimals = 6
        ang_key = np.round(ang_deg_all, angle_round_decimals)
        unique_angles = np.array(sorted(np.unique(ang_key)))

        # Extract MGE solution arrays
        sol = np.asarray(m.sol, dtype=float)
        if sol.ndim != 2 or sol.shape[0] < 3:
            sol = np.vstack(m.sol)

        total_counts = np.asarray(sol[0], dtype=float)
        sigma_pix = np.asarray(sol[1], dtype=float)
        q_obs = np.asarray(sol[2], dtype=float)

        valid_g = (
            np.isfinite(total_counts) & np.isfinite(sigma_pix) & np.isfinite(q_obs)
            & (sigma_pix > 0) & (q_obs > 0) & (total_counts > 0)
        )

        total_counts = total_counts[valid_g]
        sigma_pix = sigma_pix[valid_g]
        q_obs = q_obs[valid_g]

        order = np.argsort(sigma_pix)
        total_counts = total_counts[order]
        sigma_pix = sigma_pix[order]
        q_obs = q_obs[order]

        def component_profiles_at_angle(rvals_pix, ang_deg):
            """
            Return profiles of each Gaussian component at a given polar angle.
            Output shape = (n_r, n_gauss)
            Units match mge_model_counts_at_polar_points: counts / pix.
            """
            rvals_pix = np.asarray(rvals_pix, dtype=float)[:, None]
            th = np.deg2rad(float(ang_deg))
            c2 = np.cos(th) ** 2
            s2 = np.sin(th) ** 2

            amp = total_counts / (2.0 * np.pi * sigma_pix**2 * q_obs)
            expo = -0.5 * (rvals_pix**2 / sigma_pix**2) * (c2 + s2 / q_obs**2)
            return amp[None, :] * np.exp(expo)

        # ---------------------------
        # 1) Radial profiles by angle
        # ---------------------------
        nang = len(unique_angles)
        ncols = min(3, nang)
        nrows = int(np.ceil(nang / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.2 * ncols, 4.2 * nrows),
            squeeze=False
        )
        axes = axes.ravel()

        for i, ang in enumerate(unique_angles):
            ax = axes[i]
            mask = np.isfinite(r_pix) & np.isfinite(y_dat) & np.isclose(ang_key, ang)

            if not np.any(mask):
                ax.set_visible(False)
                continue

            r_this_arc = r_arc[mask]
            y_this = y_dat[mask]

            y_model = mge_model_counts_at_polar_points(
                rgrid, np.full_like(rgrid, ang, dtype=float), m.sol
            )

            y_comp = component_profiles_at_angle(rgrid, ang)

            ax.scatter(r_this_arc, y_this, s=8, alpha=0.45, label="data")

            for j in range(y_comp.shape[1]):
                ax.plot(
                    rgrid_arc, y_comp[:, j],
                    linewidth=0.6, alpha=0.25
                )

            ax.plot(rgrid_arc, y_model, linewidth=1.8, label="total model")

            ax.set_title(rf"$\theta = {ang:.1f}^\circ$")
            ax.set_xlabel("R (arcsec)")
            ax.set_ylabel("counts / pix")

            good_y = np.concatenate([y_this[np.isfinite(y_this)], y_model[np.isfinite(y_model)]])
            good_y_pos = good_y[good_y > 0]
            if good_y_pos.size > 0:
                ax.set_xscale("log")
                ax.set_yscale("log")
                y_model_pos = y_model[y_model > 0]
                if y_model_pos.size > 0 and np.any(np.isfinite(y_this)):
                    ax.set_ylim(np.nanmin(y_model_pos) / 5, np.nanmax(y_this) * 1.3)

            ax.grid(alpha=0.2)
            if i == 0:
                ax.legend(fontsize=9)

        for j in range(nang, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{self.prefix}: radial profile by sector angle  (absdev={m.absdev:.4f})", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(
            os.path.join(self.checkplot_dir, f"{self.prefix}_21_radial_profiles_by_angle.pdf"),
            dpi=self.dpi
        )
        plt.close(fig)

        # ---------------------------
        # 2) Fractional residuals by angle
        # ---------------------------
        good = np.isfinite(y_dat) & np.isfinite(y_fit_pts) & (y_dat != 0)
        frac = np.full_like(y_dat, np.nan, dtype=float)
        frac[good] = 1.0 - (y_fit_pts[good] / y_dat[good])

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.2 * ncols, 3.8 * nrows),
            squeeze=False,
            sharex=False,
            sharey=True
        )
        axes = axes.ravel()

        for i, ang in enumerate(unique_angles):
            ax = axes[i]
            mask = good & np.isclose(ang_key, ang)

            if not np.any(mask):
                ax.set_visible(False)
                continue

            ax.scatter(r_arc[mask], frac[mask], s=8, alpha=0.5)
            ax.axhline(0.0, linewidth=1.0)
            ax.set_title(rf"$\theta = {ang:.1f}^\circ$")
            ax.set_xlabel("R (arcsec)")
            ax.set_ylabel(r"$1 - y_{\rm fit}/y$")

            xpos = r_arc[mask]
            xpos = xpos[np.isfinite(xpos) & (xpos > 0)]
            if xpos.size > 0:
                ax.set_xscale("log")

            ax.grid(alpha=0.2)

        for j in range(nang, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{self.prefix}: fractional residuals by sector angle", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(
            os.path.join(self.checkplot_dir, f"{self.prefix}_22_frac_residuals_by_angle.pdf"),
            dpi=self.dpi
        )
        plt.close(fig)

    def _plot_fit_contours_and_ratio(self):
        m = self.fit_result

        half_size_pix = int(max(10, self.contour_half_size_arcsec / self.pixel_scale))

        # Use image-plane angle used by the sector extraction / fit
        theta = getattr(self, "theta", None)
        if theta is None:
            theta = getattr(self, "theta_deg", None)
        if theta is None:
            theta = self.pa

        bounds, model_cut = build_mge_model_image_cutout(
            self.img.shape,
            m.sol,
            theta,
            (self.xc, self.yc),
            half_size_pix=half_size_pix,
            oversample=self.contour_oversample,
        )

        x1, x2, y1, y2 = bounds
        data_cut = self.img[y1:y2, x1:x2]
        good_cut = self.goodmask[y1:y2, x1:x2]

        # Make sure model_cut has same axis ordering as data_cut
        if model_cut.shape != data_cut.shape:
            if model_cut.T.shape == data_cut.shape:
                model_cut = model_cut.T
            else:
                raise ValueError(
                    f"model_cut shape {model_cut.shape} does not match "
                    f"data_cut shape {data_cut.shape}"
                )
        else:
            # If square, compare both orientations
            if model_cut.shape[0] == model_cut.shape[1]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    valid0 = good_cut & np.isfinite(data_cut) & np.isfinite(model_cut) & (data_cut > 0) & (model_cut > 0)
                    validT = good_cut & np.isfinite(data_cut) & np.isfinite(model_cut.T) & (data_cut > 0) & (model_cut.T > 0)

                    score0 = np.inf
                    scoreT = np.inf

                    if np.count_nonzero(valid0) > 20:
                        score0 = np.nanmedian(np.abs(np.log(data_cut[valid0] / model_cut[valid0])))

                    if np.count_nonzero(validT) > 20:
                        scoreT = np.nanmedian(np.abs(np.log(data_cut[validT] / model_cut.T[validT])))

                if scoreT < score0:
                    model_cut = model_cut.T

        v = data_cut[good_cut]
        v = v[np.isfinite(v)]

        if v.size > 0:
            levels = np.percentile(v, [70, 80, 90, 95, 97, 99])
            levels = np.unique(levels[np.isfinite(levels)])

            if levels.size >= 3:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.imshow(
                    _stretch_for_display(data_cut, goodmask=good_cut),
                    origin="lower",
                    extent=[x1, x2, y1, y2],
                )
                ax.contour(
                    model_cut,
                    levels=levels,
                    linewidths=1.0,
                    origin="lower",
                    extent=[x1, x2, y1, y2]
                )
                ax.plot([self.xc], [self.yc], marker="+", markersize=14)
                ax.set_title(f"{self.prefix}: model contours over data (cutout)")
                ax.set_xlabel("x (pix)")
                ax.set_ylabel("y (pix)")
                fig.tight_layout()
                fig.savefig(
                    os.path.join(self.checkplot_dir, f"{self.prefix}_23_model_contours_over_data.png"),
                    dpi=self.dpi
                )
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(7, 7))
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = data_cut / model_cut
                ax.imshow(
                    _stretch_for_display(ratio, goodmask=good_cut),
                    origin="lower",
                    extent=[x1, x2, y1, y2]
                )
                ax.set_title(f"{self.prefix}: residual (cutout)")
                ax.set_xlabel("x (pix)")
                ax.set_ylabel("y (pix)")
                fig.tight_layout()
                fig.savefig(
                    os.path.join(self.checkplot_dir, f"{self.prefix}_24_data_over_model_ratio.png"),
                    dpi=self.dpi
                )
                plt.close(fig)
    

    # def _plot_fit_contours(self, show=False, smoothing=None):
    #     """
    #     Plot data isophotes and MGE model contours.

    #     Parameters
    #     ----------
    #     show : bool
    #         If True, display the figure.
    #     smoothing : float or None
    #         Gaussian sigma in pixels applied only to the data contours.
    #         If None or <= 0, no smoothing is applied.
    #     """
    #     m = self.fit_result

    #     half_size_pix = int(max(10, self.contour_half_size_arcsec / self.pixel_scale))

    #     # Use the same image-plane angle used by the sector extraction / fit
    #     theta = getattr(self, "theta", None)
    #     if theta is None:
    #         theta = getattr(self, "theta_deg", None)
    #     if theta is None:
    #         theta = self.pa

    #     bounds, model_cut = build_mge_model_image_cutout(
    #         self.img.shape,
    #         m.sol,
    #         theta,
    #         (self.xc, self.yc),
    #         half_size_pix=half_size_pix,
    #         oversample=self.contour_oversample,
    #     )

    #     x1, x2, y1, y2 = bounds
    #     data_cut = self.img[y1:y2, x1:x2]
    #     good_cut = self.goodmask[y1:y2, x1:x2]

    #     # Make sure model_cut has same axis ordering as data_cut
    #     if model_cut.shape != data_cut.shape:
    #         if model_cut.T.shape == data_cut.shape:
    #             model_cut = model_cut.T
    #         else:
    #             raise ValueError(
    #                 f"model_cut shape {model_cut.shape} does not match "
    #                 f"data_cut shape {data_cut.shape}"
    #             )
    #     else:
    #         # If square, compare both orientations and keep the one that matches best
    #         if model_cut.shape[0] == model_cut.shape[1]:
    #             with np.errstate(divide="ignore", invalid="ignore"):
    #                 valid0 = (
    #                     good_cut
    #                     & np.isfinite(data_cut)
    #                     & np.isfinite(model_cut)
    #                     & (data_cut > 0)
    #                     & (model_cut > 0)
    #                 )
    #                 validT = (
    #                     good_cut
    #                     & np.isfinite(data_cut)
    #                     & np.isfinite(model_cut.T)
    #                     & (data_cut > 0)
    #                     & (model_cut.T > 0)
    #                 )

    #                 score0 = np.inf
    #                 scoreT = np.inf

    #                 if np.count_nonzero(valid0) > 20:
    #                     score0 = np.nanmedian(
    #                         np.abs(np.log(data_cut[valid0] / model_cut[valid0]))
    #                     )

    #                 if np.count_nonzero(validT) > 20:
    #                     scoreT = np.nanmedian(
    #                         np.abs(np.log(data_cut[validT] / model_cut.T[validT]))
    #                     )

    #             if scoreT < score0:
    #                 model_cut = model_cut.T

    #     # ------------------------------------------------------------------
    #     # Mask-aware smoothing of the data only
    #     # ------------------------------------------------------------------
    #     data_for_contours = np.array(data_cut, dtype=float, copy=True)

    #     if smoothing is not None and smoothing > 0:
    #         valid = good_cut & np.isfinite(data_cut)

    #         weights = valid.astype(float)
    #         values = np.where(valid, data_cut, 0.0)

    #         smooth_num = gaussian_filter(values * weights, sigma=smoothing, mode="nearest")
    #         smooth_den = gaussian_filter(weights, sigma=smoothing, mode="nearest")

    #         with np.errstate(invalid="ignore", divide="ignore"):
    #             data_for_contours = smooth_num / smooth_den

    #         data_for_contours[smooth_den <= 1e-6] = np.nan
    #         data_for_contours[~good_cut] = np.nan

    #     # Valid positive data values for defining contour levels
    #     valid_data = good_cut & np.isfinite(data_for_contours) & (data_for_contours > 0)
    #     v = data_for_contours[valid_data]

    #     if v.size == 0:
    #         return

    #     # Use logarithmically spaced levels
    #     lo = np.nanpercentile(v, 70)
    #     hi = np.nanpercentile(v, 99.7)

    #     if not np.isfinite(lo) or not np.isfinite(hi):
    #         return

    #     if hi > lo and lo > 0:
    #         levels = np.geomspace(lo, hi, 8)
    #     else:
    #         levels = np.percentile(v, [70, 80, 90, 95, 97, 99])

    #     levels = np.unique(levels[np.isfinite(levels)])
    #     if levels.size < 3:
    #         return

    #     extent = [x1, x2, y1, y2]

    #     fig, ax = plt.subplots(figsize=(7, 7))

    #     # Smoothed data isophotes
    #     ax.contour(
    #         data_for_contours,
    #         levels=levels,
    #         colors="cyan",
    #         linewidths=0.6,
    #         origin="lower",
    #         extent=extent,
    #         alpha=0.9,
    #     )

    #     # MGE model contours at the same levels
    #     ax.contour(
    #         model_cut,
    #         levels=levels,
    #         colors="red",
    #         linewidths=1.0,
    #         linestyles="--",
    #         origin="lower",
    #         extent=extent,
    #     )

    #     ax.plot([self.xc], [self.yc], marker="+", markersize=14, color="yellow", mew=1.5)

    #     legend_handles = [
    #         Line2D([0], [0], color="cyan", lw=1.5, label="Data isophotes"),
    #         Line2D([0], [0], color="red", lw=1.5, ls="--", label="MGE model contours"),
    #         Line2D([0], [0], color="yellow", marker="+", lw=0, markersize=10, label="Center"),
    #     ]
    #     ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    #     title = f"{self.prefix}: data isophotes + MGE model contours"
    #     if smoothing is not None and smoothing > 0:
    #         title += f" (Gaussian σ={smoothing:.2f} pix)"
    #     ax.set_title(title)

    #     ax.set_xlabel("x (pix)")
    #     ax.set_ylabel("y (pix)")
    #     fig.tight_layout()
    #     fig.savefig(
    #         os.path.join(
    #             self.checkplot_dir,
    #             f"{self.prefix}_23_data_isophotes_and_model_contours.png",
    #         ),
    #         dpi=self.dpi,
    #     )
    #     if show:
    #         plt.show()
    #     plt.close(fig)
    def _plot_fit_contours(
        self,
        show=False,
        smoothing=0.0,
        magstep=0.5,
        minlevel=None,
        nlevels=None,
    ):
        m = self.fit_result

        half_size_pix = int(max(10, self.contour_half_size_arcsec / self.pixel_scale))

        theta = getattr(self, "theta", None)
        if theta is None:
            theta = getattr(self, "theta_deg", None)
        if theta is None:
            theta = self.pa

        bounds, model_cut = build_mge_model_image_cutout(
            self.img.shape,
            m.sol,
            theta,
            (self.xc, self.yc),
            half_size_pix=half_size_pix,
            oversample=self.contour_oversample,
        )

        x1, x2, y1, y2 = bounds
        data_cut = self.img[y1:y2, x1:x2]
        good_cut = self.goodmask[y1:y2, x1:x2]

        if model_cut.shape != data_cut.shape:
            if model_cut.T.shape == data_cut.shape:
                model_cut = model_cut.T
            else:
                raise ValueError(
                    f"model_cut shape {model_cut.shape} does not match "
                    f"data_cut shape {data_cut.shape}"
                )

        # very mild optional smoothing, just to suppress pixel noise
        data_for_contours = np.array(data_cut, dtype=float, copy=True)
        if smoothing is not None and smoothing > 0:
            valid = good_cut & np.isfinite(data_cut)
            weights = valid.astype(float)
            values = np.where(valid, data_cut, 0.0)

            num = gaussian_filter(values * weights, sigma=smoothing, mode="nearest")
            den = gaussian_filter(weights, sigma=smoothing, mode="nearest")

            with np.errstate(divide="ignore", invalid="ignore"):
                data_for_contours = num / den

            data_for_contours[den <= 1e-8] = np.nan
            data_for_contours[~good_cut] = np.nan

        levels = _native_like_mge_levels(
            data_for_contours,
            good_cut,
            magstep=magstep,
            minlevel=minlevel,
            nlevels=nlevels,
        )
        if levels is None:
            return

        extent = [x1, x2, y1, y2]

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.contour(
            data_for_contours,
            levels=levels,
            colors="k",
            linewidths=0.6,
            origin="lower",
            extent=extent,
        )

        ax.contour(
            model_cut,
            levels=levels,
            colors="r",
            linewidths=1.0,
            linestyles="--",
            origin="lower",
            extent=extent,
        )

        ax.plot([self.xc], [self.yc], marker="+", markersize=14, color="gold", mew=1.5)

        legend_handles = [
            Line2D([0], [0], color="k", lw=1.2, label="Data isophotes"),
            Line2D([0], [0], color="r", lw=1.2, ls="--", label="MGE model contours"),
            Line2D([0], [0], color="gold", marker="+", lw=0, markersize=10, label="Center"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True)

        ax.set_title(
            f"{self.prefix}: contours (magstep={magstep}"
            + (f", smooth={smoothing:.2f}px" if smoothing and smoothing > 0 else "")
            + ")"
        )
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                self.checkplot_dir,
                f"{self.prefix}_23_data_isophotes_and_model_contours.png",
            ),
            dpi=self.dpi,
        )

        if show:
            plt.show()
        plt.close(fig)


    # runners
    def run_all(self, force_find=False, force_sectors=False, force_fit=False, save=True, load=True):
        if self.has_manual_geometry():
            self.run_sectors(force=force_sectors, save=save, load=load)
        else:
            self.run_find(force=force_find, save=save, load=load)
            self.run_sectors(force=force_sectors, save=save, load=load)

        self.run_fit(force=force_fit, save=save, load=load)
        return self.results_dict()

    def results_dict(self):
        if self.fit_result is None:
            raise RuntimeError("fit_result is not available yet. Run run_fit() or run_all().")

        total_counts, sigma_pix, q_obs = self.fit_result.sol
        surf = total_counts / (2.0 * np.pi * q_obs * sigma_pix**2)
        sigma_arcsec = sigma_pix * float(self.pixel_scale)
        table = np.vstack([surf, sigma_pix, sigma_arcsec, q_obs, total_counts]).T

        out = {
            "center_pix": (self.xc, self.yc),
            "pa_deg": self.pa,
            "eps": self.eps,
            "theta_deg": self.theta,
            "manual_geometry": self.has_manual_geometry(),
            "sky_mean": self.sky_mean,
            "sky_sigma": self.sky_sigma,
            "find_result": self.find_result,
            "sectors": self.sectors_result,
            "mgefit": self.fit_result,
            "table_cols": ["surf_counts_per_pix", "sigma_pix", "sigma_arcsec", "q_obs", "total_counts"],
            "table": table,
            "deprojection_config": self._deprojection_config,
            "checkplot_dir": self.checkplot_dir,
            "cache_dir": self.cache_dir,
        }

        if self.deprojection_is_configured():
            try:
                dmge = self.deprojected_mge
                out["deprojected_mge"] = dmge
            except Exception as exc:
                out["deprojected_mge_error"] = str(exc)

        return out

    def save_final_results(self, filename=None):
        if filename is None:
            if self.cache_dir is None:
                raise ValueError("Provide filename or set cache_dir.")
            filename = os.path.join(self.cache_dir, f"{self.prefix}_final_results.pkl")

        with open(filename, "wb") as f:
            pickle.dump(self.results_dict(), f)

        return filename

    def plot_deprojected_density_map(
        self,
        inc_deg=None,
        distance_mpc=None,
        sb_unit=None,
        ml=1.0,
        half_size_arcsec=30.0,
        npix=400,
        save=True,
        mass_density=False,
    ):
        """
        Plot the deprojected axisymmetric MGE density on a meridional (R,z) plane.

        Parameters
        ----------
        inc_deg : float or None
            Inclination in degrees. If None, use the configured deprojection inclination.
        distance_mpc : float or None
            Distance in Mpc. If given together with sb_unit, axes are in pc and the
            density is the physical emissivity in W Hz^-1 pc^-3.
        sb_unit : str or None
            Surface-brightness unit of the image, e.g. 'MJy/sr'.
        ml : float
            Multiplicative scale factor applied only if mass_density=True.
        half_size_arcsec : float
            Half-size of the plotted box in arcsec.
        npix : int
            Number of pixels per axis in the output density map.
        save : bool
            Save the figure to checkplot_dir.
        mass_density : bool
            If True, multiply the density by ml before plotting.
        """
        if self.fit_result is None:
            raise RuntimeError("No fit_result found. Run run_fit() first.")

        if inc_deg is None:
            inc_deg = self._deprojection_config.get("inclination_deg", None)

        if inc_deg is None:
            raise ValueError(
                "No inclination provided. Pass inc_deg or call set_deprojection(...) first."
            )

        if distance_mpc is None:
            distance_mpc = self._deprojection_config.get("distance", None)

        if sb_unit is None:
            sb_unit = self._deprojection_config.get("sb_unit", None)

        use_physical = (distance_mpc is not None) and (sb_unit is not None)

        dmge = self.get_deprojected_mge(
            inclination_deg=inc_deg,
            distance=distance_mpc,
            sb_unit=sb_unit,
        )

        if use_physical:
            pc_per_arcsec = self._distance_to_pc(distance_mpc) * ARCSEC_TO_RAD
            half_size = half_size_arcsec * pc_per_arcsec
            x, z, rho = dmge.grid(half_size=half_size, npix=npix, physical=True)
            xlab = "R (mirrored) [pc]"
            ylab = "z [pc]"
            unit_label = dmge.physical_density_unit
        else:
            x, z, rho = dmge.grid(half_size=half_size_arcsec, npix=npix, physical=False)
            xlab = "R (mirrored) [arcsec]"
            ylab = "z [arcsec]"
            unit_label = dmge.native_density_unit

        if mass_density:
            rho = ml * rho
            unit_label = f"scaled {unit_label}"

        rho_plot = np.full_like(rho, np.nan, dtype=float)
        good = np.isfinite(rho) & (rho > 0)
        rho_plot[good] = np.log10(rho[good])

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            rho_plot,
            origin="lower",
            extent=[x.min(), x.max(), z.min(), z.max()],
            aspect="equal",
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"log10 density [{unit_label}]")

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{self.prefix}: deprojected axisymmetric density (i={float(inc_deg):.1f} deg)")
        fig.tight_layout()

        if save:
            if self.checkplot_dir is None:
                raise ValueError("checkplot_dir is None, so the plot cannot be saved.")
            out = os.path.join(
                self.checkplot_dir,
                f"{self.prefix}_25_deprojected_density_i{float(inc_deg):.1f}.png"
            )
            fig.savefig(out, dpi=self.dpi)

        plt.close(fig)

        return {
            "x": x,
            "z": z,
            "rho": rho,
            "qintr": dmge.qintr,
            "sigma_arcsec": dmge.sigma_arcsec,
            "sigma_pc": dmge.sigma_pc,
            "rho0_native_per_arcsec": dmge.rho0_native_per_arcsec,
            "rho0_W_hz_pc3": dmge.rho0_W_hz_pc3,
            "unit": unit_label,
            "deprojected_mge": dmge,
        }