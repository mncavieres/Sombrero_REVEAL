import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mgefit as mge
from astropy.io import fits


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
    x = r * np.cos(th)
    y = r * np.sin(th)

    xx = x[:, None]
    yy = y[:, None]
    sig2 = sigma[None, :] ** 2
    q2 = q[None, :] ** 2

    expo = -0.5 * (xx**2 + (yy**2) / q2) / sig2
    return np.sum(surf[None, :] * np.exp(expo), axis=1)


def polar_points_to_image_xy(radius_pix, angle_deg, pa_deg, center_pix_xy):
    """
    Convert (radius, angle from major axis) -> image (x,y) pixels for overlay.

    center_pix_xy = (xc, yc) = (x, y) = (col, row).
    Returns x=col, y=row for imshow(origin='lower').
    """
    xc, yc = center_pix_xy
    x0 = float(xc)
    y0 = float(yc)

    r = np.asarray(radius_pix, float)
    th = np.deg2rad(np.asarray(angle_deg, float))
    xp = r * np.cos(th)
    yp = r * np.sin(th)

    pa = np.deg2rad(pa_deg)
    x_img = x0 + xp * np.sin(pa) + yp * np.cos(pa)
    y_img = y0 + xp * np.cos(pa) - yp * np.sin(pa)
    return x_img, y_img


def build_mge_model_image_cutout(img_shape, sol, pa_deg, center_pix_xy,
                                 half_size_pix=400, oversample=1):
    """
    Build a model image cutout (counts/pix) from the MGE solution.
    NOTE: If you supplied a PSF to fit_sectors, sol is deconvolved.
    """
    ny, nx = img_shape
    xc, yc = center_pix_xy
    x0 = float(xc)
    y0 = float(yc)

    x1 = int(max(0, np.floor(x0 - half_size_pix)))
    x2 = int(min(nx, np.ceil(x0 + half_size_pix)))
    y1 = int(max(0, np.floor(y0 - half_size_pix)))
    y2 = int(min(ny, np.ceil(y0 + half_size_pix)))

    xs = np.arange(x1, x2, 1 / oversample)
    ys = np.arange(y1, y2, 1 / oversample)
    X, Y = np.meshgrid(xs, ys)

    pa = np.deg2rad(pa_deg)
    dx = X - x0
    dy = Y - y0
    xp = dx * np.sin(pa) + dy * np.cos(pa)
    yp = dx * np.cos(pa) - dy * np.sin(pa)

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


class MGEFitter:
    """
    Resumable MGE fitting pipeline with per-stage disk caching.

    You can either:
      1) use run_find() to get geometry from mge.find_galaxy, or
      2) manually provide geometry via:
            center=(xc, yc), pa_deg=..., eps=..., theta_deg=...
         and skip run_find().
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
        # NEW: manual geometry
        center=None,          # (xc, yc)
        pa_deg=None,
        eps=None,
        theta_deg=None,
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

        if self.checkplot_dir is not None:
            _ensure_dir(self.checkplot_dir)
        if self.cache_dir is not None:
            _ensure_dir(self.cache_dir)

        # keep your original goodmask convention unchanged
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

        self.find_result = None
        self.sectors_result = None
        self.fit_result = None

        # NEW: manual geometry storage
        self._manual_center = None
        self._manual_pa = None
        self._manual_eps = None
        self._manual_theta = None

        if center is not None or pa_deg is not None or eps is not None or theta_deg is not None:
            self.set_manual_geometry(center=center, pa_deg=pa_deg, eps=eps, theta_deg=theta_deg)

    # -------------------------
    # manual geometry
    # -------------------------
    def set_manual_geometry(self, *, center=None, pa_deg=None, eps=None, theta_deg=None):
        """
        Set manual geometry so run_find() is not required.

        Parameters
        ----------
        center : tuple
            (xc, yc) = (x, y) pixel center
        pa_deg : float
            Position angle in degrees
        eps : float
            Ellipticity = 1 - q
        theta_deg : float or None
            Optional theta used by mge.print_contours. If None, defaults to pa_deg.
        """
        if center is not None:
            if len(center) != 2:
                raise ValueError("center must be a 2-element tuple/list (xc, yc)")
            self._manual_center = (float(center[0]), float(center[1]))

        if pa_deg is not None:
            self._manual_pa = float(pa_deg)

        if eps is not None:
            self._manual_eps = float(eps)

        if theta_deg is not None:
            self._manual_theta = float(theta_deg)

    def clear_manual_geometry(self):
        """Remove manual geometry and go back to using run_find()."""
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
        return float(self.find_result.xpeak)

    @property
    def yc(self):
        if self._manual_center is not None:
            return self._manual_center[1]
        if self.find_result is None:
            return None
        return float(self.find_result.ypeak)

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
            f"Found galaxy center at (x, y) = ({self.xc:.2f}, {self.yc:.2f}), "
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

    # -------------------------
    # stage 2: sectors_photometry
    # -------------------------
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

        def _do_sectors():
            return mge.sectors_photometry(
                self.img_work,
                eps=self.eps,
                ang=self.pa,
                xc=self.xc,
                yc=self.yc,
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
        # try:
        #     ax.contour((~self.goodmask).astype(float), levels=[0.5], linewidths=0.8,
        #                alpha=0.8, origin="lower", colors="orange")
        # except Exception:
        #     pass

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

                manual = cached.get("manual_geometry", None)
                if manual is not None:
                    self._manual_center = manual["center"]
                    self._manual_pa = manual["pa_deg"]
                    self._manual_eps = manual["eps"]
                    self._manual_theta = manual["theta_deg"]

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
                bulge_disk=False, # this worked just fine without bulge_disk = True
            )

        m, new_figs = _capture_new_figures(_do_fit)
        self.fit_result = m

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
                    "sectors_result": self.sectors_result,
                    "fit_result": self.fit_result,
                }
            )

        return self.fit_result

    def _plot_fit_profiles_and_residuals(self):
        s = self.sectors_result
        m = self.fit_result

        r_arc = np.asarray(s.radius) * self.pixel_scale
        y_dat = np.asarray(s.counts)
        y_fit_pts = mge_model_counts_at_polar_points(s.radius, s.angle, m.sol)

        rpos = np.asarray(s.radius)
        rpos = rpos[np.isfinite(rpos) & (rpos > 0)]
        rmin = max(np.nanmin(rpos), 0.5) if rpos.size else 0.5
        rmax = np.nanmax(s.radius)
        rgrid = np.geomspace(rmin, rmax, 250)
        rgrid_arc = rgrid * self.pixel_scale

        y_major = mge_model_counts_at_polar_points(rgrid, np.zeros_like(rgrid), m.sol)
        y_minor = mge_model_counts_at_polar_points(rgrid, np.full_like(rgrid, 90.0), m.sol)
        y_45 = mge_model_counts_at_polar_points(rgrid, np.full_like(rgrid, 45.0), m.sol)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.scatter(r_arc, y_dat, s=6, alpha=0.35, label="data (all sectors)")
        ax.plot(rgrid_arc, y_major, linewidth=1.8, label="model (major axis)")
        ax.plot(rgrid_arc, y_minor, linewidth=1.8, label="model (minor axis)")
        ax.plot(rgrid_arc, y_45, linewidth=1.2, label="model (45 deg)")
        ax.set_xlabel("R (arcsec)")
        ax.set_ylabel("counts / pix")

        pos = y_dat[np.isfinite(y_dat) & (y_dat > 0)]
        if pos.size > 0 and np.nanmin(pos) > 0:
            ax.set_yscale("log")

        ax.set_title(f"{self.prefix}: radial profile data vs model  (absdev={m.absdev:.4f})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_21_radial_profile_data_vs_model.png"), dpi=self.dpi)
        plt.close(fig)

        good = np.isfinite(y_dat) & np.isfinite(y_fit_pts) & (y_dat != 0)
        frac = np.full_like(y_dat, np.nan, dtype=float)
        frac[good] = 1.0 - (y_fit_pts[good] / y_dat[good])

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.scatter(r_arc[good], frac[good], s=6, alpha=0.35)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("R (arcsec)")
        ax.set_ylabel(r"$1 - y_{\rm fit}/y$")
        ax.set_title(f"{self.prefix}: fractional residuals")
        fig.tight_layout()
        fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_22_frac_residual_1_minus_yfit_over_y.png"), dpi=self.dpi)
        plt.close(fig)

    def _plot_fit_contours_and_ratio(self):
        m = self.fit_result

        # mge.print_contours(
        #     self.img, self.theta, self.xc, self.yc, m.sol,
        #     scale=self.pixel_scale,
        #     sigmapsf=self.sigmapsf,
        #     normpsf=self.normpsf,
        #     binning=9,
        #     minlevel=self.minlevel
        # )

        half_size_pix = int(max(10, self.contour_half_size_arcsec / self.pixel_scale))
        bounds, model_cut = build_mge_model_image_cutout(
            self.img.shape, m.sol, self.pa, (self.xc, self.yc),
            half_size_pix=half_size_pix,
            oversample=self.contour_oversample,
        )
        x1, x2, y1, y2 = bounds
        data_cut = self.img[y1:y2, x1:x2]
        good_cut = self.goodmask[y1:y2, x1:x2]

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
                    extent=[x1, x2, y1, y2]
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
                fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_23_model_contours_over_data.png"), dpi=self.dpi)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(7, 7))
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = data_cut / model_cut
                ax.imshow(
                    _stretch_for_display(ratio, goodmask=good_cut),
                    origin="lower",
                    extent=[x1, x2, y1, y2]
                )
                ax.plot([self.xc], [self.yc], marker="+", markersize=14)
                ax.set_title(f"{self.prefix}: data/model ratio (cutout)")
                ax.set_xlabel("x (pix)")
                ax.set_ylabel("y (pix)")
                fig.tight_layout()
                fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_24_data_over_model_ratio.png"), dpi=self.dpi)
                plt.close(fig)

    # -------------------------
    # high-level helpers
    # -------------------------
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

        return {
            "center_pix": (self.xc, self.yc),   # (x, y)
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
            "checkplot_dir": self.checkplot_dir,
            "cache_dir": self.cache_dir,
        }

    def save_final_results(self, filename=None):
        if filename is None:
            if self.cache_dir is None:
                raise ValueError("Provide filename or set cache_dir.")
            filename = os.path.join(self.cache_dir, f"{self.prefix}_final_results.pkl")

        with open(filename, "wb") as f:
            pickle.dump(self.results_dict(), f)

        return filename