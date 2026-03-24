import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mgefit as mge


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

        self._manual_center = None    # always public (x, y)
        self._manual_pa = None
        self._manual_eps = None
        self._manual_theta = None

        if center is not None or pa_deg is not None or eps is not None or theta_deg is not None:
            self.set_manual_geometry(center=center, pa_deg=pa_deg, eps=eps, theta_deg=theta_deg)

    # -------------------------
    # conversion helpers
    # -------------------------
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

    # -------------------------
    # manual geometry
    # -------------------------
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
                bulge_disk=self.bulge_disk,
                negative=self.allow_negative
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

    # def _plot_fit_profiles_and_residuals(self):
    #     s = self.sectors_result
    #     m = self.fit_result

    #     r_arc = np.asarray(s.radius) * self.pixel_scale
    #     y_dat = np.asarray(s.counts)
    #     y_fit_pts = mge_model_counts_at_polar_points(s.radius, s.angle, m.sol)

    #     rpos = np.asarray(s.radius)
    #     rpos = rpos[np.isfinite(rpos) & (rpos > 0)]
    #     rmin = max(np.nanmin(rpos), 0.5) if rpos.size else 0.5
    #     rmax = np.nanmax(s.radius)
    #     rgrid = np.geomspace(rmin, rmax, 250)
    #     rgrid_arc = rgrid * self.pixel_scale

    #     y_major = mge_model_counts_at_polar_points(rgrid, np.zeros_like(rgrid), m.sol)
    #     y_minor = mge_model_counts_at_polar_points(rgrid, np.full_like(rgrid, 90.0), m.sol)
    #     y_45 = mge_model_counts_at_polar_points(rgrid, np.full_like(rgrid, 45.0), m.sol)

    #     fig, ax = plt.subplots(figsize=(7.5, 5.5))
    #     ax.scatter(r_arc, y_dat, s=6, alpha=0.35, label="data (all sectors)")
    #     ax.plot(rgrid_arc, y_major, linewidth=1.8, label="model (major axis)")
    #     ax.plot(rgrid_arc, y_minor, linewidth=1.8, label="model (minor axis)")
    #     ax.plot(rgrid_arc, y_45, linewidth=1.2, label="model (45 deg)")
    #     ax.set_xlabel("R (arcsec)")
    #     ax.set_ylabel("counts / pix")

    #     pos = y_dat[np.isfinite(y_dat) & (y_dat > 0)]
    #     if pos.size > 0 and np.nanmin(pos) > 0:
    #         ax.set_yscale("log")
    #         ax.set_xscale("log")

    #     ax.set_title(f"{self.prefix}: radial profile data vs model  (absdev={m.absdev:.4f})")
    #     ax.legend(fontsize=9)
    #     fig.tight_layout()
    #     fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_21_radial_profile_data_vs_model.png"), dpi=self.dpi)
    #     plt.close(fig)

    #     good = np.isfinite(y_dat) & np.isfinite(y_fit_pts) & (y_dat != 0)
    #     frac = np.full_like(y_dat, np.nan, dtype=float)
    #     frac[good] = 1.0 - (y_fit_pts[good] / y_dat[good])

    #     fig, ax = plt.subplots(figsize=(7.5, 4.8))
    #     ax.scatter(r_arc[good], frac[good], s=6, alpha=0.35)
    #     ax.axhline(0.0, linewidth=1.0)
    #     ax.set_xlabel("R (arcsec)")
    #     ax.set_ylabel(r"$1 - y_{\rm fit}/y$")
    #     ax.set_title(f"{self.prefix}: fractional residuals")
    #     fig.tight_layout()
    #     fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_22_frac_residual_1_minus_yfit_over_y.png"), dpi=self.dpi)
    #     plt.close(fig)

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

        # Unique sector angles (rounded a little to avoid float duplication issues)
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

        valid_g = np.isfinite(total_counts) & np.isfinite(sigma_pix) & np.isfinite(q_obs) \
                & (sigma_pix > 0) & (q_obs > 0) & (total_counts > 0)

        total_counts = total_counts[valid_g]
        sigma_pix = sigma_pix[valid_g]
        q_obs = q_obs[valid_g]

        # Sort components by sigma so the family of Gaussians is easier to read
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

            # Total model for this angle
            y_model = mge_model_counts_at_polar_points(
                rgrid, np.full_like(rgrid, ang, dtype=float), m.sol
            )

            # Individual Gaussian components for this angle
            y_comp = component_profiles_at_angle(rgrid, ang)  # (nr, ngauss)

            # Plot data
            ax.scatter(r_this_arc, y_this, s=8, alpha=0.45, label="data")

            # Plot individual Gaussians
            # Keep all components, but make them subtle
            for j in range(y_comp.shape[1]):
                ax.plot(
                    rgrid_arc, y_comp[:, j],
                    linewidth=0.6, alpha=0.25
                )

            # Plot total model
            ax.plot(rgrid_arc, y_model, linewidth=1.8, label="total model")

            ax.set_title(rf"$\theta = {ang:.1f}^\circ$")
            ax.set_xlabel("R (arcsec)")
            ax.set_ylabel("counts / pix")

            good_y = np.concatenate([y_this[np.isfinite(y_this)], y_model[np.isfinite(y_model)]])
            if good_y.size > 0 and np.nanmin(good_y[good_y > 0]) > 0:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylim(np.nanmin(y_model[y_model > 0]) / 5, np.nanmax(y_this)*1.3)
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

    # def _plot_fit_contours_and_ratio(self):
    #     m = self.fit_result

    #     half_size_pix = int(max(10, self.contour_half_size_arcsec / self.pixel_scale))
    #     bounds, model_cut = build_mge_model_image_cutout(
    #         self.img.shape, m.sol, self.pa, (self.xc, self.yc),
    #         half_size_pix=half_size_pix,
    #         oversample=self.contour_oversample,
    #     )
    #     x1, x2, y1, y2 = bounds
    #     data_cut = self.img[y1:y2, x1:x2]
    #     good_cut = self.goodmask[y1:y2, x1:x2]

    #     v = data_cut[good_cut]
    #     v = v[np.isfinite(v)]

    #     if v.size > 0:
    #         levels = np.percentile(v, [70, 80, 90, 95, 97, 99])
    #         levels = np.unique(levels[np.isfinite(levels)])

    #         if levels.size >= 3:
    #             fig, ax = plt.subplots(figsize=(7, 7))
    #             ax.imshow(
    #                 _stretch_for_display(data_cut, goodmask=good_cut),
    #                 origin="lower",
    #                 extent=[x1, x2, y1, y2]
    #             )
    #             ax.contour(
    #                 model_cut,
    #                 levels=levels,
    #                 linewidths=1.0,
    #                 origin="lower",
    #                 extent=[x1, x2, y1, y2]
    #             )
    #             ax.plot([self.xc], [self.yc], marker="+", markersize=14)
    #             ax.set_title(f"{self.prefix}: model contours over data (cutout)")
    #             ax.set_xlabel("x (pix)")
    #             ax.set_ylabel("y (pix)")
    #             fig.tight_layout()
    #             fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_23_model_contours_over_data.png"), dpi=self.dpi)
    #             plt.close(fig)

    #             fig, ax = plt.subplots(figsize=(7, 7))
    #             with np.errstate(divide="ignore", invalid="ignore"):
    #                 ratio = data_cut / model_cut
    #             ax.imshow(
    #                 _stretch_for_display(ratio, goodmask=good_cut),
    #                 origin="lower",
    #                 extent=[x1, x2, y1, y2]
    #             )
    #             ax.plot([self.xc], [self.yc], marker="+", markersize=14)
    #             ax.set_title(f"{self.prefix}: data/model ratio (cutout)")
    #             ax.set_xlabel("x (pix)")
    #             ax.set_ylabel("y (pix)")
    #             fig.tight_layout()
    #             fig.savefig(os.path.join(self.checkplot_dir, f"{self.prefix}_24_data_over_model_ratio.png"), dpi=self.dpi)
    #             plt.close(fig)
    def _plot_fit_contours_and_ratio(self):
        m = self.fit_result

        half_size_pix = int(max(10, self.contour_half_size_arcsec / self.pixel_scale))

        # IMPORTANT:
        # Use the image-plane angle used by the sector extraction / fit, not the sky PA.
        # In many pipelines self.pa is an astronomical PA, while the model-image builder
        # wants the image-frame theta.
        theta = getattr(self, "theta", None)
        if theta is None:
            theta = getattr(self, "theta_deg", None)
        if theta is None:
            theta = self.pa  # fallback only if no theta attribute exists

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

        # Make sure model_cut has the same axis ordering as data_cut: [y, x]
        if model_cut.shape != data_cut.shape:
            if model_cut.T.shape == data_cut.shape:
                model_cut = model_cut.T
            else:
                raise ValueError(
                    f"model_cut shape {model_cut.shape} does not match "
                    f"data_cut shape {data_cut.shape}"
                )
        else:
            # If the cutout is square, shape alone cannot tell whether x/y are swapped.
            # Compare both orientations and keep the one that matches the data better.
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
                    #cmap='Grays'
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
                #ax.plot([self.xc], [self.yc], marker="+", markersize=14)
                ax.set_title(f"{self.prefix}: residual (cutout)")
                ax.set_xlabel("x (pix)")
                ax.set_ylabel("y (pix)")
                fig.tight_layout()
                fig.savefig(
                    os.path.join(self.checkplot_dir, f"{self.prefix}_24_data_over_model_ratio.png"),
                    dpi=self.dpi
                )
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

        return {
            "center_pix": (self.xc, self.yc),   # public convention: (x, y)
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
    
    def plot_deprojected_density_map(
        self,
        inc_deg,
        distance_mpc=None,
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
        inc_deg : float
            Inclination in degrees. 90 = edge-on.
        distance_mpc : float or None
            Distance in Mpc. If given, axes are in pc and density is per pc^3.
            If None, axes stay in pixels and density is in arbitrary/pixel^3 units.
        ml : float
            Mass-to-light ratio. Only applied if mass_density=True.
        half_size_arcsec : float
            Half-size of plotted box in arcsec (or converted to pixels if distance_mpc is None).
        npix : int
            Number of pixels per axis in the output density map.
        save : bool
            Save the figure to checkplot_dir.
        mass_density : bool
            If True, multiply luminosity density by ml.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if self.fit_result is None:
            raise RuntimeError("No fit_result found.")

        # MGE solution from mge_fit_sectors: total_counts, sigma_pix, q_obs
        total_counts, sigma_pix, qobs = self.fit_result.sol
        total_counts = np.asarray(total_counts, dtype=float)
        sigma_pix = np.asarray(sigma_pix, dtype=float)
        qobs = np.asarray(qobs, dtype=float)

        inc = np.deg2rad(inc_deg)
        sini = np.sin(inc)
        cosi = np.cos(inc)

        if np.isclose(sini, 0.0):
            raise ValueError("inc_deg must be > 0 for axisymmetric deprojection.")

        # Oblate axisymmetric intrinsic flattening:
        # qintr^2 = (qobs^2 - cos(i)^2) / sin(i)^2
        qintr2 = (qobs**2 - cosi**2) / (sini**2)

        if np.any(qintr2 <= 0):
            qmin = np.min(qobs)
            inc_min = np.degrees(np.arccos(np.clip(qmin, -1.0, 1.0)))
            raise ValueError(
                f"Chosen inclination {inc_deg:.2f} deg is too low for this MGE. "
                f"Need roughly inc_deg >= {inc_min:.2f} deg."
            )

        qintr = np.sqrt(qintr2)

        # Convert sigma to the working length unit
        sigma_arcsec = sigma_pix * self.pixel_scale

        if distance_mpc is None:
            # Work in pixels
            sigma = sigma_pix
            half_size = half_size_arcsec / self.pixel_scale
            xlab = "R (mirrored) [pix]"
            ylab = "z [pix]"
            unit_label = "arb / pix^3"
        else:
            # Convert arcsec -> pc
            pc_per_arcsec = distance_mpc * 1.0e6 / 206265.0
            sigma = sigma_arcsec * pc_per_arcsec
            half_size = half_size_arcsec * pc_per_arcsec
            xlab = "R (mirrored) [pc]"
            ylab = "z [pc]"
            unit_label = "L / pc^3"

        # Peak projected surface brightness for each Gaussian
        # surf = total_counts / (2*pi*qobs*sigma^2)
        surf = total_counts / (2.0 * np.pi * qobs * sigma**2)

        # Intrinsic central density for each Gaussian
        dens0 = surf * qobs / (qintr * sigma * np.sqrt(2.0 * np.pi))

        if mass_density:
            dens0 = ml * dens0
            unit_label = unit_label.replace("L", "M")

        # Build meridional grid. Mirror R for display symmetry.
        x = np.linspace(-half_size, half_size, npix)
        z = np.linspace(-half_size, half_size, npix)
        xx, zz = np.meshgrid(x, z)
        RR = np.abs(xx)

        rho = np.zeros_like(RR)

        # Sum intrinsic 3D Gaussians:
        # rho_k(R,z) = dens0_k * exp[-(R^2 + z^2/q_k^2)/(2 sigma_k^2)]
        for d0, s, q in zip(dens0, sigma, qintr):
            rho += d0 * np.exp(-0.5 * (RR**2 + (zz**2) / (q**2)) / s**2)

        # Plot log density
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
        ax.set_title(f"{self.prefix}: deprojected axisymmetric density (i={inc_deg:.1f} deg)")

        fig.tight_layout()

        if save:
            out = os.path.join(
                self.checkplot_dir,
                f"{self.prefix}_25_deprojected_density_i{inc_deg:.1f}.png"
            )
            fig.savefig(out, dpi=self.dpi)

        plt.close(fig)

        return {
            "x": x,
            "z": z,
            "rho": rho,
            "qintr": qintr,
            "dens0": dens0,
            "sigma": sigma,
        }

    def compute_enclosed_mass_profile(
        self,
        ml,
        radii_arcsec=None,
        rmax_arcsec=120,
        npts=300,
        flux_unit=u.MJy / u.sr,
        flux_to_lsun=None,
        distance=None,
        inclination_deg=87,
        checkplots_path=None,
        prefix="mge",
        show=False,
        dpi=600,
    ):
        """
        Compute the enclosed mass profile from the fitted MGE.

        By default this returns the *projected* enclosed mass profile M(<R)
        inside circular apertures on the sky.

        If `inclination_deg` is provided, this instead performs an
        axisymmetric-oblate deprojection of the MGE and returns the *3D
        spherical* enclosed mass profile M(<r).

        Parameters
        ----------
        ml : astropy.units.Quantity
            Mass-to-light ratio, with units equivalent to Msun/Lsun.

        radii_arcsec : array-like, optional
            Radii in arcsec where the profile is evaluated.
            If None, an automatic logarithmic grid is generated.

        rmax_arcsec : float, optional
            Maximum radius in arcsec for the automatic grid.

        npts : int, optional
            Number of radius points for the automatic grid.

        flux_unit : astropy.units.Unit, optional
            Surface-brightness unit of the fitted image.
            Default is MJy/sr.

        flux_to_lsun : astropy.units.Quantity, optional
            Conversion factor from integrated flux to luminosity,
            with units equivalent to Lsun / (integrated flux unit).
            Needed unless the integrated MGE is already in luminosity units.

        distance : astropy.units.Quantity, optional
            Distance to the galaxy. Required if `flux_unit` is per physical area
            (e.g. Lsun/pc^2 or Jy/pc^2). If None, tries self.distance or
            self.distance_mpc if available.

        inclination_deg : float, optional
            Inclination in degrees, with 90 = edge-on.
            If provided, compute the deprojected 3D spherical enclosed profile.
            Assumes an oblate axisymmetric deprojection.

        checkplots_path : str, optional
            Directory where the enclosed-mass checkplot will be saved.
            If None, tries self.checkplots_path, otherwise uses ".".

        prefix : str, optional
            Prefix for saved checkplot filename.

        show : bool, optional
            If True, show the plot interactively.

        dpi : int, optional
            DPI of saved figure.

        Returns
        -------
        out : dict
            Dictionary containing the profile and metadata.
        """
        if getattr(self, "fit_result", None) is None or not hasattr(self.fit_result, "sol"):
            raise RuntimeError(
                "run_fit() must be executed before compute_enclosed_mass_profile()."
            )

        ml = u.Quantity(ml)
        if not ml.unit.is_equivalent(u.Msun / u.Lsun):
            raise u.UnitConversionError("`ml` must have units equivalent to Msun/Lsun.")
        ml = ml.to(u.Msun / u.Lsun)

        sol = np.asarray(self.fit_result.sol, dtype=float)
        if sol.ndim != 2 or sol.shape[0] < 3:
            raise ValueError(
                "fit_result.sol must contain at least [total_counts, sigma_pixels, q_obs]."
            )

        total_counts = np.asarray(sol[0], dtype=float)
        sigma_pix = np.asarray(sol[1], dtype=float)
        q_obs = np.asarray(sol[2], dtype=float)

        if np.any(sigma_pix <= 0):
            raise ValueError("All MGE sigma values must be positive.")

        q_obs = np.clip(q_obs, 1e-6, 1.0)
        sigma_arcsec = sigma_pix * float(self.pixel_scale)

        if radii_arcsec is None:
            if rmax_arcsec is None:
                rmax_arcsec = 10.0 * np.max(sigma_arcsec)
            rmin_arcsec = max(0.25 * float(self.pixel_scale), 1e-4 * np.max(sigma_arcsec))
            radii_arcsec = np.geomspace(rmin_arcsec, rmax_arcsec, int(npts))
        else:
            radii_arcsec = np.asarray(radii_arcsec, dtype=float)
            if radii_arcsec.ndim != 1 or radii_arcsec.size == 0:
                raise ValueError("`radii_arcsec` must be a non-empty 1D array.")
            if np.any(radii_arcsec < 0):
                raise ValueError("`radii_arcsec` must be non-negative.")
            if np.any(np.diff(radii_arcsec) <= 0):
                raise ValueError("`radii_arcsec` must be strictly increasing.")

        sb_unit = u.Unit(flux_unit)
        pix_scale = float(self.pixel_scale) * u.arcsec

        if distance is None:
            if hasattr(self, "distance") and self.distance is not None:
                distance = self.distance
            elif hasattr(self, "distance_mpc") and self.distance_mpc is not None:
                distance = self.distance_mpc if isinstance(self.distance_mpc, u.Quantity) else self.distance_mpc * u.Mpc

        # Convert one pixel into the area corresponding to the denominator of flux_unit
        if sb_unit.is_equivalent(u.Jy / u.sr):
            pixel_area = (pix_scale.to(u.rad).value ** 2) * u.sr

        elif sb_unit.is_equivalent(u.Jy / u.arcsec**2) or sb_unit.is_equivalent(u.Lsun / u.arcsec**2):
            pixel_area = pix_scale**2

        elif sb_unit.is_equivalent(u.Jy / u.pc**2) or sb_unit.is_equivalent(u.Lsun / u.pc**2):
            if distance is None:
                raise ValueError(
                    "`distance` is required when `flux_unit` is per physical area (e.g. pc^2)."
                )
            distance = u.Quantity(distance).to(u.pc)
            pixel_size = distance * pix_scale.to(u.rad).value
            pixel_area = pixel_size**2

        else:
            raise ValueError(
                "Unsupported `flux_unit`. Use a surface-brightness unit like "
                "MJy/sr, Jy/arcsec^2, Lsun/arcsec^2, or Lsun/pc^2."
            )

        # total_counts are in (image unit) * pixel^2
        component_flux = total_counts * sb_unit * pixel_area

        # Convert integrated flux to luminosity
        if component_flux.unit.is_equivalent(u.Lsun):
            component_lum = component_flux.to(u.Lsun)
        else:
            if flux_to_lsun is None:
                raise ValueError(
                    "With the current `flux_unit`, the integrated MGE is not in luminosity units. "
                    "Pass `flux_to_lsun` with units equivalent to "
                    f"Lsun / ({component_flux.unit})."
                )
            flux_to_lsun = u.Quantity(flux_to_lsun)
            target_unit = u.Lsun / component_flux.unit
            if not flux_to_lsun.unit.is_equivalent(target_unit):
                raise u.UnitConversionError(
                    f"`flux_to_lsun` must have units equivalent to {target_unit}."
                )
            component_lum = (component_flux * flux_to_lsun).to(u.Lsun)

        prepend_zero = radii_arcsec[0] > 0.0
        if prepend_zero:
            r_eval = np.concatenate(([0.0], radii_arcsec))
        else:
            r_eval = radii_arcsec.copy()

        rr = r_eval[:, None]                # (nr, 1)
        ss = sigma_arcsec[None, :]          # (1, ngauss)

        if inclination_deg is None:
            profile_type = "projected"
            qq = q_obs[None, :]

            t = rr**2 / (2.0 * ss**2)
            beta = 0.5 * (1.0 / qq**2 - 1.0)

            # dL/dR for observed 2D elliptical Gaussian inside circular aperture
            kernel = (rr / (ss**2 * qq)) * np.exp(-t) * i0e(beta * t)

            dLdr_val = np.sum(component_lum.value[None, :] * kernel, axis=1)
            dLdr = dLdr_val * component_lum.unit / u.arcsec

            intrinsic_q = None
            inclination_used = None

        else:
            profile_type = "spherical_deprojected"
            inclination_deg = float(inclination_deg)

            if not (0.0 < inclination_deg <= 90.0):
                raise ValueError("`inclination_deg` must satisfy 0 < inclination_deg <= 90.")

            inc_rad = np.deg2rad(inclination_deg)
            sini = np.sin(inc_rad)
            cosi = np.cos(inc_rad)

            # For oblate axisymmetric deprojection: q_obs^2 = cos^2 i + q_intr^2 sin^2 i
            if np.any(q_obs < cosi - 1e-10):
                min_allowed = np.degrees(np.arccos(np.min(q_obs)))
                raise ValueError(
                    f"This MGE cannot be deprojected at inclination {inclination_deg:.2f} deg "
                    f"under the oblate-axisymmetric assumption. "
                    f"A valid inclination must satisfy i >= {min_allowed:.2f} deg."
                )

            q_intr = np.sqrt(np.maximum(q_obs**2 - cosi**2, 0.0)) / sini
            q_intr = np.clip(q_intr, 1e-6, 1.0)
            qq = q_intr[None, :]

            # 3D density:
            # rho(R,z) = L / [ (sqrt(2pi) sigma)^3 q ] * exp[-(R^2 + z^2/q^2)/(2 sigma^2)]
            #
            # Spherical enclosed mass derivative:
            # dL/dr = ∮ rho(r,theta) r^2 dOmega
            A = rr**2 / (2.0 * ss**2)
            C = A * (1.0 / qq**2 - 1.0)

            C_safe = np.where(C > 0.0, C, 1.0)
            ang_int = np.where(
                C < 1e-12,
                2.0 * np.exp(-A),
                np.exp(-A) * np.sqrt(np.pi / C_safe) * erf(np.sqrt(C_safe))
            )

            kernel = (rr**2 / (np.sqrt(2.0 * np.pi) * ss**3 * qq)) * ang_int

            dLdr_val = np.sum(component_lum.value[None, :] * kernel, axis=1)
            dLdr = dLdr_val * component_lum.unit / u.arcsec

            intrinsic_q = q_intr
            inclination_used = inclination_deg

        enclosed_lum_val = np.zeros_like(r_eval, dtype=float)
        enclosed_lum_val[1:] = np.cumsum(
            0.5 * (dLdr_val[1:] + dLdr_val[:-1]) * np.diff(r_eval)
        )

        enclosed_lum = enclosed_lum_val * component_lum.unit
        enclosed_mass = (enclosed_lum * ml).to(u.Msun)
        dMdr = (dLdr * ml).to(u.Msun / u.arcsec)

        if prepend_zero:
            enclosed_lum = enclosed_lum[1:]
            enclosed_mass = enclosed_mass[1:]
            dLdr = dLdr[1:]
            dMdr = dMdr[1:]

        total_flux = np.sum(component_flux)
        total_lum = np.sum(component_lum)
        total_mass = (total_lum * ml).to(u.Msun)
        enclosed_fraction = (enclosed_lum / total_lum).to_value(u.dimensionless_unscaled)

        # Save checkplot
        if checkplots_path is None:
            checkplots_path = getattr(self, "checkplots_path", ".")
        os.makedirs(checkplots_path, exist_ok=True)

        if profile_type == "projected":
            plot_name = f"{prefix}_enclosed_mass_profile_projected.png"
            title = "Projected enclosed mass profile"
            xlabel = r"$R$ [arcsec]"
        else:
            plot_name = f"{prefix}_enclosed_mass_profile_3d.png"
            title = f"3D spherical enclosed mass profile (i={inclination_deg:.1f} deg)"
            xlabel = r"$r$ [arcsec]"

        plot_path = os.path.join(checkplots_path, plot_name)

        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        ax.plot(radii_arcsec, enclosed_mass.to_value(u.Msun), lw=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Enclosed mass [$M_\odot$]")
        ax.set_title(title)

        if np.all(radii_arcsec > 0):
            ax.set_xscale("log")
        if np.all(enclosed_mass.to_value(u.Msun) > 0):
            ax.set_yscale("log")

        txt = (
            f"Total mass = {total_mass.to_value(u.Msun):.3e} Msun\n"
            f"M/L = {ml.to_value(u.Msun/u.Lsun):.3g} Msun/Lsun"
        )
        ax.text(
            0.04, 0.96, txt,
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return {
            "profile_type": profile_type,
            "radii_arcsec": radii_arcsec * u.arcsec,
            "enclosed_luminosity": enclosed_lum,
            "enclosed_mass": enclosed_mass,
            "dLdr": dLdr,
            "dMdr": dMdr,
            "total_flux": total_flux,
            "total_luminosity": total_lum,
            "total_mass": total_mass,
            "enclosed_fraction": enclosed_fraction,
            "inclination_deg": inclination_used,
            "intrinsic_q": intrinsic_q,
            "checkplot_path": plot_path,
        }