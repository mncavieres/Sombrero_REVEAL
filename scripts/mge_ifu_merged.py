import os
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
    Evaluate the *surface brightness* model (counts/pix) at polar points (radius, angle),
    where angle is measured from the major axis (as in mgefit sector photometry).
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

def polar_points_to_image_xy(radius_pix, angle_deg, pa_deg, center_pix_rc):
    """
    Convert (radius, angle from major axis) -> image (x,y) pixels for overlay.
    center_pix_rc = (row, col) = (xc, yc) from mgefit.
    Returns x=col, y=row for imshow(origin='lower').
    """
    xc_row, yc_col = center_pix_rc
    x0 = float(yc_col)
    y0 = float(xc_row)

    r = np.asarray(radius_pix, float)
    th = np.deg2rad(np.asarray(angle_deg, float))
    xp = r * np.cos(th)
    yp = r * np.sin(th)

    pa = np.deg2rad(pa_deg)
    x_img = x0 + xp * np.sin(pa) + yp * np.cos(pa)
    y_img = y0 + xp * np.cos(pa) - yp * np.sin(pa)
    return x_img, y_img

def build_mge_model_image_cutout(img_shape, sol, pa_deg, center_pix_rc,
                                 half_size_pix=400, oversample=1):
    """
    Build a model image cutout (counts/pix) from the MGE solution.
    NOTE: If you supplied a PSF to fit_sectors, sol is deconvolved.
    """
    ny, nx = img_shape
    xc_row, yc_col = center_pix_rc
    x0 = float(yc_col)
    y0 = float(xc_row)

    x1 = int(max(0, np.floor(x0 - half_size_pix)))
    x2 = int(min(nx, np.ceil(x0 + half_size_pix)))
    y1 = int(max(0, np.floor(y0 - half_size_pix)))
    y2 = int(min(ny, np.ceil(y0 + half_size_pix)))

    xs = np.arange(x1, x2, 1/oversample)
    ys = np.arange(y1, y2, 1/oversample)
    X, Y = np.meshgrid(xs, ys)

    pa = np.deg2rad(pa_deg)
    dx = X - x0
    dy = Y - y0
    xp =  dx * np.sin(pa) + dy * np.cos(pa)
    yp =  dx * np.cos(pa) - dy * np.sin(pa)

    total_counts, sigma, q = sol
    total_counts = np.asarray(total_counts, float)
    sigma = np.asarray(sigma, float)
    q = np.asarray(q, float)
    surf = total_counts / (2.0 * np.pi * sigma**2 * q)

    model = np.zeros_like(X, dtype=float)
    for s0, sj, qj in zip(surf, sigma, q):
        expo = -0.5 * (xp**2 + (yp**2) / (qj*qj)) / (sj*sj)
        model += s0 * np.exp(expo)

    if oversample > 1:
        oy = (y2 - y1) * oversample
        ox = (x2 - x1) * oversample
        model = model[:oy, :ox]
        model = model.reshape((y2 - y1), oversample, (x2 - x1), oversample).mean(axis=(1, 3))

    return (x1, x2, y1, y2), model


# ----------------------------
# Main fitting function (now saves checkplots "as it goes")
# ----------------------------
def fit_mge_from_f200(
    img_f200,
    dust_mask,
    *,
    pixel_scale=1.0,                 # arcsec/pix
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
    # NEW: checkplot saving
    checkplot_dir=None,              # if None -> no files saved
    prefix="f200",
    dpi=300,
    max_points_overlay=200000,
    contour_half_size_arcsec=80,
    contour_oversample=1,
):
    """
    Fit an MGE model to a 2D F200 image, excluding dusty pixels via a mask,
    and save checkplots immediately after each stage.

    If checkplot_dir is not None:
      - saves native mgefit diagnostic figs for find_galaxy / sectors_photometry / fit_sectors (when plot=True)
      - saves custom overlays + radial profile + residuals + contours
    """
    img = np.asarray(img_f200, dtype=float)
    if img.ndim != 2:
        raise ValueError("img_f200 must be a 2D array")

    mask = np.asarray(dust_mask)
    if mask.shape != img.shape:
        raise ValueError("dust_mask must have the same shape as img_f200")
    if mask.dtype != bool:
        mask = mask.astype(bool)

    goodmask = (~mask) if dust_mask_is_bad else mask

    # Fill masked pixels for find_galaxy stability
    img_for_find = img.copy()
    fill = np.nanmedian(img[goodmask]) if np.any(goodmask) else np.nanmedian(img)
    img_for_find[~goodmask] = fill
    if not np.all(np.isfinite(img_for_find)):
        raise ValueError("Image contains NaN/Inf even after filling masked pixels")

    # Optional sky subtraction
    sky_mean = None
    sky_sigma = None
    img_work = img.copy()
    if subtract_sky:
        sky_mean, sky_sigma = mge.sky_level(img_for_find[goodmask], plot=plot, quiet=quiet)
        img_work = img_work - sky_mean
        img_for_find = img_for_find - sky_mean

    # Set up output directory early
    if checkplot_dir is not None:
        _ensure_dir(checkplot_dir)

    # -------------------------
    # 1) find_galaxy + save plots right away
    # -------------------------
    def _do_find():
        return mge.find_galaxy(
            img_for_find,
            fraction=find_fraction,
            binning=find_binning,
            plot=plot,
            quiet=quiet
        )

    f, new_figs = _capture_new_figures(_do_find)

    if checkplot_dir is not None:
        # Save native find_galaxy(plot=True) diagnostic(s)
        if plot and len(new_figs) > 0:
            paths = [
                os.path.join(checkplot_dir, f"{prefix}_00_find_galaxy_diagnostic_{i+1:02d}.png")
                for i in range(len(new_figs))
            ]
            _save_and_close_figs(new_figs, paths, dpi=dpi)

        # Custom geometry overlay
        xc, yc = f.xpeak, f.ypeak  # (row, col)
        pa = float(f.pa)
        eps = float(f.eps)
        qbar = 1.0 - eps

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(_stretch_for_display(img, goodmask=goodmask), origin="lower")
        try:
            ax.contour((~goodmask).astype(float), levels=[0.5], linewidths=0.8, alpha=0.8, origin="lower")
        except Exception:
            pass
        ax.plot([yc], [xc], marker="+", markersize=14)

        L_arcsec = 20.0
        L_pix = L_arcsec / pixel_scale
        pa_rad = np.deg2rad(pa)
        dx = L_pix * np.sin(pa_rad)
        dy = L_pix * np.cos(pa_rad)
        ax.plot([yc - dx, yc + dx], [xc - dy, xc + dy], linewidth=1.2)

        for a_arc in [5.0, 15.0, 30.0]:
            a_pix = a_arc / pixel_scale
            b_pix = a_pix * qbar
            e = Ellipse((yc, xc), width=2*a_pix, height=2*b_pix, angle=(90.0 - pa),
                        fill=False, linewidth=1.0)
            ax.add_patch(e)

        ax.set_title(f"{prefix}: geometry overlay  (PA={pa:.2f} deg, eps={eps:.3f})")
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
        fig.tight_layout()
        fig.savefig(os.path.join(checkplot_dir, f"{prefix}_01_geometry_overlay.png"), dpi=dpi)
        plt.close(fig)

    # Store geometry
    xc, yc = f.xpeak, f.ypeak
    eps = f.eps
    pa = f.pa

    # -------------------------
    # 2) sectors_photometry + save plots right away
    # -------------------------
    def _do_sectors():
        return mge.sectors_photometry(
            img_work,
            eps=eps,
            ang=pa,
            xc=xc,
            yc=yc,
            mask=goodmask,
            n_sectors=n_sectors,
            minlevel=minlevel,
            plot=plot
        )

    s, new_figs = _capture_new_figures(_do_sectors)

    if checkplot_dir is not None:
        # Save native sectors_photometry(plot=True) diagnostic(s)
        if plot and len(new_figs) > 0:
            paths = [
                os.path.join(checkplot_dir, f"{prefix}_10_sectors_photometry_{i+1:02d}.png")
                for i in range(len(new_figs))
            ]
            _save_and_close_figs(new_figs, paths, dpi=dpi)

        # Custom: sampled points overlay
        x_img, y_img = polar_points_to_image_xy(s.radius, s.angle, float(pa), (xc, yc))
        npts = x_img.size
        if npts > max_points_overlay:
            rng = np.random.default_rng(12345)
            idx = rng.choice(npts, size=max_points_overlay, replace=False)
        else:
            idx = slice(None)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(_stretch_for_display(img, goodmask=goodmask), origin="lower")
        try:
            ax.contour((~goodmask).astype(float), levels=[0.5], linewidths=0.8, alpha=0.8, origin="lower", colors='orange')
        except Exception:
            pass
        ax.scatter(x_img[idx], y_img[idx], s=1, alpha=0.4, color='orange')
        ax.plot([yc], [xc], marker="+", markersize=14)
        ax.set_title(f"{prefix}: sampled photometry points (sectors)")
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
        fig.tight_layout()
        fig.savefig(os.path.join(checkplot_dir, f"{prefix}_11_sectors_sampled_points.png"), dpi=dpi)
        plt.close(fig)

        # Custom: sector-by-sector radial profiles (data only)
        r_arc = np.asarray(s.radius) * pixel_scale
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

        if np.nanmin(y_dat[y_dat > 0]) > 0:
            ax.set_yscale("log")
        ax.set_xlabel("R (arcsec)")
        ax.set_ylabel("counts / pix")
        ax.set_title(f"{prefix}: sector radial profiles (data)")
        fig.tight_layout()
        fig.savefig(os.path.join(checkplot_dir, f"{prefix}_12_sector_profiles.pdf"), dpi=dpi)
        plt.close(fig)

    # -------------------------
    # 3) fit_sectors + save plots right away
    # -------------------------
    def _do_fit():
        return mge.fit_sectors(
            s.radius, s.angle, s.counts, eps,
            linear=linear,
            ngauss=ngauss,
            qbounds=qbounds,
            outer_slope=outer_slope,
            sigmapsf=sigmapsf,
            normpsf=normpsf,
            scale=pixel_scale,
            plot=plot,
            quiet=quiet,
            bulge_disk=True,
        )

    m, new_figs = _capture_new_figures(_do_fit)

    if checkplot_dir is not None:
        # Save native fit_sectors(plot=True) diagnostic(s)
        if plot and len(new_figs) > 0:
            paths = [
                os.path.join(checkplot_dir, f"{prefix}_20_fit_sectors_{i+1:02d}.png")
                for i in range(len(new_figs))
            ]
            _save_and_close_figs(new_figs, paths, dpi=dpi)

        # Custom: radial profile data vs model (arcsec)
        r_arc = np.asarray(s.radius) * pixel_scale
        y_dat = np.asarray(s.counts)
        y_fit_pts = mge_model_counts_at_polar_points(s.radius, s.angle, m.sol)

        rpos = np.asarray(s.radius)
        rpos = rpos[np.isfinite(rpos) & (rpos > 0)]
        rmin = max(np.nanmin(rpos), 0.5) if rpos.size else 0.5
        rmax = np.nanmax(s.radius)
        rgrid = np.geomspace(rmin, rmax, 250)
        rgrid_arc = rgrid * pixel_scale

        y_major = mge_model_counts_at_polar_points(rgrid, np.zeros_like(rgrid), m.sol)
        y_minor = mge_model_counts_at_polar_points(rgrid, np.full_like(rgrid, 90.0), m.sol)
        y_45    = mge_model_counts_at_polar_points(rgrid, np.full_like(rgrid, 45.0), m.sol)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.scatter(r_arc, y_dat, s=6, alpha=0.35, label="data (all sectors)")
        ax.plot(rgrid_arc, y_major, linewidth=1.8, label="model (major axis)")
        ax.plot(rgrid_arc, y_minor, linewidth=1.8, label="model (minor axis)")
        ax.plot(rgrid_arc, y_45,    linewidth=1.2, label="model (45 deg)")
        ax.set_xlabel("R (arcsec)")
        ax.set_ylabel("counts / pix")
        if np.nanmin(y_dat[y_dat > 0]) > 0:
            ax.set_yscale("log")
        ax.set_title(f"{prefix}: radial profile data vs model  (absdev={m.absdev:.4f})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(checkplot_dir, f"{prefix}_21_radial_profile_data_vs_model.png"), dpi=dpi)
        plt.close(fig)

        # Custom: 1 - yfit/y vs radius
        good = np.isfinite(y_dat) & np.isfinite(y_fit_pts) & (y_dat != 0)
        frac = np.full_like(y_dat, np.nan, dtype=float)
        frac[good] = 1.0 - (y_fit_pts[good] / y_dat[good])

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.scatter(r_arc[good], frac[good], s=6, alpha=0.35)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("R (arcsec)")
        ax.set_ylabel(r"$1 - y_{\rm fit}/y$")
        ax.set_title(f"{prefix}: fractional residuals")
        fig.tight_layout()
        fig.savefig(os.path.join(checkplot_dir, f"{prefix}_22_frac_residual_1_minus_yfit_over_y.png"), dpi=dpi)
        plt.close(fig)

        # Custom: model contours over data + data/model ratio (cutout)
        half_size_pix = int(max(10, contour_half_size_arcsec / pixel_scale))
        bounds, model_cut = build_mge_model_image_cutout(
            img.shape, m.sol, float(pa), (xc, yc),
            half_size_pix=half_size_pix,
            oversample=contour_oversample,
        )
        x1, x2, y1, y2 = bounds
        data_cut = img[y1:y2, x1:x2]
        good_cut = goodmask[y1:y2, x1:x2]

        v = data_cut[good_cut]
        v = v[np.isfinite(v)]
        if v.size > 0:
            levels = np.percentile(v, [70, 80, 90, 95, 97, 99])
            levels = np.unique(levels[np.isfinite(levels)])
            if levels.size >= 3:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.imshow(_stretch_for_display(data_cut, goodmask=good_cut), origin="lower",
                          extent=[x1, x2, y1, y2])
                ax.contour(model_cut, levels=levels, linewidths=1.0,
                           origin="lower", extent=[x1, x2, y1, y2])
                ax.plot([yc], [xc], marker="+", markersize=14)
                ax.set_title(f"{prefix}: model contours over data (cutout)")
                ax.set_xlabel("x (pix)")
                ax.set_ylabel("y (pix)")
                fig.tight_layout()
                fig.savefig(os.path.join(checkplot_dir, f"{prefix}_23_model_contours_over_data.png"), dpi=dpi)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(7, 7))
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = data_cut / model_cut
                ax.imshow(_stretch_for_display(ratio, goodmask=good_cut), origin="lower",
                          extent=[x1, x2, y1, y2])
                ax.plot([yc], [xc], marker="+", markersize=14)
                ax.set_title(f"{prefix}: data/model ratio (cutout)")
                ax.set_xlabel("x (pix)")
                ax.set_ylabel("y (pix)")
                fig.tight_layout()
                fig.savefig(os.path.join(checkplot_dir, f"{prefix}_24_data_over_model_ratio.png"), dpi=dpi)
                plt.close(fig)

    # -------------------------
    # Package outputs
    # -------------------------
    total_counts, sigma_pix, q_obs = m.sol
    surf = total_counts / (2.0 * np.pi * q_obs * sigma_pix**2)
    sigma_arcsec = sigma_pix * float(pixel_scale)
    table = np.vstack([surf, sigma_pix, sigma_arcsec, q_obs, total_counts]).T

    return {
        "center_pix": (xc, yc),
        "pa_deg": pa,
        "eps": eps,
        "sky_mean": sky_mean,
        "sky_sigma": sky_sigma,
        "sectors": s,
        "mgefit": m,
        "table_cols": ["surf_counts_per_pix", "sigma_pix", "sigma_arcsec", "q_obs", "total_counts"],
        "table": table,
        "checkplot_dir": checkplot_dir,
    }

if __name__ == "__main__":
    img_f200 = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/photometry/f200w_ifu_coadd_masked.fits')[0].data
    dust_mask = fits.open('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/dust_mask/f200_mask_1.fits')[0].data

    nan_mask = np.isnan(img_f200)
    if np.any(nan_mask):
        print(f"Found {np.sum(nan_mask)} NaN pixels in the image. Replacing with 0 and adding to dust mask.")
        img_f200[nan_mask] = 0.0
        dust_mask = dust_mask | nan_mask

    checkplot_dir = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_merged_ifu"

    res = fit_mge_from_f200(
        img_f200, dust_mask,
        pixel_scale=0.031,
        subtract_sky=False,
        linear=False,
        ngauss=24,
        plot=True,
        checkplot_dir=checkplot_dir,
        prefix="sombrero_f200",
        contour_half_size_arcsec=80,
        contour_oversample=1,
    )

    print("Center (pix):", res["center_pix"])
    print("PA (deg):", res["pa_deg"], "eps:", res["eps"])
    print("absdev:", res["mgefit"].absdev)
    print(res["table_cols"])
    print(res["table"])

    import pickle
    with open(os.path.join(checkplot_dir, 'mge_fit_results.pkl'), 'wb') as f:
        pickle.dump(res, f)