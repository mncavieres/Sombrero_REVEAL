import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/scripts/mge_class')
from mge_gen2 import MGEFitter
from astropy.io import fits
import numpy as np


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def brightest_pixel_near(img, x0, y0, halfsize=50, goodmask=None):
    """
    Return brightest pixel near an initial guess.

    Public convention:
        input/output are (x, y) = (col, row)

    NumPy indexing:
        img[y, x]
    """
    ny, nx = img.shape

    x1 = max(0, int(round(x0 - halfsize)))
    x2 = min(nx, int(round(x0 + halfsize + 1)))
    y1 = max(0, int(round(y0 - halfsize)))
    y2 = min(ny, int(round(y0 + halfsize + 1)))

    cut = np.array(img[y1:y2, x1:x2], copy=True)

    if goodmask is not None:
        gm = goodmask[y1:y2, x1:x2]
        cut[~gm] = -np.inf

    iy, ix = np.unravel_index(np.nanargmax(cut), cut.shape)
    return x1 + ix, y1 + iy



def central_los_profile_aperture(
        runner,
        aperture_arcsec,
        incl_deg=None,
        frac=0.90,
        n_s=2000,
        n_y=401,
        max_s_arcsec=None,
        sigmas_are_arcsec=False,
        ml=1.0,
    ):
    """
    Compute the LOS mass profile through the galaxy center, but integrated over
    a circular projected aperture of radius `aperture_arcsec`.

    This assumes an axisymmetric oblate deprojection of the runner MGE.

    Parameters
    ----------
    runner : object
        Must contain:
            runner.fit_result.sol = [total_counts, sigma, q_obs]
        and, if sigmas_are_arcsec=False:
            runner.pixel_scale   # arcsec/pixel

    aperture_arcsec : float
        Radius of the circular aperture on the sky, centered on the fitted galaxy center.

    incl_deg : float, optional
        Inclination in degrees. If None, tries:
            runner.incl_deg
            runner.inclination_deg
            runner.inclination

    frac : float, optional
        Enclosed LOS fraction to report. Default 0.90.

    n_s : int, optional
        Number of LOS grid points on s >= 0.

    n_y : int, optional
        Number of integration points along projected y inside the aperture.
        Odd values are nice so y=0 is included.

    max_s_arcsec : float, optional
        Maximum LOS half-range to integrate to. If None, set automatically.

    sigmas_are_arcsec : bool, optional
        If False, assumes MGE sigmas are in pixels and converts using runner.pixel_scale.
        If True, assumes MGE sigmas are already in arcsec.

    ml : float, optional
        Multiplicative mass-to-light ratio scaling. Affects normalization only,
        not the fractional radii such as s90.

    Returns
    -------
    result : dict
        Keys:
            s_arcsec              : symmetric LOS coordinates [-max_s, +max_s]
            dMds                  : mass-per-arcsec along LOS within aperture
            mean_rho_in_aperture  : dMds / aperture_area
            ds_arcsec             : LOS spacing
            mass_per_bin          : approximate mass in each LOS bin
            S_arcsec              : non-negative LOS radius array
            cumfrac               : enclosed LOS mass fraction within |s| < S
            s_frac_arcsec         : LOS half-length enclosing `frac`
            s50_arcsec            : LOS half-length enclosing 50%
            s90_arcsec            : LOS half-length enclosing 90%
            s95_arcsec            : LOS half-length enclosing 95%
            aperture_arcsec       : input aperture radius
            qintr                 : intrinsic axial ratios
            sigma_arcsec          : Gaussian sigmas in arcsec
    """
    try:
        from scipy.special import erf
    except Exception:
        from math import erf as _erf
        erf = np.vectorize(_erf)

    if aperture_arcsec <= 0:
        raise ValueError("aperture_arcsec must be > 0")

    # ---------------------------
    # Inclination
    # ---------------------------
    if incl_deg is None:
        for name in ("incl_deg", "inclination_deg", "inclination"):
            if hasattr(runner, name):
                incl_deg = float(getattr(runner, name))
                break
        else:
            raise ValueError("No inclination found on runner. Pass incl_deg explicitly.")

    inc = np.deg2rad(float(incl_deg))
    sini = np.sin(inc)
    cosi = np.cos(inc)

    if np.isclose(sini, 0.0):
        raise ValueError("Face-on deprojection is non-unique; inclination must be > 0.")

    # ---------------------------
    # Read MGE solution
    # ---------------------------
    if not hasattr(runner, "fit_result") or runner.fit_result is None:
        raise ValueError("runner.fit_result is missing.")

    sol = np.asarray(runner.fit_result.sol, dtype=float)
    if sol.shape[0] != 3:
        raise ValueError(
            "Expected runner.fit_result.sol to have shape (3, N): "
            "[total_counts, sigma, q_obs]"
        )

    L = sol[0].copy() * ml
    sigma = sol[1].copy()
    qobs = sol[2].copy()

    if not sigmas_are_arcsec:
        if not hasattr(runner, "pixel_scale"):
            raise ValueError(
                "runner.pixel_scale is required when sigmas_are_arcsec=False."
            )
        sigma = sigma * float(runner.pixel_scale)

    # ---------------------------
    # Oblate axisymmetric deprojection
    # q_intr^2 = (q_obs^2 - cos(i)^2) / sin(i)^2
    # ---------------------------
    qintr2 = (qobs**2 - cosi**2) / (sini**2)

    if np.any(qintr2 < -1e-12):
        raise ValueError(
            f"Inclination {incl_deg:.3f} deg is too low for this MGE. "
            f"Need q_obs >= cos(i) = {cosi:.5f}, but min(q_obs) = {qobs.min():.5f}."
        )

    qintr2 = np.clip(qintr2, 0.0, None)
    qintr = np.sqrt(qintr2)

    if np.any(qintr < 1e-10):
        raise ValueError(
            "At least one Gaussian deprojects to q_intr ~ 0, making the 3D density singular."
        )

    # Characteristic LOS width for the pencil-beam central LOS
    a_los = sigma * qintr / qobs

    if max_s_arcsec is None:
        max_s_arcsec = 8.0 * np.max(a_los) + aperture_arcsec

    # Only compute s >= 0, then mirror
    S = np.linspace(0.0, max_s_arcsec, int(n_s))
    y = np.linspace(-aperture_arcsec, aperture_arcsec, int(n_y))

    # x-integral through circular aperture boundary for each y:
    # integral_{-sqrt(R^2-y^2)}^{+sqrt(R^2-y^2)} exp(-x^2/(2 sigma^2)) dx
    y2 = y**2
    xmax = np.sqrt(np.clip(aperture_arcsec**2 - y2, 0.0, None))

    # Total dM/ds inside the circular aperture
    dMds_pos = np.zeros_like(S)

    # Loop over Gaussians
    for Lj, sigj, qj, qoj in zip(L, sigma, qintr, qobs):
        amp = Lj / (((2.0 * np.pi) ** 1.5) * sigj**3 * qj)

        # Density in observer coordinates:
        # exponent = (x'^2 + A y'^2 + 2 B y' s + C s^2)/(2 sigma^2)
        A = cosi**2 + (sini**2) / (qj**2)
        B = sini * cosi * (1.0 / (qj**2) - 1.0)
        C = sini**2 + (cosi**2) / (qj**2)

        # Integrate x analytically inside the circle
        xint = np.sqrt(2.0 * np.pi) * sigj * erf(xmax / (np.sqrt(2.0) * sigj))

        # Build 2D exponent on (y, S)
        expo = -(
            A * y[:, None] ** 2
            + 2.0 * B * y[:, None] * S[None, :]
            + C * S[None, :] ** 2
        ) / (2.0 * sigj**2)

        integrand_y = amp * np.exp(expo) * xint[:, None]

        dMds_pos += np.trapz(integrand_y, y, axis=0)

    # Mirror to negative s
    s_full = np.concatenate((-S[:0:-1], S))
    dMds_full = np.concatenate((dMds_pos[:0:-1], dMds_pos))

    ds = S[1] - S[0]
    area_ap = np.pi * aperture_arcsec**2
    mean_rho_in_aperture = dMds_full / area_ap
    mass_per_bin = dMds_full * ds

    # Cumulative enclosed fraction within |s| < S
    # Since profile is even:
    cum_pos = np.zeros_like(S)
    if len(S) > 1:
        cum_pos[1:] = 2.0 * np.cumsum(0.5 * (dMds_pos[:-1] + dMds_pos[1:]) * np.diff(S))

    total_mass_los = cum_pos[-1]
    if total_mass_los <= 0:
        raise RuntimeError("Computed non-positive total LOS mass.")

    cumfrac = cum_pos / total_mass_los

    def radius_at_fraction(f):
        return np.interp(float(f), cumfrac, S)

    s50 = radius_at_fraction(0.50)
    s90 = radius_at_fraction(0.90)
    s95 = radius_at_fraction(0.95)
    sfrac = radius_at_fraction(frac)

    return {
        "s_arcsec": s_full,
        "dMds": dMds_full,
        "mean_rho_in_aperture": mean_rho_in_aperture,
        "ds_arcsec": ds,
        "mass_per_bin": mass_per_bin,
        "S_arcsec": S,
        "cumfrac": cumfrac,
        "s_frac_arcsec": sfrac,
        "s50_arcsec": s50,
        "s90_arcsec": s90,
        "s95_arcsec": s95,
        "aperture_arcsec": aperture_arcsec,
        "qintr": qintr,
        "sigma_arcsec": sigma,
        "incl_deg": incl_deg,
        "frac": frac,
    }


if __name__ == "__main__":

    img_f200 = fits.open(
        '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/IFU/photometry/f200w_ifu_coadd_masked.fits'
    )[0].data

    dust_mask = fits.open(
        '/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/dust_mask/f200_mask_1.fits'
    )[0].data

    nan_mask = np.isnan(img_f200)
    if np.any(nan_mask):
        print(f"Found {np.sum(nan_mask)} NaN pixels in the image. Replacing with 0 and adding to dust mask.")
        img_f200[nan_mask] = 0.0
        dust_mask = dust_mask | nan_mask

    checkplot_dir = "/Users/mncavieres/Documents/2026-1/Sombrero_REVEAL/Data/mge_test_nosky_0deg_pa_positive_gauss"
    _ensure_dir(checkplot_dir)

    runner = MGEFitter(
        img_f200,
        dust_mask,
        pixel_scale=0.031,
        subtract_sky=True,
        linear=False,
        ngauss=30,
        plot=True,
        checkplot_dir=checkplot_dir,
        cache_dir=checkplot_dir,
        prefix="sombrero_f200",
        contour_half_size_arcsec=80,
        contour_oversample=1,
        n_sectors=19,
        allow_negative=False,
        bulge_disk=False,
    )

    x_peak, y_peak = brightest_pixel_near(
        img_f200, 7538, 7333, halfsize=40, goodmask=runner.goodmask
    )

    print(f"Initial guess for galaxy center (x, y) [pix]: ({x_peak:.2f}, {y_peak:.2f})")
    print(f"Pixel value at center img[y, x] = img[{int(round(y_peak))}, {int(round(x_peak))}] = "
          f"{img_f200[int(round(y_peak)), int(round(x_peak))]}")

    runner.set_manual_geometry(
        center=(x_peak, y_peak),   # correct public convention: (x, y)
        pa_deg=0,#90.78185872429874,#87.2,
        eps=0.7060956459920877, #0.35,
        theta_deg=90.78185872429874,#87.2,
    )

    print(f"Stored manual center in runner: (x, y) = ({runner.xc:.2f}, {runner.yc:.2f})")

    # remember to set things to force=True if you want to re-run steps that have already been cached
    runner.run_sectors()  # this will use the new geometry and overwrite any previous sectors
    runner.run_fit() # 

    # now get the deprojected density map and LOS profile in a central aperture
    res = central_los_profile_aperture(
    runner,
    aperture_arcsec=10.5,   # circular projected aperture radius
    incl_deg=87.2,         # or omit if stored on runner
    frac=0.90,
    sigmas_are_arcsec=False
    )

    print(f"Within a {res['aperture_arcsec']:.2f}\" aperture:")
    print(f"  50% LOS mass radius = ±{res['s50_arcsec']:.3f}\"")
    print(f"  90% LOS mass radius = ±{res['s90_arcsec']:.3f}\"")
    print(f"  95% LOS mass radius = ±{res['s95_arcsec']:.3f}\"")

    # convert to physical units using distance = 9.55 Mpc (from SIMBAD for Sombrero)
    distance_mpc = 9.55
    arcsec_to_kpc = distance_mpc * 1e3 * np.tan(np.deg2rad(1.0 / 3600.0))
    print(f"  50% LOS mass radius = ±{res['s50_arcsec']*arcsec_to_kpc:.3f} kpc")
    print(f"  90% LOS mass radius = ±{res['s90_arcsec']*arcsec_to_kpc:.3f} kpc")
    print(f"  95% LOS mass radius = ±{res['s95_arcsec']*arcsec_to_kpc:.3f} kpc")

    # plot the histogram in physical units as well
    

    plt.figure(figsize=(6, 4))
    plt.plot(res["s_arcsec"]*arcsec_to_kpc, res["dMds"])
    plt.axvline(-res["s90_arcsec"]*arcsec_to_kpc, ls="--")
    plt.axvline(+res["s90_arcsec"]*arcsec_to_kpc, ls="--")
    plt.xlabel("LOS distance from center [kpc]")
    plt.ylabel(r"$dM/ds$ inside projected circular aperture")
    plt.tight_layout()
    plt.savefig(os.path.join(checkplot_dir, "los_mass_profile.png"))
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(res["S_arcsec"]*arcsec_to_kpc, res["cumfrac"])
    plt.axhline(0.90, ls="--")
    plt.axvline(res["s90_arcsec"]*arcsec_to_kpc, ls="--")
    plt.xlabel(r"Enclosed LOS half-length $|s|$ [kpc]")
    plt.ylabel("Enclosed LOS mass fraction")
    plt.tight_layout()
    plt.savefig(os.path.join(checkplot_dir, "los_mass_cumulative_fraction.png"))
    plt.show()


    # out = runner.plot_deprojected_density_map(
    # inc_deg=87.2,
    # distance_mpc=9.55,   # https://simbad.cds.unistra.fr/simbad/sim-ref?bibcode=2025ApJ...978...77B
    # ml=1.0,
    # half_size_arcsec=40.0,
    # npix=600,
    # mass_density=False,  # True if you want rho_* after applying M/L
    # )   
    
    # res = runner.run_all(
    #     force_sectors=True,
    #     force_fit=True,
    # )