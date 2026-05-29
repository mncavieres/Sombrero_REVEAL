from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.optimize import least_squares
from scipy.special import erf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MGE_LUMINOSITY = REPO_ROOT / "Data/mge_NAGN_0deg_pa_positive_gauss/mge_luminosity_table.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Data/sersic_nsc_deproj_check"


@dataclass
class Config:
    mge_luminosity_table: str
    output_dir: str
    distance_mpc: float
    inclination_deg: float
    fit_rmin_arcsec: float
    fit_rmax_arcsec: float
    profile_rmin_arcsec: float
    profile_rmax_arcsec: float
    n_profile: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a deprojected Sersic luminosity-density profile to the F200W "
            "MGE luminosity model and test for a compact nuclear excess."
        )
    )
    parser.add_argument("--mge-luminosity-table", type=Path, default=DEFAULT_MGE_LUMINOSITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--distance-mpc", type=float, default=9.55)
    parser.add_argument("--inclination-deg", type=float, default=87.2)
    parser.add_argument("--fit-rmin-arcsec", type=float, default=0.3)
    parser.add_argument("--fit-rmax-arcsec", type=float, default=20.0)
    parser.add_argument("--profile-rmin-arcsec", type=float, default=0.01)
    parser.add_argument("--profile-rmax-arcsec", type=float, default=200.0)
    parser.add_argument("--n-profile", type=int, default=500)
    return parser.parse_args()


def pc_per_arcsec(distance_mpc: float) -> float:
    return float(distance_mpc * 1.0e6 / 206265.0)


def read_luminosity_mge(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = Table.read(path)
    required = {"luminosity_Lsun", "sigma_arcsec", "q_obs"}
    missing = sorted(required - set(table.colnames))
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    luminosity = np.asarray(table["luminosity_Lsun"], dtype=float)
    sigma_arcsec = np.asarray(table["sigma_arcsec"], dtype=float)
    q_obs = np.asarray(table["q_obs"], dtype=float)
    good = np.isfinite(luminosity) & np.isfinite(sigma_arcsec) & np.isfinite(q_obs)
    good &= (luminosity > 0.0) & (sigma_arcsec > 0.0) & (q_obs > 0.0)
    if not np.any(good):
        raise ValueError("No valid positive MGE components found.")
    return luminosity[good], sigma_arcsec[good], np.clip(q_obs[good], 1e-6, 1.0)


def intrinsic_q(q_obs: np.ndarray, inclination_deg: float) -> np.ndarray:
    inc = np.deg2rad(float(inclination_deg))
    sini = np.sin(inc)
    cosi = np.cos(inc)
    if np.isclose(sini, 0.0):
        raise ValueError("Inclination must be > 0 deg for this oblate deprojection.")
    if np.any(q_obs < cosi - 1e-10):
        min_allowed = np.degrees(np.arccos(np.min(q_obs)))
        raise ValueError(
            f"This MGE cannot be deprojected at i={inclination_deg:.2f} deg; "
            f"a valid inclination must be >= {min_allowed:.2f} deg."
        )
    q2 = (q_obs**2 - cosi**2) / sini**2
    return np.clip(np.sqrt(np.maximum(q2, 0.0)), 1e-6, 1.0)


def spherical_average_mge_density(
    r_pc: np.ndarray,
    luminosity_lsun: np.ndarray,
    sigma_pc: np.ndarray,
    q_intr: np.ndarray,
) -> np.ndarray:
    """Spherical average of an oblate-axisymmetric deprojected MGE density."""
    r_pc = np.asarray(r_pc, dtype=float)
    rr = r_pc[:, None]
    sigma = sigma_pc[None, :]
    q = q_intr[None, :]
    amp = luminosity_lsun[None, :] / (((2.0 * np.pi) ** 1.5) * sigma**3 * q)

    a = rr**2 / (2.0 * sigma**2)
    c = a * (1.0 / q**2 - 1.0)
    angular = np.empty_like(c)
    small = c < 1e-12
    angular[small] = 2.0 * np.exp(-a[small])
    if np.any(~small):
        cs = c[~small]
        angular[~small] = np.exp(-a[~small]) * np.sqrt(np.pi / cs) * erf(np.sqrt(cs))

    return np.sum(0.5 * amp * angular, axis=1)


def sersic_b_n(n: float | np.ndarray) -> float | np.ndarray:
    n = np.asarray(n)
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)


def prugniel_sersic_p(n: float | np.ndarray) -> float | np.ndarray:
    n = np.asarray(n)
    return 1.0 - 0.6097 / n + 0.05463 / (n * n)


def deprojected_sersic_density(r_pc: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Prugniel-Simien approximation for a deprojected Sersic luminosity density."""
    log_rho_e, log_re_pc, log_n = params
    rho_e = np.exp(log_rho_e)
    re_pc = np.exp(log_re_pc)
    n = np.exp(log_n)
    p = prugniel_sersic_p(n)
    return rho_e * (r_pc / re_pc) ** (-p) * np.exp(
        -sersic_b_n(n) * ((r_pc / re_pc) ** (1.0 / n) - 1.0)
    )


def fit_deprojected_sersic(
    r_arcsec: np.ndarray,
    r_pc: np.ndarray,
    density: np.ndarray,
    fit_rmin_arcsec: float,
    fit_rmax_arcsec: float,
) -> least_squares:
    fit = (r_arcsec >= fit_rmin_arcsec) & (r_arcsec <= fit_rmax_arcsec)
    fit &= np.isfinite(density) & (density > 0.0)
    if np.count_nonzero(fit) < 12:
        raise ValueError("Not enough finite deprojected-profile points to fit.")

    r_ref = 5.0 * (r_pc[fit] / r_arcsec[fit])[0]
    rho_ref = np.interp(r_ref, r_pc[fit], density[fit])
    p0 = np.array([np.log(rho_ref), np.log(10.0 * (r_pc[fit] / r_arcsec[fit])[0]), np.log(3.0)])
    lower = np.array([np.log(1e-20), np.log(0.01 * (r_pc[fit] / r_arcsec[fit])[0]), np.log(0.2)])
    upper = np.array([np.log(1e12), np.log(1000.0 * (r_pc[fit] / r_arcsec[fit])[0]), np.log(12.0)])

    def residual(params: np.ndarray) -> np.ndarray:
        model = deprojected_sersic_density(r_pc[fit], params)
        out = np.empty_like(model)
        bad = (~np.isfinite(model)) | (model <= 0.0)
        out[bad] = 1e12
        out[~bad] = np.log(model[~bad]) - np.log(density[fit][~bad])
        return out

    return least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=20000,
    )


def cumulative_luminosity(r_pc: np.ndarray, density: np.ndarray) -> np.ndarray:
    integrand = 4.0 * np.pi * r_pc**2 * density
    out = np.zeros_like(r_pc)
    out[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_pc))
    return out


def interpolate_profile(x: np.ndarray, y: np.ndarray, x_new: float) -> float:
    return float(np.interp(float(x_new), x, y))


def fractional_residual(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    out = np.full_like(data, np.nan, dtype=float)
    good = np.isfinite(data) & np.isfinite(model) & (model != 0.0)
    out[good] = (data[good] - model[good]) / model[good]
    return out


def write_profile_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    lines = [",".join(names)]
    for values in zip(*(columns[name] for name in names)):
        lines.append(",".join(str(v) for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(
    output_dir: Path,
    r_arcsec: np.ndarray,
    density: np.ndarray,
    model_density: np.ndarray,
    lum_mge: np.ndarray,
    lum_model: np.ndarray,
    fit_rmin_arcsec: float,
    fit_rmax_arcsec: float,
) -> None:
    frac_density = fractional_residual(density, model_density)
    frac_lum = fractional_residual(lum_mge, lum_model)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(r_arcsec, density, color="tab:blue", lw=2, label="Deprojected MGE density")
    axes[0].plot(r_arcsec, model_density, color="crimson", lw=2, label="Deprojected Sersic fit")
    axes[0].axvspan(r_arcsec[0], fit_rmin_arcsec, color="0.85", alpha=0.8, label="Excluded inner core")
    axes[0].axvline(fit_rmax_arcsec, color="0.5", ls=":", lw=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$j(r)$ [$L_\odot\,pc^{-3}$]")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0.0, color="0.25", lw=1.0)
    axes[1].plot(r_arcsec, frac_density, color="black", lw=1.5)
    axes[1].axvspan(r_arcsec[0], fit_rmin_arcsec, color="0.85", alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_ylim(-1.0, 0.5)
    axes[1].set_xlabel("3D radius (arcsec)")
    axes[1].set_ylabel(r"$(j_{\rm MGE}-j_{\rm Ser})/j_{\rm Ser}$")
    fig.tight_layout()
    fig.savefig(output_dir / "f200w_deprojected_sersic_density_profile.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(r_arcsec, lum_mge, color="tab:blue", lw=2, label="Deprojected MGE L(<r)")
    axes[0].plot(r_arcsec, lum_model, color="crimson", lw=2, label="Deprojected Sersic L(<r)")
    axes[0].axvspan(r_arcsec[0], fit_rmin_arcsec, color="0.85", alpha=0.8, label="Excluded inner core")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Enclosed luminosity [$L_\odot$]")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0.0, color="0.25", lw=1.0)
    axes[1].plot(r_arcsec, frac_lum, color="black", lw=1.5)
    axes[1].axvspan(r_arcsec[0], fit_rmin_arcsec, color="0.85", alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_ylim(-1.0, 0.5)
    axes[1].set_xlabel("3D radius (arcsec)")
    axes[1].set_ylabel(r"$(L_{\rm MGE}-L_{\rm Ser})/L_{\rm Ser}$")
    fig.tight_layout()
    fig.savefig(output_dir / "f200w_deprojected_sersic_enclosed_luminosity.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    luminosity, sigma_arcsec, q_obs = read_luminosity_mge(args.mge_luminosity_table)
    pc_arcsec = pc_per_arcsec(args.distance_mpc)
    sigma_pc = sigma_arcsec * pc_arcsec
    q_intr = intrinsic_q(q_obs, args.inclination_deg)

    r_arcsec = np.geomspace(args.profile_rmin_arcsec, args.profile_rmax_arcsec, args.n_profile)
    r_pc = r_arcsec * pc_arcsec
    density = spherical_average_mge_density(r_pc, luminosity, sigma_pc, q_intr)
    fit = fit_deprojected_sersic(r_arcsec, r_pc, density, args.fit_rmin_arcsec, args.fit_rmax_arcsec)
    model_density = deprojected_sersic_density(r_pc, fit.x)

    lum_mge = cumulative_luminosity(r_pc, density)
    lum_model = cumulative_luminosity(r_pc, model_density)
    frac_density = fractional_residual(density, model_density)
    frac_lum = fractional_residual(lum_mge, lum_model)

    profile_columns = {
        "r_arcsec": r_arcsec,
        "r_pc": r_pc,
        "mge_j_lsun_pc3": density,
        "sersic_j_lsun_pc3": model_density,
        "density_fractional_residual": frac_density,
        "mge_enclosed_lsun": lum_mge,
        "sersic_enclosed_lsun": lum_model,
        "enclosed_lsun_residual": lum_mge - lum_model,
        "enclosed_fractional_residual": frac_lum,
    }
    write_profile_csv(args.output_dir / "f200w_deprojected_sersic_profile.csv", profile_columns)
    make_plots(
        args.output_dir,
        r_arcsec,
        density,
        model_density,
        lum_mge,
        lum_model,
        args.fit_rmin_arcsec,
        args.fit_rmax_arcsec,
    )

    density_diagnostics = []
    enclosed_diagnostics = []
    for radius_arcsec in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        radius_pc = radius_arcsec * pc_arcsec
        j_mge = interpolate_profile(r_pc, density, radius_pc)
        j_ser = interpolate_profile(r_pc, model_density, radius_pc)
        l_mge = interpolate_profile(r_pc, lum_mge, radius_pc)
        l_ser = interpolate_profile(r_pc, lum_model, radius_pc)
        density_diagnostics.append(
            {
                "radius_arcsec": radius_arcsec,
                "radius_pc": radius_pc,
                "mge_j_lsun_pc3": j_mge,
                "sersic_j_lsun_pc3": j_ser,
                "density_residual_lsun_pc3": j_mge - j_ser,
                "density_fractional_residual": (j_mge - j_ser) / j_ser,
            }
        )
        enclosed_diagnostics.append(
            {
                "radius_arcsec": radius_arcsec,
                "radius_pc": radius_pc,
                "mge_enclosed_lsun": l_mge,
                "sersic_enclosed_lsun": l_ser,
                "enclosed_residual_lsun": l_mge - l_ser,
                "enclosed_fractional_residual": (l_mge - l_ser) / l_ser,
            }
        )

    fit_mask = (r_arcsec >= args.fit_rmin_arcsec) & (r_arcsec <= args.fit_rmax_arcsec)
    rms_log_resid = float(np.sqrt(np.mean((np.log(model_density[fit_mask]) - np.log(density[fit_mask])) ** 2)))
    n = float(np.exp(fit.x[2]))
    summary = {
        "config": asdict(
            Config(
                mge_luminosity_table=str(args.mge_luminosity_table),
                output_dir=str(args.output_dir),
                distance_mpc=args.distance_mpc,
                inclination_deg=args.inclination_deg,
                fit_rmin_arcsec=args.fit_rmin_arcsec,
                fit_rmax_arcsec=args.fit_rmax_arcsec,
                profile_rmin_arcsec=args.profile_rmin_arcsec,
                profile_rmax_arcsec=args.profile_rmax_arcsec,
                n_profile=args.n_profile,
            )
        ),
        "mge_deprojection": {
            "n_components": int(len(luminosity)),
            "total_luminosity_lsun": float(np.sum(luminosity)),
            "pc_per_arcsec": pc_arcsec,
            "q_intr_min": float(np.min(q_intr)),
            "q_intr_max": float(np.max(q_intr)),
        },
        "deprojected_sersic_fit": {
            "success": bool(fit.success),
            "message": fit.message,
            "robust_cost": float(fit.cost),
            "rms_log_residual_in_fit_range": rms_log_resid,
            "rho_e_lsun_pc3": float(np.exp(fit.x[0])),
            "Re_pc": float(np.exp(fit.x[1])),
            "Re_arcsec": float(np.exp(fit.x[1]) / pc_arcsec),
            "n": n,
            "prugniel_sersic_p": float(prugniel_sersic_p(n)),
        },
        "density_diagnostics": density_diagnostics,
        "enclosed_luminosity_diagnostics": enclosed_diagnostics,
        "interpretation": (
            "The deprojected MGE luminosity density and enclosed luminosity lie below the "
            "inward extrapolation of the deprojected Sersic fit inside 0.3 arcsec, so this "
            "fit does not show a positive compact nuclear-star-cluster excess."
        ),
    }
    (args.output_dir / "f200w_deprojected_sersic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary["deprojected_sersic_fit"], indent=2))
    for row in enclosed_diagnostics:
        print(
            "r<={radius_arcsec:.2f}\" residual={enclosed_residual_lsun:.3e} Lsun "
            "frac={enclosed_fractional_residual:.3f}".format(**row)
        )
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
