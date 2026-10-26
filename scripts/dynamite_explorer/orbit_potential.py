"""Re-integrate DYNAMITE orbits in the model potential, for 3D visualisation.

DYNAMITE does not store orbit trajectories, only time-averaged projections.
To draw orbits in the intrinsic (deprojected) frame we start from DYNAMITE's
own initial conditions (``datfil/begin.dat`` for tubes, ``beginbox.dat`` for
boxes) and integrate them again in the potential described by
``infil/parameters_pot.in``. The potential is a NumPy port of the Fortran
routines ``triaxpotent.f90`` (deprojected MGE, Cappellari 2002 eq. 12) and
``dmpotent.f90`` (Plummer black hole plus NFW halo, type 1).

Units: lengths in arcsec, velocities in km/s, so time is arcsec/(km/s).

The trajectories are for visualisation. The Fortran integrator uses its own
interpolated potential grid and integrates for 200 periods, so a re-integrated
orbit follows the same family and extent but not the identical path.
"""
import numpy as np
from scipy.integrate import solve_ivp

# constants exactly as in legacy_fortran/iniparam_f.f90
G_KM = 6.67428e-11 * 1.98892e30 / 1e9          # km^3 / (s^2 Msun)
PARSEC_KM = 1.4959787068e8 * (648e3 / np.pi)
H0_PER_S = 7.0e-5 / PARSEC_KM                  # 70 km/s/Mpc in 1/s

# Fortran switches to a point mass beyond this many intrinsic sigmas
OUTER_APPROX = 300.0
# Gauss-Legendre nodes for the t-integral of Cappellari (2002) eq. 12
_GL_T, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_T = 0.5 * (_GL_T + 1.0)
_GL_W = 0.5 * _GL_W


def read_parameters_pot(path):
    """Parse ``infil/parameters_pot.in`` (layout from iniparam_f.f90)."""
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    n = int(lines[0][0])
    mge = np.array([[float(v) for v in ln[:4]] for ln in lines[1:n + 1]])
    rest = lines[n + 1:]
    out = dict(
        surf_pc=mge[:, 0], sigobs_arcsec=mge[:, 1], qobs=mge[:, 2],
        psi_twist=mge[:, 3],
        distance=float(rest[0][0]),
        theta=float(rest[1][0]), phi=float(rest[1][1]), psi=float(rest[1][2]),
        upsilon=float(rest[2][0]), mbh=float(rest[3][0]),
        softl_arcsec=float(rest[4][0]),
        nE=int(rest[5][0]), logrmin=float(rest[5][1]), logrmax=float(rest[5][2]),
        nI2=int(rest[6][0]), nI3=int(rest[7][0]), dithering=int(rest[8][0]),
        quad_nr=int(rest[9][0]), quad_nth=int(rest[10][0]),
        quad_nph=int(rest[11][0]),
        dm_type=int(rest[12][0]), dm_params=[float(v) for v in rest[13]],
    )
    return out


def deproject_mge(qobs, theta_deg, phi_deg, psi_deg, psi_twist=0.0):
    """Intrinsic p_j, q_j and sigma_intr/sigma_obs per Gaussian (tp_setup)."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    psi = np.radians(np.asarray(psi_twist) + psi_deg)
    secth, cotph = 1.0 / np.cos(th), 1.0 / np.tan(ph)
    delp = 1.0 - qobs ** 2
    nom1minq2 = delp * (2 * np.cos(2 * psi)
                        + np.sin(2 * psi) * (secth * cotph - np.cos(th) * np.tan(ph)))
    nomp2minq2 = delp * (2 * np.cos(2 * psi)
                         + np.sin(2 * psi) * (np.cos(th) * cotph - secth * np.tan(ph)))
    denom = 2 * np.sin(th) ** 2 * (delp * np.cos(psi)
                                   * (np.cos(psi) + secth * cotph * np.sin(psi)) - 1)
    q2 = 1.0 - nom1minq2 / denom
    p2 = q2 + nomp2minq2 / denom
    q, p = np.sqrt(q2), np.sqrt(p2)
    sig_ratio = np.sqrt(qobs / np.sqrt((p * np.cos(th)) ** 2 + (q * np.sin(th)) ** 2
                                       * ((p * np.cos(ph)) ** 2 + np.sin(ph) ** 2)))
    return p, q, sig_ratio


class DynamitePotential:
    """Stars (deprojected MGE) + Plummer BH + NFW halo, as in the Fortran."""

    def __init__(self, pars):
        self.pars = pars
        self.kpa = pars['distance'] * 1e6 * np.tan(np.pi / 648e3) * PARSEC_KM  # km/arcsec
        self.G = G_KM / self.kpa                     # (km/s)^2 arcsec / Msun
        pc_per_arcsec = self.kpa / PARSEC_KM
        surf = pars['surf_pc'] * pars['upsilon'] * pc_per_arcsec ** 2   # Msun/arcsec^2
        qobs, sigobs = pars['qobs'], pars['sigobs_arcsec']
        p, q, ratio = deproject_mge(qobs, pars['theta'], pars['phi'], pars['psi'],
                                    pars['psi_twist'])
        self.p, self.q = p, q
        self.sig = sigobs * ratio
        dens = surf * qobs * sigobs ** 2 / (np.sqrt(2 * np.pi) * p * q * self.sig ** 3)
        self.V0 = 4 * np.pi * self.G * self.sig ** 2 * p * q * dens
        self.mass_gauss = (2 * np.pi) ** 1.5 * dens * p * q * self.sig ** 3
        self.mstars = 2 * np.pi * np.sum(surf * qobs * sigobs ** 2)
        self.mbh = pars['mbh']
        self.soft = pars['softl_arcsec']
        self.nfw = None
        if pars['dm_type'] == 1:
            c, f = pars['dm_params'][:2]
            rho_crit = 3 * H0_PER_S ** 2 / (8 * np.pi * G_KM) * self.kpa ** 3  # Msun/arcsec^3
            rhoc = (200 / 3) * rho_crit * c ** 3 / (np.log(1 + c) - c / (1 + c))
            rc = (3 / (800 * np.pi * rho_crit * c ** 3) * f * self.mstars) ** (1 / 3)
            self.nfw = dict(c=c, f=f, rc=rc, mscale=4 * np.pi * rhoc * rc ** 3,
                            m200=800 * np.pi / 3 * rho_crit * rc ** 3 * c ** 3)
        elif pars['dm_type'] != 0:
            raise NotImplementedError(f"dark halo type {pars['dm_type']}")

    # ---------------------------------------------------------------- stars
    def _stars(self, x, y, z, want_pot=False):
        r2 = x * x + y * y + z * z
        r = np.sqrt(r2)
        sig = self.sig[:, None]
        far = r > OUTER_APPROX * self.sig
        tmax = np.minimum(1.0, 12.0 * self.sig / max(r, 1e-300))
        t = tmax[:, None] * _GL_T[None, :]
        w = tmax[:, None] * _GL_W[None, :]
        t2 = t * t
        d = 1.0 - (1.0 - self.p[:, None] ** 2) * t2
        e = 1.0 - (1.0 - self.q[:, None] ** 2) * t2
        a = np.exp(-t2 / (2 * sig ** 2) * (x * x + y * y / d + z * z / e)) / np.sqrt(d * e)
        if want_pot:
            pot = self.V0 * np.sum(w * a, axis=1)
            pm = np.sqrt(np.pi / 2) * self.sig * self.V0 / max(r, 1e-300)
            return np.sum(np.where(far, pm, pot))
        base = -t2 * a / sig ** 2 * w
        ax = self.V0 * x * np.sum(base, axis=1)
        ay = self.V0 * y * np.sum(base / d, axis=1)
        az = self.V0 * z * np.sum(base / e, axis=1)
        if far.any():
            pm = -np.sqrt(np.pi / 2) * self.sig * self.V0 / max(r, 1e-300) ** 3
            ax = np.where(far, pm * x, ax)
            ay = np.where(far, pm * y, ay)
            az = np.where(far, pm * z, az)
        return ax.sum(), ay.sum(), az.sum()

    def components_accel(self, x, y, z):
        """Accelerations (stars, bh, dm) at one point, each a 3-tuple."""
        stars = self._stars(x, y, z)
        r2 = x * x + y * y + z * z
        tb = -self.G * self.mbh * (r2 + self.soft ** 2) ** -1.5
        bh = (tb * x, tb * y, tb * z)
        dm = (0.0, 0.0, 0.0)
        if self.nfw is not None:
            r = np.sqrt(r2)
            s = r / self.nfw['rc']
            td = -self.G * self.nfw['mscale'] / r2 * (np.log1p(s) - s / (1 + s)) / r
            dm = (td * x, td * y, td * z)
        return stars, bh, dm

    def accel(self, x, y, z):
        s, b, d = self.components_accel(x, y, z)
        return s[0] + b[0] + d[0], s[1] + b[1] + d[1], s[2] + b[2] + d[2]

    def potential(self, x, y, z):
        """Gravitational potential (negative), in (km/s)^2."""
        r2 = x * x + y * y + z * z
        pot = self._stars(x, y, z, want_pot=True)
        pot += self.G * self.mbh / np.sqrt(r2 + self.soft ** 2)
        if self.nfw is not None:
            r = np.sqrt(r2)
            pot += self.G * self.nfw['mscale'] / r * np.log1p(r / self.nfw['rc'])
        return -pot

    def circular_velocity(self, r_arcsec, axis=0):
        """v_c along one principal axis, total and per component (km/s)."""
        out = {k: [] for k in ('total', 'stars', 'bh', 'dm')}
        for r in np.atleast_1d(r_arcsec):
            pos = [0.0, 0.0, 0.0]
            pos[axis] = r
            comps = self.components_accel(*pos)
            vals = [np.sqrt(max(-c[axis] * r, 0.0)) for c in comps]
            out['stars'].append(vals[0])
            out['bh'].append(vals[1])
            out['dm'].append(vals[2])
            out['total'].append(np.sqrt(vals[0] ** 2 + vals[1] ** 2 + vals[2] ** 2))
        return {k: np.array(v) for k, v in out.items()}


def read_begin(path):
    """Read begin.dat / beginbox.dat -> header grid shape and IC table."""
    with open(path) as f:
        shape = tuple(int(v) for v in f.readline().split())
    data = np.loadtxt(path, skiprows=1)
    return shape, data


def bundle_ic_row(iE, iI2, iI3, shape, ndither, dither=None):
    """Row index in begin.dat of one dither of orbit bundle (iE, iI2, iI3).

    begin.dat lists the dithered start grid (nE*nd, nI2*nd, nI3*nd) with the
    I3 index fastest. ``dither`` = (a, b, c) in 0..nd-1, default the centre.
    """
    if dither is None:
        dither = (ndither // 2,) * 3
    a, b, c = dither
    i1, i2, i3 = iE * ndither + a, iI2 * ndither + b, iI3 * ndither + c
    return (i1 * shape[1] + i2) * shape[2] + i3


def integrate_orbit(pot, ic_row, kpa, n_periods=15, n_out=1500, rtol=1e-9):
    """Integrate one begin.dat row. Returns (t, pos[n,3] arcsec, vel[n,3])."""
    x0 = ic_row[3:6] / kpa
    v0 = ic_row[6:9].copy()
    rc, vc = ic_row[9] / kpa, ic_row[11]
    period = 2 * np.pi * rc / vc if vc > 0 else 2 * np.pi * np.linalg.norm(x0) / max(np.linalg.norm(v0), 1.0)
    t_end = n_periods * period

    def rhs(_, s):
        ax, ay, az = pot.accel(s[0], s[1], s[2])
        return [s[3], s[4], s[5], ax, ay, az]

    t_eval = np.linspace(0, t_end, n_out)
    sol = solve_ivp(rhs, (0, t_end), np.concatenate([x0, v0]), method='DOP853',
                    rtol=rtol, atol=1e-12 * max(np.linalg.norm(x0), 1e-6),
                    t_eval=t_eval)
    return sol.t, sol.y[:3].T, sol.y[3:].T, period


def orbit_averages(pos, vel):
    """Time averages as written by integrator_find_orbtype (orbclass.out)."""
    x, y, z = pos.T
    vx, vy, vz = vel.T
    lx, ly, lz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    r = np.sqrt(x * x + y * y + z * z).mean()
    vrms2 = np.mean(vx ** 2 + vy ** 2 + vz ** 2 + 2 * (vx * vy + vy * vz + vz * vx))
    # orbit type as in the Fortran: sign changes of each L component
    lxc, lyc, lzc = lx.max() * lx.min(), ly.max() * ly.min(), lz.max() * lz.min()
    otype = 5
    if lxc > 0 and lyc < 0 and lzc < 0:
        otype = 1
    if lxc < 0 and lyc > 0 and lzc < 0:
        otype = 2
    if lxc < 0 and lyc < 0 and lzc > 0:
        otype = 3
    if lxc < 0 and lyc < 0 and lzc < 0:
        otype = 4
    return dict(Lx=lx.mean(), Ly=ly.mean(), Lz=lz.mean(), r=r, vrms=np.sqrt(vrms2),
                type=otype)
