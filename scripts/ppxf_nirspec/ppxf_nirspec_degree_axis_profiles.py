#!/usr/bin/env python3
"""
Build major/minor-axis kinematic profiles for the PHOENIX degree study.

The script reads completed case folders from
ppxf_nirspec_phoenix_powerbin_degree_study.py, rotates the bin coordinates by a
user-provided angle, and compares pseudo-slit profiles of velocity and
dispersion for all completed degree/mdegree cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_ppxf_nirspec_axis_profiles"),
)

import matplotlib

matplotlib.use("Agg")

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUBE_PATH = ROOT / "Data/IFU/antoine/sombrero_nirspec_1150_p1293_g235_wicked.fits"
DEFAULT_STUDY_DIR = ROOT / "Data/ppxf_nirspec/antoine_wicked_powerbin_sn120_lsf_degree_study"
CASE_RE = re.compile(r"deg(?P<degree>\d+)_mdeg(?P<mdegree>\d+)$")


@dataclass(frozen=True)
class CaseProduct:
    label: str
    degree: int
    mdegree: int
    fits_path: Path


@dataclass(frozen=True)
class ProfileRow:
    case_label: str
    degree: int
    mdegree: int
    axis: str
    quantity: str
    radius_arcsec: float
    median: float
    p16: float
    p84: float
    n: int


@dataclass(frozen=True)
class BrightnessProfileRow:
    axis: str
    radius_arcsec: float
    median_brightness: float
    p16_brightness: float
    p84_brightness: float
    normalized_median_brightness: float
    n: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def case_label(degree: int, mdegree: int) -> str:
    return f"deg{degree:02d}_mdeg{mdegree:02d}"


def parse_case_pairs(text: str | None) -> list[tuple[int, int]] | None:
    if text is None:
        return None
    pairs: list[tuple[int, int]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            d_text, m_text = chunk.split(":", maxsplit=1)
        elif "/" in chunk:
            d_text, m_text = chunk.split("/", maxsplit=1)
        else:
            raise ValueError(f"Could not parse case '{chunk}'. Use degree:mdegree.")
        pairs.append((int(d_text), int(m_text)))
    return pairs


def load_case_pairs_from_config(study_dir: Path) -> list[tuple[int, int]] | None:
    config_path = study_dir / "degree_study_config.json"
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text())
    pairs: list[tuple[int, int]] = []
    for item in payload.get("case_pairs", []):
        if "degree" in item and "mdegree" in item:
            pairs.append((int(item["degree"]), int(item["mdegree"])))
    return pairs or None


def expected_fits_path(study_dir: Path, cube_stem: str, target_sn: float, degree: int, mdegree: int) -> Path:
    label = case_label(degree, mdegree)
    base = study_dir / label / f"{cube_stem}_phoenix_powerbin_lsf_sn{int(round(target_sn))}_kinematics"
    return base.with_suffix(".fits")


def parse_case_from_dir(path: Path) -> tuple[int, int] | None:
    match = CASE_RE.match(path.name)
    if match is None:
        return None
    return int(match.group("degree")), int(match.group("mdegree"))


def discover_case_products(
    study_dir: Path,
    cube_stem: str,
    target_sn: float,
    case_pairs: list[tuple[int, int]] | None,
) -> list[CaseProduct]:
    products: list[CaseProduct] = []

    ordered_pairs = case_pairs
    if ordered_pairs is None:
        ordered_pairs = load_case_pairs_from_config(study_dir)

    if ordered_pairs is not None:
        for degree, mdegree in ordered_pairs:
            fits_path = expected_fits_path(study_dir, cube_stem, target_sn, degree, mdegree)
            if fits_path.exists():
                products.append(
                    CaseProduct(
                        label=case_label(degree, mdegree),
                        degree=degree,
                        mdegree=mdegree,
                        fits_path=fits_path,
                    )
                )
        return products

    for fits_path in sorted(study_dir.glob("deg*_mdeg*/*_kinematics.fits")):
        parsed = parse_case_from_dir(fits_path.parent)
        if parsed is None:
            continue
        degree, mdegree = parsed
        products.append(
            CaseProduct(
                label=fits_path.parent.name,
                degree=degree,
                mdegree=mdegree,
                fits_path=fits_path,
            )
        )
    products.sort(key=lambda item: (item.degree, item.mdegree))
    return products


def rotate_xy(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t
    return x_rot, y_rot


def inverse_rotated_line(
    radius: np.ndarray,
    axis: str,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    if axis == "major":
        x_rot = radius
        y_rot = np.zeros_like(radius)
    elif axis == "minor":
        x_rot = np.zeros_like(radius)
        y_rot = radius
    else:
        raise ValueError(f"Unknown axis '{axis}'")
    x = x_rot * cos_t + y_rot * sin_t
    y = -x_rot * sin_t + y_rot * cos_t
    return x, y


def profile_edges(coord: np.ndarray, bin_width: float) -> np.ndarray:
    finite = coord[np.isfinite(coord)]
    if finite.size == 0:
        return np.array([], dtype=float)
    lo = np.floor(np.nanmin(finite) / bin_width) * bin_width
    hi = np.ceil(np.nanmax(finite) / bin_width) * bin_width
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.array([], dtype=float)
    return np.arange(lo, hi + 1.5 * bin_width, bin_width)


def binned_profile(
    coord: np.ndarray,
    values: np.ndarray,
    bin_width: float,
    min_per_bin: int,
) -> list[tuple[float, float, float, float, int]]:
    edges = profile_edges(coord, bin_width)
    if edges.size < 2:
        return []
    rows: list[tuple[float, float, float, float, int]] = []
    coord = np.asarray(coord, dtype=float)
    values = np.asarray(values, dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (coord >= lo) & (coord < hi) & np.isfinite(values)
        n = int(np.count_nonzero(in_bin))
        if n < min_per_bin:
            continue
        vals = values[in_bin]
        rows.append(
            (
                float(0.5 * (lo + hi)),
                float(np.nanmedian(vals)),
                float(np.nanpercentile(vals, 16.0)),
                float(np.nanpercentile(vals, 84.0)),
                n,
            )
        )
    return rows


def table_names(tab) -> set[str]:
    return set(tab.names or [])


def read_case_profiles(
    product: CaseProduct,
    rotation_deg: float,
    slit_half_width_arcsec: float,
    radial_bin_width_arcsec: float,
    min_per_bin: int,
    velocity_column: str,
    goodfit_only: bool,
) -> list[ProfileRow]:
    with fits.open(product.fits_path) as hdul:
        tab = hdul["BIN_RESULTS"].data
        names = table_names(tab)
        if velocity_column not in names:
            raise KeyError(
                f"{product.fits_path} does not contain column '{velocity_column}'. "
                f"Available columns: {', '.join(sorted(names))}"
            )
        if "SIGMA" not in names:
            raise KeyError(f"{product.fits_path} does not contain SIGMA")

        x = np.asarray(tab["X"], dtype=float)
        y = np.asarray(tab["Y"], dtype=float)
        x_rot, y_rot = rotate_xy(x, y, rotation_deg)
        finite_base = np.isfinite(x_rot) & np.isfinite(y_rot)
        if goodfit_only and "GOODFIT" in names:
            finite_base &= np.asarray(tab["GOODFIT"], dtype=bool)

        quantities = {
            velocity_column: np.asarray(tab[velocity_column], dtype=float),
            "SIGMA": np.asarray(tab["SIGMA"], dtype=float),
        }

    rows: list[ProfileRow] = []
    axis_specs = {
        "major": (x_rot, np.abs(y_rot) <= slit_half_width_arcsec),
        "minor": (y_rot, np.abs(x_rot) <= slit_half_width_arcsec),
    }
    for axis, (coord, axis_mask) in axis_specs.items():
        base_mask = finite_base & axis_mask
        for quantity, values in quantities.items():
            mask = base_mask & np.isfinite(values)
            for radius, med, p16, p84, n in binned_profile(
                coord[mask],
                values[mask],
                radial_bin_width_arcsec,
                min_per_bin,
            ):
                rows.append(
                    ProfileRow(
                        case_label=product.label,
                        degree=product.degree,
                        mdegree=product.mdegree,
                        axis=axis,
                        quantity=quantity,
                        radius_arcsec=radius,
                        median=med,
                        p16=p16,
                        p84=p84,
                        n=n,
                    )
                )
    return rows


def write_profile_csv(path: Path, rows: list[ProfileRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)


def write_brightness_csv(path: Path, rows: list[BrightnessProfileRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)


def map_extent(header: fits.Header, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    ny, nx = shape
    pixsize = float(header.get("PIXSIZE", 1.0))
    center_row = float(header.get("CENROW", (ny + 1) / 2.0)) - 1.0
    center_col = float(header.get("CENCOL", (nx + 1) / 2.0)) - 1.0
    return (
        float((-0.5 - center_col) * pixsize),
        float((nx - 0.5 - center_col) * pixsize),
        float((-0.5 - center_row) * pixsize),
        float((ny - 0.5 - center_row) * pixsize),
    )


def map_coordinates(header: fits.Header, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    pixsize = float(header.get("PIXSIZE", np.nan))
    center_row = float(header.get("CENROW", (ny + 1) / 2.0)) - 1.0
    center_col = float(header.get("CENCOL", (nx + 1) / 2.0)) - 1.0
    row, col = np.indices(shape, dtype=float)
    return (col - center_col) * pixsize, (row - center_row) * pixsize


def load_signal_map(product: CaseProduct) -> tuple[np.ndarray, fits.Header]:
    with fits.open(product.fits_path) as hdul:
        if "SIGNAL_MAP" not in hdul:
            raise KeyError(f"{product.fits_path} does not contain SIGNAL_MAP")
        signal = np.asarray(hdul["SIGNAL_MAP"].data, dtype=float)
        header = hdul[0].header.copy()
    return signal, header


def brightness_profiles(
    signal_map: np.ndarray,
    header: fits.Header,
    rotation_deg: float,
    slit_half_width_arcsec: float,
    radial_bin_width_arcsec: float,
    min_per_bin: int,
) -> list[BrightnessProfileRow]:
    x, y = map_coordinates(header, signal_map.shape)
    x_rot, y_rot = rotate_xy(x, y, rotation_deg)
    norm = np.nanmax(signal_map[np.isfinite(signal_map)])
    if not np.isfinite(norm) or norm == 0:
        norm = 1.0

    rows: list[BrightnessProfileRow] = []
    axis_specs = {
        "major": (x_rot.ravel(), np.abs(y_rot.ravel()) <= slit_half_width_arcsec),
        "minor": (y_rot.ravel(), np.abs(x_rot.ravel()) <= slit_half_width_arcsec),
    }
    values = signal_map.ravel()
    for axis, (coord, axis_mask) in axis_specs.items():
        mask = axis_mask & np.isfinite(values)
        for radius, med, p16, p84, n in binned_profile(
            coord[mask],
            values[mask],
            radial_bin_width_arcsec,
            min_per_bin,
        ):
            rows.append(
                BrightnessProfileRow(
                    axis=axis,
                    radius_arcsec=radius,
                    median_brightness=med,
                    p16_brightness=p16,
                    p84_brightness=p84,
                    normalized_median_brightness=float(med / norm),
                    n=n,
                )
            )
    return rows


def rows_to_arrays(
    rows: list[ProfileRow],
    case_label_value: str,
    axis: str,
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row.case_label == case_label_value and row.axis == axis and row.quantity == quantity
    ]
    selected.sort(key=lambda row: row.radius_arcsec)
    return (
        np.array([row.radius_arcsec for row in selected], dtype=float),
        np.array([row.median for row in selected], dtype=float),
        np.array([row.p16 for row in selected], dtype=float),
        np.array([row.p84 for row in selected], dtype=float),
        np.array([row.n for row in selected], dtype=int),
    )


def brightness_rows_to_arrays(
    rows: list[BrightnessProfileRow],
    axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row.axis == axis]
    selected.sort(key=lambda row: row.radius_arcsec)
    return (
        np.array([row.radius_arcsec for row in selected], dtype=float),
        np.array([row.normalized_median_brightness for row in selected], dtype=float),
        np.array([row.p16_brightness for row in selected], dtype=float),
        np.array([row.p84_brightness for row in selected], dtype=float),
        np.array([row.n for row in selected], dtype=int),
    )


def plot_kinematic_profiles(
    outdir: Path,
    products: list[CaseProduct],
    rows: list[ProfileRow],
    rotation_deg: float,
    slit_half_width_arcsec: float,
    velocity_column: str,
) -> Path:
    velocity_label = "V - systemic [km/s]" if velocity_column == "V_REL_KMS" else f"{velocity_column} [km/s]"
    quantity_specs = [
        ("major", velocity_column, "Major-Axis Velocity", velocity_label),
        ("minor", velocity_column, "Minor-Axis Velocity", velocity_label),
        ("major", "SIGMA", "Major-Axis Dispersion", "sigma [km/s]"),
        ("minor", "SIGMA", "Minor-Axis Dispersion", "sigma [km/s]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True, sharex=False)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(products), 1)))

    for ax, (axis, quantity, title, ylabel) in zip(axes.ravel(), quantity_specs):
        if quantity == velocity_column:
            ax.axhline(0.0, color="black", lw=0.7, alpha=0.35)
        for color, product in zip(colors, products):
            radius, med, p16, p84, _n = rows_to_arrays(rows, product.label, axis, quantity)
            if radius.size == 0:
                continue
            label = f"d={product.degree}, m={product.mdegree}"
            ax.plot(radius, med, marker="o", ms=3.0, lw=1.2, color=color, label=label)
            ax.fill_between(radius, p16, p84, color=color, alpha=0.08, lw=0)
        ax.set_title(title)
        ax.set_xlabel(f"{axis} coordinate after {rotation_deg:.1f} deg rotation [arcsec]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0, 0].legend(loc="best", fontsize=8, ncols=2)
    fig.suptitle(
        f"Degree Study Axis Profiles | slit half-width = {slit_half_width_arcsec:.2f} arcsec",
        fontsize=13,
    )
    outpath = outdir / "degree_study_major_minor_velocity_sigma_profiles.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_brightness_centering(
    outdir: Path,
    signal_map: np.ndarray,
    header: fits.Header,
    rows: list[BrightnessProfileRow],
    rotation_deg: float,
    slit_half_width_arcsec: float,
) -> Path:
    finite = signal_map[np.isfinite(signal_map)]
    if finite.size == 0:
        raise ValueError("SIGNAL_MAP has no finite values")
    floor = float(np.nanpercentile(finite, 1.0))
    scale = float(np.nanpercentile(finite, 95.0) - floor)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(finite))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    display = np.arcsinh(np.clip(signal_map - floor, 0.0, None) / scale)
    extent = map_extent(header, signal_map.shape)
    xlim = max(abs(extent[0]), abs(extent[1]))
    ylim = max(abs(extent[2]), abs(extent[3]))
    radius_max = max(xlim, ylim)
    radius = np.linspace(-radius_max, radius_max, 200)
    major_x, major_y = inverse_rotated_line(radius, "major", rotation_deg)
    minor_x, minor_y = inverse_rotated_line(radius, "minor", rotation_deg)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(
        display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="gray_r",
        aspect="equal",
    )
    ax.plot(major_x, major_y, color="tab:red", lw=1.2, label="major")
    ax.plot(minor_x, minor_y, color="tab:blue", lw=1.2, label="minor")
    ax.scatter([0.0], [0.0], marker="+", s=90, color="gold", linewidths=1.5, label="center")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_title("Collapsed Cube Brightness")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="asinh scaled flux")

    for ax, axis, color in [(axes[1], "major", "tab:red"), (axes[2], "minor", "tab:blue")]:
        radius_profile, norm_brightness, _p16, _p84, _n = brightness_rows_to_arrays(rows, axis)
        ax.axvline(0.0, color="black", lw=0.7, alpha=0.35)
        ax.plot(radius_profile, norm_brightness, marker="o", ms=3.0, lw=1.2, color=color)
        ax.set_xlabel(f"{axis} coordinate after {rotation_deg:.1f} deg rotation [arcsec]")
        ax.set_ylabel("normalized median brightness")
        ax.set_title(f"{axis.capitalize()}-Axis Brightness Profile")
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=min(-0.05, float(np.nanmin(norm_brightness)) - 0.05) if norm_brightness.size else -0.05)

    fig.suptitle(
        f"Brightness Centering Check | slit half-width = {slit_half_width_arcsec:.2f} arcsec",
        fontsize=13,
    )
    outpath = outdir / "degree_study_brightness_centering.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make rotated major/minor-axis profiles for pPXF degree-study kinematics.",
    )
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE_PATH)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-sn", type=float, default=120.0)
    parser.add_argument("--case-pairs", type=str, default=None)
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=-18.0,
        help="Coordinate rotation applied before extracting profiles. Default: -18 deg.",
    )
    parser.add_argument("--slit-half-width-arcsec", type=float, default=0.15)
    parser.add_argument("--radial-bin-width-arcsec", type=float, default=0.10)
    parser.add_argument("--min-per-bin", type=int, default=1)
    parser.add_argument("--velocity-column", type=str, default="V_REL_KMS")
    parser.add_argument(
        "--include-badfits",
        action="store_true",
        help="Include bins with GOODFIT=False. Default is to use only good-fit bins.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cube_path = args.cube_path.expanduser().resolve()
    study_dir = args.study_dir.expanduser().resolve()
    outdir = ensure_dir((args.output_dir or study_dir).expanduser().resolve())
    case_pairs = parse_case_pairs(args.case_pairs)
    products = discover_case_products(study_dir, cube_path.stem, float(args.target_sn), case_pairs)
    if not products:
        raise FileNotFoundError(
            f"No completed degree-study kinematics FITS files found in {study_dir}. "
            "Run the degree study first, or pass --case-pairs for the completed cases."
        )

    all_rows: list[ProfileRow] = []
    for product in products:
        print(f"[axis-profiles] Reading {product.label}: {product.fits_path}")
        all_rows.extend(
            read_case_profiles(
                product,
                rotation_deg=float(args.rotation_deg),
                slit_half_width_arcsec=float(args.slit_half_width_arcsec),
                radial_bin_width_arcsec=float(args.radial_bin_width_arcsec),
                min_per_bin=max(1, int(args.min_per_bin)),
                velocity_column=str(args.velocity_column).upper(),
                goodfit_only=not bool(args.include_badfits),
            )
        )

    profile_csv = outdir / "degree_study_major_minor_velocity_sigma_profiles.csv"
    write_profile_csv(profile_csv, all_rows)

    signal_map, signal_header = load_signal_map(products[0])
    brightness_rows = brightness_profiles(
        signal_map,
        signal_header,
        rotation_deg=float(args.rotation_deg),
        slit_half_width_arcsec=float(args.slit_half_width_arcsec),
        radial_bin_width_arcsec=float(args.radial_bin_width_arcsec),
        min_per_bin=max(1, int(args.min_per_bin)),
    )
    brightness_csv = outdir / "degree_study_brightness_axis_profiles.csv"
    write_brightness_csv(brightness_csv, brightness_rows)

    kin_plot = plot_kinematic_profiles(
        outdir,
        products,
        all_rows,
        rotation_deg=float(args.rotation_deg),
        slit_half_width_arcsec=float(args.slit_half_width_arcsec),
        velocity_column=str(args.velocity_column).upper(),
    )
    brightness_plot = plot_brightness_centering(
        outdir,
        signal_map,
        signal_header,
        brightness_rows,
        rotation_deg=float(args.rotation_deg),
        slit_half_width_arcsec=float(args.slit_half_width_arcsec),
    )

    config = {
        "cube_path": str(cube_path),
        "study_dir": str(study_dir),
        "output_dir": str(outdir),
        "target_sn": float(args.target_sn),
        "rotation_deg": float(args.rotation_deg),
        "slit_half_width_arcsec": float(args.slit_half_width_arcsec),
        "radial_bin_width_arcsec": float(args.radial_bin_width_arcsec),
        "min_per_bin": max(1, int(args.min_per_bin)),
        "velocity_column": str(args.velocity_column).upper(),
        "goodfit_only": not bool(args.include_badfits),
        "products": [asdict(product) | {"fits_path": str(product.fits_path)} for product in products],
        "profile_csv": str(profile_csv),
        "brightness_csv": str(brightness_csv),
        "kinematic_profile_plot": str(kin_plot),
        "brightness_centering_plot": str(brightness_plot),
    }
    (outdir / "degree_study_axis_profile_config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"[axis-profiles] Kinematic profiles CSV : {profile_csv}")
    print(f"[axis-profiles] Brightness profiles CSV : {brightness_csv}")
    print(f"[axis-profiles] Kinematic profile plot: {kin_plot}")
    print(f"[axis-profiles] Brightness check plot : {brightness_plot}")


if __name__ == "__main__":
    main()
