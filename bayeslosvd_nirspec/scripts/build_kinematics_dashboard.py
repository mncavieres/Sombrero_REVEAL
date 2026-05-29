#!/usr/bin/env python3
"""Build a self-contained BAYES-LOSVD kinematics dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MOMENTS = ("vel", "sigma", "h3", "h4")
MAP_ROTATION_DEG = -18.0
CUBE_FIT_RANGE_UM = (2.10, 2.398)
CONTOUR_PERCENTILES = (50, 60, 70, 80, 88, 94, 97, 99)


def rotate_coordinates(x, y, angle_deg=MAP_ROTATION_DEG):
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return x * c - y * s, x * s + y * c


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def compact_array(values, ndigits=6):
    arr = np.asarray(values, dtype=float)
    rounded = np.round(arr, ndigits)
    flat = rounded.ravel()
    return [finite_float(v) for v in flat]


def compact_segment(segment, ndigits=5):
    arr = np.asarray(segment, dtype=float)
    arr = np.round(arr, ndigits)
    return [[finite_float(x), finite_float(y)] for x, y in arr]


def normalized_losvd_for_bin(group, xvel_order, xplot):
    if "losvd" not in group:
        return np.full(len(xplot), np.nan)
    losvd = np.asarray(group["losvd"])
    if losvd.ndim == 2 and losvd.shape[0] > 2:
        losvd = losvd[2]
    losvd = np.asarray(losvd, dtype=float)[xvel_order]
    norm = np.trapz(losvd, xplot)
    if np.isfinite(norm) and norm != 0:
        losvd = losvd / norm
    return losvd


def losvd_moments(xvel, losvd):
    good = np.isfinite(xvel) & np.isfinite(losvd)
    if np.count_nonzero(good) < 3:
        return {key: np.nan for key in MOMENTS}
    x = xvel[good]
    y = losvd[good]
    area = np.sum(y)
    if not np.isfinite(area) or area <= 0:
        return {key: np.nan for key in MOMENTS}
    mean = np.sum(x * y) / area
    var = np.sum((x - mean) ** 2 * y) / area
    if not np.isfinite(var) or var <= 0:
        return {"vel": mean, "sigma": np.nan, "h3": np.nan, "h4": np.nan}
    sigma = np.sqrt(var)
    w = (x - mean) / sigma
    skew = np.sum((w**3) * y) / area
    excess_kurtosis = np.sum((w**4) * y) / area - 3.0
    return {"vel": mean, "sigma": sigma, "h3": skew, "h4": excess_kurtosis}


def percentile_limits(values, low=2.0, high=98.0):
    arr = np.asarray(values, dtype=float)
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return None, None
    lo, hi = np.nanpercentile(good, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(good)
        hi = np.nanmax(good)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def interpolate_to_grid(x, y, values, grid_size=140, preferred_method="cubic"):
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    xi = np.linspace(np.nanmin(x[good]), np.nanmax(x[good]), grid_size)
    yi = np.linspace(np.nanmin(y[good]), np.nanmax(y[good]), grid_size)
    xx, yy = np.meshgrid(xi, yi)

    if np.count_nonzero(good) < 4:
        return xi, yi, np.full_like(xx, np.nan)

    methods = [preferred_method, "linear", "nearest"]
    last = None
    for method in methods:
        try:
            zz = griddata((x[good], y[good]), values[good], (xx, yy), method=method, fill_value=np.nan)
            if np.any(np.isfinite(zz)):
                return xi, yi, zz
            last = zz
        except Exception:
            continue
    if last is None:
        last = np.full_like(xx, np.nan)
    return xi, yi, last


def science_cube_hdu(hdul):
    for hdu in hdul:
        if hdu.data is not None and np.asarray(hdu.data).ndim == 3:
            return hdu
    raise ValueError("No 3D science cube HDU found.")


def orient_cube_nlambda_first(data, header):
    cube = np.asarray(data, dtype=float)
    naxis3 = int(header.get("NAXIS3", cube.shape[0]))
    if cube.shape[0] == naxis3:
        return cube
    if cube.shape[-1] == naxis3:
        return np.moveaxis(cube, -1, 0)
    spectral_axis = int(np.argmax(cube.shape))
    return np.moveaxis(cube, spectral_axis, 0) if spectral_axis != 0 else cube


def wavelength_axis_um(header, nlambda):
    crval = float(header.get("CRVAL3", 0.0))
    crpix = float(header.get("CRPIX3", 1.0))
    cdelt = header.get("CDELT3", header.get("CD3_3", 1.0))
    wave = crval + (np.arange(nlambda, dtype=float) + 1.0 - crpix) * float(cdelt)
    unit = str(header.get("CUNIT3", "")).strip().lower()
    if "ang" in unit:
        wave = wave / 1.0e4
    elif unit == "nm":
        wave = wave / 1.0e3
    elif unit in {"m", "meter", "metre"}:
        wave = wave * 1.0e6
    elif not unit and np.nanmedian(wave) > 100.0:
        wave = wave / 1.0e4
    return wave


def pixel_scale_arcsec(header):
    scale = header.get("CDELT1", header.get("CD1_1", None))
    if scale is None:
        scale = header.get("CDELT2", header.get("CD2_2", None))
    if scale is None:
        return 0.1
    scale = abs(float(scale))
    return scale * 3600.0 if scale < 0.01 else scale


def contour_levels(values, percentiles=CONTOUR_PERCENTILES):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    positive = finite[finite > 0]
    sample = positive if positive.size >= 10 else finite
    levels = np.nanpercentile(sample, percentiles)
    levels = np.unique(np.round(levels[np.isfinite(levels)], 10))
    if levels.size == 0:
        return []
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    return [float(level) for level in levels if lo < level < hi]


def contour_segments(xgrid, ygrid, values, levels, rotate=False):
    if not levels:
        return []
    fig, ax = plt.subplots(figsize=(2, 2))
    try:
        cs = ax.contour(xgrid, ygrid, values, levels=levels)
        contours = []
        for level, segments in zip(cs.levels, cs.allsegs):
            packed = []
            for segment in segments:
                if len(segment) < 2:
                    continue
                x = segment[:, 0]
                y = segment[:, 1]
                if rotate:
                    x, y = rotate_coordinates(x, y)
                    segment = np.column_stack([x, y])
                packed.append(compact_segment(segment))
            if packed:
                contours.append({"level": finite_float(level), "segments": packed})
        return contours
    finally:
        plt.close(fig)


def read_ifu_image(cube_path: Path, grid_size: int):
    if cube_path is None or not cube_path.exists():
        return None

    with fits.open(cube_path, memmap=False) as hdul:
        hdu = science_cube_hdu(hdul)
        cube = orient_cube_nlambda_first(hdu.data, hdu.header)
        wave_um = wavelength_axis_um(hdu.header, cube.shape[0])
        pixsize = pixel_scale_arcsec(hdu.header)

    fit = (wave_um >= CUBE_FIT_RANGE_UM[0]) & (wave_um <= CUBE_FIT_RANGE_UM[1])
    if np.count_nonzero(fit) == 0:
        fit = np.isfinite(wave_um)
    image = np.nanmedian(cube[fit], axis=0)
    image = np.asarray(image, dtype=float)

    signal_for_center = np.where(np.isfinite(image), image, -np.inf)
    if not np.any(np.isfinite(signal_for_center)):
        return None
    center_row, center_col = np.unravel_index(int(np.nanargmax(signal_for_center)), image.shape)

    rows, cols = np.indices(image.shape)
    x = (cols - center_col) * pixsize
    y = (rows - center_row) * pixsize
    xrot, yrot = rotate_coordinates(x, y)

    xi, yi, zz = interpolate_to_grid(
        xrot.ravel(),
        yrot.ravel(),
        image.ravel(),
        grid_size=grid_size,
        preferred_method="linear",
    )
    levels = contour_levels(image)
    image_contours = contour_segments(x, y, image, levels, rotate=True)
    auto_lo = float(np.nanmin(image[np.isfinite(image)]))
    auto_hi = float(np.nanmax(image[np.isfinite(image)]))
    robust_lo, robust_hi = percentile_limits(image, low=1.0, high=99.0)

    return {
        "source": str(cube_path),
        "fitRangeUm": [CUBE_FIT_RANGE_UM[0], CUBE_FIT_RANGE_UM[1]],
        "center": {
            "row": int(center_row),
            "col": int(center_col),
            "xArcsec": 0.0,
            "yArcsec": 0.0,
        },
        "pixscaleArcsec": float(pixsize),
        "shape": [int(image.shape[0]), int(image.shape[1])],
        "extent": [
            float(np.nanmin(xrot)),
            float(np.nanmax(xrot)),
            float(np.nanmin(yrot)),
            float(np.nanmax(yrot)),
        ],
        "values": compact_array(image, ndigits=8),
        "grid": {
            "x": compact_array(xi, ndigits=6),
            "y": compact_array(yi, ndigits=6),
            "z": compact_array(zz, ndigits=8),
            "nx": int(len(xi)),
            "ny": int(len(yi)),
        },
        "contours": image_contours,
        "stats": {
            "auto": [auto_lo, auto_hi],
            "robust": [robust_lo, robust_hi],
            "symmetric": [-max(abs(auto_lo), abs(auto_hi)), max(abs(auto_lo), abs(auto_hi))],
        },
    }


def combined_extent(results, image_payload=None):
    extents = [result["extent"] for result in results]
    if image_payload is not None:
        extents.append(image_payload["extent"])
    arr = np.asarray(extents, dtype=float)
    return [
        float(np.nanmin(arr[:, 0])),
        float(np.nanmax(arr[:, 1])),
        float(np.nanmin(arr[:, 2])),
        float(np.nanmax(arr[:, 3])),
    ]


def read_mge_overlay(mge_table: Path, extent, grid_size: int):
    if mge_table is None or not mge_table.exists():
        return None

    table = np.genfromtxt(mge_table, delimiter=",", names=True, dtype=float)
    table = np.atleast_1d(table)
    names = table.dtype.names or ()
    sigma = np.asarray(table["sigma_arcsec"], dtype=float)
    q = np.asarray(table["q_obs"], dtype=float)
    if "luminosity_Lsun" in names:
        weight = np.asarray(table["luminosity_Lsun"], dtype=float)
    elif "total_flux" in names:
        weight = np.asarray(table["total_flux"], dtype=float)
    else:
        weight = np.ones_like(sigma)

    valid = np.isfinite(sigma) & np.isfinite(q) & np.isfinite(weight) & (sigma > 0) & (q > 0) & (weight > 0)
    sigma = sigma[valid]
    q = q[valid]
    weight = weight[valid]
    if sigma.size == 0:
        return None

    xmin, xmax, ymin, ymax = extent
    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(xi, yi)
    model = np.zeros_like(xx, dtype=float)
    for sig, flattening, flux in zip(sigma, q, weight):
        amp = flux / (2.0 * np.pi * sig**2 * flattening)
        radius_term = (xx / sig) ** 2 + (yy / (sig * flattening)) ** 2
        model += amp * np.exp(-0.5 * radius_term)

    levels = contour_levels(model)
    return {
        "source": str(mge_table),
        "center": [0.0, 0.0],
        "frame": "rotated map frame; major axis horizontal",
        "nGaussians": int(sigma.size),
        "contours": contour_segments(xx, yy, model, levels, rotate=False),
        "levels": [finite_float(level) for level in levels],
    }


def read_result(path: Path, label: str | None, grid_size: int):
    with h5py.File(path, "r") as handle:
        fit_type = str(handle.attrs.get("fit_type", path.parent.name)).strip() or path.parent.name
        x_raw = np.asarray(handle["in/xbin"], dtype=float)
        y_raw = np.asarray(handle["in/ybin"], dtype=float)
        x, y = rotate_coordinates(x_raw, y_raw)
        bin_snr = np.asarray(handle["in/bin_snr"], dtype=float) if "in/bin_snr" in handle else np.full_like(x, np.nan)
        xvel = np.asarray(handle["in/xvel"], dtype=float)
        order = np.argsort(xvel)
        xplot = xvel[order]
        bin_ids = sorted((int(k) for k in handle["out"].keys()))

        moments = {key: np.full(len(x), np.nan) for key in MOMENTS}
        losvds = {}
        bins = []
        for bin_id in bin_ids:
            group = handle[f"out/{bin_id}"]
            losvd = normalized_losvd_for_bin(group, order, xplot)
            derived = losvd_moments(xplot, losvd)
            for key in MOMENTS:
                moments[key][bin_id] = derived[key]
            losvds[str(bin_id)] = compact_array(losvd, ndigits=8)
            bins.append(
                {
                    "id": bin_id,
                    "x": finite_float(x[bin_id]),
                    "y": finite_float(y[bin_id]),
                    "r": finite_float(np.hypot(x_raw[bin_id], y_raw[bin_id])),
                    "snr": finite_float(bin_snr[bin_id]) if bin_id < len(bin_snr) else None,
                }
            )

        grids = {}
        stats = {}
        for key in MOMENTS:
            values = moments[key]
            good = np.isfinite(values)
            if not np.any(good):
                grids[key] = None
                stats[key] = None
                continue
            xi, yi, zz = interpolate_to_grid(x, y, values, grid_size=grid_size)
            auto_lo = float(np.nanmin(values[good]))
            auto_hi = float(np.nanmax(values[good]))
            robust_lo, robust_hi = percentile_limits(values)
            absmax = float(np.nanmax(np.abs(values[good])))
            grids[key] = {
                "x": compact_array(xi, ndigits=6),
                "y": compact_array(yi, ndigits=6),
                "z": compact_array(zz, ndigits=6),
                "nx": int(len(xi)),
                "ny": int(len(yi)),
            }
            stats[key] = {
                "auto": [auto_lo, auto_hi],
                "robust": [robust_lo, robust_hi],
                "symmetric": [-absmax, absmax],
            }

        return {
            "label": label or path.parent.name,
            "fitType": fit_type,
            "source": str(path),
            "extent": [
                float(np.nanmin(x)),
                float(np.nanmax(x)),
                float(np.nanmin(y)),
                float(np.nanmax(y)),
            ],
            "xvel": compact_array(xplot, ndigits=6),
            "bins": bins,
            "moments": {key: compact_array(values, ndigits=6) for key, values in moments.items()},
            "losvds": losvds,
            "grids": grids,
            "stats": stats,
        }


def html_template(payload):
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BAYES-LOSVD NIRSpec Dashboard</title>
<style>
:root {{
  --bg: #f4f5f2;
  --panel: #ffffff;
  --ink: #172026;
  --muted: #5b6870;
  --line: #d8dedc;
  --accent: #0f766e;
  --accent-2: #b6465f;
  --shadow: 0 12px 34px rgba(23, 32, 38, 0.08);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}}
header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 24px 12px;
  border-bottom: 1px solid var(--line);
}}
h1 {{
  margin: 0;
  font-size: 22px;
  line-height: 1.15;
  font-weight: 760;
  letter-spacing: 0;
}}
.meta {{
  margin-top: 4px;
  color: var(--muted);
  font-size: 13px;
}}
.toolbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding: 14px 24px;
  align-items: end;
}}
.group {{
  min-width: 0;
}}
.result-group {{ flex: 1 1 190px; }}
.moment-group {{ flex: 1.5 1 330px; }}
.view-group {{ flex: 0.9 1 210px; }}
.overlay-group {{ flex: 1 1 260px; }}
.limits-group {{ flex: 1.7 1 480px; }}
label {{
  display: block;
  font-size: 11px;
  font-weight: 720;
  color: var(--muted);
  margin: 0 0 6px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
select, input {{
  width: 100%;
  height: 36px;
  border: 1px solid var(--line);
  background: white;
  color: var(--ink);
  border-radius: 7px;
  padding: 0 10px;
  font: inherit;
  font-size: 14px;
}}
.segmented, .inline {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}}
.checks {{
  min-height: 36px;
  align-items: center;
}}
.check {{
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 36px;
  margin: 0;
  padding: 0 10px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: white;
  color: var(--ink);
  font-size: 13px;
  font-weight: 650;
  text-transform: none;
  letter-spacing: 0;
}}
.check input {{
  width: auto;
  height: auto;
  margin: 0;
}}
.check:has(input:disabled) {{
  color: #a2aca9;
  background: #eef1ef;
}}
button {{
  height: 36px;
  border: 1px solid var(--line);
  background: white;
  color: var(--ink);
  border-radius: 7px;
  padding: 0 12px;
  font: inherit;
  font-size: 14px;
  cursor: pointer;
}}
button:hover {{ border-color: #9aa6a4; }}
button.active {{
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}}
button:disabled {{
  color: #a2aca9;
  background: #eef1ef;
  cursor: not-allowed;
}}
.limits {{
  display: grid;
  grid-template-columns: minmax(90px, 1fr) minmax(90px, 1fr) repeat(3, auto);
  gap: 6px;
}}
main {{
  display: grid;
  grid-template-columns: minmax(620px, 1.35fr) minmax(430px, 0.85fr);
  gap: 16px;
  padding: 0 24px 24px;
}}
.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  min-width: 0;
}}
.panel-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px 0;
}}
.panel-title {{
  font-weight: 760;
  font-size: 15px;
}}
.status {{
  color: var(--muted);
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.map-shell {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 76px;
  gap: 8px;
  padding: 8px 14px 14px;
}}
canvas {{
  display: block;
  width: 100%;
}}
#mapCanvas {{
  height: min(68vh, 720px);
  min-height: 520px;
  cursor: crosshair;
}}
#colorbarCanvas {{
  height: min(68vh, 720px);
  min-height: 520px;
}}
.side {{
  display: grid;
  grid-template-rows: auto auto minmax(280px, 1fr);
  gap: 12px;
}}
.summary {{
  padding: 14px;
}}
.summary-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}}
.metric {{
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 9px 10px;
}}
.metric .name {{
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.metric .value {{
  margin-top: 4px;
  font-size: 18px;
  font-weight: 760;
  font-variant-numeric: tabular-nums;
}}
.select-row {{
  display: grid;
  grid-template-columns: minmax(90px, 1fr) auto auto;
  gap: 6px;
  margin-top: 12px;
}}
.chips {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 12px;
  min-height: 36px;
}}
.chip {{
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 30px;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0 8px 0 10px;
  font-size: 13px;
  background: #f9faf8;
}}
.chip.active {{
  border-color: var(--accent);
  color: var(--accent);
  font-weight: 720;
}}
.chip button {{
  border: 0;
  padding: 0;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  color: inherit;
  background: transparent;
}}
.losvd-panel {{
  padding: 12px 14px 14px;
}}
#losvdCanvas {{
  height: 330px;
}}
footer {{
  color: var(--muted);
  font-size: 12px;
  padding: 0 24px 18px;
}}
@media (max-width: 1160px) {{
  .toolbar, main {{
    grid-template-columns: 1fr;
  }}
  #mapCanvas, #colorbarCanvas {{
    height: 560px;
  }}
}}
</style>
</head>
<body>
<header>
  <div>
    <h1>BAYES-LOSVD NIRSpec Dashboard</h1>
    <div class="meta">AGN-subtracted G235H cube, MAP products, maps rotated -18 deg</div>
  </div>
  <div class="status" id="sourceStatus"></div>
</header>

<section class="toolbar">
  <div class="group result-group">
    <label for="resultSelect">Result</label>
    <select id="resultSelect"></select>
  </div>
  <div class="group moment-group">
    <label>Moment</label>
    <div class="segmented" id="momentButtons"></div>
  </div>
  <div class="group view-group">
    <label>Map View</label>
    <div class="segmented" id="viewButtons">
      <button type="button" data-view="bins" class="active">Bins</button>
      <button type="button" data-view="interp">Interpolated</button>
    </div>
  </div>
  <div class="group overlay-group">
    <label>Overlays</label>
    <div class="inline checks">
      <label class="check"><input id="mgeToggle" type="checkbox"> MGE</label>
      <label class="check"><input id="imageIsoToggle" type="checkbox"> Image isophotes</label>
    </div>
  </div>
  <div class="group limits-group">
    <label>Colorbar Limits</label>
    <div class="limits">
      <input id="vminInput" type="number" step="any" aria-label="Colorbar minimum">
      <input id="vmaxInput" type="number" step="any" aria-label="Colorbar maximum">
      <button type="button" id="autoBtn">Auto</button>
      <button type="button" id="robustBtn">Robust</button>
      <button type="button" id="symmetricBtn">Symmetric</button>
    </div>
  </div>
</section>

<main>
  <section class="panel">
    <div class="panel-head">
      <div class="panel-title" id="mapTitle">Velocity</div>
      <div class="status" id="mapStatus"></div>
    </div>
    <div class="map-shell">
      <canvas id="mapCanvas"></canvas>
      <canvas id="colorbarCanvas"></canvas>
    </div>
  </section>

  <section class="side">
    <section class="panel summary">
      <div class="panel-title">Selected Bin</div>
      <div class="summary-grid" id="metricGrid"></div>
      <div class="select-row">
        <input id="binInput" type="number" min="0" step="1" placeholder="Bin ID">
        <button type="button" id="addBinBtn">Add</button>
        <button type="button" id="clearBtn">Clear</button>
      </div>
      <div class="chips" id="selectedChips"></div>
    </section>

    <section class="panel losvd-panel">
      <div class="panel-head" style="padding:0 0 8px">
        <div class="panel-title">LOSVD</div>
        <div class="status" id="losvdStatus"></div>
      </div>
      <canvas id="losvdCanvas"></canvas>
    </section>
  </section>
</main>

<footer id="dashboardFooter"></footer>

<script>
const DATA = {data_json};

const moments = [
  {{key: "vel", label: "Velocity", unit: "km/s", cmap: "RdBu_r"}},
  {{key: "sigma", label: "Sigma", unit: "km/s", cmap: "magma"}},
  {{key: "h3", label: "h3 / skew", unit: "", cmap: "coolwarm"}},
  {{key: "h4", label: "h4 / kurt", unit: "", cmap: "coolwarm"}},
  {{key: "image", label: "Image", unit: "cube flux", cmap: "gray"}}
];
const palette = ["#0f766e", "#b6465f", "#2f5aa8", "#c27418", "#6f4aa7", "#54733c", "#8f3b32", "#2f747f"];
const LOSVD_X_LIMIT = 1200;
const state = {{
  result: 0,
  moment: "vel",
  view: "bins",
  showMge: false,
  showImageContours: false,
  selected: [],
  active: null,
  vmin: null,
  vmax: null,
  plot: null
}};

const resultSelect = document.getElementById("resultSelect");
const momentButtons = document.getElementById("momentButtons");
const viewButtons = document.getElementById("viewButtons");
const mgeToggle = document.getElementById("mgeToggle");
const imageIsoToggle = document.getElementById("imageIsoToggle");
const vminInput = document.getElementById("vminInput");
const vmaxInput = document.getElementById("vmaxInput");
const autoBtn = document.getElementById("autoBtn");
const robustBtn = document.getElementById("robustBtn");
const symmetricBtn = document.getElementById("symmetricBtn");
const mapCanvas = document.getElementById("mapCanvas");
const cbarCanvas = document.getElementById("colorbarCanvas");
const losvdCanvas = document.getElementById("losvdCanvas");
const sourceStatus = document.getElementById("sourceStatus");
const mapStatus = document.getElementById("mapStatus");
const mapTitle = document.getElementById("mapTitle");
const metricGrid = document.getElementById("metricGrid");
const binInput = document.getElementById("binInput");
const addBinBtn = document.getElementById("addBinBtn");
const clearBtn = document.getElementById("clearBtn");
const selectedChips = document.getElementById("selectedChips");
const losvdStatus = document.getElementById("losvdStatus");
const footer = document.getElementById("dashboardFooter");

function currentResult() {{
  return DATA.results[state.result];
}}

function momentInfo(key = state.moment) {{
  return moments.find((m) => m.key === key);
}}

function finiteValues(values) {{
  return values.filter((v) => Number.isFinite(v));
}}

function formatValue(value, unit = "") {{
  if (!Number.isFinite(value)) return "n/a";
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 1 : abs >= 10 ? 2 : 3;
  return `${{value.toFixed(digits)}}${{unit ? " " + unit : ""}}`;
}}

function setupCanvas(canvas) {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {{ctx, width: rect.width, height: rect.height}};
}}

function clamp01(x) {{
  return Math.max(0, Math.min(1, x));
}}

function hexToRgb(hex) {{
  const s = hex.replace("#", "");
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}}

function interpColor(stops, t) {{
  t = clamp01(t);
  for (let i = 0; i < stops.length - 1; i += 1) {{
    const a = stops[i];
    const b = stops[i + 1];
    if (t >= a[0] && t <= b[0]) {{
      const local = (t - a[0]) / Math.max(1e-12, b[0] - a[0]);
      const ca = hexToRgb(a[1]);
      const cb = hexToRgb(b[1]);
      const r = Math.round(ca[0] + local * (cb[0] - ca[0]));
      const g = Math.round(ca[1] + local * (cb[1] - ca[1]));
      const bl = Math.round(ca[2] + local * (cb[2] - ca[2]));
      return `rgb(${{r}},${{g}},${{bl}})`;
    }}
  }}
  return stops[stops.length - 1][1];
}}

function colorFor(value, cmap, vmin, vmax) {{
  if (!Number.isFinite(value)) return "rgba(180, 188, 184, 0.25)";
  let t = (value - vmin) / Math.max(1e-12, vmax - vmin);
  if (cmap === "gray") {{
    return interpColor([[0, "#11181c"], [0.22, "#303a3d"], [0.58, "#a8b1a9"], [1, "#fbfbf4"]], t);
  }}
  if (cmap === "magma") {{
    return interpColor([[0, "#08051f"], [0.25, "#4b1279"], [0.5, "#b53679"], [0.75, "#f47d3b"], [1, "#fcfdbf"]], t);
  }}
  if (cmap === "coolwarm") {{
    return interpColor([[0, "#3b4cc0"], [0.5, "#f6f7f7"], [1, "#b40426"]], t);
  }}
  return interpColor([[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]], t);
}}

function defaultLimits(mode = "robust") {{
  const stats = state.moment === "image" ? DATA.image?.stats : currentResult().stats[state.moment];
  if (!stats) return [0, 1];
  const pair = stats[mode] || stats.robust || stats.auto;
  if (!pair || pair[0] === null || pair[1] === null || pair[1] <= pair[0]) return [0, 1];
  return pair;
}}

function setLimits(vmin, vmax) {{
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmax <= vmin) {{
    [vmin, vmax] = defaultLimits("robust");
  }}
  state.vmin = vmin;
  state.vmax = vmax;
  vminInput.value = Number(vmin.toPrecision(7));
  vmaxInput.value = Number(vmax.toPrecision(7));
  render();
}}

function resetLimits(mode) {{
  const [lo, hi] = defaultLimits(mode);
  setLimits(lo, hi);
}}

function hasMoment(result, key) {{
  if (key === "image") return Boolean(DATA.image && DATA.image.grid);
  const stats = result.stats[key];
  return Boolean(stats);
}}

function initControls() {{
  DATA.results.forEach((result, idx) => {{
    const option = document.createElement("option");
    option.value = String(idx);
    option.textContent = result.label;
    resultSelect.appendChild(option);
  }});

  moments.forEach((m) => {{
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.moment = m.key;
    button.textContent = m.label;
    if (m.key === state.moment) button.classList.add("active");
    button.addEventListener("click", () => {{
      if (button.disabled) return;
      state.moment = m.key;
      resetLimits(m.key === "vel" || m.key === "h3" || m.key === "h4" ? "symmetric" : "robust");
      updateMomentButtons();
    }});
    momentButtons.appendChild(button);
  }});

  viewButtons.querySelectorAll("button").forEach((button) => {{
    button.addEventListener("click", () => {{
      state.view = button.dataset.view;
      viewButtons.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === button));
      render();
    }});
  }});

  const mgeAvailable = Boolean(DATA.mge && DATA.mge.contours && DATA.mge.contours.length);
  const imageContoursAvailable = Boolean(DATA.image && DATA.image.contours && DATA.image.contours.length);
  mgeToggle.disabled = !mgeAvailable;
  imageIsoToggle.disabled = !imageContoursAvailable;
  mgeToggle.checked = state.showMge;
  imageIsoToggle.checked = state.showImageContours;
  mgeToggle.addEventListener("change", () => {{
    state.showMge = mgeToggle.checked;
    render();
  }});
  imageIsoToggle.addEventListener("change", () => {{
    state.showImageContours = imageIsoToggle.checked;
    render();
  }});

  resultSelect.addEventListener("change", () => {{
    state.result = Number(resultSelect.value);
    if (!hasMoment(currentResult(), state.moment)) {{
      state.moment = hasMoment(currentResult(), "vel") ? "vel" : Object.keys(currentResult().stats).find((k) => currentResult().stats[k]);
    }}
    updateMomentButtons();
    resetLimits(state.moment === "sigma" ? "robust" : "symmetric");
  }});

  autoBtn.addEventListener("click", () => resetLimits("auto"));
  robustBtn.addEventListener("click", () => resetLimits("robust"));
  symmetricBtn.addEventListener("click", () => resetLimits("symmetric"));
  vminInput.addEventListener("change", () => setLimits(Number(vminInput.value), state.vmax));
  vmaxInput.addEventListener("change", () => setLimits(state.vmin, Number(vmaxInput.value)));
  addBinBtn.addEventListener("click", () => selectBin(Number(binInput.value), true));
  binInput.addEventListener("keydown", (event) => {{
    if (event.key === "Enter") selectBin(Number(binInput.value), true);
  }});
  clearBtn.addEventListener("click", () => {{
    state.selected = [];
    state.active = null;
    render();
  }});

  mapCanvas.addEventListener("click", handleMapClick);
  window.addEventListener("resize", render);
}}

function updateMomentButtons() {{
  const result = currentResult();
  momentButtons.querySelectorAll("button").forEach((button) => {{
    const key = button.dataset.moment;
    const available = hasMoment(result, key);
    button.disabled = !available;
    button.classList.toggle("active", key === state.moment);
  }});
}}

function xScale(x, plot) {{
  return plot.left + (x - plot.xmin) / (plot.xmax - plot.xmin) * plot.width;
}}

function yScale(y, plot) {{
  return plot.top + (plot.ymax - y) / (plot.ymax - plot.ymin) * plot.height;
}}

function drawAxes(ctx, plot, width, height) {{
  ctx.save();
  ctx.strokeStyle = "#8e9a97";
  ctx.fillStyle = "#48535a";
  ctx.lineWidth = 1;
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.beginPath();
  ctx.rect(plot.left, plot.top, plot.width, plot.height);
  ctx.stroke();

  const ntick = 5;
  for (let i = 0; i < ntick; i += 1) {{
    const f = i / (ntick - 1);
    const xv = plot.xmin + f * (plot.xmax - plot.xmin);
    const xs = xScale(xv, plot);
    ctx.beginPath();
    ctx.moveTo(xs, plot.top + plot.height);
    ctx.lineTo(xs, plot.top + plot.height + 5);
    ctx.stroke();
    ctx.fillText(xv.toFixed(1), xs, plot.top + plot.height + 8);

    const yv = plot.ymin + f * (plot.ymax - plot.ymin);
    const ys = yScale(yv, plot);
    ctx.beginPath();
    ctx.moveTo(plot.left - 5, ys);
    ctx.lineTo(plot.left, ys);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(yv.toFixed(1), plot.left - 8, ys);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
  }}

  ctx.fillStyle = "#172026";
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText("x_rot (arcsec)", plot.left + plot.width / 2, height - 22);
  ctx.save();
  ctx.translate(18, plot.top + plot.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("y_rot (arcsec)", 0, 0);
  ctx.restore();
  ctx.restore();
}}

function drawGrid(ctx, grid, info, plot) {{
  if (!grid) return;
  const nx = grid.nx;
  const ny = grid.ny;
  const xs = grid.x;
  const ys = grid.y;
  const z = grid.z;
  for (let j = 0; j < ny - 1; j += 1) {{
    const y0 = yScale(ys[j], plot);
    const y1 = yScale(ys[j + 1], plot);
    const top = Math.min(y0, y1);
    const h = Math.max(1, Math.abs(y1 - y0) + 0.8);
    for (let i = 0; i < nx - 1; i += 1) {{
      const value = z[j * nx + i];
      if (!Number.isFinite(value)) continue;
      const x0 = xScale(xs[i], plot);
      const x1 = xScale(xs[i + 1], plot);
      const left = Math.min(x0, x1);
      const w = Math.max(1, Math.abs(x1 - x0) + 0.8);
      ctx.fillStyle = colorFor(value, info.cmap, state.vmin, state.vmax);
      ctx.fillRect(left, top, w, h);
    }}
  }}
}}

function drawInterpolated(ctx, result, values, info, plot) {{
  drawGrid(ctx, result.grids[state.moment], info, plot);
  result.bins.forEach((bin) => {{
    const px = xScale(bin.x, plot);
    const py = yScale(bin.y, plot);
    ctx.fillStyle = "rgba(18, 28, 32, 0.25)";
    ctx.beginPath();
    ctx.arc(px, py, 2.1, 0, Math.PI * 2);
    ctx.fill();
  }});
}}

function drawBins(ctx, result, values, info, plot) {{
  result.bins.forEach((bin) => {{
    const value = values[bin.id];
    const px = xScale(bin.x, plot);
    const py = yScale(bin.y, plot);
    ctx.fillStyle = colorFor(value, info.cmap, state.vmin, state.vmax);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.74)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.arc(px, py, 4.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }});
}}

function drawSelection(ctx, result, plot) {{
  state.selected.forEach((id, idx) => {{
    const bin = result.bins[id];
    if (!bin) return;
    const px = xScale(bin.x, plot);
    const py = yScale(bin.y, plot);
    ctx.strokeStyle = palette[idx % palette.length];
    ctx.lineWidth = id === state.active ? 3.2 : 2.0;
    ctx.beginPath();
    ctx.arc(px, py, id === state.active ? 9 : 7, 0, Math.PI * 2);
    ctx.stroke();
  }});
}}

function drawContourSet(ctx, contourSource, plot, color, width, dash = []) {{
  const contours = contourSource?.contours || [];
  if (!contours.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  contours.forEach((contour) => {{
    contour.segments.forEach((segment) => {{
      if (!segment.length) return;
      ctx.beginPath();
      segment.forEach((point, idx) => {{
        const px = xScale(point[0], plot);
        const py = yScale(point[1], plot);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }});
      ctx.stroke();
    }});
  }});
  ctx.restore();
}}

function drawCenterMarker(ctx, plot) {{
  const px = xScale(0, plot);
  const py = yScale(0, plot);
  if (!Number.isFinite(px) || !Number.isFinite(py)) return;
  ctx.save();
  ctx.strokeStyle = "#111827";
  ctx.fillStyle = "#ffffff";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(px - 6, py);
  ctx.lineTo(px + 6, py);
  ctx.moveTo(px, py - 6);
  ctx.lineTo(px, py + 6);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(px, py, 2.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}}

function drawOverlays(ctx, plot) {{
  const anyOverlay = (state.showMge && DATA.mge) || (state.showImageContours && DATA.image);
  if (state.showImageContours && DATA.image) {{
    drawContourSet(ctx, DATA.image, plot, "#f59e0b", 1.45, [6, 4]);
  }}
  if (state.showMge && DATA.mge) {{
    drawContourSet(ctx, DATA.mge, plot, "#101820", 1.55);
  }}
  if (anyOverlay) drawCenterMarker(ctx, plot);
}}

function drawMap() {{
  const result = currentResult();
  const info = momentInfo();
  const isImage = state.moment === "image" && DATA.image;
  const values = isImage ? DATA.image.values : result.moments[state.moment];
  const {{ctx, width, height}} = setupCanvas(mapCanvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const [xmin, xmax, ymin, ymax] = isImage ? DATA.image.extent : result.extent;
  const padX = Math.max(0.05, (xmax - xmin) * 0.04);
  const padY = Math.max(0.05, (ymax - ymin) * 0.04);
  const plot = {{
    left: 58,
    top: 18,
    width: Math.max(100, width - 78),
    height: Math.max(100, height - 72),
    xmin: xmin - padX,
    xmax: xmax + padX,
    ymin: ymin - padY,
    ymax: ymax + padY
  }};
  state.plot = plot;

  if (!values || finiteValues(values).length === 0) {{
    ctx.fillStyle = "#5b6870";
    ctx.font = "16px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${{info.label}} is not available for this result`, width / 2, height / 2);
  }} else {{
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.left, plot.top, plot.width, plot.height);
    ctx.clip();
    if (isImage) {{
      drawGrid(ctx, DATA.image.grid, info, plot);
    }} else if (state.view === "interp") {{
      drawInterpolated(ctx, result, values, info, plot);
    }} else {{
      drawBins(ctx, result, values, info, plot);
    }}
    drawOverlays(ctx, plot);
    drawSelection(ctx, result, plot);
    ctx.restore();
  }}
  drawAxes(ctx, plot, width, height);
  drawColorbar(info);
}}

function drawColorbar(info) {{
  const {{ctx, width, height}} = setupCanvas(cbarCanvas);
  ctx.clearRect(0, 0, width, height);
  const left = 16;
  const top = 18;
  const barW = 20;
  const barH = height - 72;
  for (let j = 0; j < barH; j += 1) {{
    const t = 1 - j / Math.max(1, barH - 1);
    const value = state.vmin + t * (state.vmax - state.vmin);
    ctx.fillStyle = colorFor(value, info.cmap, state.vmin, state.vmax);
    ctx.fillRect(left, top + j, barW, 1);
  }}
  ctx.strokeStyle = "#8e9a97";
  ctx.strokeRect(left, top, barW, barH);
  ctx.fillStyle = "#48535a";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const ntick = 5;
  for (let i = 0; i < ntick; i += 1) {{
    const f = i / (ntick - 1);
    const y = top + barH * (1 - f);
    const value = state.vmin + f * (state.vmax - state.vmin);
    ctx.beginPath();
    ctx.moveTo(left + barW, y);
    ctx.lineTo(left + barW + 5, y);
    ctx.stroke();
    ctx.fillText(formatValue(value), left + barW + 8, y);
  }}
  ctx.save();
  ctx.translate(width - 12, top + barH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#172026";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(info.unit || "value", 0, 0);
  ctx.restore();
}}

function handleMapClick(event) {{
  const result = currentResult();
  const plot = state.plot;
  if (!plot) return;
  const rect = mapCanvas.getBoundingClientRect();
  const sx = event.clientX - rect.left;
  const sy = event.clientY - rect.top;
  if (sx < plot.left || sx > plot.left + plot.width || sy < plot.top || sy > plot.top + plot.height) return;
  let best = null;
  let bestD2 = Infinity;
  result.bins.forEach((bin) => {{
    const px = xScale(bin.x, plot);
    const py = yScale(bin.y, plot);
    const d2 = (px - sx) ** 2 + (py - sy) ** 2;
    if (d2 < bestD2) {{
      bestD2 = d2;
      best = bin.id;
    }}
  }});
  if (best !== null) selectBin(best, true);
}}

function selectBin(id, additive = true) {{
  const result = currentResult();
  if (!Number.isInteger(id) || id < 0 || id >= result.bins.length) return;
  state.active = id;
  if (additive && !state.selected.includes(id)) {{
    state.selected.push(id);
    if (state.selected.length > 8) state.selected.shift();
  }} else if (!additive) {{
    state.selected = [id];
  }}
  if (!state.selected.includes(id)) state.selected.push(id);
  binInput.value = String(id);
  render();
}}

function removeBin(id) {{
  state.selected = state.selected.filter((v) => v !== id);
  if (state.active === id) state.active = state.selected.length ? state.selected[state.selected.length - 1] : null;
  render();
}}

function drawMetrics() {{
  const result = currentResult();
  const id = state.active ?? state.selected[state.selected.length - 1] ?? null;
  const metrics = [];
  if (id === null || !result.bins[id]) {{
    metrics.push(["bin", "n/a"], ["x", "n/a"], ["y", "n/a"], ["V", "n/a"], ["sigma", "n/a"], ["h3/h4", "n/a"]);
  }} else {{
    const bin = result.bins[id];
    metrics.push(
      ["bin", String(id)],
      ["x", formatValue(bin.x, "arcsec")],
      ["y", formatValue(bin.y, "arcsec")],
      ["V", formatValue(result.moments.vel[id], "km/s")],
      ["sigma", formatValue(result.moments.sigma[id], "km/s")],
      ["h3", formatValue(result.moments.h3[id])],
      ["h4", formatValue(result.moments.h4[id])],
      ["S/N", formatValue(bin.snr)]
    );
  }}
  metricGrid.innerHTML = "";
  metrics.forEach(([name, value]) => {{
    const div = document.createElement("div");
    div.className = "metric";
    div.innerHTML = `<div class="name">${{name}}</div><div class="value">${{value}}</div>`;
    metricGrid.appendChild(div);
  }});

  selectedChips.innerHTML = "";
  state.selected.forEach((binId, idx) => {{
    const chip = document.createElement("span");
    chip.className = `chip${{binId === state.active ? " active" : ""}}`;
    chip.style.borderColor = palette[idx % palette.length];
    chip.innerHTML = `<span>bin ${{binId}}</span>`;
    chip.addEventListener("click", () => {{
      state.active = binId;
      render();
    }});
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "x";
    close.addEventListener("click", (event) => {{
      event.stopPropagation();
      removeBin(binId);
    }});
    chip.appendChild(close);
    selectedChips.appendChild(chip);
  }});
}}

function drawLosvd() {{
  const result = currentResult();
  const {{ctx, width, height}} = setupCanvas(losvdCanvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const xvel = result.xvel;
  const selected = state.selected.filter((id) => result.losvds[String(id)]);
  if (!selected.length) {{
    ctx.fillStyle = "#5b6870";
    ctx.font = "15px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No bins selected", width / 2, height / 2);
    losvdStatus.textContent = "";
    return;
  }}
  const allY = selected.flatMap((id) => result.losvds[String(id)].filter((v) => Number.isFinite(v)));
  const xmin = -LOSVD_X_LIMIT;
  const xmax = LOSVD_X_LIMIT;
  const ymin = 0;
  const ymax = Math.max(...allY) * 1.08;
  const plot = {{left: 54, top: 18, width: width - 78, height: height - 62, xmin, xmax, ymin, ymax}};
  ctx.strokeStyle = "#8e9a97";
  ctx.strokeRect(plot.left, plot.top, plot.width, plot.height);

  function xs(v) {{ return plot.left + (v - xmin) / (xmax - xmin) * plot.width; }}
  function ys(v) {{ return plot.top + (ymax - v) / (ymax - ymin) * plot.height; }}

  const zero = xs(0);
  ctx.strokeStyle = "#a2aca9";
  ctx.setLineDash([2, 3]);
  ctx.beginPath();
  ctx.moveTo(zero, plot.top);
  ctx.lineTo(zero, plot.top + plot.height);
  ctx.stroke();
  ctx.setLineDash([]);

  selected.forEach((id, idx) => {{
    const y = result.losvds[String(id)];
    ctx.strokeStyle = palette[idx % palette.length];
    ctx.lineWidth = id === state.active ? 2.8 : 1.8;
    ctx.beginPath();
    y.forEach((value, i) => {{
      const px = xs(xvel[i]);
      const py = ys(value);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }});
    ctx.stroke();
  }});

  ctx.fillStyle = "#48535a";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i < 5; i += 1) {{
    const f = i / 4;
    const xv = xmin + f * (xmax - xmin);
    const px = xs(xv);
    ctx.beginPath();
    ctx.moveTo(px, plot.top + plot.height);
    ctx.lineTo(px, plot.top + plot.height + 5);
    ctx.stroke();
    ctx.fillText(xv.toFixed(0), px, plot.top + plot.height + 8);
  }}
  ctx.fillStyle = "#172026";
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText("velocity (km/s)", plot.left + plot.width / 2, height - 20);
  ctx.save();
  ctx.translate(17, plot.top + plot.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("LOSVD", 0, 0);
  ctx.restore();
  losvdStatus.textContent = selected.map((id) => `bin ${{id}}`).join(", ");
}}

function render() {{
  const result = currentResult();
  const info = momentInfo();
  const isImage = state.moment === "image" && DATA.image;
  resultSelect.value = String(state.result);
  updateMomentButtons();
  sourceStatus.textContent = result.fitType;
  mapTitle.textContent = `${{info.label}} map`;
  const overlayText = [
    state.showMge && DATA.mge ? "MGE" : null,
    state.showImageContours && DATA.image ? "image isophotes" : null
  ].filter(Boolean).join(" + ");
  const baseStatus = isImage
    ? `median IFU image, peak-centered at spaxel (${{DATA.image.center.col}}, ${{DATA.image.center.row}})`
    : `${{state.view === "interp" ? "cubic interpolated" : "bin centers"}}`;
  mapStatus.textContent = overlayText ? `${{baseStatus}}; overlays: ${{overlayText}}` : baseStatus;
  const sources = DATA.results.map((r) => r.source);
  if (DATA.image?.source) sources.push(DATA.image.source);
  if (DATA.mge?.source) sources.push(DATA.mge.source);
  footer.textContent = `Sources: ${{sources.join(" | ")}}`;
  drawMap();
  drawMetrics();
  drawLosvd();
}}

initControls();
updateMomentButtons();
resetLimits("symmetric");
selectBin(0, false);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        action="append",
        nargs="+",
        metavar=("PATH", "LABEL"),
        help="Result HDF5 and optional label. May be supplied multiple times.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid-size", type=int, default=140)
    parser.add_argument("--image-grid-size", type=int, default=170)
    parser.add_argument("--cube", type=Path, default=Path("Data/IFU/david_subs/g235h_agn_sub.fits"))
    parser.add_argument(
        "--mge-table",
        type=Path,
        default=Path("Data/mge_NAGN_0deg_pa_positive_gauss/mge_luminosity_table.csv"),
    )
    args = parser.parse_args()

    if not args.results:
        parser.error("At least one --results PATH [LABEL] entry is required.")

    results = []
    for entry in args.results:
        path = Path(entry[0])
        label = " ".join(entry[1:]) if len(entry) > 1 else None
        results.append(read_result(path, label, grid_size=args.grid_size))

    image_payload = read_ifu_image(args.cube, grid_size=args.image_grid_size)
    mge_payload = read_mge_overlay(
        args.mge_table,
        combined_extent(results, image_payload=image_payload),
        grid_size=args.image_grid_size,
    )

    payload = {
        "generatedBy": "build_kinematics_dashboard.py",
        "gridSize": args.grid_size,
        "imageGridSize": args.image_grid_size,
        "rotationDeg": MAP_ROTATION_DEG,
        "image": image_payload,
        "mge": mge_payload,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_template(payload), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
