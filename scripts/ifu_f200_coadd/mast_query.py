from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from filter_catalog import compatible_filters_for_ifu, normalize_filter_name
from fits_utils import selected_hdu


@dataclass
class MastQueryResult:
    image_paths: list[Path]
    download_dir: Path
    observations_path: Path
    products_path: Path
    candidate_products_path: Path
    filtered_products_path: Path
    fallback_options_path: Path | None
    download_manifest_path: Path | None
    requested_filter_name: str
    center_ra_deg: float
    center_dec_deg: float
    radius_arcsec: float
    filter_name: str
    instrument_name: str
    product_subgroup: str
    n_observations: int
    n_products: int
    n_candidate_products: int
    n_filtered_products: int
    n_seed_products: int
    n_downloaded: int
    fallback_used: bool
    filters_found: list[str]


def ifu_search_region(
    ifu_path: str | Path,
    *,
    ifu_hdu_index: int | None = None,
    padding_arcsec: float = 5.0,
) -> tuple[SkyCoord, u.Quantity]:
    """Estimate a MAST query center/radius from the IFU celestial footprint."""
    with fits.open(ifu_path, memmap=True) as hdul:
        ifu_hdu = selected_hdu(hdul, ifu_hdu_index, ndim=3)
        ny, nx = ifu_hdu.data.shape[-2:]
        wcs = WCS(ifu_hdu.header).celestial

    x0 = 0.5 * (nx - 1)
    y0 = 0.5 * (ny - 1)
    center = wcs.pixel_to_world(x0, y0)
    corners = wcs.pixel_to_world(
        np.array([0.0, nx - 1.0, nx - 1.0, 0.0]),
        np.array([0.0, 0.0, ny - 1.0, ny - 1.0]),
    )
    radius = np.nanmax(center.separation(corners).to_value(u.arcsec)) * u.arcsec
    radius = radius + float(padding_arcsec) * u.arcsec
    return center.icrs, radius


def _write_table(path: Path, table: Table, *, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", overwrite=overwrite)
    return path


def _lower_column(table: Table, name: str) -> np.ndarray:
    if name not in table.colnames:
        return np.full(len(table), "", dtype=object)
    return np.asarray([str(value).lower() for value in table[name]], dtype=object)


_FILTER_PATTERN = re.compile(r"\b(F\d{3}(?:W2|W|M|N)|WLP4)\b", re.IGNORECASE)


def _filter_sort_key(filter_name: str) -> tuple[int, str]:
    if filter_name == "WLP4":
        return (10_000, filter_name)
    match = re.search(r"\d{3}", filter_name)
    return (int(match.group(0)) if match else 9_999, filter_name)


def filters_found_in_tables(products: Table, observations: Table | None = None) -> list[str]:
    """Extract unique NIRCam filter names from MAST product/observation tables."""
    names: set[str] = set()
    tables = [products]
    if observations is not None:
        tables.append(observations)

    for table in tables:
        for column in ("filters", "filter", "productFilename", "obs_id", "obsid"):
            if column not in table.colnames:
                continue
            for value in table[column]:
                names.update(normalize_filter_name(match.group(1)) for match in _FILTER_PATTERN.finditer(str(value)))

    return sorted(names, key=_filter_sort_key)


def _project_lonlat(ra_deg: np.ndarray, dec_deg: np.ndarray, center: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    """Project small sky regions to local tangent-plane arcsec coordinates."""
    dra = (ra_deg - center.ra.deg + 180.0) % 360.0 - 180.0
    x = dra * np.cos(np.deg2rad(center.dec.deg)) * 3600.0
    y = (dec_deg - center.dec.deg) * 3600.0
    return x, y


def s_region_to_geometry(s_region: str, center: SkyCoord):
    """Convert a MAST s_region value to a Shapely geometry in local arcsec coordinates."""
    try:
        from shapely.geometry import Point, Polygon
    except ImportError as exc:
        raise RuntimeError("MAST footprint connectivity requires shapely.") from exc

    tokens = str(s_region).replace(",", " ").split()
    if not tokens:
        return None
    shape = tokens[0].upper()
    offset = 2 if len(tokens) > 1 and tokens[1].upper() in {"ICRS", "FK5"} else 1
    values = [float(token) for token in tokens[offset:]]

    if shape == "POLYGON" and len(values) >= 6:
        coords = np.asarray(values, dtype=float).reshape(-1, 2)
        x, y = _project_lonlat(coords[:, 0], coords[:, 1], center)
        geom = Polygon(zip(x, y))
        if not geom.is_valid:
            geom = geom.buffer(0)
        return geom

    if shape == "CIRCLE" and len(values) >= 3:
        x, y = _project_lonlat(np.asarray([values[0]]), np.asarray([values[1]]), center)
        return Point(float(x[0]), float(y[0])).buffer(float(values[2]) * 3600.0)

    return None


def _obsid_to_geometry(observations: Table, center: SkyCoord) -> dict[str, object]:
    mapping = {}
    if "obsid" not in observations.colnames or "s_region" not in observations.colnames:
        return mapping
    for row in observations:
        geom = s_region_to_geometry(row["s_region"], center)
        if geom is not None and not geom.is_empty:
            mapping[str(row["obsid"])] = geom
    return mapping


def _product_obsid(row) -> str | None:
    for key in ("parent_obsid", "obsID", "obsid"):
        if key in row.colnames:
            value = row[key]
            if value is not None and str(value) not in {"", "--"}:
                return str(value)
    return None


def select_connected_footprint_products(
    products: Table,
    observations: Table,
    center: SkyCoord,
    *,
    tolerance_arcsec: float = 1.0,
) -> tuple[Table, int]:
    """Keep the footprint-connected product component seeded by IFU-center overlap."""
    try:
        from shapely.geometry import Point
    except ImportError as exc:
        raise RuntimeError("MAST footprint connectivity requires shapely.") from exc

    if len(products) == 0:
        return products, 0

    obs_geoms = _obsid_to_geometry(observations, center)
    center_point = Point(0.0, 0.0)
    product_geoms = []
    for row in products:
        product_geoms.append(obs_geoms.get(_product_obsid(row)))

    seed = [
        idx
        for idx, geom in enumerate(product_geoms)
        if geom is not None and (geom.buffer(tolerance_arcsec).contains(center_point) or geom.distance(center_point) <= tolerance_arcsec)
    ]
    if not seed:
        return products[:0], 0

    keep = set(seed)
    queue = list(seed)
    while queue:
        idx = queue.pop(0)
        geom_i = product_geoms[idx]
        if geom_i is None:
            continue
        geom_i_buffered = geom_i.buffer(tolerance_arcsec)
        for j, geom_j in enumerate(product_geoms):
            if j in keep or geom_j is None:
                continue
            if geom_i_buffered.intersects(geom_j) or geom_i.distance(geom_j) <= tolerance_arcsec:
                keep.add(j)
                queue.append(j)

    indices = sorted(keep)
    selected = products[indices].copy()
    selected["footprint_seed"] = np.asarray([idx in seed for idx in indices], dtype=bool)
    selected["footprint_component"] = np.ones(len(selected), dtype=int)
    return selected, len(seed)


def filter_jwst_imaging_products(
    products: Table,
    *,
    filter_name: str = "F200W",
    product_subgroup: str = "I2D",
    science_only: bool = True,
    max_products: int | None = None,
) -> Table:
    """Select calibrated JWST imaging products suitable for mosaicking."""
    keep = np.ones(len(products), dtype=bool)
    subgroup = _lower_column(products, "productSubGroupDescription")
    filenames = _lower_column(products, "productFilename")
    extensions = _lower_column(products, "extension")
    product_type = _lower_column(products, "productType")

    keep &= subgroup == product_subgroup.lower()
    keep &= np.char.endswith(filenames.astype(str), ".fits") | (extensions == "fits")
    if filter_name:
        keep &= np.char.find(filenames.astype(str), filter_name.lower()) >= 0
    if science_only:
        keep &= product_type == "science"

    filtered = products[keep]
    if "productFilename" in filtered.colnames and len(filtered) > 1:
        filtered.sort("productFilename")
        _, unique_idx = np.unique(np.asarray(filtered["productFilename"], dtype=str), return_index=True)
        filtered = filtered[np.sort(unique_idx)]
    if max_products is not None and len(filtered) > max_products:
        filtered = filtered[:max_products]
    return filtered


def downloaded_fits_paths(manifest: Table) -> list[Path]:
    """Extract successful local FITS paths from an astroquery MAST download manifest."""
    if "Local Path" not in manifest.colnames:
        return []
    status = _lower_column(manifest, "Status")
    paths = []
    for i, raw_path in enumerate(manifest["Local Path"]):
        if len(status) and status[i] not in {"complete", "downloaded", "local", "exists"}:
            continue
        path = Path(str(raw_path))
        if path.suffix.lower() == ".fits" and path.exists():
            paths.append(path)
    return paths


def _query_observations_and_products(Observations, criteria: dict) -> tuple[Table, Table]:
    observations = Observations.query_criteria(**criteria)
    products = Observations.get_product_list(observations) if len(observations) else Table()
    return observations, products


def _select_products_for_filter(
    products: Table,
    observations: Table,
    center: SkyCoord,
    *,
    filter_name: str,
    product_subgroup: str,
    connectivity: bool,
    connect_tolerance_arcsec: float,
) -> tuple[Table, Table, int]:
    candidates = filter_jwst_imaging_products(
        products,
        filter_name=filter_name,
        product_subgroup=product_subgroup,
        science_only=True,
    )
    if connectivity:
        selected, n_seed = select_connected_footprint_products(
            candidates,
            observations,
            center,
            tolerance_arcsec=connect_tolerance_arcsec,
        )
    else:
        selected = candidates
        n_seed = 0
    return candidates, selected, n_seed


def _choose_fallback_filter(
    options: Table,
    *,
    requested_filter: str,
    mode: str,
    fallback_filter: str | None,
) -> str | None:
    valid = options[np.asarray(options["n_selected"], dtype=int) > 0]
    if len(valid) == 0:
        return None
    valid.sort(["ifu_response_fraction", "n_selected"])
    valid.reverse()

    if fallback_filter:
        fallback_filter = normalize_filter_name(fallback_filter)
        matches = valid[np.asarray(valid["filter"], dtype=str) == fallback_filter]
        if len(matches) == 0:
            names = ", ".join(np.asarray(valid["filter"], dtype=str))
            raise RuntimeError(f"Requested fallback filter {fallback_filter} is not available. Available: {names}")
        return fallback_filter

    if mode == "off":
        return None

    if mode == "auto":
        return str(valid["filter"][0])

    if mode != "prompt":
        raise ValueError(f"Unknown fallback mode: {mode}")

    lines = [
        f"No connected {requested_filter} imaging was found. Compatible alternatives with MAST imaging are:",
    ]
    for idx, row in enumerate(valid, start=1):
        lines.append(
            f"  {idx}. {row['filter']} "
            f"(selected={row['n_selected']}, seed={row['n_seed']}, "
            f"IFU response fraction={row['ifu_response_fraction']:.3f})"
        )
    lines.append("Choose a filter number to use, or press Enter to stop: ")
    prompt = "\n".join(lines)

    if not sys.stdin.isatty():
        names = ", ".join(np.asarray(valid["filter"], dtype=str))
        raise RuntimeError(
            f"No connected {requested_filter} imaging was found. Compatible alternatives are: {names}. "
            "Run interactively, pass --mast-filter-fallback auto, or pass --mast-fallback-filter FILTER."
        )

    answer = input(prompt).strip()
    if not answer:
        return None
    try:
        index = int(answer)
    except ValueError as exc:
        raise RuntimeError(f"Invalid fallback filter choice: {answer}") from exc
    if index < 1 or index > len(valid):
        raise RuntimeError(f"Fallback filter choice {index} is out of range.")
    return str(valid["filter"][index - 1])


def query_and_download_matching_jwst_imaging(
    *,
    ifu_path: str | Path,
    output_dir: str | Path,
    ifu_hdu_index: int | None = None,
    filter_name: str = "F200W",
    radius_arcsec: float | None = None,
    min_radius_arcsec: float | None = None,
    padding_arcsec: float = 5.0,
    instrument_name: str = "NIRCAM/IMAGE",
    product_subgroup: str = "I2D",
    calib_level: int = 3,
    proposal_id: str | None = None,
    max_products: int | None = None,
    connectivity: bool = True,
    connect_tolerance_arcsec: float = 1.0,
    throughput_dir: str | Path | None = None,
    filter_fallback_mode: str = "prompt",
    fallback_filter: str | None = None,
    fallback_min_response_fraction: float = 0.75,
    dry_run: bool = False,
    overwrite: bool = True,
) -> MastQueryResult:
    """Find and optionally download matching JWST imaging from MAST."""
    try:
        from astroquery.mast import Observations
    except ImportError as exc:
        raise RuntimeError(
            "MAST discovery requires astroquery. Install it in the tool environment "
            "or create the environment from scripts/ifu_f200_coadd/environment.yml."
        ) from exc

    center, footprint_radius = ifu_search_region(
        ifu_path,
        ifu_hdu_index=ifu_hdu_index,
        padding_arcsec=padding_arcsec,
    )
    if radius_arcsec is None:
        query_radius = footprint_radius
        if min_radius_arcsec is not None:
            query_radius = max(query_radius, float(min_radius_arcsec) * u.arcsec)
    else:
        query_radius = float(radius_arcsec) * u.arcsec

    requested_filter_name = normalize_filter_name(filter_name)
    active_filter_name = requested_filter_name

    criteria = {
        "coordinates": center,
        "radius": query_radius,
        "obs_collection": "JWST",
        "dataproduct_type": "image",
        "instrument_name": instrument_name,
        "filters": active_filter_name,
        "calib_level": calib_level,
    }
    if proposal_id:
        criteria["proposal_id"] = proposal_id

    download_dir = Path(output_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    observations, products = _query_observations_and_products(Observations, criteria)
    filters_found = filters_found_in_tables(products, observations)
    candidates, filtered, n_seed_products = _select_products_for_filter(
        products,
        observations,
        center,
        filter_name=active_filter_name,
        product_subgroup=product_subgroup,
        connectivity=connectivity,
        connect_tolerance_arcsec=connect_tolerance_arcsec,
    )

    fallback_options_path = None
    fallback_used = False
    if len(filtered) == 0 and filter_fallback_mode != "off" and throughput_dir is not None:
        compatible, (wave_min, wave_max) = compatible_filters_for_ifu(
            ifu_path,
            throughput_dir,
            ifu_hdu_index=ifu_hdu_index,
            min_response_fraction=fallback_min_response_fraction,
        )
        all_criteria = dict(criteria)
        all_criteria.pop("filters", None)
        all_observations, all_products = _query_observations_and_products(Observations, all_criteria)
        filters_found = filters_found_in_tables(all_products, all_observations)

        option_rows = []
        filter_tables = {}
        for row in compatible:
            alt_filter = normalize_filter_name(row["filter"])
            if alt_filter == requested_filter_name:
                continue
            alt_candidates, alt_selected, alt_seed = _select_products_for_filter(
                all_products,
                all_observations,
                center,
                filter_name=alt_filter,
                product_subgroup=product_subgroup,
                connectivity=connectivity,
                connect_tolerance_arcsec=connect_tolerance_arcsec,
            )
            filter_tables[alt_filter] = (alt_candidates, alt_selected, alt_seed)
            option_rows.append(
                {
                    "filter": alt_filter,
                    "ifu_wave_min_micron": wave_min,
                    "ifu_wave_max_micron": wave_max,
                    "ifu_response_fraction": float(row["ifu_response_fraction"]),
                    "wave_min_micron": float(row["wave_min_micron"]),
                    "wave_max_micron": float(row["wave_max_micron"]),
                    "weighted_mean_micron": float(row["weighted_mean_micron"]),
                    "n_candidates": len(alt_candidates),
                    "n_selected": len(alt_selected),
                    "n_seed": int(alt_seed),
                }
            )

        options = Table(rows=option_rows)
        if len(options):
            options.sort(["ifu_response_fraction", "n_selected"])
            options.reverse()
        fallback_options_path = _write_table(download_dir / "mast_filter_fallback_options.ecsv", options, overwrite=overwrite)
        choice = _choose_fallback_filter(
            options,
            requested_filter=requested_filter_name,
            mode=filter_fallback_mode,
            fallback_filter=fallback_filter,
        )
        if choice is not None:
            active_filter_name = choice
            observations = all_observations
            products = all_products
            candidates, filtered, n_seed_products = filter_tables[choice]
            fallback_used = True

    if max_products is not None and len(filtered) > max_products:
        filtered = filtered[:max_products]

    observations_path = _write_table(download_dir / "mast_observations.ecsv", observations, overwrite=overwrite)
    products_path = _write_table(download_dir / "mast_products_all.ecsv", products, overwrite=overwrite)
    candidate_products_path = _write_table(download_dir / "mast_products_candidates.ecsv", candidates, overwrite=overwrite)
    filtered_products_path = _write_table(download_dir / "mast_products_selected.ecsv", filtered, overwrite=overwrite)

    manifest_path = None
    image_paths: list[Path] = []
    if len(filtered) and not dry_run:
        manifest = Observations.download_products(filtered, download_dir=str(download_dir))
        manifest_path = _write_table(download_dir / "mast_download_manifest.ecsv", manifest, overwrite=overwrite)
        image_paths = downloaded_fits_paths(manifest)

    return MastQueryResult(
        image_paths=image_paths,
        download_dir=download_dir,
        observations_path=observations_path,
        products_path=products_path,
        candidate_products_path=candidate_products_path,
        filtered_products_path=filtered_products_path,
        fallback_options_path=fallback_options_path,
        download_manifest_path=manifest_path,
        requested_filter_name=requested_filter_name,
        center_ra_deg=float(center.ra.deg),
        center_dec_deg=float(center.dec.deg),
        radius_arcsec=float(query_radius.to_value(u.arcsec)),
        filter_name=active_filter_name,
        instrument_name=instrument_name,
        product_subgroup=product_subgroup,
        n_observations=len(observations),
        n_products=len(products),
        n_candidate_products=len(candidates),
        n_filtered_products=len(filtered),
        n_seed_products=n_seed_products,
        n_downloaded=len(image_paths),
        fallback_used=fallback_used,
        filters_found=filters_found,
    )
