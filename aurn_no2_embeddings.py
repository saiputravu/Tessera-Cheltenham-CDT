"""Minimal AURN and TESSERA helpers for `cheltenham_similarity.ipynb`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def pick(cols: list[str] | pd.Index, *candidates: str) -> str | None:
    """Find the first column whose lowercase name contains one of `candidates`."""
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        for lc, orig in low.items():
            if cand in lc:
                return orig
    return None


def load_sites(pollutant: str = "NO2") -> pd.DataFrame:
    """Download AURN metadata and return unique candidate sites for `pollutant`."""
    import pyaurn

    # importMeta() drops duplicate site_id rows before filtering, which can drop NO2 rows.
    meta = pyaurn.__dict__["__download_and_import_RData_file"](
        "http://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData"
    )

    code = pick(meta.columns, "code", "site_id")
    lat = pick(meta.columns, "latitude", "lat")
    lon = pick(meta.columns, "longitude", "lon")
    stype = pick(meta.columns, "site_type", "location_type", "type")
    var = pick(meta.columns, "variable", "parameter", "pollutant")
    name_col = pick(meta.columns, "site_name", "name")
    zone_col = pick(meta.columns, "zone")
    la_col = pick(meta.columns, "local_authority", "authority")

    if None in (code, lat, lon, var):
        sys.exit(f"Missing required metadata columns: {code=}, {lat=}, {lon=}, {var=}")

    meta = meta[meta[var].astype(str).str.upper() == pollutant.upper()]
    col_map: dict[str, str] = {code: "code", lat: "lat", lon: "lon"}
    if stype:
        col_map[stype] = "site_type"
    if name_col:
        col_map[name_col] = "site_name"
    if zone_col:
        col_map[zone_col] = "zone"
    if la_col:
        col_map[la_col] = "local_authority"

    sites = meta[list(col_map)].rename(columns=col_map).drop_duplicates(subset=["code"]).copy()
    if "site_type" not in sites:
        sites["site_type"] = "unknown"
    return sites.reset_index(drop=True)


def annual_pollutant(
    sites: pd.DataFrame,
    year: int,
    pollutant: str = "NO2",
    min_capture: float = 0.75,
) -> pd.DataFrame:
    """Download hourly readings for `year` and keep sites with >= `min_capture` valid hours."""
    from pyaurn import importAURN

    rows: list[dict[str, object]] = []
    target_col_key = pollutant.lower().replace(".", "")
    for i, code in enumerate(sites["code"]):
        try:
            df = importAURN(code, [year])
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        val_col = pick(df.columns, target_col_key, pollutant.lower())
        if val_col is None:
            continue
        series = pd.to_numeric(df[val_col], errors="coerce")
        not_na_pct = float(series.notna().mean())
        if not_na_pct >= min_capture:
            rows.append({"code": code, "no2": float(series.mean()), "notNAPercentage": not_na_pct})
        if (i + 1) % 25 == 0:
            print(f"  fetched {i + 1}/{len(sites)}")

    return sites.merge(pd.DataFrame(rows), on="code")


def extract_full_patches(
    sites: pd.DataFrame,
    year: int,
    out_path: Path,
    half_lat_deg: float = 0.0029,
) -> None:
    """Download and cache full raw (H, W, 128) TESSERA patches for every site."""
    from geotessera import GeoTesseraZarr

    gt = GeoTesseraZarr()
    patches: dict[str, np.ndarray] = {}
    for _, row in sites.iterrows():
        code = str(row["code"])
        lon, lat = float(row["lon"]), float(row["lat"])
        half_lon_deg = half_lat_deg / max(np.cos(np.radians(lat)), 0.2)
        bbox = (lon - half_lon_deg, lat - half_lat_deg, lon + half_lon_deg, lat + half_lat_deg)
        try:
            mosaic, _, _ = gt.read_region(bbox, year=year)
            patches[code] = np.asarray(mosaic, dtype=np.float32)
        except Exception as e:
            print(f"  patch failed for {code}: {type(e).__name__}")
        if len(patches) % 25 == 0 and len(patches) > 0:
            print(f"  saved {len(patches)}/{len(sites)} raw patches")

    np.savez_compressed(out_path, **patches)
    print(f"Saved {len(patches)} full patches to {out_path}")


def summarize_patches(
    patch_path: Path,
    codes: list[str] | pd.Series,
    radius_300m_px: int = 15,
) -> dict[str, np.ndarray]:
    """Compute 128-D summary vectors from stored raw (H, W, 128) patches."""
    npz = np.load(patch_path)
    point_list, w300_list, w640_mean_list, w640_std_list = [], [], [], []

    for code in [str(c) for c in codes]:
        if code not in npz:
            nan_vec = np.full(128, np.nan, dtype=float)
            point_list.append(nan_vec)
            w300_list.append(nan_vec)
            w640_mean_list.append(nan_vec)
            w640_std_list.append(nan_vec)
            continue

        arr = np.asarray(npz[code], dtype=float)
        h, w, _ = arr.shape
        cy, cx = h // 2, w // 2

        point_list.append(arr[cy, cx])

        y0, y1 = max(0, cy - radius_300m_px), min(h, cy + radius_300m_px)
        x0, x1 = max(0, cx - radius_300m_px), min(w, cx + radius_300m_px)
        flat_300 = arr[y0:y1, x0:x1].reshape(-1, 128)
        ok_300 = np.isfinite(flat_300).all(axis=1)
        w300_list.append(np.mean(flat_300[ok_300], axis=0) if ok_300.any() else np.full(128, np.nan))

        flat_640 = arr.reshape(-1, 128)
        ok_640 = np.isfinite(flat_640).all(axis=1)
        if ok_640.any():
            w640_mean_list.append(np.mean(flat_640[ok_640], axis=0))
            w640_std_list.append(np.std(flat_640[ok_640], axis=0))
        else:
            w640_mean_list.append(np.full(128, np.nan))
            w640_std_list.append(np.full(128, np.nan))

    return {
        "point": np.asarray(point_list, dtype=float),
        "window_300m": np.asarray(w300_list, dtype=float),
        "window_640m_mean": np.asarray(w640_mean_list, dtype=float),
        "window_640m_std": np.asarray(w640_std_list, dtype=float),
    }
