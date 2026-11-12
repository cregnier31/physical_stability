"""Readers for the CloudFerro OceanBench forecasts and GLOENS Zarr stores.

Source conventions are adapted from:
cregnier31/eddy-tracking-intercomparison, branch
feature/gloens-ensemble-zos-clean:
- eddy_detect/zarr_forecasts.py
- eddy_load/gloens_mean.py

Those original loaders are SSH-only. This reader explicitly requires both 3D
temperature and salinity; it never estimates stratification from zos.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import xarray as xr


ENDPOINT = "https://s3.waw3-1.cloudferro.com"
BUCKET = "oceanbench-bucket"
MODEL_STORES = {
    "GLONET": ("public/ml-forecast-outputs/glonet", "longitude", "latitude"),
    "GLOW": ("dev/ml-forecast-outputs/glowcascade_v5_ring", "lon", "lat"),
    "GLO12": ("public/GLO12", "longitude", "latitude"),
}
SYSTEMS = (*MODEL_STORES, "GLOENS")

# Candidates are explicit and limited; never silently use zos for N2.
TEMPERATURE_NAMES = ("thetao", "votemper", "temperature", "theta", "CT")
SALINITY_NAMES = ("so", "vosaline", "salinity", "salt", "SA")
DEPTH_NAMES = ("depth", "deptht", "deptho", "z", "lev", "depthu")


@dataclass(frozen=True)
class FieldNames:
    temperature: str
    salinity: str
    depth: str
    latitude: str
    longitude: str


def forecast_store(system: str, init_date: str) -> str:
    """CloudFerro store root. GLOENS requires an explicit 3D store path."""
    system = system.upper()
    if system not in MODEL_STORES:
        raise ValueError(f"{system}: select GLONET/GLOW/GLO12 or pass a GLOENS store")
    if len(init_date) != 8 or not init_date.isdigit():
        raise ValueError("init_date must be YYYYMMDD")
    return f"{BUCKET}/{MODEL_STORES[system][0]}/{init_date}.zarr"


@contextmanager
def open_cloudferro(store: str, endpoint: str = ENDPOINT, anonymous: bool = True) -> Iterator[xr.Dataset]:
    """Open S3 Zarr lazily; use subset before invoking eager GSW."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=anonymous, client_kwargs={"endpoint_url": endpoint})
    mapper = s3fs.S3Map(root=store.removeprefix("s3://").rstrip("/"), s3=fs, check=False)
    with xr.open_zarr(mapper, consolidated=False, chunks="auto") as ds:
        yield ds


def _choose(ds: xr.Dataset, names: tuple[str, ...], label: str, override: str | None = None) -> str:
    if override:
        if override not in ds:
            raise KeyError(f"{label} override '{override}' not found")
        return override
    for name in names:
        if name in ds:
            return name
    raise KeyError(f"No {label} variable among {names}; available: {list(ds.data_vars)}")


def detect_fields(
    ds: xr.Dataset, system: str,
    temperature: str | None = None, salinity: str | None = None,
    depth: str | None = None, latitude: str | None = None,
    longitude: str | None = None,
) -> FieldNames:
    """Require 3D T/S; provide informative failure for SSH-only products."""
    system = system.upper()
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system}; choose {SYSTEMS}")
    lon_default, lat_default = MODEL_STORES.get(system, ("longitude", "latitude"))[1:]
    t = _choose(ds, TEMPERATURE_NAMES, "temperature", temperature)
    s = _choose(ds, SALINITY_NAMES, "salinity", salinity)
    z = _choose(ds, DEPTH_NAMES, "depth", depth)
    la = _choose(ds, (lat_default, "nav_lat", "latitude", "lat"), "latitude", latitude)
    lo = _choose(ds, (lon_default, "nav_lon", "longitude", "lon"), "longitude", longitude)
    if ds[z].ndim != 1:
        raise ValueError("Only fixed 1D depth coordinates are supported")
    zd = ds[z].dims[0]
    if zd not in ds[t].dims or zd not in ds[s].dims:
        raise ValueError(
            f"{system}: temperature/salinity require vertical dimension '{zd}'. "
            "An SSH-only 2D store cannot yield N2; obtain the 3D T/S store."
        )
    if ds[t].ndim < 3 or ds[s].ndim < 3:
        raise ValueError(f"{system}: expected at least 3D temperature and salinity")
    if ds[t].dims != ds[s].dims:
        raise ValueError(f"{system}: T and S dims differ; collocate them before N2")
    if "ens" in ds[t].dims and system != "GLOENS":
        raise ValueError("Unexpected ensemble dimension in a deterministic/AI store")
    if system == "GLOENS" and "ens" not in ds[t].dims:
        raise ValueError("GLOENS: 3D T/S must contain an 'ens' dimension")
    return FieldNames(t, s, z, la, lo)


def _lon_180(lon: xr.DataArray) -> xr.DataArray:
    return ((lon + 180) % 360) - 180


def subset_region(
    ds: xr.Dataset, fields: FieldNames,
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
) -> xr.Dataset:
    """Select limited geographic region without loading the 3D fields.

    Supports 1D rectilinear coordinates and 2D curvilinear GLOENS coordinates.
    Longitude windows must not cross the dateline.
    """
    if not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90):
        raise ValueError("Invalid non-dateline-crossing geographic bounds")
    lon = ds[fields.longitude]
    lat = ds[fields.latitude]
    if lon.ndim == lat.ndim == 1:
        lon_idx = np.flatnonzero(
            (_lon_180(lon).values >= lon_min) & (_lon_180(lon).values <= lon_max)
        )
        lat_idx = np.flatnonzero((lat.values >= lat_min) & (lat.values <= lat_max))
        if not len(lon_idx) or not len(lat_idx):
            raise ValueError("No horizontal points in requested region")
        return ds.isel({lon.dims[0]: lon_idx, lat.dims[0]: lat_idx})
    if lon.ndim == lat.ndim == 2 and lon.dims == lat.dims:
        mask = (
            (_lon_180(lon) >= lon_min) & (_lon_180(lon) <= lon_max) &
            (lat >= lat_min) & (lat <= lat_max)
        )
        # Computing a small coordinate-only mask is acceptable.
        found = np.argwhere(np.asarray(mask.values))
        if not found.size:
            raise ValueError("No horizontal points in requested region")
        (y0, x0), (y1, x1) = found.min(axis=0), found.max(axis=0)
        cropped = ds.isel({lon.dims[0]: slice(y0, y1 + 1), lon.dims[1]: slice(x0, x1 + 1)})
        cropped_mask = mask.isel({lon.dims[0]: slice(y0, y1 + 1), lon.dims[1]: slice(x0, x1 + 1)})
        return cropped.where(cropped_mask)
    raise ValueError("Longitude and latitude must both be 1D or both 2D")


def select_day(
    ds: xr.Dataset, first_day: int = 0, last_day: int | None = None,
    time_dim: str = "time",
) -> xr.Dataset:
    """Select forecast lead indices [first_day, last_day), not initialization dates."""
    if time_dim not in ds.dims:
        raise KeyError(f"Forecast requires '{time_dim}' dimension")
    stop = ds.sizes[time_dim] if last_day is None else min(last_day, ds.sizes[time_dim])
    if not (0 <= first_day < stop):
        raise ValueError("Empty/invalid forecast lead window")
    return ds.isel({time_dim: slice(first_day, stop)})


def prepare_ts(
    ds: xr.Dataset, fields: FieldNames, system: str, ensemble: str = "mean",
) -> tuple[xr.Dataset, dict]:
    """Normalize 3D T/S names, while preserving original coordinates/dimensions.

    'mean' forms mean(T), mean(S) before EOS. For ensemble mean(N2), use
    'members' to retain ensemble members and summarize N2 afterwards.
    """
    if ensemble not in ("mean", "members"):
        raise ValueError("ensemble must be 'mean' or 'members'")
    system = system.upper()
    t, s = ds[fields.temperature], ds[fields.salinity]
    if system == "GLOENS" and ensemble == "mean":
        if ds.sizes["ens"] != 50:
            raise ValueError(f"Expected 50 GLOENS members; found {ds.sizes['ens']}")
        # Preserve the same member mask for T and S; do not mix unpaired members.
        paired = np.isfinite(t) & np.isfinite(s)
        t = t.where(paired).mean("ens", skipna=True)
        s = s.where(paired).mean("ens", skipna=True)
    return xr.Dataset(
        {"thetao": t, "so": s},
        coords={
            "deptht": ds[fields.depth].rename({fields.depth: "deptht"})
                      if fields.depth != "deptht" else ds[fields.depth],
            "nav_lat": ds[fields.latitude],
            "nav_lon": ds[fields.longitude],
        },
        attrs={"source_system": system, "ensemble_processing": ensemble},
    ), {"temperature": "thetao", "salinity": "so", "depth": "deptht",
        "latitude": "nav_lat", "longitude": "nav_lon"}
