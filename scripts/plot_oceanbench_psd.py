#!/usr/bin/env python3
"""Compare OceanBench-inspired zonal SSH/SST PSD across GLOENS/GLONET/GLOW/GLO12.

Reference: mercator-ocean/oceanbench OB_AI-TT_DEMO oceanbench/core/psd.py
and assets/glonet_demo_sample.report.ipynb.
This is a regional 1D zonal FFT adaptation, not a bitwise match to xrft.
GLOENS CLASS4 mean .npy files have no SSH; GLOENS contributes SST only.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physical_stability.forecasts import ENDPOINT, forecast_store, open_cloudferro
from physical_stability.gloens_npy import load_gloens_mean, DEFAULT_GLOENS_GRID
from physical_stability.psd import zonal_psd, band_metrics

VARIABLES = {"sst": ("thetao", "votemper", "temperature", "theta", "CT"),
             "ssh": ("zos", "ssh", "sossheig", "sea_surface_height_above_geoid")}
SYSTEMS = ("GLOENS", "GLONET", "GLOW", "GLO12")


def field_and_coords(ds, variable, lat_min, lat_max, lon_min, lon_max, depth_index):
    names = VARIABLES[variable]
    name = next((n for n in names if n in ds), None)
    if name is None:
        raise KeyError(f"Missing {variable} ({names}); data variables: {list(ds.data_vars)}")
    da = ds[name]
    if variable == "sst":
        depth_dim = next((d for d in da.dims if d.lower() in
                          ("depth", "deptht", "deptho", "z", "lev", "depthu")), None)
        if depth_dim is None:
            raise ValueError(f"{name}: no vertical dimension")
        da = da.isel({depth_dim: depth_index})
    for dim in list(da.dims):
        if dim == "time":
            if da.sizes[dim] != 1:
                raise ValueError("Expected one time after lead selection")
            da = da.isel({dim: 0}, drop=True)
        elif dim == "ens":
            da = da.mean(dim, skipna=True)
    lat_name = next((v for v in ("latitude", "lat", "nav_lat") if v in ds), None)
    lon_name = next((v for v in ("longitude", "lon", "nav_lon") if v in ds), None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"Missing geographic coordinates; available: {list(ds.coords)}")
    lat, lon = ds[lat_name], ds[lon_name]
    if lat.ndim == lon.ndim == 1:
        yd, xd = lat.dims[0], lon.dims[0]
        la = np.asarray(lat.values, float)
        lo = ((np.asarray(lon.values, float) + 180) % 360) - 180
        yi = np.flatnonzero((la >= lat_min) & (la <= lat_max))
        xi = np.flatnonzero((lo >= lon_min) & (lo <= lon_max))
        if yi.size < 3 or xi.size < 8:
            raise ValueError("Insufficient lat/lon grid points in bounding box")
        xi = xi[np.argsort(lo[xi])]
        subset = da.isel({yd: yi, xd: xi}).transpose(yd, xd)
        return np.asarray(subset.values), lo[xi], la[yi]
    if lat.ndim == lon.ndim == 2:
        # Native curvilinear GLOENS -> sample on common regular longitude rows.
        # Never use longitude FFT directly on curvilinear x-index lines.
        la = np.asarray(lat.values)
        lo = ((np.asarray(lon.values) + 180) % 360) - 180
        valid = (la >= lat_min) & (la <= lat_max) & (lo >= lon_min) & (lo <= lon_max)
        iy, ix = np.where(valid)
        if iy.size < 8:
            raise ValueError("No GLOENS cells in selected region")
        y0, y1 = iy.min(), iy.max() + 1
        x0, x1 = ix.min(), ix.max() + 1
        yd, xd = lat.dims
        values = np.asarray(da.isel({yd: slice(y0, y1), xd: slice(x0, x1)})
                            .transpose(yd, xd).values, float)
        return values, lo[y0:y1, x0:x1], la[y0:y1, x0:x1]
    raise ValueError("Expected either 1D or 2D lat/lon coordinates")


def common_grid(fields, latitude_count, longitude_count, bounds):
    """Interpolate model-native data to identical horizontal sampling positions."""
    from scipy.interpolate import griddata
    lat_min, lat_max, lon_min, lon_max = bounds
    target_lat = np.linspace(lat_min, lat_max, latitude_count)
    target_lon = np.linspace(lon_min, lon_max, longitude_count)
    lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
    sampled = {}
    for system, (values, lon, lat) in fields.items():
        if np.ndim(lon) == 1:
            lons, lats = np.meshgrid(lon, lat)
        else:
            lons, lats = lon, lat
        finite = np.isfinite(values) & np.isfinite(lons) & np.isfinite(lats)
        if finite.sum() < 8:
            raise ValueError(f"{system}: insufficient valid values for interpolation")
        points = np.column_stack((lons[finite], lats[finite]))
        sampled[system] = griddata(points, values[finite], (lon_grid, lat_grid),
                                   method="linear", fill_value=np.nan)
    return target_lat, target_lon, sampled


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variable", choices=("sst", "ssh"), default="sst")
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    p.add_argument("--gloens-init", default="20241205")
    p.add_argument("--gloens-lead", type=int, default=0)
    p.add_argument("--regular-init", default="20241204")
    p.add_argument("--regular-lead", type=int, default=1)
    p.add_argument("--npy-dir", default="/sortref/calval/CLASS4_scores")
    p.add_argument("--grid-file", default=DEFAULT_GLOENS_GRID)
    p.add_argument("--lon-min", type=float, default=-50)
    p.add_argument("--lon-max", type=float, default=-20)
    p.add_argument("--lat-min", type=float, default=25)
    p.add_argument("--lat-max", type=float, default=35)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--min-valid-fraction", type=float, default=0.85)
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("imgs"))
    args = p.parse_args()
    if not (-180 <= args.lon_min < args.lon_max <= 180 and -80 <= args.lat_min < args.lat_max <= 80):
        p.error("Invalid region bounds")
    if args.n_lon < 8 or args.n_lat < 3 or not 0 < args.min_valid_fraction <= 1:
        p.error("Require n-lon >= 8, n-lat >= 3 and 0 < min-valid-fraction <= 1")
    systems = list(dict.fromkeys(args.systems))
    if args.variable == "ssh" and "GLOENS" in systems:
        print("Skipping GLOENS: its precomputed CLASS4 ensemble mean has no SSH", flush=True)
        systems.remove("GLOENS")
    if not systems:
        p.error("No systems with available requested variable")
    gtime = datetime.strptime(args.gloens_init, "%Y%m%d") + timedelta(days=args.gloens_lead)
    rtime = datetime.strptime(args.regular_init, "%Y%m%d") + timedelta(days=args.regular_lead)
    if "GLOENS" in systems and len(systems) > 1 and gtime != rtime:
        p.error(f"Misaligned valid dates: GLOENS {gtime.date()} vs regular {rtime.date()}")

    fields = {}
    for system in systems:
        if system == "GLOENS":
            ds = load_gloens_mean(args.npy_dir, args.grid_file, args.gloens_init,
                                  args.gloens_lead, include_currents=False)
            try:
                fields[system] = field_and_coords(
                    ds, args.variable, args.lat_min, args.lat_max,
                    args.lon_min, args.lon_max, args.depth_index)
            finally:
                ds.close()
        else:
            store = forecast_store(system, args.regular_init)
            print(f"Opening {system}: s3://{store}", flush=True)
            with open_cloudferro(store, endpoint=args.endpoint,
                                 anonymous=not args.authenticated) as ds:
                if args.regular_lead < 0 or args.regular_lead >= ds.sizes["time"]:
                    raise ValueError(f"{system}: lead {args.regular_lead} unavailable")
                fields[system] = field_and_coords(
                    ds.isel(time=slice(args.regular_lead, args.regular_lead + 1)),
                    args.variable, args.lat_min, args.lat_max,
                    args.lon_min, args.lon_max, args.depth_index)

    bounds = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    target_lat, target_lon, sampled = common_grid(fields, args.n_lat, args.n_lon, bounds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, metrics = {}, []
    lat_middle = float(np.mean(target_lat))
    for system, field in sampled.items():
        spectrum = zonal_psd(field, target_lon, lat_middle,
                             min_valid_fraction=args.min_valid_fraction)
        spectrum.attrs["system"] = system
        spectrum.attrs["variable"] = args.variable
        spectrum.attrs["date"] = str(gtime.date() if system == "GLOENS" else rtime.date())
        spectrum.attrs["region"] = f"{args.lon_min}:{args.lon_max} E, {args.lat_min}:{args.lat_max} N"
        spectrum.to_netcdf(args.output_dir / f"psd_{args.variable}_{system.lower()}_{gtime:%Y%m%d}.nc")
        results[system] = spectrum
        metrics.append({"model": system, "variable": args.variable, **band_metrics(spectrum),
                        "accepted_rows": spectrum.attrs["accepted_rows"],
                        "mean_valid_fraction": spectrum.attrs["mean_accepted_row_valid_fraction"]})
    csv_path = args.output_dir / f"psd_{args.variable}_metrics_{gtime:%Y%m%d}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    for system, spectrum in results.items():
        freq = spectrum.freq_lon.values
        power = spectrum.psd.values
        use = (freq > 0) & (power > 0)
        ax.loglog(1 / freq[use], power[use], label=system)
    for scale in (500, 2000):
        ax.axvline(scale, linestyle="--", color="0.5", linewidth=0.8)
    ax.set_xlabel("Zonal wavelength (km)")
    ax.set_ylabel(f"{args.variable.upper()} zonal PSD (variable² km)")
    ax.set_title(f"OceanBench-inspired spatial PSD | {gtime:%Y-%m-%d} | {args.lat_min:g}–{args.lat_max:g}°N")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    ax.invert_xaxis()
    png_path = args.output_dir / f"psd_{args.variable}_comparison_{gtime:%Y%m%d}.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path} and {csv_path}", flush=True)


if __name__ == "__main__":
    main()
