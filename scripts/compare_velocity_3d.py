#!/usr/bin/env python3
"""Compare 3D velocity diagnostics at one initialization and lead index.

Regular 1D lat/lon coordinates and collocated uo/vo only. Refuses unsupported
native staggered grids; no implicit C-grid interpolation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import xarray as xr

from physical_stability.forecasts import ENDPOINT, forecast_store, open_cloudferro
from physical_stability.velocity_dynamics import velocity_diagnostics, depth_kinetic_energy_psd
from plot_oceanbench_psd import common_grid
from physical_stability.gloens_npy import load_gloens_mean, DEFAULT_GLOENS_GRID


def select_field(ds, candidates, depth_name, lat_name, lon_name, lead):
    name = next((n for n in candidates if n in ds), None)
    if name is None:
        raise KeyError(f"No variable among {candidates}")
    a = ds[name].isel(time=lead) if "time" in ds[name].dims else ds[name]
    required = (depth_name, lat_name, lon_name)
    if set(a.dims) != set(required):
        raise ValueError(f"{name}: expected collocated dims {required}, got {a.dims}")
    return np.asarray(a.transpose(*required).values, float)


def process(ds, args):
    lat_name = next((n for n in ("latitude", "lat") if n in ds.coords), None)
    lon_name = next((n for n in ("longitude", "lon") if n in ds.coords), None)
    depth_name = next((n for n in ("depth", "deptht", "deptho", "z", "lev")
                       if n in ds.coords), None)
    if not all((lat_name, lon_name, depth_name)):
        raise ValueError("Expected 1D lat, lon, depth coordinates")
    lat, lon, depth = (np.asarray(ds[n].values, float)
                       for n in (lat_name, lon_name, depth_name))
    if ds[depth_name].ndim != 1:
        raise ValueError("Depth must be 1D")
    curvilinear = ds[lat_name].ndim == ds[lon_name].ndim == 2
    if not curvilinear and any(ds[n].ndim != 1 for n in (lat_name, lon_name)):
        raise ValueError("Horizontal coordinates must both be 1D or both 2D")
    if not np.all(np.diff(depth) > 0):
        raise ValueError("Depth must be increasing positive-down")
    lon180 = (lon + 180) % 360 - 180
    if curvilinear:
        mask = ((lat >= args.lat_min) & (lat <= args.lat_max) &
                (lon180 >= args.lon_min) & (lon180 <= args.lon_max))
        yi, xi = np.where(mask)
        if yi.size < 8:
            raise ValueError("Insufficient regional grid coverage")
        y = slice(int(yi.min()), int(yi.max()) + 1)
        x = slice(int(xi.min()), int(xi.max()) + 1)
    else:
        y = np.flatnonzero((lat >= args.lat_min) & (lat <= args.lat_max))
        x = np.flatnonzero((lon180 >= args.lon_min) & (lon180 <= args.lon_max))
        if len(y) < 3 or len(x) < 8:
            raise ValueError("Insufficient regional grid coverage")
        x = x[np.argsort(lon180[x])]
    if args.max_depth is not None:
        depth_idx = np.flatnonzero(depth <= args.max_depth)
    else:
        depth_idx = np.arange(len(depth))
    if len(depth_idx) < 3:
        raise ValueError("At least three depth levels are required")
    if curvilinear:
        yd, xd = ds[lat_name].dims
        def read_current(candidates):
            name = next((n for n in candidates if n in ds), None)
            if name is None:
                raise KeyError(f"Missing velocity component among {candidates}")
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=args.lead)
            if set(da.dims) != {depth_name, yd, xd}:
                raise ValueError(f"{name}: velocity grid does not match T-grid; "
                                 "explicit U/V collocation required")
            return np.asarray(da.isel({depth_name: depth_idx, yd: y, xd: x})
                              .transpose(depth_name, yd, xd).values, float)
        u = read_current(("uo", "vozocrtx", "u"))
        v = read_current(("vo", "vomecrty", "v"))
        source_lon, source_lat = lon180[y, x], lat[y, x]
        inside = ((source_lon >= args.lon_min) & (source_lon <= args.lon_max) &
                  (source_lat >= args.lat_min) & (source_lat <= args.lat_max))
        u = np.where(inside[None], u, np.nan)
        v = np.where(inside[None], v, np.nan)
    else:
        def read_regular(candidates):
            name = next((n for n in candidates if n in ds), None)
            if name is None:
                raise KeyError(f"Missing velocity component among {candidates}")
            da = ds[name]
            if "time" in da.dims:
                da = da.isel(time=args.lead)
            if set(da.dims) != {depth_name, lat_name, lon_name}:
                raise ValueError(f"{name}: staggered U/V grids require explicit collocation")
            return np.asarray(da.isel({depth_name: depth_idx, lat_name: y, lon_name: x})
                              .transpose(depth_name, lat_name, lon_name).values, float)
        u = read_regular(("uo", "vozocrtx", "u"))
        v = read_regular(("vo", "vomecrty", "v"))
        source_lon, source_lat = lon180[x], lat[y]
    bounds = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    ul, vl = [], []
    for uk, vk in zip(u, v):
        _, _, sampled = common_grid({"u": (uk, source_lon, source_lat),
                                     "v": (vk, source_lon, source_lat)},
                                    args.n_lat, args.n_lon, bounds)
        ul.append(sampled["u"])
        vl.append(sampled["v"])
    ul, vl = np.asarray(ul), np.asarray(vl)
    target_lon = np.linspace(args.lon_min, args.lon_max, args.n_lon)
    target_lat = np.linspace(args.lat_min, args.lat_max, args.n_lat)
    z = depth[depth_idx]
    # Do not calculate SSH ageostrophic residual here: it requires a separately
    # validated collocation between the SSH grid and velocity grid.
    dyn = velocity_diagnostics(ul, vl, z, target_lon, target_lat)
    psd = depth_kinetic_energy_psd(ul, vl, z, target_lon, target_lat,
                                   min_valid_fraction=args.min_valid_fraction)
    dyn["ke_psd"] = psd.ke_psd
    dyn["mean_ke_profile"] = dyn.kinetic_energy.mean(("latitude", "longitude"), skipna=True)
    return dyn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", nargs="+", default=["GLONET", "GLOW", "GLO12"],
                   choices=["GLONET", "GLOW", "GLO12", "GLOENS"])
    p.add_argument("--gloens-npy-dir", default="/sortref/calval/CLASS4_scores")
    p.add_argument("--gloens-grid-file", default=DEFAULT_GLOENS_GRID)
    p.add_argument("--gloens-init", help="GLOENS initialization YYYYMMDD; defaults to --init")
    p.add_argument("--gloens-lead", type=int, help="GLOENS lead; defaults to --lead")
    p.add_argument("--init", required=True, help="YYYYMMDD")
    p.add_argument("--lead", type=int, default=1)
    p.add_argument("--max-depth", type=float, default=1000)
    p.add_argument("--lon-min", type=float, default=-50)
    p.add_argument("--lon-max", type=float, default=-20)
    p.add_argument("--lat-min", type=float, default=25)
    p.add_argument("--lat-max", type=float, default=35)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--min-valid-fraction", type=float, default=0.85)
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("results/velocity"))
    p.add_argument("--no-plots", action="store_true", help="Skip automatic PNG generation")
    args = p.parse_args()
    if args.lead < 0:
        p.error("Lead index must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    from contextlib import nullcontext
    from datetime import datetime, timedelta
    for model in args.systems:
        if model == "GLOENS":
            gi = args.gloens_init or args.init
            gl = args.gloens_lead if args.gloens_lead is not None else args.lead
            if datetime.strptime(gi, "%Y%m%d") + timedelta(days=gl) != datetime.strptime(args.init, "%Y%m%d") + timedelta(days=args.lead):
                raise ValueError("GLOENS and other models must have the same valid date")
            dataset = load_gloens_mean(args.gloens_npy_dir, args.gloens_grid_file, gi, gl, include_currents=True)
            if not {"uo", "vo"} <= set(dataset.data_vars):
                raise ValueError("GLOENS precomputed mean lacks uo/vo currents")
            context = nullcontext(dataset)
        else:
            context = open_cloudferro(forecast_store(model, args.init), endpoint=args.endpoint,
                                      anonymous=not args.authenticated)
        with context as ds:
            if model == "GLOENS":
                original_lead = args.lead
                args.lead = 0
                try:
                    result = process(ds, args)
                finally:
                    args.lead = original_lead
            else:
                result = process(ds, args)
            result.attrs.update(model=model, init=args.init, lead_index=args.lead)
            path = args.output_dir / f"{model.lower()}_{args.init}_lead{args.lead}.nc"
            result.to_netcdf(path)
            output_files.append(path)
            print(f"Saved {path}", flush=True)
    if not args.no_plots:
        from plot_velocity_diagnostics import render
        render(output_files, args.output_dir / "plots", (0, 5, 10))
        print(f"Saved PNG plots under {args.output_dir / 'plots'}", flush=True)


if __name__ == "__main__":
    main()
