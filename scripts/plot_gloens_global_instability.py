#!/usr/bin/env python3
"""Global GLOENS N² instability map on a Robinson projection.

Example:
    PYTHONPATH=src python scripts/plot_gloens_global_instability.py --init-date 20241205 --lead 0

Requires numpy, xarray, gsw, matplotlib, cartopy and a NetCDF writer.
Reads one narrow latitude-index strip at a time to limit peak memory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from physical_stability.diagnostics import compute_n2
from physical_stability.gloens_npy import DEFAULT_GLOENS_GRID, load_gloens_mean
from physical_stability.masks import MASK_PATHS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-date", default="20241205", help="YYYYMMDD")
    p.add_argument("--lead", type=int, default=0)
    p.add_argument("--npy-dir", default="/sortref/calval/CLASS4_scores")
    p.add_argument("--grid-file", default=DEFAULT_GLOENS_GRID)
    p.add_argument("--mask-file", default=str(MASK_PATHS["GLOENS"]))
    p.add_argument("--epsilon", type=float, default=1e-8, help="Instability when N² < -epsilon, s^-2")
    p.add_argument("--chunk-y", type=int, default=16)
    p.add_argument("--color-max", type=float, default=10.0, help="Upper plotted percentage")
    p.add_argument("--output-prefix", default=None)
    args = p.parse_args()
    if args.epsilon < 0 or not np.isfinite(args.epsilon) or args.chunk_y < 1 or args.color_max <= 0:
        p.error("epsilon must be finite and nonnegative; chunk-y and color-max must be positive")

    ds = load_gloens_mean(
        args.npy_dir, args.grid_file, args.init_date, args.lead, include_currents=False
    )
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    lat, lon = ds.latitude.values, ds.longitude.values
    depths = ds.depth.values
    valid_count = np.zeros((ny, nx), dtype=np.int16)
    unstable_count = np.zeros((ny, nx), dtype=np.int16)
    minimum_n2 = np.full((ny, nx), np.nan, dtype=np.float32)

    with xr.open_dataset(args.mask_file, decode_times=False) as mesh:
        if "tmask" not in mesh:
            raise KeyError(f"Missing 3D tmask in {args.mask_file}")
        mask = mesh["tmask"]
        if mask.shape[-3:] != (len(depths), ny, nx):
            raise ValueError(f"Mask shape {mask.shape} does not match GLOENS {(len(depths), ny, nx)}")
        for y0 in range(0, ny, args.chunk_y):
            y1 = min(y0 + args.chunk_y, ny)
            sel = slice(y0, y1)
            wet = mask.isel(y=sel).squeeze(drop=True).values.astype(bool)
            t = np.where(wet, ds.thetao.isel(time=0, y=sel).values, np.nan)
            s = np.where(wet, ds.so.isel(time=0, y=sel).values, np.nan)
            chunk = xr.Dataset(
                {
                    "thetao": (("deptht", "y", "x"), t),
                    "so": (("deptht", "y", "x"), s),
                    "nav_lat": (("y", "x"), lat[sel]),
                    "nav_lon": (("y", "x"), lon[sel]),
                },
                coords={"deptht": depths, "y": ds.y.values[sel], "x": ds.x.values},
            )
            n2 = compute_n2(chunk).values
            finite = np.isfinite(n2)
            valid_count[sel] = finite.sum(axis=0)
            unstable_count[sel] = (finite & (n2 < -args.epsilon)).sum(axis=0)
            min_values = np.where(finite, n2, np.inf).min(axis=0)
            min_values[~np.isfinite(min_values)] = np.nan
            minimum_n2[sel] = min_values.astype(np.float32)
            print(f"Processed y={y0}:{y1} of {ny}", flush=True)

    fraction = np.full((ny, nx), np.nan, dtype=np.float32)
    np.divide(100.0 * unstable_count, valid_count, out=fraction, where=valid_count > 0)

    prefix = args.output_prefix or f"gloens_instability_{args.init_date}_lead{args.lead:03d}"
    nc_path = Path(prefix + ".nc")
    png_path = Path(prefix + ".png")
    output = xr.Dataset(
        {
            "unstable_fraction_percent": (("y", "x"), fraction, {"units": "%"}),
            "minimum_n2": (("y", "x"), minimum_n2, {"units": "s-2"}),
            "valid_layer_count": (("y", "x"), valid_count),
            "unstable_layer_count": (("y", "x"), unstable_count),
        },
        coords={
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
        attrs={
            "model": "GLOENS",
            "initialization_date": args.init_date,
            "lead_days": args.lead,
            "epsilon_s-2": args.epsilon,
            "note": "Unstable layers are defined by N2 < -epsilon, among finite N2 layers.",
        },
    )
    output.to_netcdf(nc_path)

    fig = plt.figure(figsize=(16, 8.5), facecolor="white")
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    ax.set_global()
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad((1, 1, 1, 0))
    im = ax.pcolormesh(
        lon, lat, np.ma.masked_invalid(fraction),
        transform=ccrs.PlateCarree(), shading="auto",
        cmap=cmap, vmin=0, vmax=args.color_max, rasterized=True, zorder=1,
    )
    ax.add_feature(cfeature.LAND, facecolor="#dedede", edgecolor="none", zorder=3)
    ax.coastlines(resolution="110m", linewidth=0.5, color="#555555", zorder=4)
    ax.gridlines(linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
    bar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.045, shrink=0.72, extend="max")
    bar.set_label(f"Valid vertical layers with N² < -{args.epsilon:g} s⁻² (%)")
    ax.set_title(f"GLOENS | Global static instability | {args.init_date} | lead {args.lead}", fontsize=16, pad=18)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    total = int(valid_count.sum())
    print(f"Saved {png_path} and {nc_path}")
    print(f"Finite N² layer-cells: {total:,}")
    print(f"Unstable fraction of finite layer-cells: {100 * unstable_count.sum() / total:.6f}%" if total else "No valid N² cells")


if __name__ == "__main__":
    main()
