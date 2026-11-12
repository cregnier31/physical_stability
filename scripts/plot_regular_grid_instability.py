#!/usr/bin/env python3
"""Plot native-grid global N² instability frequency for GLONET, GLOW and GLO12.

Requires access to CloudFerro OceanBench S3 Zarr datasets and local model masks.
No spatial regridding is performed: each map uses its model's native grid.
Run: PYTHONPATH=src python scripts/plot_regular_grid_instability.py --init-date 20241205 --lead 0 --eos eos80
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from physical_stability.diagnostics import compute_n2
from physical_stability.forecasts import (
    ENDPOINT, detect_fields, forecast_store, open_cloudferro, prepare_ts,
)
from physical_stability.masks import MASK_PATHS, MASK_CANDIDATES


SYSTEMS = ("GLONET", "GLOW", "GLO12")


def mask_for_chunk(grid: xr.Dataset, nz: int, ny: int, nx: int, y0: int, y1: int):
    name = next((key for key in MASK_CANDIDATES if key in grid), None)
    if name is None:
        raise KeyError(f"No T mask found; available: {list(grid.data_vars)}")
    mask = grid[name].squeeze(drop=True)
    if mask.ndim not in (2, 3):
        raise ValueError(f"{name}: expected 2D or 3D mask, got {mask.dims}")
    if mask.shape[-2:] != (ny, nx):
        raise ValueError(f"{name}: mask horizontal shape {mask.shape[-2:]} != {(ny, nx)}")
    if mask.ndim == 3 and mask.shape[0] != nz:
        raise ValueError(f"{name}: mask vertical levels {mask.shape[0]} != {nz}")
    part = mask.isel({mask.dims[-2]: slice(y0, y1)}).values
    if not np.all(np.isfinite(part)) or not np.all((part == 0) | (part == 1)):
        raise ValueError(f"{name}: expected finite binary mask")
    if part.ndim == 2:
        part = np.broadcast_to(part, (nz, y1 - y0, nx))
    return part.astype(bool), name


def diagnose_store(store, endpoint, authenticated):
    """List a few S3 entries to distinguish missing paths from Zarr issues."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=not authenticated, client_kwargs={"endpoint_url": endpoint})
    try:
        for target in (store, store.rsplit("/", 1)[0]):
            try:
                entries = fs.ls(target, detail=False)[:12]
                print(f"S3 listing {target}: {entries}", flush=True)
            except Exception as exc:
                print(f"S3 listing {target} failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        fs.close()


def calculate(system, args):
    store = forecast_store(system, args.init_date)
    with ExitStack() as stack:
        print(f"Opening {system}: s3://{store} via {args.endpoint}", flush=True)
        try:
            ds = stack.enter_context(open_cloudferro(store, endpoint=args.endpoint, anonymous=not args.authenticated))
        except Exception as exc:
            print(f"Cannot open {system} forecast: {type(exc).__name__}: {exc}", flush=True)
            diagnose_store(store, args.endpoint, args.authenticated)
            raise RuntimeError(
                f"{system}: failed opening s3://{store}; check listing above for the correct "
                "initialization-date path, permissions, or Zarr layout."
            ) from exc
        fields = detect_fields(ds, system)
        if args.lead >= ds.sizes["time"]:
            raise ValueError(f"{system}: lead {args.lead} exceeds available time dimension")
        selected = ds.isel(time=slice(args.lead, args.lead + 1))
        standard, options = prepare_ts(selected, fields, system)
        ny, nx, nz = (standard.sizes[k] for k in ("y", "x", "deptht"))
        lat = standard.nav_lat.values
        lon = standard.nav_lon.values
        valid_count = np.zeros((ny, nx), dtype=np.int16)
        unstable_count = np.zeros((ny, nx), dtype=np.int16)
        mask_ds = stack.enter_context(xr.open_dataset(args.mask_dir / MASK_PATHS[system].name, decode_times=False))
        mask_name = None
        for y0 in range(0, ny, args.chunk_y):
            y1 = min(y0 + args.chunk_y, ny)
            wet, mask_name = mask_for_chunk(mask_ds, nz, ny, nx, y0, y1)
            strip = standard.isel(y=slice(y0, y1)).copy()
            # Drop the singleton forecast time before the eager GSW calculation.
            strip = strip.isel(time=0, drop=True)
            strip["thetao"] = strip.thetao.where(
                xr.DataArray(wet, dims=("deptht", "y", "x"),
                             coords={k: strip.coords[k] for k in ("deptht", "y", "x")})
            )
            strip["so"] = strip.so.where(
                xr.DataArray(wet, dims=("deptht", "y", "x"),
                             coords={k: strip.coords[k] for k in ("deptht", "y", "x")})
            )
            n2 = compute_n2(strip, eos=args.eos, **options).values
            good = np.isfinite(n2)
            valid_count[y0:y1] = good.sum(axis=0)
            unstable_count[y0:y1] = (good & (n2 < -args.epsilon)).sum(axis=0)
            print(f"{system}: rows {y0}:{y1} / {ny}", flush=True)
        frequency = np.full((ny, nx), np.nan, dtype=np.float32)
        np.divide(100.0 * unstable_count, valid_count, out=frequency, where=valid_count > 0)
        return {
            "system": system, "lat": lat, "lon": lon, "frequency": frequency,
            "valid": valid_count, "unstable": unstable_count, "store": store,
            "mask_name": mask_name, "valid_time": str(selected.time.values[0]),
        }


def render(ax, result, vmax):
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad((1, 1, 1, 0))
    im = ax.pcolormesh(
        result["lon"], result["lat"], np.ma.masked_invalid(result["frequency"]),
        transform=ccrs.PlateCarree(), shading="auto", cmap=cmap,
        vmin=0, vmax=vmax, rasterized=True, zorder=1,
    )
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#dddddd", edgecolor="none", zorder=3)
    ax.coastlines(resolution="110m", linewidth=0.45, color="#555555", zorder=4)
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.35, linestyle="--")
    ax.set_title(result["system"], fontsize=15, pad=10)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-date", default="20241205")
    p.add_argument("--lead", type=int, default=0)
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--eos", choices=("eos80", "teos10"), required=True,
                   help="eos80: practical salinity/potential temperature; teos10: SA/CT")
    p.add_argument("--color-max", type=float, default=100.0)
    p.add_argument("--chunk-y", type=int, default=8)
    p.add_argument("--mask-dir", type=Path, default=Path("/data/rd_exchange/cregnier/static"))
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("imgs"))
    args = p.parse_args()
    if args.lead < 0 or args.chunk_y < 1 or not np.isfinite(args.epsilon) or args.epsilon < 0 or not 0 < args.color_max <= 100:
        p.error("lead >= 0, chunk-y > 0, epsilon finite and >= 0, and 0 < color-max <= 100 are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for system in dict.fromkeys(args.systems):
        result = calculate(system, args)
        results.append(result)
        stem = f"{system.lower()}_instability_{args.init_date}_lead{args.lead:03d}"
        xr.Dataset(
            {
                "unstable_fraction_percent": (("y", "x"), result["frequency"], {"units": "%"}),
                "valid_layer_count": (("y", "x"), result["valid"]),
                "unstable_layer_count": (("y", "x"), result["unstable"]),
            },
            coords={
                "latitude": (("y", "x"), result["lat"]),
                "longitude": (("y", "x"), result["lon"]),
            },
            attrs={
                "system": system, "store": result["store"], "mask_variable": result["mask_name"],
                "valid_time": result["valid_time"], "epsilon_s-2": args.epsilon,
                "eos": args.eos, "note": "Native-grid fraction of valid vertical interfaces with N2 < -epsilon",
            },
        ).to_netcdf(args.output_dir / (stem + ".nc"))

        fig = plt.figure(figsize=(15, 8), facecolor="white")
        ax = fig.add_subplot(111, projection=ccrs.Robinson())
        im = render(ax, result, args.color_max)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.7,
                     extend="max" if args.color_max < 100 else "neither",
                     label=f"Valid vertical layers with N² < -{args.epsilon:g} s⁻² (%)")
        fig.suptitle(f"{args.init_date} · lead {args.lead} · {args.eos}", fontsize=13)
        fig.savefig(args.output_dir / (stem + ".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)
        total = int(result["valid"].sum())
        print(f"{system}: unstable fraction {100 * result['unstable'].sum() / total:.5f}%" if total else f"{system}: no valid layers")

    fig, axs = plt.subplots(
        1, len(results), figsize=(7 * len(results), 5.5),
        subplot_kw={"projection": ccrs.Robinson()}, squeeze=False,
    )
    for ax, result in zip(axs[0], results):
        im = render(ax, result, args.color_max)
    fig.colorbar(
        im, ax=list(axs[0]), orientation="horizontal", pad=0.06, shrink=0.7,
        extend="max" if args.color_max < 100 else "neither",
        label=f"Valid vertical layers with N² < -{args.epsilon:g} s⁻² (%)",
    )
    fig.suptitle(f"Native-grid static-instability frequency | {args.init_date} | lead {args.lead} | {args.eos}", fontsize=15)
    path = args.output_dir / f"regular_models_instability_{args.init_date}_lead{args.lead:03d}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved combined figure: {path}")


if __name__ == "__main__":
    main()
