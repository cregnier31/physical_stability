#!/usr/bin/env python3
"""Exploratory North Atlantic density-displacement sections across four forecasts.

Daily output cannot establish internal-wave propagation. Density anomalies are
spatially high-pass filtered proxies and can include eddies/fronts.
Default valid time 2024-12-05: GLOENS init 20241205 lead 0; regular
models init 20241204 lead 1.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import xarray as xr
import gsw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physical_stability.forecasts import ENDPOINT, forecast_store, open_cloudferro, detect_fields
from physical_stability.gloens_npy import load_gloens_mean, DEFAULT_GLOENS_GRID
from physical_stability.masks import MASK_PATHS, MASK_CANDIDATES

SYSTEMS = ("GLOENS", "GLONET", "GLOW", "GLO12")


def positions(lat, lon, latitude, west, east, count):
    """Nearest wet-grid candidates along a constant-latitude transect."""
    targets = np.linspace(west, east, count)
    la = np.asarray(lat)
    lo = ((np.asarray(lon) + 180) % 360) - 180
    if la.ndim == lo.ndim == 1:
        yy = np.full(count, int(np.nanargmin(abs(la - latitude))), dtype=int)
        xx = np.array([np.nanargmin(abs(lo - value)) for value in targets], dtype=int)
    elif la.ndim == lo.ndim == 2:
        yy, xx = [], []
        for value in targets:
            distance = (la - latitude) ** 2 + ((lo - value) * np.cos(np.deg2rad(latitude))) ** 2
            idx = np.nanargmin(distance)
            y, x = np.unravel_index(idx, la.shape)
            yy.append(y)
            xx.append(x)
        yy, xx = np.asarray(yy), np.asarray(xx)
    else:
        raise ValueError("Expected both 1D or both 2D geographic coordinates")
    return yy, xx


def mask_at_points(path, yy, xx, nz, ny, nx):
    with xr.open_dataset(path, decode_times=False) as ds:
        name = next((v for v in MASK_CANDIDATES if v in ds), None)
        if name is None:
            raise KeyError(f"No wet mask in {path}")
        m = ds[name].squeeze(drop=True)
        if m.shape[-2:] != (ny, nx):
            raise ValueError(f"{path}: mask shape {m.shape[-2:]} differs from grid {(ny,nx)}")
        if m.ndim == 3:
            if m.shape[0] != nz:
                raise ValueError(f"{path}: vertical mask mismatch")
            wet = np.asarray(m.values[:, yy, xx])
        elif m.ndim == 2:
            wet = np.broadcast_to(np.asarray(m.values[yy, xx]), (nz, len(xx)))
        else:
            raise ValueError("Expected 2D/3D wet mask")
        if not np.all(np.isfinite(wet)) or not np.all((wet == 0) | (wet == 1)):
            raise ValueError("Mask must be finite and binary")
        return wet.astype(bool)


def get_section(system, args):
    if system == "GLOENS":
        ds = load_gloens_mean(args.npy_dir, args.grid_file, args.gloens_init, args.gloens_lead, include_currents=False).isel(time=0, drop=True)
        try:
            return sample(ds, "thetao", "so", "depth", "latitude", "longitude", system, args)
        finally:
            ds.close()
    store = forecast_store(system, args.regular_init)
    print(f"Opening {system}: s3://{store}", flush=True)
    with open_cloudferro(store, endpoint=args.endpoint, anonymous=not args.authenticated) as ds:
        f = detect_fields(ds, system)
        if args.regular_lead >= ds.sizes["time"]:
            raise ValueError(f"{system}: lead {args.regular_lead} unavailable")
        one = ds.isel(time=args.regular_lead)
        return sample(one, f.temperature, f.salinity, f.depth, f.latitude, f.longitude, system, args)


def sample(ds, tname, sname, zname, latname, lonname, system, args):
    lat, lon = ds[latname].values, ds[lonname].values
    yy, xx = positions(lat, lon, args.latitude, args.lon_min, args.lon_max, args.points)
    t = ds[tname]
    s = ds[sname]
    zd = ds[zname].dims[0]
    if lat.ndim == 1:
        yd, xd = ds[latname].dims[0], ds[lonname].dims[0]
        latitude = lat[yy]
        longitude = lon[xx]
    else:
        yd, xd = ds[latname].dims
        latitude = lat[yy, xx]
        longitude = lon[yy, xx]
    index = {"station": np.arange(len(xx))}
    select = {yd: xr.DataArray(yy, dims="station", coords=index),
              xd: xr.DataArray(xx, dims="station", coords=index)}
    temp = np.asarray(t.isel(select).transpose(zd, "station").values, dtype=float)
    salt = np.asarray(s.isel(select).transpose(zd, "station").values, dtype=float)
    depth = np.asarray(ds[zname].values, dtype=float)
    wet = mask_at_points(args.mask_dir / MASK_PATHS[system].name, yy, xx, len(depth),
                         t.sizes[yd], t.sizes[xd])
    temp[~wet] = np.nan
    salt[~wet] = np.nan
    if args.eos == "eos80":
        p = gsw.p_from_z(-depth[:, None], latitude[None, :])
        SA = gsw.SA_from_SP(salt, p, longitude[None, :], latitude[None, :])
        CT = gsw.CT_from_pt(SA, temp)
    else:
        SA, CT = salt, temp
    # Density referenced to the same pressure to isolate lateral variations.
    rho = gsw.sigma0(SA, CT)
    # Same physical window for every model, on a shared longitude sampling grid.
    # Gaussian weights use great-circle distance along the transect (km).
    # Require sufficient wet coverage; do not fill dry or missing stations.
    station_km = 111.195 * np.cos(np.deg2rad(args.latitude)) * (np.asarray(longitude) - longitude[0])
    spacing_km = np.median(np.abs(np.diff(station_km)))
    if not np.isfinite(spacing_km) or spacing_km <= 0:
        raise ValueError("Invalid or duplicate longitude sampling positions")
    radius_km = args.background_km / 2
    background = np.full_like(rho, np.nan)
    for j in range(len(longitude)):
        distances = np.abs(station_km - station_km[j])
        weights = np.where(distances <= radius_km, 1.0, 0.0)
        if weights.sum() < 3:
            continue
        finite = np.isfinite(rho)
        weighted = finite * weights[None, :]
        coverage = weighted.sum(axis=1) / weights.sum()
        numerator = np.sum(np.where(finite, rho, 0.0) * weights[None, :], axis=1)
        denominator = weighted.sum(axis=1)
        background[:, j] = np.divide(numerator, denominator,
                                      out=np.full(len(depth), np.nan),
                                      where=(coverage >= args.min_coverage) & (denominator > 0))
    anomaly = rho - background
    # Positive-down depth: stable density increases with depth.
    # Compute derivatives only where adjacent background levels are finite.
    drhodz = np.gradient(background, depth, axis=0)
    # Avoid singular displacements in weak stratification.
    eta = np.divide(-anomaly, drhodz, out=np.full_like(anomaly, np.nan),
                    where=drhodz > args.min_gradient)
    eta[~np.isfinite(rho)] = np.nan
    n2 = 9.81 / 1027.0 * drhodz
    ep = 0.5 * n2 * eta ** 2
    ep[(n2 <= 0) | ~np.isfinite(eta)] = np.nan
    return dict(system=system, depth=depth, lon=((longitude + 180) % 360) - 180,
                lat=latitude, eta=eta, ep=ep, n2=n2, anomaly=anomaly,
                wet=wet, background=background)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gloens-init", default="20241205")
    p.add_argument("--gloens-lead", type=int, default=0)
    p.add_argument("--regular-init", default="20241204")
    p.add_argument("--regular-lead", type=int, default=1)
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    p.add_argument("--npy-dir", default="/sortref/calval/CLASS4_scores")
    p.add_argument("--grid-file", default=DEFAULT_GLOENS_GRID)
    p.add_argument("--mask-dir", type=Path, default=Path("/data/rd_exchange/cregnier/static"))
    p.add_argument("--latitude", type=float, default=30)
    p.add_argument("--lon-min", type=float, default=-50)
    p.add_argument("--lon-max", type=float, default=-20)
    p.add_argument("--points", type=int, default=161)
    p.add_argument("--background-km", type=float, default=200,
                   help="Full width of the horizontal running-mean background window (km)")
    p.add_argument("--min-coverage", type=float, default=0.7,
                   help="Minimum fraction of valid stations in a background window")
    p.add_argument("--min-gradient", type=float, default=0.001, help="Minimum background sigma0 gradient (kg m^-4)")
    p.add_argument("--max-depth", type=float, default=800)
    p.add_argument("--eta-limit", type=float, default=100)
    p.add_argument("--eos", choices=("eos80", "teos10"), default="eos80")
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("imgs"))
    args = p.parse_args()
    if args.points < 5 or args.background_km <= 0 or not 0 < args.min_coverage <= 1:
        p.error("--points >= 5, background-km > 0 and 0 < min-coverage <= 1 required")
    if args.min_gradient <= 0 or args.eta_limit <= 0:
        p.error("Gradient and displacement limits must be positive")
    gtime = datetime.strptime(args.gloens_init, "%Y%m%d") + timedelta(days=args.gloens_lead)
    rtime = datetime.strptime(args.regular_init, "%Y%m%d") + timedelta(days=args.regular_lead)
    if "GLOENS" in args.systems and len(args.systems) > 1 and gtime != rtime:
        p.error(f"Valid times differ: GLOENS {gtime.date()} vs regular {rtime.date()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [get_section(s, args) for s in dict.fromkeys(args.systems)]
    fig, axs = plt.subplots(len(results), 1, figsize=(13, 3.4 * len(results)),
                            sharex=True, squeeze=False, constrained_layout=True)
    summary = []
    for ax, result in zip(axs[:, 0], results):
        common = (result["depth"] >= 0) & (result["depth"] <= args.max_depth)
        eta_values = result["eta"][common]
        ep_values = result["ep"][common]
        finite_eta = np.isfinite(eta_values)
        finite_ep = np.isfinite(ep_values)
        summary.append({
            "system": result["system"], "valid_time": (gtime if result["system"] == "GLOENS" else rtime).isoformat(),
            "max_depth_m": args.max_depth, "background_km": args.background_km,
            "valid_displacement_cells": int(finite_eta.sum()),
            "displacement_rms_m": float(np.sqrt(np.mean(eta_values[finite_eta] ** 2))) if finite_eta.any() else float("nan"),
            "displacement_std_m": float(np.std(eta_values[finite_eta])) if finite_eta.any() else float("nan"),
            "mean_ep_proxy_J_kg": float(np.mean(ep_values[finite_ep])) if finite_ep.any() else float("nan"),
            "valid_ep_cells": int(finite_ep.sum()),
            "wet_cells": int(result["wet"][common].sum()),
        })
        field = np.ma.masked_invalid(result["eta"])
        im = ax.pcolormesh(result["lon"], result["depth"], field, shading="auto",
                           cmap="RdBu_r", vmin=-args.eta_limit, vmax=args.eta_limit)
        ax.set_ylim(args.max_depth, 0)
        ax.set_facecolor("#e0e0e0")
        ax.set_ylabel(f'{result["system"]}\nDepth (m)')
        fig.colorbar(im, ax=ax, label="Density-displacement proxy (m)", extend="both")
        ds = xr.Dataset(
            {"displacement_proxy_m": (("depth", "station"), result["eta"]),
             "potential_energy_proxy_J_kg": (("depth", "station"), result["ep"]),
             "background_n2_approx_s2": (("depth", "station"), result["n2"]),
             "sigma0_anomaly_kg_m3": (("depth", "station"), result["anomaly"]),
             "background_sigma0_kg_m3": (("depth", "station"), result["background"]),
             "wet_mask": (("depth", "station"), result["wet"].astype(np.int8))},
            coords={"depth": result["depth"], "station": np.arange(len(result["lon"])),
                    "longitude": ("station", result["lon"]),
                    "latitude": ("station", result["lat"])},
            attrs={"model": result["system"], "valid_time": gtime.isoformat() if result["system"] == "GLOENS" else rtime.isoformat(),
                   "eos": args.eos, "background_km": args.background_km, "min_coverage": args.min_coverage,
                   "warning": "Spatial density-displacement proxy; not a verified internal-wave signal. Sigma0 gradient is approximate."})
        ds.to_netcdf(args.output_dir / f'internal_wave_proxy_{result["system"].lower()}_{gtime:%Y%m%d}.nc')
    csv_path = args.output_dir / f"internal_wave_proxy_summary_{gtime:%Y%m%d}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved {csv_path}", flush=True)
    axs[-1, 0].set_xlabel("Longitude (degrees east)")
    fig.suptitle(f"North Atlantic density-displacement proxies | {gtime:%Y-%m-%d} | {args.latitude:g}°N")
    output = args.output_dir / f"internal_wave_proxy_sections_{gtime:%Y%m%d}.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
