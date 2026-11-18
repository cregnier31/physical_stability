#!/usr/bin/env python3
"""Annual, paired, lead-resolved zonal SSH/SST spectra for three forecast systems.

Writes one NetCDF with (model, init, lead, freq_lon), plus summary CSV.
Missing stores/leads are logged; incomplete model triples are excluded.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from physical_stability.forecasts import ENDPOINT, forecast_store, open_cloudferro
from physical_stability.psd import zonal_psd, band_metrics
from plot_oceanbench_psd import field_and_coords, common_grid


def run(args):
    starts = np.arange(np.datetime64(args.start), np.datetime64(args.end) + np.timedelta64(1, "D"),
                       np.timedelta64(1, "D"))
    models = tuple(args.systems)
    bounds = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    target_lon = np.linspace(args.lon_min, args.lon_max, args.n_lon)
    lat_middle = 0.5 * (args.lat_min + args.lat_max)
    # Frequency grid is fixed by the target longitude spacing.
    dx = 111.195 * np.cos(np.deg2rad(lat_middle)) * np.diff(target_lon).mean()
    frequency = np.fft.rfftfreq(args.n_lon, dx)
    leads = list(range(args.lead_start, args.lead_stop + 1))
    cube = np.full((len(models), len(starts), len(leads), len(frequency)), np.nan)
    rows = []
    failures = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, start in enumerate(starts):
        init = str(start).replace("-", "")
        # Open each model store once for all lead indices.
        stores = {}
        try:
            for model in models:
                try:
                    context = open_cloudferro(forecast_store(model, init), endpoint=args.endpoint,
                                              anonymous=not args.authenticated)
                    ds = context.__enter__()
                    stores[model] = (context, ds)
                except Exception as exc:
                    failures.append((init, model, "open", str(exc)))
            for j, lead in enumerate(leads):
                fields = {}
                for model, (_, ds) in stores.items():
                    try:
                        if lead >= ds.sizes["time"]:
                            raise IndexError("lead index outside time dimension")
                        fields[model] = field_and_coords(
                            ds.isel(time=slice(lead, lead + 1)), args.variable,
                            args.lat_min, args.lat_max, args.lon_min, args.lon_max, 0)
                    except Exception as exc:
                        failures.append((init, model, str(lead), str(exc)))
                if set(fields) != set(models):
                    continue
                try:
                    _, _, sampled = common_grid(fields, args.n_lat, args.n_lon, bounds)
                    spectra = {}
                    for model in models:
                        spectra[model] = zonal_psd(sampled[model], target_lon, lat_middle,
                                                   min_valid_fraction=args.min_valid_fraction)
                except Exception as exc:
                    failures.append((init, "paired", str(lead), str(exc)))
                    continue
                for m, model in enumerate(models):
                    psd = spectra[model]
                    cube[m, i, j] = psd.psd.values
                    metrics = band_metrics(psd)
                    rows.append({"model": model, "init": init, "lead_index": lead,
                                 "valid_date": str(start + np.timedelta64(lead, "D")),
                                 "accepted_rows": psd.attrs["accepted_rows"], **metrics})
        finally:
            for context, _ in stores.values():
                context.__exit__(None, None, None)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(starts)} initializations", flush=True)

    output = xr.Dataset(
        {"psd": (("model", "init", "lead", "freq_lon"), cube)},
        coords={"model": list(models), "init": starts, "lead": leads,
                "freq_lon": frequency},
        attrs={"variable": args.variable, "region": str(bounds),
               "lead_semantics": "zero-based dataset time index, not guaranteed elapsed days",
               "method": "paired regional, remapped zonal PSD; incomplete triples excluded",
               "sampling": "common regular grid, linear interpolation"})
    output.psd.attrs["units"] = "(input units)^2 km"
    output.freq_lon.attrs["units"] = "cycles/km"
    output.to_netcdf(args.output_dir / f"annual_{args.variable}_psd.nc")
    for filename, fields, data in [
        ("metrics.csv", list(rows[0]) if rows else ["model", "init", "lead_index"], rows),
        ("failures.csv", ["init", "model", "lead", "error"],
         [dict(zip(["init", "model", "lead", "error"], row)) for row in failures]),
    ]:
        with (args.output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    print(f"Saved {args.output_dir}/annual_{args.variable}_psd.nc; "
          f"{len(rows)//len(models)} paired cases; {len(failures)} logged failures")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variable", choices=["ssh", "sst"], default="ssh")
    p.add_argument("--systems", nargs="+", choices=["GLONET", "GLOW", "GLO12"],
                   default=["GLONET", "GLOW", "GLO12"])
    p.add_argument("--start", required=True, help="First initialization YYYY-MM-DD")
    p.add_argument("--end", required=True, help="Last initialization YYYY-MM-DD inclusive")
    p.add_argument("--lead-start", type=int, default=0)
    p.add_argument("--lead-stop", type=int, default=6)
    p.add_argument("--lon-min", type=float, default=-50)
    p.add_argument("--lon-max", type=float, default=-20)
    p.add_argument("--lat-min", type=float, default=25)
    p.add_argument("--lat-max", type=float, default=35)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--min-valid-fraction", type=float, default=0.85)
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("results/annual_psd"))
    a = p.parse_args()
    if (a.start > a.end or a.lead_start < 0 or a.lead_stop < a.lead_start or
            len(set(a.systems)) != len(a.systems) or a.n_lon < 8 or a.n_lat < 3 or
            not -180 <= a.lon_min < a.lon_max <= 180 or
            not -80 <= a.lat_min < a.lat_max <= 80):
        p.error("Invalid date, lead, model, grid or geographical bounds")
    run(a)


if __name__ == "__main__":
    main()
