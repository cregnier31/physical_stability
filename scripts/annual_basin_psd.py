#!/usr/bin/env python3
"""2024 basin-sector x latitude-band zonal SST/SSH spectra.

Basin sectors are explicitly approximate geographic windows, NOT hydrographic
basin polygons. Longitude segments are never concatenated across oceans.
Spectra are interpolated onto a common physical wavenumber axis before
aggregation. Missing four-system SST cases are excluded together.
"""
from __future__ import annotations
import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import xarray as xr
from physical_stability.forecasts import ENDPOINT, forecast_store, open_cloudferro
from physical_stability.gloens_npy import DEFAULT_GLOENS_GRID, load_gloens_mean
from physical_stability.psd import zonal_psd
from plot_oceanbench_psd import field_and_coords, common_grid

# Non-dateline-crossing sectors; deliberately do not claim exact basin masks.
SECTORS = {
    "Atlantic": [(-70., 20.)],
    "Pacific": [(-180., -70.), (120., 180.)],
    "Indian": [(20., 120.)],
    "Southern": [(-180., -60.), (-60., 60.), (60., 180.)],
}
BANDS = {
    "Tropical": (-20., 20.),
    "Midlatitude_N": (20., 60.),
    "Midlatitude_S": (-60., -20.),
    "Highlatitude_N": (60., 80.),
    "Highlatitude_S": (-90., -60.),
}
# Exclude Southern Ocean from other basins below 60S.
# Other basin sectors exclude latitudes south of 60S.
def regions():
    for basin, windows in SECTORS.items():
        for band, (lo, hi) in BANDS.items():
            if basin == "Southern" and band != "Highlatitude_S":
                continue
            if basin != "Southern" and band == "Highlatitude_S":
                continue
            for west, east in windows:
                yield basin, band, (lo, hi, west, east)


def one_spectrum(ds, variable, lead, region, nlat, nlon, min_fraction, depth=0):
    south, north, west, east = region
    # Avoid exact poles where physical zonal spacing vanishes.
    south, north = max(south, -85.), min(north, 80.)
    if north - south < 1:
        raise ValueError("Empty latitude range")
    if "time" in ds.dims:
        if lead >= ds.sizes["time"]:
            raise IndexError("Lead outside dataset")
        ds = ds.isel(time=slice(lead, lead+1))
    field = field_and_coords(ds, variable, south, north, west, east, depth)
    _, lon, data = common_grid({"field": field}, nlat, nlon,
                               (south, north, west, east))
    # Per-row FFT with latitude-specific km spacing; interpolate power to
    # a common physical wavenumber grid, preventing artificial pole shifts.
    lats = np.linspace(south, north, nlat)
    f_common = np.geomspace(1/10000., 1/100., 120)
    powers, weights = [], []
    for j, latitude in enumerate(lats):
        if abs(latitude) >= 85:
            continue
        try:
            p = zonal_psd(data["field"][j:j+1], lon, latitude,
                          min_valid_fraction=min_fraction)
        except ValueError:
            continue
        freq = np.asarray(p.freq_lon.values)
        power = np.asarray(p.psd.values)
        ok = (freq > 0) & np.isfinite(power)
        if ok.sum() < 3:
            continue
        # No extrapolation beyond the row's physically resolved frequencies.
        interp = np.interp(f_common, freq[ok], power[ok],
                           left=np.nan, right=np.nan)
        powers.append(interp)
        weights.append(np.cos(np.deg2rad(latitude)))
    if not powers:
        raise ValueError("No accepted latitude rows")
    arr = np.asarray(powers)
    w = np.asarray(weights)[:, None]
    valid = np.isfinite(arr)
    num = np.sum(np.where(valid, arr, 0.)*w, axis=0)
    den = np.sum(valid*w, axis=0)
    avg = np.divide(num, den, out=np.full(len(f_common), np.nan), where=den>0)
    return f_common, avg, len(powers)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = args.systems
    if args.variable == "ssh" and "GLOENS" in models:
        raise ValueError("GLOENS has no SSH; use SST for four-system comparisons")
    starts = []
    day = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    while day <= end:
        starts.append(day)
        day += timedelta(days=args.init_step_days)
    freqs = np.geomspace(1/10000., 1/100., 120)
    rows, failures = [], []
    # Store only complete model sets for each init/lead/region.
    for index, start in enumerate(starts, 1):
        init = start.strftime("%Y%m%d")
        with ExitStack() as stack:
            stores = {}
            for model in models:
                if model == "GLOENS":
                    continue
                try:
                    stores[model] = stack.enter_context(open_cloudferro(
                        forecast_store(model, init), endpoint=args.endpoint,
                        anonymous=not args.authenticated))
                except Exception as exc:
                    failures.append(dict(init=init, lead="", model=model,
                                         basin="", band="", error=str(exc)))
            for lead in range(args.lead_start, args.lead_stop+1):
                for basin, band, bounds in regions():
                    # Each geographic window remains separate, including Pacific
                    # and Southern Ocean segments.
                    current = {}
                    for model in models:
                        try:
                            if model == "GLOENS":
                                offset = args.gloens_init_offset_days
                                glead = lead-offset
                                if not 0 <= glead <= 27:
                                    raise ValueError("GLOENS lead out of range")
                                gi = (start+timedelta(days=offset)).strftime("%Y%m%d")
                                ds = load_gloens_mean(args.gloens_npy_dir,
                                      args.gloens_grid_file, gi, glead,
                                      include_currents=False)
                                try:
                                    f, p, nrows = one_spectrum(ds, args.variable, 0,
                                        bounds, args.n_lat, args.n_lon,
                                        args.min_valid_fraction)
                                finally:
                                    ds.close()
                            else:
                                f, p, nrows = one_spectrum(stores[model], args.variable,
                                    lead, bounds, args.n_lat, args.n_lon,
                                    args.min_valid_fraction)
                            current[model] = (p, nrows)
                        except Exception as exc:
                            failures.append(dict(init=init, lead=lead, model=model,
                                basin=basin, band=band, error=str(exc)))
                    if set(current) != set(models):
                        continue
                    for model, (power, nrows) in current.items():
                        for k, value in zip(freqs, power):
                            if np.isfinite(value):
                                rows.append(dict(model=model, init=init, lead=lead,
                                    basin=basin, band=band,
                                    lon_min=bounds[2], lon_max=bounds[3],
                                    frequency_cpkm=float(k), psd=float(value),
                                    accepted_rows=nrows))
        print(f"Processed {index}/{len(starts)} initializations: {init}", flush=True)
        write_csv(args.output_dir/"spectra.csv", rows,
            ["model","init","lead","basin","band","lon_min","lon_max",
             "frequency_cpkm","psd","accepted_rows"])
        write_csv(args.output_dir/"failures.csv", failures,
            ["init","lead","model","basin","band","error"])
    if not rows:
        raise RuntimeError("No complete comparisons; inspect failures.csv")
    import pandas as pd
    frame = pd.DataFrame(rows)
    # Window averaging is by accepted latitude rows and geographic sector width.
    # This is a sector-area proxy, not an exact wet-cell-area integration.
    frame["weight"] = frame.accepted_rows*(frame.lon_max-frame.lon_min)
    group = frame.groupby(["model","init","lead","basin","band","frequency_cpkm"])
    combined = group.apply(lambda g: np.average(g.psd, weights=g.weight),
                           include_groups=False).rename("psd").reset_index()
    combined.to_csv(args.output_dir/"per_case.csv", index=False)
    summary = combined.groupby(["model","lead","basin","band","frequency_cpkm"]).agg(
        annual_mean_psd=("psd","mean"), n_initializations=("psd","count"),
        annual_std_psd=("psd","std")).reset_index()
    summary.to_csv(args.output_dir/"annual_summary.csv", index=False)
    print(f"Saved {len(combined)} per-case frequency rows and annual summary")


def write_csv(path, rows, cols):
    with path.open("w", newline="") as file:
        w = csv.DictWriter(file, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variable", choices=["sst","ssh"], default="sst")
    p.add_argument("--systems", nargs="+",
                   choices=["GLONET","GLOW","GLO12","GLOENS"],
                   default=["GLONET","GLOW","GLO12","GLOENS"])
    p.add_argument("--start", default="2024-01-03")
    p.add_argument("--end", default="2024-12-25")
    p.add_argument("--init-step-days", type=int, default=7)
    p.add_argument("--lead-start", type=int, default=0)
    p.add_argument("--lead-stop", type=int, default=8)
    p.add_argument("--gloens-init-offset-days", type=int, default=0)
    p.add_argument("--gloens-npy-dir", default="/sortref/calval/CLASS4_scores")
    p.add_argument("--gloens-grid-file", default=DEFAULT_GLOENS_GRID)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--min-valid-fraction", type=float, default=.85)
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("results/annual_basin_psd"))
    a = p.parse_args()
    if (a.start>a.end or a.init_step_days<1 or a.lead_start<0 or
        a.lead_stop<a.lead_start or a.n_lon<16 or a.n_lat<3 or
        not 0<a.min_valid_fraction<=1 or len(set(a.systems))!=len(a.systems)):
        p.error("Invalid arguments")
    run(a)


if __name__ == "__main__":
    main()
