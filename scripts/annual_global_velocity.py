#!/usr/bin/env python3
"""Annual, paired, area-weighted near-surface velocity diagnostics.

Outputs one row per model/init/lead/metric and an across-init summary.
All four systems must be present at a given valid time for a paired case.
No interpolation between grids: statistics describe each native resolution.
This is NOT EKE and is NOT a verification score.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from physical_stability.forecasts import ENDPOINT, forecast_store, open_cloudferro
from physical_stability.gloens_npy import DEFAULT_GLOENS_GRID
from global_velocity_maps import (
    FIELDS, _regular_field, _gloens_field, R,
)


def weighted_stats(values, weights):
    """Area-weighted moments and weighted quantiles, excluding invalid cells."""
    a = np.asarray(values, dtype=float).ravel()
    w = np.broadcast_to(weights, values.shape).ravel().astype(float)
    ok = np.isfinite(a) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return None
    a, w = a[ok], w[ok]
    total = np.sum(w)
    if not np.isfinite(total) or total <= 0:
        return None
    mean = np.sum(a * w) / total
    rms = np.sqrt(np.sum(a*a*w) / total)
    std = np.sqrt(max(0., np.sum((a-mean)**2*w) / total))
    ix = np.argsort(a)
    ordered = a[ix]
    cum = (np.cumsum(w[ix]) - .5*w[ix]) / total
    q05, q50, q95 = np.interp([.05, .5, .95], cum, ordered)
    return dict(mean=mean, std=std, rms=rms, p05=q05, median=q50,
                p95=q95, mean_abs=np.sum(np.abs(a)*w)/total,
                valid_area_m2=total, valid_cells=int(len(a)))


def regular_area(lat, lon):
    """Spherical latitude-longitude cell area from coordinate midpoints."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if len(lat) < 3 or len(lon) < 3:
        raise ValueError("Need >=3 points per horizontal axis")
    lat_edges = np.empty(len(lat)+1)
    lat_edges[1:-1] = .5*(lat[:-1]+lat[1:])
    lat_edges[0] = lat[0] - .5*(lat[1]-lat[0])
    lat_edges[-1] = lat[-1] + .5*(lat[-1]-lat[-2])
    lat_edges = np.clip(lat_edges, -90., 90.)
    lon_edges = np.empty(len(lon)+1)
    lon_edges[1:-1] = .5*(lon[:-1]+lon[1:])
    lon_edges[0] = lon[0] - .5*(lon[1]-lon[0])
    lon_edges[-1] = lon[-1] + .5*(lon[-1]-lon[-2])
    dx = np.deg2rad(np.diff(lon_edges))
    dy = np.diff(np.sin(np.deg2rad(lat_edges)))
    if np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError("Nonmonotonic regular coordinates")
    return R*R*dy[:, None]*dx[None, :]


def dates(start, end, step):
    d = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")
    while d <= last:
        yield d
        d += timedelta(days=step)


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = tuple(args.systems)
    metrics = list(FIELDS)
    metric_cols = ["mean", "std", "rms", "p05", "median", "p95",
                   "mean_abs", "valid_area_m2", "valid_cells"]
    columns = ["model", "init", "lead", "valid_date", "metric", "depth_m"] + metric_cols
    rows, failures, paired = [], [], []
    # Cache the horizontal grid area; use F-cell area for F-point diagnostics.
    area_t = area_f = lat_t = lat_f = None
    if "GLOENS" in models:
        with xr.open_dataset(args.gloens_grid_file) as grid:
            def arr(name):
                return np.asarray(grid[name].squeeze(drop=True).values, float)
            area_t = arr("e1t") * arr("e2t")
            area_f = arr("e1f") * arr("e2f")
            lat_t, lat_f = arr("gphit"), arr("gphif")
    for index, init_dt in enumerate(dates(args.start, args.end, args.init_step_days), 1):
        init = init_dt.strftime("%Y%m%d")
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
                                         error=f"open: {exc}"))
            for lead in range(args.lead_start, args.lead_stop+1):
                case = {}
                for model in models:
                    try:
                        if model == "GLOENS":
                            gi = init_dt + timedelta(days=args.gloens_init_offset_days)
                            glead = lead - args.gloens_init_offset_days
                            if not 0 <= glead <= 27:
                                raise ValueError(f"GLOENS lead {glead} outside 0..27")
                            data, coords, depth = _gloens_field(
                                args, glead, gi.strftime("%Y%m%d"))
                            weights = {}
                            for metric in data:
                                fgrid = metric in ("relative_vorticity", "rossby_number")
                                area = area_f if fgrid else area_t
                                lat = lat_f if fgrid else lat_t
                                weights[metric] = np.where(lat <= args.max_latitude,
                                                           area, 0.)
                        else:
                            if model not in stores:
                                continue
                            ds = stores[model]
                            if lead >= ds.sizes["time"]:
                                raise IndexError("Lead outside time dimension")
                            data, lat, lon, depth = _regular_field(
                                ds, lead, args.depth_index, args.max_latitude)
                            area = regular_area(lat, lon)
                            weights = {m: area for m in data}
                        current = {}
                        for metric in metrics:
                            if metric not in data:
                                raise ValueError(f"Missing {metric} for {model}")
                            stats = weighted_stats(data[metric], weights[metric])
                            if stats is None:
                                raise ValueError(f"No valid area for {metric}")
                            current[metric] = dict(model=model, init=init,
                                lead=lead, valid_date=(init_dt+timedelta(days=lead)).strftime("%Y-%m-%d"),
                                metric=metric, depth_m=depth, **stats)
                        case[model] = current
                    except Exception as exc:
                        failures.append(dict(init=init, lead=lead, model=model,
                                             error=str(exc)))
                if args.paired_only and set(case) != set(models):
                    continue
                for model, entries in case.items():
                    rows.extend(entries.values())
                if set(case) == set(models):
                    paired.append((init, lead))
        print(f"Processed {index} initialization(s): {init}; paired cases {len(paired)}",
              flush=True)
        # Persist partial progress after each initialization, useful for HPC jobs.
        write_csv(args.output_dir/"per_case.csv", rows, columns)
        write_csv(args.output_dir/"failures.csv", failures,
                  ["init", "lead", "model", "error"])

    if not rows:
        raise RuntimeError("No valid cases; inspect failures.csv")
    frame = pd.DataFrame(rows)
    # Each initialization is one statistical replicate; do not pool grid cells
    # or treat leads as independent samples.
    summaries = []
    for (model, lead, metric), group in frame.groupby(["model", "lead", "metric"], sort=True):
        item = dict(model=model, lead=int(lead), metric=metric,
                    n_initializations=int(group.init.nunique()),
                    mean_of_area_means=float(group["mean"].mean()),
                    sd_across_initializations=float(group["mean"].std(ddof=1))
                    if len(group) > 1 else float("nan"),
                    median_area_mean=float(group["mean"].median()),
                    mean_spatial_rms=float(group["rms"].mean()),
                    mean_spatial_p95=float(group["p95"].mean()),
                    mean_valid_area_m2=float(group["valid_area_m2"].mean()))
        summaries.append(item)
    write_csv(args.output_dir/"annual_summary.csv", summaries, list(summaries[0]))
    write_csv(args.output_dir/"paired_cases.csv",
              [dict(init=i, lead=l) for i,l in paired], ["init", "lead"])
    print(f"Saved {len(rows)} per-case metric rows; {len(paired)} paired cases; "
          f"{len(failures)} failures to {args.output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", nargs="+",
                   choices=["GLONET", "GLOW", "GLO12", "GLOENS"],
                   default=["GLONET", "GLOW", "GLO12", "GLOENS"])
    p.add_argument("--start", default="2024-01-03")
    p.add_argument("--end", default="2024-12-25")
    p.add_argument("--init-step-days", type=int, default=7)
    p.add_argument("--lead-start", type=int, default=0)
    p.add_argument("--lead-stop", type=int, default=8)
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--max-latitude", type=float, default=80.)
    p.add_argument("--gloens-init-offset-days", type=int, default=0)
    p.add_argument("--gloens-npy-dir", default="/sortref/calval/CLASS4_scores")
    p.add_argument("--gloens-grid-file", default=DEFAULT_GLOENS_GRID)
    p.add_argument("--gloens-native-uv-confirmed", action="store_true")
    p.add_argument("--paired-only", action="store_true", default=True)
    p.add_argument("--allow-unpaired", dest="paired_only", action="store_false")
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("results/annual_global_velocity"))
    args = p.parse_args()
    if (args.start > args.end or args.init_step_days < 1 or args.lead_start < 0 or
        args.lead_stop < args.lead_start or args.depth_index < 0 or
        not 5 < args.max_latitude <= 80 or len(set(args.systems)) != len(args.systems)):
        p.error("Invalid date/lead/depth/latitude/system arguments")
    if "GLOENS" in args.systems and not args.gloens_native_uv_confirmed:
        p.error("Verify native GLOENS U/V staggering before using --gloens-native-uv-confirmed")
    run(args)


if __name__ == "__main__":
    main()
