#!/usr/bin/env python3
"""Annual mean 2D maps of six near-surface velocity diagnostics at selected leads.

Means each diagnostic across paired weekly initializations at native grid points.
Does NOT calculate diagnostics from annual-mean U/V (nonlinear operations differ).
GLOENS input is the precomputed ensemble-mean velocity, not individual members.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from global_velocity_maps import (
    FIELDS, _regular_field, _gloens_field, plot,
)
from physical_stability.forecasts import ENDPOINT, forecast_store, open_cloudferro
from physical_stability.gloens_npy import DEFAULT_GLOENS_GRID

MODELS = ("GLONET", "GLOW", "GLO12", "GLOENS")


def dates(start, end, step):
    current = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")
    while current <= last:
        yield current
        current += timedelta(days=step)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2024-01-03")
    p.add_argument("--end", default="2024-12-25")
    p.add_argument("--init-step-days", type=int, default=7)
    p.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--max-latitude", type=float, default=80.)
    p.add_argument("--gloens-init-offset-days", type=int, default=1)
    p.add_argument("--gloens-npy-dir", default="/sortref/calval/CLASS4_scores")
    p.add_argument("--gloens-grid-file", default=DEFAULT_GLOENS_GRID)
    p.add_argument("--gloens-native-uv-confirmed", action="store_true")
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("results/annual_global_velocity"))
    p.add_argument("--max-plot-points", type=int, default=500000)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--skip-plots", action="store_true")
    args = p.parse_args()
    if not args.gloens_native_uv_confirmed:
        p.error("Confirm native GLOENS U/V staggering with --gloens-native-uv-confirmed")
    if (args.start > args.end or args.init_step_days < 1 or args.depth_index < 0 or
        not 5 < args.max_latitude <= 80 or len(set(args.leads)) != len(args.leads) or
        min(args.leads) < args.gloens_init_offset_days or
        max(args.leads) - args.gloens_init_offset_days > 27):
        p.error("Invalid date, lead, depth, latitude or GLOENS offset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    counts = []
    initializations = list(dates(args.start, args.end, args.init_step_days))
    for lead in args.leads:
        sums, n_valid, coords, depths = {}, {}, {}, {}
        n_paired = 0
        for index, init_dt in enumerate(initializations, 1):
            init = init_dt.strftime("%Y%m%d")
            case = {}
            for model in MODELS:
                try:
                    if model == "GLOENS":
                        gi = init_dt + timedelta(days=args.gloens_init_offset_days)
                        glead = lead - args.gloens_init_offset_days
                        data, grid_coords, depth = _gloens_field(
                            args, glead, gi.strftime("%Y%m%d"))
                    else:
                        with open_cloudferro(forecast_store(model, init),
                                             endpoint=args.endpoint,
                                             anonymous=not args.authenticated) as ds:
                            data, lat, lon, depth = _regular_field(
                                ds, lead, args.depth_index, args.max_latitude)
                        grid_coords = {metric: (lat, lon) for metric in FIELDS}
                    if any(metric not in data or not np.isfinite(data[metric]).any()
                           for metric in FIELDS):
                        raise ValueError("Missing diagnostic or no finite grid cells")
                    for metric in FIELDS:
                        key = (model, metric)
                        shape = np.shape(data[metric])
                        if key in sums and sums[key].shape != shape:
                            raise ValueError(f"Grid shape changed for {key}: {shape}")
                        if key in coords:
                            oldlat, oldlon = coords[key]
                            newlat, newlon = grid_coords[metric]
                            if (not np.array_equal(oldlat, newlat, equal_nan=True) or
                                not np.array_equal(oldlon, newlon, equal_nan=True)):
                                raise ValueError(f"Grid coordinates changed for {key}")
                    case[model] = (data, grid_coords, depth)
                except Exception as exc:
                    failures.append(dict(init=init, lead=lead, model=model, error=str(exc)))
            # All four systems must exist for a date/lead: same valid-date sample.
            if set(case) != set(MODELS):
                print(f"lead {lead} init {init}: skipped unpaired case", flush=True)
                continue
            for model, (data, grid_coords, depth) in case.items():
                depths[model] = depth
                for metric in FIELDS:
                    key = (model, metric)
                    values = np.asarray(data[metric], dtype=np.float64)
                    finite = np.isfinite(values)
                    if key not in sums:
                        sums[key] = np.zeros_like(values, dtype=np.float64)
                        n_valid[key] = np.zeros_like(values, dtype=np.uint16)
                        coords[key] = grid_coords[metric]
                    sums[key] += np.where(finite, values, 0.)
                    n_valid[key] += finite.astype(np.uint16)
            n_paired += 1
            print(f"lead {lead} init {init}: paired {n_paired}/{index}", flush=True)
        if not n_paired:
            print(f"No paired initializations for lead {lead}; skipping", flush=True)
            counts.append(dict(lead=lead, n_paired=0))
            continue
        results = {}
        data_dir = args.output_dir / "annual_maps_data" / f"lead{lead:02d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        for model in MODELS:
            results[model] = {}
            for metric in FIELDS:
                key = (model, metric)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean = np.where(n_valid[key] > 0, sums[key] / n_valid[key], np.nan)
                lat, lon = coords[key]
                results[model][metric] = (mean, lat, lon)
                np.savez_compressed(data_dir / f"{model}_{metric}.npz",
                                    mean=mean.astype(np.float32),
                                    count=n_valid[key], latitude=lat, longitude=lon)
        counts.append(dict(lead=lead, n_paired=n_paired))
        if not args.skip_plots:
            plot(results, depths, args.output_dir / "maps" / f"lead{lead:02d}",
                 args.start[:4], lead, args.depth_index, args.max_plot_points,
                 args.dpi, title_label=f"annual mean {args.start} to {args.end} (n={n_paired})")
        print(f"lead {lead}: saved {n_paired} paired-case annual maps", flush=True)
    for filename, rows, columns in (
        ("annual_map_counts.csv", counts, ["lead", "n_paired"]),
        ("annual_map_failures.csv", failures, ["init", "lead", "model", "error"]),
    ):
        with (args.output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
