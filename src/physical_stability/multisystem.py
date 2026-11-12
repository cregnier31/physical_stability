"""Compare fixed-z 3D temperature/salinity stratification across CloudFerro systems.

Pointwise comparisons are attempted only on identical geographic, depth,
and time grids. Native-grid stability metrics are always reported.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from .diagnostics import compare_n2, compute_n2, summarize_n2
from .gloens_npy import DEFAULT_GLOENS_GRID, load_gloens_mean
from .masks import MASK_PATHS, wet_mask_for_ts, mask_ts_pairs, region_indices
from .forecasts import (
    ENDPOINT, SYSTEMS, detect_fields, forecast_store, open_cloudferro,
    prepare_ts, select_day, subset_region,
)

SUMMARY_FIELDS = (
    "system", "lead", "valid_time", "source_store", "eos",
    "ensemble_processing", "valid_cell_count",
    "unstable_cell_fraction", "mean_negative_n2_s-2", "mean_n2_s-2",
)


def standalone_summary(n2: xr.DataArray, epsilon: float) -> dict:
    """Grid-cell statistics; ensemble mode pools cells across members."""
    values = np.asarray(n2.values)
    good = np.isfinite(values)
    count = int(good.sum())
    if count == 0:
        return {
            "valid_cell_count": 0,
            "unstable_cell_fraction": float("nan"),
            "mean_negative_n2_s-2": float("nan"),
            "mean_n2_s-2": float("nan"),
        }
    a = values[good]
    return {
        "valid_cell_count": count,
        "unstable_cell_fraction": float(np.mean(a < -epsilon)),
        "mean_negative_n2_s-2": float(np.mean(np.maximum(-a, 0))),
        "mean_n2_s-2": float(np.mean(a)),
    }


def load_configuration(path: str | None) -> dict:
    """Optional JSON per-system variable and EOS overrides."""
    if path is None:
        return {}
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError("Configuration JSON must be an object keyed by model")
    unknown = set(config) - set(SYSTEMS)
    if unknown:
        raise ValueError(f"Unknown configuration systems: {sorted(unknown)}")
    for system, options in config.items():
        if not isinstance(options, dict):
            raise ValueError(f"{system} configuration must be an object")
        if set(options) - {"temperature", "salinity", "depth", "latitude", "longitude", "eos"}:
            raise ValueError(f"Unsupported configuration key(s) for {system}")
    return config


def run_multisystem(args) -> tuple[Path, Path]:
    """Run each requested lead separately; no full-global T/S read."""
    if not np.isfinite(args.epsilon) or args.epsilon < 0:
        raise ValueError("--epsilon must be finite and >= 0")
    systems = list(dict.fromkeys(name.upper() for name in args.systems))
    if not systems or any(s not in SYSTEMS for s in systems):
        raise ValueError(f"Systems must be among: {SYSTEMS}")
    npy_dir = getattr(args, "gloens_npy_dir", None)
    grid_file = getattr(args, "gloens_grid_file", None)
    npy_mode = bool(npy_dir)
    if npy_mode and not grid_file:
        raise ValueError("--gloens-grid-file is required for .npy arrays")
    if "GLOENS" in systems and not (args.gloens_store or npy_mode):
        raise ValueError(
            "GLOENS needs --gloens-store pointing to a *3D T/S* Zarr store. "
            "eddy_load/gloens_mean.py handles 2D zos only and cannot supply N2."
        )
    config = load_configuration(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    paths: dict[tuple[str, int], Path] = {}
    timestamps: dict[tuple[str, int], str] = {}
    from contextlib import nullcontext
    for system in systems:
        store = (
            (str(npy_dir) if npy_mode else args.gloens_store) if system == "GLOENS"
            else forecast_store(system, args.init_date)
        )
        overrides = config.get(system, {})
        eos = overrides.get("eos", args.eos)
        if eos not in ("eos80", "teos10"):
            raise ValueError(f"{system}: invalid EOS convention {eos}")
        if system == "GLOENS" and npy_mode:
            if args.gloens_ensemble != "mean":
                raise ValueError("Precomputed .npy files provide ensemble means, not members")
            context = nullcontext(None)
        else:
            context = open_cloudferro(
                store, endpoint=args.endpoint, anonymous=not args.authenticated
            )
        with context as ds:
            if system == "GLOENS" and npy_mode:
                leads = range(args.first_day, min(args.last_day, 28))
                if not list(leads):
                    raise ValueError("No GLOENS leads in requested window")
            else:
                fields = detect_fields(
                    ds, system,
                    **{k: overrides.get(k) for k in
                       ("temperature", "salinity", "depth", "latitude", "longitude")}
                )
                regional = subset_region(
                    ds, fields, args.lon_min, args.lon_max,
                    args.lat_min, args.lat_max,
                )
                # A model's shorter native forecast does not truncate other systems.
                stop = min(args.last_day, regional.sizes["time"])
                leads = range(args.first_day, max(args.first_day, stop))
            for lead in leads:
                if system == "GLOENS" and npy_mode:
                    one = load_gloens_mean(
                        npy_dir, grid_file, args.init_date, lead,
                        axis_order=getattr(args, "gloens_axes", "depth,y,x"),
                    )
                    fields = detect_fields(
                        one, system,
                        **{k: overrides.get(k) for k in
                           ("temperature", "salinity", "depth", "latitude", "longitude")}
                    )
                    one = subset_region(
                        one, fields, args.lon_min, args.lon_max,
                        args.lat_min, args.lat_max,
                    )
                    # Precomputed mean has no ensemble axis.
                    standard, options = prepare_ts(
                        one, fields, system, ensemble="mean"
                    )
                else:
                    one = regional.isel(time=slice(lead, lead + 1))
                    standard, options = prepare_ts(
                        one, fields, system, ensemble=args.gloens_ensemble
                    )
                mask_path = getattr(args, "mask_dir", None)
                if mask_path is not None:
                    mask_path = Path(mask_path) / MASK_PATHS[system].name
                else:
                    mask_path = MASK_PATHS[system]
                if getattr(args, "no_static_masks", False):
                    pass
                else:
                    indices = region_indices(
                        one if system == "GLOENS" and npy_mode else ds,
                        fields, args.lon_min, args.lon_max,
                        args.lat_min, args.lat_max,
                    )
                    # In the local GLOENS path, 'one' is already cropped.
                    # Its original global grid is required to locate the mask.
                    if system == "GLOENS" and npy_mode:
                        full_grid = load_gloens_mean(
                            npy_dir, grid_file, args.init_date, lead,
                            axis_order=getattr(args, "gloens_axes", "depth,y,x"),
                            include_currents=False,
                        )
                        indices = region_indices(
                            full_grid, fields, args.lon_min, args.lon_max,
                            args.lat_min, args.lat_max,
                        )
                    wet = wet_mask_for_ts(
                        mask_path, standard["thetao"],
                        horizontal_indices=indices,
                    )
                    standard = mask_ts_pairs(standard, wet)
                n2 = compute_n2(standard, eos=eos, **options)
                timestamp = str(one["time"].values[0])
                filename = out / system / f"n2_{system}_{args.init_date}_lead{lead:03d}.nc"
                filename.parent.mkdir(parents=True, exist_ok=True)
                xr.Dataset(
                    {"N2": n2},
                    attrs={
                        "source_system": system,
                        "source_store": store,
                        "eos_input_convention": eos,
                        "ensemble_processing": (
                            "precomputed_50_member_mean" if system == "GLOENS" and npy_mode
                            else args.gloens_ensemble if system == "GLOENS" else "none"
                        ),
                        "valid_time": timestamp,
                        "lead_index": lead,
                        "note": "N2 is calculated between fixed T-point depth levels using TEOS-10 GSW",
                    },
                ).to_netcdf(filename)
                paths[(system, lead)] = filename
                timestamps[(system, lead)] = timestamp
                scores.append(
                    {
                        "system": system, "lead": lead,
                        "valid_time": timestamp, "source_store": store,
                        "eos": eos,
                        "ensemble_processing": (
                            "precomputed_50_member_mean" if system == "GLOENS" and npy_mode
                            else args.gloens_ensemble if system == "GLOENS" else "none"
                        ),
                        **standalone_summary(n2, args.epsilon),
                    }
                )
                print(f"{system} lead {lead}: {filename}", flush=True)
    summary_file = out / "n2_multisystem_summary.csv"
    with summary_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(scores)

    # Compare against deterministic GLO12 only if all physical coordinates match.
    comparisons = []
    if "GLO12" in systems:
        for system in systems:
            if system == "GLO12":
                continue
            for lead in range(args.first_day, args.last_day):
                key = (system, lead)
                baseline = ("GLO12", lead)
                if key not in paths or baseline not in paths:
                    continue
                status = {
                    "system": system,
                    "reference": "GLO12",
                    "lead": lead,
                    "status": "",
                    "reason": "",
                }
                if timestamps[key] != timestamps[baseline]:
                    status.update(status="skipped", reason="Forecast valid time differs")
                    comparisons.append(status)
                    continue
                with xr.open_dataset(paths[key]) as a, xr.open_dataset(paths[baseline]) as b:
                    try:
                        # GLOENS ensemble member mode is not a single deterministic field.
                        if "ens" in a["N2"].dims:
                            raise ValueError(
                                "Ensemble members cannot be paired as a single field; "
                                "use --gloens-ensemble mean or compare each member separately"
                            )
                        result = compare_n2(b["N2"], a["N2"], epsilon=args.epsilon)
                        metrics = summarize_n2(result)
                        status.update(status="paired", reason="", **metrics)
                        paired_file = (
                            out / system / f"versus_GLO12_lead{lead:03d}.nc"
                        )
                        result.to_netcdf(paired_file)
                    except (ValueError, KeyError) as exc:
                        status.update(status="skipped", reason=str(exc))
                comparisons.append(status)
    comparison_file = out / "pointwise_comparisons.json"
    comparison_file.write_text(
        json.dumps(comparisons, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return summary_file, comparison_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-date", required=True, help="Forecast initialization YYYYMMDD")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--mask-dir", help="Override directory of model static masks")
    parser.add_argument("--no-static-masks", action="store_true", help="Disable masks (diagnostic only)")
    parser.add_argument("--gloens-npy-dir", help="Local CLASS4_scores ensemble-mean .npy directory")
    parser.add_argument("--gloens-grid-file", default=DEFAULT_GLOENS_GRID,
                        help="NEMO domain_cfg file (default: iORCA025_domcfg_v3.1.nc)")
    parser.add_argument("--gloens-axes", default="depth,y,x",
                        help="Explicit .npy axis order, e.g. depth,y,x")
    parser.add_argument("--gloens-store", help="S3 path to 3D T/S GLOENS Zarr (not 2DT zos)")
    parser.add_argument("--gloens-ensemble", choices=("mean", "members"), default="mean")
    parser.add_argument("--output", default="stability_results")
    parser.add_argument("--first-day", type=int, default=0)
    parser.add_argument("--last-day", type=int, default=28, help="Exclusive lead index; GLOENS supports 0..27")
    parser.add_argument("--lon-min", type=float, default=-80.0)
    parser.add_argument("--lon-max", type=float, default=0.0)
    parser.add_argument("--lat-min", type=float, default=20.0)
    parser.add_argument("--lat-max", type=float, default=65.0)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--authenticated", action="store_true", help="Use S3 environment credentials")
    parser.add_argument("--eos", choices=("eos80", "teos10"), required=True,
                        help="Explicit input T/S convention; can be overridden per-system in --config")
    parser.add_argument("--config", help="JSON file of per-system variable names and eos")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    args = parser.parse_args(argv)
    if args.last_day <= args.first_day:
        parser.error("--last-day must be larger than --first-day")
    summary, comparison = run_multisystem(args)
    print(f"Native-grid summary: {summary}")
    print(f"Pointwise comparison status: {comparison}")


if __name__ == "__main__":
    main()
