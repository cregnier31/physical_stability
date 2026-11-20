#!/usr/bin/env python3
"""Compare 10 reproducibly sampled GLOENS SSH members with deterministic systems.

Requires an original GLOENS Zarr store containing *individual* SSH members.
The precomputed CLASS4 .npy ensemble means are not valid inputs.
Regional prototype; not yet basin-masked global PSD.
"""
from __future__ import annotations
import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from physical_stability.forecasts import ENDPOINT, forecast_store, open_cloudferro
from physical_stability.psd import zonal_psd, band_metrics
from plot_oceanbench_psd import field_and_coords, common_grid

DETERMINISTIC = ("GLONET", "GLOW", "GLO12")


def choose_members(ds, dim, n, seed):
    if dim not in ds.dims:
        raise KeyError(f"GLOENS raw store lacks member dimension {dim!r}; dimensions: {dict(ds.sizes)}")
    if ds.sizes[dim] < n:
        raise ValueError(f"Requested {n} members, only {ds.sizes[dim]} available")
    rng = np.random.default_rng(seed)
    # Sample *positions* so this also works with noninteger member identifiers.
    positions = np.sort(rng.choice(ds.sizes[dim], size=n, replace=False))
    labels = ds[dim].values if dim in ds.coords else np.arange(ds.sizes[dim])
    return [(int(pos), str(labels[pos])) for pos in positions]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gloens-store-template", required=True,
                   help="Original member-resolved SSH Zarr path, with {init} placeholder")
    p.add_argument("--gloens-member-dim", default="ens")
    p.add_argument("--gloens-ssh-name", default="zos")
    p.add_argument("--gloens-init-offset-days", type=int, default=0)
    p.add_argument("--n-members", type=int, default=10)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--start", default="2024-01-03")
    p.add_argument("--end", default="2024-12-25")
    p.add_argument("--init-step-days", type=int, default=7)
    p.add_argument("--lead-start", type=int, default=0)
    p.add_argument("--lead-stop", type=int, default=8)
    p.add_argument("--lat-min", type=float, default=25.)
    p.add_argument("--lat-max", type=float, default=35.)
    p.add_argument("--lon-min", type=float, default=-50.)
    p.add_argument("--lon-max", type=float, default=-20.)
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--min-valid-fraction", type=float, default=.85)
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("results/gloens_member_ssh_psd"))
    a = p.parse_args()
    if ("{init}" not in a.gloens_store_template or a.n_members<1 or
        a.init_step_days<1 or a.lead_start<0 or a.lead_stop<a.lead_start or
        a.n_lon<8 or a.n_lat<3 or not 0<a.min_valid_fraction<=1):
        p.error("Invalid store template, member count, lead, grid or coverage")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["init","valid_date","lead","model","member_position","member_label",
              "frequency_cpkm","psd"]
    metrics_fields = ["init","valid_date","lead","model","member_position","member_label",
                      "accepted_rows","total_variance","Near-grid scale energy",
                      "Near-grid scale fraction","Regional scale energy",
                      "Regional scale fraction","Large scale energy",
                      "Large scale fraction","Near-grid / Regional noise index"]
    failures = []
    spectra = []
    metrics = []
    sampled_members = []
    start = datetime.strptime(a.start, "%Y-%m-%d")
    end = datetime.strptime(a.end, "%Y-%m-%d")
    date = start
    while date <= end:
        init = date.strftime("%Y%m%d")
        gloens_date = date+timedelta(days=a.gloens_init_offset_days)
        gloens_init = gloens_date.strftime("%Y%m%d")
        try:
            with ExitStack() as stack:
                ds_map = {model:stack.enter_context(open_cloudferro(
                    forecast_store(model, init), endpoint=a.endpoint,
                    anonymous=not a.authenticated)) for model in DETERMINISTIC}
                raw = stack.enter_context(open_cloudferro(
                    a.gloens_store_template.format(init=gloens_init),
                    endpoint=a.endpoint, anonymous=not a.authenticated))
                if a.gloens_ssh_name not in raw:
                    raise KeyError(f"SSH variable {a.gloens_ssh_name!r} missing; available: {list(raw.data_vars)}")
                members = choose_members(raw, a.gloens_member_dim, a.n_members, a.seed)
                for pos, label in members:
                    sampled_members.append(dict(init=init,gloens_init=gloens_init,
                                                member_position=pos,member_label=label))
                for lead in range(a.lead_start,a.lead_stop+1):
                    gloens_lead = lead-a.gloens_init_offset_days
                    if gloens_lead < 0 or gloens_lead >= raw.sizes.get("time",0):
                        failures.append(dict(init=init,lead=lead,error="GLOENS lead unavailable for aligned valid date"))
                        continue
                    bounds = (a.lat_min,a.lat_max,a.lon_min,a.lon_max)
                    native = {}
                    try:
                        for model, ds in ds_map.items():
                            if lead >= ds.sizes.get("time",0):
                                raise IndexError(f"{model} lead {lead} unavailable")
                            native[(model,"")] = field_and_coords(
                                ds.isel(time=slice(lead,lead+1)), "ssh", *bounds, 0)
                        for pos, label in members:
                            member_ds = raw.isel({a.gloens_member_dim:pos}).rename(
                                {a.gloens_ssh_name:"zos"} if a.gloens_ssh_name!="zos" else {})
                            native[("GLOENS",label)] = field_and_coords(
                                member_ds.isel(time=slice(gloens_lead,gloens_lead+1)),
                                "ssh", *bounds, 0)
                        _,lon, sampled = common_grid(native,a.n_lat,a.n_lon,bounds)
                        current = {}
                        for key, grid in sampled.items():
                            current[key] = zonal_psd(grid,lon,(a.lat_min+a.lat_max)/2,
                                                     min_valid_fraction=a.min_valid_fraction)
                    except Exception as exc:
                        failures.append(dict(init=init,lead=lead,error=str(exc)))
                        continue
                    valid_date=(date+timedelta(days=lead)).strftime("%Y-%m-%d")
                    member_positions={label:pos for pos,label in members}
                    for (model,label), psd in current.items():
                        base=dict(init=init,valid_date=valid_date,lead=lead,
                                  model=model,member_position=member_positions.get(label,""),
                                  member_label=label)
                        for freq,power in zip(psd.freq_lon.values,psd.psd.values):
                            spectra.append(dict(**base,frequency_cpkm=float(freq),psd=float(power)))
                        metrics.append(dict(**base,accepted_rows=psd.attrs["accepted_rows"],
                                            **band_metrics(psd)))
        except Exception as exc:
            failures.append(dict(init=init,lead="",error=str(exc)))
        print(f"Processed {init}: {len(metrics)} completed member/model spectra",flush=True)
        date+=timedelta(days=a.init_step_days)
    for name, cols, records in [
        ("spectra.csv",fields,spectra),("metrics.csv",metrics_fields,metrics),
        ("sampled_members.csv",["init","gloens_init","member_position","member_label"],sampled_members),
        ("failures.csv",["init","lead","error"],failures)]:
        with (a.output_dir/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=cols)
            writer.writeheader();writer.writerows(records)
    if not spectra:
        raise RuntimeError(f"No spectra produced; inspect {a.output_dir/'failures.csv'}")
    print(f"Saved {len(metrics)} spectra in {a.output_dir}")


if __name__=="__main__":
    main()
