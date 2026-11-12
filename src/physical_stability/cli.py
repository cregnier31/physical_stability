"""Command-line interface for NEMO and AI N² comparison."""
from __future__ import annotations

import argparse
import json
import xarray as xr
from .diagnostics import compute_n2, compare_n2, summarize_n2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare NEMO reference vs AI N² stability")
    parser.add_argument("--reference", required=True, help="Reference NetCDF")
    parser.add_argument("--prediction", required=True, help="AI prediction NetCDF")
    parser.add_argument("--output", default="stability_comparison.nc")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path")
    parser.add_argument("--temperature", default="thetao")
    parser.add_argument("--salinity", default="so")
    parser.add_argument("--depth", default="deptht")
    parser.add_argument("--latitude", default="nav_lat")
    parser.add_argument("--longitude", default="nav_lon")
    parser.add_argument("--eos", choices=("eos80", "teos10"), default="eos80")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    a = parser.parse_args(argv)
    opts = dict(temperature=a.temperature, salinity=a.salinity, depth=a.depth,
                latitude=a.latitude, longitude=a.longitude, eos=a.eos)
    with xr.open_dataset(a.reference) as ref, xr.open_dataset(a.prediction) as pred:
        nref = compute_n2(ref, **opts)
        npred = compute_n2(pred, **opts)
        compared = compare_n2(nref, npred, epsilon=a.epsilon)
        summary = summarize_n2(compared)
        compared.to_netcdf(a.output)
    print(json.dumps(summary, indent=2))
    if a.summary:
        with open(a.summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
