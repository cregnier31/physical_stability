#!/usr/bin/env python3
"""Read-only preflight for private GLOENS and static mask files.

Run on the compute cluster after installing the package:
  python scripts/validate_cluster_inputs.py --init-date 20241205

This script does not load full forecast arrays into memory or write results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import xarray as xr

from physical_stability.gloens_npy import filename
from physical_stability.masks import MASK_PATHS

NPY_DIR = Path("/sortref/calval/CLASS4_scores")
GRID = Path("/data/rd_exchange/cregnier/static/iORCA025_domcfg_v3.1.nc")


def inspect_nc(path: Path, candidates: tuple[str, ...]):
    print(f"\nNETCDF {path}", flush=True)
    if not path.is_file():
        print("  MISSING")
        return False
    with xr.open_dataset(path, decode_times=False) as ds:
        print("  dimensions:", dict(ds.sizes))
        for name in candidates:
            if name in ds:
                v = ds[name]
                print(f"  {name}: dims={v.dims}, shape={v.shape}, dtype={v.dtype}, units={v.attrs.get('units')}")
        print("  available data variables:", list(ds.data_vars)[:35])
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-date", default="20241205")
    p.add_argument("--npy-dir", type=Path, default=NPY_DIR)
    p.add_argument("--grid-file", type=Path, default=GRID)
    args = p.parse_args()

    ok = inspect_nc(
        args.grid_file,
        ("deptht", "gdept_1d", "gdept_0", "gphit", "glamt",
         "nav_lat", "nav_lon", "e3t_0", "tmask"),
    )
    for system, path in MASK_PATHS.items():
        print(f"\n{system} mask:", flush=True)
        ok = inspect_nc(path, ("tmask", "mask", "mask_thetao", "wet_mask", "wet", "tmaskutil")) and ok

    print("\nGLOENS .npy arrays (memory-mapped headers only):", flush=True)
    for lead in (0, 8, 9, 27):
        for var in ("thetao", "so", "uo", "vo"):
            path = filename(args.npy_dir, var, lead, args.init_date)
            if not path.is_file():
                print("  MISSING", path)
                if var in ("thetao", "so"):
                    ok = False
                continue
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            print(f"  lead={lead:03d} {var}: shape={a.shape}, dtype={a.dtype}")
            if var in ("thetao", "so") and a.ndim != 3:
                print("  WARNING: expected 3D arrays; verify axis convention")
                ok = False

    print("\nPreflight:", "PASS (metadata present; grid compatibility still needs checking)" if ok else "INCOMPLETE")
    print("No physical N2 values were calculated by this metadata preflight.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
