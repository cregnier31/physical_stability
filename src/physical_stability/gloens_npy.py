"""Load precomputed GLOENS 50-member means from local CLASS4_scores .npy files.

A .npy contains no dimension/coordinate metadata. Require an independent
NetCDF grid file and an explicit axis order; never infer spatial positions.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import xarray as xr

VARIABLES = ("thetao", "so", "uo", "vo")
PREFIX = "glo4-ens50_ensmean"


def filename(directory: str | Path, variable: str, lead: int, init_date: str) -> Path:
    if variable not in VARIABLES:
        raise ValueError(f"Unknown GLOENS variable: {variable}")
    if not 0 <= lead <= 27:
        raise ValueError("GLOENS lead must be 000 through 027")
    datetime.strptime(init_date, "%Y%m%d")
    return Path(directory) / f"{PREFIX}_{variable}_leadtime_{lead:03d}_{init_date}.npy"


def load_gloens_mean(
    directory: str | Path,
    grid_file: str | Path,
    init_date: str,
    lead: int,
    axis_order: str = "depth,y,x",
    include_currents: bool = True,
) -> xr.Dataset:
    """Build one lead with explicit 3D depth/y/x and 2D latitude/longitude.

    Grid NetCDF must contain 1D 'deptht' (metres positive down) and 2D
    'nav_lat'/'nav_lon' (y,x). If array axes differ, pass --gloens-axes.
    """
    dims = tuple(part.strip() for part in axis_order.split(","))
    if len(dims) != 3 or set(dims) != {"depth", "y", "x"}:
        raise ValueError("axis_order must be a permutation of depth,y,x")
    grid_path = Path(grid_file)
    if not grid_path.is_file():
        raise FileNotFoundError(f"GLOENS grid metadata not found: {grid_path}")
    with xr.open_dataset(grid_path) as grid:
        for key in ("deptht", "nav_lat", "nav_lon"):
            if key not in grid:
                raise KeyError(f"Grid file requires '{key}'")
        depth = np.asarray(grid["deptht"].values)
        lat = np.asarray(grid["nav_lat"].values)
        lon = np.asarray(grid["nav_lon"].values)
    if depth.ndim != 1 or lat.ndim != 2 or lon.shape != lat.shape:
        raise ValueError("Grid must contain 1D deptht and matching 2D nav_lat/nav_lon")
    if depth.size < 2 or not np.all(np.diff(depth) > 0) or depth[0] < 0:
        raise ValueError("Depth levels must be increasing, positive-down metres")
    sizes = {"depth": len(depth), "y": lat.shape[0], "x": lat.shape[1]}
    expected = tuple(sizes[d] for d in dims)
    vars_to_read = VARIABLES if include_currents else VARIABLES[:2]
    data = {}
    for name in vars_to_read:
        path = filename(directory, name, lead, init_date)
        if not path.is_file():
            if name in ("uo", "vo") and include_currents:
                # Currents are optional for N2; retain both only as a pair.
                continue
            raise FileNotFoundError(f"Missing required GLOENS file: {path}")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.shape != expected:
            raise ValueError(
                f"{path.name}: shape {array.shape} != expected {expected} "
                f"for axis order {axis_order}; verify the grid and axes"
            )
        data[name] = (dims, array)
    if ("uo" in data) != ("vo" in data):
        raise ValueError("Both uo and vo are required to retain currents")
    valid_time = np.datetime64(
        datetime.strptime(init_date, "%Y%m%d") + timedelta(days=lead), "ns"
    )
    ds = xr.Dataset(
        data,
        coords={
            "depth": depth, "y": np.arange(sizes["y"]),
            "x": np.arange(sizes["x"]),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), ((lon + 180) % 360) - 180),
        },
        attrs={
            "source_system": "GLOENS",
            "ensemble_processing": "precomputed_50_member_mean",
            "lead_index": lead,
            "time_assumption": "One daily lead increment from initialization date",
            "grid_source": str(grid_path),
        },
    )
    return ds.expand_dims(time=[valid_time])
