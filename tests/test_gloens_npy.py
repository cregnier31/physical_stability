"""Tests for the 28-lead precomputed GLOENS ensemble-mean format."""
import numpy as np
import pytest
import xarray as xr

from physical_stability.gloens_npy import filename, load_gloens_mean
from physical_stability.forecasts import detect_fields, prepare_ts


def test_local_mean_and_current_pair(tmp_path):
    grid = xr.Dataset(
        coords={"deptht": [0., 10., 20.], "y": [0, 1], "x": [0, 1]},
        data_vars={
            "nav_lat": (("y", "x"), [[30., 30.], [31., 31.]]),
            "nav_lon": (("y", "x"), [[-60., -59.], [-60., -59.]]),
        },
    )
    grid_path = tmp_path / "grid.nc"
    grid.to_netcdf(grid_path)
    for var in ("thetao", "so", "uo", "vo"):
        np.save(filename(tmp_path, var, 27, "20241205"), np.ones((3, 2, 2)))
    ds = load_gloens_mean(tmp_path, grid_path, "20241205", 27)
    assert set(("thetao", "so", "uo", "vo")).issubset(ds)
    assert str(ds.time.values[0]).startswith("2025-01-01")
    fields = detect_fields(ds, "GLOENS")
    standard, _ = prepare_ts(ds, fields, "GLOENS")
    assert standard.thetao.shape == (1, 3, 2, 2)
    assert "ens" not in standard.dims


def test_shape_validation(tmp_path):
    grid_path = tmp_path / "grid.nc"
    xr.Dataset(
        coords={"deptht": [0., 10.], "y": [0, 1], "x": [0, 1]},
        data_vars={
            "nav_lat": (("y", "x"), np.zeros((2, 2))),
            "nav_lon": (("y", "x"), np.zeros((2, 2))),
        },
    ).to_netcdf(grid_path)
    for var in ("thetao", "so"):
        np.save(filename(tmp_path, var, 0, "20241205"), np.ones((3, 2, 2)))
    with pytest.raises(ValueError, match="shape"):
        load_gloens_mean(tmp_path, grid_path, "20241205", 0, axis_order="y,x,depth")
    with pytest.raises(ValueError, match="lead"):
        filename(tmp_path, "so", 28, "20241205")
