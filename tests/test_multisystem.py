"""Synthetic tests of multi-system store conventions (no remote S3 required)."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from physical_stability.forecasts import (
    MODEL_STORES, detect_fields, forecast_store, prepare_ts, select_day,
    subset_region,
)
from physical_stability.multisystem import run_multisystem


def regular_store(system, salinity_offset=0.0):
    lon_name, lat_name = MODEL_STORES[system][1:]
    lon = [280., 290., 300., 310.]
    lat = [20., 30., 40.]
    depth = [0., 10., 30.]
    shape = (2, 3, 3, 4)
    return xr.Dataset(
        {
            "thetao": (("time", "depth", lat_name, lon_name), np.full(shape, 12.)),
            "so": (("time", "depth", lat_name, lon_name),
                   np.broadcast_to(
                       (np.array(depth)[None, :, None, None] * .001
                        + 34.8 + salinity_offset),
                       shape
                   ).copy()),
            "zos": (("time", lat_name, lon_name), np.zeros((2, 3, 4))),
        },
        coords={
            "time": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
            "depth": depth,
            lon_name: lon,
            lat_name: lat,
        },
    )


def gloens_3d():
    shape = (2, 50, 3, 3, 4)
    return xr.Dataset(
        {
            "thetao": (("time", "ens", "depth", "y", "x"), np.full(shape, 12.)),
            "so": (("time", "ens", "depth", "y", "x"),
                   np.broadcast_to(
                       (34.5 + np.array([0., .2, .5]))[None, None, :, None, None],
                       shape
                   ).copy()),
            "latitude": (("y", "x"), np.broadcast_to(np.array([20., 30., 40.])[:, None], (3, 4))),
            "longitude": (("y", "x"), np.broadcast_to(np.array([-80., -70., -60., -50.])[None, :], (3, 4))),
        },
        coords={
            "time": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
            "ens": np.arange(50),
            "depth": [0., 10., 30.],
        },
    )


@pytest.mark.parametrize("system", ["GLONET", "GLOW", "GLO12"])
def test_regular_system_grid_and_subset(system):
    data = regular_store(system)
    fields = detect_fields(data, system)
    cropped = subset_region(data, fields, -80, -50, 20, 40)
    day = select_day(cropped, 0, 1)
    result, options = prepare_ts(day, fields, system)
    assert result["thetao"].dims == ("time", "deptht", "y", "x")
    assert result.sizes["time"] == 1
    assert result["nav_lat"].shape == (3, 4)
    assert result["nav_lon"].shape == (3, 4)
    assert options["depth"] == "deptht"


def test_expected_s3_paths():
    assert forecast_store("GLONET", "20260101") == (
        "oceanbench-bucket/public/ml-forecast-outputs/glonet/20260101.zarr"
    )
    assert forecast_store("GLOW", "20260101") == (
        "oceanbench-bucket/dev/ml-forecast-outputs/glowcascade_v5_ring/20260101.zarr"
    )


def test_gloens_mean_and_members():
    ds = gloens_3d()
    names = detect_fields(ds, "GLOENS")
    cropped = subset_region(ds, names, -80, -50, 20, 40)
    mean, _ = prepare_ts(cropped, names, "GLOENS", "mean")
    members, _ = prepare_ts(cropped, names, "GLOENS", "members")
    assert "ens" not in mean.dims
    assert members.sizes["ens"] == 50
    assert mean.so.shape == (2, 3, 3, 4)
    np.testing.assert_allclose(mean.so.values, members.so.mean("ens").values)


def test_ssh_only_rejected():
    ds = gloens_3d().drop_vars(["thetao", "so"])
    with pytest.raises(KeyError, match="temperature"):
        detect_fields(ds, "GLOENS")


def test_pipeline_with_synthetic_zarr_store(monkeypatch, tmp_path):
    from physical_stability import multisystem

    stores = {
        "glonet": regular_store("GLONET"),
        "glowcascade": regular_store("GLOW"),
        "GLO12": regular_store("GLO12", salinity_offset=.05),
        "gloens3d": gloens_3d(),
    }

    @contextmanager
    def fake_open(store, **kwargs):
        matched = [v for k, v in stores.items() if k in store]
        assert len(matched) == 1
        yield matched[0]

    # This test checks orchestration, not the physical GSW calculation.
    def fake_n2(ds, **kwargs):
        a = ds.so
        depth = a.deptht.values
        dz = np.diff(depth)
        top = a.isel(deptht=slice(0, -1)).values
        bot = a.isel(deptht=slice(1, None)).values
        data = (bot - top) / dz.reshape((1,) * (top.ndim - 3) + (len(dz), 1, 1))
        dims = tuple("depth_mid" if d == "deptht" else d for d in a.dims)
        coords = {d: a.coords[d] for d in a.dims if d != "deptht" and d in a.coords}
        coords["depth_mid"] = (depth[:-1] + depth[1:]) / 2
        coords["nav_lat"] = ds.nav_lat
        coords["nav_lon"] = ds.nav_lon
        return xr.DataArray(data, dims=dims, coords=coords)

    monkeypatch.setattr(multisystem, "open_cloudferro", fake_open)
    monkeypatch.setattr(multisystem, "compute_n2", fake_n2)
    args = SimpleNamespace(
        systems=["GLONET", "GLOW", "GLO12", "GLOENS"],
        gloens_store="s3://bucket/gloens3d.zarr",
        gloens_ensemble="mean", init_date="20260101",
        output=str(tmp_path), first_day=0, last_day=1,
        lon_min=-80., lon_max=-50., lat_min=20., lat_max=40.,
        endpoint="unused", authenticated=False, eos="eos80",
        config=None, epsilon=1e-8, no_static_masks=True,
    )
    summary, comparisons = run_multisystem(args)
    assert summary.is_file()
    assert comparisons.is_file()
    import csv
    import json
    with summary.open() as fh:
        rows = list(csv.DictReader(fh))
    assert {row["system"] for row in rows} == set(args.systems)
    with comparisons.open() as fh:
        statuses = json.load(fh)
    assert len(statuses) == 3
    assert all(item["status"] == "paired" for item in statuses)
