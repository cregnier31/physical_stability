import numpy as np
import pytest
import xarray as xr

from physical_stability.masks import wet_mask_for_ts, mask_ts_pairs, region_indices
from physical_stability.forecasts import detect_fields


def test_regional_3d_mask(tmp_path):
    path = tmp_path / "mask.nc"
    mask = np.ones((3, 4, 5), dtype="i1")
    mask[:, 1, 2] = 0
    xr.Dataset({"tmask": (("z", "y", "x"), mask)}).to_netcdf(path)
    ts = xr.DataArray(
        np.ones((1, 3, 2, 3)),
        dims=("time", "deptht", "y", "x"),
        coords={"deptht": [0., 10., 20.], "y": [0, 1], "x": [0, 1, 2]},
    )
    wet = wet_mask_for_ts(path, ts, horizontal_indices=(np.array([1, 2]), np.array([1, 2, 3])))
    assert wet.shape == (3, 2, 3)
    assert not bool(wet.isel(deptht=1, y=0, x=1))
    ds = xr.Dataset({"thetao": ts, "so": ts + 35})
    masked = mask_ts_pairs(ds, wet)
    assert np.isnan(masked.thetao.isel(time=0, deptht=0, y=0, x=1))


def test_mask_mismatch_fails(tmp_path):
    path = tmp_path / "mask.nc"
    xr.Dataset({"tmask": (("y", "x"), np.ones((4, 5)))}).to_netcdf(path)
    ts = xr.DataArray(np.ones((3, 2, 3)), dims=("deptht", "y", "x"),
                      coords={"deptht": [0, 10, 20], "y": [0, 1], "x": [0, 1, 2]})
    with pytest.raises(ValueError, match="horizontal shape"):
        wet_mask_for_ts(path, ts)
