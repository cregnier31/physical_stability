import numpy as np
import pytest
import xarray as xr

from physical_stability import compute_n2, compare_n2, summarize_n2


def sample_dataset(stable=True):
    z = [0., 20., 50., 100.]
    # Increasing salinity with depth is statically stable.
    salinity = np.array([34.0, 34.2, 34.5, 34.8]) if stable else np.array([35.0, 34.7, 34.4, 34.0])
    return xr.Dataset(
        {
            "thetao": (("time_counter", "deptht", "y", "x"), np.full((1, 4, 2, 2), 10.)),
            "so": (("time_counter", "deptht", "y", "x"), np.broadcast_to(salinity[None, :, None, None], (1, 4, 2, 2)).copy()),
            "nav_lat": (("y", "x"), np.full((2, 2), 45.)),
            "nav_lon": (("y", "x"), np.full((2, 2), -20.)),
        },
        coords={"time_counter": [0], "deptht": z, "y": [0, 1], "x": [0, 1]},
    )


def test_n2_stable_and_midpoints():
    n2 = compute_n2(sample_dataset())
    assert n2.shape == (1, 3, 2, 2)
    np.testing.assert_allclose(n2.depth_mid, [10., 35., 75.])
    assert np.all(n2.values > 0)


def test_n2_unstable_and_disagreement():
    a = compute_n2(sample_dataset(stable=True))
    b = compute_n2(sample_dataset(stable=False))
    assert np.all(b.values < 0)
    compared = compare_n2(a, b)
    metrics = summarize_n2(compared)
    assert metrics["reference_unstable_fraction"] == 0.
    assert metrics["prediction_unstable_fraction"] == 1.
    assert metrics["ai_only_fraction_of_stable_reference"] == 1.


def test_reject_nonmonotonic_depth():
    ds = sample_dataset().assign_coords(deptht=[0., 30., 20., 100.])
    with pytest.raises(ValueError, match="Depth"):
        compute_n2(ds)


def test_mismatched_coordinates_rejected():
    a = compute_n2(sample_dataset())
    b = a.assign_coords(y=[4, 5])
    with pytest.raises(ValueError):
        compare_n2(a, b)
