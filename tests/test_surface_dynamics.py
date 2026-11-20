import numpy as np
import pytest

from physical_stability.surface_dynamics import (
    GRAVITY, OMEGA, R_EARTH, surface_diagnostics,
)


def test_linear_ssh_geostrophic_velocity():
    lat = np.linspace(25, 35, 41)
    lon = np.linspace(-50, -20, 61)
    ssh = 1e-6 * R_EARTH * np.deg2rad(lat)[:, None] * np.ones((1, lon.size))
    ds = surface_diagnostics(ssh, lon, lat)
    f = 2 * OMEGA * np.sin(np.deg2rad(lat))
    expected = np.broadcast_to((-GRAVITY * 1e-6 / f)[:, None], (lat.size, lon.size))
    np.testing.assert_allclose(ds.ug.values, expected, rtol=1e-8)
    np.testing.assert_allclose(ds.vg.values, 0, atol=1e-10)


def test_constant_fields():
    lat = np.linspace(20, 40, 31)
    lon = np.linspace(-60, -20, 51)
    ssh = np.zeros((lat.size, lon.size))
    ds = surface_diagnostics(ssh, lon, lat, sst=np.ones_like(ssh) * 15)
    for name in ("ug", "vg", "relative_vorticity", "strain",
                 "okubo_weiss", "sst_gradient"):
        np.testing.assert_allclose(ds[name].values, 0, atol=1e-12)


def test_equatorial_mask_and_bad_coords():
    lat = np.linspace(-10, 10, 21)
    lon = np.linspace(-40, -20, 31)
    ds = surface_diagnostics(np.ones((21, 31)), lon, lat)
    assert np.isnan(ds.ug.sel(latitude=0)).all()
    with pytest.raises(ValueError):
        surface_diagnostics(np.ones((21, 31)), lon[::-1], lat)
