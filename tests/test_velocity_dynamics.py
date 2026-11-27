import numpy as np
import pytest

from physical_stability.velocity_dynamics import (
    velocity_diagnostics, depth_kinetic_energy_psd,
)


def grid():
    z = np.array([0., 20., 50., 100.])
    lat = np.linspace(25, 35, 9)
    lon = np.linspace(-50, -20, 32)
    return z, lat, lon


def test_constant_velocity():
    z, lat, lon = grid()
    u = np.ones((len(z), len(lat), len(lon))) * 0.4
    v = np.zeros_like(u)
    ds = velocity_diagnostics(u, v, z, lon, lat, n2=np.ones_like(u)*1e-5)
    np.testing.assert_allclose(ds.kinetic_energy, 0.08)
    np.testing.assert_allclose(ds.relative_vorticity, 0, atol=1e-14)
    np.testing.assert_allclose(ds.horizontal_divergence, 0, atol=1e-14)
    np.testing.assert_allclose(ds.vertical_shear_squared, 0, atol=1e-14)
    assert np.isnan(ds.gradient_richardson_number).all()


def test_linear_vertical_shear_and_ri():
    z, lat, lon = grid()
    u = np.broadcast_to(0.01*z[:, None, None], (len(z), len(lat), len(lon)))
    v = np.zeros_like(u)
    ds = velocity_diagnostics(u, v, z, lon, lat, n2=np.full_like(u, 1e-4))
    np.testing.assert_allclose(ds.vertical_shear_squared, 1e-4, rtol=1e-10)
    np.testing.assert_allclose(ds.gradient_richardson_number, 1, rtol=1e-10)


def test_spectrum_and_shape_validation():
    z, lat, lon = grid()
    rng = np.random.default_rng(4)
    u = rng.normal(size=(len(z), len(lat), len(lon)))
    v = np.zeros_like(u)
    psd = depth_kinetic_energy_psd(u, v, z, lon, lat)
    assert psd.ke_psd.shape == (len(z), len(lon)//2 + 1)
    assert np.all(psd.ke_psd >= 0)
    with pytest.raises(ValueError):
        velocity_diagnostics(u, v[:, :, :-1], z, lon, lat)
    with pytest.raises(ValueError):
        velocity_diagnostics(u, v, z[::-1], lon, lat)
