"""Unit tests for native OceanBench-inspired PSD helper."""
import numpy as np
import pytest

from physical_stability.psd import band_metrics, zonal_psd


def test_zonal_sinusoid_peak_and_energy():
    n = 256
    lon = np.linspace(-50, -20, n)
    lat = 30.0
    dx = 111.195 * np.cos(np.deg2rad(lat)) * (lon[1]-lon[0])
    wavelength = n * dx / 8
    x = np.arange(n) * dx
    field = np.broadcast_to(np.sin(2 * np.pi * x / wavelength), (6, n)).copy()
    result = zonal_psd(field, lon, lat)
    freq = result.freq_lon.values
    peak = np.argmax(result.psd.values[1:]) + 1
    assert abs(1/freq[peak] - wavelength) < .05 * wavelength
    assert result.attrs["accepted_rows"] == 6
    assert band_metrics(result)["total_variance"] > 0


def test_zonal_rejects_irregular_longitude():
    lon = np.linspace(-50, -20, 64)
    lon[20] += 0.2
    with pytest.raises(ValueError, match="equally spaced"):
        zonal_psd(np.ones((4, 64)), lon, 30)


def test_zonal_missing_rows_and_coverage():
    lon = np.linspace(-50, -20, 64)
    field = np.random.default_rng(7).normal(size=(3, 64))
    field[0, :30] = np.nan
    spectrum = zonal_psd(field, lon, 30)
    assert spectrum.attrs["accepted_rows"] == 2
