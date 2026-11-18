"""OceanBench-inspired zonal spatial power spectral density (PSD).

Based on the methodology in mercator-ocean/oceanbench, branch OB_AI-TT_DEMO,
oceanbench/core/psd.py (EUPL-1.2). This is an independent NumPy implementation
for fixed-latitude, equally spaced longitude sections; it does not claim
bitwise equivalence to the xrft/Tukey/nfactor=2 OceanBench algorithm.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

BANDS_KM = {"Near-grid scale": (0.0, 500.0),
            "Regional scale": (500.0, 2000.0),
            "Large scale": (2000.0, np.inf)}


def zonal_psd(section: np.ndarray, longitude_deg: np.ndarray, latitude_deg: float,
              min_valid_fraction: float = 0.85) -> xr.Dataset:
    """Zonal one-sided PSD averaged over rows, in field-units squared per cycle/km.

    Parameters
    ----------
    section: array (lat, longitude)
        Must be uniformly spaced over longitude without a periodic seam.
    longitude_deg: 1D coordinate with strictly increasing, regular spacing.
    latitude_deg: representative latitude for zonal distance conversion.
    min_valid_fraction: reject rows missing more than this fraction of points.

    NaN gaps on retained rows are filled by linear interpolation, including
    endpoints. This can alter high-wavenumber energy, so coverage is reported.
    Detrends each row linearly and applies a Tukey window with alpha=0.5,
    normalizing by mean squared window amplitude.
    """
    a = np.asarray(section, dtype=float)
    lon = np.asarray(longitude_deg, dtype=float)
    if a.ndim != 2 or a.shape[1] != lon.size or lon.size < 8:
        raise ValueError("section must be (latitude, longitude) with >= 8 zonal points")
    dlon = np.diff(lon)
    if not (np.all(dlon > 0) and np.allclose(dlon, np.median(dlon), rtol=1e-3, atol=1e-6)):
        raise ValueError("Longitudes must be strictly increasing and equally spaced")
    if not (-85 < latitude_deg < 85):
        raise ValueError("Select a regional latitude band within 85 degrees")
    dx = 111.195 * np.cos(np.deg2rad(latitude_deg)) * np.median(dlon)
    if dx <= 0:
        raise ValueError("Invalid zonal spacing in km")
    n = lon.size
    # Tukey window alpha=0.5; taper one quarter of the samples at each edge.
    q = np.linspace(0, 1, n)
    window = np.ones(n)
    left = q < .25
    right = q > .75
    window[left] = 0.5 * (1 + np.cos(np.pi * (q[left] / .25 - 1)))
    window[right] = 0.5 * (1 + np.cos(np.pi * ((q[right] - .75) / .25)))
    freq = np.fft.rfftfreq(n, d=dx)
    power_rows = []
    good_fractions = []
    x = np.arange(n)
    for row in a:
        finite = np.isfinite(row)
        fraction = float(finite.mean())
        if fraction < min_valid_fraction or finite.sum() < 8:
            continue
        filled = np.interp(x, x[finite], row[finite])
        trend = np.polyval(np.polyfit(x, filled, 1), x)
        fft = np.fft.rfft((filled - trend) * window)
        # One-sided periodogram integrates to approximately window-normalized variance.
        power = np.abs(fft)**2 * dx / (n * np.mean(window**2))
        if n % 2 == 0:
            power[1:-1] *= 2
        else:
            power[1:] *= 2
        power_rows.append(power)
        good_fractions.append(fraction)
    if not power_rows:
        raise ValueError("No latitude rows with adequate finite data for PSD")
    result = xr.Dataset(
        {"psd": ("freq_lon", np.mean(power_rows, axis=0))},
        coords={"freq_lon": freq},
        attrs={"zonal_spacing_km": float(dx),
               "latitude_degrees": float(latitude_deg),
               "accepted_rows": len(power_rows),
               "mean_accepted_row_valid_fraction": float(np.mean(good_fractions)),
               "method": "one-sided FFT periodogram, Tukey 0.5, linear detrend",
               "note": "Independent approximation of OceanBench zonal PSD, not exact xrft output"},
    )
    result["psd"].attrs["units"] = "(input units)^2 km"
    result["freq_lon"].attrs["units"] = "cycles/km"
    return result


def band_metrics(psd: xr.Dataset, bands_km=None) -> dict[str, float]:
    """Band-integrated one-sided spectral energy using bin-width sums."""
    bands = BANDS_KM if bands_km is None else bands_km
    freq = np.asarray(psd.freq_lon.values)
    power = np.asarray(psd.psd.values)
    positive = freq > 0
    if len(freq) < 2:
        raise ValueError("PSD needs more than one Fourier frequency")
    df = float(freq[1]-freq[0])
    total = float(np.sum(power[positive]) * df)
    metrics = {"total_variance": total}
    for name, (lo, hi) in bands.items():
        wavelength = np.divide(1, freq, out=np.full_like(freq, np.inf), where=positive)
        selected = positive & (wavelength >= lo) & (wavelength < hi)
        energy = float(np.sum(power[selected]) * df)
        metrics[f"{name} energy"] = energy
        metrics[f"{name} fraction"] = energy / total if total > 0 else np.nan
    denom = metrics["Regional scale energy"]
    metrics["Near-grid / Regional noise index"] = (
        metrics["Near-grid scale energy"] / denom if denom > 0 else np.nan
    )
    return metrics
