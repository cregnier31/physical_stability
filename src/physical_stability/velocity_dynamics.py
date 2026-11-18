"""Three-dimensional horizontal-velocity diagnostics on a collocated regular grid.

Velocities must be eastward/northward in m/s and collocated at tracer points.
Native NEMO U/V staggering requires a separate, metric-aware collocation step.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .surface_dynamics import _derivatives, surface_diagnostics


def _check_depth(depth):
    z = np.asarray(depth, dtype=float)
    if z.ndim != 1 or z.size < 3 or not np.all(np.isfinite(z)):
        raise ValueError("Depth must be a finite 1D array of at least 3 levels")
    if not np.all(np.diff(z) > 0):
        raise ValueError("Depth must be strictly increasing positive-down in metres")
    return z


def velocity_diagnostics(uo, vo, depth, longitude, latitude, n2=None, ssh=None,
                         min_abs_latitude=5.0):
    """Calculate 3D horizontal circulation diagnostics.

    Parameters
    ----------
    uo, vo : arrays (depth, latitude, longitude)
        Collocated east/north velocities in m/s.
    depth : 1D positive-down depth coordinate, metres.
    n2 : optional array (depth, latitude, longitude), s^-2
        Must be collocated at velocity depths (not midpoint N2 levels).
    ssh : optional 2D SSH in metres for surface ageostrophic residual.
    """
    z = _check_depth(depth)
    u, v = np.asarray(uo, float), np.asarray(vo, float)
    lat, lon = np.asarray(latitude, float), np.asarray(longitude, float)
    shape = (z.size, lat.size, lon.size)
    if u.shape != shape or v.shape != shape:
        raise ValueError(f"uo and vo must have shape {shape}")
    valid = np.isfinite(u) & np.isfinite(v)
    u, v = np.where(valid, u, np.nan), np.where(valid, v, np.nan)
    ux, uy, vx, vy = [], [], [], []
    for uk, vk in zip(u, v):
        a, b = _derivatives(uk, lon, lat)
        c, d = _derivatives(vk, lon, lat)
        ux.append(a); uy.append(b); vx.append(c); vy.append(d)
    ux, uy, vx, vy = (np.stack(a) for a in (ux, uy, vx, vy))
    zeta = vx - uy
    strain = np.hypot(ux - vy, vx + uy)
    f = 2 * 7.2921159e-5 * np.sin(np.deg2rad(lat))[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rossby = np.where(np.abs(lat)[None, :, None] >= min_abs_latitude,
                          zeta / f, np.nan)
    dims = ("depth", "latitude", "longitude")
    def variable(a, units):
        return (dims, a, {"units": units})
    data = {
        "kinetic_energy": variable(0.5 * (u*u + v*v), "m2 s-2"),
        "speed": variable(np.hypot(u, v), "m s-1"),
        "relative_vorticity": variable(zeta, "s-1"),
        "rossby_number": variable(rossby, "1"),
        "horizontal_divergence": variable(ux + vy, "s-1"),
        "strain": variable(strain, "s-1"),
        "okubo_weiss": variable(strain**2 - zeta**2, "s-2"),
    }
    # Vertical derivatives need three adjacent wet levels. Never bridge NaN gaps.
    du_dz = np.gradient(u, z, axis=0, edge_order=2)
    dv_dz = np.gradient(v, z, axis=0, edge_order=2)
    wet = valid.copy()
    wet[1:-1] &= valid[:-2] & valid[2:]
    wet[0] &= valid[1] & valid[2]
    wet[-1] &= valid[-2] & valid[-3]
    shear2 = np.where(wet, du_dz**2 + dv_dz**2, np.nan)
    data["vertical_shear_squared"] = variable(shear2, "s-2")
    if n2 is not None:
        strat = np.asarray(n2, float)
        if strat.shape != shape:
            raise ValueError("n2 must be collocated at the velocity depth levels")
        with np.errstate(divide="ignore", invalid="ignore"):
            ri = np.where(shear2 > 0, strat / shear2, np.nan)
        data["gradient_richardson_number"] = variable(ri, "1")
    ds = xr.Dataset(data, coords={"depth": z, "latitude": lat, "longitude": lon},
                    attrs={"velocity_grid": "collocated regular latitude-longitude",
                           "vertical_coordinate": "positive down (m)",
                           "warning": "Do not pass native staggered NEMO U/V without collocation"})
    if ssh is not None:
        geo = surface_diagnostics(ssh, lon, lat,
                                  min_abs_latitude=min_abs_latitude)
        ds["ageostrophic_u"] = (("latitude", "longitude"), u[0] - geo.ug.values,
                               {"units": "m s-1"})
        ds["ageostrophic_v"] = (("latitude", "longitude"), v[0] - geo.vg.values,
                               {"units": "m s-1"})
        ds["ageostrophic_ke"] = (("latitude", "longitude"),
                                0.5*((u[0]-geo.ug.values)**2 + (v[0]-geo.vg.values)**2),
                                {"units": "m2 s-2"})
    return ds


def depth_kinetic_energy_psd(uo, vo, depth, longitude, latitude,
                             min_valid_fraction=0.85):
    """Zonal velocity KE spectrum at each depth, m3/s2 per cycle/km.

    PSD(u)+PSD(v) divided by two; this is an eddy/variance spectrum
    after zonal detrending, not the full mean-flow kinetic energy.
    """
    from .psd import zonal_psd
    z = _check_depth(depth)
    u, v = np.asarray(uo, float), np.asarray(vo, float)
    if u.shape != v.shape or u.shape != (z.size, len(latitude), len(longitude)):
        raise ValueError("Velocity arrays and coordinates have inconsistent shapes")
    spectra = []
    for uk, vk in zip(u, v):
        # Shared finite mask prevents different accepted rows for u and v.
        common = np.isfinite(uk) & np.isfinite(vk)
        pu = zonal_psd(np.where(common, uk, np.nan), longitude,
                       float(np.mean(latitude)), min_valid_fraction)
        pv = zonal_psd(np.where(common, vk, np.nan), longitude,
                       float(np.mean(latitude)), min_valid_fraction)
        spectra.append(0.5 * (pu.psd + pv.psd))
    return xr.Dataset({"ke_psd": xr.concat(spectra, dim=xr.IndexVariable("depth", z))},
                      attrs={"method": "0.5*(zonal PSD(u)+zonal PSD(v)); detrended",
                             "warning": "Spectral variance, not total kinetic energy"})
