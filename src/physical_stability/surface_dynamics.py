"""Regular-grid surface dynamics: geostrophic currents, vorticity, strain and fronts."""
from __future__ import annotations
import numpy as np
import xarray as xr

OMEGA = 7.2921159e-5
R_EARTH = 6371000.0
GRAVITY = 9.80665


def _derivatives(field, longitude, latitude):
    a = np.asarray(field, float)
    lon = np.asarray(longitude, float)
    lat = np.asarray(latitude, float)
    if a.ndim != 2 or a.shape != (lat.size, lon.size) or min(a.shape) < 3:
        raise ValueError("Expected a 2D (latitude, longitude) field, at least 3x3")
    if not (np.all(np.diff(lon) > 0) and np.all(np.diff(lat) > 0)):
        raise ValueError("Coordinates must be strictly increasing")
    if np.max(np.abs(lat)) >= 89:
        raise ValueError("Too close to the poles")
    dlat, dlon = np.gradient(a, np.deg2rad(lat), np.deg2rad(lon), edge_order=2)
    dx = dlon / (R_EARTH * np.cos(np.deg2rad(lat))[:, None])
    dy = dlat / R_EARTH
    valid = np.isfinite(a)
    vx, vy = valid.copy(), valid.copy()
    vx[:, 1:-1] &= valid[:, :-2] & valid[:, 2:]
    vx[:, 0] &= valid[:, 1] & valid[:, 2]
    vx[:, -1] &= valid[:, -2] & valid[:, -3]
    vy[1:-1, :] &= valid[:-2, :] & valid[2:, :]
    vy[0, :] &= valid[1, :] & valid[2, :]
    vy[-1, :] &= valid[-2, :] & valid[-3, :]
    return np.where(vx, dx, np.nan), np.where(vy, dy, np.nan)


def surface_diagnostics(ssh, longitude, latitude, sst=None, min_abs_latitude=5.0):
    """Compute geostrophic surface diagnostics on a regular lat/lon grid.

    SSH in metres; optional SST in degrees Celsius. Not a native NEMO
    staggered-grid operator. Regridding can modify small-scale gradients.
    """
    lon = np.asarray(longitude, float)
    lat = np.asarray(latitude, float)
    etax, etay = _derivatives(ssh, lon, lat)
    f = 2 * OMEGA * np.sin(np.deg2rad(lat))[:, None]
    safe = np.abs(lat)[:, None] >= min_abs_latitude
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(safe, -GRAVITY * etay / f, np.nan)
        v = np.where(safe, GRAVITY * etax / f, np.nan)
    ux, uy = _derivatives(u, lon, lat)
    vx, vy = _derivatives(v, lon, lat)
    zeta = vx - uy
    strain = np.hypot(ux - vy, vx + uy)
    dims = ("latitude", "longitude")
    def var(a, units):
        return (dims, a, {"units": units})
    data = {
        "ug": var(u, "m s-1"),
        "vg": var(v, "m s-1"),
        "geostrophic_speed": var(np.hypot(u, v), "m s-1"),
        "geostrophic_ke": var(0.5 * (u*u + v*v), "m2 s-2"),
        "relative_vorticity": var(zeta, "s-1"),
        "rossby_proxy": var(np.where(safe, zeta / f, np.nan), "1"),
        "strain": var(strain, "s-1"),
        "okubo_weiss": var(strain**2 - zeta**2, "s-2"),
    }
    if sst is not None:
        tx, ty = _derivatives(sst, lon, lat)
        data["sst_gradient"] = var(1000 * np.hypot(tx, ty), "degC km-1")
    return xr.Dataset(data, coords={"latitude": lat, "longitude": lon},
                      attrs={"method": "spherical metric gradients on regular lat/lon grid",
                             "caveat": "Not a native-grid NEMO operator; gradients sensitive to remapping"})
