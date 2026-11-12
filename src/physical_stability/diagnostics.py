"""TEOS-10 buoyancy frequency and paired static-stability diagnostics.

Inputs use NEMO T-point fields on fixed one-dimensional positive-down z levels.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import gsw


def compute_n2(
    ds: xr.Dataset,
    temperature: str = "thetao",
    salinity: str = "so",
    depth: str = "deptht",
    latitude: str = "nav_lat",
    longitude: str = "nav_lon",
    eos: str = "eos80",
) -> xr.DataArray:
    """Compute N² in s⁻² at midpoints between adjacent NEMO T depths.

    eos80 expects potential temperature (ITS-90 °C) and Practical Salinity.
    teos10 expects Conservative Temperature (°C) and Absolute Salinity (g/kg).

    This fixed-z implementation is eager (GSW consumes NumPy arrays).
    """
    t = ds[temperature]
    s = ds[salinity]
    z = ds[depth]
    lat = ds[latitude]
    lon = ds[longitude]
    if z.ndim != 1:
        raise ValueError("Fixed-z input requires a 1D depth coordinate")
    zdim = z.dims[0]
    if zdim not in t.dims or zdim not in s.dims:
        raise ValueError("Temperature and salinity must share the depth dimension")
    if t.dims != s.dims:
        s = s.transpose(*t.dims)
    if t.shape != s.shape:
        raise ValueError("Temperature and salinity shapes must match")
    zz = np.asarray(z.values, dtype=float)
    if zz.size < 2 or not np.all(np.isfinite(zz)) or np.any(np.diff(zz) <= 0) or zz[0] < 0:
        raise ValueError("Depth must have at least two increasing nonnegative levels (m)")
    if eos not in ("eos80", "teos10"):
        raise ValueError("eos must be eos80 or teos10")
    # Explicit broadcast avoids an xarray/NumPy axis-order mismatch.
    other = [d for d in t.dims if d != zdim]
    T = t.transpose(*other, zdim)
    S = s.transpose(*other, zdim)
    T, S, LAT, LON, Z = xr.broadcast(T, S, lat, lon, z)
    T = T.transpose(*other, zdim)
    S = S.transpose(*other, zdim)
    LAT = LAT.transpose(*other, zdim)
    LON = LON.transpose(*other, zdim)
    Z = Z.transpose(*other, zdim)
    lat_values = np.asarray(LAT.values, dtype=float)
    p = gsw.p_from_z(-np.asarray(Z.values, dtype=float), lat_values)
    if eos == "eos80":
        SA = gsw.SA_from_SP(np.asarray(S.values), p, np.asarray(LON.values), lat_values)
        CT = gsw.CT_from_pt(SA, np.asarray(T.values))
    else:
        SA = np.asarray(S.values)
        CT = np.asarray(T.values)
    n2, _ = gsw.Nsquared(SA, CT, p, lat=lat_values, axis=-1)
    midpoint = 0.5 * (zz[:-1] + zz[1:])
    coords = {d: T.coords[d] for d in other if d in T.coords}
    coords["depth_mid"] = midpoint
    # Carry physical positions so identical grid indices cannot mask grid differences.
    for name, position in ((latitude, lat), (longitude, lon)):
        if name not in coords and all(d in other for d in position.dims):
            coords[name] = position
    result = xr.DataArray(
        n2, dims=(*other, "depth_mid"), coords=coords, name="N2",
        attrs={"units": "s-2", "long_name": "Brunt-Vaisala frequency squared"},
    )
    return result.transpose(*( "depth_mid" if d == zdim else d for d in t.dims))


def compare_n2(reference: xr.DataArray, prediction: xr.DataArray, epsilon: float = 1e-8) -> xr.Dataset:
    """Create paired masks and differences on *identical* coordinates.

    AI-only instability means AI is unstable where reference is stable, not
    necessarily that the AI is physically false.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be nonnegative and finite")
    for name in ("nav_lat", "nav_lon"):
        if (name in reference.coords) != (name in prediction.coords):
            raise ValueError(f"Both N2 inputs must carry the same {name} coordinate")
        if name in reference.coords:
            a = np.asarray(reference.coords[name].values)
            b = np.asarray(prediction.coords[name].values)
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-6, rtol=0, equal_nan=True):
                raise ValueError(f"Physical grids differ at {name}; regrid T/S before comparing N2")
    ref, ai = xr.align(reference, prediction, join="exact")
    if ref.dims != ai.dims:
        raise ValueError("N2 arrays must have identical dimension ordering; transpose first")
    if ref.shape != ai.shape:
        raise ValueError("N2 arrays must have identical shapes")
    valid = np.isfinite(ref) & np.isfinite(ai)
    ref_unstable = (ref < -epsilon) & valid
    ai_unstable = (ai < -epsilon) & valid
    ai_only = ai_unstable & (ref > epsilon) & valid
    return xr.Dataset(
        {
            "n2_reference": ref.where(valid),
            "n2_prediction": ai.where(valid),
            "n2_difference": (ai - ref).where(valid),
            "valid_pair": valid,
            "reference_unstable": ref_unstable,
            "prediction_unstable": ai_unstable,
            "ai_only_unstable": ai_only,
        },
        attrs={"epsilon_s-2": float(epsilon), "note": "AI-only means reference-stable disagreement, not proven physical impossibility."},
    )


def summarize_n2(comparison: xr.Dataset, epsilon: float | None = None) -> dict[str, float]:
    """Return *unweighted grid-cell* fractions on paired valid cells."""
    valid = comparison["valid_pair"]
    ref = comparison["n2_reference"]
    ai = comparison["n2_prediction"]
    if epsilon is None:
        epsilon = float(comparison.attrs["epsilon_s-2"])
    total = int(valid.sum().item())
    if total == 0:
        raise ValueError("No finite paired grid cells")
    stable_ref = valid & (ref > epsilon)
    n_stable = int(stable_ref.sum().item())
    sqerr = ((ai - ref) ** 2).where(valid)
    return {
        "valid_cell_count": total,
        "reference_unstable_fraction": float(comparison["reference_unstable"].sum().item() / total),
        "prediction_unstable_fraction": float(comparison["prediction_unstable"].sum().item() / total),
        "ai_only_fraction_of_stable_reference": float(comparison["ai_only_unstable"].sum().item() / n_stable) if n_stable else float("nan"),
        "prediction_mean_negative_n2_magnitude_s-2": float((-ai.clip(max=0)).where(valid).sum().item() / total),
        "n2_rmse_s-2": float(np.sqrt(sqerr.sum().item() / total)),
    }
