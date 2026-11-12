"""Static wet masks for model-native stratification diagnostics.

The actual private files must be inspected on the compute node. Mask names
are detected conservatively; no all-ocean fallback is allowed.
"""
from pathlib import Path
import numpy as np
import xarray as xr

STATIC_DIR = Path("/data/rd_exchange/cregnier/static")
MASK_PATHS = {
    "GLOW": STATIC_DIR / "mask_GLOW.nc",
    "GLONET": STATIC_DIR / "mask_GLONET.nc",
    "GLO12": STATIC_DIR / "mask_GL012.nc",
    "GLOENS": STATIC_DIR / "iORCA025_meshmask_v3.1_binary.nc",
}
MASK_CANDIDATES = ("tmask", "mask", "mask_thetao", "wet_mask", "wet", "tmaskutil")


def wet_mask_for_ts(
    path: str | Path,
    ts: xr.DataArray,
    mask_variable: str | None = None,
    horizontal_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> xr.DataArray:
    """Return a boolean mask on canonical (deptht,y,x) T points.

    2D masks broadcast through the water column; 3D T masks are preferred.
    A grid or vertical mismatch fails closed. This routine does not infer
    bottom bathymetry from 2D wet masks.
    """
    with xr.open_dataset(path) as grid:
        name = mask_variable or next((v for v in MASK_CANDIDATES if v in grid), None)
        if name is None or name not in grid:
            raise KeyError(
                f"No recognized T-point mask in {path}; available: {list(grid.data_vars)}. "
                "Specify --mask-variable-SYSTEM explicitly."
            )
        mask = grid[name].squeeze(drop=True)
        if mask.ndim not in (2, 3):
            raise ValueError(f"{name} has {mask.ndim} dimensions; expected 2D/3D T mask")
        target = (ts.sizes["deptht"], ts.sizes["y"], ts.sizes["x"])
        if horizontal_indices is not None:
            yi, xi = horizontal_indices
            mask = mask.isel({mask.dims[-2]: np.asarray(yi), mask.dims[-1]: np.asarray(xi)})
        if mask.ndim == 2 and mask.shape != target[1:]:
            raise ValueError(f"{name}: horizontal shape {mask.shape} != {target[1:]}")
        if mask.ndim == 3 and mask.shape != target:
            raise ValueError(f"{name}: 3D shape {mask.shape} != {target}")
        data = np.asarray(mask.values)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{name} contains nonfinite values")
        if not np.all((data == 0) | (data == 1)):
            raise ValueError(f"{name} must be a binary 0/1 wet mask")
        if mask.ndim == 2:
            data = np.broadcast_to(data, target)
        return xr.DataArray(
            data.astype(bool), dims=("deptht", "y", "x"),
            coords={d: ts.coords[d] for d in ("deptht", "y", "x")},
            name="wet_t",
            attrs={"mask_source": str(path), "mask_variable": name},
        )


def mask_ts_pairs(ds: xr.Dataset, wet: xr.DataArray) -> xr.Dataset:
    """Apply T wet mask before density; invalid interfaces stay missing."""
    return ds.assign(
        thetao=ds["thetao"].where(wet),
        so=ds["so"].where(wet),
    )
