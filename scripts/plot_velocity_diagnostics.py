#!/usr/bin/env python3
"""Publication-style PNG plots for collocated 3D velocity diagnostics.

Eckert IV is a global equal-area projection; regional domains may appear small.
Land is Natural Earth grey; Cartopy may download coastline data on first use.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

MAP_FIELDS = (
    "kinetic_energy", "speed", "relative_vorticity", "rossby_number",
    "horizontal_divergence", "strain", "okubo_weiss",
    "vertical_shear_squared", "gradient_richardson_number",
    "ageostrophic_u", "ageostrophic_v", "ageostrophic_ke",
)
SIGNED = {"relative_vorticity", "rossby_number", "horizontal_divergence",
          "okubo_weiss", "ageostrophic_u", "ageostrophic_v"}
CMAP = "RdBu_r"


def _limits(arrays, signed):
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays
                             if np.isfinite(a).any()]) if any(np.isfinite(a).any() for a in arrays) else np.array([])
    if not finite.size:
        return None
    if signed:
        v = float(np.nanpercentile(np.abs(finite), 98))
        return (-v if v else -1, v if v else 1)
    lo, hi = np.nanpercentile(finite, [2, 98])
    return (float(lo), float(hi)) if hi > lo else (float(lo), float(lo + 1))


def plot_maps(datasets, output, depth_indices=(0,), dpi=180):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    output.mkdir(parents=True, exist_ok=True)
    models = list(datasets)
    for field in MAP_FIELDS:
        available = [m for m in models if field in datasets[m]]
        if not available:
            continue
        depth_based = "depth" in datasets[available[0]][field].dims
        for iz in (depth_indices if depth_based else (None,)):
            arrays = []
            panels = []
            for model in available:
                da = datasets[model][field]
                if depth_based:
                    if iz >= da.sizes["depth"]:
                        continue
                    da = da.isel(depth=iz)
                if set(da.dims) != {"latitude", "longitude"}:
                    continue
                arrays.append(np.asarray(da.values, float))
                panels.append((model, da))
            if not panels:
                continue
            lim = _limits(arrays, field in SIGNED)
            if lim is None:
                continue
            fig = plt.figure(figsize=(6.3 * len(panels), 4.9))
            axes = []
            for j, (model, da) in enumerate(panels):
                ax = fig.add_subplot(1, len(panels), j + 1, projection=ccrs.EckertIV())
                ax.set_global()
                ax.add_feature(cfeature.LAND, facecolor="#b9b9b9",
                               edgecolor="#666666", linewidth=.35, zorder=5)
                ax.coastlines(resolution="110m", linewidth=.4, zorder=6)
                ax.gridlines(linewidth=.3, color="0.5", alpha=.45)
                opts = dict(transform=ccrs.PlateCarree(), shading="auto",
                            cmap=CMAP if field in SIGNED else "viridis",
                            vmin=lim[0], vmax=lim[1], rasterized=True)
                mesh = ax.pcolormesh(da.longitude.values, da.latitude.values,
                                     da.values, **opts)
                ax.set_title(model, fontsize=12, weight="bold")
                axes.append(ax)
            unit = panels[0][1].attrs.get("units", "")
            label = field.replace("_", " ").title()
            fig.suptitle(label + (f" | depth index {iz}" if depth_based else ""),
                         fontsize=15, weight="bold")
            fig.colorbar(mesh, ax=axes, orientation="horizontal",
                         fraction=.06, pad=.08, label=f"{label} [{unit}]")
            suffix = f"_depth{iz}" if depth_based else ""
            fig.savefig(output / f"map_{field}{suffix}.png", dpi=dpi,
                        bbox_inches="tight", facecolor="white")
            plt.close(fig)


def plot_profiles(datasets, output, dpi=180):
    output.mkdir(parents=True, exist_ok=True)
    for field in ("mean_ke_profile", "vertical_shear_squared",
                  "gradient_richardson_number", "kinetic_energy",
                  "relative_vorticity", "strain"):
        fig, ax = plt.subplots(figsize=(6.5, 7))
        plotted = False
        for model, ds in datasets.items():
            if field not in ds or "depth" not in ds[field].dims:
                continue
            da = ds[field]
            reduce = [d for d in da.dims if d != "depth"]
            profile = da.mean(reduce, skipna=True) if reduce else da
            ax.plot(profile.values, profile.depth.values, label=model, lw=2)
            plotted = True
        if plotted:
            ax.invert_yaxis()
            ax.set(xlabel=field.replace("_", " "), ylabel="Depth (m)",
                   title=f"Vertical profile: {field.replace('_', ' ')}")
            ax.grid(alpha=.25); ax.legend()
            fig.savefig(output / f"profile_{field}.png", dpi=dpi,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_spectra(datasets, output, dpi=180):
    output.mkdir(parents=True, exist_ok=True)
    if not any("ke_psd" in ds for ds in datasets.values()):
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = False
    for model, ds in datasets.items():
        if "ke_psd" not in ds:
            continue
        da = ds.ke_psd
        dim = next((d for d in da.dims if d != "depth"), None)
        if dim is None:
            continue
        freq = np.asarray(da[dim].values, float)
        power = np.asarray(da.isel(depth=0).values, float)
        good = (freq > 0) & (power > 0) & np.isfinite(power)
        if good.any():
            ax.loglog(1/freq[good], power[good], label=model, lw=2)
            plotted = True
    if plotted:
        ax.invert_xaxis()
        ax.set(xlabel="Zonal wavelength (km)", ylabel="KE variance PSD",
               title="Near-surface horizontal KE spectrum")
        ax.grid(alpha=.25, which="both"); ax.legend()
        fig.savefig(output / "ke_psd_surface_comparison.png", dpi=dpi,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for model, ds in datasets.items():
        if "ke_psd" not in ds:
            continue
        da = ds.ke_psd
        dim = next((d for d in da.dims if d != "depth"), None)
        if dim is None:
            continue
        freq = np.asarray(da[dim].values, float)
        valid = freq > 0
        power = np.asarray(da.values, float)[:, valid]
        with np.errstate(divide="ignore", invalid="ignore"):
            logpower = np.where(power > 0, np.log10(power), np.nan)
        if not np.isfinite(logpower).any():
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        mesh = ax.pcolormesh(1/freq[valid], da.depth.values, logpower,
                             shading="auto", cmap="viridis", rasterized=True)
        ax.set_xscale("log"); ax.invert_xaxis(); ax.invert_yaxis()
        ax.set(xlabel="Zonal wavelength (km)", ylabel="Depth (m)",
               title=f"{model}: depth-dependent KE variance PSD")
        fig.colorbar(mesh, ax=ax, label="log10(KE variance PSD)")
        fig.savefig(output / f"ke_psd_depth_{model.lower()}.png", dpi=dpi,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)


def render(files, output, depths, dpi=180):
    datasets = {}
    try:
        for path in files:
            ds = xr.open_dataset(path)
            name = str(ds.attrs.get("model", path.stem.split("_")[0])).upper()
            if name in datasets:
                ds.close()
                raise ValueError(f"Duplicate model {name}")
            datasets[name] = ds
        plot_maps(datasets, output / "maps", depths, dpi)
        plot_profiles(datasets, output / "profiles", dpi)
        plot_spectra(datasets, output / "spectra", dpi)
    finally:
        for ds in datasets.values():
            ds.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path, help="3D diagnostic NetCDF files")
    p.add_argument("--output-dir", type=Path, default=Path("results/velocity/plots"))
    p.add_argument("--depth-indices", nargs="+", type=int, default=[0, 5, 10])
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()
    if any(i < 0 for i in args.depth_indices):
        p.error("Depth indices must be nonnegative")
    render(args.files, args.output_dir, args.depth_indices, args.dpi)


if __name__ == "__main__":
    main()
