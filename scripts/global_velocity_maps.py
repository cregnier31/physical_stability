#!/usr/bin/env python3
"""Global, selected-depth ocean velocity maps with native-grid operators.

GLOENS requires explicit confirmation that saved uo/vo retain native NEMO U/V
positions. The NEMO curl and divergence use finite-volume metrics; polar-fold
and open-boundary rows are masked. Regular grids use spherical derivatives.
This is not a global 3D annual EKE calculation.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physical_stability.forecasts import ENDPOINT, forecast_store, open_cloudferro
from physical_stability.gloens_npy import load_gloens_mean, DEFAULT_GLOENS_GRID

R = 6371000.
OMEGA = 7.2921159e-5
FIELDS = ("kinetic_energy", "relative_vorticity", "horizontal_divergence",
          "strain", "okubo_weiss", "rossby_number")
UNITS = dict(kinetic_energy="m² s⁻²", relative_vorticity="s⁻¹",
             horizontal_divergence="s⁻¹", strain="s⁻¹",
             okubo_weiss="s⁻²", rossby_number="1")
SIGNED = {"relative_vorticity", "horizontal_divergence", "okubo_weiss", "rossby_number"}


def _finite_neighbors(a):
    ok = np.isfinite(a)
    # Explicitly exclude open boundaries and points adjacent to missing values.
    out = ok.copy()
    out[1:-1, 1:-1] &= (ok[:-2, 1:-1] & ok[2:, 1:-1] &
                          ok[1:-1, :-2] & ok[1:-1, 2:])
    out[[0, -1], :] = False
    out[:, [0, -1]] = False
    return out


def regular_diagnostics(u, v, lat, lon):
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Regular model requires 1D lat/lon")
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise ValueError("Coordinates must increase monotonically")
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    cosphi = np.cos(phi)[:, None]
    if np.any(np.abs(lat) >= 89.5):
        raise ValueError("Regular-grid poles require a specialized operator; exclude |lat| >= 89.5")
    # Covariant spherical formulas for east/north physical velocity components.
    dudlam = np.gradient(u, lam, axis=1, edge_order=2)
    dvdlam = np.gradient(v, lam, axis=1, edge_order=2)
    dudphi = np.gradient(u, phi, axis=0, edge_order=2)
    dvdphi = np.gradient(v, phi, axis=0, edge_order=2)
    ux = dudlam / (R * cosphi)
    vx = dvdlam / (R * cosphi)
    uy = dudphi / R
    vy = dvdphi / R
    curvature = np.tan(phi)[:, None] / R
    zeta = vx - uy + u * curvature
    divergence = ux + vy - v * curvature
    normal = ux - vy + v * curvature
    shear = vx + uy - u * curvature
    strain = np.hypot(normal, shear)
    valid = _finite_neighbors(u) & _finite_neighbors(v)
    f = 2 * OMEGA * np.sin(phi)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rossby = np.where(np.abs(lat)[:, None] >= 5, zeta / f, np.nan)
    return {k: np.where(valid, a, np.nan) for k, a in {
        "kinetic_energy": .5 * (u*u + v*v), "relative_vorticity": zeta,
        "horizontal_divergence": divergence, "strain": strain,
        "okubo_weiss": strain*strain - zeta*zeta, "rossby_number": rossby}.items()}


def nemo_diagnostics(u, v, grid, max_north_lat=80.):
    """Metric-aware C-grid strain/OW at T points, excluding polar-fold region.

    Face-normal gradients are computed on native U/V points using NEMO
    horizontal scale factors. F-point shear and curl are averaged to T
    before forming Okubo-Weiss. These are local grid-axis derivatives:
    they do not include curvilinear basis-rotation/Christoffel terms.
    Accordingly the result is a metric-aware approximation, not an exact
    covariant strain tensor on the rotated ORCA grid.
    """
    def g(name):
        return np.asarray(grid[name].squeeze(drop=True).values, float)
    e1u, e2u, e1v, e2v = (g(n) for n in ("e1u", "e2u", "e1v", "e2v"))
    e1t, e2t, e1f, e2f = (g(n) for n in ("e1t", "e2t", "e1f", "e2f"))
    lat_t, lat_f = g("gphit"), g("gphif")
    if any(a.shape != u.shape for a in
           (v, e1u, e2u, e1v, e2v, e1t, e2t, e1f, e2f, lat_t, lat_f)):
        raise ValueError("NEMO metric/velocity shapes differ")
    with np.errstate(divide="ignore", invalid="ignore"):
        ut = .5 * (u[1:, 1:] + u[1:, :-1])
        vt = .5 * (v[1:, 1:] + v[:-1, 1:])
        ke = np.full_like(u, np.nan)
        ke[1:, 1:] = .5 * (ut**2 + vt**2)

        du_dx = np.full_like(u, np.nan)
        dv_dy = np.full_like(u, np.nan)
        du_dx[1:, 1:] = (u[1:, 1:] - u[1:, :-1]) / e1t[1:, 1:]
        dv_dy[1:, 1:] = (v[1:, 1:] - v[:-1, 1:]) / e2t[1:, 1:]
        normal = du_dx - dv_dy

        div = np.full_like(u, np.nan)
        div[1:, 1:] = ((e2u[1:, 1:] * u[1:, 1:] -
                        e2u[1:, :-1] * u[1:, :-1]) +
                       (e1v[1:, 1:] * v[1:, 1:] -
                        e1v[:-1, 1:] * v[:-1, 1:])) / (e1t[1:, 1:] * e2t[1:, 1:])

        dvdx_f = (v[:-1, 1:] - v[:-1, :-1]) / e1f[:-1, :-1]
        dudy_f = (u[1:, :-1] - u[:-1, :-1]) / e2f[:-1, :-1]
        shear_f = np.full_like(u, np.nan)
        shear_f[:-1, :-1] = dvdx_f + dudy_f
        curl = np.full_like(u, np.nan)
        curl[:-1, :-1] = ((e2v[:-1, 1:] * v[:-1, 1:] -
                           e2v[:-1, :-1] * v[:-1, :-1]) -
                          (e1u[1:, :-1] * u[1:, :-1] -
                           e1u[:-1, :-1] * u[:-1, :-1])) / (e1f[:-1, :-1] * e2f[:-1, :-1])

        # F(i,j) lies northeast of T(i,j) in this indexing convention.
        # Average four adjacent F values onto each interior T location.
        shear_t = np.full_like(u, np.nan)
        curl_t = np.full_like(u, np.nan)
        shear_t[1:-1, 1:-1] = .25 * (
            shear_f[:-2, :-2] + shear_f[1:-1, :-2] +
            shear_f[:-2, 1:-1] + shear_f[1:-1, 1:-1])
        curl_t[1:-1, 1:-1] = .25 * (
            curl[:-2, :-2] + curl[1:-1, :-2] +
            curl[:-2, 1:-1] + curl[1:-1, 1:-1])
        strain = np.hypot(normal, shear_t)
        ow = strain**2 - curl_t**2
        f = 2 * OMEGA * np.sin(np.deg2rad(lat_f))
        rossby = np.where(np.abs(lat_f) >= 5., curl / f, np.nan)

    out = {"kinetic_energy": ke, "horizontal_divergence": div,
           "relative_vorticity": curl, "strain": strain,
           "okubo_weiss": ow, "rossby_number": rossby}
    for key, a in out.items():
        lat = lat_f if key in ("relative_vorticity", "rossby_number") else lat_t
        a[~np.isfinite(lat) | (lat > max_north_lat)] = np.nan
        a[[0, -1], :] = np.nan
        a[:, [0, -1]] = np.nan
        if key in ("strain", "okubo_weiss"):
            a[:2, :] = np.nan
            a[-2:, :] = np.nan
            a[:, :2] = np.nan
            a[:, -2:] = np.nan
    return out


def _regular_field(ds, lead, depth_index, max_lat):
    latname = next((n for n in ("latitude", "lat") if n in ds.coords), None)
    lonname = next((n for n in ("longitude", "lon") if n in ds.coords), None)
    zname = next((n for n in ("depth", "deptht", "deptho", "z", "lev") if n in ds.coords), None)
    if not all((latname, lonname, zname)):
        raise ValueError("Missing regular-grid coordinates")
    lat = np.asarray(ds[latname].values, float)
    lon = np.asarray(ds[lonname].values, float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Expected regular-grid coordinates")
    yi = np.flatnonzero(np.isfinite(lat) & ((lat <= max_lat) & (lat > -88.)))
    if yi.size < 3:
        raise ValueError("Insufficient latitude coverage")
    xi = np.argsort((lon + 180) % 360 - 180)
    lon_sorted = ((lon[xi] + 180) % 360) - 180
    if np.any(np.diff(lon_sorted) <= 0):
        raise ValueError("Duplicate longitudes after wrapping")
    if lat[yi[0]] > lat[yi[-1]]:
        yi = yi[::-1]
    def get(names):
        name = next((n for n in names if n in ds), None)
        if name is None:
            raise KeyError(f"Missing {names}")
        da = ds[name]
        if "time" in da.dims:
            da = da.isel(time=lead)
        if set(da.dims) != {zname, latname, lonname}:
            raise ValueError(f"{name}: expected collocated regular U/V grid, got {da.dims}")
        return np.asarray(da.isel({zname: depth_index, latname: yi, lonname: xi})
                          .transpose(latname, lonname).values, float)
    u = get(("uo", "vozocrtx", "u"))
    v = get(("vo", "vomecrty", "v"))
    depth = float(ds[zname].values[depth_index])
    return regular_diagnostics(u, v, lat[yi], lon_sorted), lat[yi], lon_sorted, depth


def _gloens_field(args, lead, init):
    ds = load_gloens_mean(args.gloens_npy_dir, args.gloens_grid_file, init, lead,
                          include_currents=True)
    try:
        if not {"uo", "vo"} <= set(ds.data_vars):
            raise ValueError("GLOENS uo/vo files are required")
        u = np.asarray(ds.uo.isel(time=0, depth=args.depth_index).values, float)
        v = np.asarray(ds.vo.isel(time=0, depth=args.depth_index).values, float)
        depth = float(ds.depth.values[args.depth_index])
    finally:
        ds.close()
    with xr.open_dataset(args.gloens_grid_file) as grid:
        result = nemo_diagnostics(u, v, grid, max_north_lat=args.max_latitude)
        lat_t = np.asarray(grid.gphit.squeeze(drop=True).values)
        lon_t = np.asarray(grid.glamt.squeeze(drop=True).values)
        lat_f = np.asarray(grid.gphif.squeeze(drop=True).values)
        lon_f = np.asarray(grid.glamf.squeeze(drop=True).values)
    coords = {k: ((lat_f, lon_f) if k in ("relative_vorticity", "rossby_number") else (lat_t, lon_t))
              for k in result}
    return result, coords, depth


def plot(results, depths, output, init, lead, depth_index, max_plot_points, dpi):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    output.mkdir(parents=True, exist_ok=True)
    order = ("GLONET", "GLOW", "GLO12", "GLOENS")
    for field in FIELDS:
        available = [(m, *results[m][field]) for m in order
                     if m in results and field in results[m]]
        if not available:
            continue
        allvalid = [a[np.isfinite(a)] for _, a, _, _ in available
                    if np.isfinite(a).any()]
        if not allvalid:
            continue
        sample = np.concatenate([a[::max(1, len(a)//100000)] for a in allvalid])
        if field in SIGNED:
            limit = float(np.nanpercentile(np.abs(sample), 98))
            vmin, vmax = -max(limit, 1e-16), max(limit, 1e-16)
        else:
            vmin, vmax = 0., max(float(np.nanpercentile(sample, 98)), 1e-16)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                                 subplot_kw={"projection": ccrs.EckertIV()})
        last_artist = None
        for ax, model in zip(axes.flat, order):
            ax.set_global()
            ax.add_feature(cfeature.LAND, facecolor="#b9b9b9",
                           edgecolor="#555555", linewidth=.3, zorder=4)
            ax.coastlines(resolution="110m", linewidth=.4, zorder=5)
            import matplotlib.ticker as mticker
            from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=.45, color=".45", alpha=.55,
                              linestyle="--")
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
            gl.ylocator = mticker.FixedLocator(np.arange(-80, 81, 20))
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 8, "color": ".3"}
            gl.ylabel_style = {"size": 8, "color": ".3"}
            display_name = "GLONET3" if model == "GLOW" else model
            ax.set_title(f"{display_name}  |  {depths[model]:.1f} m" if model in depths else display_name,
                         fontsize=13, weight="bold", pad=8)
            if model not in results or field not in results[model]:
                reason = ("Not requested" if model not in results else
                          "Diagnostic not available on this grid")
                ax.text(.5, .5, reason, transform=ax.transAxes,
                        ha="center", va="center", fontsize=10,
                        bbox=dict(facecolor="white", alpha=.9, edgecolor="none"),
                        zorder=8)
                continue
            data, lat, lon = results[model][field]
            if np.ndim(lat) == 1:
                llon, llat = np.meshgrid(lon, lat)
            else:
                llon, llat = lon, lat
            stride = max(1, int(np.ceil(np.sqrt(data.size / max_plot_points))))
            sl = (slice(None, None, stride), slice(None, None, stride))
            valid = (np.isfinite(data[sl]) & np.isfinite(llat[sl]) &
                     np.isfinite(llon[sl]))
            if not np.any(valid):
                continue
            last_artist = ax.scatter(
                llon[sl][valid], llat[sl][valid], c=data[sl][valid],
                s=1.1 if stride == 1 else 2.2, marker="s", linewidths=0,
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r" if field in SIGNED else "viridis",
                vmin=vmin, vmax=vmax, rasterized=True)
        fig.suptitle(f"Global {field.replace('_',' ').title()} | {init} | lead {lead} | "
                     "near-surface velocity" if depth_index == 0 else f"depth index {depth_index}", fontsize=16, weight="bold", y=.98)
        fig.subplots_adjust(left=.035, right=.965, top=.91, bottom=.15,
                            wspace=.06, hspace=.12)
        if last_artist is not None:
            cb_ax = fig.add_axes([.27, .075, .46, .025])
            fig.colorbar(last_artist, cax=cb_ax, orientation="horizontal",
                         label=f"{field.replace('_', ' ')} ({UNITS[field]})")
        path = output / f"global_{field}_depth{depth_index}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", nargs="+", default=["GLONET", "GLOW", "GLO12", "GLOENS"],
                   choices=["GLONET", "GLOW", "GLO12", "GLOENS"])
    p.add_argument("--init", required=True, help="Regular model initialization YYYYMMDD")
    p.add_argument("--lead", type=int, default=1)
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--max-latitude", type=float, default=80.)
    p.add_argument("--gloens-init", help="GLOENS initialization YYYYMMDD")
    p.add_argument("--gloens-lead", type=int)
    p.add_argument("--gloens-npy-dir", default="/sortref/calval/CLASS4_scores")
    p.add_argument("--gloens-grid-file", default=DEFAULT_GLOENS_GRID)
    p.add_argument("--gloens-native-uv-confirmed", action="store_true",
                   help="Explicitly attest that GLOENS uo/vo are native staggered U/V face velocities")
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--authenticated", action="store_true")
    p.add_argument("--max-plot-points", type=int, default=500000)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--output-dir", type=Path, default=Path("results/global_velocity"))
    args = p.parse_args()
    if args.lead < 0 or args.depth_index < 0 or not 5 < args.max_latitude <= 80. or args.max_plot_points < 1000:
        p.error("Invalid lead, depth, latitude or plot point count")
    if "GLOENS" in args.systems and not args.gloens_native_uv_confirmed:
        p.error("GLOENS needs --gloens-native-uv-confirmed after verifying uo/vo grid positions")
    results = {}
    depths = {}
    for model in args.systems:
        if model == "GLOENS":
            gi = args.gloens_init or args.init
            gl = args.gloens_lead if args.gloens_lead is not None else args.lead
            if datetime.strptime(gi, "%Y%m%d") + timedelta(days=gl) != datetime.strptime(args.init, "%Y%m%d") + timedelta(days=args.lead):
                raise ValueError("GLOENS valid date differs from regular models")
            data, coords, depth = _gloens_field(args, gl, gi)
            results[model] = {k: (v, *coords[k]) for k, v in data.items()}
        else:
            with open_cloudferro(forecast_store(model, args.init), endpoint=args.endpoint,
                                 anonymous=not args.authenticated) as ds:
                data, lat, lon, depth = _regular_field(ds, args.lead, args.depth_index,
                                                       args.max_latitude)
                results[model] = {k: (v, lat, lon) for k, v in data.items()}
        depths[model] = depth
        print(f"Computed {model} at depth {depth:.1f} m", flush=True)
    plot(results, depths, args.output_dir / "maps" / f"lead{args.lead:02d}",
         args.init, args.lead, args.depth_index, args.max_plot_points, args.dpi)


if __name__ == "__main__":
    main()
