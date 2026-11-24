#!/usr/bin/env python3
"""Plot deterministic SSH PSD and individual GLOENS member spectra.

Input: spectra.csv from scripts/gloens_member_ssh_psd.py.
GLOENS mean/spread are computed in PSD space, never from ensemble-mean SSH.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"GLONET": "#0072B2", "GLOW": "#009E73", "GLO12": "#D55E00",
          "GLOENS": "#7B3294"}


def add_region_inset(ax, bounds: tuple[float, float, float, float], name: str | None) -> None:
    """Show selected lon/lat rectangle on a small global locator map.

    Cartopy coastlines are used if installed; a graticule locator is the
    dependency-free fallback. Bounds are (south, north, west, east).
    """
    from matplotlib.patches import Rectangle
    south, north, west, east = bounds
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError(f"Invalid region bounds: {bounds}")
    rect_style = dict(facecolor="#E64B35", edgecolor="#8B1A1A",
                      alpha=0.5, linewidth=1.1, zorder=5)
    try:
        import cartopy.crs as ccrs
        inset = ax.inset_axes([0.025, 0.035, 0.19, 0.16],
                              projection=ccrs.PlateCarree())
        inset.set_global()
        inset.set_facecolor("#EAF3F8")
        inset.coastlines(resolution="110m", linewidth=0.45)
        inset.add_patch(Rectangle((west, south), east-west, north-south,
                                  transform=ccrs.PlateCarree(), **rect_style))
        inset.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)
    except ImportError:
        inset = ax.inset_axes([0.025, 0.035, 0.19, 0.16])
        inset.set_facecolor("#EAF3F8")
        inset.set_xlim(-180, 180)
        inset.set_ylim(-90, 90)
        inset.set_aspect("auto")
        inset.set_xticks([-180, -90, 0, 90, 180])
        inset.set_yticks([-60, 0, 60])
        inset.grid(linewidth=0.35, alpha=0.5)
        inset.add_patch(Rectangle((west, south), east-west, north-south,
                                  **rect_style))
        inset.tick_params(labelsize=5, length=1)
    inset.set_title(name or "Study region", fontsize=8, pad=2)
    inset.set_zorder(10)


def plot_group(frame: pd.DataFrame, output: Path, title: str, bounds=None, region_name=None) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    deterministic = ["GLONET", "GLOW", "GLO12"]
    for model in deterministic:
        part = frame.loc[frame.model == model].sort_values("frequency_cpkm")
        if part.empty:
            continue
        if part.frequency_cpkm.duplicated().any():
            raise ValueError(f"Duplicate frequency bins for {model}; filter to one initialization and lead")
        f = part.frequency_cpkm.to_numpy(float)
        y = part.psd.to_numpy(float)
        good = (f > 0) & np.isfinite(y) & (y > 0)
        ax.loglog(1 / f[good], y[good], label=model, color=COLORS[model], linewidth=2)

    ensemble = frame.loc[frame.model == "GLOENS"].copy()
    if not ensemble.empty:
        if ensemble.member_label.isna().any():
            raise ValueError("GLOENS member_label missing; individual members are required")
        member_spectra = ensemble.pivot(
            index="frequency_cpkm", columns="member_label", values="psd"
        ).sort_index()
        freq = member_spectra.index.to_numpy(float)
        positive = freq > 0
        member_spectra = member_spectra.iloc[positive]
        freq = freq[positive]
        wavelength = 1 / freq
        values = member_spectra.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite GLOENS PSD values; inspect spectra.csv")
        for i, member in enumerate(member_spectra.columns):
            ax.loglog(wavelength, values[:, i], color=COLORS["GLOENS"],
                      alpha=0.18, linewidth=0.8,
                      label="GLOENS individual members" if i == 0 else None)
        mean = values.mean(axis=1)
        std = values.std(axis=1, ddof=1) if values.shape[1] > 1 else np.zeros_like(mean)
        lower = np.maximum(mean - std, np.finfo(float).tiny)
        upper = mean + std
        ax.fill_between(wavelength, lower, upper, color=COLORS["GLOENS"],
                        alpha=0.24, label="GLOENS mean ± 1 SD")
        ax.loglog(wavelength, mean, color=COLORS["GLOENS"], linewidth=1.3,
                  label=f"GLOENS mean PSD (n={values.shape[1]})")

    for scale in (100, 500, 2000):
        ax.axvline(scale, color="0.45", linestyle="--", linewidth=1)
        ax.text(scale, 0.02, f"{scale:,} km", rotation=90,
                transform=ax.get_xaxis_transform(), ha="right", va="bottom",
                color="0.4", fontsize=9)
    ax.set_xlabel("Zonal wavelength (km)")
    ax.set_ylabel("SSH PSD (m² km)")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    if bounds is not None:
        add_region_inset(ax, bounds, region_name)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spectra", type=Path,
                   default=Path("results/gloens_ssh_test/spectra.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/gloens_ssh_test/figures"))
    p.add_argument("--init", help="Optional initialization YYYYMMDD")
    p.add_argument("--lead", type=int, help="Optional forecast lead")
    p.add_argument("--lat-min", type=float)
    p.add_argument("--lat-max", type=float)
    p.add_argument("--lon-min", type=float)
    p.add_argument("--lon-max", type=float)
    p.add_argument("--region-name", help="Label for the locator map")
    a = p.parse_args()
    coords = (a.lat_min, a.lat_max, a.lon_min, a.lon_max)
    if any(x is not None for x in coords) and not all(x is not None for x in coords):
        p.error("Specify all four bounds: --lat-min, --lat-max, --lon-min, --lon-max")
    bounds = coords if all(x is not None for x in coords) else None

    df = pd.read_csv(a.spectra, dtype={"init": str, "model": str,
                                      "member_label": str})
    required = {"init", "valid_date", "lead", "model", "member_label",
                "frequency_cpkm", "psd"}
    missing = required - set(df.columns)
    if missing:
        p.error(f"Missing columns: {sorted(missing)}")
    if a.init is not None:
        df = df.loc[df.init == a.init]
    if a.lead is not None:
        df = df.loc[df.lead == a.lead]
    if df.empty:
        p.error("No spectra match the requested initialization and lead")
    for (init, lead, valid_date), group in df.groupby(
            ["init", "lead", "valid_date"], sort=True):
        plot_group(group, a.output_dir / f"ssh_psd_{init}_lead{int(lead):02d}.png",
                   f"SSH zonal PSD | init {init} | valid {valid_date} | lead {int(lead)}",
                   bounds=bounds, region_name=a.region_name)


if __name__ == "__main__":
    main()
