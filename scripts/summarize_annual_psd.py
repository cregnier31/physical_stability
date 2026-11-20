#!/usr/bin/env python3
"""Plot annual lead-dependent PSDs, ratios and seasonal spectra from annual_psd.py."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bootstrap_mean_ci(values, block=14, nboot=300, seed=42):
    """Moving-block bootstrap across initialization dates, preserving date order."""
    a = np.asarray(values, float)
    if a.ndim != 2:
        raise ValueError("Expected init x frequency array")
    n = len(a)
    if n == 0:
        raise ValueError("No initialization dates")
    rng = np.random.default_rng(seed)
    length = min(block, n)
    reps = []
    for _ in range(nboot):
        indices = []
        while len(indices) < n:
            start = rng.integers(0, n - length + 1)
            indices.extend(range(start, start + length))
        sample = a[np.asarray(indices[:n])]
        count = np.isfinite(sample).sum(axis=0)
        reps.append(np.divide(np.nansum(sample, axis=0), count,
                              out=np.full(a.shape[1], np.nan), where=count > 0))
    return np.nanpercentile(reps, [2.5, 97.5], axis=0)


def plot(ds, output, reference, leads, block, nboot):
    output.mkdir(parents=True, exist_ok=True)
    freq = ds.freq_lon.values
    good = freq > 0
    wavelength = 1 / freq[good]
    for lead in leads:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model in ds.model.values:
            arr = ds.psd.sel(model=model, lead=lead).values
            mean = np.nanmean(arr, axis=0)[good]
            ci = bootstrap_mean_ci(arr, block=block, nboot=nboot)
            line, = ax.loglog(wavelength, mean, label=str(model))
            ax.fill_between(wavelength, ci[0, good], ci[1, good],
                            color=line.get_color(), alpha=.15)
        ax.set(xlabel="Zonal wavelength (km)", ylabel="PSD (field units squared km)",
               title=f"Annual mean PSD | lead index {lead}")
        ax.invert_xaxis(); ax.grid(alpha=.2, which="both"); ax.legend()
        fig.savefig(output / f"mean_psd_lead_{lead}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        if reference in ds.model.values:
            fig, ax = plt.subplots(figsize=(9, 5))
            ref = np.nanmean(ds.psd.sel(model=reference, lead=lead).values, axis=0)[good]
            for model in ds.model.values:
                if model == reference:
                    continue
                numerator = np.nanmean(ds.psd.sel(model=model, lead=lead).values,
                                       axis=0)[good]
                ratio = np.divide(numerator, ref, out=np.full_like(ref, np.nan),
                                  where=ref > 0)
                ax.semilogx(wavelength, ratio, label=f"{model}/{reference}")
            ax.axhline(1, color="0.4", linestyle="--")
            ax.set(xlabel="Zonal wavelength (km)", ylabel="Mean PSD ratio",
                   title=f"Spectral ratio | lead index {lead}")
            ax.invert_xaxis(); ax.grid(alpha=.2); ax.legend()
            fig.savefig(output / f"ratio_lead_{lead}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
    for lead in leads:
        for season, months in {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
                               "JJA": [6, 7, 8], "SON": [9, 10, 11]}.items():
            selected = ds.sel(init=ds.init.dt.month.isin(months))
            if not selected.sizes["init"]:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            for model in ds.model.values:
                a = selected.psd.sel(model=model, lead=lead).values
                if not np.isfinite(a).any():
                    continue
                ax.loglog(wavelength, np.nanmean(a, axis=0)[good], label=str(model))
            ax.set(xlabel="Zonal wavelength (km)", ylabel="PSD (field units squared km)",
                   title=f"{season} PSD | lead index {lead}")
            ax.invert_xaxis(); ax.grid(alpha=.2, which="both"); ax.legend()
            fig.savefig(output / f"season_{season}_lead_{lead}.png",
                        dpi=160, bbox_inches="tight")
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/annual_plots"))
    p.add_argument("--reference", default="GLO12")
    p.add_argument("--leads", nargs="+", type=int)
    p.add_argument("--bootstrap-block", type=int, default=14)
    p.add_argument("--bootstrap-reps", type=int, default=300)
    args = p.parse_args()
    if args.bootstrap_block < 1 or args.bootstrap_reps < 10:
        p.error("Bootstrap block must be positive and reps >= 10")
    with xr.open_dataset(args.input) as ds:
        leads = args.leads if args.leads is not None else list(ds.lead.values)
        plot(ds, args.output_dir, args.reference, leads,
             args.bootstrap_block, args.bootstrap_reps)


if __name__ == "__main__":
    main()
