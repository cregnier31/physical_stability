#!/usr/bin/env python3
"""Plot basin/latitude-band PSD output from annual_basin_psd.py.

Uses per_case.csv (individual initialization) or annual_summary.csv
(mean over initializations). No ensemble-member spread is inferred.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"GLONET": "#0072B2", "GLOW": "#009E73",
          "GLO12": "#D55E00", "GLOENS": "#7B3294"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("results/annual_basin_psd_test"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--source", choices=["per_case", "annual_summary"], default="per_case")
    p.add_argument("--init", help="YYYYMMDD; only valid for per_case")
    p.add_argument("--lead", type=int)
    p.add_argument("--basin")
    p.add_argument("--band")
    p.add_argument("--variable", choices=["sst", "ssh"], default="sst")
    a = p.parse_args()
    if a.init and a.source != "per_case":
        p.error("--init requires --source per_case")
    path = a.input_dir / (a.source + ".csv")
    if not path.exists():
        p.error(f"Missing {path}; annual_basin_psd.py must finish to write this file")
    df = pd.read_csv(path, dtype={"init": str, "model": str})
    ycol = "psd" if a.source == "per_case" else "annual_mean_psd"
    required = {"model", "lead", "basin", "band", "frequency_cpkm", ycol}
    if a.source == "per_case":
        required.add("init")
    missing = required - set(df.columns)
    if missing:
        p.error(f"Missing columns: {sorted(missing)}")
    for column, value in (("init", a.init), ("lead", a.lead),
                          ("basin", a.basin), ("band", a.band)):
        if value is not None:
            df = df.loc[df[column] == value]
    if df.empty:
        p.error("No matching rows")
    outdir = a.output_dir or (a.input_dir / "figures")
    grouping = ["init", "lead", "basin", "band"] if a.source == "per_case" else [
        "lead", "basin", "band"]
    for keys, subset in df.groupby(grouping, sort=True):
        meta = dict(zip(grouping, keys))
        fig, ax = plt.subplots(figsize=(10, 6))
        for model in ("GLONET", "GLOW", "GLO12", "GLOENS"):
            part = subset.loc[subset.model == model].sort_values("frequency_cpkm")
            if part.empty:
                continue
            f = part.frequency_cpkm.to_numpy(float)
            y = part[ycol].to_numpy(float)
            valid = (f > 0) & np.isfinite(y) & (y > 0)
            if not valid.any():
                continue
            ax.loglog(1/f[valid], y[valid], color=COLORS[model],
                      label=model, linewidth=2)
            if a.source == "annual_summary" and "annual_std_psd" in part:
                sd = part.annual_std_psd.to_numpy(float)
                good = valid & np.isfinite(sd)
                if good.any():
                    ax.fill_between(1/f[good],
                        np.maximum(y[good]-sd[good], np.finfo(float).tiny),
                        y[good]+sd[good], color=COLORS[model], alpha=0.12)
        for scale in (500, 2000):
            ax.axvline(scale, linestyle="--", color="0.5", linewidth=0.9)
        ax.invert_xaxis()
        ax.set_xlabel("Zonal wavelength (km)")
        ax.set_ylabel(f"{a.variable.upper()} PSD ({'m² km' if a.variable == 'ssh' else '°C² km'})")
        case = f"init {meta['init']} | " if "init" in meta else "mean across initializations | "
        ax.set_title(f"{a.variable.upper()} zonal PSD | {case}lead {meta['lead']}\n"
                     f"{meta['basin']} / {meta['band']}")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        stem = "_".join(str(meta[k]) for k in grouping)
        output = outdir / f"{a.variable}_psd_{a.source}_{stem}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
