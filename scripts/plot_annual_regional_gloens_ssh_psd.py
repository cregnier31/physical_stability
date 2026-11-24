#!/usr/bin/env python3
"""Plot annual mean member-resolved SSH spectra for completed regional jobs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from plot_gloens_member_ssh_psd import plot_group
from run_regional_gloens_ssh_psd import BANDS, SECTORS

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lead", type=int, default=1)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--n-members", type=int, default=50)
    a = p.parse_args()
    for path in sorted(a.input_dir.glob("*/spectra.csv")):
        region = path.parent.name
        try:
            df = pd.read_csv(path, dtype={"init": str, "member_label": str})
            df = df[(df.lead == a.lead) & df.init.str.startswith(str(a.year))].copy()
            df["member_label"] = df.member_label.fillna("")
            if df.empty:
                continue
            expected = {"GLONET", "GLOW", "GLO12", "GLOENS"}
            good = []
            for init, g in df.groupby("init"):
                if set(g.model) != expected:
                    continue
                if g.loc[g.model == "GLOENS", "member_label"].nunique() != a.n_members:
                    continue
                if g.duplicated(["model", "member_label", "frequency_cpkm"]).any():
                    continue
                if not np.isfinite(g.psd).all():
                    continue
                bins = [tuple(sorted(x.frequency_cpkm)) for _, x in
                        g.groupby(["model", "member_label"])]
                if len(set(bins)) != 1:
                    continue
                good.append(init)
            df = df[df.init.isin(good)]
            if df.empty:
                print(f"{region}: no complete paired cases", flush=True)
                continue
            # Require identical member labels and frequency bins across dates.
            keys = ["model", "member_label", "frequency_cpkm"]
            sizes = df.groupby(keys).init.nunique()
            if not (sizes == len(good)).all():
                raise ValueError("Member labels or frequency grid change across dates")
            mean = df.groupby(keys, as_index=False).psd.mean()
            bounds = None
            if region == "Atlantic_Tropical_open_ocean":
                bounds = (-20, 20, -30, -15)
            else:
                sector, remainder = region.split("_", 1)
                band, index = remainder.rsplit("_sector", 1)
                south, north = BANDS[band]
                west, east = SECTORS[sector][int(index)-1]
                bounds = (south, north, west, east)
            output = a.output_dir / f"{region}_{a.year}_lead{a.lead:02d}.png"
            plot_group(mean, output,
                       f"SSH zonal PSD | {a.year} | lead {a.lead} | n={len(good)} initializations",
                       bounds=bounds, region_name=region)
            print(f"{region}: {len(good)} paired initializations", flush=True)
        except Exception as exc:
            print(f"{region}: skipped: {exc}", flush=True)

if __name__ == "__main__":
    main()
