#!/usr/bin/env python3
"""Plot lead-dependent annual global velocity diagnostics from annual_summary.csv."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"GLOW": "GLONET3"}
UNITS = {"kinetic_energy": "m² s⁻²", "relative_vorticity": "s⁻¹",
         "horizontal_divergence": "s⁻¹", "strain": "s⁻¹",
         "okubo_weiss": "s⁻²", "rossby_number": "1"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("results/annual_global_velocity/annual_summary.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/annual_global_velocity/figures"))
    args = p.parse_args()
    df = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in UNITS:
        subset = df[df.metric == metric]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for model, group in subset.groupby("model", sort=True):
            group = group.sort_values("lead")
            ax.plot(group.lead, group.mean_spatial_rms, marker="o",
                    label=LABELS.get(model, model))
        ax.set_title(f"Annual global {metric.replace('_', ' ')} — spatial RMS")
        ax.set_xlabel("Forecast lead index (not necessarily elapsed days)")
        ax.set_ylabel(f"Area-weighted spatial RMS ({UNITS[metric]})")
        ax.grid(alpha=.25)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=min(4, len(labels)), loc="lower center",
                   bbox_to_anchor=(0.5, 0.015), frameon=False)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        fig.savefig(args.output_dir / f"annual_{metric}_rms.png", dpi=180)
        plt.close(fig)
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
