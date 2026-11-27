#!/usr/bin/env python3
"""Plot paired annual basin spectra from basin_ssh_mean_psd.py."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--lead",type=int,default=1)
    a=p.parse_args()
    paths=sorted(a.input_dir.glob(f"*_lead{a.lead:02d}.csv"))
    if not paths: p.error("No completed case files")
    df=pd.concat([pd.read_csv(x,dtype={"init":str}) for x in paths],ignore_index=True)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    colors={"GLONET":"#0072B2","GLOW":"#009E73","GLO12":"#D55E00","GLOENS":"#7B3294"}
    for region,g in df.groupby("region"):
        fig,ax=plt.subplots(figsize=(10,6))
        mean=g.groupby(["model","frequency_cpkm"]).psd.agg(["mean","count"]).reset_index()
        for m,c in colors.items():
            x=mean[mean.model==m].sort_values("frequency_cpkm")
            ax.loglog(1/x.frequency_cpkm,x["mean"],color=c,lw=2,label={"GLOW":"GLONET3","GLOENS":"GLOENS ensemble-mean SSH"}.get(m,m))
        for scale in (100,500,2000):
            ax.axvline(scale,color=".45",ls="--",lw=1)
            ax.text(scale,.02,f"{scale:,} km",rotation=90,transform=ax.get_xaxis_transform(),ha="right",va="bottom")
        ax.invert_xaxis();ax.grid(True,which="both",alpha=.2)
        ax.set(xlabel="Zonal wavelength (km)",ylabel="SSH PSD (m² km)",title=f"{region} | lead {a.lead} | {g.init.nunique()} initializations")
        fig.legend(*ax.get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,.09,1,1))
        fig.savefig(a.output_dir/f"{region}_lead{a.lead:02d}.png",dpi=200)
        plt.close(fig)
        mean.to_csv(a.output_dir/f"{region}_lead{a.lead:02d}_mean.csv",index=False)
if __name__=="__main__":
    main()
