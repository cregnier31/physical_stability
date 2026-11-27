#!/usr/bin/env python3
"""Compute GLOENS member PSDs, then plot against existing paired spectra."""
import argparse
from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import pandas as pd
from basin_ssh_oceanbench_psd import REGIONS, geometry, spectrum_rows, open_cloudferro, ENDPOINT
from gloens_member_ssh_psd import detect_layout

def compute(a):
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for date in pd.date_range(a.start,a.end,freq="7D"):
        init=date.strftime("%Y%m%d")
        out=a.output_dir/f"{init}_lead01.csv"
        if out.exists(): continue
        gi=(date+timedelta(days=1)).strftime("%Y%m%d")
        ge=(date+timedelta(days=28)).strftime("%Y%m%d")
        try:
            with open_cloudferro(f"MOISICEEF/glo4-ens50_ng_1d-m_{gi}-{ge}_2DT-oce_fcst_R{gi}.zarr",endpoint=a.endpoint,anonymous=True) as ds:
                layout=detect_layout(ds,SimpleNamespace(**{"gloens_"+k:None for k in ("ssh_name","lat_name","lon_name","member_dim","time_dim")}))
                md=layout["member"]
                da=ds[layout["ssh"]].isel({layout["time"]:0})
                lat,lon=ds[layout["latitude"]],ds[layout["longitude"]]
                if lat.ndim==lon.ndim==1:
                    dims=(lat.dims[0],lon.dims[0])
                    lo,la=np.meshgrid(lon.values,lat.values)
                else:
                    dims=lat.dims
                    la,lo=lat.values,lon.values
                if da.sizes[md]!=50: raise ValueError("Expected 50 members")
                labels=np.asarray(da[md].values).astype(str)
                if len(set(labels))!=50: raise ValueError("Duplicate member labels")
                geo={k:geometry(la,lo,b) for k,b in REGIONS.items()}
                records=[]
                for i,label in enumerate(labels):
                    v=da.isel({md:i}).transpose(*dims).values
                    for k,(rows,dx) in geo.items():
                        f,p,n,d=spectrum_rows(v,rows,dx)
                        if not np.isfinite(p).all(): raise ValueError("Nonfinite PSD")
                        records.extend(dict(init=init,lead=1,region=k,model="GLOENS",member_label=label,frequency_cpkm=x,psd=y) for x,y in zip(f,p))
                    print(init,"member",i+1,"/ 50",flush=True)
            tmp=out.with_suffix(".tmp")
            pd.DataFrame(records).to_csv(tmp,index=False);tmp.replace(out)
        except Exception as exc:
            print(init,"FAILED",repr(exc),flush=True)
            with (a.output_dir/"failures.log").open("a") as log: print(init,repr(exc),file=log)

def plot(a):
    from plot_gloens_member_ssh_psd import add_region_inset, COLORS
    import matplotlib.pyplot as plt
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pairs=[]
    for path in sorted(a.member_dir.glob("*_lead01.csv")):
        other=a.mean_dir/path.name
        if not other.exists(): continue
        m=pd.read_csv(path,dtype={"init":str,"member_label":str})
        d=pd.read_csv(other,dtype={"init":str})
        if set(m.region)!=set(REGIONS) or set(d.region)!=set(REGIONS): raise ValueError(f"Incomplete regions: {path}")
        pairs.append((m,d))
    if not pairs: raise ValueError("No paired dates")
    for region,b in REGIONS.items():
        members=[]; systems=[]
        for m,d in pairs:
            m=m[m.region==region]
            d=d[d.region==region]
            mp=m.pivot(index="frequency_cpkm",columns="member_label",values="psd").sort_index()
            if mp.shape[1]!=50 or not np.isfinite(mp.values).all(): raise ValueError("Incomplete members")
            dp={k:g.set_index("frequency_cpkm").psd.sort_index() for k,g in d.groupby("model")}
            if set(dp)!=set(COLORS): raise ValueError("Incomplete systems")
            if not np.array_equal(mp.index,dp["GLOENS"].index): raise ValueError("GLOENS frequency mismatch")
            if members:
                if not mp.index.equals(members[0].index) or not mp.columns.equals(members[0].columns): raise ValueError("Member grid or labels changed")
                if any(not dp[k].index.equals(systems[0][k].index) for k in dp): raise ValueError("System frequency grid changed")
            members.append(mp);systems.append(dp)
        vals=np.mean([m.values for m in members],axis=0)
        f=members[0].index.values
        fig,ax=plt.subplots(figsize=(10.5,6.5))
        for k,c in COLORS.items():
            x=systems[0][k].index.values
            y=np.mean([d[k].values for d in systems],axis=0)
            ax.loglog(1/x,y,color=c,lw=2,ls="--" if k=="GLOENS" else "-",label={"GLOW":"GLONET3","GLOENS":"GLOENS ensemble-mean SSH PSD"}.get(k,k))
        c=COLORS["GLOENS"]
        for i in range(50):
            ax.loglog(1/f,vals[:,i],color=c,alpha=.18,lw=.7,label="GLOENS individual members" if i==0 else None)
        mu=vals.mean(axis=1);sd=vals.std(axis=1,ddof=1)
        ax.fill_between(1/f,np.maximum(mu-sd,np.finfo(float).tiny),mu+sd,color=c,alpha=.22,label="GLOENS mean ± 1 SD")
        ax.loglog(1/f,mu,color=c,lw=2,label="GLOENS mean PSD (50 members)")
        for scale in (100,500,2000):
            ax.axvline(scale,color=".45",ls="--",lw=1)
            ax.text(scale,.02,f"{scale:,} km",rotation=90,transform=ax.get_xaxis_transform(),ha="right",va="bottom")
        ax.invert_xaxis();ax.grid(True,which="both",alpha=.2)
        ax.set(xlabel="Zonal wavelength (km)",ylabel="SSH PSD (m² km)",title=f"{region} | lead 1 | {len(pairs)} initializations")
        add_region_inset(ax,b,region.replace("_"," "))
        fig.legend(*ax.get_legend_handles_labels(),loc="lower center",ncol=3,fontsize=8,frameon=False)
        fig.tight_layout(rect=(0,.13,1,1))
        fig.savefig(a.output_dir/f"{region}_lead01.png",dpi=200);plt.close(fig)
        pd.DataFrame({"frequency_cpkm":f,"mean_psd":mu,"sd_psd":sd}).to_csv(a.output_dir/f"{region}_members_summary.csv",index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    c=sub.add_parser("compute")
    c.add_argument("--start",default="2024-01-03")
    c.add_argument("--end",default="2024-12-25")
    c.add_argument("--endpoint",default=ENDPOINT)
    c.add_argument("--output-dir",type=Path,required=True)
    q=sub.add_parser("plot")
    for name in ("member-dir","mean-dir","output-dir"): q.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    (compute if a.mode=="compute" else plot)(a)
if __name__=="__main__": main()
