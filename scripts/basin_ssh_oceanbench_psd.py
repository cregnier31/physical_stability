#!/usr/bin/env python3
"""OceanBench-style zonal PSD, without spatial interpolation.

Regular grids use native longitude rows. Curvilinear GLOENS uses native
x rows as an approximation, only where selected rows are contiguous and
approximately zonal. Zero-fill missing data, mean physical spacing,
equal row weights, and OceanBench xrft settings.
"""
import argparse
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import xrft
from basin_ssh_mean_psd import REGIONS as ORIGINAL, MODELS, field, xyz, forecast_store, open_cloudferro, ENDPOINT

REGIONS={k:b for k,b in ORIGINAL.items() if k!="Arctic"}
REGIONS["Southern"]=(-60,-40,-180,180)

def spectrum(v,lat,lon,bounds):
    s,n,w,e=bounds
    rel=(lon-w)%360
    inside=(lat>=s)&(lat<n)&(rel<e-w)&np.isfinite(lat)&np.isfinite(lon)
    rows=[]; spacings=[]
    for j in np.flatnonzero(inside.any(axis=1)):
        ids=np.flatnonzero(inside[j])
        if len(ids)<16: continue
        # Longitude ordering must preserve native cyclic adjacency.
        ids=ids[np.argsort(rel[j,ids])]
        if not np.all(np.diff(ids)%v.shape[1]==1): continue
        pos=xyz(lat[j,ids],lon[j,ids])
        dx=12742*np.arcsin(np.clip(np.linalg.norm(np.diff(pos,axis=0),axis=1)/2,0,1))
        if not np.all(np.isfinite(dx)&(dx>0)): continue
        if np.any(111.195*np.abs(np.diff(lat[j,ids]))>.5*dx): continue
        if np.max(np.abs(dx/np.mean(dx)-1))>.1: continue
        rows.append(np.nan_to_num(v[j,ids],nan=0.,posinf=0.,neginf=0.))
        spacings.append(dx)
    if not rows: raise ValueError("No qualifying native rows")
    # A rectangular FFT needs equal row lengths. Retain modal length,
    # rather than pad or resample native curvilinear rows.
    lengths,counts=np.unique([len(r) for r in rows],return_counts=True)
    length=lengths[np.argmax(counts)]
    selected=[i for i,r in enumerate(rows) if len(r)==length]
    dx=float(np.mean(np.concatenate([spacings[i] for i in selected])))
    da=xr.DataArray(np.stack([rows[i] for i in selected]).astype("float32"),
        dims=("lat","lon"),coords={"lon":np.arange(length)*dx*1000})
    ps=xrft.power_spectrum(da,dim=["lon"],detrend="linear",window="tukey",
        window_correction=True,true_amplitude=True,truncate=True)
    ps=ps.mean("lat",skipna=True)
    ps=ps.sel(freq_lon=ps.freq_lon>0).compute()
    return ps.freq_lon.values*1000,ps.values/1000,len(selected),dx

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start",default="2024-01-03")
    p.add_argument("--end",default="2024-12-25")
    p.add_argument("--output-dir",type=Path,default=Path("results/basin_ssh_oceanbench_2024"))
    p.add_argument("--endpoint",default=ENDPOINT)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for date in pd.date_range(a.start,a.end,freq="7D"):
        init=date.strftime("%Y%m%d")
        output=a.output_dir/f"{init}_lead01.csv"
        if output.exists(): continue
        gi=(date+timedelta(days=1)).strftime("%Y%m%d")
        ge=(date+timedelta(days=28)).strftime("%Y%m%d")
        records=[]
        try:
            with ExitStack() as stack:
                stores={m:stack.enter_context(open_cloudferro(forecast_store(m,init),endpoint=a.endpoint,anonymous=True)) for m in MODELS[:-1]}
                stores["GLOENS"]=stack.enter_context(open_cloudferro(f"MOISICEEF/glo4-ens50_ng_1d-m_{gi}-{ge}_2DT-oce_fcst_R{gi}.zarr",endpoint=a.endpoint,anonymous=True))
                for m in MODELS:
                    v,lat,lon=field(stores[m],int(m!="GLOENS"),m=="GLOENS")
                    for k,b in REGIONS.items():
                        f,power,count,dx=spectrum(v,lat,lon,b)
                        records.extend(dict(init=init,lead=1,region=k,model=m,frequency_cpkm=x,psd=y,n_segments=count,dx_km=dx) for x,y in zip(f,power) if np.isfinite(y))
                        print(init,m,k,count,dx,flush=True)
            tmp=output.with_suffix(".tmp")
            pd.DataFrame(records).to_csv(tmp,index=False);tmp.replace(output)
        except Exception as exc:
            print(init,"FAILED",repr(exc),flush=True)
            with (a.output_dir/"failures.log").open("a") as log:
                print(init,repr(exc),file=log)
if __name__=="__main__":
    main()
