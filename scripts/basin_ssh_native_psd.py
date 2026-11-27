#!/usr/bin/env python3
"""Native-grid SSH spectra. No spatial resampling.

Regular grids: zonal sections. Curvilinear grid: native x sections,
restricted to approximately east-west, nearly uniformly spaced segments.
Spectral bin integration uses native Fourier bin overlap.
"""
import argparse
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import periodogram, windows
from basin_ssh_mean_psd import REGIONS, MODELS, field, xyz, forecast_store, open_cloudferro, ENDPOINT

def sections(lat,lon,bounds):
    s,n,w,e=bounds
    # Rotate each native row at the geographical western boundary.
    # Preserve native adjacency; never sort individual curvilinear points.
    for j in range(lat.shape[0]):
        la,lo=lat[j],lon[j]
        rel=(lo-w)%360
        inside=np.isfinite(la)&np.isfinite(lo)&(la>=s)&(la<n)&(rel<e-w)
        if not inside.any(): continue
        # Native global grids are cyclic in x; check actual seam distance.
        pos=xyz(la,lo)
        dist=6371*2*np.arcsin(np.clip(np.linalg.norm(np.diff(pos,axis=0),axis=1)/2,0,1))
        typical=np.nanmedian(dist[dist>0])
        seam=6371*2*np.arcsin(np.clip(np.linalg.norm(pos[0]-pos[-1])/2,0,1))
        cyclic=np.isfinite(seam) and seam>0 and seam<1.5*typical
        order=np.arange(len(la))
        if cyclic:
            cut=np.nanargmin(rel)
            order=np.roll(order,-cut)
        la,lo,inside=la[order],lo[order],inside[order]
        pos=pos[order]
        dx=6371*2*np.arcsin(np.clip(np.linalg.norm(np.diff(pos,axis=0),axis=1)/2,0,1))
        north=111.195*np.abs(np.diff(la))
        # Reject strongly meridional edges, grid folds and large spacing jumps.
        good=np.isfinite(dx)&(dx>0)&(north<=0.5*dx)
        yield j,order,inside,dx,good

def native_spectrum(v, geometry, edges):
    numerator=np.zeros(len(edges)-1)
    denominator=np.zeros_like(numerator)
    count=0
    for j,order,inside,dx,good in geometry:
        row=v[j,order]
        wet=inside&np.isfinite(row)
        starts=np.flatnonzero(wet & np.r_[True,~wet[:-1]|~good])
        ends=np.flatnonzero(wet & np.r_[~wet[1:]|~good,True])+1
        for a,b in zip(starts,ends):
            if b-a<16: continue
            spacing=dx[a:b-1]
            d=float(np.mean(spacing))
            if (b-a)*d<1000 or np.max(np.abs(spacing/d-1))>.05: continue
            f,p=periodogram(row[a:b],fs=1/d,window=windows.tukey(b-a,.5),
                            detrend="linear",scaling="density")
            df=f[1]-f[0]
            # Exclude DC and the terminal Fourier bin (Nyquist edge).
            left=f[1:-1]-df/2; right=f[1:-1]+df/2
            if not len(left): continue
            # Cumulative integral of piecewise constant native spectral density.
            cumulative=np.r_[0,np.cumsum(p[1:-1]*df)]
            boundaries=np.r_[left[0],right]
            integ=np.interp(edges,boundaries,cumulative)
            overlap=np.maximum(0,np.minimum(edges[1:],right[-1])-np.maximum(edges[:-1],left[0]))
            full=overlap>=np.diff(edges)*(1-1e-8)
            weight=(b-a)*d
            numerator[full]+=np.diff(integ)[full]/np.diff(edges)[full]*weight
            denominator[full]+=weight
            count+=1
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0),count

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start",default="2024-01-03")
    p.add_argument("--end",default="2024-12-25")
    p.add_argument("--output-dir",type=Path,default=Path("results/basin_ssh_native_2024"))
    p.add_argument("--endpoint",default=ENDPOINT)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    edges=np.geomspace(1/40000,1,181)
    freq=np.sqrt(edges[:-1]*edges[1:])
    cache={}
    for date in pd.date_range(a.start,a.end,freq="7D"):
        init=date.strftime("%Y%m%d")
        output=a.output_dir/f"{init}_lead01.csv"
        if output.exists(): continue
        gi=(date+timedelta(days=1)).strftime("%Y%m%d")
        ge=(date+timedelta(days=28)).strftime("%Y%m%d")
        try:
            with ExitStack() as stack:
                stores={m:stack.enter_context(open_cloudferro(forecast_store(m,init),endpoint=a.endpoint,anonymous=True)) for m in MODELS[:-1]}
                stores["GLOENS"]=stack.enter_context(open_cloudferro(f"MOISICEEF/glo4-ens50_ng_1d-m_{gi}-{ge}_2DT-oce_fcst_R{gi}.zarr",endpoint=a.endpoint,anonymous=True))
                records=[]
                for m in MODELS:
                    v,lat,lon=field(stores[m],int(m!="GLOENS"),m=="GLOENS")
                    if m not in cache:
                        cache[m]=(lat.copy(),lon.copy(),{k:list(sections(lat,lon,b)) for k,b in REGIONS.items()})
                    oldlat,oldlon,geometry=cache[m]
                    if not (np.array_equal(lat,oldlat,equal_nan=True) and np.array_equal(lon,oldlon,equal_nan=True)):
                        raise ValueError(f"{m}: native grid changed")
                    for k,g in geometry.items():
                        power,count=native_spectrum(v,g,edges)
                        if not np.isfinite(power).any():
                            print(init,m,k,"NO ACCEPTED SPECTRAL BINS",flush=True)
                        records.extend(dict(init=init,lead=1,region=k,model=m,frequency_cpkm=f,psd=y,n_segments=count) for f,y in zip(freq,power) if np.isfinite(y))
                if not records: raise ValueError("No spectra")
                tmp=output.with_suffix(".tmp")
                pd.DataFrame(records).to_csv(tmp,index=False);tmp.replace(output)
                print(init,"finished",flush=True)
        except Exception as exc:
            print(init,"FAILED",repr(exc),flush=True)
            with (a.output_dir/"failures.log").open("a") as log:
                print(init,repr(exc),file=log)
if __name__=="__main__":
    main()
