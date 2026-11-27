#!/usr/bin/env python3
"""Paired basin SSH PSDs, using ensemble-mean SSH for GLOENS.

Interpolate each global field once per lead using nearest native wet/dry
grid cells, with a distance limit. FFT only contiguous common wet segments.
Latitude-specific distances and common physical frequency bins.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.signal import periodogram, windows
from physical_stability.forecasts import forecast_store, open_cloudferro, ENDPOINT
from gloens_member_ssh_psd import detect_layout

REGIONS = {
    "Atlantic_Tropical": (-20,20,-70,20),
    "Atlantic_Midlatitude_N": (20,60,-70,20),
    "Atlantic_Midlatitude_S": (-40,-20,-70,20),
    "Pacific_Tropical": (-20,20,120,290),
    "Pacific_Midlatitude_N": (20,60,120,290),
    "Pacific_Midlatitude_S": (-40,-20,120,290),
    "Indian_Tropical": (-20,20,20,120),
    "Indian_Midlatitude_S": (-40,-20,20,120),
    "Arctic": (60,89,-180,180),
    "Southern": (-70,-40,-180,180),
}
MODELS = ("GLONET","GLOW","GLO12","GLOENS")
def xyz(lat, lon):
    a,b = np.deg2rad(lat),np.deg2rad(lon)
    return np.column_stack((np.cos(a).ravel()*np.cos(b).ravel(),
                            np.cos(a).ravel()*np.sin(b).ravel(),np.sin(a).ravel()))

def field(ds, lead, raw=False):
    if raw:
        from types import SimpleNamespace
        layout = detect_layout(ds, SimpleNamespace(**{"gloens_"+k:None for k in
            ("ssh_name","lat_name","lon_name","member_dim","time_dim")}))
        da = ds[layout["ssh"]].isel({layout["time"]:lead})
        da = da.mean(layout["member"],skipna=False)
        lat,lon = ds[layout["latitude"]],ds[layout["longitude"]]
    else:
        da = ds[next(k for k in ("zos","sossheig","ssh","sea_surface_height_above_geoid") if k in ds)].isel(time=lead)
        lat = ds[next(k for k in ("latitude","lat","nav_lat") if k in ds)]
        lon = ds[next(k for k in ("longitude","lon","nav_lon") if k in ds)]
    if lat.ndim == lon.ndim == 1:
        da = da.transpose(lat.dims[0],lon.dims[0])
        lo,la = np.meshgrid(lon.values,lat.values)
    else:
        da = da.transpose(*lat.dims)
        la,lo = lat.values,lon.values
    return np.asarray(da.values),la,lo

def targets(bounds, dx, dlat):
    s,n,w,e = bounds
    rows=[]
    for lat in np.arange(s+dlat/2,n,dlat):
        length=111.195*np.cos(np.deg2rad(lat))*(e-w)
        count=max(8,int(np.ceil(length/dx)))
        lon=w+(np.arange(count)+.5)*(e-w)/count
        rows.append((lat,lon,length/count))
    return rows

def spectra(values, rows, freq):
    result=[]
    offset=0
    for lat,lon,dx in rows:
        n=len(lon)
        a=values[:,offset:offset+n]; offset+=n
        wet=np.isfinite(a).all(axis=0)
        # Join segments crossing the coordinate seam in full circles.
        if n*dx >= 359.99*111.195*np.cos(np.deg2rad(lat)) and not wet.all():
            cut=int(np.flatnonzero(~wet)[0])
            a=np.roll(a,-cut,axis=1); wet=np.roll(wet,-cut)
        edges=np.flatnonzero(np.diff(np.r_[False,wet,False]))
        for start,end in edges.reshape(-1,2):
            size=end-start
            if size*dx<1000 or size<16: continue
            f,p=periodogram(a[:,start:end],fs=1/dx,axis=-1,
                window=windows.tukey(size,.5),detrend="linear",scaling="density")
            valid=(freq>=f[1])&(freq<=f[-1])
            out=np.full((len(MODELS),len(freq)),np.nan)
            for i in range(len(MODELS)):
                out[i,valid]=np.interp(freq[valid],f[1:],p[i,1:])
            result.append((out,size*dx))
    if not result: raise ValueError("No common wet segments >= 1000 km")
    a=np.stack([x[0] for x in result]); w=np.array([x[1] for x in result])[:,None,None]
    den=np.sum(np.isfinite(a)*w,axis=0)
    return np.divide(np.nansum(a*w,axis=0),den,out=np.full_like(den,np.nan),where=den>0),len(result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start",default="2024-01-03")
    p.add_argument("--end",default="2024-12-25")
    p.add_argument("--leads",nargs="+",type=int,default=[1])
    p.add_argument("--dx-km",type=float,default=5)
    p.add_argument("--dlat",type=float,default=1)
    p.add_argument("--output-dir",type=Path,default=Path("results/basin_ssh_mean_2024"))
    p.add_argument("--endpoint",default=ENDPOINT)
    a=p.parse_args()
    if min(a.leads)<1 or a.dx_km<=0 or a.dlat<=0: p.error("Invalid lead or spacing")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows={k:targets(b,a.dx_km,a.dlat) for k,b in REGIONS.items()}
    queries={k:xyz(np.concatenate([np.full(len(lon),lat) for lat,lon,dx in rr]),
                   np.concatenate([lon for lat,lon,dx in rr])) for k,rr in rows.items()}
    freq=np.geomspace(1/40000,1/(2*a.dx_km),180)
    cache={}
    failures=[]
    for date in pd.date_range(a.start,a.end,freq="7D"):
        init=date.strftime("%Y%m%d")
        gdate=date+timedelta(days=1)
        gi=gdate.strftime("%Y%m%d"); ge=(gdate+timedelta(days=27)).strftime("%Y%m%d")
        try:
            with ExitStack() as stack:
                stores={m:stack.enter_context(open_cloudferro(forecast_store(m,init),endpoint=a.endpoint,anonymous=True)) for m in MODELS[:-1]}
                stores["GLOENS"]=stack.enter_context(open_cloudferro(f"MOISICEEF/glo4-ens50_ng_1d-m_{gi}-{ge}_2DT-oce_fcst_R{gi}.zarr",endpoint=a.endpoint,anonymous=True))
                for lead in a.leads:
                    output=a.output_dir/f"{init}_lead{lead:02d}.csv"
                    if output.exists(): continue
                    try:
                        sampled={k:[] for k in rows}
                        for m in MODELS:
                            v,lat,lon=field(stores[m],lead-(m=="GLOENS"),m=="GLOENS")
                            if m not in cache:
                                valid=np.isfinite(lat)&np.isfinite(lon)
                                indices=np.flatnonzero(valid)
                                tree=cKDTree(xyz(lat[valid],lon[valid]))
                                maps={}
                                for k,q in queries.items():
                                    dist,idx=tree.query(q,workers=-1)
                                    maps[k]=(indices[idx],dist*6371<=50)
                                cache[m]=(lat.copy(),lon.copy(),maps)
                            oldlat,oldlon,maps=cache[m]
                            if not (np.array_equal(lat,oldlat,equal_nan=True) and np.array_equal(lon,oldlon,equal_nan=True)):
                                raise ValueError(f"{m}: native grid changed")
                            for k,(idx,near) in maps.items():
                                sampled[k].append(np.where(near,v.ravel()[idx],np.nan))
                        records=[]
                        for k,rr in rows.items():
                            try:
                                power,nseg=spectra(np.stack(sampled[k]),rr,freq)
                                for m,power_m in zip(MODELS,power):
                                    records.extend(dict(init=init,lead=lead,region=k,model=m,frequency_cpkm=f,psd=y,n_segments=nseg) for f,y in zip(freq,power_m) if np.isfinite(y))
                            except Exception as exc:
                                failures.append(dict(init=init,lead=lead,region=k,error=str(exc)))
                        if records:
                            tmp=output.with_suffix(".tmp")
                            pd.DataFrame(records).to_csv(tmp,index=False);tmp.replace(output)
                    except Exception as exc:
                        failures.append(dict(init=init,lead=lead,region="all",error=str(exc)))
                    print(init,lead,"finished",flush=True)
        except Exception as exc:
            failures.append(dict(init=init,lead="",region="all",error=str(exc)))
        if failures:
            pd.DataFrame(failures).to_csv(a.output_dir/"failures_latest_run.csv",index=False)
if __name__=="__main__":
    main()
