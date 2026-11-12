# Physical stability diagnostics for NEMO and AI ocean models

A reproducible toolkit for comparing static stratification in NEMO reference simulations and AI-generated temperature/salinity predictions. It computes the Brunt–Väisälä frequency squared (N²) with TEOS-10 and diagnoses candidate spurious density inversions.

## Installation

```bash
pip install -e '.[test]'
```

## Quick start

```bash
physical-stability compare --reference nemo.nc --prediction ai.nc --output results.nc --eos eos80 --temperature thetao --salinity so --depth deptht --latitude nav_lat --longitude nav_lon
```

The example above assumes NEMO's EOS-80 convention: potential temperature (°C) and Practical Salinity (PSS-78), both at T points. Use `--eos teos10` **only** if input temperature is Conservative Temperature and salinity is Absolute Salinity (g/kg). Inspect your NEMO `nn_eos`, namelist, and output metadata before selecting a convention.

Input fields should have a common time and horizontal grid, one-dimensional positive-down, increasing depth at T centers, and latitude/longitude on the horizontal grid. The program uses TEOS-10 GSW `Nsquared`, which yields values between adjacent T levels. It is an independent diagnostic, not an exact reproduction of NEMO's native `bn2` discretization. For z-star or moving vertical coordinates, use the actual instantaneous T-point depths; the current CLI supports fixed 1D z levels only.

Outputs include `n2_reference`, `n2_prediction`, `n2_difference`, `reference_unstable`, `prediction_unstable`, and `ai_only_unstable` at `depth_mid`, plus unweighted paired-grid-cell summary metrics. The AI-only mask identifies *disagreement* with the reference, not proof that the AI is nonphysical: real convection can cause negative N², and the reference may be wrong. Evaluate against observations and forcing before drawing physical conclusions.

## Interpretation

With a positive-up z coordinate, stable stratification corresponds to N² > 0. Weakly negative near-neutral estimates should be treated with a chosen tolerance, supplied through `--epsilon` (default `1e-8 s^-2`, illustrative only). Keep a consistent finite-data mask when comparing the two models. The metrics are grid-cell based, not volume weighted; for rigorous multi-depth averages use NEMO cell metrics and representative volumes, and avoid treating vertically averaged fields as instantaneous stability.

## Development

```bash
pytest -q
```

## Scope and limitations

- TEOS-10 conversions and squared buoyancy frequency use GSW.
- Nonlinear EOS and cabbeling make temperature or salinity gradients alone insufficient to infer stability.
- A positive N² everywhere is not sufficient for correct internal-wave physics: compare pycnocline strength/depth and vertical N² profiles too.
- Avoid applying the EOS-80 conversion twice to TEOS-10 output.
- Outputs may be memory-intensive for large full-domain grids; chunk processing will be needed for production-scale runs.


## Multi-system CloudFerro comparison

This workflow follows the remote-store conventions in the
[eddy-tracking-intercomparison code](https://github.com/cregnier31/eddy-tracking-intercomparison/tree/feature/gloens-ensemble-zos-clean):
`eddy_detect/zarr_forecasts.py` (GLONET/GLOW/GLO12 paths) and
`eddy_load/gloens_mean.py` (GLOENS anonymous S3 access, 50 ensemble members
and curvilinear 2D geographic coordinates).

**Important:** These original scripts process sea-surface height (`zos`).
N² is *not* calculated from SSH: the stability tool requires a store with
3D temperature and salinity on T-point z levels. Availability and names of
those fields in the live Zarr stores have **not** been verified; the reader
will raise a descriptive exception if they are missing.

Install optional CloudFerro/Zarr dependencies:

```bash
pip install -e '.[cloud,test]'
```

To analyze the first lead day of GLONET, GLOW and GLO12 in the North Atlantic:

```bash
physical-stability-multisystem \
  --init-date 20260101 \
  --systems GLONET GLOW GLO12 \
  --first-day 0 --last-day 1 \
  --lon-min -80 --lon-max 0 --lat-min 20 --lat-max 65 \
  --eos eos80 \
  --output results
```

For all four systems, **supply a verified GLOENS store containing 3D T/S**
rather than the 2DT SSH store:

```bash
physical-stability-multisystem \
  --init-date 20260101 \
  --systems GLONET GLOW GLO12 GLOENS \
  --gloens-store 's3://YOUR_BUCKET/YOUR_VERIFIED_3D_TS_FORECAST.zarr' \
  --first-day 0 --last-day 3 \
  --eos eos80 \
  --gloens-ensemble mean \
  --output results
```

The CloudFerro default endpoint is `https://s3.waw3-1.cloudferro.com`.
Use `--authenticated` if the store requires configured IAM/environment
credentials. The code subselects longitude/latitude and one lead day before
eager GSW calculations. Forecast lead indices are 0-based, and `--last-day`
is exclusive. A single day is the safe default for initial validation.

### Model-specific field and equation-of-state mapping

If stores use different variable names or EOS conventions, provide a JSON
configuration file:

```json
{
  "GLONET": {"temperature": "thetao", "salinity": "so", "depth": "depth",
             "eos": "eos80"},
  "GLOW": {"temperature": "thetao", "salinity": "so", "depth": "depth",
           "eos": "eos80"},
  "GLO12": {"temperature": "thetao", "salinity": "so", "depth": "depth",
             "eos": "eos80"},
  "GLOENS": {"temperature": "thetao", "salinity": "so", "depth": "depth",
              "latitude": "latitude", "longitude": "longitude",
              "eos": "eos80"}
}
```

Pass `--config field_mapping.json`. The values above are **examples to
verify**, not claims about the live Zarr metadata. When the data already
contain Conservative Temperature and Absolute Salinity, set
`eos: "teos10"`; otherwise use `eos80` only when the actual fields are
potential temperature and Practical Salinity. A wrong convention may
generate false stability conclusions.

### Outputs and interpretation

- `results/<SYSTEM>/n2_<SYSTEM>_<INIT>_leadNNN.nc`: N² and native-grid
  geographic coordinates at vertical midlevels, with model provenance.
- `results/n2_multisystem_summary.csv`: per-system, per-lead count and
  unweighted unstable fraction, mean N², and mean negative N² magnitude.
- `results/pointwise_comparisons.json`: whether pointwise comparisons
  against GLO12 were possible and paired-grid-cell metrics where they were.
- `results/<SYSTEM>/versus_GLO12_leadNNN.nc`: paired differences and masks,
  **only** if times, dimensions, and physical grids are identical.

No implicit cross-grid interpolation is done. GLOENS has 2D geographic
coordinates and typically a different grid; GLO12 pointwise comparisons
are therefore likely to be marked `skipped` until a validated T/S
remapping stage is added. Remap **temperature and salinity before computing
N²**, not two precomputed N² fields. Different native-grid unweighted
fractions should not be interpreted as volume-weighted performance scores.

`--gloens-ensemble mean` computes **N²(mean(T), mean(S))**, respecting
paired finite T/S ensemble members; this does not equal the ensemble
mean of member-wise N² because seawater density is nonlinear.
`--gloens-ensemble members` retains all 50 member N² fields; its overall
summary pools valid member grid cells. It does not do a one-to-one comparison
of ensemble members against the single GLO12 field.

The classification threshold `--epsilon` defaults to
`1e-8 s^-2` only as an illustrative tolerance. Negative N² is a
potential static instability, not automatic proof of AI failure.
Additional physical evaluation should compare with observations, surface
forcing, mixing and the pycnocline's vertical structure.


### Precomputed 28-day GLOENS CLASS4 ensemble means (preferred)

The local GLOENS data can be read from
`/sortref/calval/CLASS4_scores/` with filenames such as
`glo4-ens50_ensmean_thetao_leadtime_000_20241205.npy` through
`glo4-ens50_ensmean_thetao_leadtime_027_20241205.npy`.
Matching `so` is required; `uo` and `vo` are read together when both
are available, but are not used to compute N².

**Important:** NumPy .npy files do not include geolocation, depth
coordinates, units or axis labels. Supply a trusted NetCDF grid file with
1D `deptht` in metres and 2D `nav_lat` and `nav_lon` on (y,x).
Verify the actual array shape and dimension order before using the tool.
For example, if each file is ordered (depth,y,x):

```bash
physical-stability-multisystem \
  --init-date 20241205 \
  --systems GLONET GLOW GLO12 GLOENS \
  --gloens-npy-dir /sortref/calval/CLASS4_scores \
  --gloens-grid-file /path/to/verified_gloens_grid.nc \
  --gloens-axes depth,y,x \
  --first-day 0 --last-day 28 \
  --eos eos80 --output results
```

This analyzes every available GLOENS lead from 000 to 027. Shorter
GLONET/GLOW/GLO12 forecasts stop at their actual Zarr time length
without truncating GLOENS. GLOW is expected to have 9 days; the others
are checked against their actual store metadata rather than assuming 10.
Pointwise comparisons are attempted only for leads available in both
models, matching valid times, and identical geographic grids. The
GLOENS mean fields are already averaged over 50 members, so no ensemble
reduction is performed again.

The loader assumes one daily lead increment after the initialization
date. Confirm this convention from the CLASS4 production workflow before
interpreting cross-system valid-time comparisons. This code has not
verified the contents or array dimensions of the private local directory.
