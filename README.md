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
