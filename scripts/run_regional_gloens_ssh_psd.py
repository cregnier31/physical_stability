#!/usr/bin/env python3
"""Run the existing 50-member SSH PSD comparison over ocean sectors and latitude bands.

Each sector is a rectangular diagnostic window, not a hydrographic basin mask.
Pacific and Southern sectors are split into continuous longitude intervals.
Runs one region at a time and writes a summary CSV for successful/failed jobs.
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from pathlib import Path

BANDS = {
    "Tropical": (-20, 20),
    "Midlatitude_N": (20, 60),
    "Midlatitude_S": (-60, -20),
    "Highlatitude_N": (60, 80),
    "Highlatitude_S": (-80, -60),
}
SECTORS = {
    "Atlantic": [(-70, 20)],
    "Pacific": [(-180, -70), (120, 180)],
    "Indian": [(20, 120)],
    "Southern": [(-180, -60), (-60, 60), (60, 180)],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2024-01-03")
    p.add_argument("--end", default="2024-12-25")
    p.add_argument("--init-step-days", type=int, default=7)
    p.add_argument("--lead-start", type=int, default=1)
    p.add_argument("--lead-stop", type=int, default=8)
    p.add_argument("--gloens-init-offset-days", type=int, default=1)
    p.add_argument("--n-members", type=int, default=50)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--n-lat", type=int, default=25)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--min-valid-fraction", type=float, default=0.85)
    p.add_argument("--sectors", nargs="+", choices=list(SECTORS),
                   default=["Atlantic", "Pacific", "Indian", "Southern"])
    p.add_argument("--bands", nargs="+", choices=list(BANDS),
                   default=list(BANDS))
    p.add_argument("--output-dir", type=Path, default=Path("results/regional_ssh_50members"))
    p.add_argument("--plot", action="store_true",
                   help="Plot completed regions with plot_gloens_member_ssh_psd.py")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--endpoint", help="Override CloudFerro endpoint")
    p.add_argument("--authenticated", action="store_true")
    args = p.parse_args()
    if args.lead_start < args.gloens_init_offset_days:
        p.error("Lead start must be >= GLOENS init offset to compare aligned valid dates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    worker = script_dir / "gloens_member_ssh_psd.py"
    plotter = script_dir / "plot_gloens_member_ssh_psd.py"
    status_file = args.output_dir / "run_status.csv"
    jobs = []
    for sector in args.sectors:
        for band in args.bands:
            if sector == "Southern" and band != "Highlatitude_S":
                continue
            if sector != "Southern" and band == "Highlatitude_S":
                continue
            south, north = BANDS[band]
            for index, (west, east) in enumerate(SECTORS[sector], 1):
                name = f"{sector}_{band}_sector{index}"
                jobs.append((name, south, north, west, east))
    if not jobs:
        p.error("No region matches selected sectors/bands")

    def record(name, result, message):
        with status_file.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, result, message])

    with status_file.open("w", newline="") as f:
        csv.writer(f).writerow(["region", "status", "message"])
    for i, (name, south, north, west, east) in enumerate(jobs, 1):
        destination = args.output_dir / name
        cmd = [
            sys.executable, str(worker),
            "--start", args.start, "--end", args.end,
            "--init-step-days", str(args.init_step_days),
            "--lead-start", str(args.lead_start),
            "--lead-stop", str(args.lead_stop),
            "--gloens-init-offset-days", str(args.gloens_init_offset_days),
            "--n-members", str(args.n_members), "--seed", str(args.seed),
            "--n-lat", str(args.n_lat), "--n-lon", str(args.n_lon),
            "--min-valid-fraction", str(args.min_valid_fraction),
            "--lat-min", str(south), "--lat-max", str(north),
            "--lon-min", str(west), "--lon-max", str(east),
            "--output-dir", str(destination),
        ]
        if args.endpoint:
            cmd.extend(["--endpoint", args.endpoint])
        if args.authenticated:
            cmd.append("--authenticated")
        print(f"[{i}/{len(jobs)}] {name}: {south}..{north} N, {west}..{east} E", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            record(name, "failed", f"PSD exit code {result.returncode}; see {destination}/failures.csv")
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
            continue
        if args.plot:
            plotted = subprocess.run([
                sys.executable, str(plotter),
                "--spectra", str(destination / "spectra.csv"),
                "--output-dir", str(destination / "figures"),
            ], check=False)
            if plotted.returncode != 0:
                record(name, "plot_failed", f"Plot exit code {plotted.returncode}")
                if not args.continue_on_error:
                    raise SystemExit(plotted.returncode)
                continue
        record(name, "ok", str(destination))
    print(f"Finished {len(jobs)} regional jobs; status: {status_file}")


if __name__ == "__main__":
    main()
