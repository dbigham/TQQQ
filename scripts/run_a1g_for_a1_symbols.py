#!/usr/bin/env python3
"""
Run the A1g experiment for all curated A1 symbols and save artifacts per symbol.

This script uses the batch runner helpers in scripts/rerun_symbols.py to invoke
strategy_tqqq_reserve.py with per-symbol output paths (PNG + CSV) and --no-show.
"""

from __future__ import annotations

import argparse
from typing import List

import os
import sys
import subprocess

# Ensure imports resolve relative to the repo root when executed from scripts/
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from a1_symbol_stats import A1_SYMBOL_STATS


def run_symbol(symbol: str, experiment: str, start: str | None, end: str | None, fred_series: str | None, dry_run: bool) -> int:
    folder_name = "qqq" if symbol.upper() == "QQQ" else symbol.lower()
    symbol_dir = os.path.join("symbols", folder_name)
    os.makedirs(symbol_dir, exist_ok=True)
    plot_path = os.path.join(symbol_dir, f"strategy_{folder_name}_reserve_{experiment}.png")
    csv_path = os.path.join(symbol_dir, f"strategy_{folder_name}_reserve_{experiment}.csv")
    summary_path = os.path.join(symbol_dir, f"{folder_name}_{experiment}_summary.md")

    cmd = [
        sys.executable,
        "strategy_tqqq_reserve.py",
        "--base-symbol",
        symbol,
        "--experiment",
        experiment,
        "--save-plot",
        plot_path,
        "--save-csv",
        csv_path,
        "--save-summary",
        summary_path,
        "--no-show",
    ]
    if start:
        cmd.extend(["--start", start])
    if end:
        cmd.extend(["--end", end])
    if fred_series:
        cmd.extend(["--fred-series", fred_series])

    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return 0

    print(f"Running {symbol} ({experiment}) …")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A1g for all curated A1 symbols")
    parser.add_argument("--experiment", default="A1g", help="Experiment id to run (default A1g)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    parser.add_argument("--fred-series", default=None, help="FRED series id override (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    symbols: List[str] = sorted(A1_SYMBOL_STATS.keys())

    failures: List[str] = []
    for sym in symbols:
        code = run_symbol(sym, args.experiment, args.start, args.end, args.fred_series, args.dry_run)
        if code != 0:
            failures.append(sym)

    if failures:
        print("Failures:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
