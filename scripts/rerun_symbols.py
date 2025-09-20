#!/usr/bin/env python3
"""
Batch runner for strategy_tqqq_reserve.py across a list of symbols or all detected symbols.

Features:
- Accepts --symbols SYMBOL [SYMBOL ...] or --all to auto-discover symbol folders
- Pass-through of --experiment/--start/--end/--fred-series
- Automatically sets per-symbol output paths for plot, CSV, and summary
- Uses --no-show by default for non-interactive runs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable, List


EXCLUDE_DIRS = {
    "__pycache__",
    "scripts",
    "symbol_data",
    "temperature_cache",
}


def discover_symbols(root: str) -> List[str]:
    symbols: List[str] = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if name in EXCLUDE_DIRS:
            continue
        # Basic heuristic: treat any directory in root as a symbol folder
        # Map folder name -> ticker symbol (uppercased, preserving hyphens and carets)
        symbols.append(name.upper())
    symbols.sort()
    return symbols


def run_symbol(symbol: str, experiment: str, start: str | None, end: str | None, fred_series: str | None, dry_run: bool) -> int:
    symbol_dir = "qqq" if symbol.upper() == "QQQ" else symbol.lower()
    os.makedirs(symbol_dir, exist_ok=True)
    plot_path = os.path.join(symbol_dir, f"strategy_{symbol_dir}_reserve_{experiment}.png")
    csv_path = os.path.join(symbol_dir, f"strategy_{symbol_dir}_reserve_{experiment}.csv")
    # Summary path is optional; the strategy writes a sensible default if omitted

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
    parser = argparse.ArgumentParser(description="Re-run strategy across symbols, saving plots/CSVs/summaries")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", nargs="+", help="List of ticker symbols to run (e.g., QQQ NVDA BTC-USD)")
    group.add_argument("--all", action="store_true", help="Auto-discover all symbol folders at repo root")
    parser.add_argument("--experiment", default="A1", help="Experiment id to run (default A1)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    parser.add_argument("--fred-series", default=None, help="FRED series id override (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(repo_root)

    if args.all:
        syms = discover_symbols(repo_root)
    else:
        syms = [s.upper() for s in args.symbols]

    failures: List[str] = []
    for sym in syms:
        code = run_symbol(sym, args.experiment, args.start, args.end, args.fred_series, args.dry_run)
        if code != 0:
            failures.append(sym)

    if failures:
        print("Failures:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


