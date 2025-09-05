"""
Unified Nasdaq (1971→today) time series builder using Yahoo Finance

Overview
--------
This script creates a single continuous daily time series that approximates the Nasdaq-100 (ish) from the start of the Nasdaq in 1971 through today by grafting together:

1) QQQ (Invesco QQQ Trust) — used wherever QQQ has data
2) ^NDX (Nasdaq-100 index) — scaled to match QQQ at their first overlap, used before QQQ begins
3) ^IXIC (Nasdaq Composite index) — scaled to match the scaled ^NDX at their first overlap, used before ^NDX begins

Data Source
-----------
Yahoo Finance via the `yfinance` Python library. Prices are adjusted (dividends/splits) to maintain continuity in the presence of corporate actions. Adjusted historical values can change over time as providers revise data; minor differences from past runs are expected.

Method
------
- Download adjusted daily closes for tickers: QQQ, ^NDX, ^IXIC (Auto-adjusted close via yfinance).
- Determine seam (anchor) dates:
  - seam_qqq_ndx: first date where both QQQ and ^NDX have data (can override via --seam-qqq-ndx)
  - seam_ixic_ndx: first date where both ^IXIC and ^NDX have data (can override via --seam-ixic-ndx)
- Compute scaling factors to preserve continuity:
  - scale_ndx_to_qqq = QQQ[seam_qqq_ndx] / NDX[seam_qqq_ndx]
  - scale_ixic_to_ndx = (NDX[seam_ixic_ndx] * scale_ndx_to_qqq) / IXIC[seam_ixic_ndx]
- Build unified series:
  - For dates < seam_ixic_ndx: use IXIC * scale_ixic_to_ndx (source = "IXIC_scaled")
  - For seam_ixic_ndx ≤ dates < seam_qqq_ndx: use NDX * scale_ndx_to_qqq (source = "NDX_scaled")
  - For dates ≥ seam_qqq_ndx: use QQQ as-is (source = "QQQ")
- Output:
  - Saves CSV (default: unified_nasdaq.csv) with columns: date, close, source
  - Prints values at: dataset start, seam_ixic_ndx, seam_qqq_ndx, and dataset end
  - Shows a log-scale plot of the unified series

CLI
---
python build_unified_nasdaq.py [--out unified_nasdaq.csv] [--upgrade] \
  [--seam-qqq-ndx YYYY-MM-DD] [--seam-ixic-ndx YYYY-MM-DD] [--no-plot] \
  [--pre-ndx-plus2pct]

- --upgrade: attempts to `pip install -U yfinance pandas matplotlib` before running
- Seam flags let you pin anchor dates if you want reproducibility across runs

Notes
-----
- Yahoo symbols: QQQ, ^NDX, ^IXIC
- Minor differences vs historical runs are normal due to adjusted data updates
- The resulting series is an approximation for research, not investment advice
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from typing import Optional, Tuple


def upgrade_packages_if_requested(upgrade: bool) -> None:
    if not upgrade:
        return
    pkgs = [
        "yfinance",
        "pandas",
        "matplotlib",
    ]
    cmd = [sys.executable, "-m", "pip", "install", "-U", *pkgs]
    print("Upgrading packages:", " ".join(pkgs))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print("Warning: package upgrade failed:", exc)


def import_libs():
    # Import inside a function so optional upgrade can run first
    import pandas as pd  # type: ignore
    import yfinance as yf  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    return pd, yf, plt


def download_adjusted_series(yf, pd, ticker: str, start: str = "1970-01-01"):
    df = yf.download(
        tickers=ticker,
        start=start,
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    # When auto_adjust=True, 'Close' is adjusted close
    if "Close" not in df.columns:
        raise RuntimeError(f"Expected 'Close' in data for {ticker}")
    close_obj = df["Close"].copy()
    # If we happen to get a DataFrame (e.g., MultiIndex columns), reduce to a Series
    if hasattr(close_obj, "columns"):
        if close_obj.shape[1] == 1:
            s = close_obj.iloc[:, 0]
        else:
            # Choose the first column deterministically
            first_col = list(close_obj.columns)[0]
            s = close_obj[first_col]
    else:
        s = close_obj
    s.name = ticker
    # Normalize index to naive dates (no tz)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[~s.index.duplicated(keep="first")]
    s = s.dropna()
    return s


def find_first_overlap_date(pd, a, b) -> dt.date:
    intersection = a.index.intersection(b.index)
    if len(intersection) == 0:
        raise RuntimeError("No overlapping dates found between series")
    return pd.to_datetime(intersection.min()).date()


def parse_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    return dt.date.fromisoformat(value)


def build_unified(pd, qqq, ndx, ixic, seam_qqq_ndx_date: dt.date, seam_ixic_ndx_date: dt.date, ixic_extra_cagr: float = 0.0):
    # Extract anchor values
    seam_qqq_ndx_ts = pd.to_datetime(seam_qqq_ndx_date)
    seam_ixic_ndx_ts = pd.to_datetime(seam_ixic_ndx_date)

    if seam_qqq_ndx_ts not in qqq.index or seam_qqq_ndx_ts not in ndx.index:
        raise RuntimeError(
            f"Seam QQQ/NDX date {seam_qqq_ndx_date} must be present in both QQQ and ^NDX series"
        )
    if seam_ixic_ndx_ts not in ndx.index or seam_ixic_ndx_ts not in ixic.index:
        raise RuntimeError(
            f"Seam IXIC/NDX date {seam_ixic_ndx_date} must be present in both ^IXIC and ^NDX series"
        )

    qqq_anchor = float(qqq.loc[[seam_qqq_ndx_ts]].iloc[0])
    ndx_anchor_at_qqq = float(ndx.loc[[seam_qqq_ndx_ts]].iloc[0])
    scale_ndx_to_qqq = qqq_anchor / ndx_anchor_at_qqq

    ndx_anchor_at_ixic_unscaled = float(ndx.loc[[seam_ixic_ndx_ts]].iloc[0])
    ndx_anchor_at_ixic_scaled = ndx_anchor_at_ixic_unscaled * scale_ndx_to_qqq
    ixic_anchor = float(ixic.loc[[seam_ixic_ndx_ts]].iloc[0])
    scale_ixic_to_ndx = ndx_anchor_at_ixic_scaled / ixic_anchor

    # Build segments
    ixic_scaled = (ixic[ixic.index < seam_ixic_ndx_ts] * scale_ixic_to_ndx).to_frame("close")
    if ixic_extra_cagr and abs(ixic_extra_cagr) > 0.0:
        # Apply extra CAGR to pre-NDX IXIC so that earlier points are adjusted
        # by (1 + cagr)^(-years_to_seam), keeping the seam value unchanged.
        years_to_seam = (seam_ixic_ndx_ts - ixic_scaled.index) / pd.Timedelta(days=1)
        years_to_seam = years_to_seam / 365.25
        factor = (1.0 + float(ixic_extra_cagr)) ** (-years_to_seam)
        ixic_scaled["close"] = ixic_scaled["close"].to_numpy() * factor.to_numpy()
        ixic_source_label = f"IXIC_scaled_extra{ixic_extra_cagr:+.2%}"
    else:
        ixic_source_label = "IXIC_scaled"
    ixic_scaled["source"] = ixic_source_label

    ndx_scaled = (ndx[(ndx.index >= seam_ixic_ndx_ts) & (ndx.index < seam_qqq_ndx_ts)] * scale_ndx_to_qqq).to_frame("close")
    ndx_scaled["source"] = "NDX_scaled"

    qqq_part = qqq[qqq.index >= seam_qqq_ndx_ts].to_frame("close")
    qqq_part["source"] = "QQQ"

    unified = pd.concat([ixic_scaled, ndx_scaled, qqq_part]).sort_index()

    # Compute seam and endpoints from unified
    start_date = pd.to_datetime(unified.index.min()).date()
    end_date = pd.to_datetime(unified.index.max()).date()
    start_value = float(unified.iloc[0]["close"]) if not unified.empty else float("nan")
    end_value = float(unified.iloc[-1]["close"]) if not unified.empty else float("nan")

    # Values exactly at seams (as present in unified)
    ixic_to_ndx_value = float(unified.loc[pd.to_datetime(seam_ixic_ndx_date)]["close"])  # from NDX_scaled
    qqq_seam_value = float(unified.loc[pd.to_datetime(seam_qqq_ndx_date)]["close"])      # from QQQ

    meta = {
        "scale_ndx_to_qqq": scale_ndx_to_qqq,
        "scale_ixic_to_ndx": scale_ixic_to_ndx,
        "seam_qqq_ndx_date": seam_qqq_ndx_date.isoformat(),
        "seam_ixic_ndx_date": seam_ixic_ndx_date.isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "start_value": start_value,
        "ixic_to_ndx_value": ixic_to_ndx_value,
        "qqq_seam_value": qqq_seam_value,
        "end_value": end_value,
        "ixic_extra_cagr": float(ixic_extra_cagr),
    }

    return unified, meta


def main():
    parser = argparse.ArgumentParser(description="Build unified Nasdaq time series from QQQ, ^NDX, ^IXIC")
    parser.add_argument("--out", default="unified_nasdaq.csv", help="Output CSV path")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade yfinance/pandas/matplotlib before running")
    parser.add_argument("--seam-qqq-ndx", dest="seam_qqq_ndx", default=None, help="Anchor date (YYYY-MM-DD) for QQQ⇄NDX seam")
    parser.add_argument("--seam-ixic-ndx", dest="seam_ixic_ndx", default=None, help="Anchor date (YYYY-MM-DD) for IXIC⇄NDX seam")
    parser.add_argument("--no-plot", action="store_true", help="Skip showing the log plot")
    parser.add_argument("--pre-ndx-plus2pct", action="store_true", help="Apply +2% annual CAGR to pre-NDX IXIC segment")
    args = parser.parse_args()

    upgrade_packages_if_requested(args.upgrade)
    pd, yf, plt = import_libs()

    # Download
    qqq = download_adjusted_series(yf, pd, "QQQ", start="1970-01-01")
    ndx = download_adjusted_series(yf, pd, "^NDX", start="1970-01-01")
    ixic = download_adjusted_series(yf, pd, "^IXIC", start="1970-01-01")

    # Determine seams if not provided
    seam_qqq_ndx = parse_date(args.seam_qqq_ndx)
    if seam_qqq_ndx is None:
        seam_qqq_ndx = find_first_overlap_date(pd, qqq, ndx)

    seam_ixic_ndx = parse_date(args.seam_ixic_ndx)
    if seam_ixic_ndx is None:
        seam_ixic_ndx = find_first_overlap_date(pd, ixic, ndx)

    # Build unified
    ixic_extra_cagr = 0.02 if args.pre_ndx_plus2pct else 0.0
    unified, meta = build_unified(pd, qqq, ndx, ixic, seam_qqq_ndx, seam_ixic_ndx, ixic_extra_cagr=ixic_extra_cagr)

    # Save CSV (date, close, source)
    out_path = args.out
    out_df = unified.copy()
    out_df.index = out_df.index.date  # dates only
    out_df.index.name = "date"
    out_df.to_csv(out_path)

    # Save metadata sidecar .json for reproducibility
    meta_path = out_path.rsplit(".", 1)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Print seam/start/end values
    print("\nUnified Nasdaq dataset saved:")
    print("  CSV:", out_path)
    print("  Meta:", meta_path)
    print("\nValues:")
    print(f"  Start {meta['start_date']}: {meta['start_value']:.6f}")
    print(f"  IXIC→NDX seam {meta['seam_ixic_ndx_date']}: {meta['ixic_to_ndx_value']:.6f}")
    print(f"  NDX→QQQ seam {meta['seam_qqq_ndx_date']}: {meta['qqq_seam_value']:.6f}")
    print(f"  End {meta['end_date']}: {meta['end_value']:.6f}")
    print("\nScaling factors:")
    print(f"  scale_ndx_to_qqq: {meta['scale_ndx_to_qqq']:.8f}")
    print(f"  scale_ixic_to_ndx: {meta['scale_ixic_to_ndx']:.8f}")

    # Plot
    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(unified.index, unified["close"], label="Unified series")
        ax.axvline(pd.to_datetime(meta["seam_ixic_ndx_date"]), color="gray", linestyle="--", alpha=0.5, label="IXIC→NDX seam")
        ax.axvline(pd.to_datetime(meta["seam_qqq_ndx_date"]), color="gray", linestyle=":", alpha=0.5, label="NDX→QQQ seam")
        title = "Unified Nasdaq (QQQ + scaled ^NDX + scaled ^IXIC)"
        if meta.get("ixic_extra_cagr", 0.0):
            title += " [+2% pre-NDX]" if abs(meta["ixic_extra_cagr"] - 0.02) < 1e-9 else f" [+{meta['ixic_extra_cagr']*100:.2f}% pre-NDX]"
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted close (log scale)")
        ax.legend()
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


