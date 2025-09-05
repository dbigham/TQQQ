"""
TQQQ simulator using unified Nasdaq dataset and FRED interest rates

Overview
--------
This script simulates the daily value path of a 3x leveraged Nasdaq proxy (like TQQQ) using
the unified dataset built by build_unified_nasdaq.py. It overlays the simulated series with
actual TQQQ (adjusted close) from 2010 onward to compare behavior.

Model
-----
Let pct_change_t be the daily percentage change of the unified series from t-1 to t.
Let r_annual_t be the annual interest rate (percent, e.g., 5 for 5%) from FRED.

Simulation update for day t:
  value_t = value_{t-1}
            * (1 + leverage * pct_change_t)
            * (1 - annual_fee / trading_days_per_year)
            * (1 - ((borrowed_fraction * (r_annual_t / 100.0) / trading_days_per_year) / borrow_divisor))

Defaults:
- leverage = 3.0
- annual_fee = 0.0095 (0.95%)
- borrowed_fraction = leverage - 1 (e.g., 2.0)
- borrow_divisor = 0.7 (captures empirical extra borrowing costs)
- trading_days_per_year = 252

Interest Rates
--------------
Fetches a FRED rate series (default: FEDFUNDS). Attempts in this order:
1) fredapi (if installed and API key provided via --fred-api-key or env FRED_API_KEY)
2) pandas_datareader (if installed)
3) Direct CSV from fredgraph (no key) at https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES

Output
------
- Plot with actual TQQQ and simulated series (log y-axis). Annotations show start/end values for both.
- Optional CSV with simulated series values and actual TQQQ for comparison.

CLI
---
python simulate_tqqq.py [--csv unified_nasdaq.csv] [--fred-series FEDFUNDS] [--fred-api-key KEY] \
  [--leverage 3.0] [--annual-fee 0.0095] [--borrow-divisor 0.7] [--trading-days 252] \
  [--save-csv sim_tqqq.csv] [--save-plot sim_tqqq.png] [--no-show]
"""

from __future__ import annotations

import argparse
import os
from typing import Optional


def import_libs():
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import yfinance as yf  # type: ignore
    return pd, np, plt, yf


def load_unified(pd, path: str):
    df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError("Expected 'date' and 'close' in unified CSV")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["close"]).copy()
    df = df[df["close"] > 0]
    df = df.set_index("date")
    return df


def fetch_fred_series(pd, series_id: str, start, end, api_key: Optional[str]):
    # Try fredapi
    if api_key:
        try:
            from fredapi import Fred  # type: ignore
            fred = Fred(api_key=api_key)
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s = s.rename("rate").to_frame()
            s.index = pd.to_datetime(s.index)
            return s
        except Exception:
            pass
    # Try pandas_datareader
    try:
        from pandas_datareader import data as pdr  # type: ignore
        s = pdr.DataReader(series_id, "fred", start, end)
        s = s.rename(columns={series_id: "rate"})
        return s
    except Exception:
        pass
    # Try direct CSV from fredgraph (no key)
    try:
        import io
        import urllib.request
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        with urllib.request.urlopen(url) as resp:
            csv_bytes = resp.read()
        s = pd.read_csv(io.BytesIO(csv_bytes))
        # Columns: typically DATE or observation_date, plus the series value column
        value_col = series_id
        if value_col not in s.columns and value_col.upper() in s.columns:
            value_col = value_col.upper()
        date_col = "DATE"
        if date_col not in s.columns:
            if "observation_date" in s.columns:
                date_col = "observation_date"
            elif "date" in s.columns:
                date_col = "date"
        s = s.rename(columns={date_col: "date", value_col: "rate"})
        s["date"] = pd.to_datetime(s["date"])
        s = s.set_index("date")["rate"].to_frame()
        s = s.loc[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))]
        return s
    except Exception:
        pass
    raise RuntimeError("Failed to fetch FRED series via all methods")


def fetch_tqqq_adjusted(pd, yf, start, end):
    df = yf.download(
        tickers="TQQQ",
        start=str(start.date()),
        end=str((end + pd.Timedelta(days=1)).date()),
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError("Failed to fetch TQQQ from Yahoo Finance")
    close_obj = df["Close"].copy()
    # Coerce to Series if DataFrame (e.g., MultiIndex columns)
    if hasattr(close_obj, "columns"):
        if close_obj.shape[1] == 1:
            s = close_obj.iloc[:, 0]
        else:
            first_col = list(close_obj.columns)[0]
            s = close_obj[first_col]
    else:
        s = close_obj
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[~s.index.duplicated(keep="first")]
    s = s.dropna()
    s.name = "TQQQ_actual"
    return s.to_frame()


def simulate(pd, np, unified_df, rates_df, tqqq_df, leverage: float, annual_fee: float, trading_days: int, borrow_divisor: float):
    # Align dates: intersection of unified and TQQQ
    start = max(unified_df.index.min(), tqqq_df.index.min())
    end = min(unified_df.index.max(), tqqq_df.index.max())
    unified = unified_df.loc[(unified_df.index >= start) & (unified_df.index <= end)].copy()
    tqqq = tqqq_df.loc[(tqqq_df.index >= start) & (tqqq_df.index <= end)].copy()

    # Compute daily pct change from unified
    unified["ret"] = unified["close"].pct_change()

    # Rates: forward-fill to trading days
    rates = rates_df.rename(columns={rates_df.columns[0]: "rate"}) if rates_df.shape[1] == 1 else rates_df.copy()
    rates.index = pd.to_datetime(rates.index)
    rates = rates.sort_index()
    rates = rates.loc[(rates.index >= start) & (rates.index <= end)]
    # Reindex to unified dates and ffill/backfill edges
    rates = rates.reindex(unified.index).ffill().bfill()

    # Prepare arrays
    dates = unified.index.to_numpy()
    rets = unified["ret"].to_numpy()
    rates_annual = rates["rate"].to_numpy()

    sim = np.empty_like(rets, dtype=float)
    sim[:] = np.nan
    # Initial simulated value equals actual
    sim[0] = float(tqqq["TQQQ_actual"].iloc[0])

    borrowed_fraction = leverage - 1.0
    daily_fee = annual_fee / float(trading_days)

    for i in range(1, len(rets)):
        pct_change = rets[i] if np.isfinite(rets[i]) else 0.0
        rate_ann = rates_annual[i]
        daily_borrow_cost = borrowed_fraction * ((rate_ann / 100.0) / float(trading_days))
        factor = (1.0 + leverage * pct_change) * (1.0 - daily_fee) * (1.0 - (daily_borrow_cost / borrow_divisor))
        sim[i] = sim[i - 1] * factor

    sim_df = pd.DataFrame({
        "TQQQ_sim": sim,
    }, index=unified.index)

    # Merge with actual
    merged = tqqq.join(sim_df, how="inner")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Simulate TQQQ using unified dataset and FRED rates")
    parser.add_argument("--csv", default="unified_nasdaq.csv", help="Path to unified dataset CSV")
    parser.add_argument("--fred-series", default="FEDFUNDS", help="FRED series ID for interest rate (default FEDFUNDS)")
    parser.add_argument("--fred-api-key", default=None, help="FRED API key (optional; else tries other methods)")
    parser.add_argument("--leverage", type=float, default=3.0, help="Leverage multiple (default 3.0)")
    parser.add_argument("--annual-fee", type=float, default=0.0095, help="Annual management fee (default 0.0095)")
    parser.add_argument("--borrow-divisor", type=float, default=0.7, help="Divisor applied to daily borrow cost (default 0.7)")
    parser.add_argument("--trading-days", type=int, default=252, help="Trading days per year (default 252)")
    parser.add_argument("--save-csv", default=None, help="If set, saves merged actual/simulated CSV here")
    parser.add_argument("--save-plot", default=None, help="If set, saves plot PNG here")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    args = parser.parse_args()

    pd, np, plt, yf = import_libs()

    unified = load_unified(pd, args.csv)
    start = unified.index.min()
    end = unified.index.max()

    api_key = args.fred_api_key or os.environ.get("FRED_API_KEY")
    rates = fetch_fred_series(pd, args.fred_series, start, end, api_key)
    tqqq = fetch_tqqq_adjusted(pd, yf, start, end)

    merged = simulate(pd, np, unified, rates, tqqq, args.leverage, args.annual_fee, args.trading_days, args.borrow_divisor)

    # Print start/end values for both actual and simulated
    s_date = merged.index.min().date()
    e_date = merged.index.max().date()
    actual_start = float(merged.iloc[0]["TQQQ_actual"]) if not merged.empty else float("nan")
    sim_start = float(merged.iloc[0]["TQQQ_sim"]) if not merged.empty else float("nan")
    actual_end = float(merged.iloc[-1]["TQQQ_actual"]) if not merged.empty else float("nan")
    sim_end = float(merged.iloc[-1]["TQQQ_sim"]) if not merged.empty else float("nan")

    print(f"Span: {s_date} → {e_date}")
    print(f"  Actual TQQQ start: {actual_start:.4f}")
    print(f"  Sim TQQQ start:    {sim_start:.4f}")
    print(f"  Actual TQQQ end:   {actual_end:.4f}")
    print(f"  Sim TQQQ end:      {sim_end:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(merged.index, merged["TQQQ_actual"], label="TQQQ actual", color="#1f77b4")
    ax.semilogy(merged.index, merged["TQQQ_sim"], label="TQQQ simulated", color="#d62728")

    # Annotations (avoid overlap with legend; place near edges)
    ax.annotate(f"Actual start\n{actual_start:.4f}", xy=(merged.index[0], merged["TQQQ_actual"].iloc[0]),
                xytext=(20, 20), textcoords="offset points",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.annotate(f"Sim start\n{sim_start:.4f}", xy=(merged.index[0], merged["TQQQ_sim"].iloc[0]),
                xytext=(20, -30), textcoords="offset points",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.annotate(f"Actual end\n{actual_end:.4f}", xy=(merged.index[-1], merged["TQQQ_actual"].iloc[-1]),
                xytext=(-120, 20), textcoords="offset points",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.annotate(f"Sim end\n{sim_end:.4f}", xy=(merged.index[-1], merged["TQQQ_sim"].iloc[-1]),
                xytext=(-120, -30), textcoords="offset points",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_title("TQQQ Actual vs Simulated (3x Unified Nasdaq with Fees & Borrow Costs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted close (log scale)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if args.save_csv:
        out = merged.copy()
        out.index = out.index.date
        out.index.name = "date"
        out.to_csv(args.save_csv)

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=150)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()


