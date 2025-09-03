"""
Iterative constant-growth curve fit for unified Nasdaq dataset

Overview
--------
This script fits an exponential (constant annual growth rate) curve to a time series,
robustly handling outliers via iterative outlier rejection:

1) Fit using all points (log-linear fit: ln(price) ~ a + b * years)
2) Refit after excluding points whose relative error vs the first fit exceeds a threshold
3) Refit again after excluding points whose relative error vs the second fit exceeds a stricter threshold

Model
-----
price(t_years) = A * (1 + r)^(t_years)

Taking logs: ln(price) = ln(A) + t_years * ln(1 + r)
We fit ln(price) against t_years with linear regression (numpy.polyfit).
Then:
  A = exp(intercept)
  r = exp(slope) - 1

Input
-----
- CSV created by build_unified_nasdaq.py: columns [date, close, source]
  - date: YYYY-MM-DD
  - close: adjusted close (scaled unified series)

Output
------
- Printed initial value A (at start date) with 4 decimals
- Printed annual growth rate r with two decimal places (percentage)
- Plot showing:
  - Raw series
  - Final fitted curve (exponential), on log y-axis for readability
  - Text annotation with A and r
- Optional saved plot (PNG) when --save-plot is given

CLI
---
python fit_constant_growth.py [--csv unified_nasdaq.csv] [--no-show] [--save-plot FIT.png] \
  [--thresh2 0.35] [--thresh3 0.15]

Thresholds are relative error cutoffs (absolute), e.g. 0.35 = 35%.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple


def import_libs():
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    return pd, np, plt


def load_series(pd, csv_path: str):
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError("Expected columns 'date' and 'close' in CSV")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["close"]).copy()
    df = df[df["close"] > 0]
    start_ts = df["date"].iloc[0]
    # Use 365.25 days per year
    t_years = (df["date"] - start_ts).dt.total_seconds() / (365.25 * 24 * 3600)
    df["t_years"] = t_years
    return df, start_ts.date()


def fit_log_linear(np, t_years, prices) -> Tuple[float, float]:
    # Fit ln(price) = intercept + slope * t_years
    y = np.log(prices)
    x = np.asarray(t_years)
    slope, intercept = np.polyfit(x, y, 1)
    A = float(math.exp(intercept))
    r = float(math.exp(slope) - 1.0)
    return A, r


def compute_relative_errors(np, A: float, r: float, t_years, prices):
    pred = A * np.power(1.0 + r, t_years)
    rel_err = np.abs((prices - pred) / pred)
    return pred, rel_err


def iterative_fit(np, t_years, prices, thresholds: List[float]):
    # Step 1: initial fit
    A1, r1 = fit_log_linear(np, t_years, prices)
    pred1, rel1 = compute_relative_errors(np, A1, r1, t_years, prices)

    # Step 2: exclude > thresholds[0]
    mask2 = rel1 <= thresholds[0]
    if not mask2.any():
        raise RuntimeError(
            f"No points remain after applying threshold {thresholds[0]:.3f} in step 2; "
            "try using a larger --thresh2 value."
        )
    A2, r2 = fit_log_linear(np, t_years[mask2], prices[mask2])
    pred2, rel2 = compute_relative_errors(np, A2, r2, t_years, prices)

    # Step 3: exclude > thresholds[1] based on step-2 fit
    mask3 = rel2 <= thresholds[1]
    if not mask3.any():
        raise RuntimeError(
            f"No points remain after applying threshold {thresholds[1]:.3f} in step 3; "
            "try using a larger --thresh3 value."
        )
    A3, r3 = fit_log_linear(np, t_years[mask3], prices[mask3])

    return (A1, r1, pred1, rel1), (A2, r2, pred2, rel2, mask2), (A3, r3, mask3)


def main():
    parser = argparse.ArgumentParser(description="Iterative constant-growth fit with outlier rejection")
    parser.add_argument("--csv", default="unified_nasdaq.csv", help="Input CSV path from build_unified_nasdaq.py")
    parser.add_argument("--thresh2", type=float, default=0.35, help="Relative error cutoff for step 2 (default 0.35)")
    parser.add_argument("--thresh3", type=float, default=0.15, help="Relative error cutoff for step 3 (default 0.15)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    parser.add_argument("--save-plot", default=None, help="If set, saves the plot to this PNG path")
    args = parser.parse_args()

    pd, np, plt = import_libs()

    df, start_date = load_series(pd, args.csv)
    t = df["t_years"].to_numpy()
    p = df["close"].to_numpy()

    (A1, r1, pred1, rel1), (A2, r2, pred2, rel2, mask2), (A3, r3, mask3) = iterative_fit(
        np, t, p, thresholds=[args.thresh2, args.thresh3]
    )

    # Final parameters
    A_final, r_final = A3, r3

    # Print summary
    print(f"Start date: {start_date}")
    print(f"Initial value (A): {A_final:.4f}")
    print(f"Annual growth rate (r): {r_final * 100.0:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(df["date"], p, label="Unified data", color="#1f77b4", alpha=0.7)

    # Fitted curve on full domain
    pred_final = A_final * np.power(1.0 + r_final, t)
    ax.semilogy(df["date"], pred_final, label="Fitted constant growth", color="#d62728")

    # Annotation
    text = f"A (start): {A_final:.4f}\nAnnual growth: {r_final*100.0:.2f}%"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title("Iterative Constant-Growth Fit (Unified Nasdaq)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close (log scale)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=150)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()


