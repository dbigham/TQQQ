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
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

from tqqq import FitResult, load_price_csv, iterative_constant_growth


def run_constant_growth_fit(
    csv_path: str,
    thresh2: float,
    thresh3: float,
) -> Tuple[pd.DataFrame, pd.Timestamp, FitResult]:
    """Load the unified dataset and run the 3-step fitter."""

    df, start_ts = load_price_csv(csv_path, add_elapsed_years=True)
    t = df["t_years"].to_numpy(dtype=float)
    prices = df["close"].to_numpy(dtype=float)
    result = iterative_constant_growth(t, prices, thresholds=[thresh2, thresh3])
    return df, start_ts, result


def fit_constant_growth_from_csv(
    csv_path: str = "unified_nasdaq.csv",
    thresh2: float = 0.35,
    thresh3: float = 0.15,
) -> FitResult:
    """Convenience API used by other modules/tests."""

    _, _, result = run_constant_growth_fit(csv_path, thresh2, thresh3)
    return result


def main():
    parser = argparse.ArgumentParser(description="Iterative constant-growth fit with outlier rejection")
    parser.add_argument("--csv", default="unified_nasdaq.csv", help="Input CSV path from build_unified_nasdaq.py")
    parser.add_argument("--thresh2", type=float, default=0.35, help="Relative error cutoff for step 2 (default 0.35)")
    parser.add_argument("--thresh3", type=float, default=0.15, help="Relative error cutoff for step 3 (default 0.15)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    parser.add_argument("--save-plot", default=None, help="If set, saves the plot to this PNG path")
    args = parser.parse_args()

    df, start_ts, result = run_constant_growth_fit(args.csv, args.thresh2, args.thresh3)
    t = df["t_years"].to_numpy(dtype=float)
    prices = df["close"].to_numpy(dtype=float)

    final_step = result.final
    A_final, r_final = final_step.A, final_step.r
    pred_final = final_step.predictions

    # Print summary
    print(f"Start date: {start_ts.date()}")
    print(f"Initial value (A): {A_final:.4f}")
    print(f"Annual growth rate (r): {r_final * 100.0:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(df["date"], prices, label="Unified data", color="#1f77b4", alpha=0.7)

    # Fitted curve on full domain
    ax.semilogy(df["date"], pred_final, label="Fitted constant growth", color="#d62728")

    # Annotation
    text = f"A (start): {A_final:.4f}\nAnnual growth: {r_final*100.0:.2f}%"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title(f"Iterative Constant-Growth Fit (Unified Nasdaq, CAGR: {r_final*100.0:.2f}%)")
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


