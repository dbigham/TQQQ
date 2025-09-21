"""
Nasdaq "temperature" based on fitted constant-growth curve

Overview
--------
This script defines a Nasdaq temperature as the ratio of actual price to a
fitted constant-growth curve:

  temperature(date) = actual_price(date) / fitted_price(date)

Interpretation examples:
- T = 1.0 → price is exactly on the fitted curve
- T = 1.5 → price is 50% above the curve
- T = 0.5 → price is 50% below the curve

How it works
------------
1) Loads the unified dataset CSV produced by build_unified_nasdaq.py.
2) Fits a constant annual-growth curve using a robust 3-step iterative
   outlier rejection (log-linear fit of ln(price) vs. years since start):
   - Step 1: Fit on all points.
   - Step 2: Refit excluding points with relative error > thresh2.
   - Step 3: Refit excluding points with relative error > thresh3.
3) Computes temperature for each date and plots it with reference lines:
   - Solid line at T = 1.0
   - Dotted lines at T = 1.5 and T = 0.5

API
---
The module exposes a function:

  get_nasdaq_temperature(date_input) -> float

It returns the temperature for the specified date (string YYYY-MM-DD or datetime-like).
If the date is not a trading day, it uses the most recent available prior date.

CLI
---
python nasdaq_temperature.py [--csv unified_nasdaq.csv] [--thresh2 0.35] [--thresh3 0.15] \
  [--date YYYY-MM-DD] [--save-plot temperature.png] [--no-show]
"""

from __future__ import annotations

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqqq import load_price_csv, iterative_constant_growth


class NasdaqTemperature:
    def __init__(self, csv_path: str = "unified_nasdaq.csv", thresh2: float = 0.35, thresh3: float = 0.15):
        df, start_ts = load_price_csv(csv_path, set_index=True, add_elapsed_years=True)
        self.df = df
        self.start_ts = start_ts
        t = df["t_years"].to_numpy(dtype=float)
        prices = df["close"].to_numpy(dtype=float)
        result = iterative_constant_growth(t, prices, thresholds=[thresh2, thresh3])
        final = result.final
        self.A = float(final.A)
        self.r = float(final.r)
        self._predictions = final.predictions

    def get_temperature(self, date_input) -> float:
        ts = pd.to_datetime(date_input)
        # Use last available date on/before ts
        if ts not in self.df.index:
            prev = self.df.index[self.df.index <= ts]
            if len(prev) == 0:
                raise ValueError("Date precedes available data range")
            ts = prev.max()
        price = float(self.df.loc[ts, "close"])  # actual
        t_years = (ts - self.start_ts).days / 365.25
        pred = float(self.A * (1.0 + self.r) ** t_years)
        return price / pred


def plot_temperature(df, A: float, r: float, save_plot: Optional[str], no_show: bool):
    t_years = df["t_years"].to_numpy(dtype=float)
    pred = A * np.power(1.0 + r, t_years)
    temp = df["close"].to_numpy(dtype=float) / pred

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, temp, label="Temperature", color="#1f77b4")

    ax.axhline(1.0, color="gray", linestyle="-", alpha=0.8, label="T = 1.0")
    ax.axhline(1.5, color="gray", linestyle=":", alpha=0.7, label="T = 1.5")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="T = 0.5")

    ax.set_title("Nasdaq Temperature (Actual / Fitted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (ratio)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_plot:
        fig.savefig(save_plot, dpi=150)
    if not no_show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compute and plot Nasdaq temperature vs fitted curve")
    parser.add_argument("--csv", default="unified_nasdaq.csv", help="Path to unified dataset CSV")
    parser.add_argument("--thresh2", type=float, default=0.35, help="Outlier cutoff for step 2 (default 0.35)")
    parser.add_argument("--thresh3", type=float, default=0.15, help="Outlier cutoff for step 3 (default 0.15)")
    parser.add_argument("--date", default=None, help="If provided, prints the temperature for this date (YYYY-MM-DD)")
    parser.add_argument("--save-plot", default=None, help="If set, saves the temperature plot to this PNG path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    args = parser.parse_args()

    temp_model = NasdaqTemperature(csv_path=args.csv, thresh2=args.thresh2, thresh3=args.thresh3)

    if args.date:
        tval = temp_model.get_temperature(args.date)
        print(f"Temperature on {args.date}: {tval:.4f}")

    plot_temperature(temp_model.df, temp_model.A, temp_model.r, args.save_plot, args.no_show)


def get_nasdaq_temperature(date_input, csv_path: str = "unified_nasdaq.csv", thresh2: float = 0.35, thresh3: float = 0.15) -> float:
    model = NasdaqTemperature(csv_path=csv_path, thresh2=thresh2, thresh3=thresh3)
    return model.get_temperature(date_input)


if __name__ == "__main__":
    main()


