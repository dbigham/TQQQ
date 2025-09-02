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
from typing import Optional, Tuple


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
    df = df.set_index("date")
    start_ts = df.index.min()
    t_years = (df.index - start_ts).days / 365.25
    df["t_years"] = t_years
    return df, start_ts


def fit_log_linear(np, t_years, prices) -> Tuple[float, float]:
    y = np.log(prices)
    x = np.asarray(t_years)
    slope, intercept = np.polyfit(x, y, 1)
    import math
    A = float(math.exp(intercept))
    r = float(math.exp(slope) - 1.0)
    return A, r


def compute_relative_errors(np, A: float, r: float, t_years, prices):
    pred = A * np.power(1.0 + r, t_years)
    rel_err = np.abs((prices - pred) / pred)
    return pred, rel_err


def iterative_fit(np, t_years, prices, thresh2: float, thresh3: float):
    A1, r1 = fit_log_linear(np, t_years, prices)
    _, rel1 = compute_relative_errors(np, A1, r1, t_years, prices)

    mask2 = rel1 <= thresh2
    A2, r2 = fit_log_linear(np, t_years[mask2], prices[mask2])
    _, rel2 = compute_relative_errors(np, A2, r2, t_years, prices)

    mask3 = rel2 <= thresh3
    A3, r3 = fit_log_linear(np, t_years[mask3], prices[mask3])
    return A3, r3


class NasdaqTemperature:
    def __init__(self, csv_path: str = "unified_nasdaq.csv", thresh2: float = 0.35, thresh3: float = 0.15):
        pd, np, _ = import_libs()
        self._pd = pd
        self._np = np
        self.df, self.start_ts = load_series(pd, csv_path)
        t = self.df["t_years"].to_numpy()
        p = self.df["close"].to_numpy()
        self.A, self.r = iterative_fit(np, t, p, thresh2, thresh3)

    def get_temperature(self, date_input) -> float:
        pd = self._pd
        np = self._np
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


def plot_temperature(pd, np, plt, df, A: float, r: float, start_ts, save_plot: Optional[str], no_show: bool):
    t_years = df["t_years"].to_numpy()
    pred = A * np.power(1.0 + r, t_years)
    temp = df["close"].to_numpy() / pred

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

    pd, np, plt = import_libs()

    temp_model = NasdaqTemperature(csv_path=args.csv, thresh2=args.thresh2, thresh3=args.thresh3)

    if args.date:
        tval = temp_model.get_temperature(args.date)
        print(f"Temperature on {args.date}: {tval:.4f}")

    plot_temperature(pd, np, plt, temp_model.df, temp_model.A, temp_model.r, temp_model.start_ts, args.save_plot, args.no_show)


def get_nasdaq_temperature(date_input, csv_path: str = "unified_nasdaq.csv", thresh2: float = 0.35, thresh3: float = 0.15) -> float:
    model = NasdaqTemperature(csv_path=csv_path, thresh2=thresh2, thresh3=thresh3)
    return model.get_temperature(date_input)


if __name__ == "__main__":
    main()


