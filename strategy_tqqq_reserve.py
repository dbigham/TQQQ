"""
TQQQ + cash-reserve strategy simulator with temperature and momentum filters

Overview
--------
This script simulates an allocation strategy that pairs a leveraged Nasdaq proxy (TQQQ-like)
with a cash reserve. Allocation to TQQQ is based on the market "temperature" (actual price
relative to a fitted constant-growth curve) and is adjusted by recent momentum filters and
interest-rate conditions. The reserve earns the current interest rate.

Key Concepts
------------
- Temperature T: ratio actual/fitted. T=1 on curve, 1.5 means 50% above, 0.5 is 50% below.
- Allocation targets by T (linear interpolation between anchors):
  - T >= 1.5 → 20% in TQQQ
  - T = 1.0 → 80% in TQQQ
  - T <= 0.9 → 100% in TQQQ
- Never add to TQQQ when T > 1.3 (buys blocked; sells allowed).
- Monthly rebalancing cadence ≈ 22 trading days. If filters block on the due day,
  keep checking daily until a day without blocks, then reset the 22-day cadence.
- Momentum buy-block filters (delay buying on a due day):
  - QQQ fell ≥ 3% over last 3 trading days
  - QQQ fell ≥ 3% over last 6 trading days
  - QQQ fell ≥ 2% over last 12 trading days
  - QQQ fell ≥ 0% over last 22 trading days
  - T > 1.3
- Momentum sell-block filters (delay selling on a due day):
  - QQQ gained ≥ 3% over last 3 trading days
  - QQQ gained ≥ 3% over last 6 trading days
  - QQQ gained ≥ 2.25% over last 12 trading days
  - QQQ gained ≥ 0.75% over last 22 trading days
- Crash de-risk rule: If QQQ dropped > 15.25% over last 22 trading days, move 100% to cash immediately.
- Interest-rate taper: Between 10% and 12% annual rates, reduce TQQQ allocation linearly to 0 at 12%.

Leverage and Returns
--------------------
We simulate TQQQ-like daily growth using unified Nasdaq returns and borrowing costs (via FRED rate):
  daily_factor = (1 + leverage * daily_return)
                 * (1 - annual_fee / trading_days)
                 * (1 - (borrowed_fraction * (rate/100) / trading_days) / borrow_divisor)
where leverage=3.0, borrowed_fraction=2.0, annual_fee=0.0095, borrow_divisor=0.7, trading_days=252.
Cash grows by: (1 + (rate/100) / trading_days) per day.

Inputs
------
- unified_nasdaq.csv from build_unified_nasdaq.py (date, close)
- Interest rate from FRED (default FEDFUNDS), fetched without key via fredgraph CSV (or other fallbacks)
- Temperature fit via nasdaq_temperature.py (robust 3-step fit)

Outputs
-------
- Plot 1: normalized unified series vs strategy portfolio value (same start = 1.0). Legend includes CAGR.
- Plot 2: temperature (black) with reference lines at T=1.0 (solid), T=1.5 and T=0.5 (dotted), and
  deployed proportion to TQQQ (green) on same axis.

Run
---
python strategy_tqqq_reserve.py [--csv unified_nasdaq.csv] [--experiment A25] \
  [--start 2010-01-01] [--end 2025-01-10] [--fred-series FEDFUNDS] [--leverage 3.0] \
  [--annual-fee 0.0095] [--borrow-divisor 0.7] [--trading-days 252] \
  [--initial-capital 100000] [--print-rebalances] [--save-plot strategy_tqqq.png] [--no-show]

The CLI defaults to the best-performing experiment; as of the latest calibration this is
A25 (Momentum Surge Overlay).
"""

from __future__ import annotations

import argparse
import math
from typing import Optional


def import_libs():
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    return pd, np, plt


def fetch_fred_series(pd, series_id: str, start, end):
    # Try fredapi with env key if available
    import os
    api_key = os.environ.get("FRED_API_KEY")
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
    # Fallback to fredgraph CSV
    try:
        import io
        import urllib.request
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        with urllib.request.urlopen(url) as resp:
            csv_bytes = resp.read()
        s = pd.read_csv(io.BytesIO(csv_bytes))
        # fredgraph currently uses 'observation_date' for the date column but older
        # exports used 'DATE'. Support either for robustness.
        date_col = "DATE" if "DATE" in s.columns else "observation_date"
        value_col = series_id if series_id in s.columns else series_id.upper()
        s = s.rename(columns={date_col: "date", value_col: "rate"})
        s["date"] = pd.to_datetime(s["date"])
        s = s.set_index("date")["rate"].to_frame()
        s = s.loc[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))]
        return s
    except Exception as exc:
        raise RuntimeError("Failed to fetch FRED series") from exc


def load_unified(pd, csv_path: str):
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError("Expected columns 'date' and 'close'")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["close"]).copy()
    df = df[df["close"] > 0]
    df = df.set_index("date")
    return df


def compute_temperature_series(pd, np, df, csv_path: str = "unified_nasdaq.csv"):
    """Compute temperature series using the provided CSV for the fit."""
    # Use the robust fit from nasdaq_temperature module
    from nasdaq_temperature import NasdaqTemperature  # type: ignore
    model = NasdaqTemperature(csv_path=csv_path)
    # Align to df
    start_ts = model.start_ts
    t_years = (df.index - start_ts).days / 365.25
    pred = model.A * np.power(1.0 + model.r, t_years)
    temp = df["close"].to_numpy() / pred
    return temp, model.A, model.r, start_ts


def target_allocation_from_temperature(T: float) -> float:
    # Piecewise-linear interpolation:
    if T <= 0.9:
        return 1.0
    if T >= 1.5:
        return 0.2
    if T <= 1.0:
        # from (0.9,1.0) to (1.0,0.8)
        return 1.0 - 2.0 * (T - 0.9)
    # from (1.0,0.8) to (1.5,0.2)
    return 0.8 - 1.2 * (T - 1.0)


def apply_rate_taper(p: float, rate_annual_pct: float) -> float:
    # Reduce allocation linearly to 0 by 12%. No change at 10%.
    if rate_annual_pct <= 10.0:
        return p
    if rate_annual_pct >= 12.0:
        return 0.0
    factor = (12.0 - rate_annual_pct) / 2.0  # 10%→1, 11%→0.5, 12%→0
    factor = max(0.0, min(1.0, factor))
    return p * factor


def apply_cold_leverage(
    p: float,
    T: float,
    rate_annual_pct: float,
    ret_22: float,
    *,
    boost: float = 0.2,
    temperature_threshold: float = 0.8,
    rate_threshold: float = 5.0,
    extra_boost: float = 0.0,
    extra_temperature_threshold: float = 0.75,
    extra_rate_threshold: float = 4.0,
    extra_ret22_threshold: float = 0.02,
) -> float:
    """Boost allocation when conditions look favorable.

    Parameters allow experiments to fine‑tune when and how much extra leverage
    is deployed. The default matches the original A2 behaviour of adding up to
    20% exposure (capped at 120%) when the market is cold but stable and rates
    are low. Experiments can optionally specify a second tier "extra" boost
    that further increases exposure when conditions are extremely favourable.
    """
    if (
        T < temperature_threshold
        and rate_annual_pct < rate_threshold
        and not math.isnan(ret_22)
        and ret_22 >= 0.0
    ):
        max_p = 1.0 + boost
        p = min(max_p, p + boost)
        if (
            extra_boost > 0.0
            and T < extra_temperature_threshold
            and rate_annual_pct < extra_rate_threshold
            and ret_22 >= extra_ret22_threshold
        ):
            max_p = 1.0 + boost + extra_boost
            p = min(max_p, p + extra_boost)
        return p
    return p


# Mean-reversion leverage helper
# -------------------------------
def apply_mean_reversion_kicker(
    p: float,
    T: float,
    rate_annual_pct: float,
    ret_3: float,
    ret_6: float,
    ret_22: float,
    *,
    temperature_threshold: float = 0.78,
    rate_threshold: float = 6.0,
    ret22_max: float = -0.04,
    ret3_min: float = 0.015,
    ret6_min: float = -0.01,
    boost: float = 0.25,
    cap: float = 1.6,
) -> float:
    """Add contrarian leverage after sharp drawdowns that start to rebound."""

    if math.isnan(ret_22) or math.isnan(ret_3) or math.isnan(ret_6):
        return p
    if (
        T < temperature_threshold
        and rate_annual_pct < rate_threshold
        and ret_22 <= ret22_max
        and ret_3 >= ret3_min
        and ret_6 >= ret6_min
    ):
        return min(cap, p + boost)
    return p


def apply_drawdown_turbo(
    p: float,
    drawdown: float,
    T: float,
    rate_annual_pct: float,
    ret_6: float,
    ret_12: float,
    *,
    drawdown_threshold: float = -0.25,
    temperature_ceiling: float = 1.05,
    rate_threshold: float = 6.5,
    ret6_min: float = 0.0,
    ret12_min: float = -0.02,
    boost: float = 0.35,
    cap: float = 1.9,
) -> float:
    """Increase exposure when recovering from deep strategy drawdowns."""

    if math.isnan(ret_6) or math.isnan(ret_12):
        return p
    if (
        drawdown <= drawdown_threshold
        and T < temperature_ceiling
        and rate_annual_pct < rate_threshold
        and ret_6 >= ret6_min
        and ret_12 >= ret12_min
    ):
        return min(cap, p + boost)
    return p


def apply_temperature_slope_boost(
    p: float,
    T: float,
    temp_ch_11: float,
    temp_ch_22: float,
    rate_annual_pct: float,
    ret_3: float,
    ret_6: float,
    *,
    temperature_ceiling: float = 1.05,
    rate_threshold: float = 6.0,
    temp_ch11_min: float = 0.03,
    temp_ch22_min: float = 0.08,
    ret3_min: float = 0.0,
    ret6_min: float = -0.01,
    boost: float = 0.25,
    extra_boost: float = 0.20,
    extra_temperature_ceiling: float = 0.95,
    extra_temp_ch11_min: float = 0.05,
    extra_temp_ch22_min: float = 0.12,
    cap: float = 2.0,
) -> float:
    """Add leverage when temperature is rising rapidly from discounted levels."""

    if math.isnan(temp_ch_11) or math.isnan(temp_ch_22) or math.isnan(ret_3) or math.isnan(ret_6):
        return p
    if (
        T < temperature_ceiling
        and rate_annual_pct < rate_threshold
        and temp_ch_11 >= temp_ch11_min
        and temp_ch_22 >= temp_ch22_min
        and ret_3 >= ret3_min
        and ret_6 >= ret6_min
    ):
        p = min(cap, p + boost)
        if (
            T < extra_temperature_ceiling
            and temp_ch_11 >= extra_temp_ch11_min
            and temp_ch_22 >= extra_temp_ch22_min
        ):
            p = min(cap, p + extra_boost)
    return p


def apply_temperature_curve_boost(
    base_p: float,
    T: float,
    *,
    pivot: float = 0.88,
    slope: float = 2.2,
    max_extra: float = 0.4,
    cap: float = 1.35,
) -> float:
    """Shift the baseline temperature curve to allow >100% exposure when very cold."""

    if T >= pivot:
        return min(cap, base_p)
    extra = slope * (pivot - T)
    extra = max(0.0, min(max_extra, extra))
    return min(cap, base_p + extra)


def apply_temperature_leverage_ladder(
    base_p: float,
    T: float,
    ret_6: float,
    ret_22: float,
    *,
    steps: list[dict[str, float]],
    cap: float = 2.0,
) -> float:
    """Ensure a minimum allocation when temperature falls below ladder steps."""

    updated = base_p
    for step in sorted(steps, key=lambda s: s.get("temp_max", 0.0)):
        temp_max = step.get("temp_max")
        min_alloc = step.get("min_allocation")
        ret6_min = step.get("ret6_min")
        ret22_min = step.get("ret22_min")
        if temp_max is None or min_alloc is None:
            continue
        meets = T <= temp_max
        if ret6_min is not None:
            meets = meets and (not math.isnan(ret_6) and ret_6 >= float(ret6_min))
        if ret22_min is not None:
            meets = meets and (not math.isnan(ret_22) and ret_22 >= float(ret22_min))
        if meets:
            updated = max(updated, float(min_alloc))
    return min(cap, updated)


def apply_rate_scaled_multiplier(
    p: float,
    rate_annual_pct: float,
    *,
    rate_floor: float = 2.5,
    rate_ceiling: float = 7.0,
    max_scale: float = 1.2,
    cap: float = 2.5,
) -> float:
    """Scale allocation up when financing rates are low."""

    if rate_annual_pct <= rate_floor:
        scale = max_scale
    elif rate_annual_pct >= rate_ceiling:
        scale = 1.0
    else:
        span = rate_ceiling - rate_floor
        scale = 1.0 + (max_scale - 1.0) * (rate_ceiling - rate_annual_pct) / span
    return min(cap, p * scale)


def apply_momentum_guard(
    p: float,
    ret_6: float,
    ret_22: float,
    *,
    ret6_floor: float = -0.08,
    ret22_floor: float = -0.15,
    scale: float = 0.55,
    floor: float = 0.6,
) -> float:
    """Dial back leverage if intermediate momentum is severely negative."""

    triggers = []
    if not math.isnan(ret_6) and ret_6 <= ret6_floor:
        triggers.append(True)
    if not math.isnan(ret_22) and ret_22 <= ret22_floor:
        triggers.append(True)
    if triggers:
        return max(floor, p * scale)
    return p


def apply_drawdown_guard(
    p: float,
    drawdown: float,
    ret_6: float,
    ret_22: float,
    *,
    drawdown_threshold: float = -0.35,
    ret6_max: float = -0.02,
    ret22_max: float = -0.05,
    cap: float = 1.2,
) -> float:
    """Cap exposure during deep drawdowns until momentum improves."""

    if drawdown <= drawdown_threshold:
        cond6 = math.isnan(ret_6) or ret_6 <= ret6_max
        cond22 = math.isnan(ret_22) or ret_22 <= ret22_max
        if cond6 and cond22:
            return min(cap, p)
    return p


def apply_volatility_squeeze_boost(
    p: float,
    vol_22: float,
    vol_66: float,
    ret_6: float,
    ret_22: float,
    *,
    vol22_threshold: float = 0.018,
    vol_ratio_max: float = 0.75,
    ret6_min: float = 0.02,
    ret22_min: float = 0.0,
    boost: float = 0.25,
    cap: float = 2.1,
) -> float:
    """Boost exposure when volatility compresses while momentum turns positive."""

    if math.isnan(vol_22) or math.isnan(vol_66) or math.isnan(ret_6) or math.isnan(ret_22):
        return p
    if vol_22 <= 0:
        return p
    ratio = vol_22 / vol_66 if vol_66 > 0 else float("inf")
    if (
        vol_22 <= vol22_threshold
        and ratio <= vol_ratio_max
        and ret_6 >= ret6_min
        and ret_22 >= ret22_min
    ):
        return min(cap, p + boost)
    return p


def apply_macro_tailwind_boost(
    p: float,
    yc_spread: float,
    credit_spread: float,
    yc_change_22: float,
    credit_change_22: float,
    rate_annual_pct: float,
    *,
    yc_min: float = -0.4,
    credit_max: float = 2.5,
    yc_change_min: float = 0.2,
    credit_change_max: float = -0.1,
    rate_threshold: float = 6.5,
    boost: float = 0.3,
    cap: float = 2.3,
) -> float:
    """Boost exposure when macro indicators signal an improving backdrop."""

    if (
        math.isnan(yc_spread)
        or math.isnan(credit_spread)
        or math.isnan(yc_change_22)
        or math.isnan(credit_change_22)
    ):
        return p
    if (
        yc_spread >= yc_min
        and credit_spread <= credit_max
        and yc_change_22 >= yc_change_min
        and credit_change_22 <= credit_change_max
        and rate_annual_pct < rate_threshold
    ):
        return min(cap, p + boost)
    return p


def apply_recovery_hyperdrive(
    p: float,
    *,
    drawdown: float,
    T: float,
    ret_6: float,
    ret_22: float,
    temp_ch11: float,
    vol_22: float,
    vol_66: float,
    rate_annual_pct: float,
    yc_spread: float,
    credit_spread: float,
    yc_change_22: float,
    credit_change_22: float,
    drawdown_threshold: float = -0.32,
    temperature_max: float = 0.9,
    ret6_min: float = 0.015,
    ret22_min: float = -0.08,
    temp_ch11_min: float = 0.025,
    vol22_threshold: float = 0.02,
    vol_ratio_max: float = 0.85,
    rate_threshold: float = 6.0,
    yc_min: float = -0.4,
    credit_max: float = 2.6,
    yc_change_min: float = 0.1,
    credit_change_max: float = -0.05,
    target: float = 2.3,
) -> float:
    """Force exposure up to a high watermark when multiple recovery signals align."""

    if any(
        math.isnan(val)
        for val in (
            drawdown,
            T,
            ret_6,
            ret_22,
            temp_ch11,
            vol_22,
            vol_66,
            rate_annual_pct,
            yc_spread,
            credit_spread,
            yc_change_22,
            credit_change_22,
        )
    ):
        return p
    if vol_22 <= 0 or vol_66 <= 0:
        return p
    vol_ratio = vol_22 / vol_66
    if (
        drawdown <= drawdown_threshold
        and T <= temperature_max
        and ret_6 >= ret6_min
        and ret_22 >= ret22_min
        and temp_ch11 >= temp_ch11_min
        and vol_22 <= vol22_threshold
        and vol_ratio <= vol_ratio_max
        and rate_annual_pct < rate_threshold
        and yc_spread >= yc_min
        and credit_spread <= credit_max
        and yc_change_22 >= yc_change_min
        and credit_change_22 <= credit_change_max
    ):
        return max(p, target)
    return p


# Momentum leverage helpers
# -------------------------
def apply_hot_momentum_leverage(
    p: float,
    T: float,
    rate_annual_pct: float,
    ret_3: float,
    ret_6: float,
    ret_12: float,
    ret_22: float,
    *,
    boost: float = 0.2,
    temperature_max: float = 1.15,
    rate_threshold: float = 5.0,
    ret22_threshold: float = 0.05,
    ret3_threshold: float = 0.0,
    ret6_threshold: float = 0.0,
    ret12_threshold: float = 0.0,
    extra_boost: float = 0.2,
    extra_temperature_max: float = 1.05,
    extra_rate_threshold: float = 4.0,
    extra_ret22_threshold: float = 0.10,
    cap: float = 1.6,
    extra_cap: float = 1.8,
) -> float:
    """Temporarily boost exposure during strong uptrends."""

    if (
        T < temperature_max
        and rate_annual_pct < rate_threshold
        and not math.isnan(ret_22)
        and ret_22 >= ret22_threshold
        and ret_3 >= ret3_threshold
        and ret_6 >= ret6_threshold
        and ret_12 >= ret12_threshold
    ):
        p = min(cap, p + boost)
        if (
            extra_boost > 0.0
            and T < extra_temperature_max
            and rate_annual_pct < extra_rate_threshold
            and ret_22 >= extra_ret22_threshold
        ):
            p = min(extra_cap, p + extra_boost)
    return p


def apply_momentum_stop(
    p: float,
    ret_3: float,
    ret_12: float,
    *,
    stop_ret12: float = -0.02,
    stop_ret3: float = -0.01,
    base_p: float,
) -> float:
    """Revert to baseline allocation if momentum fades."""

    if ret_12 <= stop_ret12 or ret_3 <= stop_ret3:
        return base_p
    return p


def apply_macro_filter(
    p: float,
    yc_spread: float,
    credit_spread: float,
    *,
    yc_threshold: float = 0.0,
    credit_threshold: float = 2.0,
    reduce_to: float = 0.0,
    require_both: bool = True,
) -> float:
    """Reduce allocation when macro indicators signal elevated risk.

    If ``require_both`` is True, both the yield-curve spread and the credit
    spread must breach their respective thresholds to trigger the reduction.
    Otherwise either condition alone will trigger it.
    """

    yc_bad = not math.isnan(yc_spread) and yc_spread <= yc_threshold
    credit_bad = not math.isnan(credit_spread) and credit_spread >= credit_threshold
    triggered = (yc_bad and credit_bad) if require_both else (yc_bad or credit_bad)
    if triggered:
        return min(p, reduce_to)
    return p


def apply_volatility_adjustment(
    p: float,
    vol_22: float,
    ret_22: float,
    *,
    low_threshold: float = 0.012,
    high_threshold: float = 0.025,
    low_boost: float = 0.15,
    high_cut: float = 0.10,
    low_ret_threshold: float = 0.0,
    high_ret_threshold: float = 0.0,
    cap: float = 1.6,
    floor: float = 0.0,
) -> float:
    """Adjust allocation based on recent realized volatility.

    When volatility is very low and returns are non-negative, lean into the
    trend by adding exposure. When volatility is elevated and returns have
    turned negative, pull back towards cash.
    """

    if math.isnan(vol_22):
        return p
    if (
        vol_22 <= low_threshold
        and not math.isnan(ret_22)
        and ret_22 >= low_ret_threshold
    ):
        return min(cap, p + low_boost)
    if (
        vol_22 >= high_threshold
        and not math.isnan(ret_22)
        and ret_22 <= high_ret_threshold
    ):
        return max(floor, p - high_cut)
    return p


# Strategy parameters
# -------------------
# cold_leverage:
#   boost: fraction of extra exposure to add when the rule triggers
#   temperature_threshold: only boost when Nasdaq temperature is below this value
#   rate_threshold: only boost when the annual interest rate is below this percentage
#   extra_boost: additional exposure when deep-cold conditions are met
#   extra_temperature_threshold: temperature must be below this for extra boost
#   extra_rate_threshold: rate must be below this for extra boost
#   extra_ret22_threshold: 22-day return must be at least this for extra boost
# hot_momentum_leverage:
#   boost: exposure added when strong positive momentum is present
#   temperature_max: only apply boost if temperature is below this ceiling
#   rate_threshold: only apply boost if rates are below this percentage
#   ret22_threshold/ret3_threshold/ret6_threshold/ret12_threshold: minimum recent returns
#   extra_boost: additional boost under even stronger conditions
#   extra_temperature_max: tighter temperature ceiling for extra boost
#   extra_rate_threshold: tighter rate ceiling for extra boost
#   extra_ret22_threshold: higher 22-day return needed for extra boost
#   cap/extra_cap: overall exposure caps for first and second boost
#   stop_ret12/stop_ret3: revert to baseline if momentum drops below these
# macro_filter:
#   yc_threshold: trigger when yield-curve spread falls below this
#   credit_threshold: trigger when credit spread rises above this
#   reduce_to: allocation fraction to reduce to when triggered
#   require_both: if True, both conditions must trigger; otherwise either

# Strategy experiment definitions. Each experiment can enable features or tune
# parameters to explore different behaviours without littering conditional
# logic throughout the simulation.
EXPERIMENTS = {
    "A1": {},  # Baseline strategy with no extra features
    "A2": {
        "cold_leverage": {
            "boost": 0.2,
            "temperature_threshold": 0.8,
            "rate_threshold": 5.0,
        }
    },
    "A3": {
        "cold_leverage": {
            "boost": 0.2,
            "temperature_threshold": 0.8,
            "rate_threshold": 5.0,
            "extra_boost": 0.2,
            "extra_temperature_threshold": 0.75,
            "extra_rate_threshold": 4.0,
            "extra_ret22_threshold": 0.02,
        }
    },
    "A4": {
        "cold_leverage": {
            "boost": 0.2,
            "temperature_threshold": 0.8,
            "rate_threshold": 5.0,
            "extra_boost": 0.2,
            "extra_temperature_threshold": 0.75,
            "extra_rate_threshold": 4.0,
            "extra_ret22_threshold": 0.02,
        },
        "hot_momentum_leverage": {
            "boost": 0.2,
            "temperature_max": 1.15,
            "rate_threshold": 5.0,
            "ret22_threshold": 0.05,
            "ret3_threshold": 0.0,
            "ret6_threshold": 0.0,
            "ret12_threshold": 0.0,
            "extra_boost": 0.2,
            "extra_temperature_max": 1.05,
            "extra_rate_threshold": 4.0,
            "extra_ret22_threshold": 0.10,
            "stop_ret12": -0.02,
            "stop_ret3": -0.01,
        },
    },
}

# A5 builds on A4 with a macro risk-off filter
EXPERIMENTS["A5"] = {
    **EXPERIMENTS["A4"],
    "macro_filter": {
        "yc_threshold": -0.2,
        "credit_threshold": 2.15,
        "reduce_to": 0.0,
        "require_both": True,
    },
}

# A6 adds a realized-volatility tilt: lean in when calm, pull back when stormy
EXPERIMENTS["A6"] = {
    **EXPERIMENTS["A5"],
    "volatility_adjustment": {
        "low_threshold": 0.013,
        "low_ret_threshold": 0.0,
        "low_boost": 0.25,
        "high_threshold": 0.05,
        "high_ret_threshold": -0.10,
        "high_cut": 0.0,
        "cap": 1.85,
    },
}

# A7 enables faster re-entries when allocation drifts too far from target
EXPERIMENTS["A7"] = {
    **EXPERIMENTS["A5"],
    "intra_cycle_rebalance": {
        "threshold": 0.12,
        "min_gap_days": 8,
        "direction": "buy",
    },
}

# A8 loosens buy blocks when the market is extremely cold
EXPERIMENTS["A8"] = {
    **EXPERIMENTS["A7"],
    "buy_block_relax": {
        "temp_threshold": 0.84,
        "r3_limit": -0.04,
        "r6_limit": -0.04,
        "r12_limit": -0.025,
        "r22_limit": -0.005,
        "temp_limit_cold": 1.34,
        "temp_limit_default": 1.3,
    },
}

# A9 keeps some exposure during macro risk-off regimes instead of going fully to cash
EXPERIMENTS["A9"] = {
    **EXPERIMENTS["A7"],
    "macro_filter": {
        "yc_threshold": -0.2,
        "credit_threshold": 2.15,
        "reduce_to": 0.4,
        "require_both": True,
    },
}

# A10 increases the momentum overlay aggressiveness to chase strong rallies harder
EXPERIMENTS["A10"] = {
    **EXPERIMENTS["A7"],
    "hot_momentum_leverage": {
        "boost": 0.25,
        "temperature_max": 1.18,
        "rate_threshold": 5.5,
        "ret22_threshold": 0.04,
        "ret3_threshold": -0.01,
        "ret6_threshold": 0.0,
        "ret12_threshold": 0.0,
        "extra_boost": 0.25,
        "extra_temperature_max": 1.08,
        "extra_rate_threshold": 5.0,
        "extra_ret22_threshold": 0.08,
        "stop_ret12": -0.015,
        "stop_ret3": -0.005,
        "cap": 1.9,
        "extra_cap": 2.05,
    },
}

# A11 deploys more leverage in deep cold regimes with slightly looser requirements
EXPERIMENTS["A11"] = {
    **EXPERIMENTS["A7"],
    "cold_leverage": {
        "boost": 0.30,
        "temperature_threshold": 0.85,
        "rate_threshold": 6.0,
        "extra_boost": 0.40,
        "extra_temperature_threshold": 0.75,
        "extra_rate_threshold": 5.0,
        "extra_ret22_threshold": -0.03,
    },
    "intra_cycle_rebalance": {
        "threshold": 0.11,
        "min_gap_days": 6,
        "direction": "buy",
    },
}

# A12 adds a mean-reversion kicker after deep drawdowns begin to rebound
EXPERIMENTS["A12"] = {
    **EXPERIMENTS["A11"],
    "mean_reversion_kicker": {
        "temperature_threshold": 0.78,
        "rate_threshold": 6.0,
        "ret22_max": -0.05,
        "ret3_min": 0.02,
        "ret6_min": -0.005,
        "boost": 0.30,
        "cap": 1.7,
    },
}

# A13 introduces a drawdown-aware turbo to lean in during recoveries
EXPERIMENTS["A13"] = {
    **EXPERIMENTS["A12"],
    "drawdown_turbo": {
        "drawdown_threshold": -0.28,
        "temperature_ceiling": 1.03,
        "rate_threshold": 6.5,
        "ret6_min": 0.01,
        "ret12_min": -0.01,
        "boost": 0.35,
        "cap": 1.85,
    },
}

# A14 captures steep temperature recoveries with a slope-based booster
EXPERIMENTS["A14"] = {
    **EXPERIMENTS["A13"],
    "temperature_slope_boost": {
        "temperature_ceiling": 1.02,
        "rate_threshold": 6.0,
        "temp_ch11_min": 0.035,
        "temp_ch22_min": 0.09,
        "ret3_min": 0.0,
        "ret6_min": 0.005,
        "boost": 0.30,
        "extra_boost": 0.30,
        "extra_temperature_ceiling": 0.92,
        "extra_temp_ch11_min": 0.06,
        "extra_temp_ch22_min": 0.14,
        "cap": 2.05,
    },
}

# A15 shifts the temperature curve itself to deploy more in deep discounts
EXPERIMENTS["A15"] = {
    **EXPERIMENTS["A14"],
    "temperature_curve_boost": {
        "pivot": 0.88,
        "slope": 2.4,
        "max_extra": 0.45,
        "cap": 1.45,
    },
}

# A16 layers in a volatility-squeeze boost to lean in when calm after storms
EXPERIMENTS["A16"] = {
    **EXPERIMENTS["A15"],
    "volatility_squeeze_boost": {
        "vol22_threshold": 0.017,
        "vol_ratio_max": 0.7,
        "ret6_min": 0.015,
        "ret22_min": -0.005,
        "boost": 0.28,
        "cap": 2.15,
    },
}

# A17 rewards improving macro backdrops with an additional tailwind boost
EXPERIMENTS["A17"] = {
    **EXPERIMENTS["A16"],
    "macro_tailwind_boost": {
        "yc_min": -0.35,
        "credit_max": 2.4,
        "yc_change_min": 0.18,
        "credit_change_max": -0.08,
        "rate_threshold": 6.2,
        "boost": 0.32,
        "cap": 2.25,
    },
}

# A18 introduces a composite hyperdrive when every recovery signal aligns
EXPERIMENTS["A18"] = {
    **EXPERIMENTS["A17"],
    "recovery_hyperdrive": {
        "drawdown_threshold": -0.34,
        "temperature_max": 0.88,
        "ret6_min": 0.02,
        "ret22_min": -0.06,
        "temp_ch11_min": 0.03,
        "vol22_threshold": 0.019,
        "vol_ratio_max": 0.8,
        "rate_threshold": 6.0,
        "yc_min": -0.35,
        "credit_max": 2.4,
        "yc_change_min": 0.16,
        "credit_change_max": -0.07,
        "target": 2.35,
    },
}

# A19 installs a cold-temperature leverage ladder to guarantee higher baselines
EXPERIMENTS["A19"] = {
    **EXPERIMENTS["A18"],
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.2},
            {"temp_max": 0.82, "min_allocation": 1.5},
            {"temp_max": 0.78, "min_allocation": 1.8},
            {"temp_max": 0.74, "min_allocation": 2.1},
            {"temp_max": 0.70, "min_allocation": 2.3},
        ],
        "cap": 2.35,
    },
}

# A20 scales the ladder exposure when money is cheap
EXPERIMENTS["A20"] = {
    **EXPERIMENTS["A19"],
    "rate_scaled_leverage": {
        "rate_floor": 2.5,
        "rate_ceiling": 6.5,
        "max_scale": 1.18,
        "cap": 2.45,
    },
}

# A21 adds a momentum guard so high leverage waits for trend confirmation
EXPERIMENTS["A21"] = {
    **EXPERIMENTS["A20"],
    "momentum_guard": {
        "ret6_floor": -0.06,
        "ret22_floor": -0.12,
        "scale": 0.45,
        "floor": 0.7,
    },
}

# A22 applies a global allocation cap to tame extreme stack-ups
EXPERIMENTS["A22"] = {
    **EXPERIMENTS["A19"],
    "global_allocation_cap": 2.45,
}

# A23 gates the ladder steps on recovering momentum to avoid crash-time overexposure
EXPERIMENTS["A23"] = {
    **EXPERIMENTS["A22"],
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.2, "ret22_min": -0.08},
            {"temp_max": 0.82, "min_allocation": 1.5, "ret6_min": 0.0, "ret22_min": -0.05},
            {"temp_max": 0.78, "min_allocation": 1.8, "ret6_min": 0.01, "ret22_min": -0.02},
            {"temp_max": 0.74, "min_allocation": 2.05, "ret6_min": 0.015, "ret22_min": 0.0},
            {"temp_max": 0.70, "min_allocation": 2.3, "ret6_min": 0.02, "ret22_min": 0.02},
        ],
        "cap": 2.35,
    },
    "global_allocation_cap": 2.4,
}

# A24 adds a drawdown guard that reins in leverage until momentum mends
EXPERIMENTS["A24"] = {
    **EXPERIMENTS["A23"],
    "drawdown_guard": {
        "drawdown_threshold": -0.34,
        "ret6_max": -0.01,
        "ret22_max": -0.04,
        "cap": 1.45,
    },
}

# A25 retools the hot momentum overlay to press harder during broad rallies
EXPERIMENTS["A25"] = {
    **EXPERIMENTS["A22"],
    "hot_momentum_leverage": {
        "boost": 0.35,
        "temperature_max": 1.12,
        "rate_threshold": 6.0,
        "ret22_threshold": 0.06,
        "ret3_threshold": 0.01,
        "ret6_threshold": 0.015,
        "ret12_threshold": 0.02,
        "extra_boost": 0.25,
        "extra_temperature_max": 1.02,
        "extra_rate_threshold": 5.5,
        "extra_ret22_threshold": 0.09,
        "stop_ret12": -0.01,
        "stop_ret3": -0.005,
        "cap": 2.3,
        "extra_cap": 2.45,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Simulate TQQQ+reserve strategy with temperature & filters")
    parser.add_argument("--csv", default="unified_nasdaq.csv", help="Unified dataset CSV path")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD) inclusive")
    parser.add_argument("--end", default="2025-01-10", help="End date (YYYY-MM-DD) inclusive")
    parser.add_argument("--fred-series", default="FEDFUNDS", help="FRED rate series (default FEDFUNDS)")
    parser.add_argument(
        "--experiment",
        default="A25",
        choices=sorted(EXPERIMENTS.keys()),
        help="Strategy experiment to run (default A25, best-known)"
    )
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--annual-fee", type=float, default=0.0095)
    parser.add_argument("--borrow-divisor", type=float, default=0.7)
    parser.add_argument("--trading-days", type=int, default=252)
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1.0,
        help="Initial portfolio value for the simulation (default 1.0)",
    )
    parser.add_argument(
        "--print-rebalances",
        action="store_true",
        help="Print a detailed log of executed rebalances",
    )
    parser.add_argument("--save-plot", default=None, help="If set, save output figure to this PNG path")
    parser.add_argument("--save-csv", default=None, help="If set, save daily debug CSV here")
    # Debug mode for rebalance decisions
    parser.add_argument("--debug-start", default=None, help="Start date (YYYY-MM-DD) for rebalance debug output")
    parser.add_argument("--debug-end", default=None, help="End date (YYYY-MM-DD) for rebalance debug output (inclusive)")
    parser.add_argument("--debug-csv", default=None, help="If set, write rebalance debug CSV to this path (default strategy_tqqq_reserve_debug.csv if a debug range is provided)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment.upper()
    config = EXPERIMENTS[experiment]

    pd, np, plt = import_libs()
    df = load_unified(pd, args.csv)
    if args.start:
        start_filter_ts = pd.to_datetime(args.start)
        df = df.loc[df.index >= start_filter_ts]
    end_ts = pd.to_datetime(args.end)
    df = df.loc[df.index <= end_ts]
    if df.empty:
        raise RuntimeError("No data available in the requested date range")
    start_ts = df.index.min()

    # Daily returns for unified (QQQ proxy)
    df["ret"] = df["close"].pct_change().fillna(0.0)
    # Momentum windows use change from t-(N-1) to t
    df["ret_3"] = df["close"] / df["close"].shift(2) - 1.0
    df["ret_6"] = df["close"] / df["close"].shift(5) - 1.0
    df["ret_12"] = df["close"] / df["close"].shift(11) - 1.0
    df["ret_22"] = df["close"] / df["close"].shift(21) - 1.0
    df["vol_22"] = df["ret"].rolling(window=22).std()
    df["vol_66"] = df["ret"].rolling(window=66).std()

    # Rates
    rates = fetch_fred_series(pd, args.fred_series, start_ts, end_ts)
    rates = rates.rename(columns={rates.columns[0]: "rate"}) if rates.shape[1] == 1 else rates
    rates.index = pd.to_datetime(rates.index)
    rates = rates.sort_index()
    rates = rates.reindex(df.index).ffill().bfill()

    # Macro indicators: yield-curve spread (T10Y2Y) and credit spread (BAA10Y)
    yc = fetch_fred_series(pd, "T10Y2Y", start_ts, end_ts)
    yc.index = pd.to_datetime(yc.index)
    yc = yc.sort_index()
    cs = fetch_fred_series(pd, "BAA10Y", start_ts, end_ts)
    cs.index = pd.to_datetime(cs.index)
    cs = cs.sort_index()
    macro = pd.DataFrame({
        "yc_spread": yc["rate"],
        "credit_spread": cs["rate"],
    })
    macro = macro.reindex(df.index).ffill().bfill()
    macro["yc_change_22"] = macro["yc_spread"].diff(22)
    macro["credit_change_22"] = macro["credit_spread"].diff(22)

    # Temperature series
    temp, A, r, fit_start_ts = compute_temperature_series(pd, np, df, csv_path=args.csv)
    df["temp"] = temp
    df["temp_ch_11"] = df["temp"] / df["temp"].shift(11) - 1.0
    df["temp_ch_22"] = df["temp"] / df["temp"].shift(21) - 1.0

    # Simulated TQQQ-like and cash daily factors
    leverage = float(args.leverage)
    borrowed_fraction = leverage - 1.0
    daily_fee = float(args.annual_fee) / float(args.trading_days)

    rate_ann = rates["rate"].to_numpy()
    rets = df["ret"].to_numpy()
    cash_factor = 1.0 + (rate_ann / 100.0) / float(args.trading_days)
    daily_borrow = borrowed_fraction * ((rate_ann / 100.0) / float(args.trading_days))
    tqqq_factor = (1.0 + leverage * rets) * (1.0 - daily_fee) * (1.0 - (daily_borrow / float(args.borrow_divisor)))

    # Simulation state
    dates = df.index.to_numpy()
    num_days = len(df)
    port_tqqq = np.zeros(num_days, dtype=float)
    port_cash = np.zeros(num_days, dtype=float)
    port_total = np.zeros(num_days, dtype=float)
    deployed_p = np.zeros(num_days, dtype=float)
    base_p_series = np.zeros(num_days, dtype=float)
    target_p_series = np.zeros(num_days, dtype=float)
    block_buy_series = np.zeros(num_days, dtype=int)
    block_sell_series = np.zeros(num_days, dtype=int)
    forced_derisk_series = np.zeros(num_days, dtype=int)

    # Debug collection for rebalance and crash-rule decisions
    debug_rows = []
    debug_start_ts = pd.to_datetime(args.debug_start) if args.debug_start else None
    debug_end_ts = pd.to_datetime(args.debug_end) if args.debug_end else None
    if debug_start_ts is not None and debug_end_ts is None:
        debug_end_ts = debug_start_ts

    # Start with all in cash
    initial_capital = float(args.initial_capital)
    port_tqqq[0] = 0.0
    port_cash[0] = initial_capital
    port_total[0] = port_cash[0] + port_tqqq[0]
    deployed_p[0] = 0.0

    # Rebalance scheduling
    next_rebalance_idx = 0  # day 0 is an eligible rebalance day after updates
    eps = 1e-6
    last_rebalance_idx = -22
    peak_total = port_total[0]

    rebalance_entries = []

    def add_rebalance_entry(
        index: int,
        new_tqqq: float,
        new_cash: float,
        prev_tqqq: float,
        prev_cash: float,
        *,
        reason: str = "rebalance",
    ) -> None:
        trade_eps = 1e-9
        delta_tqqq = float(new_tqqq - prev_tqqq)
        delta_cash = float(new_cash - prev_cash)
        if abs(delta_tqqq) < trade_eps and abs(delta_cash) < trade_eps:
            return
        total_after = float(new_tqqq + new_cash)
        p_tqqq = 0.0 if total_after <= 0 else float(new_tqqq / total_after)
        entry_date = pd.Timestamp(dates[index])
        rebalance_entries.append(
            {
                "index": int(index),
                "date": entry_date,
                "p_tqqq": p_tqqq,
                "p_cash": 1.0 - p_tqqq,
                "delta_tqqq": delta_tqqq,
                "delta_cash": delta_cash,
                "total_after": total_after,
                "tqqq_value": float(new_tqqq),
                "cash_value": float(new_cash),
                "reason": reason,
            }
        )

    for i in range(num_days):
        # Update holdings based on previous day growth (for i=0 this applies one day of cash growth)
        if i > 0:
            base_factor = tqqq_factor[i]
            port_tqqq[i] = port_tqqq[i - 1] * base_factor
            port_cash[i] = port_cash[i - 1] * cash_factor[i]
        else:
            port_tqqq[i] = port_tqqq[0] * tqqq_factor[0]
            port_cash[i] = port_cash[0] * cash_factor[0]

        total = port_tqqq[i] + port_cash[i]
        curr_p = 0.0 if total <= 0 else port_tqqq[i] / total
        drawdown = 0.0 if peak_total <= 0 else (total / peak_total) - 1.0

        T = df["temp"].iloc[i]
        rate_today = float(rate_ann[i])
        base_p = target_allocation_from_temperature(T)
        tcurve_cfg = config.get("temperature_curve_boost")
        if tcurve_cfg:
            base_p = apply_temperature_curve_boost(base_p, T, **tcurve_cfg)
        rate_scale_cfg = config.get("rate_scaled_leverage")
        if rate_scale_cfg:
            base_p = apply_rate_scaled_multiplier(base_p, rate_today, **rate_scale_cfg)
        r3 = df["ret_3"].iloc[i]
        r6 = df["ret_6"].iloc[i]
        r12 = df["ret_12"].iloc[i]
        r22 = df["ret_22"].iloc[i]
        ladder_cfg = config.get("temperature_leverage_ladder")
        if ladder_cfg:
            base_p = apply_temperature_leverage_ladder(base_p, T, r6, r22, **ladder_cfg)
        temp_ch11 = df["temp_ch_11"].iloc[i]
        temp_ch22 = df["temp_ch_22"].iloc[i]
        vol22 = df["vol_22"].iloc[i]
        vol66 = df["vol_66"].iloc[i]
        yc_today = macro["yc_spread"].iloc[i]
        cs_today = macro["credit_spread"].iloc[i]
        yc_change22 = macro["yc_change_22"].iloc[i]
        cs_change22 = macro["credit_change_22"].iloc[i]
        guard_cfg = config.get("momentum_guard")
        if guard_cfg:
            base_p = apply_momentum_guard(base_p, r6, r22, **guard_cfg)
        target_p = apply_rate_taper(base_p, rate_today)
        dd_guard_cfg = config.get("drawdown_guard")
        if dd_guard_cfg:
            target_p = apply_drawdown_guard(target_p, drawdown, r6, r22, **dd_guard_cfg)
        cl_cfg = config.get("cold_leverage")
        if cl_cfg:
            target_p = apply_cold_leverage(target_p, T, rate_today, r22, **cl_cfg)
        mrk_cfg = config.get("mean_reversion_kicker")
        if mrk_cfg:
            target_p = apply_mean_reversion_kicker(
                target_p,
                T,
                rate_today,
                r3,
                r6,
                r22,
                **mrk_cfg,
            )
        dd_cfg = config.get("drawdown_turbo")
        if dd_cfg:
            target_p = apply_drawdown_turbo(
                target_p,
                drawdown,
                T,
                rate_today,
                r6,
                r12,
                **dd_cfg,
            )
        tsb_cfg = config.get("temperature_slope_boost")
        if tsb_cfg:
            target_p = apply_temperature_slope_boost(
                target_p,
                T,
                temp_ch11,
                temp_ch22,
                rate_today,
                r3,
                r6,
                **tsb_cfg,
            )
        vs_cfg = config.get("volatility_squeeze_boost")
        if vs_cfg:
            target_p = apply_volatility_squeeze_boost(
                target_p,
                vol22,
                vol66,
                r6,
                r22,
                **vs_cfg,
            )
        vol_cfg = config.get("volatility_adjustment")
        if vol_cfg:
            target_p = apply_volatility_adjustment(target_p, vol22, r22, **vol_cfg)
        target_p_base = target_p
        hml_cfg = config.get("hot_momentum_leverage")
        if hml_cfg:
            hml_main = {k: v for k, v in hml_cfg.items() if k not in ("stop_ret12", "stop_ret3")}
            target_p = apply_hot_momentum_leverage(
                target_p, T, rate_today, r3, r6, r12, r22, **hml_main
            )
            stop_cfg = {k: hml_cfg[k] for k in ("stop_ret12", "stop_ret3") if k in hml_cfg}
            target_p = apply_momentum_stop(
                target_p, r3, r12, base_p=target_p_base, **stop_cfg
            )
        macro_cfg = config.get("macro_filter")
        if macro_cfg:
            target_p = apply_macro_filter(target_p, yc_today, cs_today, **macro_cfg)
        macro_tail_cfg = config.get("macro_tailwind_boost")
        if macro_tail_cfg:
            target_p = apply_macro_tailwind_boost(
                target_p,
                yc_today,
                cs_today,
                yc_change22,
                cs_change22,
                rate_today,
                **macro_tail_cfg,
            )
        hyper_cfg = config.get("recovery_hyperdrive")
        if hyper_cfg:
            target_p = apply_recovery_hyperdrive(
                target_p,
                drawdown=drawdown,
                T=T,
                ret_6=r6,
                ret_22=r22,
                temp_ch11=temp_ch11,
                vol_22=vol22,
                vol_66=vol66,
                rate_annual_pct=rate_today,
                yc_spread=yc_today,
                credit_spread=cs_today,
                yc_change_22=yc_change22,
                credit_change_22=cs_change22,
                **hyper_cfg,
            )
        global_cap = config.get("global_allocation_cap")
        if global_cap is not None:
            target_p = min(float(global_cap), target_p)
        base_p_series[i] = base_p
        target_p_series[i] = target_p

        intra_cfg = config.get("intra_cycle_rebalance")
        if (
            intra_cfg
            and i < next_rebalance_idx
            and i >= last_rebalance_idx + int(intra_cfg.get("min_gap_days", 5))
        ):
            threshold = float(intra_cfg.get("threshold", 0.15))
            direction = intra_cfg.get("direction", "both").lower()
            diff = target_p - curr_p
            trigger_buy = direction in ("both", "buy") and diff >= threshold
            trigger_sell = direction in ("both", "sell") and diff <= -threshold
            if trigger_buy or trigger_sell:
                next_rebalance_idx = i

        # (Moved crash de-risk to rebalance-due section to enforce cadence)

        # Rebalance logic
        acted = False
        if i >= next_rebalance_idx:
            # First, check the 22d crash de-risk. If triggered, act immediately and skip momentum filters.
            ret22 = r22
            forced_today = False
            if not math.isnan(ret22) and ret22 <= -0.1525 and port_tqqq[i] > 0:
                prev_tqqq = port_tqqq[i]
                prev_cash = port_cash[i]
                # Capture debug BEFORE action
                if debug_start_ts is not None:
                    ts_i = dates[i]
                    if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                        prev_close_22 = df["close"].shift(21).iloc[i] if i >= 21 else float('nan')
                        prev_date_22 = pd.to_datetime(dates[i - 21]).date() if i >= 21 else None
                        debug_rows.append({
                            "date": pd.to_datetime(ts_i).date(),
                            "event": "forced_derisk",
                            "price": float(df["close"].iloc[i]),
                            "ref_date_22d": prev_date_22,
                            "price_22d_ago": float(prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_abs": float(df["close"].iloc[i] - prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_rel": float(ret22),
                            "sell_all_threshold": -0.1525,
                            "sell_all_ratio": (float(ret22) / -0.1525) if -0.1525 != 0 else float('nan'),
                            "temp_T": float(df["temp"].iloc[i]) if not math.isnan(df["temp"].iloc[i]) else float('nan'),
                            "rate_annual_pct": float(rate_ann[i]),
                            "curr_p_before": float(0.0 if (port_tqqq[i] + port_cash[i]) <= 0 else port_tqqq[i] / (port_tqqq[i] + port_cash[i])),
                            "action": "sell_all_to_cash"
                        })
                port_cash[i] = total
                port_tqqq[i] = 0.0
                total = port_tqqq[i] + port_cash[i]
                curr_p = 0.0
                next_rebalance_idx = i + 22
                forced_derisk_series[i] = 1
                acted = True
                forced_today = True
                last_rebalance_idx = i
                add_rebalance_entry(i, port_tqqq[i], port_cash[i], prev_tqqq, prev_cash, reason="forced_derisk")
                # Update debug AFTER action
                if debug_start_ts is not None:
                    ts_i = dates[i]
                    if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                        debug_rows[-1]["curr_p_after"] = float(curr_p)
            if forced_today:
                port_total[i] = port_tqqq[i] + port_cash[i]
                deployed_p[i] = curr_p
                continue
            # Determine direction
            if target_p > curr_p + eps:
                # Buy-side filters
                block_buy = False
                # Thresholds can be relaxed when the market is very cold
                buy_relax_cfg = config.get("buy_block_relax")
                limit_r3 = -0.03
                limit_r6 = -0.03
                limit_r12 = -0.02
                limit_r22 = 0.0
                temp_limit = 1.3
                if buy_relax_cfg:
                    temp_threshold_relax = float(buy_relax_cfg.get("temp_threshold", 0.85))
                    if T <= temp_threshold_relax:
                        limit_r3 = float(buy_relax_cfg.get("r3_limit", limit_r3))
                        limit_r6 = float(buy_relax_cfg.get("r6_limit", limit_r6))
                        limit_r12 = float(buy_relax_cfg.get("r12_limit", limit_r12))
                        limit_r22 = float(buy_relax_cfg.get("r22_limit", limit_r22))
                        temp_limit = float(buy_relax_cfg.get("temp_limit_cold", temp_limit))
                    else:
                        temp_limit = float(buy_relax_cfg.get("temp_limit_default", temp_limit))
                # r3, r6, r12, r22 already computed above
                trig_buy_r3 = (not math.isnan(r3) and r3 <= limit_r3)
                trig_buy_r6 = (not math.isnan(r6) and r6 <= limit_r6)
                trig_buy_r12 = (not math.isnan(r12) and r12 <= limit_r12)
                trig_buy_r22 = (not math.isnan(r22) and r22 <= limit_r22)
                trig_buy_T = (T > temp_limit)
                if trig_buy_r3 or trig_buy_r6 or trig_buy_r12 or trig_buy_r22 or trig_buy_T:
                    block_buy = True
                block_buy_series[i] = 1 if block_buy else 0
                # Debug log for decision day
                if debug_start_ts is not None:
                    ts_i = dates[i]
                    if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                        prev_close_22 = df["close"].shift(21).iloc[i] if i >= 21 else float('nan')
                        prev_date_22 = pd.to_datetime(dates[i - 21]).date() if i >= 21 else None
                        debug_rows.append({
                            "date": pd.to_datetime(ts_i).date(),
                            "event": "rebalance_due",
                            "decision": "buy",
                            "acted": (not block_buy),
                            "blocked_buy": bool(block_buy),
                            "blocked_reason_buy_r3_le_-3pct": bool(trig_buy_r3),
                            "blocked_reason_buy_r6_le_-3pct": bool(trig_buy_r6),
                            "blocked_reason_buy_r12_le_-2pct": bool(trig_buy_r12),
                            "blocked_reason_buy_r22_le_0pct": bool(trig_buy_r22),
                            "blocked_reason_buy_T_gt_1.3": bool(trig_buy_T),
                            "price": float(df["close"].iloc[i]),
                            "ref_date_22d": prev_date_22,
                            "price_22d_ago": float(prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_abs": float(df["close"].iloc[i] - prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_rel": float(r22) if not math.isnan(r22) else float('nan'),
                            "ret_3": float(r3) if not math.isnan(r3) else float('nan'),
                            "ret_6": float(r6) if not math.isnan(r6) else float('nan'),
                            "ret_12": float(r12) if not math.isnan(r12) else float('nan'),
                            "ret_22": float(r22) if not math.isnan(r22) else float('nan'),
                            "limit_buy_r3": float(limit_r3),
                            "limit_buy_r6": float(limit_r6),
                            "limit_buy_r12": float(limit_r12),
                            "limit_buy_r22": float(limit_r22),
                            "ratio_buy_r3": (float(r3) / float(limit_r3)) if (not math.isnan(r3) and limit_r3 != 0.0) else float('nan'),
                            "ratio_buy_r6": (float(r6) / float(limit_r6)) if (not math.isnan(r6) and limit_r6 != 0.0) else float('nan'),
                            "ratio_buy_r12": (float(r12) / float(limit_r12)) if (not math.isnan(r12) and limit_r12 != 0.0) else float('nan'),
                            "ratio_buy_r22": (float(r22) / float(limit_r22)) if (not math.isnan(r22) and limit_r22 != 0.0) else float('nan'),
                            "temp_T": float(T),
                            "base_p": float(base_p),
                            "rate_annual_pct": float(rate_today),
                            "target_p": float(target_p),
                            "curr_p_before": float(curr_p)
                        })
                if not block_buy:
                    prev_tqqq = port_tqqq[i]
                    prev_cash = port_cash[i]
                    port_tqqq[i] = total * target_p
                    port_cash[i] = total - port_tqqq[i]
                    total = port_tqqq[i] + port_cash[i]
                    curr_p = target_p
                    next_rebalance_idx = i + 22
                    last_rebalance_idx = i
                    acted = True
                    add_rebalance_entry(i, port_tqqq[i], port_cash[i], prev_tqqq, prev_cash)
                    if debug_start_ts is not None:
                        ts_i = dates[i]
                        if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                            debug_rows[-1]["curr_p_after"] = float(curr_p)
            elif target_p < curr_p - eps:
                # Sell-side filters
                block_sell = False
                # r3, r6, r12, r22 already computed above
                trig_sell_r3 = (not math.isnan(r3) and r3 >= 0.03)
                trig_sell_r6 = (not math.isnan(r6) and r6 >= 0.03)
                trig_sell_r12 = (not math.isnan(r12) and r12 >= 0.0225)
                trig_sell_r22 = (not math.isnan(r22) and r22 >= 0.0075)
                if trig_sell_r3 or trig_sell_r6 or trig_sell_r12 or trig_sell_r22:
                    block_sell = True
                block_sell_series[i] = 1 if block_sell else 0
                # Debug log for decision day
                if debug_start_ts is not None:
                    ts_i = dates[i]
                    if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                        prev_close_22 = df["close"].shift(21).iloc[i] if i >= 21 else float('nan')
                        prev_date_22 = pd.to_datetime(dates[i - 21]).date() if i >= 21 else None
                        debug_rows.append({
                            "date": pd.to_datetime(ts_i).date(),
                            "event": "rebalance_due",
                            "decision": "sell",
                            "acted": (not block_sell),
                            "blocked_sell": bool(block_sell),
                            "blocked_reason_sell_r3_ge_3pct": bool(trig_sell_r3),
                            "blocked_reason_sell_r6_ge_3pct": bool(trig_sell_r6),
                            "blocked_reason_sell_r12_ge_2.25pct": bool(trig_sell_r12),
                            "blocked_reason_sell_r22_ge_0.75pct": bool(trig_sell_r22),
                            "price": float(df["close"].iloc[i]),
                            "ref_date_22d": prev_date_22,
                            "price_22d_ago": float(prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_abs": float(df["close"].iloc[i] - prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_rel": float(r22) if not math.isnan(r22) else float('nan'),
                            "ret_3": float(r3) if not math.isnan(r3) else float('nan'),
                            "ret_6": float(r6) if not math.isnan(r6) else float('nan'),
                            "ret_12": float(r12) if not math.isnan(r12) else float('nan'),
                            "ret_22": float(r22) if not math.isnan(r22) else float('nan'),
                            "limit_sell_r3": 0.03,
                            "limit_sell_r6": 0.03,
                            "limit_sell_r12": 0.0225,
                            "limit_sell_r22": 0.0075,
                            "ratio_sell_r3": (float(r3) / 0.03) if (not math.isnan(r3)) else float('nan'),
                            "ratio_sell_r6": (float(r6) / 0.03) if (not math.isnan(r6)) else float('nan'),
                            "ratio_sell_r12": (float(r12) / 0.0225) if (not math.isnan(r12)) else float('nan'),
                            "ratio_sell_r22": (float(r22) / 0.0075) if (not math.isnan(r22)) else float('nan'),
                            "temp_T": float(T),
                            "base_p": float(base_p),
                            "rate_annual_pct": float(rate_today),
                            "target_p": float(target_p),
                            "curr_p_before": float(curr_p)
                        })
                if not block_sell:
                    prev_tqqq = port_tqqq[i]
                    prev_cash = port_cash[i]
                    port_tqqq[i] = total * target_p
                    port_cash[i] = total - port_tqqq[i]
                    total = port_tqqq[i] + port_cash[i]
                    curr_p = target_p
                    next_rebalance_idx = i + 22
                    last_rebalance_idx = i
                    acted = True
                    add_rebalance_entry(i, port_tqqq[i], port_cash[i], prev_tqqq, prev_cash)
                    if debug_start_ts is not None:
                        ts_i = dates[i]
                        if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                            debug_rows[-1]["curr_p_after"] = float(curr_p)
            else:
                # No significant change, but treat as a completed rebalance to keep cadence
                next_rebalance_idx = i + 22
                last_rebalance_idx = i
                acted = True
                if debug_start_ts is not None:
                    ts_i = dates[i]
                    if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                        prev_close_22 = df["close"].shift(21).iloc[i] if i >= 21 else float('nan')
                        prev_date_22 = pd.to_datetime(dates[i - 21]).date() if i >= 21 else None
                        r3 = df["ret_3"].iloc[i]
                        r6 = df["ret_6"].iloc[i]
                        r12 = df["ret_12"].iloc[i]
                        r22 = df["ret_22"].iloc[i]
                        debug_rows.append({
                            "date": pd.to_datetime(ts_i).date(),
                            "event": "rebalance_due",
                            "decision": "hold",
                            "acted": True,
                            "price": float(df["close"].iloc[i]),
                            "ref_date_22d": prev_date_22,
                            "price_22d_ago": float(prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_abs": float(df["close"].iloc[i] - prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_rel": float(r22) if not math.isnan(r22) else float('nan'),
                            "ret_3": float(r3) if not math.isnan(r3) else float('nan'),
                            "ret_6": float(r6) if not math.isnan(r6) else float('nan'),
                            "ret_12": float(r12) if not math.isnan(r12) else float('nan'),
                            "ret_22": float(r22) if not math.isnan(r22) else float('nan'),
                            "temp_T": float(T),
                            "base_p": float(base_p),
                            "rate_annual_pct": float(rate_today),
                            "target_p": float(target_p),
                            "curr_p_before": float(curr_p),
                            "curr_p_after": float(curr_p)
                        })

        port_total[i] = port_tqqq[i] + port_cash[i]
        deployed_p[i] = curr_p
        peak_total = max(peak_total, port_total[i])

    # Normalized lines for plotting
    unified_norm = df["close"].to_numpy() / df["close"].iloc[0]
    strategy_norm = port_total / port_total[0]

    # CAGR for strategy
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (strategy_norm[-1] / strategy_norm[0]) ** (1.0 / years) - 1.0
    # Print CAGR summary for convenience
    print(f"Strategy span: {df.index[0].date()} -> {df.index[-1].date()} ({years:.2f} years)")
    print(f"Strategy CAGR: {cagr * 100.0:.2f}%")

    if args.print_rebalances:
        print()
        if len(rebalance_entries) == 0:
            print("No rebalances were executed in the specified range.")
        else:
            rebalance_df = pd.DataFrame(rebalance_entries)
            start_date = pd.Timestamp(df.index[0])
            initial_total = float(port_total[0]) if port_total[0] > 0 else float("nan")
            cagr_values = []
            for _, row in rebalance_df.iterrows():
                total_after = float(row["total_after"])
                date = pd.Timestamp(row["date"])
                days_elapsed = (date - start_date).days
                if (
                    days_elapsed <= 0
                    or math.isnan(initial_total)
                    or initial_total <= 0
                    or total_after <= 0
                ):
                    cagr_values.append(float("nan"))
                else:
                    years_elapsed = days_elapsed / 365.25
                    if years_elapsed <= 0:
                        cagr_values.append(float("nan"))
                    else:
                        cagr_values.append((total_after / initial_total) ** (1.0 / years_elapsed) - 1.0)
            rebalance_df["cagr"] = cagr_values
            display_df = rebalance_df.copy()
            display_df["date"] = display_df["date"].dt.date
            display_df["tqqq_pct"] = display_df["p_tqqq"] * 100.0
            display_df["tbill_pct"] = display_df["p_cash"] * 100.0
            display_df["portfolio_value"] = display_df["total_after"]
            display_df["cagr_pct"] = display_df["cagr"] * 100.0
            display_df["action"] = display_df["delta_tqqq"].apply(
                lambda x: "Buy" if x > 0 else ("Sell" if x < 0 else "Hold")
            )

            def fmt_pct(val: float) -> str:
                if pd.isna(val):
                    return "   n/a"
                return f"{val:6.2f}%"

            def fmt_amt(val: float) -> str:
                return f"{val:,.2f}"

            columns = [
                "date",
                "action",
                "tqqq_pct",
                "tbill_pct",
                "delta_tqqq",
                "delta_cash",
                "portfolio_value",
                "cagr_pct",
            ]
            formatters = {
                "tqqq_pct": fmt_pct,
                "tbill_pct": fmt_pct,
                "delta_tqqq": fmt_amt,
                "delta_cash": fmt_amt,
                "portfolio_value": fmt_amt,
                "cagr_pct": fmt_pct,
            }
            print("Rebalance log:")
            print(display_df[columns].to_string(index=False, formatters=formatters))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(df.index, unified_norm, label="Unified (normalized)", color="#1f77b4", alpha=0.7)
    ax1.plot(df.index, strategy_norm, label=f"Strategy (CAGR {cagr*100.0:.2f}%)", color="#d62728")
    ax1.set_title("Unified Nasdaq vs Strategy Portfolio Value (start=1.0)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Value (normalized, log)")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="upper left")

    ax2.plot(df.index, df["temp"], label="Temperature", color="black")
    ax2.axhline(1.0, color="gray", linestyle="-", alpha=0.8)
    ax2.axhline(1.5, color="gray", linestyle=":", alpha=0.7)
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax2.plot(df.index, deployed_p, label="Proportion in TQQQ", color="green", alpha=0.8)
    ax2.set_title("Temperature and Deployment")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("T, deployment (0-1)")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    if args.save_plot:
        fig.savefig(args.save_plot, dpi=150)
    if args.save_csv:
        out = df.copy()
        out = out[
            [
                "close",
                "ret",
                "ret_3",
                "ret_6",
                "ret_12",
                "ret_22",
                "temp",
                "temp_ch_11",
                "temp_ch_22",
                "vol_22",
                "vol_66",
            ]
        ].copy()
        out["rate"] = rate_ann
        out["yc_spread"] = macro["yc_spread"].to_numpy()
        out["credit_spread"] = macro["credit_spread"].to_numpy()
        out["yc_change_22"] = macro["yc_change_22"].to_numpy()
        out["credit_change_22"] = macro["credit_change_22"].to_numpy()
        out["port_tqqq"] = port_tqqq
        out["port_cash"] = port_cash
        out["port_total"] = port_total
        out["deployed_p"] = deployed_p
        out["base_p"] = base_p_series
        out["target_p"] = target_p_series
        out["block_buy"] = block_buy_series
        out["block_sell"] = block_sell_series
        out["forced_derisk"] = forced_derisk_series
        out.index = out.index.date
        out.index.name = "date"
        out.to_csv(args.save_csv)
    # Rebalance debug CSV output
    if (args.debug_start or args.debug_end) and len(debug_rows) > 0:
        debug_df = pd.DataFrame(debug_rows)
        debug_df = debug_df.sort_values("date")
        debug_path = args.debug_csv if args.debug_csv else "strategy_tqqq_reserve_debug.csv"
        debug_df.to_csv(debug_path, index=False)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()


