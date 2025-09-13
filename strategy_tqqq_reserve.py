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
python strategy_tqqq_reserve.py [--csv unified_nasdaq.csv] [--experiment A1] [--end 2025-01-10] \
  [--fred-series FEDFUNDS] [--leverage 3.0] [--annual-fee 0.0095] [--borrow-divisor 0.7] \
  [--trading-days 252] [--save-plot strategy_tqqq.png] [--no-show]
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


def main():
    parser = argparse.ArgumentParser(description="Simulate TQQQ+reserve strategy with temperature & filters")
    parser.add_argument("--csv", default="unified_nasdaq.csv", help="Unified dataset CSV path")
    parser.add_argument("--end", default="2025-01-10", help="End date (YYYY-MM-DD) inclusive")
    parser.add_argument("--fred-series", default="FEDFUNDS", help="FRED rate series (default FEDFUNDS)")
    parser.add_argument(
        "--experiment",
        default="A1",
        choices=sorted(EXPERIMENTS.keys()),
        help="Strategy experiment to run (default A1)"
    )
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--annual-fee", type=float, default=0.0095)
    parser.add_argument("--borrow-divisor", type=float, default=0.7)
    parser.add_argument("--trading-days", type=int, default=252)
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
    end_ts = pd.to_datetime(args.end)
    df = df.loc[df.index <= end_ts]
    start_ts = df.index.min()

    # Daily returns for unified (QQQ proxy)
    df["ret"] = df["close"].pct_change().fillna(0.0)
    # Momentum windows use change from t-(N-1) to t
    df["ret_3"] = df["close"] / df["close"].shift(2) - 1.0
    df["ret_6"] = df["close"] / df["close"].shift(5) - 1.0
    df["ret_12"] = df["close"] / df["close"].shift(11) - 1.0
    df["ret_22"] = df["close"] / df["close"].shift(21) - 1.0

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

    # Temperature series
    temp, A, r, fit_start_ts = compute_temperature_series(pd, np, df, csv_path=args.csv)
    df["temp"] = temp

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

    # Start with all in cash, normalized portfolio = 1.0
    port_tqqq[0] = 0.0
    port_cash[0] = 1.0
    port_total[0] = port_cash[0] + port_tqqq[0]
    deployed_p[0] = 0.0

    # Rebalance scheduling
    next_rebalance_idx = 0  # day 0 is an eligible rebalance day after updates
    eps = 1e-6

    for i in range(num_days):
        # Update holdings based on previous day growth (for i=0 this applies one day of cash growth)
        if i > 0:
            port_tqqq[i] = port_tqqq[i - 1] * tqqq_factor[i]
            port_cash[i] = port_cash[i - 1] * cash_factor[i]
        else:
            port_tqqq[i] = port_tqqq[0] * tqqq_factor[0]
            port_cash[i] = port_cash[0] * cash_factor[0]

        total = port_tqqq[i] + port_cash[i]
        curr_p = 0.0 if total <= 0 else port_tqqq[i] / total

        # (Moved crash de-risk to rebalance-due section to enforce cadence)

        # Rebalance logic
        acted = False
        if i >= next_rebalance_idx:
            # First, check the 22d crash de-risk. If triggered, act immediately and skip momentum filters.
            ret22 = df["ret_22"].iloc[i]
            forced_today = False
            if not math.isnan(ret22) and ret22 <= -0.1525 and port_tqqq[i] > 0:
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
                total = port_cash[i]
                curr_p = 0.0
                next_rebalance_idx = i + 22
                forced_derisk_series[i] = 1
                acted = True
                forced_today = True
                # Update debug AFTER action
                if debug_start_ts is not None:
                    ts_i = dates[i]
                    if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                        debug_rows[-1]["curr_p_after"] = float(curr_p)
            if forced_today:
                port_total[i] = port_tqqq[i] + port_cash[i]
                deployed_p[i] = curr_p
                continue
            T = df["temp"].iloc[i]
            base_p = target_allocation_from_temperature(T)
            rate_today = float(rate_ann[i])
            r3 = df["ret_3"].iloc[i]
            r6 = df["ret_6"].iloc[i]
            r12 = df["ret_12"].iloc[i]
            r22 = df["ret_22"].iloc[i]
            target_p = apply_rate_taper(base_p, rate_today)
            cl_cfg = config.get("cold_leverage")
            if cl_cfg:
                target_p = apply_cold_leverage(target_p, T, rate_today, r22, **cl_cfg)
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
                yc_today = macro["yc_spread"].iloc[i]
                cs_today = macro["credit_spread"].iloc[i]
                target_p = apply_macro_filter(target_p, yc_today, cs_today, **macro_cfg)
            base_p_series[i] = base_p
            target_p_series[i] = target_p

            # Determine direction
            if target_p > curr_p + eps:
                # Buy-side filters
                block_buy = False
                # r3, r6, r12, r22 already computed above
                trig_buy_r3 = (not math.isnan(r3) and r3 <= -0.03)
                trig_buy_r6 = (not math.isnan(r6) and r6 <= -0.03)
                trig_buy_r12 = (not math.isnan(r12) and r12 <= -0.02)
                trig_buy_r22 = (not math.isnan(r22) and r22 <= 0.0)
                trig_buy_T = (T > 1.3)
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
                            "limit_buy_r3": -0.03,
                            "limit_buy_r6": -0.03,
                            "limit_buy_r12": -0.02,
                            "limit_buy_r22": 0.0,
                            "ratio_buy_r3": (float(r3) / -0.03) if (not math.isnan(r3)) else float('nan'),
                            "ratio_buy_r6": (float(r6) / -0.03) if (not math.isnan(r6)) else float('nan'),
                            "ratio_buy_r12": (float(r12) / -0.02) if (not math.isnan(r12)) else float('nan'),
                            "ratio_buy_r22": float('nan'),
                            "temp_T": float(T),
                            "base_p": float(base_p),
                            "rate_annual_pct": float(rate_today),
                            "target_p": float(target_p),
                            "curr_p_before": float(curr_p)
                        })
                if not block_buy:
                    port_tqqq[i] = total * target_p
                    port_cash[i] = total - port_tqqq[i]
                    total = port_tqqq[i] + port_cash[i]
                    curr_p = target_p
                    next_rebalance_idx = i + 22
                    acted = True
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
                    port_tqqq[i] = total * target_p
                    port_cash[i] = total - port_tqqq[i]
                    total = port_tqqq[i] + port_cash[i]
                    curr_p = target_p
                    next_rebalance_idx = i + 22
                    acted = True
                    if debug_start_ts is not None:
                        ts_i = dates[i]
                        if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                            debug_rows[-1]["curr_p_after"] = float(curr_p)
            else:
                # No significant change, but treat as a completed rebalance to keep cadence
                next_rebalance_idx = i + 22
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

    # Normalized lines for plotting
    unified_norm = df["close"].to_numpy() / df["close"].iloc[0]
    strategy_norm = port_total / port_total[0]

    # CAGR for strategy
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (strategy_norm[-1] / strategy_norm[0]) ** (1.0 / years) - 1.0
    # Print CAGR summary for convenience
    print(f"Strategy span: {df.index[0].date()} -> {df.index[-1].date()} ({years:.2f} years)")
    print(f"Strategy CAGR: {cagr * 100.0:.2f}%")

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
        out = out[["close", "ret", "ret_3", "ret_6", "ret_12", "ret_22", "temp"]].copy()
        out["rate"] = rate_ann
        out["yc_spread"] = macro["yc_spread"].to_numpy()
        out["credit_spread"] = macro["credit_spread"].to_numpy()
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


