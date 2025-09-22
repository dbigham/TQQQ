"""
TQQQ + cash-reserve strategy simulator with temperature and momentum filters

Overview
--------
This script simulates an allocation strategy that pairs a leveraged Nasdaq proxy (TQQQ-like)
with a cash reserve. Allocation to TQQQ is based on the market "temperature" (actual price
relative to a fitted constant-growth curve) and is adjusted by recent momentum filters and
interest-rate conditions. The reserve earns the current interest rate. The underlying symbol
defaults to QQQ but can be overridden via ``--base-symbol`` to run the same framework against
other indices or assets (for example SPY, BTC-USD, NVDA). When another symbol is selected the
script downloads its history on demand via Yahoo Finance, caches the fitted growth curve, and
reuses the saved parameters and PNG diagnostics on subsequent runs.

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
- unified_nasdaq.csv from build_unified_nasdaq.py (date, close) when ``--base-symbol QQQ``
  is used; for other symbols the script downloads auto-adjusted closes (cached under
  ``symbol_data/``) without requiring pre-built CSVs.
- Interest rate from FRED (default FEDFUNDS), fetched without key via fredgraph CSV (or other fallbacks)
- Temperature fit via nasdaq_temperature.py (robust 3-step fit), cached per symbol along with
  diagnostic PNG plots in ``temperature_cache/``

Outputs
-------
- Plot 1: normalized base-symbol series vs strategy portfolio value (same start = 1.0). Legend includes CAGR.
- Plot 2: temperature (black) with reference lines at T=1.0 (solid), T=1.5 and T=0.5 (dotted), and
  deployed proportion to the leveraged sleeve (green) on the same axis.
- Fit diagnostics: ``fit_constant_growth*.png`` and temperature plots saved with the symbol in
  the filename when not running the QQQ baseline.

Run
---
python strategy_tqqq_reserve.py [--base-symbol QQQ] [--csv unified_nasdaq.csv] [--experiment A36] \
  [--start 2010-01-01] [--end 2025-01-10] [--fred-series FEDFUNDS] [--leverage 3.0] \
  [--annual-fee 0.0095] [--borrow-divisor 0.7] [--trading-days 252] \
  [--initial-capital 100000] [--print-rebalances] [--save-plot strategy_tqqq.png] [--no-show]

The CLI defaults to the best-performing experiment; as of the latest calibration this is
A36 (High-Leverage Ramp), which delivers ~45.31% CAGR through 2025-01-10.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Mapping, Optional, Sequence

from tqqq import fetch_fred_series as download_fred_series
from tqqq import ensure_fundamentals
from tqqq import iterative_constant_growth, load_price_csv


TEMPERATURE_CACHE_DIR = "temperature_cache"
SYMBOL_DATA_DIR = "symbol_data"

# Global default controlling whether the simulator automatically replaces
# "leveraged ETF + cash" allocations with the low-drag replication described in
# ``docs/leverage_exposure_replication.md``. Experiments may override this on a
# per-profile basis via the ``use_leverage_replication`` flag.
DEFAULT_USE_LEVERAGE_REPLICATION = False


def import_libs():
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    return pd, np, plt


def get_fred_series(series_id: str, start, end):
    return download_fred_series(series_id, start, end, api_key=os.environ.get("FRED_API_KEY"))


def load_unified(pd, csv_path: str):
    df, _ = load_price_csv(csv_path, set_index=True)
    return df


def load_symbol_history(pd, symbol: str, *, csv_override: Optional[str] = None):
    """Load price history for the requested base symbol.

    For QQQ the unified dataset remains the default. For any symbol the caller
    may pass ``csv_override`` pointing at a CSV with ``date``/``close`` columns
    (identical format to ``unified_nasdaq.csv``). If no CSV is provided or the
    symbol is not QQQ, the function looks for a cached download in
    ``SYMBOL_DATA_DIR`` and falls back to Yahoo Finance when necessary.
    """

    symbol_upper = symbol.upper()

    if csv_override:
        if not os.path.exists(csv_override):
            raise FileNotFoundError(f"CSV override not found: {csv_override}")
        df = load_unified(pd, csv_override)
        return df, os.path.abspath(csv_override)

    if symbol_upper == "QQQ":
        default_csv = "unified_nasdaq.csv"
        if os.path.exists(default_csv):
            df = load_unified(pd, default_csv)
            return df, os.path.abspath(default_csv)

    cache_name = f"{symbol_upper}.csv"
    cache_path = os.path.join(SYMBOL_DATA_DIR, cache_name)
    if os.path.exists(cache_path):
        df = load_unified(pd, cache_path)
        # Refresh cache if it's stale (e.g., missing recent months)
        try:
            latest_cached = pd.to_datetime(df.index.max())
            refresh_cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
            if latest_cached < refresh_cutoff:
                df_new = download_symbol_history(pd, symbol_upper)
                latest_new = pd.to_datetime(df_new.index.max())
                if latest_new > latest_cached:
                    # Overwrite cache with fresher data
                    os.makedirs(SYMBOL_DATA_DIR, exist_ok=True)
                    df_to_save = df_new.copy()
                    df_to_save.index.name = "date"
                    df_to_save.to_csv(cache_path)
                    df = df_new
        except Exception:
            # Non-fatal: if refresh fails (e.g., offline), use cached data
            pass
        return df, os.path.abspath(cache_path)

    df = download_symbol_history(pd, symbol_upper)
    os.makedirs(SYMBOL_DATA_DIR, exist_ok=True)
    df_to_save = df.copy()
    df_to_save.index.name = "date"
    df_to_save.to_csv(cache_path)
    return df, os.path.abspath(cache_path)


def download_symbol_history(pd, symbol: str):
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via runtime usage
        raise RuntimeError(
            "yfinance is required to download price history for non-QQQ symbols"
        ) from exc

    data = yf.download(symbol, period="max", auto_adjust=True, progress=False, threads=False)
    if data.empty:
        # Some tickers (e.g., recent IPOs or timezone-less listings) fail via download();
        # fall back to the per-ticker history API before giving up.
        ticker = yf.Ticker(symbol)
        try:
            data = ticker.history(period="max", auto_adjust=True)
        except Exception as exc:  # pragma: no cover - runtime fallback
            raise RuntimeError(f"No price data returned for symbol {symbol}") from exc
        if data.empty:
            raise RuntimeError(f"No price data returned for symbol {symbol}")

    close_series = None
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        for key in ("Adj Close", "Close"):
            if key in level0:
                slice_df = data.xs(key, axis=1, level=0)
                if isinstance(slice_df, pd.DataFrame):
                    close_series = slice_df.iloc[:, 0]
                else:
                    close_series = slice_df
                break
    else:
        for key in ("Adj Close", "Close"):
            if key in data.columns:
                close_series = data[key]
                break

    if close_series is None:
        raise RuntimeError(f"Downloaded data for {symbol} does not contain Close/Adj Close columns")

    series = close_series.astype(float).copy()
    series = series.dropna()
    df = pd.DataFrame({"close": series})
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[df.index.notna()]
    df = df[df["close"] > 0]
    if df.empty:
        raise RuntimeError(f"No valid price rows for symbol {symbol}")
    return df


def ensure_temperature_assets(
    pd,
    np,
    plt,
    symbol: str,
    df_full,
    *,
    thresh2=0.35,
    thresh3=0.15,
    manual_model: Optional[Mapping[str, object]] = None,
):
    """Return temperature fit parameters, caching them alongside diagnostic plots.

    When ``manual_model`` is provided the usual iterative fit is bypassed and the
    constant-growth curve is solved from the supplied growth rate and
    temperature/date anchor instead. The resolved parameters are cached with the
    manual metadata so subsequent runs reuse the same curve.
    """

    os.makedirs(TEMPERATURE_CACHE_DIR, exist_ok=True)
    symbol_upper = symbol.upper()
    cache_path = os.path.join(TEMPERATURE_CACHE_DIR, f"{symbol_upper}.json")

    data_start = pd.Timestamp(df_full.index.min()).normalize().date().isoformat()
    data_end = pd.Timestamp(df_full.index.max()).normalize().date().isoformat()
    rows = int(df_full.shape[0])

    manual_spec: Optional[dict[str, object]] = None
    manual_anchor_ts = None
    desired_fit_method = "manual_override" if manual_model else "iterative_fit"
    if manual_model is not None:
        if not isinstance(manual_model, Mapping):
            raise TypeError("manual_model must be a mapping when provided")
        if "growth_rate" not in manual_model:
            raise KeyError("manual_model requires a 'growth_rate' field")
        if "anchor_date" not in manual_model or "anchor_temperature" not in manual_model:
            raise KeyError("manual_model requires 'anchor_date' and 'anchor_temperature' fields")

        growth_rate_raw = float(manual_model["growth_rate"])
        if not math.isfinite(growth_rate_raw):
            raise ValueError("Manual growth rate must be finite")
        growth_rate = growth_rate_raw / 100.0 if abs(growth_rate_raw) > 1.0 else growth_rate_raw
        if growth_rate <= -1.0:
            raise ValueError("Manual growth rate must keep 1 + r positive")

        anchor_temperature = float(manual_model["anchor_temperature"])
        if not math.isfinite(anchor_temperature) or anchor_temperature <= 0:
            raise ValueError("Manual anchor temperature must be positive")

        anchor_ts_candidate = pd.to_datetime(manual_model["anchor_date"])
        if pd.isna(anchor_ts_candidate):
            raise ValueError("Manual anchor date is not parseable")
        manual_anchor_ts = pd.Timestamp(anchor_ts_candidate).normalize()

        manual_spec = {
            "growth_rate": growth_rate,
            "anchor_temperature": anchor_temperature,
            "anchor_date": manual_anchor_ts.date().isoformat(),
        }

    cached = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            data_matches = (
                cached.get("data_start") == data_start
                and cached.get("data_end") == data_end
                and cached.get("rows") == rows
            )
            thresholds_match = True
            if manual_spec is None:
                thresholds_match = (
                    float(cached.get("thresh2", thresh2)) == float(thresh2)
                    and float(cached.get("thresh3", thresh3)) == float(thresh3)
                )
            cached_manual = cached.get("manual_model")
            if manual_spec is None:
                manual_match = cached_manual in (None, {})
            else:
                manual_match = cached_manual == manual_spec
            method_match = cached.get("fit_method", "iterative_fit") == desired_fit_method
            if not (data_matches and thresholds_match and manual_match and method_match):
                cached = None
        except Exception:
            cached = None

    if cached:
        start_ts = pd.to_datetime(cached["start_ts"])
        A = float(cached["A"])
        r = float(cached["r"])
        # Regenerate plots if they were removed
        curve_png = cached.get("curve_plot")
        temp_png = cached.get("temperature_plot")
        regenerate_curve = curve_png and not os.path.exists(curve_png)
        regenerate_temp = temp_png and not os.path.exists(temp_png)
        if regenerate_curve or regenerate_temp:
            curve_png, temp_png = save_temperature_plots(
                pd,
                np,
                plt,
                symbol_upper,
                df_full,
                A,
                r,
                start_ts,
                force_curve=regenerate_curve,
                force_temp=regenerate_temp,
            )
            cached["curve_plot"] = curve_png
            cached["temperature_plot"] = temp_png
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(cached, fh, indent=2, sort_keys=True)
        return A, r, start_ts

    start_ts = pd.Timestamp(df_full.index.min())
    if manual_spec is not None:
        assert manual_anchor_ts is not None  # for mypy/static reasoning
        eligible = df_full.index[df_full.index <= manual_anchor_ts]
        if len(eligible) == 0:
            raise RuntimeError(
                f"Price history for {symbol_upper} does not cover anchor date {manual_spec['anchor_date']}"
            )
        anchor_ts = pd.Timestamp(eligible.max())
        anchor_price = float(df_full.loc[anchor_ts, "close"])
        t_anchor = (anchor_ts - start_ts).days / 365.25
        growth_rate = float(manual_spec["growth_rate"])
        anchor_temperature = float(manual_spec["anchor_temperature"])
        base = 1.0 + growth_rate
        if base <= 0:
            raise ValueError("Manual growth rate results in non-positive base for exponential model")
        predicted_anchor = anchor_price / anchor_temperature
        A = float(predicted_anchor / math.pow(base, t_anchor))
        r = growth_rate
    else:
        t_years = (df_full.index - start_ts).days / 365.25
        prices = df_full["close"].to_numpy()

        t_array = t_years.to_numpy() if hasattr(t_years, "to_numpy") else t_years
        result = iterative_constant_growth(t_array, prices, thresholds=[thresh2, thresh3])
        final = result.final
        A = float(final.A)
        r = float(final.r)

    curve_png, temp_png = save_temperature_plots(
        pd,
        np,
        plt,
        symbol_upper,
        df_full,
        A,
        r,
        start_ts,
    )

    payload = {
        "symbol": symbol_upper,
        "A": A,
        "r": r,
        "start_ts": pd.Timestamp(start_ts).isoformat(),
        "data_start": data_start,
        "data_end": data_end,
        "rows": rows,
        "thresh2": float(thresh2),
        "thresh3": float(thresh3),
        "curve_plot": curve_png,
        "temperature_plot": temp_png,
        "fit_method": desired_fit_method,
        "manual_model": manual_spec,
    }
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    return A, r, start_ts


def save_temperature_plots(
    pd,
    np,
    plt,
    symbol_upper: str,
    df_full,
    A: float,
    r: float,
    start_ts,
    *,
    force_curve: bool = True,
    force_temp: bool = True,
):
    """Persist the curve-fit and temperature PNG diagnostics for the symbol."""

    # Put artifacts in per-symbol directory under symbols/
    symbol_dir = os.path.join("symbols", ("qqq" if symbol_upper == "QQQ" else symbol_upper.lower()))
    os.makedirs(symbol_dir, exist_ok=True)

    curve_name = "fit_constant_growth.png" if symbol_upper == "QQQ" else f"fit_constant_growth_{symbol_upper}.png"
    temp_name = "nasdaq_temperature.png" if symbol_upper == "QQQ" else f"temperature_{symbol_upper}.png"

    curve_path = os.path.abspath(os.path.join(symbol_dir, curve_name))
    temp_path = os.path.abspath(os.path.join(symbol_dir, temp_name))

    t_years = (df_full.index - start_ts).days / 365.25
    pred = A * np.power(1.0 + r, t_years)
    temp = df_full["close"].to_numpy() / pred

    if force_curve:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(df_full.index, df_full["close"], label=f"{symbol_upper} price", color="#1f77b4", alpha=0.7)
        ax.semilogy(df_full.index, pred, label="Fitted constant growth", color="#d62728")
        ax.set_title(f"{symbol_upper} Constant-Growth Fit (CAGR {r * 100.0:.2f}%)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close (log scale)")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(loc="upper left")
        text = f"A (start): {A:.4f}\nAnnual growth: {r * 100.0:.2f}%"
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        fig.tight_layout()
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)

    if force_temp:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_full.index, temp, label="Temperature", color="black")
        ax.axhline(1.0, color="gray", linestyle="-", alpha=0.8)
        ax.axhline(1.5, color="gray", linestyle=":", alpha=0.7)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"{symbol_upper} Temperature (Actual / Fitted, CAGR {r * 100.0:.2f}%)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (ratio)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(temp_path, dpi=150)
        plt.close(fig)

    return curve_path, temp_path


def compute_temperature_series(
    pd,
    np,
    df,
    csv_path: str = "unified_nasdaq.csv",
    model_params: Optional[tuple[float, float, object]] = None,
):
    """Compute temperature series using cached parameters or by fitting a CSV."""

    if model_params is None:
        from nasdaq_temperature import NasdaqTemperature  # type: ignore

        model = NasdaqTemperature(csv_path=csv_path)
        A = float(model.A)
        r = float(model.r)
        start_ts = pd.to_datetime(model.start_ts)
    else:
        A, r, start_ts = model_params
        A = float(A)
        r = float(r)
        start_ts = pd.to_datetime(start_ts)

    t_years = (df.index - start_ts).days / 365.25
    pred = A * np.power(1.0 + r, t_years)
    temp = df["close"].to_numpy() / pred
    return temp, A, r, start_ts


def target_allocation_from_temperature(
    T: float, anchors: Optional[Sequence[Mapping[str, float]]] = None
) -> float:
    """Return the baseline allocation as a function of temperature.

    When ``anchors`` is provided it should contain temperature/allocation
    pairs that define a piecewise-linear curve. Temperatures outside the
    provided range clamp to the nearest endpoint. The default behaviour
    reproduces the historical A1 settings (100% at T<=0.9, 80% at T=1.0,
    20% at T>=1.5).
    """

    if anchors:
        # Ensure the anchors are sorted by temperature to guarantee
        # deterministic interpolation. Ignore entries missing either field.
        cleaned = [
            (float(entry["temp"]), float(entry["allocation"]))
            for entry in anchors
            if "temp" in entry and "allocation" in entry
        ]
        if cleaned:
            cleaned.sort(key=lambda item: item[0])
            temps = [item[0] for item in cleaned]
            allocs = [item[1] for item in cleaned]
            if T <= temps[0]:
                return allocs[0]
            if T >= temps[-1]:
                return allocs[-1]
            for idx in range(1, len(cleaned)):
                if T <= temps[idx]:
                    t0, a0 = cleaned[idx - 1]
                    t1, a1 = cleaned[idx]
                    if t1 == t0:
                        return a1
                    frac = (T - t0) / (t1 - t0)
                    return a0 + frac * (a1 - a0)
    # Piecewise-linear interpolation using legacy anchors when no override
    # is provided.
    if T <= 0.9:
        return 1.0
    if T >= 1.5:
        return 0.2
    if T <= 1.0:
        # from (0.9,1.0) to (1.0,0.8)
        return 1.0 - 2.0 * (T - 0.9)
    # from (1.0,0.8) to (1.5,0.2)
    return 0.8 - 1.2 * (T - 1.0)


def compute_deployment_weights(
    target_allocation: float,
    leverage: float,
    *,
    use_replication: bool,
    eps: float = 1e-9,
) -> tuple[float, float, float]:
    """Return capital weights for (1× sleeve, leveraged sleeve, cash).

    The helper keeps backwards compatibility with the legacy behaviour when
    ``use_replication`` is False by returning ``(0, target_allocation,
    1-target_allocation)``. When replication is enabled the routine follows the
    construction laid out in ``docs/leverage_exposure_replication.md``: match the
    desired exposure to the underlying while minimising how much capital is held
    in the daily-reset leveraged ETF. The implementation only applies the
    low-drag mix when the target allocation does not exceed 100% of capital,
    which preserves the original borrowing profile for higher leverage targets.
    """

    if target_allocation <= eps:
        return 0.0, 0.0, 1.0
    if not use_replication:
        return 0.0, target_allocation, 1.0 - target_allocation
    if leverage <= 1.0 + eps:
        return 0.0, target_allocation, 1.0 - target_allocation

    exposure = leverage * target_allocation
    if exposure <= 1.0 + eps:
        weight_1x = max(0.0, exposure)
        weight_leveraged = 0.0
        weight_cash = 1.0 - weight_1x
    elif target_allocation <= 1.0 + eps:
        denom = leverage - 1.0
        if denom <= eps:
            return 0.0, target_allocation, 1.0 - target_allocation
        weight_leveraged = max(0.0, min(1.0, (exposure - 1.0) / denom))
        weight_1x = 1.0 - weight_leveraged
        weight_cash = 0.0
    else:
        return 0.0, target_allocation, 1.0 - target_allocation

    weight_cash = 1.0 - weight_1x - weight_leveraged
    return weight_1x, weight_leveraged, weight_cash


def apply_rate_taper(
    p: float,
    rate_annual_pct: float,
    *,
    start: Optional[float] = 10.0,
    end: Optional[float] = 12.0,
    min_allocation: float = 0.0,
    enabled: bool = True,
) -> float:
    """Scale allocation lower as rates rise.

    ``start`` marks the annual rate (in percent) at which tapering begins and
    ``end`` marks the level where allocation reaches ``min_allocation``. If
    ``enabled`` is False or the thresholds are omitted the incoming allocation
    ``p`` is returned unchanged.
    """

    if not enabled:
        return p
    if start is None or end is None:
        return p
    if end <= start:
        return p if rate_annual_pct <= start else min(min_allocation, p)
    if rate_annual_pct <= start:
        return p
    if rate_annual_pct >= end:
        return min_allocation
    span = end - start
    frac = (end - rate_annual_pct) / span
    frac = max(0.0, min(1.0, frac))
    return min_allocation + frac * (p - min_allocation)


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


def apply_rebound_accelerator(
    p: float,
    T: float,
    temp_ch_11: float,
    temp_ch_22: float,
    ret_3: float,
    ret_6: float,
    ret_22: float,
    vol_22: float,
    vol_66: float,
    rate_annual_pct: float,
    yc_spread: float,
    credit_spread: float,
    yc_change_22: float,
    credit_change_22: float,
    *,
    temp_min: float = 0.7,
    temp_max: float = 0.97,
    rate_threshold: float = 6.2,
    ret3_min: float = 0.008,
    ret6_min: float = 0.025,
    ret22_floor: float = -0.14,
    ret22_ceiling: float = 0.025,
    temp_ch11_min: float = 0.025,
    temp_ch22_min: float = 0.06,
    vol22_max: float = 0.024,
    vol_ratio_max: float = 0.88,
    yc_min: float = -0.3,
    credit_max: float = 2.45,
    yc_change_min: float = 0.02,
    credit_change_max: float = -0.01,
    boost: float = 0.32,
    cap: float = 2.5,
    extra_boost: float = 0.12,
    extra_temp_max: float = 0.9,
    extra_ret22_floor: float = -0.01,
    extra_ret22_ceiling: float = 0.04,
    extra_vol_ratio_max: float = 0.8,
    extra_cap: float = 2.58,
) -> float:
    """Accelerate leverage during early-stage rebounds while drawdowns heal."""

    metrics = (
        temp_ch_11,
        temp_ch_22,
        ret_3,
        ret_6,
        ret_22,
        vol_22,
        vol_66,
        yc_spread,
        credit_spread,
        yc_change_22,
        credit_change_22,
    )
    if any(math.isnan(x) for x in metrics):
        return p
    if vol_22 <= 0 or vol_66 <= 0:
        return p
    ratio = vol_22 / vol_66
    macro_ok = (
        yc_spread >= yc_min
        and credit_spread <= credit_max
        and yc_change_22 >= yc_change_min
        and credit_change_22 <= credit_change_max
    )
    momentum_ok = (
        ret_3 >= ret3_min
        and ret_6 >= ret6_min
        and ret_22 >= ret22_floor
        and ret_22 <= ret22_ceiling
    )
    if (
        T >= temp_min
        and T <= temp_max
        and rate_annual_pct < rate_threshold
        and temp_ch_11 >= temp_ch11_min
        and temp_ch_22 >= temp_ch22_min
        and ratio <= vol_ratio_max
        and vol_22 <= vol22_max
        and macro_ok
        and momentum_ok
    ):
        p = min(cap, p + boost)
        if (
            extra_boost > 0.0
            and T <= extra_temp_max
            and ret_22 >= extra_ret22_floor
            and ret_22 <= extra_ret22_ceiling
            and ratio <= extra_vol_ratio_max
        ):
            p = min(extra_cap, p + extra_boost)
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


def apply_rate_tailwind_multiplier(
    p: float,
    rate_annual_pct: float,
    T: float,
    ret_6: float,
    ret_22: float,
    *,
    rate_floor: float = 3.0,
    rate_ceiling: float = 6.0,
    temp_min: float = 0.8,
    temp_max: float = 1.15,
    ret6_min: float = 0.0,
    ret22_min: float = 0.04,
    scale_max: float = 1.18,
    cap: float = 2.6,
) -> float:
    """Scale exposure when cheap rates and solid trends align."""

    if math.isnan(ret_6) or math.isnan(ret_22):
        return p
    if not (temp_min <= T <= temp_max):
        return p
    if ret_6 < ret6_min or ret_22 < ret22_min:
        return p
    if rate_annual_pct >= rate_ceiling:
        return p
    if rate_annual_pct <= rate_floor:
        scale = scale_max
    else:
        span = rate_ceiling - rate_floor
        scale = 1.0 + (scale_max - 1.0) * (rate_ceiling - rate_annual_pct) / span
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


def apply_shock_guard(
    p: float,
    ret_1: float,
    ret_3: float,
    ret_6: float,
    *,
    ret1_floor: float = -0.06,
    ret3_floor: float = -0.12,
    ret6_floor: float = -0.18,
    scale: float = 0.4,
    floor: float = 0.5,
) -> float:
    """Rapidly scale back exposure on abrupt drawdowns."""

    triggers = []
    if ret_1 <= ret1_floor:
        triggers.append(True)
    if not math.isnan(ret_3) and ret_3 <= ret3_floor:
        triggers.append(True)
    if not math.isnan(ret_6) and ret_6 <= ret6_floor:
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


def apply_regime_accelerator(
    p: float,
    T: float,
    ret_3: float,
    ret_6: float,
    ret_12: float,
    ret_22: float,
    vol_22: float,
    vol_66: float,
    rate_annual_pct: float,
    yc_spread: float,
    credit_spread: float,
    yc_change_22: float,
    credit_change_22: float,
    *,
    temp_min: float = 0.78,
    temp_max: float = 1.06,
    rate_threshold: float = 5.8,
    ret3_min: float = 0.015,
    ret6_min: float = 0.03,
    ret12_min: float = 0.05,
    ret22_min: float = 0.08,
    vol22_max: float = 0.022,
    vol_ratio_max: float = 0.82,
    yc_min: float = -0.1,
    credit_max: float = 2.2,
    yc_change_min: float = 0.05,
    credit_change_max: float = -0.03,
    boost: float = 0.25,
    cap: float = 2.55,
    extra_boost: float = 0.15,
    extra_ret22_min: float = 0.12,
    extra_temp_max: float = 0.94,
    extra_vol_ratio_max: float = 0.78,
    extra_cap: float = 2.6,
) -> float:
    """Add a regime-based accelerator when everything lines up."""

    metrics = (
        ret_3,
        ret_6,
        ret_12,
        ret_22,
        vol_22,
        vol_66,
        yc_spread,
        credit_spread,
        yc_change_22,
        credit_change_22,
    )
    if any(math.isnan(x) for x in metrics):
        return p
    if vol_22 <= 0 or vol_66 <= 0:
        return p
    ratio = vol_22 / vol_66
    macro_ok = (
        yc_spread >= yc_min
        and credit_spread <= credit_max
        and yc_change_22 >= yc_change_min
        and credit_change_22 <= credit_change_max
    )
    momentum_ok = (
        ret_3 >= ret3_min
        and ret_6 >= ret6_min
        and ret_12 >= ret12_min
        and ret_22 >= ret22_min
    )
    if (
        T >= temp_min
        and T <= temp_max
        and rate_annual_pct < rate_threshold
        and ratio <= vol_ratio_max
        and vol_22 <= vol22_max
        and macro_ok
        and momentum_ok
    ):
        p = min(cap, p + boost)
        if (
            extra_boost > 0.0
            and ret_22 >= extra_ret22_min
            and T <= extra_temp_max
            and ratio <= extra_vol_ratio_max
        ):
            p = min(extra_cap, p + extra_boost)
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
    "A1": {
        # Enable the leverage replication mix that maps partial TQQQ allocations to
        # unlevered exposure + a smaller leveraged sleeve. See
        # docs/leverage_exposure_replication.md for a walkthrough.
        "use_leverage_replication": True,
    },  # Baseline strategy with no extra features beyond the replication mix
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

# A1g replaces the classic A1 baseline with globally optimised parameters.
EXPERIMENTS["A1g"] = {
    "temperature_allocation": [
        {"temp": 0.9273427469847189, "allocation": 0.9},
        {"temp": 1.0, "allocation": 0.6914229615905205},
        {"temp": 1.3895980060223487, "allocation": 0.12368716918997923},
    ],
    "momentum_filters": {
        "buy": {
            "ret3": -0.008451681554022185,
            "ret6": -0.011054749523119571,
            "ret12": -0.004966377981066122,
            "ret22": 0.0024202479764357894,
            "temp": 1.44541683706992,
        },
        "sell": {
            "ret3": 0.07164233126484575,
            "ret6": 0.007814235979734244,
            "ret12": 0.003979505425800826,
            "ret22": 0.003273221232861071,
        },
    },
    "rate_taper": {
        "start": 10.578485326034812,
        "end": 11.648140801455359,
        "min_allocation": 0.016819225490664724,
        "enabled": True,
    },
    "crash_derisk": {
        "enabled": False,
        "threshold": None,
        "cooldown_days": 22,
    },
    "rebalance_days": 22,
}

# Provide an uppercase alias for convenience when running CLI experiments.
EXPERIMENTS["A1G"] = {**EXPERIMENTS["A1g"]}

# GOOG2X retunes A1g specifically for Google with a 2x leveraged sleeve
EXPERIMENTS["GOOG2X"] = {
    "temperature_allocation": [
        {"temp": 0.9112842466027813, "allocation": 1.5},
        {"temp": 1.0, "allocation": 0.9996099084782909},
        {"temp": 1.4141245147869759, "allocation": 0.08827971012283556},
    ],
    "momentum_filters": {
        "buy": {
            "ret3": -0.026263459101261304,
            "ret6": -0.07754344137856113,
            "ret12": -0.0047306564171121345,
            "ret22": 0.0036625749604032062,
            "temp": 1.2598906882419105,
        },
        "sell": {
            "ret3": 0.007677675400901423,
            "ret6": 0.020166327393426865,
            "ret12": 0.015591788434369168,
            "ret22": -0.0014547122384232657,
        },
    },
    "rate_taper": {
        "start": 9.99764995369902,
        "end": 12.44076444138291,
        "min_allocation": 0.17479564203226142,
        "enabled": True,
    },
    "crash_derisk": {
        "enabled": False,
        "threshold": None,
        "cooldown_days": 22,
    },
    "rebalance_days": 22,
    "leverage_override": 2.0,
    "base_symbol": "GOOGL",
}
EXPERIMENTS["GOOG2x"] = {**EXPERIMENTS["GOOG2X"]}

# GOOG2X.B caps deployment at 100% with no margin borrowing while keeping 2x sleeve
EXPERIMENTS["GOOG2X_B"] = {
    **EXPERIMENTS["GOOG2X"],
    "temperature_allocation": [
        {"temp": 0.9112842466027813, "allocation": 1.0},
        {"temp": 1.0, "allocation": 0.9996099084782909},
        {"temp": 1.4141245147869759, "allocation": 0.08827971012283556},
    ],
}
EXPERIMENTS["GOOG2XB"] = {**EXPERIMENTS["GOOG2X_B"]}
EXPERIMENTS["GOOG2X.B"] = {**EXPERIMENTS["GOOG2X_B"]}
# TESLA2X calibrates the A1g heuristics for Tesla with a 2x leveraged sleeve
EXPERIMENTS["TESLA2X"] = {
    "temperature_allocation": [
        {"temp": 0.8201083581673381, "allocation": 1.5},
        {"temp": 1.0, "allocation": 0.45},
        {"temp": 1.5570503465488417, "allocation": 0.31375185711100995},
    ],
    "momentum_filters": {
        "buy": {
            "ret3": -0.05979094145328703,
            "ret6": 0.0077659670035262855,
            "ret12": -0.007855066600683148,
            "ret22": 0.004182532001556824,
            "temp": 1.1949920627986395,
        },
        "sell": {
            "ret3": 0.03784157366662905,
            "ret6": 0.03830890778260611,
            "ret12": 0.021650430917415163,
            "ret22": 0.012962137714778287,
        },
    },
    "rate_taper": {
        "start": 10.438350002489097,
        "end": 12.991363019945322,
        "min_allocation": 0.15849329602147574,
        "enabled": True,
    },
    "crash_derisk": {
        "enabled": False,
        "threshold": None,
        "cooldown_days": 22,
    },
    "rebalance_days": 22,
    "leverage_override": 2.0,
    "base_symbol": "TSLA",
}
EXPERIMENTS["TESLA2x"] = {**EXPERIMENTS["TESLA2X"]}

# TESLA2X.B caps deployment at 100% while retaining the 2x sleeve characteristics
EXPERIMENTS["TESLA2X_B"] = {
    **EXPERIMENTS["TESLA2X"],
    "temperature_allocation": [
        {"temp": 0.8201083581673381, "allocation": 1.0},
        {"temp": 1.0, "allocation": 0.45},
        {"temp": 1.5570503465488417, "allocation": 0.31375185711100995},
    ],
}
EXPERIMENTS["TESLA2XB"] = {**EXPERIMENTS["TESLA2X_B"]}
EXPERIMENTS["TESLA2X.B"] = {**EXPERIMENTS["TESLA2X_B"]}
EXPERIMENTS["TESLA2X_B2"] = {
    "temperature_allocation": [
        {"temp": 0.9546168379713842, "allocation": 0.9910526941733343},
        {"temp": 1.0, "allocation": 0.542726817470043},
        {"temp": 1.436197375556257, "allocation": 0.2773118142633297},
    ],
    "momentum_filters": {
        "buy": {
            "ret3": -0.0016223931804276595,
            "ret6": -0.10892308862371565,
            "ret12": -0.022330213526450142,
            "ret22": None,
            "temp": 1.226046216508751,
        },
        "sell": {
            "ret3": 0.009977104443394433,
            "ret6": 0.025386756232024073,
            "ret12": 0.02247611843492726,
            "ret22": -0.00557861778446388,
        },
    },
    "rate_taper": {
        "start": 10.357193651029094,
        "end": 11.517037129998153,
        "min_allocation": 0.2171256649327028,
        "enabled": True,
    },
    "crash_derisk": {
        "enabled": False,
        "threshold": None,
        "cooldown_days": 22,
    },
    "rebalance_days": 22,
    "leverage_override": 2.0,
    "base_symbol": "TSLA",
    "temperature_model": {
        "growth_rate": 0.42,
        "anchor_date": "2025-09-18",
        "anchor_temperature": 1.26,
    },
}
EXPERIMENTS["TESLA2XB2"] = {**EXPERIMENTS["TESLA2X_B2"]}
EXPERIMENTS["TESLA2X.B2"] = {**EXPERIMENTS["TESLA2X_B2"]}
EXPERIMENTS["TESLA2X.B.2"] = {**EXPERIMENTS["TESLA2X_B2"]}
EXPERIMENTS["TESLA2x.b.2"] = {**EXPERIMENTS["TESLA2X_B2"]}
# CRM2X tunes the leveraged optimiser for Salesforce with a 2x sleeve
EXPERIMENTS["CRM2X"] = {
    "temperature_allocation": [
        {"temp": 0.943276238479186, "allocation": 1.300935182207879},
        {"temp": 1.0, "allocation": 1.1},
        {"temp": 1.2851226598530594, "allocation": 0.07214073410116358},
    ],
    "momentum_filters": {
        "buy": {
            "ret3": -0.009934536318049225,
            "ret6": 0.0010685452318550104,
            "ret12": 0.006707013120519407,
            "ret22": -0.0033135814935658452,
            "temp": 1.3449941220997192,
        },
        "sell": {
            "ret3": 0.09028928135700337,
            "ret6": 0.013594010399651791,
            "ret12": -0.026122410893363583,
            "ret22": 0.004031251686438176,
        },
    },
    "rate_taper": {
        "start": 10.63726107519519,
        "end": 13.207353213335777,
        "min_allocation": 0.17858434211873486,
        "enabled": True,
    },
    "crash_derisk": {
        "enabled": False,
        "threshold": None,
        "cooldown_days": 22,
    },
    "rebalance_days": 22,
    "leverage_override": 2.0,
    "base_symbol": "CRM",
}
EXPERIMENTS["CRM2x"] = {**EXPERIMENTS["CRM2X"]}

# CRM2X.B mirrors CRM2X but caps deployment at 100% of capital
EXPERIMENTS["CRM2X_B"] = {
    **EXPERIMENTS["CRM2X"],
    "temperature_allocation": [
        {"temp": 0.943276238479186, "allocation": 1.0},
        {"temp": 1.0, "allocation": 1.0},
        {"temp": 1.2851226598530594, "allocation": 0.07214073410116358},
    ],
}
EXPERIMENTS["CRM2XB"] = {**EXPERIMENTS["CRM2X_B"]}
EXPERIMENTS["CRM2X.B"] = {**EXPERIMENTS["CRM2X_B"]}
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

# A26 adds a snapback rung to the ladder for deep-value momentum surges
EXPERIMENTS["A26"] = {
    **EXPERIMENTS["A25"],
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.25},
            {"temp_max": 0.83, "min_allocation": 1.6},
            {"temp_max": 0.79, "min_allocation": 1.9, "ret22_min": -0.05},
            {"temp_max": 0.75, "min_allocation": 2.2, "ret6_min": 0.0, "ret22_min": -0.015},
            {"temp_max": 0.72, "min_allocation": 2.4, "ret6_min": 0.01, "ret22_min": 0.0},
            {"temp_max": 0.68, "min_allocation": 2.55, "ret6_min": 0.02, "ret22_min": 0.035},
        ],
        "cap": 2.6,
    },
    "hot_momentum_leverage": {
        "boost": 0.35,
        "temperature_max": 1.10,
        "rate_threshold": 6.0,
        "ret22_threshold": 0.065,
        "ret3_threshold": 0.012,
        "ret6_threshold": 0.018,
        "ret12_threshold": 0.022,
        "extra_boost": 0.28,
        "extra_temperature_max": 1.01,
        "extra_rate_threshold": 5.5,
        "extra_ret22_threshold": 0.10,
        "stop_ret12": -0.008,
        "stop_ret3": -0.004,
        "cap": 2.55,
        "extra_cap": 2.6,
    },
    "global_allocation_cap": 2.6,
}

# A27 layers a regime accelerator that stacks leverage when momentum, macro, and volatility align
# Lifts CAGR from 41.21% (A25) to 41.76% (+0.55 pp, +1.33% relative)
EXPERIMENTS["A27"] = {
    **EXPERIMENTS["A25"],
    "hot_momentum_leverage": {
        "boost": 0.33,
        "temperature_max": 1.09,
        "rate_threshold": 5.9,
        "ret22_threshold": 0.065,
        "ret3_threshold": 0.012,
        "ret6_threshold": 0.018,
        "ret12_threshold": 0.02,
        "extra_boost": 0.25,
        "extra_temperature_max": 1.01,
        "extra_rate_threshold": 5.4,
        "extra_ret22_threshold": 0.095,
        "stop_ret12": -0.009,
        "stop_ret3": -0.004,
        "cap": 2.5,
        "extra_cap": 2.58,
    },
    "regime_accelerator": {
        "temp_min": 0.8,
        "temp_max": 1.04,
        "rate_threshold": 5.8,
        "ret3_min": 0.015,
        "ret6_min": 0.035,
        "ret12_min": 0.055,
        "ret22_min": 0.085,
        "vol22_max": 0.021,
        "vol_ratio_max": 0.8,
        "yc_min": -0.05,
        "credit_max": 2.2,
        "yc_change_min": 0.06,
        "credit_change_max": -0.035,
        "boost": 0.28,
        "cap": 2.55,
        "extra_boost": 0.12,
        "extra_ret22_min": 0.125,
        "extra_temp_max": 0.95,
        "extra_vol_ratio_max": 0.74,
        "extra_cap": 2.6,
    },
    "global_allocation_cap": 2.55,
}

# A28 retunes the cold ladder and volatility triggers for earlier, gated leverage ramps
EXPERIMENTS["A28"] = {
    **EXPERIMENTS["A27"],
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.25},
            {"temp_max": 0.82, "min_allocation": 1.6, "ret22_min": -0.06},
            {"temp_max": 0.78, "min_allocation": 1.95, "ret22_min": -0.03},
            {"temp_max": 0.74, "min_allocation": 2.2, "ret6_min": 0.0, "ret22_min": -0.01},
            {"temp_max": 0.70, "min_allocation": 2.35, "ret6_min": 0.01, "ret22_min": 0.015},
            {"temp_max": 0.67, "min_allocation": 2.48, "ret6_min": 0.02, "ret22_min": 0.04},
        ],
        "cap": 2.5,
    },
    "temperature_curve_boost": {
        "pivot": 0.9,
        "slope": 2.6,
        "max_extra": 0.5,
        "cap": 1.5,
    },
    "volatility_squeeze_boost": {
        "vol22_threshold": 0.018,
        "vol_ratio_max": 0.72,
        "ret6_min": 0.02,
        "ret22_min": -0.002,
        "boost": 0.32,
        "cap": 2.28,
    },
    "regime_accelerator": {
        "temp_min": 0.79,
        "temp_max": 1.05,
        "rate_threshold": 5.8,
        "ret3_min": 0.014,
        "ret6_min": 0.032,
        "ret12_min": 0.05,
        "ret22_min": 0.082,
        "vol22_max": 0.0215,
        "vol_ratio_max": 0.78,
        "yc_min": -0.05,
        "credit_max": 2.15,
        "yc_change_min": 0.055,
        "credit_change_max": -0.035,
        "boost": 0.3,
        "cap": 2.55,
        "extra_boost": 0.15,
        "extra_ret22_min": 0.125,
        "extra_temp_max": 0.94,
        "extra_vol_ratio_max": 0.72,
        "extra_cap": 2.6,
    },
    "global_allocation_cap": 2.58,
}

# A29 introduces a rebound accelerator to lean in before momentum fully heals
EXPERIMENTS["A29"] = {
    **EXPERIMENTS["A27"],
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.22},
            {"temp_max": 0.82, "min_allocation": 1.55},
            {"temp_max": 0.78, "min_allocation": 1.9, "ret22_min": -0.05},
            {"temp_max": 0.74, "min_allocation": 2.15, "ret6_min": 0.0, "ret22_min": -0.015},
            {"temp_max": 0.71, "min_allocation": 2.32, "ret6_min": 0.01, "ret22_min": 0.01},
            {"temp_max": 0.68, "min_allocation": 2.45, "ret6_min": 0.02, "ret22_min": 0.035},
        ],
        "cap": 2.5,
    },
    "rebound_accelerator": {
        "temp_min": 0.72,
        "temp_max": 0.97,
        "rate_threshold": 6.0,
        "ret3_min": 0.01,
        "ret6_min": 0.028,
        "ret22_floor": -0.12,
        "ret22_ceiling": 0.02,
        "temp_ch11_min": 0.028,
        "temp_ch22_min": 0.07,
        "vol22_max": 0.023,
        "vol_ratio_max": 0.85,
        "yc_min": -0.2,
        "credit_max": 2.35,
        "yc_change_min": 0.035,
        "credit_change_max": -0.015,
        "boost": 0.34,
        "cap": 2.52,
        "extra_boost": 0.14,
        "extra_temp_max": 0.9,
        "extra_ret22_floor": -0.005,
        "extra_ret22_ceiling": 0.035,
        "extra_vol_ratio_max": 0.78,
        "extra_cap": 2.58,
    },
    "global_allocation_cap": 2.58,
}

# A30 pushes leverage higher in confirmed regimes while keeping strict gating
EXPERIMENTS["A30"] = {
    **EXPERIMENTS["A27"],
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.25},
            {"temp_max": 0.82, "min_allocation": 1.6},
            {"temp_max": 0.78, "min_allocation": 1.95, "ret22_min": -0.04},
            {"temp_max": 0.74, "min_allocation": 2.2, "ret6_min": 0.0, "ret22_min": -0.01},
            {"temp_max": 0.71, "min_allocation": 2.35, "ret6_min": 0.01, "ret22_min": 0.015},
            {"temp_max": 0.68, "min_allocation": 2.5, "ret6_min": 0.02, "ret22_min": 0.04},
        ],
        "cap": 2.55,
    },
    "hot_momentum_leverage": {
        "boost": 0.36,
        "temperature_max": 1.08,
        "rate_threshold": 5.9,
        "ret22_threshold": 0.07,
        "ret3_threshold": 0.015,
        "ret6_threshold": 0.02,
        "ret12_threshold": 0.025,
        "extra_boost": 0.28,
        "extra_temperature_max": 1.0,
        "extra_rate_threshold": 5.3,
        "extra_ret22_threshold": 0.105,
        "stop_ret12": -0.008,
        "stop_ret3": -0.003,
        "cap": 2.6,
        "extra_cap": 2.68,
    },
    "regime_accelerator": {
        "temp_min": 0.79,
        "temp_max": 1.03,
        "rate_threshold": 5.7,
        "ret3_min": 0.016,
        "ret6_min": 0.034,
        "ret12_min": 0.055,
        "ret22_min": 0.09,
        "vol22_max": 0.021,
        "vol_ratio_max": 0.78,
        "yc_min": -0.05,
        "credit_max": 2.1,
        "yc_change_min": 0.065,
        "credit_change_max": -0.04,
        "boost": 0.32,
        "cap": 2.6,
        "extra_boost": 0.18,
        "extra_ret22_min": 0.13,
        "extra_temp_max": 0.94,
        "extra_vol_ratio_max": 0.72,
        "extra_cap": 2.68,
    },
    "global_allocation_cap": 2.68,
}

# A33 adds a fast crash guard and low-rate scaling to the aggressive stack
EXPERIMENTS["A33"] = {
    **EXPERIMENTS["A30"],
    "shock_guard": {
        "ret1_floor": -0.055,
        "ret3_floor": -0.10,
        "ret6_floor": -0.16,
        "scale": 0.25,
        "floor": 0.0,
    },
    "rate_tailwind_multiplier": {
        "rate_floor": 3.0,
        "rate_ceiling": 5.5,
        "temp_min": 0.82,
        "temp_max": 1.08,
        "ret6_min": 0.012,
        "ret22_min": 0.055,
        "scale_max": 1.10,
        "cap": 2.6,
    },
    "intra_cycle_rebalance": {
        "threshold": 0.08,
        "min_gap_days": 5,
        "direction": "both",
    },
    "global_allocation_cap": 2.55,
}

# A34 raises the baseline and rate scaling while retaining the crash guard
EXPERIMENTS["A34"] = {
    **EXPERIMENTS["A33"],
    "temperature_curve_boost": {
        "pivot": 1.0,
        "slope": 2.3,
        "max_extra": 0.4,
        "cap": 1.5,
    },
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.3},
            {"temp_max": 0.82, "min_allocation": 1.65},
            {"temp_max": 0.78, "min_allocation": 2.0, "ret22_min": -0.035},
            {"temp_max": 0.74, "min_allocation": 2.25, "ret6_min": 0.005, "ret22_min": -0.005},
            {"temp_max": 0.70, "min_allocation": 2.4, "ret6_min": 0.012, "ret22_min": 0.02},
        ],
        "cap": 2.45,
    },
    "rate_tailwind_multiplier": {
        "rate_floor": 3.0,
        "rate_ceiling": 5.5,
        "temp_min": 0.82,
        "temp_max": 1.08,
        "ret6_min": 0.012,
        "ret22_min": 0.055,
        "scale_max": 1.15,
        "cap": 2.6,
    },
    "hot_momentum_leverage": {
        "boost": 0.36,
        "temperature_max": 1.08,
        "rate_threshold": 5.9,
        "ret22_threshold": 0.07,
        "ret3_threshold": 0.015,
        "ret6_threshold": 0.02,
        "ret12_threshold": 0.025,
        "extra_boost": 0.27,
        "extra_temperature_max": 0.99,
        "extra_rate_threshold": 5.3,
        "extra_ret22_threshold": 0.105,
        "stop_ret12": -0.008,
        "stop_ret3": -0.003,
        "cap": 2.6,
        "extra_cap": 2.65,
    },
    "global_allocation_cap": 2.6,
}

# A35 widens cold boosts and keeps partial exposure during macro stress
EXPERIMENTS["A35"] = {
    **EXPERIMENTS["A34"],
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.3},
            {"temp_max": 0.82, "min_allocation": 1.7},
            {"temp_max": 0.78, "min_allocation": 2.05, "ret22_min": -0.03},
            {"temp_max": 0.74, "min_allocation": 2.3, "ret6_min": 0.005, "ret22_min": -0.002},
            {"temp_max": 0.70, "min_allocation": 2.45, "ret6_min": 0.012, "ret22_min": 0.02},
            {"temp_max": 0.66, "min_allocation": 2.55, "ret6_min": 0.02, "ret22_min": 0.04},
        ],
        "cap": 2.55,
    },
    "cold_leverage": {
        "boost": 0.25,
        "temperature_threshold": 0.82,
        "rate_threshold": 5.5,
        "extra_boost": 0.2,
        "extra_temperature_threshold": 0.75,
        "extra_rate_threshold": 4.5,
        "extra_ret22_threshold": 0.015,
    },
    "macro_filter": {
        "yc_threshold": -0.2,
        "credit_threshold": 2.15,
        "reduce_to": 0.2,
        "require_both": True,
    },
    "rate_tailwind_multiplier": {
        "rate_floor": 3.0,
        "rate_ceiling": 5.4,
        "temp_min": 0.8,
        "temp_max": 1.08,
        "ret6_min": 0.015,
        "ret22_min": 0.06,
        "scale_max": 1.2,
        "cap": 2.65,
    },
    "momentum_guard": {
        "ret6_floor": -0.04,
        "ret22_floor": -0.1,
        "scale": 0.5,
        "floor": 0.55,
    },
    "global_allocation_cap": 2.65,
}

# A36 simply dials leverage higher while keeping the A27 signal stack intact
# Boosts CAGR to 45.31% (+3.55 pp, +8.5% relative vs A27)
EXPERIMENTS["A36"] = {
    **EXPERIMENTS["A27"],
    "leverage_override": 3.6,
}

# A31 blends rate tailwinds with a warmer baseline curve to stay invested longer
EXPERIMENTS["A31"] = {
    **EXPERIMENTS["A27"],
    "temperature_curve_boost": {
        "pivot": 1.02,
        "slope": 2.4,
        "max_extra": 0.45,
        "cap": 1.55,
    },
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.25},
            {"temp_max": 0.82, "min_allocation": 1.65, "ret22_min": -0.06},
            {"temp_max": 0.78, "min_allocation": 2.0, "ret22_min": -0.03},
            {"temp_max": 0.74, "min_allocation": 2.2, "ret6_min": 0.005, "ret22_min": -0.005},
            {"temp_max": 0.71, "min_allocation": 2.35, "ret6_min": 0.012, "ret22_min": 0.015},
            {"temp_max": 0.68, "min_allocation": 2.45, "ret6_min": 0.02, "ret22_min": 0.035},
        ],
        "cap": 2.5,
    },
    "hot_momentum_leverage": {
        "boost": 0.34,
        "temperature_max": 1.1,
        "rate_threshold": 5.9,
        "ret22_threshold": 0.065,
        "ret3_threshold": 0.012,
        "ret6_threshold": 0.018,
        "ret12_threshold": 0.022,
        "extra_boost": 0.25,
        "extra_temperature_max": 1.0,
        "extra_rate_threshold": 5.4,
        "extra_ret22_threshold": 0.1,
        "stop_ret12": -0.008,
        "stop_ret3": -0.004,
        "cap": 2.55,
        "extra_cap": 2.62,
    },
    "regime_accelerator": {
        "temp_min": 0.8,
        "temp_max": 1.04,
        "rate_threshold": 5.7,
        "ret3_min": 0.015,
        "ret6_min": 0.032,
        "ret12_min": 0.052,
        "ret22_min": 0.085,
        "vol22_max": 0.0215,
        "vol_ratio_max": 0.78,
        "yc_min": -0.05,
        "credit_max": 2.15,
        "yc_change_min": 0.06,
        "credit_change_max": -0.035,
        "boost": 0.3,
        "cap": 2.55,
        "extra_boost": 0.14,
        "extra_ret22_min": 0.12,
        "extra_temp_max": 0.94,
        "extra_vol_ratio_max": 0.72,
        "extra_cap": 2.62,
    },
    "rate_tailwind_multiplier": {
        "rate_floor": 3.0,
        "rate_ceiling": 5.5,
        "temp_min": 0.82,
        "temp_max": 1.1,
        "ret6_min": 0.01,
        "ret22_min": 0.05,
        "scale_max": 1.16,
        "cap": 2.55,
    },
    "global_allocation_cap": 2.6,
}

# A32 warms the base curve, leans into low-rate trends, and keeps a macro cushion
EXPERIMENTS["A32"] = {
    **EXPERIMENTS["A27"],
    "temperature_curve_boost": {
        "pivot": 1.02,
        "slope": 2.2,
        "max_extra": 0.4,
        "cap": 1.45,
    },
    "temperature_leverage_ladder": {
        "steps": [
            {"temp_max": 0.86, "min_allocation": 1.25},
            {"temp_max": 0.82, "min_allocation": 1.55},
            {"temp_max": 0.78, "min_allocation": 1.9, "ret22_min": -0.04},
            {"temp_max": 0.74, "min_allocation": 2.2, "ret6_min": 0.0, "ret22_min": -0.01},
            {"temp_max": 0.70, "min_allocation": 2.35, "ret6_min": 0.01, "ret22_min": 0.015},
        ],
        "cap": 2.4,
    },
    "macro_filter": {
        "yc_threshold": -0.2,
        "credit_threshold": 2.15,
        "reduce_to": 0.2,
        "require_both": True,
    },
    "intra_cycle_rebalance": {
        "threshold": 0.1,
        "min_gap_days": 6,
        "direction": "buy",
    },
    "rate_tailwind_multiplier": {
        "rate_floor": 3.0,
        "rate_ceiling": 5.8,
        "temp_min": 0.85,
        "temp_max": 1.12,
        "ret6_min": 0.01,
        "ret22_min": 0.05,
        "scale_max": 1.12,
        "cap": 2.4,
    },
    "global_allocation_cap": 2.45,
}


def main():
    parser = argparse.ArgumentParser(description="Simulate TQQQ+reserve strategy with temperature & filters")
    parser.add_argument(
        "--base-symbol",
        default=None,
        help="Underlying symbol/index used for the leverage sleeve (defaults to the experiment base symbol or QQQ)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path with date/close columns (defaults to unified_nasdaq.csv for QQQ)",
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD) inclusive")
    parser.add_argument("--end", default="2025-09-19", help="End date (YYYY-MM-DD) inclusive")
    parser.add_argument("--fred-series", default="FEDFUNDS", help="FRED rate series (default FEDFUNDS)")
    parser.add_argument(
        "--experiment",
        default="A36",
        choices=sorted(EXPERIMENTS.keys()),
        help="Strategy experiment to run (default A36, best-known)"
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
    parser.add_argument(
        "--disable-buy-temp-limit",
        action="store_true",
        help="Ignore temperature buy blocks so the strategy can add exposure regardless of T",
    )
    parser.add_argument("--save-plot", default=None, help="If set, save output figure to this PNG path")
    parser.add_argument("--save-csv", default=None, help="If set, save daily debug CSV here")
    parser.add_argument(
        "--save-summary",
        default=None,
        help=(
            "If set, write a markdown summary of key stats to this path; "
            "if not provided, a default '[symbol]_[experiment]_summary.md' is saved in the symbol directory"
        ),
    )
    # Debug mode for rebalance decisions
    parser.add_argument("--debug-start", default=None, help="Start date (YYYY-MM-DD) for rebalance debug output")
    parser.add_argument("--debug-end", default=None, help="End date (YYYY-MM-DD) for rebalance debug output (inclusive)")
    parser.add_argument("--debug-csv", default=None, help="If set, write rebalance debug CSV to this path (default strategy_tqqq_reserve_debug.csv if a debug range is provided)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    disable_buy_temp_limit = bool(args.disable_buy_temp_limit)
    experiment = args.experiment.upper()
    config = EXPERIMENTS[experiment]
    normalized_experiment_token = re.sub(r"[^A-Z0-9]+", "", experiment)
    artifact_experiment = None
    for candidate_key in EXPERIMENTS.keys():
        candidate_normalized = re.sub(r"[^A-Z0-9]+", "", candidate_key.upper())
        if candidate_normalized == normalized_experiment_token:
            artifact_experiment = candidate_key
            break
    if artifact_experiment is None:
        artifact_experiment = experiment
    if args.base_symbol:
        base_symbol = args.base_symbol.upper()
    else:
        config_base_symbol = config.get("base_symbol")
        base_symbol = config_base_symbol.upper() if isinstance(config_base_symbol, str) else "QQQ"

    # Optional configuration overrides shared across simulation logic
    temp_allocation_cfg = None
    if isinstance(config.get("temperature_allocation"), Sequence):
        temp_allocation_cfg = list(config.get("temperature_allocation"))

    temperature_model_cfg = config.get("temperature_model")
    if not isinstance(temperature_model_cfg, Mapping):
        temperature_model_cfg = None

    momentum_cfg = config.get("momentum_filters") if isinstance(config.get("momentum_filters"), Mapping) else None
    buy_filter_cfg = None
    sell_filter_cfg = None
    if momentum_cfg:
        buy_cfg = momentum_cfg.get("buy")
        sell_cfg = momentum_cfg.get("sell")
        buy_filter_cfg = buy_cfg if isinstance(buy_cfg, Mapping) else None
        sell_filter_cfg = sell_cfg if isinstance(sell_cfg, Mapping) else None

    def _threshold(cfg: Optional[Mapping[str, object]], key: str, default: Optional[float]) -> Optional[float]:
        if cfg is None or key not in cfg:
            return default
        value = cfg.get(key, default)
        if value is None:
            return None
        return float(value)

    buy_ret3_limit_default = _threshold(buy_filter_cfg, "ret3", -0.03)
    buy_ret6_limit_default = _threshold(buy_filter_cfg, "ret6", -0.03)
    buy_ret12_limit_default = _threshold(buy_filter_cfg, "ret12", -0.02)
    buy_ret22_limit_default = _threshold(buy_filter_cfg, "ret22", 0.0)
    buy_temp_limit_default = _threshold(buy_filter_cfg, "temp", 1.3)
    if disable_buy_temp_limit:
        buy_temp_limit_default = None

    sell_ret3_limit_default = _threshold(sell_filter_cfg, "ret3", 0.03)
    sell_ret6_limit_default = _threshold(sell_filter_cfg, "ret6", 0.03)
    sell_ret12_limit_default = _threshold(sell_filter_cfg, "ret12", 0.0225)
    sell_ret22_limit_default = _threshold(sell_filter_cfg, "ret22", 0.0075)

    rate_taper_cfg = config.get("rate_taper") if isinstance(config.get("rate_taper"), Mapping) else None

    crash_cfg = config.get("crash_derisk") if isinstance(config.get("crash_derisk"), Mapping) else None
    crash_enabled = True
    crash_threshold = -0.1525
    crash_cooldown_days = int(config.get("rebalance_days", 22))
    if crash_cfg:
        crash_enabled = bool(crash_cfg.get("enabled", True))
        crash_threshold_value = crash_cfg.get("threshold", crash_threshold)
        crash_threshold = float(crash_threshold_value) if crash_threshold_value is not None else None
        crash_cooldown_days = int(crash_cfg.get("cooldown_days", crash_cooldown_days))

    rebalance_cadence = int(config.get("rebalance_days", 22))
    if rebalance_cadence <= 0:
        rebalance_cadence = 22

    use_leverage_replication = bool(
        config.get("use_leverage_replication", DEFAULT_USE_LEVERAGE_REPLICATION)
    )

    # Ensure saved plot filenames include unique tokens without duplication.
    # Accepts a root filename (no extension) and appends `token` with an underscore
    # only if the token is not already present as a delimited segment anywhere.
    def ensure_suffix(root: str, token: str) -> str:
        lower_root = root.lower()
        token_lower = token.lower()
        # Split on common delimiters to identify existing segments
        segments = re.split(r"[_\-.]", lower_root)
        if token_lower in segments:
            return root
        return f"{root}_{token}"

    # Determine per-symbol artifact directory
    symbol_dir = os.path.join("symbols", ("qqq" if base_symbol == "QQQ" else base_symbol.lower()))
    os.makedirs(symbol_dir, exist_ok=True)

    if args.save_plot:
        root, ext = os.path.splitext(args.save_plot)
        # If caller didn't specify a directory, write into the per-symbol dir
        if not os.path.dirname(root):
            root = os.path.join(symbol_dir, os.path.basename(root))
        if base_symbol != "QQQ":
            root = ensure_suffix(root, base_symbol)
        root = ensure_suffix(root, artifact_experiment)
        args.save_plot = f"{root}{ext}"
    else:
        prefix = "strategy_qqq_reserve" if base_symbol == "QQQ" else f"strategy_{base_symbol.lower()}_reserve"
        args.save_plot = os.path.join(symbol_dir, f"{prefix}_{artifact_experiment}.png")

    # If a CSV path is provided without a directory, write it under the per-symbol dir
    if args.save_csv:
        csv_root = args.save_csv
        if not os.path.dirname(csv_root):
            args.save_csv = os.path.join(symbol_dir, os.path.basename(csv_root))

    pd, np, plt = import_libs()
    df_full, price_source = load_symbol_history(pd, base_symbol, csv_override=args.csv)
    df = df_full.copy()
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
    rates = get_fred_series(args.fred_series, start_ts, end_ts)
    rates = rates.rename(columns={rates.columns[0]: "rate"}) if rates.shape[1] == 1 else rates
    rates = rates.sort_index()
    rates = rates.reindex(df.index).ffill().bfill()

    # Macro indicators: yield-curve spread (T10Y2Y) and credit spread (BAA10Y)
    yc = get_fred_series("T10Y2Y", start_ts, end_ts)
    yc = yc.sort_index()
    cs = get_fred_series("BAA10Y", start_ts, end_ts)
    cs = cs.sort_index()
    macro = pd.DataFrame({
        "yc_spread": yc["rate"],
        "credit_spread": cs["rate"],
    })
    macro = macro.reindex(df.index).ffill().bfill()
    macro["yc_change_22"] = macro["yc_spread"].diff(22)
    macro["credit_change_22"] = macro["credit_spread"].diff(22)

    # Temperature series (fit cached per symbol)
    A_fit, r_fit, fit_start_ts_full = ensure_temperature_assets(
        pd,
        np,
        plt,
        base_symbol,
        df_full,
        manual_model=temperature_model_cfg,
    )
    temp, A, r, fit_start_ts = compute_temperature_series(
        pd,
        np,
        df,
        csv_path=price_source,
        model_params=(A_fit, r_fit, fit_start_ts_full),
    )
    df["temp"] = temp
    df["temp_ch_11"] = df["temp"] / df["temp"].shift(11) - 1.0
    df["temp_ch_22"] = df["temp"] / df["temp"].shift(21) - 1.0

    # Simulated TQQQ-like and cash daily factors
    leverage = float(config.get("leverage_override", args.leverage))
    annual_fee = float(config.get("annual_fee_override", args.annual_fee))
    borrow_divisor = float(config.get("borrow_divisor_override", args.borrow_divisor))
    borrowed_fraction = leverage - 1.0
    daily_fee = annual_fee / float(args.trading_days)

    rate_ann = rates["rate"].to_numpy()
    rets = df["ret"].to_numpy()
    cash_factor = 1.0 + (rate_ann / 100.0) / float(args.trading_days)
    daily_borrow = borrowed_fraction * ((rate_ann / 100.0) / float(args.trading_days))
    tqqq_factor = (1.0 + leverage * rets) * (1.0 - daily_fee) * (1.0 - (daily_borrow / borrow_divisor))
    underlying_factor = 1.0 + rets

    # Simulation state
    dates = df.index.to_numpy()
    num_days = len(df)
    port_unlevered = np.zeros(num_days, dtype=float)
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
    port_unlevered[0] = 0.0
    port_tqqq[0] = 0.0
    port_cash[0] = initial_capital
    port_total[0] = port_cash[0] + port_tqqq[0] + port_unlevered[0]
    deployed_p[0] = 0.0

    # Rebalance scheduling
    next_rebalance_idx = 0  # day 0 is an eligible rebalance day after updates
    eps = 1e-6
    last_rebalance_idx = -rebalance_cadence
    peak_total = port_total[0]

    rebalance_entries = []

    def add_rebalance_entry(
        index: int,
        new_unlevered: float,
        new_tqqq: float,
        new_cash: float,
        prev_unlevered: float,
        prev_tqqq: float,
        prev_cash: float,
        *,
        reason: str = "rebalance",
    ) -> None:
        trade_eps = 1e-9
        delta_unlevered = float(new_unlevered - prev_unlevered)
        delta_tqqq = float(new_tqqq - prev_tqqq)
        delta_cash = float(new_cash - prev_cash)
        if (
            abs(delta_unlevered) < trade_eps
            and abs(delta_tqqq) < trade_eps
            and abs(delta_cash) < trade_eps
        ):
            return
        total_after = float(new_unlevered + new_tqqq + new_cash)
        if total_after <= 0:
            p_tqqq = 0.0
            p_unlevered = 0.0
        else:
            p_tqqq = float(new_tqqq / total_after)
            p_unlevered = float(new_unlevered / total_after)
        entry_date = pd.Timestamp(dates[index])
        rebalance_entries.append(
            {
                "index": int(index),
                "date": entry_date,
                "p_unlevered": p_unlevered,
                "p_tqqq": p_tqqq,
                "p_cash": 1.0 - p_unlevered - p_tqqq,
                "delta_unlevered": delta_unlevered,
                "delta_tqqq": delta_tqqq,
                "delta_cash": delta_cash,
                "total_after": total_after,
                "unlevered_value": float(new_unlevered),
                "tqqq_value": float(new_tqqq),
                "cash_value": float(new_cash),
                "reason": reason,
            }
        )

    for i in range(num_days):
        # Update holdings based on previous day growth (for i=0 this applies one day of growth)
        if i > 0:
            port_unlevered[i] = port_unlevered[i - 1] * underlying_factor[i]
            port_tqqq[i] = port_tqqq[i - 1] * tqqq_factor[i]
            port_cash[i] = port_cash[i - 1] * cash_factor[i]
        else:
            port_unlevered[i] = port_unlevered[0] * underlying_factor[0]
            port_tqqq[i] = port_tqqq[0] * tqqq_factor[0]
            port_cash[i] = port_cash[0] * cash_factor[0]

        total = port_unlevered[i] + port_tqqq[i] + port_cash[i]
        if total <= 0:
            curr_p = 0.0
        else:
            curr_p = (port_unlevered[i] + leverage * port_tqqq[i]) / (leverage * total)
        drawdown = 0.0 if peak_total <= 0 else (total / peak_total) - 1.0

        T = df["temp"].iloc[i]
        rate_today = float(rate_ann[i])
        base_p = target_allocation_from_temperature(T, temp_allocation_cfg)
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
        rate_tail_cfg = config.get("rate_tailwind_multiplier")
        if rate_tail_cfg:
            base_p = apply_rate_tailwind_multiplier(
                base_p,
                rate_today,
                T,
                r6,
                r22,
                **rate_tail_cfg,
            )
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
        if rate_taper_cfg:
            rt_kwargs = {}
            for key in ("start", "end", "min_allocation", "enabled"):
                if key in rate_taper_cfg:
                    rt_kwargs[key] = rate_taper_cfg[key]
            target_p = apply_rate_taper(base_p, rate_today, **rt_kwargs)
        else:
            target_p = apply_rate_taper(base_p, rate_today)
        shock_cfg = config.get("shock_guard")
        if shock_cfg:
            target_p = apply_shock_guard(
                target_p,
                df["ret"].iloc[i],
                r3,
                r6,
                **shock_cfg,
            )
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
        rebound_cfg = config.get("rebound_accelerator")
        if rebound_cfg:
            target_p = apply_rebound_accelerator(
                target_p,
                T,
                temp_ch11,
                temp_ch22,
                r3,
                r6,
                r22,
                vol22,
                vol66,
                rate_today,
                yc_today,
                cs_today,
                yc_change22,
                cs_change22,
                **rebound_cfg,
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
        accel_cfg = config.get("regime_accelerator")
        if accel_cfg:
            target_p = apply_regime_accelerator(
                target_p,
                T,
                r3,
                r6,
                r12,
                r22,
                vol22,
                vol66,
                rate_today,
                yc_today,
                cs_today,
                yc_change22,
                cs_change22,
                **accel_cfg,
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
            if (
                crash_enabled
                and crash_threshold is not None
                and not math.isnan(ret22)
                and ret22 <= crash_threshold
                and curr_p > 0.0
            ):
                prev_unlevered = port_unlevered[i]
                prev_tqqq = port_tqqq[i]
                prev_cash = port_cash[i]
                curr_p_before_force = curr_p
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
                            "sell_all_threshold": float(crash_threshold),
                            "sell_all_ratio": (float(ret22) / float(crash_threshold)) if crash_threshold not in (0, None) else float('nan'),
                            "temp_T": float(df["temp"].iloc[i]) if not math.isnan(df["temp"].iloc[i]) else float('nan'),
                            "rate_annual_pct": float(rate_ann[i]),
                            "curr_p_before": float(curr_p_before_force),
                            "action": "sell_all_to_cash"
                        })
                prev_total = total
                port_unlevered[i] = 0.0
                port_tqqq[i] = 0.0
                port_cash[i] = prev_total
                total = prev_total
                curr_p = 0.0
                next_rebalance_idx = i + crash_cooldown_days
                forced_derisk_series[i] = 1
                acted = True
                forced_today = True
                last_rebalance_idx = i
                add_rebalance_entry(
                    i,
                    port_unlevered[i],
                    port_tqqq[i],
                    port_cash[i],
                    prev_unlevered,
                    prev_tqqq,
                    prev_cash,
                    reason="forced_derisk",
                )
                # Update debug AFTER action
                if debug_start_ts is not None:
                    ts_i = dates[i]
                    if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                        debug_rows[-1]["curr_p_after"] = float(curr_p)
            if forced_today:
                port_total[i] = port_unlevered[i] + port_tqqq[i] + port_cash[i]
                deployed_p[i] = curr_p
                continue
            # Determine direction
            if target_p > curr_p + eps:
                # Buy-side filters
                block_buy = False
                # Thresholds can be relaxed when the market is very cold
                buy_relax_cfg = config.get("buy_block_relax")
                limit_r3 = buy_ret3_limit_default
                limit_r6 = buy_ret6_limit_default
                limit_r12 = buy_ret12_limit_default
                limit_r22 = buy_ret22_limit_default
                temp_limit = None if disable_buy_temp_limit else buy_temp_limit_default
                if not disable_buy_temp_limit and buy_relax_cfg:
                    temp_threshold_relax = float(buy_relax_cfg.get("temp_threshold", 0.85))
                    if T <= temp_threshold_relax:
                        if "r3_limit" in buy_relax_cfg:
                            value = buy_relax_cfg.get("r3_limit")
                            limit_r3 = None if value is None else float(value)
                        if "r6_limit" in buy_relax_cfg:
                            value = buy_relax_cfg.get("r6_limit")
                            limit_r6 = None if value is None else float(value)
                        if "r12_limit" in buy_relax_cfg:
                            value = buy_relax_cfg.get("r12_limit")
                            limit_r12 = None if value is None else float(value)
                        if "r22_limit" in buy_relax_cfg:
                            value = buy_relax_cfg.get("r22_limit")
                            limit_r22 = None if value is None else float(value)
                        if "temp_limit_cold" in buy_relax_cfg:
                            value = buy_relax_cfg.get("temp_limit_cold")
                            temp_limit = None if value is None else float(value)
                    else:
                        if "temp_limit_default" in buy_relax_cfg:
                            value = buy_relax_cfg.get("temp_limit_default")
                            temp_limit = None if value is None else float(value)
                # r3, r6, r12, r22 already computed above
                trig_buy_r3 = (limit_r3 is not None and not math.isnan(r3) and r3 <= limit_r3)
                trig_buy_r6 = (limit_r6 is not None and not math.isnan(r6) and r6 <= limit_r6)
                trig_buy_r12 = (limit_r12 is not None and not math.isnan(r12) and r12 <= limit_r12)
                trig_buy_r22 = (limit_r22 is not None and not math.isnan(r22) and r22 <= limit_r22)
                trig_buy_T = ((not disable_buy_temp_limit) and temp_limit is not None and T > temp_limit)
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
                            "blocked_reason_buy_T_gt_limit": bool(trig_buy_T),
                            "price": float(df["close"].iloc[i]),
                            "ref_date_22d": prev_date_22,
                            "price_22d_ago": float(prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_abs": float(df["close"].iloc[i] - prev_close_22) if not math.isnan(prev_close_22) else float('nan'),
                            "change_22d_rel": float(r22) if not math.isnan(r22) else float('nan'),
                            "ret_3": float(r3) if not math.isnan(r3) else float('nan'),
                            "ret_6": float(r6) if not math.isnan(r6) else float('nan'),
                            "ret_12": float(r12) if not math.isnan(r12) else float('nan'),
                            "ret_22": float(r22) if not math.isnan(r22) else float('nan'),
                            "limit_buy_r3": float(limit_r3) if limit_r3 is not None else float('nan'),
                            "limit_buy_r6": float(limit_r6) if limit_r6 is not None else float('nan'),
                            "limit_buy_r12": float(limit_r12) if limit_r12 is not None else float('nan'),
                            "limit_buy_r22": float(limit_r22) if limit_r22 is not None else float('nan'),
                            "ratio_buy_r3": (float(r3) / float(limit_r3)) if (limit_r3 is not None and not math.isnan(r3) and limit_r3 != 0.0) else float('nan'),
                            "ratio_buy_r6": (float(r6) / float(limit_r6)) if (limit_r6 is not None and not math.isnan(r6) and limit_r6 != 0.0) else float('nan'),
                            "ratio_buy_r12": (float(r12) / float(limit_r12)) if (limit_r12 is not None and not math.isnan(r12) and limit_r12 != 0.0) else float('nan'),
                            "ratio_buy_r22": (float(r22) / float(limit_r22)) if (limit_r22 is not None and not math.isnan(r22) and limit_r22 != 0.0) else float('nan'),
                            "temp_T": float(T),
                            "base_p": float(base_p),
                            "rate_annual_pct": float(rate_today),
                            "target_p": float(target_p),
                            "curr_p_before": float(curr_p)
                        })
                if not block_buy:
                    prev_unlevered = port_unlevered[i]
                    prev_tqqq = port_tqqq[i]
                    prev_cash = port_cash[i]
                    weight_1x, weight_leveraged, weight_cash = compute_deployment_weights(
                        target_p,
                        leverage,
                        use_replication=use_leverage_replication,
                    )
                    port_unlevered[i] = total * weight_1x
                    port_tqqq[i] = total * weight_leveraged
                    port_cash[i] = total * weight_cash
                    total = port_unlevered[i] + port_tqqq[i] + port_cash[i]
                    if total <= 0:
                        curr_p = 0.0
                    else:
                        curr_p = (
                            port_unlevered[i] + leverage * port_tqqq[i]
                        ) / (leverage * total)
                    next_rebalance_idx = i + rebalance_cadence
                    last_rebalance_idx = i
                    acted = True
                    add_rebalance_entry(
                        i,
                        port_unlevered[i],
                        port_tqqq[i],
                        port_cash[i],
                        prev_unlevered,
                        prev_tqqq,
                        prev_cash,
                    )
                    if debug_start_ts is not None:
                        ts_i = dates[i]
                        if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                            debug_rows[-1]["curr_p_after"] = float(curr_p)
            elif target_p < curr_p - eps:
                # Sell-side filters
                block_sell = False
                # r3, r6, r12, r22 already computed above
                trig_sell_r3 = (
                    sell_ret3_limit_default is not None
                    and not math.isnan(r3)
                    and r3 >= sell_ret3_limit_default
                )
                trig_sell_r6 = (
                    sell_ret6_limit_default is not None
                    and not math.isnan(r6)
                    and r6 >= sell_ret6_limit_default
                )
                trig_sell_r12 = (
                    sell_ret12_limit_default is not None
                    and not math.isnan(r12)
                    and r12 >= sell_ret12_limit_default
                )
                trig_sell_r22 = (
                    sell_ret22_limit_default is not None
                    and not math.isnan(r22)
                    and r22 >= sell_ret22_limit_default
                )
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
                            "limit_sell_r3": float(sell_ret3_limit_default) if sell_ret3_limit_default is not None else float('nan'),
                            "limit_sell_r6": float(sell_ret6_limit_default) if sell_ret6_limit_default is not None else float('nan'),
                            "limit_sell_r12": float(sell_ret12_limit_default) if sell_ret12_limit_default is not None else float('nan'),
                            "limit_sell_r22": float(sell_ret22_limit_default) if sell_ret22_limit_default is not None else float('nan'),
                            "ratio_sell_r3": (float(r3) / float(sell_ret3_limit_default)) if (sell_ret3_limit_default is not None and not math.isnan(r3) and sell_ret3_limit_default != 0.0) else float('nan'),
                            "ratio_sell_r6": (float(r6) / float(sell_ret6_limit_default)) if (sell_ret6_limit_default is not None and not math.isnan(r6) and sell_ret6_limit_default != 0.0) else float('nan'),
                            "ratio_sell_r12": (float(r12) / float(sell_ret12_limit_default)) if (sell_ret12_limit_default is not None and not math.isnan(r12) and sell_ret12_limit_default != 0.0) else float('nan'),
                            "ratio_sell_r22": (float(r22) / float(sell_ret22_limit_default)) if (sell_ret22_limit_default is not None and not math.isnan(r22) and sell_ret22_limit_default != 0.0) else float('nan'),
                            "temp_T": float(T),
                            "base_p": float(base_p),
                            "rate_annual_pct": float(rate_today),
                            "target_p": float(target_p),
                            "curr_p_before": float(curr_p)
                        })
                if not block_sell:
                    prev_unlevered = port_unlevered[i]
                    prev_tqqq = port_tqqq[i]
                    prev_cash = port_cash[i]
                    weight_1x, weight_leveraged, weight_cash = compute_deployment_weights(
                        target_p,
                        leverage,
                        use_replication=use_leverage_replication,
                    )
                    port_unlevered[i] = total * weight_1x
                    port_tqqq[i] = total * weight_leveraged
                    port_cash[i] = total * weight_cash
                    total = port_unlevered[i] + port_tqqq[i] + port_cash[i]
                    if total <= 0:
                        curr_p = 0.0
                    else:
                        curr_p = (
                            port_unlevered[i] + leverage * port_tqqq[i]
                        ) / (leverage * total)
                    next_rebalance_idx = i + rebalance_cadence
                    last_rebalance_idx = i
                    acted = True
                    add_rebalance_entry(
                        i,
                        port_unlevered[i],
                        port_tqqq[i],
                        port_cash[i],
                        prev_unlevered,
                        prev_tqqq,
                        prev_cash,
                    )
                    if debug_start_ts is not None:
                        ts_i = dates[i]
                        if ts_i >= debug_start_ts and ts_i <= debug_end_ts:
                            debug_rows[-1]["curr_p_after"] = float(curr_p)
            else:
                # No significant change, but treat as a completed rebalance to keep cadence
                next_rebalance_idx = i + rebalance_cadence
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

        port_total[i] = port_unlevered[i] + port_tqqq[i] + port_cash[i]
        deployed_p[i] = curr_p
        peak_total = max(peak_total, port_total[i])

    # Normalized lines for plotting
    unified_norm = df["close"].to_numpy() / df["close"].iloc[0]
    strategy_norm = port_total / port_total[0]

    # CAGR for strategy
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (strategy_norm[-1] / strategy_norm[0]) ** (1.0 / years) - 1.0
    # Max drawdown for strategy (based on total portfolio)
    # Avoid division by zero by guarding initial value
    if len(port_total) > 0 and port_total[0] > 0:
        running_max = np.maximum.accumulate(port_total)
        drawdowns = (port_total / running_max) - 1.0
        max_drawdown = float(np.nanmin(drawdowns)) if drawdowns.size > 0 else float('nan')
    else:
        max_drawdown = float('nan')
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
            display_df["unlevered_pct"] = display_df["p_unlevered"] * 100.0
            display_df["tbill_pct"] = display_df["p_cash"] * 100.0
            display_df["portfolio_value"] = display_df["total_after"]
            display_df["cagr_pct"] = display_df["cagr"] * 100.0
            trade_eps = 1e-9

            def classify_action(row: pd.Series) -> str:
                exposure_delta = float(row.get("delta_unlevered", 0.0)) + leverage * float(
                    row.get("delta_tqqq", 0.0)
                )
                if exposure_delta > trade_eps:
                    return "Buy"
                if exposure_delta < -trade_eps:
                    return "Sell"
                return "Hold"

            display_df["action"] = display_df.apply(classify_action, axis=1)

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
                "unlevered_pct",
                "tbill_pct",
                "delta_unlevered",
                "delta_tqqq",
                "delta_cash",
                "portfolio_value",
                "cagr_pct",
            ]
            formatters = {
                "tqqq_pct": fmt_pct,
                "unlevered_pct": fmt_pct,
                "tbill_pct": fmt_pct,
                "delta_unlevered": fmt_amt,
                "delta_tqqq": fmt_amt,
                "delta_cash": fmt_amt,
                "portfolio_value": fmt_amt,
                "cagr_pct": fmt_pct,
            }
            print("Rebalance log:")
            print(display_df[columns].to_string(index=False, formatters=formatters))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    underlying_label = "Unified Nasdaq" if base_symbol == "QQQ" else f"{base_symbol} price"
    # Compute unlevered underlying CAGR over the plotted span
    years_span = (df.index[-1] - df.index[0]).days / 365.25
    underlying_cagr = (unified_norm[-1] / unified_norm[0]) ** (1.0 / years_span) - 1.0 if years_span > 0 else float('nan')

    ax1.plot(df.index, unified_norm, label=f"{underlying_label} (CAGR {underlying_cagr*100.0:.2f}%)", color="#1f77b4", alpha=0.7)
    ax1.plot(df.index, strategy_norm, label=f"Strategy (CAGR {cagr*100.0:.2f}%)", color="#d62728")
    ax1.set_title(f"{underlying_label} vs Strategy Portfolio Value (start=1.0)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Value (normalized, log)")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="upper left")

    ax2.plot(df.index, df["temp"], label="Temperature", color="black")
    ax2.axhline(1.0, color="gray", linestyle="-", alpha=0.8)
    ax2.axhline(1.5, color="gray", linestyle=":", alpha=0.7)
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
    exposure_label = "TQQQ" if base_symbol == "QQQ" else f"leveraged {base_symbol}"
    ax2.plot(df.index, deployed_p, label=f"Proportion in {exposure_label}", color="green", alpha=0.8)
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
        out["port_unlevered"] = port_unlevered
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

    # Write summary markdown
    try:
        symbol_upper = base_symbol.upper()
        symbol_dir = os.path.join("symbols", ("qqq" if base_symbol == "QQQ" else base_symbol.lower()))
        os.makedirs(symbol_dir, exist_ok=True)
        default_summary_name = f"{base_symbol.lower()}_{artifact_experiment}_summary.md"
        summary_path = args.save_summary if args.save_summary else os.path.join(symbol_dir, default_summary_name)

        # Fitted curve CAGR comes from r (annualised growth of fitted curve)
        fitted_curve_cagr = float(r)
        # Underlying CAGR already computed above as underlying_cagr
        # Strategy CAGR is cagr; max drawdown computed as max_drawdown
        # Rebalance count
        rebalance_count = int(len(rebalance_entries))

        fundamentals = ensure_fundamentals(symbol_upper, symbol_dir)

        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(f"# {symbol_upper} – Strategy {args.experiment} Summary\n\n")
            fh.write(f"- **Span**: {df.index[0].date()} → {df.index[-1].date()} ({years:.2f} years)\n")
            fh.write(f"- **Underlying ({underlying_label}) CAGR**: {underlying_cagr * 100.0:.2f}%\n")
            fh.write(f"- **Fitted curve CAGR**: {fitted_curve_cagr * 100.0:.2f}%\n")
            fh.write(f"- **Strategy CAGR**: {cagr * 100.0:.2f}%\n")
            fh.write(f"- **Max drawdown**: {max_drawdown * 100.0:.2f}%\n")
            fh.write(f"- **Rebalances executed**: {rebalance_count}\n")
            for line in fundamentals.summary_lines():
                fh.write(line + "\n")
            fh.write("\n")
            fh.write("Notes:\n\n")
            fh.write("- Temperatures and anchors per experiment govern deployment; see EXPERIMENTS.md for details.\n")
            fh.write("- Figures use simulated leveraged sleeve with fees and borrow costs.\n")
    except Exception:
        # Non-fatal: continue even if summary cannot be written
        pass
    # Rebalance debug CSV output
    if (args.debug_start or args.debug_end) and len(debug_rows) > 0:
        debug_df = pd.DataFrame(debug_rows)
        debug_df = debug_df.sort_values("date")
        if args.debug_csv:
            debug_path = args.debug_csv
        else:
            default_debug = (
                "strategy_qqq_reserve_debug.csv"
                if base_symbol == "QQQ"
                else f"strategy_{base_symbol.lower()}_reserve_debug.csv"
            )
            symbol_dir = os.path.join("symbols", ("qqq" if base_symbol == "QQQ" else base_symbol.lower()))
            os.makedirs(symbol_dir, exist_ok=True)
            debug_path = os.path.join(symbol_dir, default_debug)
        debug_df.to_csv(debug_path, index=False)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()







