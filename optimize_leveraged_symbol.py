#!/usr/bin/env python3
"""General optimiser for A1g-style leveraged symbol experiments.

This module abstracts the GOOG2x optimiser so it can be reused for other
underlyings and leverage settings.  It keeps the original perturb/sampling
logic but parameterises the data-loading, leverage assumptions, and baseline
experiment.  The CLI prints a JSON payload suitable for dropping directly
into ``strategy_tqqq_reserve.EXPERIMENTS``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

import strategy_tqqq_reserve as strategy


@dataclass
class SymbolEnvironment:
    symbol: str
    dates: np.ndarray
    ret: np.ndarray
    ret_3: np.ndarray
    ret_6: np.ndarray
    ret_12: np.ndarray
    ret_22: np.ndarray
    temperature: np.ndarray
    rate_annual: np.ndarray
    cash_factor: np.ndarray
    leveraged_factor: np.ndarray
    start: np.datetime64
    end: np.datetime64


@dataclass
class SimulationResult:
    symbol: str
    cagr: float
    final_value: float
    max_drawdown: float


@dataclass
class GlobalParams:
    anchors: List[tuple[float, float]]
    buy_thresholds: Dict[str, Optional[float]]
    sell_thresholds: Dict[str, Optional[float]]
    buy_temp_limit: Optional[float]
    rate_start: float
    rate_end: float
    rate_min_allocation: float
    rate_enabled: bool
    crash_enabled: bool
    crash_threshold: Optional[float]
    rebalance_days: int = 22

    def to_config(self) -> Dict[str, object]:
        anchors_payload = [
            {"temp": float(temp), "allocation": float(alloc)} for temp, alloc in self.anchors
        ]
        buy_cfg: Dict[str, Optional[float]] = {
            key: (None if value is None else float(value))
            for key, value in self.buy_thresholds.items()
        }
        if self.buy_temp_limit is not None or "temp" in buy_cfg:
            buy_cfg["temp"] = None if self.buy_temp_limit is None else float(self.buy_temp_limit)
        sell_cfg: Dict[str, Optional[float]] = {
            key: (None if value is None else float(value))
            for key, value in self.sell_thresholds.items()
        }
        config: Dict[str, object] = {
            "temperature_allocation": anchors_payload,
            "momentum_filters": {
                "buy": buy_cfg,
                "sell": sell_cfg,
            },
            "rate_taper": {
                "start": float(self.rate_start),
                "end": float(self.rate_end),
                "min_allocation": float(self.rate_min_allocation),
                "enabled": bool(self.rate_enabled),
            },
            "crash_derisk": {
                "enabled": bool(self.crash_enabled),
                "threshold": None if self.crash_threshold is None else float(self.crash_threshold),
                "cooldown_days": int(self.rebalance_days),
            },
            "rebalance_days": int(self.rebalance_days),
        }
        return config


@dataclass
class OptimiserSettings:
    symbol: str
    leverage: float
    annual_fee: float
    borrow_divisor: float
    trading_days: int
    end_date: str
    fred_series: str
    use_temperature_cache: bool
    fit_threshold_2: float
    fit_threshold_3: float
    csv_override: Optional[str] = None


def _anchors_to_payload(anchors: Sequence[tuple[float, float]]) -> List[Mapping[str, float]]:
    return [{"temp": float(temp), "allocation": float(alloc)} for temp, alloc in anchors]


def _local_symbol_csv(symbol: str) -> Optional[str]:
    """Return a repository CSV path for the given symbol if available."""

    upper = symbol.upper()
    cache_path = os.path.join("symbol_data", f"{upper}.csv")
    if os.path.exists(cache_path):
        return cache_path

    return None


def _load_baseline_params(experiment: str) -> GlobalParams:
    if experiment not in strategy.EXPERIMENTS:
        known = ", ".join(sorted(strategy.EXPERIMENTS))
        raise KeyError(f"Experiment '{experiment}' not found; known keys: {known}")

    cfg = strategy.EXPERIMENTS[experiment]

    anchors = [
        (float(anchor["temp"]), float(anchor["allocation"]))
        for anchor in cfg.get("temperature_allocation", [])
    ]
    momentum_cfg = cfg.get("momentum_filters", {})
    buy_cfg = dict(momentum_cfg.get("buy", {})) if isinstance(momentum_cfg, Mapping) else {}
    sell_cfg = dict(momentum_cfg.get("sell", {})) if isinstance(momentum_cfg, Mapping) else {}

    buy_thresholds = {
        key: (None if value is None else float(value))
        for key, value in buy_cfg.items()
        if key in {"ret3", "ret6", "ret12", "ret22"}
    }
    buy_temp_limit = buy_cfg.get("temp")
    buy_temp_limit_value = None if buy_temp_limit is None else float(buy_temp_limit)

    sell_thresholds = {
        key: (None if value is None else float(value))
        for key, value in sell_cfg.items()
        if key in {"ret3", "ret6", "ret12", "ret22"}
    }

    rate_cfg = dict(cfg.get("rate_taper", {}))
    rate_start = float(rate_cfg.get("start", 10.0))
    rate_end = float(rate_cfg.get("end", 12.0))
    rate_min_allocation = float(rate_cfg.get("min_allocation", 0.0))
    rate_enabled = bool(rate_cfg.get("enabled", True))

    crash_cfg = dict(cfg.get("crash_derisk", {}))
    crash_enabled = bool(crash_cfg.get("enabled", True))
    crash_threshold_raw = crash_cfg.get("threshold", None)
    crash_threshold = None if crash_threshold_raw is None else float(crash_threshold_raw)
    cooldown_days = int(cfg.get("rebalance_days", crash_cfg.get("cooldown_days", 22)))

    return GlobalParams(
        anchors=anchors,
        buy_thresholds=buy_thresholds,
        sell_thresholds=sell_thresholds,
        buy_temp_limit=buy_temp_limit_value,
        rate_start=rate_start,
        rate_end=rate_end,
        rate_min_allocation=rate_min_allocation,
        rate_enabled=rate_enabled,
        crash_enabled=crash_enabled,
        crash_threshold=crash_threshold,
        rebalance_days=cooldown_days,
    )


def _fit_temperature_parameters(pd, symbol: str, df_full, *, thresh2: float, thresh3: float):
    start_ts = pd.Timestamp(df_full.index.min())
    t_years = (df_full.index - start_ts).days / 365.25
    t_array: Iterable[float]
    if hasattr(t_years, "to_numpy"):
        t_array = t_years.to_numpy()
    else:  # pragma: no cover - legacy pandas fallback
        t_array = t_years
    prices = df_full["close"].to_numpy(dtype=float)
    result = strategy.iterative_constant_growth(t_array, prices, thresholds=[thresh2, thresh3])
    final = result.final
    return float(final.A), float(final.r), pd.Timestamp(start_ts)


def prepare_symbol_environment(symbol: str, settings: OptimiserSettings) -> SymbolEnvironment:
    pd, np_mod, plt = strategy.import_libs()
    np_local = np_mod

    csv_override = settings.csv_override or _local_symbol_csv(symbol)
    df_full_raw, price_source = strategy.load_symbol_history(
        pd, symbol, csv_override=csv_override
    )
    df_full_raw = df_full_raw.copy()
    end_ts = pd.to_datetime(settings.end_date)
    df_trimmed = df_full_raw.loc[df_full_raw.index <= end_ts]
    if df_trimmed.empty:
        raise RuntimeError(f"No price data available for {symbol}")

    rates = strategy.get_fred_series(settings.fred_series, df_trimmed.index.min(), df_trimmed.index.max())
    rates = rates.rename(columns={rates.columns[0]: "rate"}) if rates.shape[1] == 1 else rates
    rates.index = pd.to_datetime(rates.index)
    rates = rates.sort_index()

    df = df_trimmed.copy()
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["ret_3"] = df["close"] / df["close"].shift(2) - 1.0
    df["ret_6"] = df["close"] / df["close"].shift(5) - 1.0
    df["ret_12"] = df["close"] / df["close"].shift(11) - 1.0
    df["ret_22"] = df["close"] / df["close"].shift(21) - 1.0

    symbol_rates = rates.reindex(df.index).ffill().bfill()
    rate_ann = symbol_rates["rate"].to_numpy(dtype=float)

    if settings.use_temperature_cache:
        A_fit, r_fit, fit_start = strategy.ensure_temperature_assets(
            pd, np_local, plt, symbol, df_full_raw
        )
    else:
        A_fit, r_fit, fit_start = _fit_temperature_parameters(
            pd,
            symbol,
            df_full_raw,
            thresh2=settings.fit_threshold_2,
            thresh3=settings.fit_threshold_3,
        )
    temp, *_ = strategy.compute_temperature_series(
        pd,
        np_local,
        df,
        csv_path=price_source,
        model_params=(A_fit, r_fit, fit_start),
    )
    df["temp"] = temp

    rets = df["ret"].to_numpy(dtype=float)
    borrowed_fraction = settings.leverage - 1.0
    daily_fee = settings.annual_fee / settings.trading_days
    daily_borrow = borrowed_fraction * ((rate_ann / 100.0) / settings.trading_days)
    cash_factor = 1.0 + (rate_ann / 100.0) / settings.trading_days
    leveraged_factor = (
        (1.0 + settings.leverage * rets)
        * (1.0 - daily_fee)
        * (1.0 - (daily_borrow / settings.borrow_divisor))
    )

    return SymbolEnvironment(
        symbol=symbol,
        dates=df.index.to_numpy(),
        ret=rets,
        ret_3=df["ret_3"].to_numpy(dtype=float),
        ret_6=df["ret_6"].to_numpy(dtype=float),
        ret_12=df["ret_12"].to_numpy(dtype=float),
        ret_22=df["ret_22"].to_numpy(dtype=float),
        temperature=df["temp"].to_numpy(dtype=float),
        rate_annual=rate_ann,
        cash_factor=cash_factor,
        leveraged_factor=leveraged_factor,
        start=df.index[0].to_datetime64(),
        end=df.index[-1].to_datetime64(),
    )


def simulate_symbol(env: SymbolEnvironment, params: GlobalParams) -> SimulationResult:
    anchors_payload = _anchors_to_payload(params.anchors)
    n_days = env.dates.shape[0]
    port_lev = np.zeros(n_days, dtype=float)
    port_cash = np.zeros(n_days, dtype=float)
    port_total = np.zeros(n_days, dtype=float)

    port_lev[0] = 0.0
    port_cash[0] = 1.0
    port_total[0] = port_cash[0]

    next_rebalance = 0
    eps = 1e-6
    peak_total = port_total[0]
    buy_thresholds = params.buy_thresholds
    sell_thresholds = params.sell_thresholds

    for i in range(n_days):
        if i > 0:
            port_lev[i] = port_lev[i - 1] * env.leveraged_factor[i]
            port_cash[i] = port_cash[i - 1] * env.cash_factor[i]
        else:
            port_lev[i] = port_lev[0] * env.leveraged_factor[0]
            port_cash[i] = port_cash[0] * env.cash_factor[0]

        total = port_lev[i] + port_cash[i]
        curr_p = 0.0 if total <= 0 else port_lev[i] / total

        T = env.temperature[i]
        rate_today = env.rate_annual[i]
        base_p = strategy.target_allocation_from_temperature(T, anchors_payload)
        target_p = strategy.apply_rate_taper(
            base_p,
            rate_today,
            start=params.rate_start,
            end=params.rate_end,
            min_allocation=params.rate_min_allocation,
            enabled=params.rate_enabled,
        )

        if i >= next_rebalance:
            ret22 = env.ret_22[i]
            if (
                params.crash_enabled
                and params.crash_threshold is not None
                and not math.isnan(ret22)
                and ret22 <= params.crash_threshold
                and port_lev[i] > 0
            ):
                port_cash[i] = total
                port_lev[i] = 0.0
                total = port_cash[i]
                curr_p = 0.0
                next_rebalance = i + params.rebalance_days
            elif target_p > curr_p + eps:
                limit_r3 = buy_thresholds.get("ret3")
                limit_r6 = buy_thresholds.get("ret6")
                limit_r12 = buy_thresholds.get("ret12")
                limit_r22 = buy_thresholds.get("ret22")
                temp_limit = params.buy_temp_limit

                trig_r3 = limit_r3 is not None and not math.isnan(env.ret_3[i]) and env.ret_3[i] <= limit_r3
                trig_r6 = limit_r6 is not None and not math.isnan(env.ret_6[i]) and env.ret_6[i] <= limit_r6
                trig_r12 = limit_r12 is not None and not math.isnan(env.ret_12[i]) and env.ret_12[i] <= limit_r12
                trig_r22 = limit_r22 is not None and not math.isnan(env.ret_22[i]) and env.ret_22[i] <= limit_r22
                trig_temp = temp_limit is not None and T > temp_limit

                if not (trig_r3 or trig_r6 or trig_r12 or trig_r22 or trig_temp):
                    port_lev[i] = total * target_p
                    port_cash[i] = total - port_lev[i]
                    total = port_lev[i] + port_cash[i]
                    curr_p = target_p
                    next_rebalance = i + params.rebalance_days
            elif target_p < curr_p - eps:
                trig_r3 = (
                    sell_thresholds.get("ret3") is not None
                    and not math.isnan(env.ret_3[i])
                    and env.ret_3[i] >= sell_thresholds["ret3"]
                )
                trig_r6 = (
                    sell_thresholds.get("ret6") is not None
                    and not math.isnan(env.ret_6[i])
                    and env.ret_6[i] >= sell_thresholds["ret6"]
                )
                trig_r12 = (
                    sell_thresholds.get("ret12") is not None
                    and not math.isnan(env.ret_12[i])
                    and env.ret_12[i] >= sell_thresholds["ret12"]
                )
                trig_r22 = (
                    sell_thresholds.get("ret22") is not None
                    and not math.isnan(env.ret_22[i])
                    and env.ret_22[i] >= sell_thresholds["ret22"]
                )

                if not (trig_r3 or trig_r6 or trig_r12 or trig_r22):
                    port_lev[i] = total * target_p
                    port_cash[i] = total - port_lev[i]
                    total = port_lev[i] + port_cash[i]
                    curr_p = target_p
                    next_rebalance = i + params.rebalance_days
            else:
                next_rebalance = i + params.rebalance_days

        port_total[i] = port_lev[i] + port_cash[i]
        if port_total[i] > peak_total:
            peak_total = port_total[i]

    initial_value = float(port_total[0]) if port_total[0] > 0 else 1.0
    final_value = float(port_total[-1])
    years = (env.end - env.start).astype("timedelta64[D]").astype(float) / 365.25
    if final_value <= 0 or years <= 0:
        cagr = -1.0
    else:
        cagr = (final_value / initial_value) ** (1.0 / years) - 1.0
    running_max = np.maximum.accumulate(port_total)
    drawdowns = np.where(running_max > 0, (port_total / running_max) - 1.0, 0.0)
    max_drawdown = float(np.nanmin(drawdowns)) if drawdowns.size > 0 else float("nan")

    return SimulationResult(symbol=env.symbol, cagr=float(cagr), final_value=final_value, max_drawdown=max_drawdown)


def evaluate_params(env: SymbolEnvironment, params: GlobalParams) -> SimulationResult:
    return simulate_symbol(env, params)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def perturb_parameters(base: GlobalParams, rng: np.random.Generator) -> GlobalParams:
    low_temp = _clamp(base.anchors[0][0] + rng.normal(0.0, 0.01), 0.82, 0.96)
    high_temp = _clamp(base.anchors[-1][0] + rng.normal(0.0, 0.02), 1.28, 1.65)
    low_alloc = _clamp(base.anchors[0][1] + rng.normal(0.0, 0.05), 0.9, 1.5)
    mid_alloc = base.anchors[1][1] + rng.normal(0.0, 0.05)
    mid_alloc = _clamp(mid_alloc, 0.45, min(low_alloc - 0.05, 1.1))
    high_alloc = base.anchors[-1][1] + rng.normal(0.0, 0.04)
    high_alloc = _clamp(high_alloc, 0.05, min(mid_alloc - 0.05, 0.6))
    anchors = [(low_temp, low_alloc), (1.0, mid_alloc), (high_temp, high_alloc)]

    def _perturb_threshold(value: Optional[float], scale: float) -> Optional[float]:
        if value is None:
            return None
        sigma = abs(value) * scale + 0.002
        return float(value + rng.normal(0.0, sigma))

    buy_thresholds = {
        key: _perturb_threshold(base.buy_thresholds.get(key), 0.2)
        for key in ("ret3", "ret6", "ret12", "ret22")
    }
    sell_thresholds = {
        key: _perturb_threshold(base.sell_thresholds.get(key), 0.15)
        for key in ("ret3", "ret6", "ret12", "ret22")
    }

    buy_temp_limit = base.buy_temp_limit
    if rng.random() < 0.6:
        if buy_temp_limit is not None:
            buy_temp_limit = _clamp(buy_temp_limit + rng.normal(0.0, 0.03), 1.15, 1.5)
        else:
            buy_temp_limit = _clamp(1.3 + rng.normal(0.0, 0.05), 1.15, 1.5)
    elif rng.random() < 0.15:
        buy_temp_limit = None

    rate_start = _clamp(base.rate_start + rng.normal(0.0, 0.2), 8.5, 11.5)
    rate_end = _clamp(base.rate_end + rng.normal(0.0, 0.25), rate_start + 0.5, rate_start + 3.0)
    rate_min_allocation = _clamp(base.rate_min_allocation + rng.normal(0.0, 0.03), 0.0, 0.35)

    crash_enabled = base.crash_enabled if rng.random() < 0.9 else not base.crash_enabled
    crash_threshold = base.crash_threshold
    if crash_enabled:
        if crash_threshold is None:
            crash_threshold = -0.1525
        crash_threshold = _clamp(crash_threshold + rng.normal(0.0, 0.01), -0.20, -0.10)
    else:
        crash_threshold = None

    return GlobalParams(
        anchors=anchors,
        buy_thresholds=buy_thresholds,
        sell_thresholds=sell_thresholds,
        buy_temp_limit=buy_temp_limit,
        rate_start=rate_start,
        rate_end=rate_end,
        rate_min_allocation=rate_min_allocation,
        rate_enabled=base.rate_enabled,
        crash_enabled=crash_enabled,
        crash_threshold=crash_threshold,
        rebalance_days=base.rebalance_days,
    )


def sample_parameters(rng: np.random.Generator) -> GlobalParams:
    low_temp = rng.uniform(0.85, 0.95)
    mid_temp = 1.0
    high_temp = rng.uniform(1.32, 1.58)
    low_alloc = rng.uniform(1.0, 1.4)
    mid_alloc = rng.uniform(0.55, min(low_alloc, 1.05))
    high_alloc = rng.uniform(0.05, min(mid_alloc, 0.45))
    anchors = [(low_temp, low_alloc), (mid_temp, mid_alloc), (high_temp, high_alloc)]

    buy_thresholds = {
        "ret3": -0.03 * rng.uniform(0.7, 1.3),
        "ret6": -0.03 * rng.uniform(0.7, 1.3),
        "ret12": -0.02 * rng.uniform(0.7, 1.3),
        "ret22": rng.uniform(-0.01, 0.015),
    }
    if rng.random() < 0.15:
        buy_thresholds["ret22"] = None

    sell_thresholds = {
        "ret3": 0.03 * rng.uniform(0.7, 1.3),
        "ret6": 0.03 * rng.uniform(0.7, 1.3),
        "ret12": 0.0225 * rng.uniform(0.7, 1.3),
        "ret22": 0.0075 * rng.uniform(0.7, 1.3),
    }

    buy_temp_limit = rng.uniform(1.2, 1.45)
    if rng.random() < 0.1:
        buy_temp_limit = None

    rate_start = rng.uniform(9.5, 11.0)
    rate_end = rng.uniform(rate_start + 0.8, rate_start + 2.8)
    rate_min_allocation = rng.uniform(0.0, 0.25)
    rate_enabled = True

    crash_enabled = rng.random() < 0.75
    crash_threshold = rng.uniform(-0.18, -0.12) if crash_enabled else None

    return GlobalParams(
        anchors=anchors,
        buy_thresholds=buy_thresholds,
        sell_thresholds=sell_thresholds,
        buy_temp_limit=buy_temp_limit,
        rate_start=rate_start,
        rate_end=rate_end,
        rate_min_allocation=rate_min_allocation,
        rate_enabled=rate_enabled,
        crash_enabled=crash_enabled,
        crash_threshold=crash_threshold,
        rebalance_days=22,
    )


def optimise(
    env: SymbolEnvironment,
    iterations: int,
    seed: int,
    *,
    baseline_params: GlobalParams,
    baseline_result: SimulationResult,
) -> tuple[GlobalParams, SimulationResult]:
    rng = np.random.default_rng(seed)
    best_params = baseline_params
    best_result = baseline_result

    try:
        for idx in range(iterations):
            if best_params is not None and rng.random() < 0.65:
                params = perturb_parameters(best_params, rng)
            else:
                params = sample_parameters(rng)
            result = evaluate_params(env, params)
            if result.cagr > best_result.cagr:
                best_result = result
                best_params = params
                print(f"[{idx + 1:03d}/{iterations}] New best CAGR: {result.cagr * 100:.2f}%")
    except KeyboardInterrupt:
        print("\nOptimisation interrupted; returning best parameters found so far.")

    return best_params, best_result


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimise A1g-style parameters for a leveraged symbol",
    )
    parser.add_argument(
        "--symbol", default="GOOGL", help="Underlying symbol to optimise (default GOOGL)"
    )
    parser.add_argument(
        "--iterations", type=int, default=120, help="Number of random samples to evaluate (default 120)"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility (default 1)")
    parser.add_argument(
        "--leverage", type=float, default=2.0, help="Leveraged sleeve factor to simulate (default 2.0)"
    )
    parser.add_argument(
        "--annual-fee",
        type=float,
        default=0.0095,
        help="Annual fee applied to the leveraged product (default 0.0095)",
    )
    parser.add_argument(
        "--borrow-divisor",
        type=float,
        default=0.7,
        help="Borrow divisor used to approximate financing costs (default 0.7)",
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=252,
        help="Number of trading days per year for fee/rate accrual (default 252)",
    )
    parser.add_argument(
        "--end-date",
        default="2025-09-19",
        help="Use price data up to and including this date (default 2025-09-19)",
    )
    parser.add_argument(
        "--fred-series",
        default="FEDFUNDS",
        help="FRED series to use for the short-term rate (default FEDFUNDS)",
    )
    parser.add_argument(
        "--baseline",
        default="A1g",
        help="Baseline experiment name to perturb (default A1g)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV override with date/close columns for the underlying symbol",
    )
    parser.add_argument(
        "--use-temperature-cache",
        action="store_true",
        help="Reuse cached temperature fits and plots when available (default: fit fresh without saving)",
    )
    parser.add_argument(
        "--fit-threshold-2",
        type=float,
        default=0.35,
        help="Second-stage outlier rejection threshold for temperature fit (default 0.35)",
    )
    parser.add_argument(
        "--fit-threshold-3",
        type=float,
        default=0.15,
        help="Third-stage outlier rejection threshold for temperature fit (default 0.15)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    settings = OptimiserSettings(
        symbol=symbol,
        leverage=float(args.leverage),
        annual_fee=float(args.annual_fee),
        borrow_divisor=float(args.borrow_divisor),
        trading_days=int(args.trading_days),
        end_date=str(args.end_date),
        fred_series=str(args.fred_series),
        use_temperature_cache=bool(args.use_temperature_cache),
        fit_threshold_2=float(args.fit_threshold_2),
        fit_threshold_3=float(args.fit_threshold_3),
        csv_override=args.csv,
    )

    print(f"Preparing data for {symbol} with {settings.leverage:.2f}x leverage ...")
    env = prepare_symbol_environment(symbol, settings)
    baseline_params = _load_baseline_params(str(args.baseline))
    baseline_result = evaluate_params(env, baseline_params)
    print(f"Baseline {args.baseline} CAGR ({settings.leverage:.1f}x): {format_percent(baseline_result.cagr)}")
    print("Running optimisation ...")
    params, best_result = optimise(
        env,
        iterations=int(args.iterations),
        seed=int(args.seed),
        baseline_params=baseline_params,
        baseline_result=baseline_result,
    )

    print()
    print(f"Best CAGR discovered: {format_percent(best_result.cagr)}")
    print(f"Final value multiple : {best_result.final_value:.2f}x")
    print(f"Max drawdown        : {best_result.max_drawdown * 100:.2f}%")
    print()
    leverage_label = f"{settings.leverage:.1f}x".rstrip("0").rstrip(".")
    print(f"Parameter configuration for {symbol} @ {leverage_label}:")
    config_payload = params.to_config()
    config_payload["leverage_override"] = settings.leverage
    print(json.dumps(config_payload, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
