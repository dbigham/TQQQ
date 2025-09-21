#!/usr/bin/env python3
"""Global optimizer for the baseline A1 strategy parameters.

The optimizer searches across temperature allocation anchors, momentum
filters, rate-taper thresholds, and crash protections to maximise the
average CAGR across the curated symbol list used for the A1 experiment.
It produces a candidate configuration for the new A1g experiment and
prints a comparison table between the original A1 CAGRs and the tuned
results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

import strategy_tqqq_reserve as strategy
from a1_symbol_stats import A1_SYMBOL_STATS


TRADING_DAYS = 252
LEVERAGE = 3.0
ANNUAL_FEE = 0.0095
BORROW_DIVISOR = 0.7
END_DATE = "2025-01-10"


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
    tqqq_factor: np.ndarray
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


def _anchors_to_payload(anchors: Sequence[tuple[float, float]]) -> List[Mapping[str, float]]:
    return [{"temp": float(temp), "allocation": float(alloc)} for temp, alloc in anchors]


def _local_symbol_csv(symbol: str) -> Optional[str]:
    """Return a repository CSV path for the given symbol if available."""

    upper = symbol.upper()
    cache_path = os.path.join("symbol_data", f"{upper}.csv")
    if os.path.exists(cache_path):
        return cache_path

    return None


def prepare_symbol_environments(symbols: Sequence[str]) -> List[SymbolEnvironment]:
    pd, np_mod, plt = strategy.import_libs()
    np_local = np_mod  # Alias to avoid shadowing

    # Load full histories first to determine the global rate span.
    histories: Dict[str, tuple[object, object, str]] = {}
    global_start = None
    global_end = None
    end_ts = pd.to_datetime(END_DATE)
    for symbol in symbols:
        csv_override = _local_symbol_csv(symbol)
        df_full_raw, price_source = strategy.load_symbol_history(
            pd, symbol, csv_override=csv_override
        )
        df_full_raw = df_full_raw.copy()
        df_trimmed = df_full_raw.loc[df_full_raw.index <= end_ts]
        histories[symbol] = (df_trimmed, df_full_raw, price_source)
        if df_trimmed.empty:
            raise RuntimeError(f"No price data available for {symbol}")
        start = pd.Timestamp(df_trimmed.index.min())
        end = pd.Timestamp(df_trimmed.index.max())
        global_start = start if global_start is None or start < global_start else global_start
        global_end = end if global_end is None or end > global_end else global_end

    rates = strategy.fetch_fred_series(pd, "FEDFUNDS", global_start, global_end)
    rates = rates.rename(columns={rates.columns[0]: "rate"}) if rates.shape[1] == 1 else rates
    rates.index = pd.to_datetime(rates.index)
    rates = rates.sort_index()

    envs: List[SymbolEnvironment] = []
    for symbol in symbols:
        df_trimmed, df_full_raw, price_source = histories[symbol]
        df = df_trimmed.copy()
        df["ret"] = df["close"].pct_change().fillna(0.0)
        df["ret_3"] = df["close"] / df["close"].shift(2) - 1.0
        df["ret_6"] = df["close"] / df["close"].shift(5) - 1.0
        df["ret_12"] = df["close"] / df["close"].shift(11) - 1.0
        df["ret_22"] = df["close"] / df["close"].shift(21) - 1.0

        symbol_rates = rates.reindex(df.index).ffill().bfill()
        rate_ann = symbol_rates["rate"].to_numpy(dtype=float)

        A_fit, r_fit, fit_start = strategy.ensure_temperature_assets(
            pd, np_local, plt, symbol, df_full_raw
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
        borrowed_fraction = LEVERAGE - 1.0
        daily_fee = ANNUAL_FEE / TRADING_DAYS
        daily_borrow = borrowed_fraction * ((rate_ann / 100.0) / TRADING_DAYS)
        cash_factor = 1.0 + (rate_ann / 100.0) / TRADING_DAYS
        tqqq_factor = (1.0 + LEVERAGE * rets) * (1.0 - daily_fee) * (1.0 - (daily_borrow / BORROW_DIVISOR))

        envs.append(
            SymbolEnvironment(
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
                tqqq_factor=tqqq_factor,
                start=df.index[0].to_datetime64(),
                end=df.index[-1].to_datetime64(),
            )
        )

    return envs


def simulate_symbol(env: SymbolEnvironment, params: GlobalParams) -> SimulationResult:
    anchors_payload = _anchors_to_payload(params.anchors)
    n_days = env.dates.shape[0]
    port_tqqq = np.zeros(n_days, dtype=float)
    port_cash = np.zeros(n_days, dtype=float)
    port_total = np.zeros(n_days, dtype=float)
    deployed = np.zeros(n_days, dtype=float)

    port_tqqq[0] = 0.0
    port_cash[0] = 1.0
    port_total[0] = port_cash[0]
    deployed[0] = 0.0

    next_rebalance = 0
    last_rebalance = -params.rebalance_days
    eps = 1e-6
    peak_total = port_total[0]
    buy_thresholds = params.buy_thresholds
    sell_thresholds = params.sell_thresholds

    for i in range(n_days):
        if i > 0:
            port_tqqq[i] = port_tqqq[i - 1] * env.tqqq_factor[i]
            port_cash[i] = port_cash[i - 1] * env.cash_factor[i]
        else:
            port_tqqq[i] = port_tqqq[0] * env.tqqq_factor[0]
            port_cash[i] = port_cash[0] * env.cash_factor[0]

        total = port_tqqq[i] + port_cash[i]
        curr_p = 0.0 if total <= 0 else port_tqqq[i] / total
        drawdown = 0.0 if peak_total <= 0 else (total / peak_total) - 1.0

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
                and port_tqqq[i] > 0
            ):
                port_cash[i] = total
                port_tqqq[i] = 0.0
                total = port_cash[i]
                curr_p = 0.0
                next_rebalance = i + params.rebalance_days
                last_rebalance = i
                port_total[i] = port_tqqq[i] + port_cash[i]
                deployed[i] = curr_p
                continue

            if target_p > curr_p + eps:
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
                    port_tqqq[i] = total * target_p
                    port_cash[i] = total - port_tqqq[i]
                    total = port_tqqq[i] + port_cash[i]
                    curr_p = target_p
                    next_rebalance = i + params.rebalance_days
                    last_rebalance = i
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
                    port_tqqq[i] = total * target_p
                    port_cash[i] = total - port_tqqq[i]
                    total = port_tqqq[i] + port_cash[i]
                    curr_p = target_p
                    next_rebalance = i + params.rebalance_days
                    last_rebalance = i
            else:
                next_rebalance = i + params.rebalance_days
                last_rebalance = i

        port_total[i] = port_tqqq[i] + port_cash[i]
        deployed[i] = curr_p
        if port_total[i] > peak_total:
            peak_total = port_total[i]

    if port_total[0] <= 0:
        initial_value = 1.0
    else:
        initial_value = float(port_total[0])
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


def evaluate_params(envs: Sequence[SymbolEnvironment], params: GlobalParams) -> tuple[float, List[SimulationResult]]:
    results = [simulate_symbol(env, params) for env in envs]
    adjusted = [_normalise_cagr(res.cagr) for res in results]
    avg_cagr = sum(adjusted) / len(results)
    return avg_cagr, results


def baseline_parameters() -> GlobalParams:
    return GlobalParams(
        anchors=[(0.9, 1.0), (1.0, 0.8), (1.5, 0.2)],
        buy_thresholds={"ret3": -0.03, "ret6": -0.03, "ret12": -0.02, "ret22": 0.0},
        sell_thresholds={"ret3": 0.03, "ret6": 0.03, "ret12": 0.0225, "ret22": 0.0075},
        buy_temp_limit=1.3,
        rate_start=10.0,
        rate_end=12.0,
        rate_min_allocation=0.0,
        rate_enabled=True,
        crash_enabled=True,
        crash_threshold=-0.1525,
        rebalance_days=22,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalise_cagr(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return 0.0 if value <= -1.0 else value


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
        new_val = value + rng.normal(0.0, sigma)
        return float(new_val)

    buy_thresholds = {
        key: _perturb_threshold(base.buy_thresholds.get(key), 0.2)
        for key in ("ret3", "ret6", "ret12", "ret22")
    }
    sell_thresholds = {
        key: _perturb_threshold(base.sell_thresholds.get(key), 0.15)
        for key in ("ret3", "ret6", "ret12", "ret22")
    }

    buy_temp_limit = base.buy_temp_limit
    if buy_temp_limit is not None:
        buy_temp_limit = _clamp(buy_temp_limit + rng.normal(0.0, 0.03), 1.15, 1.5)
    elif rng.random() < 0.1:
        buy_temp_limit = _clamp(1.3 + rng.normal(0.0, 0.05), 1.15, 1.5)

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
    envs: Sequence[SymbolEnvironment],
    iterations: int,
    seed: int,
    *,
    baseline_params: GlobalParams,
    baseline_avg: float,
    baseline_results: Sequence[SimulationResult],
) -> tuple[GlobalParams, float, List[SimulationResult]]:
    rng = np.random.default_rng(seed)
    best_avg = baseline_avg
    best_results = list(baseline_results)
    best_params: Optional[GlobalParams] = baseline_params

    try:
        for idx in range(iterations):
            if best_params is not None and rng.random() < 0.65:
                params = perturb_parameters(best_params, rng)
            else:
                params = sample_parameters(rng)
            avg_cagr, results = evaluate_params(envs, params)
            if avg_cagr > best_avg:
                best_avg = avg_cagr
                best_params = params
                best_results = results
                print(
                    f"[{idx + 1:03d}/{iterations}] New best average CAGR: {avg_cagr * 100:.2f}%"
                )
    except KeyboardInterrupt:
        print("\nOptimisation interrupted; returning best parameters found so far.")

    if best_params is None:
        raise RuntimeError("Optimiser failed to produce any parameter set")

    return best_params, best_avg, best_results


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimise A1 baseline parameters globally")
    parser.add_argument("--iterations", type=int, default=80, help="Number of random samples to evaluate (default 80)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility (default 1)")
    args = parser.parse_args()

    symbols = sorted(A1_SYMBOL_STATS.keys())
    print(f"Preparing data for {len(symbols)} symbols ...")
    envs = prepare_symbol_environments(symbols)
    baseline_params = baseline_parameters()
    baseline_avg, baseline_results = evaluate_params(envs, baseline_params)
    print(f"Baseline average CAGR: {baseline_avg * 100:.2f}%")
    print("Running global optimisation ...")
    params, avg_cagr, results = optimise(
        envs,
        iterations=args.iterations,
        seed=args.seed,
        baseline_params=baseline_params,
        baseline_avg=baseline_avg,
        baseline_results=baseline_results,
    )

    result_map = {res.symbol: res for res in results}
    baseline_map = {res.symbol: res for res in baseline_results}

    rows: List[tuple[str, float, float, float, float]] = []
    for symbol in symbols:
        base_res = baseline_map[symbol]
        new_res = result_map[symbol]
        base_adj = _normalise_cagr(base_res.cagr)
        new_adj = _normalise_cagr(new_res.cagr)
        rows.append((symbol, base_res.cagr, new_res.cagr, base_adj, new_adj))

    rows.sort(key=lambda row: row[4] - row[3], reverse=True)

    print()
    print("Symbol | A1 CAGR | A1g CAGR | Δ (pp)")
    print("-------|---------|----------|-------")
    for symbol, base_cagr, new_cagr, base_adj, new_adj in rows:
        delta = new_adj - base_adj
        print(
            f"{symbol:5s} | {base_cagr * 100:7.2f}% | {new_cagr * 100:8.2f}% | {delta * 100:6.2f}%"
        )

    delta_values = np.array([row[4] - row[3] for row in rows], dtype=float)
    mean_delta = float(delta_values.mean()) if delta_values.size else 0.0
    median_delta = float(np.median(delta_values)) if delta_values.size else 0.0
    mean_abs_delta = float(np.mean(np.abs(delta_values))) if delta_values.size else 0.0
    improved = int(np.sum(delta_values > 0))
    worse = int(np.sum(delta_values < 0))
    flat = len(rows) - improved - worse

    print()
    print("(Collapsed series at -100% are treated as 0 for averaging, matching the published A1 stats.)")
    print()
    print(f"Average A1 CAGR : {baseline_avg * 100:.2f}%")
    print(f"Average A1g CAGR: {avg_cagr * 100:.2f}%")
    print(f"Mean delta      : {mean_delta * 100:.2f} pp")
    print(f"Median delta    : {median_delta * 100:.2f} pp")
    print(f"Mean |delta|    : {mean_abs_delta * 100:.2f} pp")
    print(f"Improved / worse / flat: {improved} / {worse} / {flat}")

    print()
    print("A1g parameter configuration:")
    config_payload = params.to_config()
    print(json.dumps(config_payload, indent=2))


if __name__ == "__main__":
    main()
