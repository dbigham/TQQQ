"""Programmatic TQQQ-style simulation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def simulate_leveraged_path(
    unified: pd.DataFrame,
    rates: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    leverage: float = 3.0,
    annual_fee: float = 0.0095,
    trading_days: int = 252,
    borrow_divisor: float = 0.7,
    actual_column: str = "actual",
    output_column: str = "simulated",
    initial_value: Optional[float] = None,
) -> pd.DataFrame:
    """Simulate a leveraged path using unified returns and borrowing costs."""

    if "close" not in unified.columns:
        raise ValueError("Unified dataframe must include a 'close' column")
    if actual_column not in actual.columns:
        raise ValueError(f"Actual dataframe missing column '{actual_column}'")

    unified = unified.sort_index()
    actual = actual.sort_index()
    rates = rates.sort_index()

    start = max(unified.index.min(), actual.index.min())
    end = min(unified.index.max(), actual.index.max())

    unified = unified.loc[(unified.index >= start) & (unified.index <= end)].copy()
    actual = actual.loc[(actual.index >= start) & (actual.index <= end)].copy()

    unified["ret"] = unified["close"].pct_change()

    rates = rates.reindex(unified.index).ffill().bfill()
    if "rate" not in rates.columns:
        raise ValueError("Rates dataframe must include a 'rate' column")

    actual_values = actual[actual_column].to_numpy(dtype=float)
    rets = unified["ret"].to_numpy(dtype=float)
    rates_arr = rates["rate"].to_numpy(dtype=float)

    sim = np.empty_like(rets, dtype=float)
    sim[:] = np.nan

    init = float(actual_values[0]) if initial_value is None else float(initial_value)
    sim[0] = init

    borrowed_fraction = leverage - 1.0
    daily_fee = annual_fee / float(trading_days)

    for i in range(1, len(sim)):
        pct_change = rets[i]
        if not np.isfinite(pct_change):
            pct_change = 0.0
        rate_ann = rates_arr[i]
        daily_borrow_cost = borrowed_fraction * ((rate_ann / 100.0) / float(trading_days))
        factor = (1.0 + leverage * pct_change) * (1.0 - daily_fee) * (1.0 - (daily_borrow_cost / borrow_divisor))
        sim[i] = sim[i - 1] * factor

    out = actual.copy()
    out[output_column] = sim
    return out


__all__ = ["simulate_leveraged_path"]
