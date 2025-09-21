"""Shared utilities for the TQQQ research codebase."""

from .dataset import PriceDataError, load_price_csv
from .fitting import FitResult, FitStep, fit_constant_growth, iterative_constant_growth
from .fred import fetch_fred_series
from .fundamentals import PERatio, ensure_pe_ratio
from .simulation import simulate_leveraged_path

__all__ = [
    "PriceDataError",
    "load_price_csv",
    "FitResult",
    "FitStep",
    "fit_constant_growth",
    "iterative_constant_growth",
    "fetch_fred_series",
    "PERatio",
    "ensure_pe_ratio",
    "simulate_leveraged_path",
]
