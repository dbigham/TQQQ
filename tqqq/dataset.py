"""Utilities for loading price series data used across scripts."""

from __future__ import annotations

from typing import Iterable, Tuple

import io
import re

import pandas as pd

_DAYS_PER_YEAR = 365.25


class PriceDataError(RuntimeError):
    """Raised when an expected price-series CSV is malformed."""


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise PriceDataError(f"Missing required column(s): {joined}")


def load_price_csv(
    path: str,
    *,
    set_index: bool = False,
    add_elapsed_years: bool = False,
    ensure_positive: bool = True,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Load a CSV containing ``date``/``close`` columns.

    Parameters
    ----------
    path:
        CSV path on disk.
    set_index:
        When True, returns a frame indexed by the ``date`` column.
    add_elapsed_years:
        When True, computes a ``t_years`` column measuring fractional
        years since the first observation (using 365.25 days/year).
    ensure_positive:
        Drop any non-positive closes before returning.

    Returns
    -------
    (DataFrame, Timestamp)
        The cleaned frame and the first timestamp in the series.
    """

    try:
        df = pd.read_csv(path)
    except Exception as e:
        # Recover from occasional malformed rows where two records were
        # concatenated or corrupted mid-line (e.g.,
        # "YYYY-MM-DD,priceYYYY-MM-DD,price" or "...86.72022-2022-..."), by
        # extracting well-formed (date, close) pairs and rebuilding the frame.
        if getattr(pd, "errors", None) is not None and isinstance(e, pd.errors.ParserError) or "ParserError" in type(e).__name__:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            pairs = re.findall(r"((?:19|20)\d{2}-\d{2}-\d{2}),\s*(-?\d+(?:\.\d+)?)", raw)
            if not pairs:
                # As a second attempt, try inserting newlines before repeated date
                # patterns and re-parse via pandas.
                fixed = re.sub(
                    r"(?<!^)(?=(?:19|20)\d{2}-\d{2}-\d{2},)",
                    "\n",
                    raw,
                    flags=re.MULTILINE,
                )
                fixed = re.sub(r"(?<=\d)-(\r?\n)", r"\1", fixed)
                df = pd.read_csv(io.StringIO(fixed))
            else:
                import pandas as _pd
                df = _pd.DataFrame(pairs, columns=["date", "close"])  # type: ignore[assignment]
                # Ensure correct dtypes before further cleaning
                df["date"] = df["date"].astype(str)
                df["close"] = df["close"].astype(float)
        else:
            raise
    _require_columns(df, ["date", "close"])

    df = df.copy()
    # Coerce any malformed date cells (e.g., stray tokens like "44") to NaT
    # and drop those rows to keep the series clean and parseable.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["close"])
    if ensure_positive:
        df = df[df["close"] > 0]

    if df.empty:
        raise PriceDataError("No rows remain after cleaning price data")

    start_ts = pd.to_datetime(df["date"].iloc[0])

    if set_index:
        df = df.set_index("date")

    if add_elapsed_years:
        if set_index:
            indexer = df.index
        else:
            indexer = pd.to_datetime(df["date"])  # type: ignore[index]
        elapsed = (indexer - start_ts) / pd.Timedelta(days=_DAYS_PER_YEAR)
        df["t_years"] = elapsed.to_numpy()

    return df, start_ts


__all__ = ["PriceDataError", "load_price_csv"]
