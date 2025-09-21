"""Helpers for downloading FRED interest-rate series with fallbacks."""

from __future__ import annotations

import io
from typing import Callable, Optional
import urllib.error
import urllib.request

import pandas as pd

Fetcher = Callable[[str, pd.Timestamp, pd.Timestamp, Optional[str]], Optional[pd.DataFrame]]


def _fetch_with_fredapi(
    series_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: Optional[str],
) -> Optional[pd.DataFrame]:
    try:
        from fredapi import Fred  # type: ignore
    except Exception:
        return None
    try:
        fred = Fred(api_key=api_key)
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        frame = series.rename("rate").to_frame()
        frame.index = pd.to_datetime(frame.index)
        return frame
    except Exception:
        return None


def _fetch_with_datareader(
    series_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    _api_key_unused: Optional[str],
) -> Optional[pd.DataFrame]:
    try:
        from pandas_datareader import data as pdr  # type: ignore
    except Exception:
        return None
    try:
        frame = pdr.DataReader(series_id, "fred", start, end)
    except Exception:
        return None
    if series_id in frame.columns:
        frame = frame.rename(columns={series_id: "rate"})
    else:
        frame.columns = ["rate"]
    return frame


def _fetch_with_csv(
    series_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    _api_key_unused: Optional[str],
    *,
    opener: Callable[[str], io.BufferedReader] = urllib.request.urlopen,
) -> Optional[pd.DataFrame]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with opener(url) as resp:
            csv_bytes = resp.read()
    except Exception:
        return None

    frame = pd.read_csv(io.BytesIO(csv_bytes))
    date_col_candidates = ["DATE", "observation_date", "date"]
    value_col_candidates = [series_id, series_id.upper(), "value"]

    date_col = next((c for c in date_col_candidates if c in frame.columns), None)
    value_col = next((c for c in value_col_candidates if c in frame.columns), None)

    if not date_col or not value_col:
        return None

    frame = frame.rename(columns={date_col: "date", value_col: "rate"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date")["rate"].to_frame()
    mask = (frame.index >= start) & (frame.index <= end)
    return frame.loc[mask]


def fetch_fred_series(
    series_id: str,
    start,
    end,
    *,
    api_key: Optional[str] = None,
    opener: Callable[[str], io.BufferedReader] = urllib.request.urlopen,
) -> pd.DataFrame:
    """Fetch ``series_id`` between ``start`` and ``end`` with layered fallbacks."""

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts > end_ts:
        raise ValueError("start must be <= end")

    attempts = (
        lambda: _fetch_with_fredapi(series_id, start_ts, end_ts, api_key),
        lambda: _fetch_with_datareader(series_id, start_ts, end_ts, api_key),
        lambda: _fetch_with_csv(series_id, start_ts, end_ts, api_key, opener=opener),
    )

    for attempt in attempts:
        frame = attempt()
        if frame is None or frame.empty:
            continue
        frame = frame.sort_index()
        if "rate" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "rate"})
        return frame

    raise RuntimeError(f"Failed to fetch FRED series '{series_id}' via fredapi, pandas-datareader, or CSV fallback")


__all__ = [
    "fetch_fred_series",
    "_fetch_with_csv",
    "_fetch_with_datareader",
    "_fetch_with_fredapi",
]
