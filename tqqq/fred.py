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
    """Fetch ``series_id`` between ``start`` and ``end`` with layered fallbacks.

    If no provider returns data within the requested window, the function
    incrementally widens the lookback so we can backfill from the most recent
    observation rather than failing.
    """

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts > end_ts:
        raise ValueError("start must be <= end")

    def normalise_frame(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
        if frame is None or frame.empty:
            return None
        frame = frame.sort_index()
        if "rate" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "rate"})
        frame = frame.loc[frame.index <= end_ts]
        if frame.empty:
            return None

        if start_ts not in frame.index:
            prior = frame.loc[frame.index < start_ts]
            if not prior.empty:
                last_row = prior.iloc[[-1]].copy()
                last_row.index = [start_ts]
                frame = pd.concat([frame, last_row])

        frame = frame.sort_index()
        if frame.index.max() >= start_ts:
            frame = frame.loc[frame.index >= start_ts]

        frame = frame[~frame.index.duplicated(keep="last")]
        return frame

    fetchers: tuple[Callable[[pd.Timestamp, pd.Timestamp], Optional[pd.DataFrame]], ...] = (
        lambda s, e: _fetch_with_fredapi(series_id, s, e, api_key),
        lambda s, e: _fetch_with_datareader(series_id, s, e, api_key),
        lambda s, e: _fetch_with_csv(series_id, s, e, api_key, opener=opener),
    )

    def try_fetch(window_start: pd.Timestamp) -> Optional[pd.DataFrame]:
        for fetcher in fetchers:
            frame = fetcher(window_start, end_ts)
            normalised = normalise_frame(frame)
            if normalised is not None and not normalised.empty:
                return normalised
        return None

    result = try_fetch(start_ts)
    if result is not None:
        return result

    lookbacks = (
        pd.Timedelta(days=30),
        pd.Timedelta(days=365),
        pd.Timedelta(days=365 * 5),
        pd.Timedelta(days=365 * 10),
    )

    for delta in lookbacks:
        window_start = start_ts - delta
        result = try_fetch(window_start)
        if result is not None:
            return result

    raise RuntimeError(
        f"Failed to fetch FRED series '{series_id}' via fredapi, pandas-datareader, or CSV fallback"
    )


__all__ = [
    "fetch_fred_series",
    "_fetch_with_csv",
    "_fetch_with_datareader",
    "_fetch_with_fredapi",
]
