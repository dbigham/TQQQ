"""Utilities for fetching and caching basic fundamental data for symbols."""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

FUNDAMENTALS_FILENAME = "fundamentals.json"


@dataclass
class FundamentalMetric:
    """Single labelled fundamental metric with optional value and timestamp."""

    label: str
    value: Optional[float]
    as_of: Optional[str]
    percent: bool = False

    def _format_value(self) -> str:
        if self.value is None:
            return "N/A"
        if self.percent:
            return f"{self.value * 100.0:.2f}%"
        return f"{self.value:.2f}"

    def as_summary_line(self) -> str:
        """Format the metric for markdown summaries."""
        date_display = self.as_of or "Unknown"
        return f"- **{self.label} (as of {date_display})**: {self._format_value()}"


@dataclass
class Fundamentals:
    """Container for key valuation metrics used across the project."""

    pe_ratio: FundamentalMetric
    peg_ratio: FundamentalMetric
    growth_rate: FundamentalMetric

    def summary_lines(self) -> list[str]:
        """Return the formatted markdown lines for all metrics."""
        return [
            self.pe_ratio.as_summary_line(),
            self.peg_ratio.as_summary_line(),
            self.growth_rate.as_summary_line(),
        ]


def _coerce_float(value: object) -> Optional[float]:
    try:
        if isinstance(value, bool):  # guard against bool being subclass of int
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_payload(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _days_between(date_iso: str, today_iso: str) -> Optional[int]:
    try:
        date_val = _dt.date.fromisoformat(date_iso)
        today_val = _dt.date.fromisoformat(today_iso)
    except ValueError:
        return None
    return (today_val - date_val).days


def _metric_from_payload(
    payload: Dict[str, object],
    *,
    value_key: str,
    as_of_key: str,
    label: str,
    percent: bool = False,
    fallback_date_keys: tuple[str, ...] = (),
) -> FundamentalMetric:
    value = _coerce_float(payload.get(value_key))
    as_of: Optional[str] = None
    raw_as_of = payload.get(as_of_key)
    if isinstance(raw_as_of, str):
        as_of = raw_as_of
    else:
        for key in fallback_date_keys:
            candidate = payload.get(key)
            if isinstance(candidate, str):
                as_of = candidate
                break
    return FundamentalMetric(label, value, as_of, percent=percent)


def _fundamentals_from_payload(payload: Dict[str, object]) -> Fundamentals:
    return Fundamentals(
        pe_ratio=_metric_from_payload(
            payload,
            value_key="pe_ratio",
            as_of_key="pe_ratio_as_of",
            label="P/E ratio",
            fallback_date_keys=("as_of",),
        ),
        peg_ratio=_metric_from_payload(
            payload,
            value_key="peg_ratio",
            as_of_key="peg_ratio_as_of",
            label="PEG ratio",
        ),
        growth_rate=_metric_from_payload(
            payload,
            value_key="growth_rate",
            as_of_key="growth_rate_as_of",
            label="Current growth rate",
            percent=True,
        ),
    )


def _write_payload(path: str, symbol: str, fundamentals: Fundamentals, source: str) -> None:
    payload = {
        "symbol": symbol,
        "source": source,
        "pe_ratio": fundamentals.pe_ratio.value,
        "pe_ratio_as_of": fundamentals.pe_ratio.as_of,
        "peg_ratio": fundamentals.peg_ratio.value,
        "peg_ratio_as_of": fundamentals.peg_ratio.as_of,
        "growth_rate": fundamentals.growth_rate.value,
        "growth_rate_as_of": fundamentals.growth_rate.as_of,
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except Exception:
        pass


def _metrics_are_fresh(fundamentals: Fundamentals, today_iso: str, *, stale_after_days: int) -> bool:
    metrics = [fundamentals.pe_ratio, fundamentals.peg_ratio, fundamentals.growth_rate]
    for metric in metrics:
        if metric.value is None:
            return False
        if metric.as_of is None:
            return False
        age = _days_between(metric.as_of, today_iso)
        if age is None or age < 0 or age > stale_after_days:
            return False
    return True


def _fetch_fundamental_values(symbol: str) -> Dict[str, Optional[float]]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:  # pragma: no cover - runtime dependency
        return {"pe_ratio": None, "peg_ratio": None, "growth_rate": None}

    ticker = yf.Ticker(symbol)
    values: Dict[str, Optional[float]] = {
        "pe_ratio": None,
        "peg_ratio": None,
        "growth_rate": None,
    }

    try:  # pragma: no branch - defensive guard against missing fast_info
        fast_info = getattr(ticker, "fast_info", None)
        getter = None
        if isinstance(fast_info, dict):
            getter = fast_info.get
        else:
            candidate_getter = getattr(fast_info, "get", None)
            if callable(candidate_getter):
                getter = candidate_getter
            elif fast_info is not None:
                def _fallback_get(key: str, default=None, *, _src=fast_info):  # type: ignore[assignment]
                    try:
                        return _src[key]
                    except Exception:
                        return default

                getter = _fallback_get
        if getter is not None:
            candidate = getter("trailing_pe")
            if candidate is None:
                candidate = getter("pe_ratio")
            values["pe_ratio"] = _coerce_float(candidate) or values["pe_ratio"]
            values["peg_ratio"] = _coerce_float(getter("peg_ratio")) or values["peg_ratio"]
            # Some tickers expose earnings growth on ``fast_info``; use if present.
            growth_candidate = getter("earnings_growth")
            values["growth_rate"] = _coerce_float(growth_candidate) or values["growth_rate"]
    except Exception:
        pass

    try:
        info = ticker.info  # type: ignore[attr-defined]
    except Exception:
        info = None

    if isinstance(info, dict):
        if values["pe_ratio"] is None:
            for key in ("trailingPE", "forwardPE"):
                values["pe_ratio"] = _coerce_float(info.get(key))
                if values["pe_ratio"] is not None:
                    break
        if values["peg_ratio"] is None:
            for key in ("trailingPegRatio", "pegRatio", "peg_ratio"):
                values["peg_ratio"] = _coerce_float(info.get(key))
                if values["peg_ratio"] is not None:
                    break
        if values["growth_rate"] is None:
            for key in (
                "earningsQuarterlyGrowth",
                "earningsGrowth",
                "revenueQuarterlyGrowth",
                "revenueGrowth",
                "ytdReturn",
                "threeYearAverageReturn",
                "fiveYearAverageReturn",
            ):
                candidate = _coerce_float(info.get(key))
                if candidate is None:
                    continue
                if key in {"ytdReturn", "threeYearAverageReturn", "fiveYearAverageReturn"} and abs(candidate) > 1:
                    candidate /= 100.0
                values["growth_rate"] = candidate
                break

    if values["peg_ratio"] is None:
        pe = values.get("pe_ratio")
        growth = values.get("growth_rate")
        if pe is not None and growth is not None and abs(growth) > 1e-9:
            # Convert a fractional growth rate into percentage terms before dividing.
            values["peg_ratio"] = pe / (growth * 100.0)

    return values


def ensure_fundamentals(
    symbol: str,
    symbol_dir: str,
    *,
    stale_after_days: int = 7,
) -> Fundamentals:
    """Ensure cached fundamentals exist for ``symbol`` and refresh when stale."""

    os.makedirs(symbol_dir, exist_ok=True)
    fundamentals_path = os.path.join(symbol_dir, FUNDAMENTALS_FILENAME)
    payload = _load_payload(fundamentals_path)
    cached = _fundamentals_from_payload(payload)
    today_iso = _dt.date.today().isoformat()

    if _metrics_are_fresh(cached, today_iso, stale_after_days=stale_after_days):
        # Backfill to the modern payload shape if required.
        if {"peg_ratio_as_of", "growth_rate_as_of"} - payload.keys():
            _write_payload(
                fundamentals_path,
                symbol,
                cached,
                str(payload.get("source") or "yfinance"),
            )
        return cached

    fetched = _fetch_fundamental_values(symbol)
    source = str(payload.get("source") or "yfinance")

    pe_value = fetched.get("pe_ratio")
    peg_value = fetched.get("peg_ratio")
    growth_value = fetched.get("growth_rate")

    if pe_value is not None:
        pe_metric = FundamentalMetric("P/E ratio", float(pe_value), today_iso)
    else:
        pe_metric = cached.pe_ratio
        if pe_metric.as_of is None and pe_metric.value is None:
            pe_metric = FundamentalMetric("P/E ratio", None, today_iso)

    if peg_value is not None:
        peg_metric = FundamentalMetric("PEG ratio", float(peg_value), today_iso)
    else:
        peg_metric = cached.peg_ratio

    if growth_value is not None:
        growth_metric = FundamentalMetric("Current growth rate", float(growth_value), today_iso, percent=True)
    else:
        growth_metric = cached.growth_rate

    refreshed = Fundamentals(pe_metric, peg_metric, growth_metric)

    _write_payload(fundamentals_path, symbol, refreshed, source)
    return refreshed


def ensure_pe_ratio(symbol: str, symbol_dir: str, *, stale_after_days: int = 7) -> FundamentalMetric:
    """Backward compatible wrapper returning only the P/E metric."""

    fundamentals = ensure_fundamentals(symbol, symbol_dir, stale_after_days=stale_after_days)
    return fundamentals.pe_ratio


__all__ = [
    "FundamentalMetric",
    "Fundamentals",
    "ensure_fundamentals",
    "ensure_pe_ratio",
]
