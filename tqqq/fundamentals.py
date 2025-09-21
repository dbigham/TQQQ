"""Utilities for fetching and caching basic fundamental data for symbols."""

from __future__ import annotations

import datetime as _dt
import json
import os
from math import isfinite as _isfinite
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

FUNDAMENTALS_FILENAME = "fundamentals.json"

PE_RATIO_BASIS = "Forward_NTM"
GROWTH_RATE_BASIS = "Next_5Y_EPS_CAGR"
GROWTH_RATE_UNITS = "percent"
PEG_RATIO_BASIS = f"{PE_RATIO_BASIS} / {GROWTH_RATE_BASIS} ({GROWTH_RATE_UNITS})"
PEG_NOT_MEANINGFUL_NOTE = "PEG not meaningful for non-positive growth"
GROWTH_ALIGNMENT_TOLERANCE = 0.02  # acceptable absolute delta in percentage points
METADATA_KEYS = (
    "pe_ratio_basis",
    "peg_ratio_basis",
    "growth_rate_basis",
    "growth_rate_units",
    "growth_rate_source",
    "peg_ratio_source",
    "peg_ratio_note",
)


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


def _positive_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not _isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _derive_growth_from_pe_and_peg(
    pe_ratio: Optional[float], peg_ratio: Optional[float]
) -> Optional[float]:
    pe_clean = _positive_or_none(pe_ratio)
    peg_clean = _positive_or_none(peg_ratio)
    if pe_clean is None or peg_clean is None:
        return None
    derived = pe_clean / peg_clean / 100.0
    if not _isfinite(derived) or derived <= 0:
        return None
    return derived


def _normalize_metric_triplet(
    pe_ratio: Optional[float],
    peg_ratio: Optional[float],
    growth_rate: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, str]]:
    """Ensure the P/E, PEG, and growth metrics are internally consistent."""

    metadata: Dict[str, str] = {}
    pe_clean = _positive_or_none(pe_ratio)
    peg_clean = _positive_or_none(peg_ratio)
    growth_clean = _coerce_float(growth_rate)
    if growth_clean is not None and (not _isfinite(growth_clean)):
        growth_clean = None

    derived_growth = _derive_growth_from_pe_and_peg(pe_clean, peg_clean)

    if growth_clean is None:
        growth_clean = derived_growth
        if derived_growth is not None:
            metadata["growth_rate_source"] = "derived_from_pe_and_peg"
    elif growth_clean <= 0 or abs(growth_clean) < 1e-9:
        metadata["peg_ratio_note"] = PEG_NOT_MEANINGFUL_NOTE
        growth_clean = float(growth_clean)
        peg_clean = None
    else:
        growth_clean = float(growth_clean)
        if derived_growth is not None and abs(derived_growth - growth_clean) > GROWTH_ALIGNMENT_TOLERANCE:
            growth_clean = derived_growth
            metadata["growth_rate_source"] = "derived_from_pe_and_peg"
        if peg_clean is None and pe_clean is not None:
            peg_clean = pe_clean / (growth_clean * 100.0)
            metadata["peg_ratio_source"] = "derived_from_pe_and_growth"

    if growth_clean is not None and (growth_clean <= 0 or abs(growth_clean) < 1e-9):
        metadata["peg_ratio_note"] = PEG_NOT_MEANINGFUL_NOTE
        peg_clean = None

    return pe_clean, peg_clean, growth_clean, metadata


def _select_preferred_ratio(
    candidates: list[Tuple[str, Optional[float]]],
) -> Tuple[Optional[float], Optional[str]]:
    for basis, value in candidates:
        numeric = _coerce_float(value)
        if numeric is None or not _isfinite(numeric):
            continue
        return numeric, basis
    return None, None


def _values_differ(first: Optional[float], second: Optional[float]) -> bool:
    if first is None or second is None:
        return (first is None) != (second is None)
    tolerance = 1e-6 * max(1.0, abs(first), abs(second))
    return abs(first - second) > tolerance


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


def _write_payload(
    path: str,
    symbol: str,
    fundamentals: Fundamentals,
    source: str,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
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
    default_metadata: Dict[str, object] = {
        "pe_ratio_basis": PE_RATIO_BASIS,
        "peg_ratio_basis": PEG_RATIO_BASIS,
        "growth_rate_basis": GROWTH_RATE_BASIS,
        "growth_rate_units": GROWTH_RATE_UNITS,
    }
    if metadata:
        for key, value in metadata.items():
            if value is not None:
                default_metadata[key] = value
    payload.update(default_metadata)
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
    values: Dict[str, Optional[float]] = {}
    metadata: Dict[str, Optional[str]] = {}
    pe_candidates: list[Tuple[str, Optional[float]]] = []
    peg_candidates: list[Tuple[str, Optional[float]]] = []
    growth_candidates: list[Tuple[str, Optional[float]]] = []

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
            pe_candidates.extend(
                [
                    ("Forward_NTM", getter("forward_pe")),
                    ("Forward_NTM", getter("forwardPE")),
                    ("Trailing_TTM", getter("trailing_pe")),
                    ("Trailing_TTM", getter("pe_ratio")),
                ]
            )
            peg_candidates.append((PEG_RATIO_BASIS, getter("peg_ratio")))
            growth_candidates.append(("fast_info_earnings_growth", getter("earnings_growth")))
    except Exception:
        pass

    try:
        info = ticker.info  # type: ignore[attr-defined]
    except Exception:
        info = None

    if isinstance(info, dict):
        pe_candidates.extend(
            [
                ("Forward_NTM", info.get("forwardPE")),
                ("Trailing_TTM", info.get("trailingPE")),
            ]
        )
        peg_candidates.extend(
            [
                (PEG_RATIO_BASIS, info.get("pegRatio")),
                (PEG_RATIO_BASIS, info.get("trailingPegRatio")),
                (PEG_RATIO_BASIS, info.get("peg_ratio")),
            ]
        )
        for key in ("earningsGrowth", "earningsQuarterlyGrowth"):
            growth_candidates.append((key, info.get(key)))

    pe_value, pe_basis = _select_preferred_ratio(pe_candidates)
    peg_value, _ = _select_preferred_ratio(peg_candidates)
    growth_value, growth_basis = _select_preferred_ratio(growth_candidates)

    pe_value, peg_value, growth_value, harmonized_metadata = _normalize_metric_triplet(
        pe_value, peg_value, growth_value
    )

    values["pe_ratio"] = pe_value
    values["peg_ratio"] = peg_value
    values["growth_rate"] = growth_value
    metadata.update(harmonized_metadata)
    if pe_basis:
        metadata.setdefault("pe_ratio_basis", pe_basis)
    if growth_basis:
        metadata.setdefault("growth_rate_basis", growth_basis)
    if peg_value is not None:
        metadata.setdefault("peg_ratio_basis", PEG_RATIO_BASIS)
    if growth_value is not None:
        metadata.setdefault("growth_rate_units", GROWTH_RATE_UNITS)

    for key, value in metadata.items():
        if value is not None:
            values[key] = value

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
    cached_metadata = {
        key: payload.get(key)
        for key in METADATA_KEYS
        if key in payload and payload.get(key) is not None
    }
    today_iso = _dt.date.today().isoformat()

    if _metrics_are_fresh(cached, today_iso, stale_after_days=stale_after_days):
        # Backfill to the modern payload shape if required.
        needs_backfill = {"peg_ratio_as_of", "growth_rate_as_of"} - payload.keys()
        metadata = dict(cached_metadata)
        if cached.peg_ratio.value is not None:
            metadata.pop("peg_ratio_note", None)
        if needs_backfill or metadata:
            _write_payload(
                fundamentals_path,
                symbol,
                cached,
                str(payload.get("source") or "yfinance"),
                metadata=metadata if metadata else None,
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
        growth_metric = FundamentalMetric(
            "Current growth rate", float(growth_value), today_iso, percent=True
        )
    else:
        growth_metric = cached.growth_rate

    refreshed = Fundamentals(pe_metric, peg_metric, growth_metric)
    metadata_updates = {
        key: fetched.get(key)
        for key in METADATA_KEYS
        if key in fetched and fetched.get(key) is not None
    }
    combined_metadata = {**cached_metadata, **metadata_updates}
    if refreshed.peg_ratio.value is not None:
        combined_metadata.pop("peg_ratio_note", None)
    _write_payload(
        fundamentals_path,
        symbol,
        refreshed,
        source,
        metadata=combined_metadata if combined_metadata else None,
    )
    return refreshed


def harmonize_cached_fundamentals(
    symbol_dir: str,
) -> Tuple[Optional[Fundamentals], bool, Optional[str]]:
    """Align cached fundamentals in ``symbol_dir`` without triggering fresh fetches."""

    fundamentals_path = os.path.join(symbol_dir, FUNDAMENTALS_FILENAME)
    payload = _load_payload(fundamentals_path)
    if not payload:
        return None, False, None

    cached = _fundamentals_from_payload(payload)
    pe_clean, peg_clean, growth_clean, metadata = _normalize_metric_triplet(
        cached.pe_ratio.value, cached.peg_ratio.value, cached.growth_rate.value
    )

    updated = any(
        (
            _values_differ(cached.pe_ratio.value, pe_clean),
            _values_differ(cached.peg_ratio.value, peg_clean),
            _values_differ(cached.growth_rate.value, growth_clean),
        )
    )

    refreshed = Fundamentals(
        FundamentalMetric("P/E ratio", pe_clean, cached.pe_ratio.as_of),
        FundamentalMetric("PEG ratio", peg_clean, cached.peg_ratio.as_of),
        FundamentalMetric(
            "Current growth rate", growth_clean, cached.growth_rate.as_of, percent=True
        ),
    )

    metadata_updates = {
        key: payload.get(key)
        for key in METADATA_KEYS
        if payload.get(key) is not None
    }
    metadata_updates.update(metadata)
    if refreshed.peg_ratio.value is not None:
        metadata_updates.pop("peg_ratio_note", None)

    symbol = str(payload.get("symbol") or os.path.basename(symbol_dir).upper())

    _write_payload(
        fundamentals_path,
        symbol,
        refreshed,
        str(payload.get("source") or "yfinance"),
        metadata=metadata_updates if metadata_updates else None,
    )

    return refreshed, updated, symbol


def ensure_pe_ratio(symbol: str, symbol_dir: str, *, stale_after_days: int = 7) -> FundamentalMetric:
    """Backward compatible wrapper returning only the P/E metric."""

    fundamentals = ensure_fundamentals(symbol, symbol_dir, stale_after_days=stale_after_days)
    return fundamentals.pe_ratio


__all__ = [
    "FundamentalMetric",
    "Fundamentals",
    "ensure_fundamentals",
    "harmonize_cached_fundamentals",
    "ensure_pe_ratio",
]
