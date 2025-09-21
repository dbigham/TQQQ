"""Utilities for fetching and caching basic fundamental data for symbols."""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

FUNDAMENTALS_FILENAME = "fundamentals.json"


@dataclass
class PERatio:
    """Representation of a symbol's price-to-earnings ratio."""

    value: Optional[float]
    as_of: Optional[str]

    def as_summary_line(self) -> str:
        """Format the ratio for markdown summaries."""
        date_display = self.as_of or "Unknown"
        value_display = "N/A" if self.value is None else f"{self.value:.2f}"
        return f"- **P/E ratio (as of {date_display})**: {value_display}"


def _load_cached(path: str) -> Tuple[Optional[float], Optional[str]]:
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return None, None
    pe = payload.get("pe_ratio")
    if isinstance(pe, (int, float)):
        pe_value: Optional[float] = float(pe)
    else:
        pe_value = None
    as_of = payload.get("as_of")
    if isinstance(as_of, str):
        return pe_value, as_of
    return pe_value, None


def _days_between(date_iso: str, today_iso: str) -> Optional[int]:
    try:
        date_val = _dt.date.fromisoformat(date_iso)
        today_val = _dt.date.fromisoformat(today_iso)
    except ValueError:
        return None
    return (today_val - date_val).days


def fetch_pe_ratio(symbol: str) -> Optional[float]:
    """Fetch the trailing P/E ratio for ``symbol`` using yfinance.

    Returns ``None`` when data is unavailable.
    """

    try:
        import yfinance as yf  # type: ignore
    except ImportError:  # pragma: no cover - runtime dependency
        return None

    ticker = yf.Ticker(symbol)
    pe_value: Optional[float] = None

    # ``fast_info`` avoids a full fundamental download and is the quickest path.
    try:  # pragma: no branch - simple guard
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            pe_value = fast_info.get("trailing_pe")
            if pe_value is None:
                pe_value = fast_info.get("pe_ratio")
            if pe_value is not None:
                pe_value = float(pe_value)
    except Exception:
        pe_value = None

    if pe_value is None:
        try:
            info = ticker.info  # type: ignore[attr-defined]
        except Exception:
            info = None
        if isinstance(info, dict):
            for key in ("trailingPE", "forwardPE"):
                raw = info.get(key)
                if raw is not None:
                    try:
                        pe_value = float(raw)
                        break
                    except (TypeError, ValueError):
                        continue

    return pe_value


def ensure_pe_ratio(symbol: str, symbol_dir: str, *, stale_after_days: int = 7) -> PERatio:
    """Ensure a cached P/E ratio exists for ``symbol``.

    The ratio is cached under ``symbol_dir`` in ``fundamentals.json``. When the
    cached data is older than ``stale_after_days`` (default seven days) or
    missing entirely, a fresh lookup is attempted.
    """

    os.makedirs(symbol_dir, exist_ok=True)
    fundamentals_path = os.path.join(symbol_dir, FUNDAMENTALS_FILENAME)
    cached_value, cached_as_of = _load_cached(fundamentals_path)
    today_iso = _dt.date.today().isoformat()

    needs_refresh = True
    if cached_as_of:
        age = _days_between(cached_as_of, today_iso)
        needs_refresh = age is None or age > stale_after_days
    elif cached_value is not None:
        # Value but no date → refresh to attach an "as of" timestamp.
        needs_refresh = True

    if not needs_refresh and cached_as_of and cached_value is not None:
        return PERatio(cached_value, cached_as_of)

    fetched_value = fetch_pe_ratio(symbol)
    if fetched_value is not None:
        payload = {
            "symbol": symbol,
            "pe_ratio": float(fetched_value),
            "as_of": today_iso,
            "source": "yfinance",
        }
        try:
            with open(fundamentals_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
        except Exception:
            pass
        return PERatio(float(fetched_value), today_iso)

    if cached_value is not None or cached_as_of is not None:
        return PERatio(cached_value, cached_as_of)

    # No cached data and lookup failed – record an explicit miss so that
    # downstream code still has a dated entry.
    payload = {
        "symbol": symbol,
        "pe_ratio": None,
        "as_of": today_iso,
        "source": "yfinance",
    }
    try:
        with open(fundamentals_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except Exception:
        pass
    return PERatio(None, today_iso)


__all__ = ["PERatio", "ensure_pe_ratio"]
