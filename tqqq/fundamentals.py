"""Utilities for fetching and caching basic fundamental data for symbols."""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

FUNDAMENTALS_FILENAME = "fundamentals.json"


def _format_ratio_line(label: str, value: Optional[float], as_of: Optional[str]) -> str:
    date_display = as_of or "Unknown"
    value_display = "N/A" if value is None else f"{value:.2f}"
    return f"- **{label} (as of {date_display})**: {value_display}"


def _write_cache(
    path: str,
    *,
    symbol: str,
    pe_value: Optional[float],
    peg_value: Optional[float],
    as_of: Optional[str],
) -> None:
    payload = {
        "symbol": symbol,
        "pe_ratio": pe_value,
        "peg_ratio": peg_value,
        "as_of": as_of,
        "source": "yfinance",
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except Exception:
        pass


@dataclass(frozen=True)
class PERatio:
    """Representation of a symbol's price-to-earnings ratio."""

    value: Optional[float]
    as_of: Optional[str]

    def as_summary_line(self) -> str:
        """Format the ratio for markdown summaries."""
        return _format_ratio_line("P/E ratio", self.value, self.as_of)


@dataclass(frozen=True)
class PEGRatio:
    """Representation of a symbol's price/earnings-to-growth ratio."""

    value: Optional[float]
    as_of: Optional[str]

    def as_summary_line(self) -> str:
        """Format the ratio for markdown summaries."""
        return _format_ratio_line("PEG ratio", self.value, self.as_of)


@dataclass(frozen=True)
class FundamentalRatios:
    """Container for cached fundamental ratios."""

    pe_ratio: PERatio
    peg_ratio: PEGRatio


def _coerce_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _load_cached(
    path: str,
) -> Tuple[Optional[float], Optional[float], Optional[str], bool]:
    if not os.path.exists(path):
        return None, None, None, False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return None, None, None, False
    pe_value = _coerce_number(payload.get("pe_ratio"))
    peg_value = _coerce_number(payload.get("peg_ratio"))
    as_of = payload.get("as_of")
    has_peg_key = "peg_ratio" in payload
    if isinstance(as_of, str):
        return pe_value, peg_value, as_of, has_peg_key
    return pe_value, peg_value, None, has_peg_key


def _fetch_finviz_ratios(
    symbol: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Fetch P/E, PEG, and growth forecasts from Finviz as a fallback."""

    try:
        import requests
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return None, None, None

    url = f"https://finviz.com/quote.ashx?t={symbol}"
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except Exception:
        return None, None, None

    soup = BeautifulSoup(response.text, "html.parser")
    labels = soup.select("td.snapshot-td2")
    if not labels:
        return None, None, None

    def parse_snapshot_value(raw: str) -> Optional[float]:
        text = raw.strip()
        if not text or text.upper() in {"-", "N/A"}:
            return None
        sanitized = text.replace(",", "")
        if sanitized.endswith("%"):
            sanitized = sanitized[:-1]
        return _coerce_number(sanitized)

    pe_value: Optional[float] = None
    peg_value: Optional[float] = None
    eps_next_5y: Optional[float] = None

    for cell in labels:
        label = cell.get_text(strip=True).upper()
        value_cell = cell.find_next_sibling("td")
        if not value_cell:
            continue
        raw_value = value_cell.get_text(strip=True)
        if label == "P/E" and pe_value is None:
            pe_value = parse_snapshot_value(raw_value)
        elif label == "PEG" and peg_value is None:
            peg_value = parse_snapshot_value(raw_value)
        elif label == "EPS NEXT 5Y" and eps_next_5y is None:
            eps_next_5y = parse_snapshot_value(raw_value)
        if pe_value is not None and peg_value is not None:
            break

    return pe_value, peg_value, eps_next_5y


def _days_between(date_iso: str, today_iso: str) -> Optional[int]:
    try:
        date_val = _dt.date.fromisoformat(date_iso)
        today_val = _dt.date.fromisoformat(today_iso)
    except ValueError:
        return None
    return (today_val - date_val).days


def fetch_fundamental_ratios(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Fetch trailing P/E and PEG ratios for ``symbol`` using yfinance."""

    try:
        import yfinance as yf  # type: ignore
    except ImportError:  # pragma: no cover - runtime dependency
        return None, None

    ticker = yf.Ticker(symbol)
    pe_value: Optional[float] = None
    peg_value: Optional[float] = None

    try:  # pragma: no branch - simple guard
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            pe_value = _coerce_number(fast_info.get("trailing_pe"))
            if pe_value is None:
                pe_value = _coerce_number(fast_info.get("pe_ratio"))
            peg_value = _coerce_number(fast_info.get("peg_ratio"))
    except Exception:
        pe_value = pe_value if pe_value is not None else None
        peg_value = peg_value if peg_value is not None else None

    growth_percent: Optional[float] = None

    if pe_value is None or peg_value is None:
        try:
            info = ticker.info  # type: ignore[attr-defined]
        except Exception:
            info = None
        if isinstance(info, dict):
            if pe_value is None:
                for key in ("trailingPE", "forwardPE"):
                    pe_value = _coerce_number(info.get(key))
                    if pe_value is not None:
                        break
            if peg_value is None:
                peg_value = _coerce_number(info.get("pegRatio"))

    if peg_value is None:
        try:
            growth_df = ticker.get_growth_estimates()  # type: ignore[attr-defined]
        except Exception:
            growth_df = None
        if growth_df is not None and hasattr(growth_df, "loc"):
            for label in ("LTG", "+1y", "+5y", "0y"):
                try:
                    row = growth_df.loc[label]
                except Exception:
                    continue
                candidate: Optional[float] = None
                if hasattr(row, "__getitem__"):
                    try:
                        candidate = _coerce_number(row["stockTrend"])
                    except Exception:
                        candidate = _coerce_number(row)
                else:
                    candidate = _coerce_number(row)
                if candidate is not None:
                    growth_percent = candidate * 100.0
                    break
        if (
            growth_percent is not None
            and pe_value is not None
            and growth_percent not in (0.0, -0.0)
        ):
            peg_value = pe_value / growth_percent

    if pe_value is None or peg_value is None:
        fallback_pe, fallback_peg, fallback_growth = _fetch_finviz_ratios(symbol)
        if pe_value is None:
            pe_value = fallback_pe
        if peg_value is None:
            peg_value = fallback_peg
        if (
            peg_value is None
            and pe_value is not None
            and fallback_growth is not None
            and fallback_growth not in (0.0, -0.0)
        ):
            peg_value = pe_value / fallback_growth

    return pe_value, peg_value


def ensure_fundamental_ratios(
    symbol: str, symbol_dir: str, *, stale_after_days: int = 7
) -> FundamentalRatios:
    """Ensure cached fundamental ratios exist for ``symbol``."""

    os.makedirs(symbol_dir, exist_ok=True)
    fundamentals_path = os.path.join(symbol_dir, FUNDAMENTALS_FILENAME)
    cached_pe, cached_peg, cached_as_of, cached_has_peg = _load_cached(
        fundamentals_path
    )
    today_iso = _dt.date.today().isoformat()

    needs_refresh = False
    if cached_as_of:
        age = _days_between(cached_as_of, today_iso)
        needs_refresh = age is None or age > stale_after_days
    else:
        needs_refresh = True

    if not cached_has_peg:
        needs_refresh = True

    if cached_pe is None or cached_peg is None:
        needs_refresh = True

    if (
        not needs_refresh
        and cached_as_of
        and (cached_pe is not None or cached_peg is not None)
    ):
        if not cached_has_peg:
            _write_cache(
                fundamentals_path,
                symbol=symbol,
                pe_value=cached_pe,
                peg_value=cached_peg,
                as_of=cached_as_of,
            )
        return FundamentalRatios(
            PERatio(cached_pe, cached_as_of),
            PEGRatio(cached_peg, cached_as_of),
        )

    fetched_pe, fetched_peg = fetch_fundamental_ratios(symbol)
    if fetched_pe is not None or fetched_peg is not None:
        _write_cache(
            fundamentals_path,
            symbol=symbol,
            pe_value=fetched_pe,
            peg_value=fetched_peg,
            as_of=today_iso,
        )
        return FundamentalRatios(
            PERatio(fetched_pe, today_iso),
            PEGRatio(fetched_peg, today_iso),
        )

    if cached_pe is not None or cached_peg is not None or cached_as_of is not None:
        if not cached_has_peg:
            _write_cache(
                fundamentals_path,
                symbol=symbol,
                pe_value=cached_pe,
                peg_value=cached_peg,
                as_of=cached_as_of,
            )
        return FundamentalRatios(
            PERatio(cached_pe, cached_as_of),
            PEGRatio(cached_peg, cached_as_of),
        )

    payload = {
        "symbol": symbol,
        "pe_ratio": None,
        "peg_ratio": None,
        "as_of": today_iso,
        "source": "yfinance",
    }
    _write_cache(
        fundamentals_path,
        symbol=symbol,
        pe_value=None,
        peg_value=None,
        as_of=today_iso,
    )
    return FundamentalRatios(PERatio(None, today_iso), PEGRatio(None, today_iso))


def ensure_pe_ratio(symbol: str, symbol_dir: str, *, stale_after_days: int = 7) -> PERatio:
    """Backward-compatible wrapper returning only the P/E ratio."""

    ratios = ensure_fundamental_ratios(
        symbol, symbol_dir, stale_after_days=stale_after_days
    )
    return ratios.pe_ratio


def ensure_peg_ratio(symbol: str, symbol_dir: str, *, stale_after_days: int = 7) -> PEGRatio:
    """Ensure a cached PEG ratio exists for ``symbol``."""

    ratios = ensure_fundamental_ratios(
        symbol, symbol_dir, stale_after_days=stale_after_days
    )
    return ratios.peg_ratio


__all__ = [
    "PERatio",
    "PEGRatio",
    "FundamentalRatios",
    "ensure_fundamental_ratios",
    "ensure_pe_ratio",
    "ensure_peg_ratio",
]
