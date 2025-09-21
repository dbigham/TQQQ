import datetime as dt
import importlib.util
import json
from pathlib import Path

import sys
import types

import pytest


_FUNDAMENTALS_SPEC = importlib.util.spec_from_file_location(
    "fundamentals_test_module",
    Path(__file__).with_name("tqqq") / "fundamentals.py",
)
assert _FUNDAMENTALS_SPEC and _FUNDAMENTALS_SPEC.loader
fundamentals = importlib.util.module_from_spec(_FUNDAMENTALS_SPEC)
sys.modules[_FUNDAMENTALS_SPEC.name] = fundamentals
_FUNDAMENTALS_SPEC.loader.exec_module(fundamentals)


def test_missing_cached_peg_triggers_refresh(tmp_path, monkeypatch):
    """PEG-less caches should be refreshed even when otherwise fresh."""

    symbol = "ABC"
    symbol_dir = tmp_path / symbol.lower()
    symbol_dir.mkdir()

    today_iso = dt.date.today().isoformat()
    cache_path = symbol_dir / fundamentals.FUNDAMENTALS_FILENAME
    cache_payload = {
        "symbol": symbol,
        "pe_ratio": 17.5,
        "as_of": today_iso,
        "source": "yfinance",
    }
    cache_path.write_text(json.dumps(cache_payload, indent=2, sort_keys=True) + "\n")

    calls = []

    def fake_fetch(requested_symbol: str):
        calls.append(requested_symbol)
        return 21.0, 1.4

    monkeypatch.setattr(fundamentals, "fetch_fundamental_ratios", fake_fetch)

    ratios = fundamentals.ensure_fundamental_ratios(symbol, str(symbol_dir))

    assert calls == [symbol]
    assert ratios.pe_ratio.value == pytest.approx(21.0)
    assert ratios.peg_ratio.value == pytest.approx(1.4)

    updated_payload = json.loads(cache_path.read_text())
    assert updated_payload["peg_ratio"] == pytest.approx(1.4)
    assert updated_payload["as_of"] == today_iso


def test_fetch_ratios_falls_back_to_finviz(monkeypatch):
    """Finviz data should be used when yfinance lacks PEG figures."""

    calls = []

    class DummyTicker:
        def __init__(self, symbol: str):
            calls.append(symbol)
            self.fast_info = {}
            self.info = {}

    dummy_module = types.SimpleNamespace(Ticker=lambda symbol: DummyTicker(symbol))
    monkeypatch.setitem(sys.modules, "yfinance", dummy_module)

    finviz_calls = []

    def fake_finviz(symbol: str):
        finviz_calls.append(symbol)
        return 17.2, 1.5, None

    monkeypatch.setattr(fundamentals, "_fetch_finviz_ratios", fake_finviz)

    pe, peg = fundamentals.fetch_fundamental_ratios("XYZ")

    assert pe == pytest.approx(17.2)
    assert peg == pytest.approx(1.5)
    assert calls == ["XYZ"]
    assert finviz_calls == ["XYZ"]
