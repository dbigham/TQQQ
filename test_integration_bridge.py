import math

import pytest

pd = pytest.importorskip("pandas")

from nasdaq_temperature import NasdaqTemperature
from strategy_tqqq_reserve import (
    evaluate_integration_request,
    evaluate_temperature_chart_request,
)
from tqqq.dataset import load_price_csv


@pytest.fixture(autouse=True)
def stub_fred(monkeypatch):
    pd = pytest.importorskip("pandas")

    def fake_fetch(series_id, start, end, api_key=None):
        index = pd.date_range(start, end, freq="D")
        if index.empty:
            index = pd.to_datetime([start])
        return pd.DataFrame({"rate": [0.05] * len(index)}, index=index)

    monkeypatch.setattr("strategy_tqqq_reserve.download_fred_series", fake_fetch)


def test_evaluate_request_without_last_rebalance():
    payload = {
        "experiment": "A1",
        "request_date": "2024-06-03",
        "leveraged_symbol": "QQQ",  # Avoid network downloads during tests
        "positions": [],
    }

    response = evaluate_integration_request(payload)

    assert response["request_date"] == payload["request_date"]
    assert response["recent_rebalance"] is None
    assert response["model"]["days_since_last_rebalance"] == 0

    symbols = {pos["symbol"] for pos in response["current_positions"]}
    expected_symbols = {
        response["base_symbol"],
        response["leveraged_symbol"],
        response["reserve_symbol"],
    }
    assert symbols == expected_symbols

    # The bridge should either request an immediate rebalance or explain why the buy is blocked.
    assert response["decision"]["action"] in {"rebalance", "hold"}
    if response["decision"]["action"] == "hold":
        assert "temperature" in response["decision"]["reason"].lower()


def test_evaluate_request_with_weekend_last_rebalance():
    payload = {
        "experiment": "A1",
        "request_date": "2024-06-03",  # Monday
        "last_rebalance": {"date": "2024-06-01"},  # Saturday
        "leveraged_symbol": "QQQ",  # Avoid network downloads during tests
        "positions": [],
    }

    response = evaluate_integration_request(payload)

    assert response["request_date"] == payload["request_date"]
    assert response["model"]["days_since_last_rebalance"] is not None
    assert response["decision"]["action"] in {"rebalance", "hold"}


def test_temperature_chart_matches_reference_model():
    payload = {
        "experiment": "A1",
        "start_date": "2020-02-01",  # Weekend – should align to next trading day
        "end_date": "2020-02-10",
    }

    response = evaluate_temperature_chart_request(payload)

    assert response["experiment"] == "A1"
    assert response["base_symbol"] == "QQQ"
    assert response["resolved_start_date"] == "2020-02-03"
    assert response["resolved_end_date"] == "2020-02-10"
    assert response["reference_temperatures"] == [0.5, 1.0, 1.5]
    assert response["fit"]["manual_override"] is False

    model = NasdaqTemperature()
    df, _ = load_price_csv("unified_nasdaq.csv", set_index=True)
    start = pd.to_datetime(response["resolved_start_date"])
    end = pd.to_datetime(response["resolved_end_date"])
    df = df[(df.index >= start) & (df.index <= end)]
    expected_dates = [ts.strftime("%Y-%m-%d") for ts in df.index]

    points = response["points"]
    assert [point["date"] for point in points] == expected_dates

    for point in points:
        ts = point["date"]
        temp_expected = model.get_temperature(ts)
        close_expected = float(df.loc[pd.to_datetime(ts), "close"])  # type: ignore[index]
        fitted_expected = close_expected / temp_expected

        assert math.isclose(point["temperature"], temp_expected, rel_tol=1e-9)
        assert math.isclose(point["close"], close_expected, rel_tol=1e-9)
        assert math.isclose(point["fitted"], fitted_expected, rel_tol=1e-9)

    assert response["fit"]["start_date"] == model.start_ts.strftime("%Y-%m-%d")
    assert math.isclose(response["fit"]["growth_rate"], model.r, rel_tol=1e-12)


def test_temperature_chart_honours_allocation_curve():
    payload = {
        "experiment": "A1g",
        "start_date": "2020-01-02",
        "end_date": "2020-01-10",
    }

    response = evaluate_temperature_chart_request(payload)

    anchors = response.get("temperature_allocation")
    assert anchors is not None and len(anchors) > 0
    for anchor in anchors:
        assert set(anchor.keys()) == {"temp", "allocation"}
        assert isinstance(anchor["temp"], float)
        assert isinstance(anchor["allocation"], float)
