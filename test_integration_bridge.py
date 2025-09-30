import pytest

pd = pytest.importorskip("pandas")

from strategy_tqqq_reserve import evaluate_integration_request


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

    # The bridge should still return a valid decision block.
    assert response["decision"]["action"] in {"rebalance", "hold"}
    assert isinstance(response["decision"]["reason"], str) and response["decision"]["reason"]


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
