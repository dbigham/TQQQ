import pytest

pd = pytest.importorskip("pandas")

from strategy_tqqq_reserve import evaluate_integration_request


def test_evaluate_request_without_last_rebalance():
    payload = {
        "experiment": "A1",
        "request_date": "2024-06-03",
        "positions": [],
    }

    response = evaluate_integration_request(payload)

    assert response["request_date"] == payload["request_date"]
    assert response["recent_rebalance"] is None
    assert response["model"]["days_since_last_rebalance"] == 0

    symbols = {pos["symbol"] for pos in response["current_positions"]}
    assert symbols == {"QQQ", "TQQQ", "CASH"}

    # The bridge should still return a valid decision block.
    assert response["decision"]["action"] in {"rebalance", "hold"}
    assert isinstance(response["decision"]["reason"], str) and response["decision"]["reason"]
