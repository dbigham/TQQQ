import pytest

pd = pytest.importorskip("pandas")

from tqqq.simulation import simulate_leveraged_path


def test_simulate_leveraged_path_basic_growth():
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    unified = pd.DataFrame({"close": [100.0, 110.0, 121.0]}, index=index)
    rates = pd.DataFrame({"rate": [0.0, 0.0, 0.0]}, index=index)
    actual = pd.DataFrame({"actual": [50.0, 55.0, 60.0]}, index=index)

    result = simulate_leveraged_path(
        unified,
        rates,
        actual,
        leverage=2.0,
        annual_fee=0.0,
        trading_days=1,
        borrow_divisor=1.0,
        actual_column="actual",
        output_column="sim",
    )

    expected = [50.0, 50.0 * (1 + 2 * 0.1), 50.0 * (1 + 2 * 0.1) * (1 + 2 * 0.1)]
    assert result["sim"].tolist() == pytest.approx(expected)
    assert result["actual"].tolist() == [50.0, 55.0, 60.0]
