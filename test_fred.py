import pytest

pd = pytest.importorskip("pandas")

from tqqq import fred


def _frame(values):
    index = pd.to_datetime(["2020-01-01", "2020-01-02"])
    return pd.DataFrame({"rate": values}, index=index)


def test_fetch_fred_series_prefers_fredapi(monkeypatch):
    called = []

    def fake_fredapi(series_id, start, end, api_key):
        called.append("fredapi")
        return _frame([1.0, 2.0])

    monkeypatch.setattr(fred, "_fetch_with_fredapi", fake_fredapi)
    monkeypatch.setattr(fred, "_fetch_with_datareader", lambda *args, **kwargs: None)
    monkeypatch.setattr(fred, "_fetch_with_csv", lambda *args, **kwargs: None)

    df = fred.fetch_fred_series("TEST", "2020-01-01", "2020-01-10")
    assert called == ["fredapi"]
    assert list(df["rate"]) == [1.0, 2.0]


def test_fetch_fred_series_falls_back_to_csv(monkeypatch):
    def fake_csv(series_id, start, end, api_key, opener=None):
        return _frame([3.0, 4.0])

    monkeypatch.setattr(fred, "_fetch_with_fredapi", lambda *args, **kwargs: None)
    monkeypatch.setattr(fred, "_fetch_with_datareader", lambda *args, **kwargs: None)
    monkeypatch.setattr(fred, "_fetch_with_csv", fake_csv)

    df = fred.fetch_fred_series("TEST", "2020-01-01", "2020-01-10")
    assert list(df["rate"]) == [3.0, 4.0]
