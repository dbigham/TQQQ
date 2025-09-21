import pytest

pd = pytest.importorskip("pandas")

from tqqq.dataset import PriceDataError, load_price_csv


def test_load_price_csv_filters_and_computes_elapsed_years(tmp_path):
    csv = tmp_path / "series.csv"
    csv.write_text(
        "date,close\n"
        "2020-01-01,100\n"
        "2020-01-02,\n"
        "2020-01-03,-5\n"
        "2020-01-04,120\n"
    )

    df, start = load_price_csv(str(csv), add_elapsed_years=True)

    assert list(df.index) == [0, 3]  # preserved row order indices
    assert start == pd.Timestamp("2020-01-01")
    assert list(df["close"]) == [100, 120]
    assert df.loc[0, "t_years"] == 0.0
    expected_years = (pd.Timestamp("2020-01-04") - start) / pd.Timedelta(days=365.25)
    assert df.loc[3, "t_years"] == pytest.approx(expected_years)


def test_load_price_csv_requires_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("date,value\n2020-01-01,1\n")

    with pytest.raises(PriceDataError):
        load_price_csv(str(csv))
