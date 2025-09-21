import pytest

np = pytest.importorskip("numpy")

from tqqq.fitting import iterative_constant_growth


def test_iterative_fit_raises_when_no_points_remaining():
    t = np.array([0.0, 1.0, 2.0])
    prices = np.array([1.0, 2.0, 1000.0])
    with pytest.raises(RuntimeError):
        iterative_constant_growth(t, prices, thresholds=[0.0, 0.0])
