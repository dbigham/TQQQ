import numpy as np
import pytest

from fit_constant_growth import iterative_fit


def test_iterative_fit_raises_when_no_points_remaining():
    t = np.array([0.0, 1.0, 2.0])
    prices = np.array([1.0, 2.0, 1000.0])
    with pytest.raises(RuntimeError):
        iterative_fit(np, t, prices, thresholds=[0.0, 0.0])
