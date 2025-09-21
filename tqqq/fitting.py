"""Shared constant-growth fitting utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FitStep:
    """A single iteration of the constant-growth fit."""

    A: float
    r: float
    predictions: np.ndarray
    relative_errors: np.ndarray
    mask: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FitResult:
    """Full 3-step iterative fit output."""

    step1: FitStep
    step2: FitStep
    step3: FitStep

    @property
    def final(self) -> FitStep:
        """Convenience accessor for the last step."""

        return self.step3


def _as_arrays(t_years, prices) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_years, dtype=float)
    p = np.asarray(prices, dtype=float)
    if t.shape != p.shape:
        raise ValueError("t_years and prices must have identical shapes")
    if t.ndim != 1:
        raise ValueError("t_years and prices must be 1-dimensional")
    return t, p


def _fit_log_linear(t_years: np.ndarray, prices: np.ndarray) -> Tuple[float, float]:
    y = np.log(prices)
    slope, intercept = np.polyfit(t_years, y, 1)
    A = float(math.exp(intercept))
    r = float(math.exp(slope) - 1.0)
    return A, r


def _compute_relative_errors(A: float, r: float, t_years: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = A * np.power(1.0 + r, t_years)
    rel_err = np.abs((prices - pred) / pred)
    return pred, rel_err


def iterative_constant_growth(
    t_years,
    prices,
    thresholds: Sequence[float],
) -> FitResult:
    """Run the 3-step constant-growth fit with outlier rejection."""

    if len(thresholds) < 2:
        raise ValueError("Expected at least two thresholds for steps 2 and 3")

    thresh2, thresh3 = float(thresholds[0]), float(thresholds[1])

    t, p = _as_arrays(t_years, prices)

    A1, r1 = _fit_log_linear(t, p)
    pred1, rel1 = _compute_relative_errors(A1, r1, t, p)
    step1 = FitStep(A=A1, r=r1, predictions=pred1, relative_errors=rel1, mask=None)

    mask2 = rel1 <= thresh2
    if not np.any(mask2):
        raise RuntimeError(
            f"No points remain after applying threshold {thresh2:.3f} in step 2; try loosening the cutoff."
        )
    A2, r2 = _fit_log_linear(t[mask2], p[mask2])
    pred2, rel2 = _compute_relative_errors(A2, r2, t, p)
    step2 = FitStep(A=A2, r=r2, predictions=pred2, relative_errors=rel2, mask=mask2)

    mask3 = rel2 <= thresh3
    if not np.any(mask3):
        raise RuntimeError(
            f"No points remain after applying threshold {thresh3:.3f} in step 3; try loosening the cutoff."
        )
    A3, r3 = _fit_log_linear(t[mask3], p[mask3])
    pred3, rel3 = _compute_relative_errors(A3, r3, t, p)
    step3 = FitStep(A=A3, r=r3, predictions=pred3, relative_errors=rel3, mask=mask3)

    return FitResult(step1=step1, step2=step2, step3=step3)


def fit_constant_growth(t_years, prices, thresholds: Sequence[float]) -> Tuple[float, float]:
    """Convenience wrapper returning only the final ``A`` and ``r``."""

    result = iterative_constant_growth(t_years, prices, thresholds)
    final = result.final
    return final.A, final.r


__all__ = ["FitResult", "FitStep", "fit_constant_growth", "iterative_constant_growth"]
