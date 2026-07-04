"""Monte Carlo standard errors and small shared helpers."""

from __future__ import annotations

import math

import numpy as np


def mcse_prop(p: float, n: int) -> float:
    """Binomial MCSE: sqrt(p (1 - p) / n)."""
    if n < 1 or p is None or math.isnan(p):
        return float("nan")
    return math.sqrt(p * (1.0 - p) / n)


def mcse_mean(x: np.ndarray) -> float:
    """MCSE of a mean: sd / sqrt(n) over finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def mcse_boot(stat_fun, n_included: int, rng: np.random.Generator,
              B: int = 500) -> float:
    """Nonparametric bootstrap MCSE for conditional diagnosands.

    Resamples the included simulations and recomputes the significance
    conditioning within each resample, so the uncertainty of the
    conditioning itself is propagated.
    """
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n_included, size=n_included)
        vals[b] = stat_fun(idx)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))
