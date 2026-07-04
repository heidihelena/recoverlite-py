"""Monte Carlo standard errors and small shared helpers."""

from __future__ import annotations

import math

import numpy as np


def mcse_prop(p: float, n: int) -> float:
    """Binomial MCSE: sqrt(p (1 - p) / n)."""
    if n < 1 or p is None or math.isnan(p):
        return float("nan")
    return math.sqrt(p * (1.0 - p) / n)


def wilson_upper(x: float, n: int, conf: float = 0.95) -> float:
    """One-sided Wilson score upper bound for a proportion x / n.

    Used for zero-count conditional diagnosands (e.g. Type S = 0): the
    point estimate is 0 but the run has not ruled out a rate up to this
    bound, so the verdict checks the threshold against the bound, not the
    point estimate (protocol section 2.4; technical comment C). For x = 0
    this reduces to z**2 / (n + z**2).
    """
    if n is None or n < 1:
        return float("nan")
    from scipy import stats
    z = float(stats.norm.ppf(conf))
    phat = x / n
    centre = phat + z ** 2 / (2 * n)
    halfwidth = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return (centre + halfwidth) / (1 + z ** 2 / n)


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
