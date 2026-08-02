"""Estimator registry: a new estimand class is a plugin, not a fork.

The declaration API (target_estimand / data_strategy / measurement /
missingness / answer_strategy) is general; the shipped estimators are
not. This registry closes that gap: `register_estimator()` makes any
callable with the standard record contract available to
`planned_analysis()`, and the four shipped trial estimators register
through the same door (simulate.py).

The record contract is the protocol's four-class failure taxonomy: a
fit function takes (analysis, data, rng) and returns a dict with keys
estimate / se / ci_lo / ci_hi / p (floats, NaN allowed) and fatal /
nonconverged / degenerate / warned (bools). Hard errors must be
recorded as fatal=True, not raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

RECORD_KEYS = frozenset(("estimate", "se", "ci_lo", "ci_hi", "p",
                         "fatal", "nonconverged", "degenerate", "warned"))


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    fit: Callable  # fit(analysis, data, rng) -> record dict
    needs_cluster: bool = False
    description: str = ""


_REGISTRY: dict[str, EstimatorSpec] = {}


def _ensure_shipped() -> None:
    # The shipped estimators register when simulate.py is imported;
    # lazy so `recoverlite.constructors` alone still validates.
    from . import simulate  # noqa: F401


def register_estimator(name: str, fit: Callable, *,
                       needs_cluster: bool = False,
                       description: str = "",
                       overwrite: bool = False) -> EstimatorSpec:
    """Register an estimator under `name` for use in planned_analysis().

    `fit(analysis, data, rng)` must honor the record contract in the
    module docstring. Overwriting a registered name (including a shipped
    one) requires `overwrite=True`: replacing an estimator silently
    would change what a declaration means.
    """
    if not name or not isinstance(name, str):
        raise ValueError("`name` must be a non-empty string")
    if not callable(fit):
        raise TypeError("`fit` must be callable")
    _ensure_shipped()
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"estimator '{name}' is already registered; pass "
            "overwrite=True to replace it deliberately")
    spec = EstimatorSpec(name=name, fit=fit, needs_cluster=bool(needs_cluster),
                         description=description)
    _REGISTRY[name] = spec
    return spec


def get_estimator(name: str) -> EstimatorSpec:
    _ensure_shipped()
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown estimator '{name}'; registered estimators: "
            f"{', '.join(sorted(_REGISTRY))}. Use register_estimator() "
            "to add one.")
    return _REGISTRY[name]


def registered_estimators() -> tuple[str, ...]:
    _ensure_shipped()
    return tuple(sorted(_REGISTRY))
