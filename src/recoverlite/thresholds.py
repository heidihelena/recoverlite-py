"""Versioned verdict threshold profiles (protocol section 2.5).

The threshold-set version string is shared with the R implementation:
the profiles are the protocol artifact, and both implementations must
agree on them exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

THRESHOLD_SET_VERSION = "recoverlite-thresholds-0.2"

_SHIPPED = {
    "lenient": dict(null_rejection_mult=1.50, power=0.70, target_bias=0.10,
                    coverage=0.900, type_s=0.05, type_m=2.00,
                    model_failure=0.05),
    "default": dict(null_rejection_mult=1.25, power=0.80, target_bias=0.05,
                    coverage=0.925, type_s=0.01, type_m=1.50,
                    model_failure=0.01),
    "strict": dict(null_rejection_mult=1.10, power=0.90, target_bias=0.025,
                   coverage=0.940, type_s=0.005, type_m=1.25,
                   model_failure=0.005),
}


@dataclass(frozen=True)
class Thresholds:
    profile: str
    null_rejection_mult: float
    power: float
    target_bias: float
    coverage: float
    type_s: float
    type_m: float
    model_failure: float
    drift: float
    overcoverage_flag: float = 0.975
    mcse_margin: float = 2.0
    min_conditional_n: int = 200
    max_width: float | None = None
    version: str = THRESHOLD_SET_VERSION
    modified: tuple[str, ...] = ()


def recovery_thresholds(profile: str = "default", *,
                        null_rejection_mult: float | None = None,
                        power: float | None = None,
                        target_bias: float | None = None,
                        coverage: float | None = None,
                        type_s: float | None = None,
                        type_m: float | None = None,
                        model_failure: float | None = None,
                        drift: float | None = None,
                        overcoverage_flag: float = 0.975,
                        mcse_margin: float = 2.0,
                        min_conditional_n: int = 200,
                        max_width: float | None = None) -> Thresholds:
    """Select a shipped profile, optionally deviating from it.

    Every deviation is recorded in `modified` and echoed in the report.
    Deviations must be chosen BEFORE simulation, never after seeing a
    verdict.
    """
    if profile not in ("default", "strict", "lenient", "estimation"):
        raise ValueError("`profile` must be default/strict/lenient/estimation")
    base = _SHIPPED["default" if profile == "estimation" else profile]

    values = dict(
        null_rejection_mult=(null_rejection_mult
                             if null_rejection_mult is not None
                             else base["null_rejection_mult"]),
        power=power if power is not None else base["power"],
        target_bias=(target_bias if target_bias is not None
                     else base["target_bias"]),
        coverage=coverage if coverage is not None else base["coverage"],
        type_s=type_s if type_s is not None else base["type_s"],
        type_m=type_m if type_m is not None else base["type_m"],
        model_failure=(model_failure if model_failure is not None
                       else base["model_failure"]),
    )
    modified = tuple(k for k in base if values[k] != base[k])
    # The drift threshold tracks the resolved bias threshold unless set.
    drift_val = drift if drift is not None else values["target_bias"]
    if drift is not None and drift != values["target_bias"]:
        modified = tuple(list(modified) + ["drift"])

    return Thresholds(profile=profile, drift=drift_val,
                      overcoverage_flag=overcoverage_flag,
                      mcse_margin=mcse_margin,
                      min_conditional_n=int(min_conditional_n),
                      max_width=max_width, modified=modified, **values)
