"""Fragility curves — deliberately OUTSIDE the PASS/RISK/FAIL verdict.

Shrinking the target effect changes the inferential target, so
pessimistic scenario rows perturb nuisance assumptions only; fragility
to the effect size (and to binding nuisance parameters) is reported as
separate curves that show where the verdict would change.
"""

from __future__ import annotations

import numpy as np

from ._utils import mcse_mean, mcse_prop
from .constructors import ClusterTrial, RecoveryDesign
from .scenarios import scenario_params
from .simulate import run_scenario


def _curve(design, specs, sims, value_name, rng):
    unit = design.target.bias_scale_unit
    rows = []
    for value, overrides in specs:
        params = scenario_params(design, overrides)
        sim = run_scenario(design, params, sims, rng)
        theta = params["effect"]
        ok = (~sim["counted_failure"]) & np.isfinite(sim["estimate"])
        est, sig = sim["estimate"][ok], sim["sig"][ok]
        n_sig = int(sig.sum())
        pw = float(sig.mean()) if ok.any() else float("nan")
        width = (sim["ci_hi"] - sim["ci_lo"])[ok]
        drift = (sim["theta_obs"][ok] - theta) / unit
        rows.append({
            value_name: value,
            "power": pw, "power_mcse": mcse_prop(pw, int(ok.sum())),
            "target_bias": float(np.mean(est - theta)) / unit,
            "coverage": float(sim["covered"][ok].mean()),
            "drift": float(drift.mean()), "drift_mcse": mcse_mean(drift),
            "type_m": (float(np.mean(np.abs(est[sig]))) / abs(theta)
                       if n_sig > 0 else float("nan")),
            "type_m_n": n_sig,
            "precision": float(width.mean()),
        })
    return rows


def effect_fragility(design: RecoveryDesign, effects=None, sims: int = 500,
                     seed: int | None = None) -> list[dict]:
    """Power / Type M / precision as the true effect shrinks."""
    rng = np.random.default_rng(seed)
    if effects is None:
        s = 1 if design.effect > 0 else -1
        effects = sorted(set(
            [f * design.effect for f in (0.25, 0.50, 0.75, 1.0)]
            + [s * design.target.sesoi]))
    specs = [(e, {"effect": e}) for e in effects]
    return _curve(design, specs, sims, "effect", rng)


def nuisance_fragility(design: RecoveryDesign, parameter: str, values,
                       sims: int = 500,
                       seed: int | None = None) -> list[dict]:
    """Fragility over a binding nuisance parameter, at the SESOI."""
    if parameter not in ("attrition_rate", "reliability", "icc"):
        raise ValueError(
            "`parameter` must be attrition_rate/reliability/icc")
    if parameter == "icc" and not isinstance(design.data_strategy,
                                             ClusterTrial):
        raise ValueError("`icc` fragility requires a cluster_trial()")
    if parameter == "attrition_rate" and design.missingness is None:
        raise ValueError(
            "`attrition_rate` fragility requires a declared attrition_model()")
    rng = np.random.default_rng(seed)
    theta = (1 if design.effect > 0 else -1) * design.target.sesoi
    specs = []
    for v in values:
        ov = {"effect": theta}
        if parameter == "attrition_rate":
            ov.update(rate_control=v, rate_treated=v)
        elif parameter == "reliability":
            ov.update(reliability=v)
        else:
            ov.update(icc=v)
        specs.append((v, ov))
    return _curve(design, specs, sims, parameter, rng)
