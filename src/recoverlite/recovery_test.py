"""recovery_test(): simulate the scenario grid and diagnose (Step 4)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .constructors import RecoveryDesign
from .diagnosands import compute_diagnosands
from .scenarios import build_scenarios
from .simulate import run_scenario
from .thresholds import Thresholds, recovery_thresholds


@dataclass
class RecoveryResult:
    design: RecoveryDesign
    runs: dict
    sims: int
    thresholds: Thresholds
    seed: int | None
    alpha: float
    scenario_request: str
    evidence_tiers: tuple[str, ...]
    elapsed_secs: float

    def __str__(self):
        lines = [f"Recovery-test result: {self.sims} simulations per "
                 f"scenario row; {len(self.runs)} row(s); "
                 f"{self.elapsed_secs:.1f} s elapsed."]
        for nm, run in self.runs.items():
            lines.append(f"\n-- {run['scenario'].label} --")
            for r in run["diagnosands"]["rows"].values():
                val = ("not estimable" if not np.isfinite(r.value)
                       else f"{r.value:8.4f} [{r.mcse:.4f}]")
                lines.append(
                    f"  {r.name:<20s} {val}  n={r.n_contributing}"
                    + ("  UNSTABLE" if r.unstable else ""))
        lines.append("\nUse verdict() and report() to evaluate.")
        return "\n".join(lines)


def recovery_test(design: RecoveryDesign, sims: int = 2000,
                  scenarios: str = "confirmatory_grid",
                  thresholds: Thresholds | None = None,
                  seed: int | None = None) -> RecoveryResult:
    """Simulate the crossed scenario grid and compute all diagnosands.

    2000 sims per row is an initial working number, not a standard: the
    relevant stopping rule is whether the MCSE is small enough to
    support the verdict. Set `seed` or the run is not reproducible.
    """
    if not isinstance(design, RecoveryDesign):
        raise TypeError("`design` must come from declare_recovery()")
    if sims < 2:
        raise ValueError("`sims` must be >= 2")
    thr = thresholds or recovery_thresholds()
    if thr.max_width is None and design.target.max_width is not None:
        thr = Thresholds(**{**thr.__dict__,
                            "max_width": design.target.max_width})
    rng = np.random.default_rng(seed)
    scs, tiers = build_scenarios(design, scenarios)
    has_attrition = (design.missingness is not None
                     and design.missingness.rate > 0)

    t0 = time.time()
    runs = {}
    for nm, sc in scs.items():
        sim = run_scenario(design, sc.params, sims, rng)
        diag = compute_diagnosands(
            sim, theta=sc.params["effect"],
            unit=design.target.bias_scale_unit, thresholds=thr,
            row_type=sc.row_type, has_attrition=has_attrition,
            alpha=design.answer_strategy.alpha, rng=rng)
        runs[nm] = {"scenario": sc, "sim_data": sim, "diagnosands": diag,
                    "theta": sc.params["effect"]}

    return RecoveryResult(
        design=design, runs=runs, sims=int(sims), thresholds=thr,
        seed=seed, alpha=design.answer_strategy.alpha,
        scenario_request=scenarios, evidence_tiers=tuple(tiers),
        elapsed_secs=time.time() - t0)
