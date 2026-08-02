"""recovery_test(): simulate the scenario grid and diagnose (Step 4).

Includes the mandatory reference-analysis calibration check: before any
scenario row is scored, the planned analysis is run on ideal data
(perfect reliability, no attrition, full compliance, declared design
structure) and must attain its nominal properties -- CI coverage at
1 - alpha and zero bias, within the MCSE stability band. A stable
deviation means the coverage and bias criteria partly measure
DGP-versus-analysis mismatch rather than design behavior, and the
verdict is capped at RISK. Same argument as the zero-effect null row,
applied to the reference analysis instead of the signal.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ._utils import mcse_mean, mcse_prop
from .constructors import RecoveryDesign
from .diagnosands import compute_diagnosands
from .scenarios import build_scenarios, scenario_params
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
    reference: dict | None = None

    def __str__(self):
        lines = [f"Recovery-test result: {self.sims} simulations per "
                 f"scenario row; {len(self.runs)} row(s); "
                 f"{self.elapsed_secs:.1f} s elapsed."]
        if self.reference is not None:
            lines.append("Reference-analysis calibration: "
                         + self.reference["status"].upper())
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


def reference_calibration(design: RecoveryDesign, sims: int,
                          rng: np.random.Generator,
                          thr: Thresholds) -> dict:
    """Run the planned analysis on ideal data and check nominal properties.

    Ideal data keeps the declared design STRUCTURE (n, allocation,
    clustering, declared ICC) and strips every nuisance the protocol
    charges to the design: reliability -> 1, attrition -> none,
    noncompliance -> 0. Two checks, each against the mcse_margin band
    that governs the verdict:

      coverage: |coverage - (1 - alpha)| vs mcse_margin MCSEs
      bias:     |E[est] - theta| / Delta  vs mcse_margin MCSEs

    Status is three-way: "attained" (no resolvable deviation at this S),
    "not_attained" (a deviation stable at this S), "not_estimable"
    (no successful fits). "attained" is a statement about resolution,
    not a proof of exactness -- the MCSEs travel with it.
    """
    theta = (1 if design.effect > 0 else -1) * design.target.sesoi
    params = scenario_params(design, {"effect": theta})
    params["reliability"] = None
    params["attrition"] = None
    if "noncompliance" in params:
        params["noncompliance"] = 0.0
    sim = run_scenario(design, params, sims, rng)

    ok = (~sim["counted_failure"]) & np.isfinite(sim["estimate"])
    n_ok = int(ok.sum())
    nominal = 1 - design.answer_strategy.alpha
    unit = design.target.bias_scale_unit
    k = thr.mcse_margin
    if n_ok == 0:
        return dict(status="not_estimable", n_ok=0, sims=int(sims),
                    nominal=nominal, coverage=float("nan"),
                    coverage_mcse=float("nan"), bias=float("nan"),
                    bias_mcse=float("nan"), deviations=(),
                    note="no successful fits on ideal data")

    cov = float(sim["covered"][ok].mean())
    cov_mcse = mcse_prop(cov, n_ok)
    est = sim["estimate"][ok]
    bias = float(np.mean(est - theta)) / unit
    bias_mcse = mcse_mean(est - theta) / unit

    deviations = []
    if math.isfinite(cov_mcse) and abs(cov - nominal) > k * cov_mcse:
        deviations.append(
            f"coverage {cov:.4f} deviates from nominal {nominal:g} by more "
            f"than {k:g} MCSE ({cov_mcse:.4f})")
    if math.isfinite(bias_mcse) and abs(bias) > k * bias_mcse:
        deviations.append(
            f"bias {bias:+.4f} Delta deviates from 0 by more than "
            f"{k:g} MCSE ({bias_mcse:.4f})")

    status = "not_attained" if deviations else "attained"
    note = ("; ".join(deviations) if deviations else
            f"no resolvable deviation at S = {sims} (coverage "
            f"{cov:.4f} [{cov_mcse:.4f}] vs nominal {nominal:g}; bias "
            f"{bias:+.4f} [{bias_mcse:.4f}] Delta)")
    return dict(status=status, n_ok=n_ok, sims=int(sims), nominal=nominal,
                coverage=cov, coverage_mcse=cov_mcse, bias=bias,
                bias_mcse=bias_mcse, deviations=tuple(deviations), note=note)


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

    # Mandatory reference-analysis calibration check (after the grid so
    # scenario-row draws stay reproducible against earlier versions for
    # the same seed).
    reference = reference_calibration(design, sims, rng, thr)

    return RecoveryResult(
        design=design, runs=runs, sims=int(sims), thresholds=thr,
        seed=seed, alpha=design.answer_strategy.alpha,
        scenario_request=scenarios, evidence_tiers=tuple(tiers),
        elapsed_secs=time.time() - t0, reference=reference)
