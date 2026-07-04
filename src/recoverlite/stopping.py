"""The algorithmic doubling stopping rule (protocol section 2.4).

`recovery_test_stable` wraps `recovery_test`, doubling the number of
simulations S from `start_sims` to a pre-declared `max_sims` ceiling
while any required threshold margin is still within `mcse_margin` Monte
Carlo standard errors of its threshold. It stops as soon as the verdict
is determined: a stable declared failure locks FAIL, or all required
margins resolve (clean PASS / determined RISK). Anything still within the
MCSE band at `max_sims` is reported as RISK -- the irresolution is itself
the finding. Protocol-identical to the R `recovery_test_stable()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .recovery_test import recovery_test
from .verdict import evaluate_criteria, verdict


@dataclass
class StoppingRecord:
    start_sims: int
    final_sims: int
    max_sims: int
    resolved: bool
    hit_ceiling: bool
    unresolved: list  # list of (row, Criterion) still within the MCSE band


def _unresolved_required(result, thr) -> list:
    """Required-row criteria whose margins are still within the MCSE band
    (either side) or whose conditional diagnosand is unstable -- the
    margins doubling S can still resolve. Mirrors verdict row selection."""
    estimation = thr.profile == "estimation"
    runs = {nm: r for nm, r in result.runs.items()
            if r["scenario"].counts_for != "informational"
            and (not estimation or r["scenario"].row_type == "target")}
    required = (["target_declared", "target_pessimistic"] if estimation else
                ["null_declared", "null_pessimistic", "target_declared",
                 "target_pessimistic"])
    out = []
    for nm in required:
        if nm not in runs:
            continue
        evs = evaluate_criteria(runs[nm]["diagnosands"],
                                runs[nm]["scenario"].row_type, thr,
                                result.alpha)
        for c in evs:
            near = (c.passed is not None and math.isfinite(c.mcse)
                    and abs(c.margin) <= thr.mcse_margin * c.mcse)
            if near or c.unstable:
                out.append((nm, c))
    return out


def recovery_test_stable(design, start_sims: int = 2000,
                         max_sims: int = 16000,
                         scenarios: str = "confirmatory_grid",
                         thresholds=None, seed: int | None = None,
                         verbose: bool = False):
    """Run recovery_test under the doubling stopping rule.

    Returns a (RecoveryResult, StoppingRecord) tuple. Doubling reuses the
    same `seed`; each (seed, S) run is reproducible.
    """
    if start_sims < 2:
        raise ValueError("`start_sims` must be >= 2")
    if max_sims < start_sims:
        raise ValueError("`max_sims` must be >= `start_sims`")
    s = int(start_sims)
    while True:
        res = recovery_test(design, sims=s, scenarios=scenarios,
                            thresholds=thresholds, seed=seed)
        thr = res.thresholds
        v = verdict(res)
        unres = _unresolved_required(res, thr)
        determined = v.verdict == "FAIL" or len(unres) == 0
        if verbose:
            print(f"S = {s}: verdict {v.verdict}; {len(unres)} required "
                  f"margin(s) within {thr.mcse_margin:g} MCSE")
        if determined or s >= max_sims:
            break
        s = min(2 * s, int(max_sims))
    record = StoppingRecord(
        start_sims=int(start_sims), final_sims=s, max_sims=int(max_sims),
        resolved=(len(unres) == 0 or v.verdict == "FAIL"),
        hit_ceiling=(s >= max_sims and len(unres) > 0 and v.verdict != "FAIL"),
        unresolved=unres)
    return res, record
