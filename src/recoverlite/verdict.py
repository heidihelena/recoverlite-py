"""The PASS/RISK/FAIL verdict rule (protocol Step 5).

PASS: all required thresholds met under all scenario rows the selected
profile requires, every margin > mcse_margin MCSEs. RISK: pessimistic-
only failure, any margin within mcse_margin MCSEs, an unstable required
conditional diagnosand, or missing required rows. FAIL: any required
threshold fails under a declared-nuisance row. The verdict is a decision
convention, not a validity classification, and the full report always
travels with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .thresholds import Thresholds, recovery_thresholds


@dataclass
class Criterion:
    criterion: str
    value: float
    mcse: float
    requirement: str
    margin: float
    passed: bool | None
    stable: bool
    unstable: bool
    n_contributing: int
    note: str = ""


@dataclass
class Verdict:
    verdict: str
    binding: str | None
    evaluations: dict
    smallest_margin: str | None
    verdict_strict: str | None
    verdict_lenient: str | None
    thresholds: Thresholds
    notes: tuple[str, ...] = ()

    def __str__(self):
        lines = [f"Recovery-test verdict: {self.verdict} "
                 f"(profile '{self.thresholds.profile}')"]
        if self.verdict_strict is not None:
            agree = len({self.verdict_strict, self.verdict,
                         self.verdict_lenient}) == 1
            lines.append(
                f"Under shipped profiles: strict {self.verdict_strict} | "
                f"default-family {self.verdict} | lenient "
                f"{self.verdict_lenient}"
                + ("" if agree else
                   "  <- profile disagreement is itself a finding"))
        if self.binding:
            lines.append(f"Binding failure mode: {self.binding}")
        if self.smallest_margin:
            lines.append(f"Smallest signed margin: {self.smallest_margin}")
        lines.append("The verdict is a decision convention, not a validity "
                     "classification; the full report() must travel with it.")
        return "\n".join(lines)


def evaluate_criteria(diag: dict, row_type: str, thr: Thresholds,
                      alpha: float) -> list[Criterion]:
    rows = diag["rows"]
    estimation = thr.profile == "estimation"
    crits: list[dict] = []

    def g(name):
        return rows[name]

    if not estimation:
        r = g("rejection_rate")
        if row_type == "null":
            lim = thr.null_rejection_mult * alpha
            crits.append(dict(criterion="target_null_rejection", d=r,
                              requirement=f"<= {lim:.4g} "
                                          f"({thr.null_rejection_mult:g} x alpha)",
                              margin=lim - r.value, unstable=False, note=""))
        else:
            crits.append(dict(criterion="power", d=r,
                              requirement=f">= {thr.power:g}",
                              margin=r.value - thr.power,
                              unstable=False, note=""))

    tb = g("target_bias")
    crits.append(dict(criterion="target_bias", d=tb,
                      requirement=f"|value| <= {thr.target_bias:g} Delta",
                      margin=thr.target_bias - abs(tb.value),
                      unstable=False, note=""))

    cov = g("coverage")
    cov_note = ""
    if math.isfinite(cov.value) and cov.value > thr.overcoverage_flag:
        cov_note = (f"overcoverage (> {thr.overcoverage_flag:g}): flagged as "
                    "inefficiency, evaluated through precision, not failure")
    crits.append(dict(criterion="coverage", d=cov,
                      requirement=f">= {thr.coverage:g} (lower bound)",
                      margin=cov.value - thr.coverage,
                      unstable=False, note=cov_note))

    if not estimation and row_type == "target":
        for nm, lim in (("type_s", thr.type_s), ("type_m", thr.type_m)):
            d = g(nm)
            crits.append(dict(criterion=nm, d=d,
                              requirement=f"<= {lim:g}",
                              margin=lim - d.value,
                              unstable=d.unstable, note=""))

    if estimation:
        d = g("estimand_drift")
        crits.append(dict(criterion="estimand_drift", d=d,
                          requirement=f"|value| <= {thr.drift:g} Delta",
                          margin=thr.drift - abs(d.value),
                          unstable=False, note=""))
        if thr.max_width is not None:
            d = g("precision")
            crits.append(dict(criterion="precision", d=d,
                              requirement="mean width <= declared max "
                                          f"{thr.max_width:g}",
                              margin=thr.max_width - d.value,
                              unstable=False, note=""))

    mfd = g("model_failure")
    crits.append(dict(criterion="model_failure", d=mfd,
                      requirement=f"<= {thr.model_failure:g} (counted classes)",
                      margin=thr.model_failure - mfd.value,
                      unstable=False, note=""))

    out = []
    for c in crits:
        d = c["d"]
        unstable = c["unstable"] or not math.isfinite(d.value)
        passed = None if unstable else c["margin"] >= 0
        stable = (not unstable and math.isfinite(d.mcse)
                  and c["margin"] > thr.mcse_margin * d.mcse)
        out.append(Criterion(criterion=c["criterion"], value=d.value,
                             mcse=d.mcse, requirement=c["requirement"],
                             margin=c["margin"], passed=passed,
                             stable=stable, unstable=unstable,
                             n_contributing=d.n_contributing,
                             note=c["note"]))
    return out


def _verdict_under(result, thr: Thresholds) -> dict:
    estimation = thr.profile == "estimation"
    runs = {nm: r for nm, r in result.runs.items()
            if r["scenario"].counts_for != "informational"
            and (not estimation or r["scenario"].row_type == "target")}
    required = (["target_declared", "target_pessimistic"] if estimation else
                ["null_declared", "null_pessimistic", "target_declared",
                 "target_pessimistic"])
    missing = [nm for nm in required if nm not in runs]

    evaluations = {nm: evaluate_criteria(r["diagnosands"],
                                         r["scenario"].row_type, thr,
                                         result.alpha)
                   for nm, r in runs.items()}

    def pick(counts_for, cond):
        found = []
        for nm, evs in evaluations.items():
            if counts_for and runs[nm]["scenario"].counts_for != counts_for:
                continue
            found += [(nm, c) for c in evs if cond(c)]
        return found

    failed_declared = pick("declared",
                           lambda c: c.passed is False)
    failed_pess = pick("pessimistic", lambda c: c.passed is False)
    narrow = pick(None, lambda c: c.passed is True and not c.stable)
    unstable = pick(None, lambda c: c.unstable)

    all_crit = [(nm, c) for nm, evs in evaluations.items() for c in evs
                if math.isfinite(c.margin)]
    smallest = None
    if all_crit:
        nm, c = min(all_crit, key=lambda t: t[1].margin)
        smallest = (f"{c.criterion} under '{nm}' "
                    f"(signed margin {c.margin:+.4f})")

    if failed_declared:
        v = "FAIL"
        binding = "Under declared-nuisance rows, " + "; ".join(
            f"{c.criterion} = {c.value:.3g} under '{nm}' violates "
            f"{c.requirement}" for nm, c in failed_declared) + "."
    elif failed_pess:
        v = "RISK"
        binding = ("Thresholds hold under declared-nuisance rows but fail "
                   "under pessimistic rows: " + "; ".join(
                       f"{c.criterion} = {c.value:.3g} under '{nm}' violates "
                       f"{c.requirement}" for nm, c in failed_pess) + ".")
    elif missing:
        v = "RISK"
        binding = (f"Required scenario rows not evaluated: "
                   f"{', '.join(missing)}. A PASS requires all rows the "
                   f"'{thr.profile}' profile requires.")
    elif narrow or unstable:
        v = "RISK"
        parts = []
        if narrow:
            parts.append("; ".join(
                f"{c.criterion} margin under '{nm}' is within "
                f"{thr.mcse_margin:g} MCSE of its threshold"
                for nm, c in narrow))
        if unstable:
            parts.append("; ".join(
                f"{c.criterion} under '{nm}' is unstable "
                f"({c.n_contributing} contributing simulations)"
                for nm, c in unstable))
        binding = ("All point estimates meet the thresholds, but simulation "
                   "precision is insufficient to confirm a PASS: "
                   + ". ".join(parts) + ". Increase `sims`.")
    else:
        v = "PASS"
        binding = None
    return dict(verdict=v, binding=binding, evaluations=evaluations,
                smallest_margin=smallest)


def verdict(result) -> Verdict:
    """Apply the verdict rule; recompute under strict and lenient."""
    thr = result.thresholds
    main = _verdict_under(result, thr)
    strict = lenient = None
    if thr.profile != "estimation":
        common = dict(overcoverage_flag=thr.overcoverage_flag,
                      mcse_margin=thr.mcse_margin,
                      min_conditional_n=thr.min_conditional_n,
                      max_width=thr.max_width)
        strict = _verdict_under(result,
                                recovery_thresholds("strict", **common))
        lenient = _verdict_under(result,
                                 recovery_thresholds("lenient", **common))
    return Verdict(verdict=main["verdict"], binding=main["binding"],
                   evaluations=main["evaluations"],
                   smallest_margin=main["smallest_margin"],
                   verdict_strict=None if strict is None
                   else strict["verdict"],
                   verdict_lenient=None if lenient is None
                   else lenient["verdict"],
                   thresholds=thr)
