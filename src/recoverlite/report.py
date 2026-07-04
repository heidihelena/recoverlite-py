"""The standalone recovery report (protocol Step 6)."""

from __future__ import annotations

import math

from . import __version__
from .constructors import ClusterTrial, TwoArmTrial
from .verdict import verdict as compute_verdict


def describe_design(d) -> list[str]:
    ds = d.data_strategy
    a = d.answer_strategy
    lines = []
    if isinstance(ds, TwoArmTrial):
        lines.append(
            f"Data strategy: two-arm randomized trial; {ds.n_per_arm} "
            f"recruited per arm (allocation {ds.allocation:g}); "
            f"baseline-outcome correlation rho = "
            f"{ds.baseline_outcome_cor:g} (baseline observed at "
            "randomization for every participant).")
        if ds.noncompliance > 0:
            lines.append(f"One-sided noncompliance: {ds.noncompliance:g} of "
                         "treated receive no treatment.")
    else:
        bound = ("no upper plausible bound declared"
                 if ds.icc_pessimistic is None
                 else f"upper plausible bound {ds.icc_pessimistic:g}")
        lines.append(
            f"Data strategy: cluster-randomized trial; {ds.n_clusters} "
            f"clusters x {ds.n_per_cluster} individuals, sizes fixed by "
            f"design (allocation {ds.allocation:g}); declared ICC "
            f"{ds.icc:g}, {bound}.")
    if d.measurement is not None:
        lines.append(
            "Measurement: classical additive error, Var(e) = 1/r - 1 with "
            f"declared reliability r = {d.measurement.reliability:g}. The "
            "raw treatment contrast is not attenuated in expectation; the "
            "error inflates residual variance (charged to the variance "
            "account, not the bias account).")
    if d.missingness is not None and d.missingness.rate > 0:
        m = d.missingness
        if m.mechanism == "differential":
            lines.append(
                "Attrition: logit Pr(dropout | Z, B) = alpha_Z + gamma_Z B, "
                "with intercepts calibrated to marginal rates control "
                f"{m.rate_control:g} / treated {m.rate_treated:g} and "
                f"baseline slopes gamma_0 = {m.baseline_slope_control:g}, "
                f"gamma_1 = {m.baseline_slope_treated:g}. Dropout depends "
                "only on observed quantities (Z, B): MAR given the baseline.")
        else:
            lines.append(f"Attrition: MCAR at marginal rates control "
                         f"{m.rate_control:g} / treated {m.rate_treated:g}.")
    extra = ""
    if a.estimator == "lmm_random_intercept":
        extra = f" (inference: {a.inference})"
    elif a.estimator == "mi_baseline_adjusted":
        extra = (f" (m = {a.m_imputations} imputations, Rubin's rules, "
                 "Barnard-Rubin df)")
    lines.append(
        f"Answer strategy: {a.estimator}{extra}, {a.formula}; two-sided "
        f"alpha {a.alpha:g}. Degenerate fits "
        f"{'COUNT' if a.degenerate_counts else 'do NOT count'} against the "
        "model-failure threshold (pre-specified); fatal errors and "
        "nonconvergence always count.")
    return lines


def report(result, file: str | None = None) -> str:
    d = result.design
    v = compute_verdict(result)
    thr = result.thresholds
    L: list[str] = []
    bar = "=" * 74

    L += [bar, f"PRE-DATA RECOVERY REPORT  (recoverlite-py {__version__})",
          bar, "", "1. TARGET",
          f"   Estimand: {d.target.estimand}",
          f"   Scale:    {d.target.scale}",
          f"   SESOI:    {d.target.sesoi:g}   Declared expected effect: "
          f"{d.effect:g}   Delta (bias/drift unit): "
          f"{d.target.bias_scale_unit:g}"]
    if d.target.max_width is not None:
        L.append(f"   Declared maximum acceptable CI width: "
                 f"{d.target.max_width:g}")
    L += ["", "2. DECLARATION"]
    L += ["   " + ln for ln in describe_design(d)]
    if d.omissions:
        L.append("   Not modeled (silence must not imply ideality):")
        L += ["     - " + om for om in d.omissions]
    L += ["", "3. SCENARIO GRID AND NULL-WORLD SPECIFICATION"]
    for run in result.runs.values():
        sc = run["scenario"]
        L += [f"   * {sc.label}", f"     {sc.rationale}"]
        if sc.row_type == "null":
            drift = run["diagnosands"]["rows"]["estimand_drift"]
            if math.isfinite(drift.mcse) and \
                    abs(drift.value) > 2 * drift.mcse:
                lab = (f"E[theta_obs] != 0 under this null world (drift "
                       f"{drift.value:.4f} [{drift.mcse:.4f}] Delta): "
                       "rejections are FALSE CLAIMS ABOUT THE TARGET, "
                       "partly induced by selection.")
            else:
                lab = ("E[theta_obs] = 0 under this null world: the "
                       "rejection rate is pure test size.")
            L.append("     Null world: theta = 0 with the declared "
                     "missingness mechanism persisting. " + lab)
    if result.evidence_tiers:
        L.append("   Evidence tiers of the pessimistic values (hierarchy: "
                 "empirical > prior-study > elicited > package default):")
        L += ["     - " + t for t in result.evidence_tiers]
    L += ["", "4. DIAGNOSANDS (value [MCSE]; n = contributing simulations)"]
    for run in result.runs.values():
        diag = run["diagnosands"]
        L += ["", f"   -- {run['scenario'].label}",
              f"      analyzed n per sim: mean "
              f"{diag['mean_n_analyzed']:.1f}; realized attrition: "
              f"{diag['mean_attrition_realized']:.3f}"]
        for r in diag["rows"].values():
            val = ("  not estimable " if not math.isfinite(r.value)
                   else f"{r.value:8.4f} [{r.mcse:.4f}]")
            L.append(f"      {r.name:<20s} {val}  n={r.n_contributing:<6d}"
                     + ("  UNSTABLE" if r.unstable else "")
                     + (f"  ({r.note})" if r.note else ""))
    L += ["", "5. THRESHOLD PROFILE AND SIGNED MARGINS",
          f"   Profile: '{thr.profile}' [{thr.version}]"
          + (f"  DEVIATIONS from shipped profile: "
             f"{', '.join(thr.modified)}" if thr.modified
             else "  (shipped values)"),
          "   Signed margin to every threshold (positive = passing):"]
    for nm, evs in v.evaluations.items():
        for c in evs:
            L.append(f"      {c.criterion:<20s} ({nm:<20s}) "
                     f"{c.requirement}  margin {c.margin:+.4f} "
                     f"[MCSE {c.mcse:.4f}]"
                     + ("  UNSTABLE" if c.unstable else "")
                     + (f"  ({c.note})" if c.note else ""))
    L += ["", f"6. VERDICT: {v.verdict}  (profile '{thr.profile}')"]
    if v.verdict_strict is not None:
        L.append(f"   Under shipped profiles: strict {v.verdict_strict} | "
                 f"default-family {v.verdict} | lenient {v.verdict_lenient}")
        if len({v.verdict_strict, v.verdict, v.verdict_lenient}) > 1:
            L.append("   Profile disagreement is itself a finding; the RISK "
                     "category exists to hold it.")
    if v.smallest_margin:
        L.append(f"   Smallest signed margin: {v.smallest_margin}")
    L.append(f"   Rule: PASS = all required rows pass with margins > "
             f"{thr.mcse_margin:g} MCSE; RISK = pessimistic-only failure, "
             "narrow margin, or unstable required diagnosand; FAIL = any "
             "failure under a declared-nuisance row.")
    L.append("   The verdict is a decision convention, not a validity "
             "classification.")
    L += ["", "7. BINDING FAILURE MODE",
          "   " + (v.binding or
                   "None: all criteria passed with stable margins.")]
    L += ["", "8. DESIGN CHANGE"]
    if v.verdict == "PASS":
        L.append("   No change required under the scenario rows this "
                 "profile requires.")
    else:
        L.append("   Rerun the recovery test on candidate repairs, each "
                 "simulated as a full declaration. Match the repair to the "
                 "failure: resources repair precision, not drift.")
    L += ["", "-" * 74,
          f"Computation: {result.sims} simulations per scenario row; "
          f"alpha = {result.alpha:g}; seed = "
          + (str(result.seed) if result.seed is not None
             else "not set (NOT reproducible)") + ";",
          f"elapsed {result.elapsed_secs:.1f} s; recoverlite-py "
          f"{__version__}. Protocol-identical Python mirror of the R "
          "implementation;",
          "results agree within Monte Carlo error, not byte-identically.",
          "A PASS is evidence about the instrument, not about the world.",
          bar]

    text = "\n".join(L)
    print(text)
    if file:
        with open(file, "w") as fh:
            fh.write(text + "\n")
    return text
