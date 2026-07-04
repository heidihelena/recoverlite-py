"""Verdict-rule tests on synthetic diagnosand tables, independent of the
simulation engine — mirrors the R package's test-verdict.R."""

import math

import pytest

from recoverlite import recovery_thresholds, verdict
from recoverlite.diagnosands import Diagnosand
from recoverlite.scenarios import Scenario


def make_diag(row_type="target", rejection=None, rej_mcse=0.005,
              bias=0.0, bias_mcse=0.002, drift=0.0, drift_mcse=0.001,
              coverage=0.955, coverage_mcse=0.005, type_s=0.0, type_m=1.05,
              n_sig=1800, cond_mcse=0.002, model_failure=0.0,
              mf_mcse=0.0005, unstable_cond=False, S=2000):
    if rejection is None:
        rejection = 0.05 if row_type == "null" else 0.90
    is_target = row_type == "target"
    nan = float("nan")
    rows = [
        Diagnosand("rejection_rate", rejection, rej_mcse, S),
        Diagnosand("target_bias", bias, bias_mcse, S),
        Diagnosand("estimator_bias", bias - drift, bias_mcse, S),
        Diagnosand("coverage", coverage, coverage_mcse, S),
        Diagnosand("analyzable_coverage", 0.95, 0.005, S),
        Diagnosand("type_s", type_s if is_target else nan,
                   cond_mcse if is_target else nan,
                   n_sig if is_target else 0,
                   unstable=unstable_cond if is_target else False),
        Diagnosand("type_m", type_m if is_target else nan,
                   cond_mcse if is_target else nan,
                   n_sig if is_target else 0,
                   unstable=unstable_cond if is_target else False),
        Diagnosand("precision", 0.5, 0.01, S),
        Diagnosand("model_failure", model_failure, mf_mcse, S),
        Diagnosand("estimand_drift", drift, drift_mcse, S),
    ]
    return {"rows": {r.name: r for r in rows}, "n_sims": S, "n_ok": S,
            "n_sig": n_sig, "mean_n_analyzed": 200.0,
            "mean_attrition_realized": 0.0,
            "failure_classes": dict(fatal=0, nonconverged=0, degenerate=0,
                                    warned=0)}


class FakeResult:
    def __init__(self, nd=None, np_=None, td=None, tp=None,
                 thresholds=None, drop=()):
        def run(name, row_type, counts_for, diag):
            sc = Scenario(name=name, label=name, params={"effect": 0.4},
                          rationale="", counts_for=counts_for,
                          row_type=row_type)
            return {"scenario": sc, "diagnosands": diag,
                    "theta": 0 if row_type == "null" else 0.4}

        runs = {
            "null_declared": run("null_declared", "null", "declared",
                                 nd or make_diag("null")),
            "null_pessimistic": run("null_pessimistic", "null",
                                    "pessimistic", np_ or make_diag("null")),
            "target_declared": run("target_declared", "target", "declared",
                                   td or make_diag("target")),
            "target_pessimistic": run("target_pessimistic", "target",
                                      "pessimistic",
                                      tp or make_diag("target")),
        }
        self.runs = {k: v for k, v in runs.items() if k not in drop}
        self.thresholds = thresholds or recovery_thresholds()
        self.alpha = 0.05
        self.sims = 2000


def test_all_pass_wide_margins_is_pass():
    v = verdict(FakeResult())
    assert v.verdict == "PASS"
    assert v.binding is None
    assert v.verdict_lenient == "PASS"


def test_inflated_target_null_rejection_fails():
    v = verdict(FakeResult(nd=make_diag("null", rejection=0.09)))
    assert v.verdict == "FAIL"
    assert "target_null_rejection" in v.binding


def test_declared_row_failures_fail():
    assert verdict(FakeResult(
        td=make_diag("target", rejection=0.55))).verdict == "FAIL"
    assert verdict(FakeResult(
        td=make_diag("target", bias=-0.17))).verdict == "FAIL"
    assert verdict(FakeResult(
        td=make_diag("target", coverage=0.90))).verdict == "FAIL"
    assert verdict(FakeResult(
        td=make_diag("target", model_failure=0.08))).verdict == "FAIL"


def test_pessimistic_only_failure_is_risk():
    v = verdict(FakeResult(tp=make_diag("target", rejection=0.70)))
    assert v.verdict == "RISK"
    assert "pessimistic" in v.binding


def test_margin_within_two_mcse_is_risk():
    v = verdict(FakeResult(td=make_diag("target", rejection=0.805,
                                        rej_mcse=0.009)))
    assert v.verdict == "RISK"
    assert "MCSE" in v.binding


def test_overcoverage_flagged_not_failed():
    v = verdict(FakeResult(td=make_diag("target", coverage=0.985)))
    assert v.verdict != "FAIL"
    ev = {c.criterion: c for c in v.evaluations["target_declared"]}
    assert ev["coverage"].passed is True
    assert "inefficiency" in ev["coverage"].note


def test_unstable_conditionals_block_pass_not_fail():
    v = verdict(FakeResult(td=make_diag("target", type_m=2.4, n_sig=20,
                                        unstable_cond=True)))
    assert v.verdict == "RISK"
    assert "unstable" in v.binding


def test_missing_required_rows_cap_at_risk():
    v = verdict(FakeResult(drop=("null_declared", "null_pessimistic")))
    assert v.verdict == "RISK"
    assert "Required scenario rows" in v.binding


def test_strict_and_lenient_can_disagree():
    v = verdict(FakeResult(td=make_diag("target", rejection=0.75),
                           tp=make_diag("target", rejection=0.75)))
    assert v.verdict == "FAIL"
    assert v.verdict_lenient == "PASS"
    assert v.verdict_strict == "FAIL"


def test_estimation_profile_uses_target_rows_and_drift():
    thr = recovery_thresholds("estimation")
    v = verdict(FakeResult(td=make_diag("target", drift=0.08),
                           thresholds=thr,
                           drop=("null_declared", "null_pessimistic")))
    assert v.verdict == "FAIL"
    assert "estimand_drift" in v.binding
    crits = [c.criterion for c in v.evaluations["target_declared"]]
    assert "power" not in crits
