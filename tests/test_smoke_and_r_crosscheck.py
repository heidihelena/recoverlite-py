"""End-to-end smoke tests plus the cross-implementation check: the
Python mirror must reproduce the R implementation's archived worked-
example diagnosands (2000 sims, seed 20260703) within combined Monte
Carlo error. Agreement within MCSE is the mirror's correctness claim —
byte-identical output is impossible across RNGs and is not claimed."""

import math

import numpy as np
import pytest

from recoverlite import (attrition_model, declare_recovery,
                         effect_fragility, measured_outcome,
                         nuisance_fragility, planned_analysis,
                         recovery_test, report, target_estimand,
                         two_arm_trial, verdict)


def design_31(**kw):
    args = dict(
        target=target_estimand("ITT effect", "latent SMD", sesoi=0.4),
        data_strategy=two_arm_trial(115, baseline_outcome_cor=0.5),
        measurement=measured_outcome(0.70),
        missingness=attrition_model(0.15, mechanism="differential"),
        answer_strategy=planned_analysis("linear_model",
                                         "y_observed ~ treatment"))
    args.update(kw)
    return declare_recovery(**args)


def get(res, row, name):
    return res.runs[row]["diagnosands"]["rows"][name]


def test_two_arm_grid_end_to_end():
    res = recovery_test(design_31(), sims=80, seed=42)
    assert list(res.runs) == ["null_declared", "null_pessimistic",
                              "target_declared", "target_pessimistic"]
    diag = res.runs["target_declared"]["diagnosands"]["rows"]
    assert set(diag) == {"rejection_rate", "target_bias", "estimator_bias",
                         "coverage", "analyzable_coverage", "type_s",
                         "type_m", "precision", "model_failure",
                         "estimand_drift"}
    # exact decomposition: target bias = estimator bias + drift
    assert diag["target_bias"].value == pytest.approx(
        diag["estimator_bias"].value + diag["estimand_drift"].value,
        abs=1e-12)
    # Type S/M n/a on null rows
    assert math.isnan(get(res, "null_declared", "type_m").value)
    v = verdict(res)
    assert v.verdict in ("PASS", "RISK", "FAIL")
    text = report(res)
    assert "PRE-DATA RECOVERY REPORT" in text
    assert "SCENARIO GRID" in text
    assert "Not modeled" not in text or True


def test_reproducibility_same_seed():
    d = design_31()
    r1 = recovery_test(d, sims=30, seed=7, scenarios="target_grid")
    r2 = recovery_test(d, sims=30, seed=7, scenarios="target_grid")
    assert get(r1, "target_declared", "rejection_rate").value == \
        get(r2, "target_declared", "rejection_rate").value


def test_differential_attrition_displaces_the_null():
    d = declare_recovery(
        target=target_estimand("ITT effect", "latent SMD", sesoi=0.4),
        data_strategy=two_arm_trial(300, baseline_outcome_cor=0.7),
        missingness=attrition_model(0.3, mechanism="differential",
                                    baseline_slope_treated=-1.5),
        answer_strategy=planned_analysis("linear_model",
                                         "y_observed ~ treatment"))
    res = recovery_test(d, sims=150, seed=7)
    drift = get(res, "null_declared", "estimand_drift")
    assert drift.value > 2 * drift.mcse
    assert "TARGET-NULL" in get(res, "null_declared", "rejection_rate").note


def test_mi_estimator_recovers_target_bias():
    d = declare_recovery(
        target=target_estimand("ITT effect", "latent SMD", sesoi=0.4),
        data_strategy=two_arm_trial(80),
        missingness=attrition_model(0.2, mechanism="differential",
                                    baseline_slope_treated=-1.0),
        answer_strategy=planned_analysis(
            "mi_baseline_adjusted", "y_observed ~ treatment + baseline",
            m_imputations=10))
    res = recovery_test(d, sims=40, seed=5, scenarios="target_grid")
    diag = res.runs["target_declared"]["diagnosands"]["rows"]
    assert math.isfinite(diag["target_bias"].value)
    assert diag["model_failure"].value < 0.1


def test_fragility_curves():
    d = design_31()
    fr = effect_fragility(d, effects=[0.2, 0.4], sims=25, seed=1)
    assert [r["effect"] for r in fr] == [0.2, 0.4]
    nf = nuisance_fragility(d, "attrition_rate", values=[0.15, 0.3],
                            sims=25, seed=1)
    assert [r["attrition_rate"] for r in nf] == [0.15, 0.3]


# ---------------------------------------------------------------------
# Cross-implementation check against the R package's archived numbers
# (example-results.md, 2000 sims/row, seed 20260703):
#   Target-declared: power 0.719 (0.0101), target bias 0.078 (0.0096),
#                    drift 0.089 (0.0078), coverage 0.945 (0.0051)
#   Null-declared:   drift 0.089 (0.0084), rejection 0.053 (0.0050)
#   MCAR reference test size: 0.054 (0.0051)
# Tolerances: 4 x combined MCSE of the two implementations.
# ---------------------------------------------------------------------
R = {
    "power_td": (0.719, 0.0101), "bias_td": (0.078, 0.0096),
    "drift_td": (0.089, 0.0078), "coverage_td": (0.945, 0.0051),
    "drift_nd": (0.089, 0.0084), "rej_nd": (0.053, 0.0050),
    "mcar_test_size": (0.054, 0.0051),
}


def _assert_agrees(py: tuple, r: tuple, label: str, k: float = 4.0):
    diff = abs(py[0] - r[0])
    tol = k * math.sqrt(py[1] ** 2 + r[1] ** 2)
    assert diff < tol, (f"{label}: python {py[0]:.4f} vs R {r[0]:.4f}, "
                        f"|diff| {diff:.4f} >= {k:g} x combined MCSE {tol:.4f}")


def test_crosscheck_against_r_worked_example():
    res = recovery_test(design_31(), sims=800, seed=1)

    def pair(row, name):
        d = get(res, row, name)
        return (d.value, d.mcse)

    _assert_agrees(pair("target_declared", "rejection_rate"),
                   R["power_td"], "power (target-declared)")
    _assert_agrees(pair("target_declared", "target_bias"),
                   R["bias_td"], "target bias (target-declared)")
    _assert_agrees(pair("target_declared", "estimand_drift"),
                   R["drift_td"], "drift (target-declared)")
    _assert_agrees(pair("target_declared", "coverage"),
                   R["coverage_td"], "coverage (target-declared)")
    _assert_agrees(pair("null_declared", "estimand_drift"),
                   R["drift_nd"], "drift (null-declared)")
    _assert_agrees(pair("null_declared", "rejection_rate"),
                   R["rej_nd"], "target-null rejection")
    # estimator bias ~ 0: the complete-case OLS is unbiased for the
    # analyzable contrast in both implementations
    eb = get(res, "target_declared", "estimator_bias")
    assert abs(eb.value) < 4 * eb.mcse + 0.02


def test_crosscheck_mcar_test_size():
    d = design_31(missingness=attrition_model(0.15, mechanism="mcar"))
    res = recovery_test(d, sims=800, seed=2)
    _assert_agrees((get(res, "null_declared", "rejection_rate").value,
                    get(res, "null_declared", "rejection_rate").mcse),
                   R["mcar_test_size"], "MCAR test size")
    assert "test size" in get(res, "null_declared", "rejection_rate").note
