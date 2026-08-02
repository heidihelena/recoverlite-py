"""Estimator registry: plugins honor the record contract and run
through the full protocol without touching package internals."""

import math

import numpy as np
import pytest
from scipy import stats

from recoverlite import (attrition_model, declare_recovery,
                         planned_analysis, recovery_test,
                         register_estimator, registered_estimators,
                         target_estimand, two_arm_trial, verdict)


def _fit_welch_ttest(analysis, data, rng):
    """Custom estimator: Welch t test on retained rows, no covariates."""
    out = dict(estimate=float("nan"), se=float("nan"), ci_lo=float("nan"),
               ci_hi=float("nan"), p=float("nan"), fatal=False,
               nonconverged=False, degenerate=False, warned=False)
    try:
        mask = data["retained"]
        y = np.asarray(data["y_observed"], dtype=float)[mask]
        z = np.asarray(data["treatment"])[mask]
        y1, y0 = y[z == 1], y[z == 0]
        est = float(y1.mean() - y0.mean())
        v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
        se = math.sqrt(v1 + v0)
        dfree = (v1 + v0) ** 2 / (v1 ** 2 / (len(y1) - 1)
                                  + v0 ** 2 / (len(y0) - 1))
        tcrit = stats.t.ppf(1 - analysis.alpha / 2, dfree)
        out.update(estimate=est, se=se,
                   p=float(2 * stats.t.sf(abs(est / se), dfree)),
                   ci_lo=est - tcrit * se, ci_hi=est + tcrit * se)
    except Exception:
        out["fatal"] = True
    return out


def test_shipped_estimators_registered():
    names = registered_estimators()
    for nm in ("linear_model", "lmm_random_intercept",
               "cluster_mean_ttest", "mi_baseline_adjusted"):
        assert nm in names


def test_unknown_estimator_names_registered_ones():
    with pytest.raises(ValueError, match="registered estimators"):
        planned_analysis("no_such_estimator", "y_observed ~ treatment")


def test_duplicate_registration_needs_overwrite():
    with pytest.raises(ValueError, match="overwrite=True"):
        register_estimator("linear_model", _fit_welch_ttest)


def test_custom_estimator_end_to_end():
    register_estimator("welch_ttest", _fit_welch_ttest,
                       description="Welch t test on retained rows",
                       overwrite=True)
    d = declare_recovery(
        target=target_estimand("ATE", "mean difference", sesoi=0.4),
        data_strategy=two_arm_trial(120),
        missingness=attrition_model(0.15),
        answer_strategy=planned_analysis("welch_ttest",
                                         "y_observed ~ treatment"))
    res = recovery_test(d, sims=150, seed=9)
    diag = res.runs["target_declared"]["diagnosands"]["rows"]
    assert math.isfinite(diag["target_bias"].value)
    assert diag["model_failure"].value == 0.0
    v = verdict(res)
    assert v.verdict in ("PASS", "RISK", "FAIL")
    # reference calibration must run for plugins too, once merged;
    # tolerated absent here so the branch stands alone
    ref = getattr(res, "reference", None)
    if ref is not None:
        assert ref["status"] in ("attained", "not_attained")


def test_malformed_record_fails_loudly():
    register_estimator("broken", lambda analysis, data, rng: {"estimate": 1.0},
                       overwrite=True)
    d = declare_recovery(
        target=target_estimand("ATE", "mean difference", sesoi=0.4),
        data_strategy=two_arm_trial(20),
        answer_strategy=planned_analysis("broken", "y_observed ~ treatment"))
    with pytest.raises(TypeError, match="record contract"):
        recovery_test(d, sims=5, seed=1, scenarios="target_grid")
