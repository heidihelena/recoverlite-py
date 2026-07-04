import pytest

from recoverlite import (attrition_model, cluster_trial, declare_recovery,
                         measured_outcome, planned_analysis, target_estimand,
                         two_arm_trial)


def test_target_estimand_defaults_bias_unit():
    t = target_estimand("ITT effect", "latent SMD", sesoi=0.4)
    assert t.bias_scale_unit == 0.4
    with pytest.raises(ValueError):
        target_estimand("x", "SMD", sesoi=0)


def test_data_strategies_validate():
    ta = two_arm_trial(100)
    assert ta.baseline_outcome_cor == 0.5
    with pytest.raises(ValueError):
        two_arm_trial(1)
    with pytest.raises(ValueError):
        two_arm_trial(100, allocation=1)
    ct = cluster_trial(16, 30, icc=0.05)
    assert ct.kind == "cluster_trial"
    with pytest.raises(ValueError):
        cluster_trial(16, 30, icc=0.05, icc_pessimistic=0.01)


def test_attrition_model_mcar_zeroes_slopes():
    a = attrition_model(0.15)
    assert a.mechanism == "differential"
    assert a.rate_control == 0.15 and a.rate_treated == 0.15
    m = attrition_model(0.15, mechanism="mcar", baseline_slope_treated=-1)
    assert m.baseline_slope_treated == 0.0


def test_planned_analysis_formula_parse_and_inference_guard():
    pa = planned_analysis("lmm_random_intercept",
                          "y_observed ~ treatment + (1 | cluster)")
    assert pa.terms["random_intercept"] == "cluster"
    assert pa.terms["fixed"] == ["treatment"]
    # Satterthwaite / KR are R-only; the error must say so
    with pytest.raises(ValueError, match="R implementation"):
        planned_analysis("lmm_random_intercept",
                         "y_observed ~ treatment + (1 | cluster)",
                         inference="satterthwaite")


def test_declare_recovery_records_omissions():
    d = declare_recovery(
        target=target_estimand("ITT effect", "latent SMD", sesoi=0.4),
        data_strategy=two_arm_trial(100),
        answer_strategy=planned_analysis("linear_model",
                                         "y_observed ~ treatment"))
    assert d.effect == 0.4
    assert any("perfectly reliable" in om for om in d.omissions)
    assert any("Attrition was not modeled" in om for om in d.omissions)


def test_cluster_estimators_require_cluster_strategy():
    with pytest.raises(ValueError, match="cluster_trial"):
        declare_recovery(
            target=target_estimand("ITT effect", "SMD", sesoi=0.4),
            data_strategy=two_arm_trial(100),
            answer_strategy=planned_analysis("cluster_mean_ttest",
                                             "y_observed ~ treatment"))
