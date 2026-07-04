import pytest

from recoverlite import (attrition_model, cluster_trial, declare_recovery,
                         measured_outcome, planned_analysis,
                         recovery_thresholds, target_estimand, two_arm_trial)
from recoverlite.scenarios import build_scenarios
from recoverlite.simulate import calibrate_dropout_intercept
from recoverlite.thresholds import THRESHOLD_SET_VERSION


def test_shipped_profiles_match_protocol_exactly():
    d = recovery_thresholds("default")
    assert (d.null_rejection_mult, d.power, d.target_bias, d.coverage,
            d.type_s, d.type_m, d.model_failure) == \
        (1.25, 0.80, 0.05, 0.925, 0.01, 1.50, 0.01)
    s = recovery_thresholds("strict")
    assert (s.null_rejection_mult, s.power, s.target_bias, s.coverage,
            s.type_s, s.type_m, s.model_failure) == \
        (1.10, 0.90, 0.025, 0.940, 0.005, 1.25, 0.005)
    l = recovery_thresholds("lenient")
    assert (l.null_rejection_mult, l.power, l.target_bias, l.coverage,
            l.type_s, l.type_m, l.model_failure) == \
        (1.50, 0.70, 0.10, 0.900, 0.05, 2.00, 0.05)
    assert d.min_conditional_n == 200
    assert d.mcse_margin == 2
    # the version string is the shared protocol artifact with R
    assert THRESHOLD_SET_VERSION == "recoverlite-thresholds-0.2"


def test_threshold_deviations_recorded_and_drift_tracks_bias():
    thr = recovery_thresholds("default", power=0.90, type_m=2.0)
    assert set(thr.modified) == {"power", "type_m"}
    e = recovery_thresholds("estimation", target_bias=0.08)
    assert e.drift == 0.08
    e2 = recovery_thresholds("estimation", target_bias=0.08, drift=0.03)
    assert e2.drift == 0.03


def _design_31():
    return declare_recovery(
        target=target_estimand("ITT effect", "latent SMD", sesoi=0.4),
        data_strategy=two_arm_trial(115),
        measurement=measured_outcome(0.70),
        missingness=attrition_model(0.15, mechanism="differential"),
        answer_strategy=planned_analysis("linear_model",
                                         "y_observed ~ treatment"))


def test_confirmatory_grid_crosses_effect_and_nuisance():
    scs, _ = build_scenarios(_design_31())
    assert list(scs) == ["null_declared", "null_pessimistic",
                         "target_declared", "target_pessimistic"]
    assert scs["null_declared"].params["effect"] == 0
    assert scs["target_declared"].params["effect"] == 0.4
    # null world keeps the declared missingness mechanism
    assert scs["null_declared"].params["attrition"]["slope_treated"] == -0.5
    p = scs["target_pessimistic"].params
    assert p["attrition"]["rate_treated"] == pytest.approx(0.225)
    assert p["reliability"] == pytest.approx(0.60)
    assert p["effect"] == 0.4  # the effect is NEVER shrunk


def test_expected_effect_row_is_informational():
    d = declare_recovery(
        target=target_estimand("ITT effect", "SMD", sesoi=0.4),
        data_strategy=two_arm_trial(100),
        answer_strategy=planned_analysis("linear_model",
                                         "y_observed ~ treatment"),
        effect=0.6)
    scs, _ = build_scenarios(d)
    assert scs["target_declared"].params["effect"] == 0.4  # SESOI
    assert scs["expected_effect"].counts_for == "informational"
    assert scs["expected_effect"].params["effect"] == 0.6


def test_icc_moves_to_declared_upper_bound_with_tier():
    d = declare_recovery(
        target=target_estimand("ITT effect", "SMD", sesoi=0.4),
        data_strategy=cluster_trial(16, 30, icc=0.05, icc_pessimistic=0.15,
                                    evidence="published ICC range"),
        answer_strategy=planned_analysis(
            "cluster_mean_ttest", "y_observed ~ treatment"))
    scs, tiers = build_scenarios(d)
    assert scs["target_pessimistic"].params["icc"] == 0.15
    assert any("published ICC range" in t for t in tiers)


def test_dropout_intercept_calibration():
    import numpy as np
    from scipy import integrate, stats

    a = calibrate_dropout_intercept(0.15, -0.5)
    marg, _ = integrate.quad(
        lambda b: 1 / (1 + np.exp(-(a - 0.5 * b))) * stats.norm.pdf(b),
        -np.inf, np.inf)
    assert marg == pytest.approx(0.15, abs=1e-4)
    # zero rate stays finite so arm interactions cannot produce NaN
    a0 = calibrate_dropout_intercept(0.0, 0.0)
    assert np.isfinite(a0)
    a1 = calibrate_dropout_intercept(0.3, -0.5)
    assert np.isfinite(a0 + (a1 - a0) * 1)
