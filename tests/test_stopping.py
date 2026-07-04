"""Tests for the algorithmic doubling stopping rule (recovery_test_stable),
mirroring the R package's stopping-rule behaviour."""

from recoverlite import (attrition_model, declare_recovery, measured_outcome,
                        planned_analysis, recovery_test_stable,
                        target_estimand, two_arm_trial, verdict)


def design_31():
    return declare_recovery(
        target=target_estimand("ITT effect", "latent SMD", sesoi=0.4),
        data_strategy=two_arm_trial(115, baseline_outcome_cor=0.5),
        measurement=measured_outcome(0.70),
        missingness=attrition_model(0.15, mechanism="differential"),
        answer_strategy=planned_analysis("linear_model",
                                         "y_observed ~ treatment"))


def test_stable_declared_failure_locks_fail_at_start():
    # Example 3.1 is a stable FAIL (bias + power), so the rule must not
    # double past the start count.
    res, rec = recovery_test_stable(design_31(), start_sims=400,
                                    max_sims=1600, seed=1)
    assert verdict(res).verdict == "FAIL"
    assert rec.final_sims == 400
    assert rec.resolved is True
    assert rec.hit_ceiling is False


def test_record_reports_doubling_trajectory_fields():
    res, rec = recovery_test_stable(design_31(), start_sims=200,
                                    max_sims=800, seed=2)
    assert rec.start_sims == 200
    assert rec.max_sims == 800
    assert rec.final_sims in (200, 400, 800)
    assert isinstance(rec.unresolved, list)
