"""Reference-analysis calibration check: the planned analysis must
attain its nominal properties on ideal data before scenario rows are
scored against it; a stable deviation caps the verdict at RISK."""

import numpy as np

from recoverlite import (cluster_trial, declare_recovery, planned_analysis,
                         recovery_test, recovery_thresholds,
                         reference_calibration, report, target_estimand,
                         two_arm_trial, verdict)


def _congenial_design(n_per_arm=200):
    return declare_recovery(
        target=target_estimand("ATE", "mean difference", sesoi=0.3),
        data_strategy=two_arm_trial(n_per_arm=n_per_arm),
        answer_strategy=planned_analysis(
            "linear_model", "y_observed ~ treatment + baseline"))


def _misspecified_design():
    # Independent-observations OLS on cluster-randomized data: SEs are
    # wrong even on ideal data, so reference coverage cannot attain
    # nominal. This is exactly the mismatch the check exists to catch.
    return declare_recovery(
        target=target_estimand("ATE", "mean difference", sesoi=0.3),
        data_strategy=cluster_trial(n_clusters=20, n_per_cluster=50,
                                    icc=0.3),
        answer_strategy=planned_analysis(
            "linear_model", "y_observed ~ treatment"))


def test_congenial_reference_attained():
    d = _congenial_design()
    thr = recovery_thresholds()
    rng = np.random.default_rng(7)
    ref = reference_calibration(d, sims=800, rng=rng, thr=thr)
    assert ref["status"] == "attained"
    assert ref["n_ok"] == 800
    assert abs(ref["coverage"] - 0.95) <= 3 * ref["coverage_mcse"] + 1e-12
    assert ref["deviations"] == ()


def test_misspecified_reference_not_attained():
    d = _misspecified_design()
    thr = recovery_thresholds()
    rng = np.random.default_rng(7)
    ref = reference_calibration(d, sims=400, rng=rng, thr=thr)
    assert ref["status"] == "not_attained"
    assert any("coverage" in dv for dv in ref["deviations"])
    # OLS point estimate stays unbiased on ideal cluster data.
    assert ref["coverage"] < 0.9


def test_recovery_test_attaches_reference():
    res = recovery_test(_congenial_design(n_per_arm=60), sims=200, seed=11)
    assert res.reference is not None
    assert res.reference["status"] in ("attained", "not_attained")
    assert "Reference-analysis calibration" in str(res)


def test_reference_deviation_caps_pass_at_risk():
    # A design whose criteria would pass, with a forced not_attained
    # reference: the verdict must be capped at RISK with the calibration
    # message binding.
    res = recovery_test(_congenial_design(), sims=600, seed=3)
    v0 = verdict(res)
    res.reference = dict(status="not_attained", n_ok=600, sims=600,
                         nominal=0.95, coverage=0.80, coverage_mcse=0.01,
                         bias=0.0, bias_mcse=0.01,
                         deviations=("coverage 0.8000 deviates from nominal "
                                     "0.95 by more than 2 MCSE (0.0100)",),
                         note="forced for test")
    v1 = verdict(res)
    assert v1.verdict != "PASS"
    if v0.verdict == "PASS":
        assert v1.verdict == "RISK"
    assert "DGP-versus-analysis mismatch" in v1.binding


def test_report_includes_calibration_section(capsys):
    res = recovery_test(_congenial_design(n_per_arm=60), sims=200, seed=11)
    text = report(res)
    capsys.readouterr()
    assert "5. REFERENCE-ANALYSIS CALIBRATION" in text
    assert "6. THRESHOLD PROFILE" in text
