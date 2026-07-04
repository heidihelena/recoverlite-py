"""recoverlite: pre-data recovery tests for planned study designs.

Protocol-identical Python mirror of the R package
(https://github.com/heidihelena/recoverlite): same declaration API,
crossed scenario grid, diagnosands, versioned threshold profiles, and
PASS/RISK/FAIL verdict rule, including the algorithmic doubling stopping
rule (`recovery_test_stable`). Satterthwaite and Kenward-Roger
mixed-model inference are implemented natively (`lmm.py`) and validated
against lmerTest/pbkrtest to numerical precision. Simulated numbers agree
with the R package within Monte Carlo error, not byte-identically.
"""

__version__ = "0.2.0"

from .constructors import (
    AttritionModel,
    ClusterTrial,
    MeasuredOutcome,
    PlannedAnalysis,
    RecoveryDesign,
    TargetEstimand,
    TwoArmTrial,
    attrition_model,
    cluster_trial,
    declare_recovery,
    measured_outcome,
    planned_analysis,
    target_estimand,
    two_arm_trial,
)
from .fragility import effect_fragility, nuisance_fragility
from .recovery_test import RecoveryResult, recovery_test
from .report import report
from .stopping import StoppingRecord, recovery_test_stable
from .thresholds import THRESHOLD_SET_VERSION, Thresholds, recovery_thresholds
from .verdict import Verdict, verdict

__all__ = [
    "THRESHOLD_SET_VERSION", "AttritionModel", "ClusterTrial",
    "MeasuredOutcome", "PlannedAnalysis", "RecoveryDesign", "RecoveryResult",
    "StoppingRecord", "TargetEstimand", "Thresholds", "TwoArmTrial", "Verdict",
    "attrition_model", "cluster_trial", "declare_recovery",
    "effect_fragility", "measured_outcome", "nuisance_fragility",
    "planned_analysis", "recovery_test", "recovery_test_stable",
    "recovery_thresholds", "report", "target_estimand", "two_arm_trial",
    "verdict",
]
