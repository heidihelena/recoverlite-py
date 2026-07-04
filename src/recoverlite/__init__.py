"""recoverlite: pre-data recovery tests for planned study designs.

Protocol-identical Python mirror of the R package
(https://github.com/heidihelena/recoverlite): same declaration API,
crossed scenario grid, diagnosands, versioned threshold profiles, and
PASS/RISK/FAIL verdict rule. Simulated numbers agree within Monte Carlo
error, not byte-identically. Satterthwaite and Kenward-Roger mixed-model
inference are available only in the R implementation.
"""

__version__ = "0.0.1.dev1"

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
from .thresholds import THRESHOLD_SET_VERSION, Thresholds, recovery_thresholds
from .verdict import Verdict, verdict

__all__ = [
    "THRESHOLD_SET_VERSION", "AttritionModel", "ClusterTrial",
    "MeasuredOutcome", "PlannedAnalysis", "RecoveryDesign", "RecoveryResult",
    "TargetEstimand", "Thresholds", "TwoArmTrial", "Verdict",
    "attrition_model", "cluster_trial", "declare_recovery",
    "effect_fragility", "measured_outcome", "nuisance_fragility",
    "planned_analysis", "recovery_test", "recovery_thresholds", "report",
    "target_estimand", "two_arm_trial", "verdict",
]
