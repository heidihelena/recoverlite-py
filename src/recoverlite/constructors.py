"""Declaration objects (protocol Steps 1-2).

Python mirror of the R package's constructors. The declaration API,
scenario grid, diagnosands, threshold profiles, and verdict rule are
protocol-identical to the R implementation; simulated numbers agree
within Monte Carlo error, not byte-identically (different RNG streams).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetEstimand:
    """The target estimand: quantity, scale, and SESOI (Step 1)."""

    estimand: str
    scale: str
    sesoi: float
    bias_scale_unit: float | None = None
    max_width: float | None = None

    def __post_init__(self):
        if not self.estimand or not isinstance(self.estimand, str):
            raise ValueError("`estimand` must be a non-empty string")
        if not (self.sesoi > 0):
            raise ValueError("`sesoi` must be strictly positive")
        if self.bias_scale_unit is None:
            object.__setattr__(self, "bias_scale_unit", self.sesoi)
        if not (self.bias_scale_unit > 0):
            raise ValueError("`bias_scale_unit` must be strictly positive")
        if self.max_width is not None and not (self.max_width > 0):
            raise ValueError("`max_width` must be strictly positive")


def target_estimand(estimand: str, scale: str, sesoi: float,
                    bias_scale_unit: float | None = None,
                    max_width: float | None = None) -> TargetEstimand:
    return TargetEstimand(estimand, scale, sesoi, bias_scale_unit, max_width)


@dataclass(frozen=True)
class TwoArmTrial:
    """Two-arm randomized trial with an observed baseline.

    B ~ N(0,1); Y* = tau Z + rho B + sqrt(1 - rho^2) eps, so the true
    outcome has unit SD given Z. `n_per_arm` is RECRUITED per arm.
    """

    n_per_arm: int
    allocation: float = 0.5
    baseline_outcome_cor: float = 0.5
    noncompliance: float = 0.0
    kind: str = field(default="two_arm_trial", init=False)

    def __post_init__(self):
        if self.n_per_arm < 2 or self.n_per_arm != int(self.n_per_arm):
            raise ValueError("`n_per_arm` must be a whole number >= 2")
        if not (0 < self.allocation < 1):
            raise ValueError("`allocation` must be strictly between 0 and 1")
        if not (0 <= self.baseline_outcome_cor < 1):
            raise ValueError("`baseline_outcome_cor` must be in [0, 1)")
        if not (0 <= self.noncompliance < 1):
            raise ValueError("`noncompliance` must be in [0, 1)")


def two_arm_trial(n_per_arm: int, allocation: float = 0.5,
                  baseline_outcome_cor: float = 0.5,
                  noncompliance: float = 0.0) -> TwoArmTrial:
    return TwoArmTrial(int(n_per_arm), allocation, baseline_outcome_cor,
                       noncompliance)


@dataclass(frozen=True)
class ClusterTrial:
    """Cluster-randomized trial; sizes fixed by design; unit total variance."""

    n_clusters: int
    n_per_cluster: int
    icc: float
    icc_pessimistic: float | None = None
    allocation: float = 0.5
    evidence: str | None = None
    kind: str = field(default="cluster_trial", init=False)

    def __post_init__(self):
        if self.n_clusters < 4:
            raise ValueError("`n_clusters` must be >= 4")
        if self.n_per_cluster < 1:
            raise ValueError("`n_per_cluster` must be >= 1")
        if not (0 <= self.icc < 1):
            raise ValueError("`icc` must be in [0, 1)")
        if self.icc_pessimistic is not None and not (
                self.icc <= self.icc_pessimistic < 1):
            raise ValueError("`icc_pessimistic` must be in [icc, 1)")
        if not (0 < self.allocation < 1):
            raise ValueError("`allocation` must be strictly between 0 and 1")


def cluster_trial(n_clusters: int, n_per_cluster: int, icc: float,
                  icc_pessimistic: float | None = None,
                  allocation: float = 0.5,
                  evidence: str | None = None) -> ClusterTrial:
    return ClusterTrial(int(n_clusters), int(n_per_cluster), icc,
                        icc_pessimistic, allocation, evidence)


@dataclass(frozen=True)
class MeasuredOutcome:
    """Classical ADDITIVE measurement error: y = y* + e, Var(e) = 1/r - 1.

    The raw treatment contrast is not attenuated in expectation; the
    error inflates residual variance (a precision/power failure, not a
    bias failure).
    """

    reliability: float
    evidence: str | None = None

    def __post_init__(self):
        if not (0 < self.reliability <= 1):
            raise ValueError("`reliability` must be in (0, 1]")


def measured_outcome(reliability: float,
                     evidence: str | None = None) -> MeasuredOutcome:
    return MeasuredOutcome(reliability, evidence)


@dataclass(frozen=True)
class AttritionModel:
    """Dropout model: logit Pr(dropout | Z, B) = alpha_Z + gamma_Z B.

    MAR given the observed baseline under "differential"; intercepts are
    calibrated so each arm's marginal rate equals its declared value.
    "mcar" drops completely at random (slopes forced to zero).
    """

    rate: float
    mechanism: str = "differential"
    rate_control: float | None = None
    rate_treated: float | None = None
    baseline_slope_treated: float = -0.5
    baseline_slope_control: float = 0.0
    max_rate: float = 0.6
    evidence: str | None = None

    def __post_init__(self):
        if self.mechanism not in ("differential", "mcar"):
            raise ValueError("`mechanism` must be 'differential' or 'mcar'")
        if not (0 <= self.rate < 1):
            raise ValueError("`rate` must be in [0, 1)")
        if not (0 < self.max_rate < 1):
            raise ValueError("`max_rate` must be in (0, 1)")
        if self.rate_control is None:
            object.__setattr__(self, "rate_control", self.rate)
        if self.rate_treated is None:
            object.__setattr__(self, "rate_treated", self.rate)
        for r in (self.rate_control, self.rate_treated):
            if not (0 <= r < 1):
                raise ValueError("arm-specific rates must be in [0, 1)")
        if self.mechanism == "mcar":
            object.__setattr__(self, "baseline_slope_treated", 0.0)
            object.__setattr__(self, "baseline_slope_control", 0.0)


def attrition_model(rate: float, mechanism: str = "differential",
                    rate_control: float | None = None,
                    rate_treated: float | None = None,
                    baseline_slope_treated: float = -0.5,
                    baseline_slope_control: float = 0.0,
                    max_rate: float = 0.6,
                    evidence: str | None = None) -> AttritionModel:
    return AttritionModel(rate, mechanism, rate_control, rate_treated,
                          baseline_slope_treated, baseline_slope_control,
                          max_rate, evidence)


_ESTIMATORS = ("linear_model", "lmm_random_intercept", "cluster_mean_ttest",
               "mi_baseline_adjusted")
_INFERENCE = ("satterthwaite", "kenward_roger", "wald_z")


@dataclass(frozen=True)
class PlannedAnalysis:
    """The answer strategy, stated exactly.

    `formula` uses the R-style syntax of the paper's API sketch, e.g.
    "y_observed ~ treatment", "y_observed ~ treatment + baseline",
    "y_observed ~ treatment + (1 | cluster)". Available columns:
    y_observed, treatment, baseline, cluster.

    For `lmm_random_intercept`, `inference` selects "satterthwaite"
    (default, matching the R implementation's default), "kenward_roger",
    or "wald_z". Satterthwaite uses the observed REML Hessian (as
    lmerTest does); Kenward-Roger uses the expected REML information and
    the KR-adjusted covariance (as pbkrtest does). Both are validated
    against R to numerical precision on shared datasets
    (tests/data/lmm_reference.json).
    """

    estimator: str
    formula: str
    alpha: float = 0.05
    inference: str = "satterthwaite"
    m_imputations: int = 20
    degenerate_counts: bool = False

    def __post_init__(self):
        if self.estimator not in _ESTIMATORS:
            raise ValueError(f"`estimator` must be one of {_ESTIMATORS}")
        if not (0 < self.alpha < 1):
            raise ValueError("`alpha` must be in (0, 1)")
        if self.estimator == "lmm_random_intercept" and \
                self.inference not in _INFERENCE:
            raise ValueError(f"`inference` must be one of {_INFERENCE}")
        if self.m_imputations < 2:
            raise ValueError("`m_imputations` must be >= 2")
        # parse the formula once; raises on malformed input
        object.__setattr__(self, "_terms", parse_formula(self.formula))

    @property
    def terms(self) -> dict:
        return self._terms


def parse_formula(formula: str) -> dict:
    """Parse the small R-style formula subset used by the protocol."""
    if "~" not in formula:
        raise ValueError("`formula` must contain '~'")
    lhs, rhs = (side.strip() for side in formula.split("~", 1))
    fixed: list[str] = []
    random_intercept = None
    for term in (t.strip() for t in rhs.split("+")):
        m = re.fullmatch(r"\(\s*1\s*\|\s*(\w+)\s*\)", term)
        if m:
            random_intercept = m.group(1)
        elif term:
            fixed.append(term)
    return {"outcome": lhs, "fixed": fixed,
            "random_intercept": random_intercept}


def planned_analysis(estimator: str, formula: str, alpha: float = 0.05,
                     inference: str = "satterthwaite",
                     m_imputations: int = 20,
                     degenerate_counts: bool = False) -> PlannedAnalysis:
    return PlannedAnalysis(estimator, formula, alpha, inference,
                           int(m_imputations), degenerate_counts)


@dataclass(frozen=True)
class RecoveryDesign:
    """The assembled declaration (Step 2)."""

    target: TargetEstimand
    data_strategy: TwoArmTrial | ClusterTrial
    answer_strategy: PlannedAnalysis
    measurement: MeasuredOutcome | None = None
    missingness: AttritionModel | None = None
    effect: float | None = None
    omissions: tuple[str, ...] = ()


def declare_recovery(target: TargetEstimand,
                     data_strategy: TwoArmTrial | ClusterTrial,
                     answer_strategy: PlannedAnalysis,
                     measurement: MeasuredOutcome | None = None,
                     missingness: AttritionModel | None = None,
                     effect: float | None = None) -> RecoveryDesign:
    """Assemble the declaration; silence is recorded, not idealized."""
    if effect is None:
        effect = target.sesoi
    if effect == 0:
        raise ValueError("`effect` must be non-zero")

    is_cluster = isinstance(data_strategy, ClusterTrial)
    est = answer_strategy.estimator
    if not is_cluster and est in ("lmm_random_intercept",
                                  "cluster_mean_ttest"):
        raise ValueError(
            "Cluster-based estimators require a cluster_trial() data strategy")

    omissions = []
    if measurement is None:
        omissions.append("Measurement reliability was not declared; outcomes "
                         "are treated as perfectly reliable.")
    if missingness is None or missingness.rate == 0:
        omissions.append("Attrition was not modeled; all recruited "
                         "observations are treated as analyzed.")
    if not is_cluster:
        omissions.append("Observations are treated as independent "
                         "(no clustering declared).")
    if is_cluster and est in ("linear_model", "mi_baseline_adjusted"):
        omissions.append("Cluster-randomized data strategy with an "
                         "independent-observations estimator: observations "
                         "are treated as independent by the analysis.")

    return RecoveryDesign(target=target, data_strategy=data_strategy,
                          answer_strategy=answer_strategy,
                          measurement=measurement, missingness=missingness,
                          effect=effect, omissions=tuple(omissions))
