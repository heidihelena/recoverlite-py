"""Scenario grid construction (protocol Step 3).

Crossed grid: null and target effects x declared and pessimistic
nuisance assumptions. The pessimistic rows perturb NUISANCE assumptions
only (attrition x1.5 capped, reliability -0.10, ICC to its declared
upper bound, noncompliance +50%); the target effect is NEVER shrunk
automatically. Target rows run at the SESOI; a declared effect above the
SESOI adds an informational expected-effect row outside the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constructors import ClusterTrial, RecoveryDesign, TwoArmTrial


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    params: dict
    rationale: str
    counts_for: str      # "declared" | "pessimistic" | "informational"
    row_type: str        # "null" | "target"


def pessimistic_overrides(design: RecoveryDesign) -> tuple[dict, str, list]:
    ds = design.data_strategy
    overrides: dict[str, Any] = {}
    rationale: list[str] = []
    tiers: list[str] = []

    if design.missingness is not None and design.missingness.rate > 0:
        a = design.missingness
        overrides["rate_control"] = min(a.rate_control * 1.5, a.max_rate)
        overrides["rate_treated"] = min(a.rate_treated * 1.5, a.max_rate)
        rationale.append(
            f"attrition rates (control {a.rate_control:g}, treated "
            f"{a.rate_treated:g}) -> ({overrides['rate_control']:g}, "
            f"{overrides['rate_treated']:g}) (x1.5, capped at {a.max_rate:g})")
        tiers.append("attrition perturbation: "
                     + (a.evidence or "package default (x1.5)"))
    if design.measurement is not None:
        r = design.measurement.reliability
        overrides["reliability"] = max(r - 0.10, 0.01)
        rationale.append(
            f"reliability {r:g} -> {overrides['reliability']:g} (-0.10, bounded)")
        tiers.append("reliability perturbation: "
                     + (design.measurement.evidence
                        or "package default (-0.10)"))
    if isinstance(ds, ClusterTrial):
        if ds.icc_pessimistic is not None:
            overrides["icc"] = ds.icc_pessimistic
            rationale.append(
                f"ICC {ds.icc:g} -> {overrides['icc']:g} (declared upper bound)")
            tiers.append("ICC upper bound: "
                         + (ds.evidence
                            or "researcher-declared (no source stated)"))
        else:
            overrides["icc"] = min(ds.icc + 0.10, 0.99)
            rationale.append(
                f"ICC {ds.icc:g} -> {overrides['icc']:g} "
                "(NO upper bound declared; icc + 0.10)")
            tiers.append("ICC upper bound: package default (icc + 0.10) - "
                         "declare `icc_pessimistic` from field evidence")
    if isinstance(ds, TwoArmTrial) and ds.noncompliance > 0:
        overrides["noncompliance"] = min(ds.noncompliance * 1.5, 0.95)
        rationale.append(
            f"noncompliance {ds.noncompliance:g} -> "
            f"{overrides['noncompliance']:g} (+50%, capped)")
        tiers.append("noncompliance perturbation: package default (+50%)")

    text = ("Nuisance perturbations only (target effect unchanged): "
            + "; ".join(rationale) + "." if rationale else
            "No nuisance assumptions were declared that admit a "
            "perturbation; pessimistic rows equal declared rows.")
    return overrides, text, tiers


def scenario_params(design: RecoveryDesign,
                    overrides: dict | None = None) -> dict:
    """Resolve design + overrides into one scenario row's parameters."""
    overrides = overrides or {}
    ds = design.data_strategy
    miss = design.missingness
    p: dict[str, Any] = {
        "kind": ds.kind,
        "effect": overrides.get("effect", design.effect),
        "reliability": overrides.get(
            "reliability",
            None if design.measurement is None
            else design.measurement.reliability),
        "allocation": ds.allocation,
    }
    if miss is not None and miss.rate > 0:
        p["attrition"] = {
            "mechanism": miss.mechanism,
            "rate_control": overrides.get("rate_control", miss.rate_control),
            "rate_treated": overrides.get("rate_treated", miss.rate_treated),
            "slope_control": miss.baseline_slope_control,
            "slope_treated": miss.baseline_slope_treated,
        }
    else:
        p["attrition"] = None
    if isinstance(ds, TwoArmTrial):
        p["n_per_arm"] = ds.n_per_arm
        p["rho"] = ds.baseline_outcome_cor
        p["noncompliance"] = overrides.get("noncompliance", ds.noncompliance)
    else:
        p["n_clusters"] = ds.n_clusters
        p["n_per_cluster"] = ds.n_per_cluster
        p["icc"] = overrides.get("icc", ds.icc)
    return p


def build_scenarios(design: RecoveryDesign,
                    scenarios: str = "confirmatory_grid"
                    ) -> tuple[dict[str, Scenario], list]:
    if scenarios not in ("confirmatory_grid", "target_grid"):
        raise ValueError(
            "`scenarios` must be 'confirmatory_grid' or 'target_grid'")
    theta_target = (1 if design.effect > 0 else -1) * design.target.sesoi
    overrides, pess_rationale, tiers = pessimistic_overrides(design)
    declared_rationale = "Nuisance assumptions exactly as declared."

    def make(name, label, theta, ov, rationale, counts_for, row_type):
        return Scenario(name=name, label=label,
                        params=scenario_params(design,
                                               {**ov, "effect": theta}),
                        rationale=rationale, counts_for=counts_for,
                        row_type=row_type)

    out: dict[str, Scenario] = {}
    if scenarios == "confirmatory_grid":
        out["null_declared"] = make(
            "null_declared", "Null-declared (theta = 0, nuisance declared)",
            0.0, {}, "Target-null rejection and calibration under the "
            "planned design. " + declared_rationale, "declared", "null")
        out["null_pessimistic"] = make(
            "null_pessimistic",
            "Null-pessimistic (theta = 0, nuisance pessimistic)",
            0.0, overrides,
            "Robustness of false-claim behavior. " + pess_rationale,
            "pessimistic", "null")
    out["target_declared"] = make(
        "target_declared",
        f"Target-declared (theta = {theta_target:g}, nuisance declared)",
        theta_target, {},
        "Power, bias, coverage, precision, drift at the SESOI. "
        + declared_rationale, "declared", "target")
    out["target_pessimistic"] = make(
        "target_pessimistic",
        f"Target-pessimistic (theta = {theta_target:g}, nuisance pessimistic)",
        theta_target, overrides,
        "Robustness of recovery at the SESOI. " + pess_rationale,
        "pessimistic", "target")
    if abs(design.effect) > design.target.sesoi:
        out["expected_effect"] = make(
            "expected_effect",
            f"Expected-effect (theta = {design.effect:g}, nuisance declared; "
            "informational)",
            design.effect, {},
            f"Secondary planning information: the declared expected effect "
            f"({design.effect:g}) exceeds the SESOI "
            f"({design.target.sesoi:g}). Verdict rows use the SESOI.",
            "informational", "target")
    return out, tiers
