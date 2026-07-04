"""Diagnosands (protocol section 2.3) with MCSEs and inclusion rules.

Inclusion rules, fixed in advance: simulations with COUNTED model
failures contribute to the model-failure rate and are excluded from
estimate-based diagnosands (which are therefore conditional on
successful analysis). Degenerate fits that do not count are retained,
and their marginal effect on coverage is reported. Conditional
diagnosands (Type S, Type M) use bootstrap MCSEs and are marked unstable
below the pre-specified minimum contributing count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._utils import mcse_boot, mcse_mean, mcse_prop


@dataclass
class Diagnosand:
    name: str
    value: float
    mcse: float
    n_contributing: int
    unstable: bool = False
    note: str = ""


def compute_diagnosands(sim: dict, theta: float, unit: float, thresholds,
                        row_type: str, has_attrition: bool, alpha: float,
                        rng: np.random.Generator) -> dict:
    S = len(sim["estimate"])
    ok = (~sim["counted_failure"]) & np.isfinite(sim["estimate"])
    n_ok = int(ok.sum())
    est = sim["estimate"][ok]
    sig = sim["sig"][ok]
    tobs = sim["theta_obs"][ok]
    n_sig = int(sig.sum())
    min_n = thresholds.min_conditional_n

    rows: list[Diagnosand] = []

    rej = float(sig.mean()) if n_ok else float("nan")
    drift_val = float(np.mean(tobs - theta)) if n_ok else float("nan")
    drift_mcse = mcse_mean(tobs - theta)
    if row_type == "null":
        if math.isfinite(drift_mcse) and abs(drift_val) > 2 * drift_mcse:
            rej_note = ("TARGET-NULL REJECTION RATE: E[theta_obs] != 0 under "
                        "this null world (drift below); rejections are false "
                        "claims about the target, partly induced by "
                        "selection - not pure test size")
        else:
            rej_note = "test size: E[theta_obs] = 0 under this null world"
    else:
        rej_note = "power at this scenario's theta"
    rows.append(Diagnosand("rejection_rate", rej, mcse_prop(rej, n_ok),
                           n_ok, note=rej_note))

    rows.append(Diagnosand(
        "target_bias", float(np.mean(est - theta)) / unit,
        mcse_mean(est - theta) / unit, n_ok,
        note=f"(E[est] - theta) / {unit:g}; "
             "= estimator_bias + estimand_drift"))
    rows.append(Diagnosand(
        "estimator_bias", float(np.mean(est - tobs)) / unit,
        mcse_mean(est - tobs) / unit, n_ok,
        note="bias for the analyzable-data contrast (analysis problem)"))

    cov = float(sim["covered"][ok].mean()) if n_ok else float("nan")
    rows.append(Diagnosand("coverage", cov, mcse_prop(cov, n_ok), n_ok))
    cov_obs = float(sim["covered_obs"][ok].mean()) if n_ok else float("nan")
    rows.append(Diagnosand(
        "analyzable_coverage", cov_obs, mcse_prop(cov_obs, n_ok), n_ok,
        note="Pr(theta_obs in CI): estimator calibration for the contrast "
             "it actually estimates (not thresholded)"))

    if row_type == "target":
        if n_sig > 0:
            sign_theta = 1.0 if theta > 0 else -1.0
            wrong = np.sign(est[sig]) != sign_theta
            ts = float(wrong.mean())

            def ts_stat(idx):
                s2, e2 = sig[idx], est[idx]
                if not s2.any():
                    return float("nan")
                return float((np.sign(e2[s2]) != sign_theta).mean())

            rows.append(Diagnosand(
                "type_s", ts, mcse_boot(ts_stat, n_ok, rng), n_sig,
                unstable=n_sig < min_n,
                note=(f"only {n_sig} significant simulations (< {min_n}); "
                      "unstable" if n_sig < min_n else "bootstrap MCSE")))

            tm = float(np.mean(np.abs(est[sig]))) / abs(theta)

            def tm_stat(idx):
                s2, e2 = sig[idx], est[idx]
                if not s2.any():
                    return float("nan")
                return float(np.mean(np.abs(e2[s2]))) / abs(theta)

            rows.append(Diagnosand(
                "type_m", tm, mcse_boot(tm_stat, n_ok, rng), n_sig,
                unstable=n_sig < min_n,
                note=(f"only {n_sig} significant simulations (< {min_n}); "
                      "unstable" if n_sig < min_n else "bootstrap MCSE")))
        else:
            for nm in ("type_s", "type_m"):
                rows.append(Diagnosand(nm, float("nan"), float("nan"), 0,
                                       unstable=True,
                                       note="no significant simulations; "
                                            "not estimable"))
    else:
        for nm in ("type_s", "type_m"):
            rows.append(Diagnosand(nm, float("nan"), float("nan"), 0,
                                   note="n/a: undefined under theta = 0"))

    width = (sim["ci_hi"] - sim["ci_lo"])[ok]
    prec_note = "mean CI width"
    if thresholds.max_width is not None and n_ok:
        p_wide = float((width > thresholds.max_width).mean())
        prec_note += (f"; Pr(width > declared max "
                      f"{thresholds.max_width:g}) = {p_wide:.3f}")
    rows.append(Diagnosand("precision",
                           float(width.mean()) if n_ok else float("nan"),
                           mcse_mean(width), n_ok, note=prec_note))

    mf = float(sim["counted_failure"].mean())
    fc = {k: int(sim[k].sum())
          for k in ("fatal", "nonconverged", "degenerate", "warned")}
    degen_ok = sim["degenerate"] & ok
    cov_degen = ""
    if degen_ok.any():
        cov_d = float(sim["covered"][degen_ok].mean())
        cov_o = float(sim["covered"][ok & ~sim["degenerate"]].mean())
        cov_degen = (f"; coverage among degenerate fits {cov_d:.3f} vs "
                     f"{cov_o:.3f} among others")
    rows.append(Diagnosand(
        "model_failure", mf, mcse_prop(mf, S), S,
        note=(f"counted classes only; all classes F/N/D/W = "
              f"{fc['fatal']}/{fc['nonconverged']}/{fc['degenerate']}/"
              f"{fc['warned']} of {S} (degenerate "
              f"{'counted' if sim['degenerate_counts'] else 'not counted'})"
              + cov_degen)))

    rows.append(Diagnosand(
        "estimand_drift", drift_val / unit, drift_mcse / unit, n_ok,
        note=("(E[theta_obs] - theta) / Delta: what the data strategy does "
              "to the question (design problem)" if has_attrition else
              "no attrition or exclusions declared; zero in expectation "
              "by construction")))

    return {
        "rows": {r.name: r for r in rows},
        "n_sims": S, "n_ok": n_ok, "n_sig": n_sig,
        "mean_n_analyzed": float(sim["n_analyzed"].mean()),
        "mean_attrition_realized": float(sim["attrition_realized"].mean()),
        "failure_classes": fc,
    }
