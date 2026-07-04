# recoverlite (Python)

[![CI](https://github.com/heidihelena/recoverlite-py/actions/workflows/ci.yaml/badge.svg)](https://github.com/heidihelena/recoverlite-py/actions/workflows/ci.yaml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Pre-data recovery tests for planned study designs** — the Python
mirror of the [R package](https://github.com/heidihelena/recoverlite).

A planned study can be unable to support its intended inferential claim
even when the researcher's substantive assumptions are correct.
`recoverlite` simulates a *declared* design–analysis pair over a crossed
scenario grid — null and target effects, each under declared and
pessimistically perturbed nuisance assumptions — and converts the
diagnosands into a **PASS / RISK / FAIL** verdict under a pre-specified,
versioned threshold profile.

## Mirror contract

This package is **protocol-identical** to the R implementation: same
declaration API, same scenario grid, same diagnosands (including the
exact decomposition *target bias = estimator bias + estimand drift*),
same threshold profiles (shared version string
`recoverlite-thresholds-0.2`), same verdict rule, same report structure.
Simulated numbers agree **within Monte Carlo error, not byte-identically**
— R and Python cannot share RNG streams. The test suite enforces the
contract by reproducing the R package's archived worked-example
diagnosands within 4× combined MCSE.

| | R package | Python mirror |
|---|---|---|
| Two-arm trial (baseline, additive measurement error, MAR/MCAR attrition, noncompliance) | ✅ | ✅ |
| Complete-case linear model | ✅ | ✅ |
| MI baseline-adjusted estimator (Rubin + Barnard-Rubin) | ✅ | ✅ |
| Cluster trial, LMM Wald-z | ✅ | ✅ (`pip install recoverlite[mixed]`) |
| Cluster-level t-test | ✅ | ✅ |
| LMM Satterthwaite / Kenward–Roger inference | ✅ | ❌ R-only (no Python equivalent) |
| Fragility curves (effect + nuisance) | ✅ | ✅ |

## Install

```bash
pip install recoverlite            # core (numpy + scipy)
pip install "recoverlite[mixed]"   # + statsmodels for mixed models
```

Not yet on PyPI? Install from source:
`pip install git+https://github.com/heidihelena/recoverlite-py`.

## The workflow in one block

```python
import recoverlite as rl

design = rl.declare_recovery(
    target=rl.target_estimand(
        estimand="ITT mean difference at 12 weeks",
        scale="latent-outcome standardized mean difference",
        sesoi=0.40,
    ),
    data_strategy=rl.two_arm_trial(n_per_arm=115),
    measurement=rl.measured_outcome(reliability=0.70),
    missingness=rl.attrition_model(rate=0.15, mechanism="differential"),
    answer_strategy=rl.planned_analysis(
        estimator="linear_model",
        formula="y_observed ~ treatment",
    ),
)

result = rl.recovery_test(design, sims=2000,
                          scenarios="confirmatory_grid", seed=1)
print(rl.verdict(result))   # PASS / RISK / FAIL (+ strict/lenient recompute)
rl.report(result)           # the standalone report always travels with it
```

Cluster designs use `cluster_trial()` with
`planned_analysis("lmm_random_intercept", "y_observed ~ treatment + (1 | cluster)")`
(Wald-z) or `"cluster_mean_ttest"`. Fragility curves —
`effect_fragility()`, `nuisance_fragility()` — are deliberately outside
the verdict.

## Citation

> Andersen, H. H. (2026). *Recovery before data: pre-data simulation
> diagnosis of planned study designs.* Working paper; preprint
> forthcoming. https://github.com/heidihelena/recoverlite

A PASS is evidence about the instrument, not about the world.

## License

[Apache License 2.0](LICENSE).
