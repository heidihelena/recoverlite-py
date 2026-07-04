---
name: recoverlite-py
description: >
  Run a pre-data recovery test on a planned study design with the
  recoverlite Python package (the protocol-identical mirror of the R
  package). Use when a user working in Python asks whether a planned
  study can detect or recover its effect, requests a power analysis or
  design feasibility check for a two-arm or cluster-randomized trial, or
  wants a PASS/RISK/FAIL design verdict with simulation diagnosands
  before data collection.
---

# Running recovery tests with recoverlite (Python)

Protocol-identical mirror of the R package
(https://github.com/heidihelena/recoverlite): same declaration API,
crossed scenario grid, diagnosands, versioned threshold profiles
(`recoverlite-thresholds-0.2`), verdict rule, and report structure.
Numbers agree with R within Monte Carlo error, not byte-identically.

```bash
pip install recoverlite            # core (numpy + scipy)
pip install "recoverlite[mixed]"   # + statsmodels for mixed models
```

```python
import recoverlite as rl

design = rl.declare_recovery(
    target=rl.target_estimand(
        estimand="<one sentence: quantity, population, scale>",
        scale="latent-outcome standardized mean difference",
        sesoi=0.40),
    data_strategy=rl.two_arm_trial(n_per_arm=115),   # RECRUITED per arm
    measurement=rl.measured_outcome(reliability=0.70),
    missingness=rl.attrition_model(rate=0.15, mechanism="differential"),
    answer_strategy=rl.planned_analysis(
        estimator="linear_model", formula="y_observed ~ treatment"))

result = rl.recovery_test(design, sims=2000,
                          scenarios="confirmatory_grid", seed=1)
print(rl.verdict(result))
rl.report(result)          # ALWAYS travels with the verdict
```

The seven rules an agent must not break are identical to the R
package's SKILL.md (github.com/heidihelena/recoverlite/blob/main/SKILL.md):
fix thresholds before simulating; never shrink the effect for the
pessimistic scenario (fragility is `rl.effect_fragility` /
`rl.nuisance_fragility`, outside the verdict); the verdict never travels
alone; a PASS is evidence about the instrument, not the world; prefer
user evidence over package defaults (`evidence=` arguments); pre-specify
degenerate-fit handling (`degenerate_counts`); always set a `seed`.

**Python-specific divergences to disclose when relevant:**

- `lmm_random_intercept` supports **Wald-z inference only** (statsmodels
  has no Satterthwaite or Kenward-Roger). For small-cluster designs
  where the inference method IS the design decision, recommend the R
  package — with few clusters, Wald-z is known to be anti-conservative
  and the recovery test will show it.
- Formulas are strings in R syntax: "y_observed ~ treatment",
  "y_observed ~ treatment + baseline",
  "y_observed ~ treatment + (1 | cluster)". Columns available:
  y_observed, treatment, baseline, cluster.
- Results are dataclasses/dicts, not data frames:
  `result.runs["target_declared"]["diagnosands"]["rows"]["power" ...]`
  — each row has .value, .mcse, .n_contributing, .unstable, .note.
- Cross-implementation agreement with the R worked examples is enforced
  by tests/test_smoke_and_r_crosscheck.py; if you change the DGP or an
  estimator, those tolerances are the contract you must keep.
