"""Simulation engine (protocol Step 4).

Each simulation records the estimate, interval, decision, a CLASSIFIED
failure indicator (fatal / nonconvergence / degenerate / warnings), and
the analyzable-data contrast theta_obs_s computed on true
(pre-measurement-error) outcomes among the analysis population — the raw
material for the estimand-drift diagnosand and the exact decomposition
target bias = estimator bias + drift.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import integrate, optimize, stats
from scipy.special import expit as _plogis


def calibrate_dropout_intercept(rate: float, slope: float) -> float:
    """Solve E_B[plogis(alpha + slope B)] = rate for B ~ N(0, 1).

    A zero rate returns a finite stand-in (-30; plogis ~ 1e-13) so the
    arm-interaction arithmetic cannot produce NaN when only one arm's
    rate is zero.
    """
    if rate <= 0:
        return -30.0
    if slope == 0:
        return float(np.log(rate / (1 - rate)))

    def marginal(a):
        val, _ = integrate.quad(
            lambda b: _plogis(a + slope * b) * stats.norm.pdf(b),
            -np.inf, np.inf)
        return val - rate

    return float(optimize.brentq(marginal, -25, 25))


def draw_data(params: dict, rng: np.random.Generator) -> dict:
    """One simulated dataset under one scenario row's parameters."""
    if params["kind"] == "two_arm_trial":
        n = 2 * params["n_per_arm"]
        m_treat = round(n * params["allocation"])
        z = np.zeros(n, dtype=int)
        z[rng.permutation(n)[:m_treat]] = 1
        baseline = rng.standard_normal(n)
        eps = rng.standard_normal(n)
        nc = params.get("noncompliance", 0.0) or 0.0
        complier = (rng.random(n) >= nc).astype(int) if nc > 0 else \
            np.ones(n, dtype=int)
        rho = params["rho"]
        eff = params["effect"]
        y_true = eff * complier * z + rho * baseline + \
            math.sqrt(1 - rho ** 2) * eps
        cluster = None
    else:
        n_cl = params["n_clusters"]
        n_pp = params["n_per_cluster"]
        n = n_cl * n_pp
        m_cl = round(n_cl * params["allocation"])
        z_cl = np.zeros(n_cl, dtype=int)
        z_cl[rng.permutation(n_cl)[:m_cl]] = 1
        cluster = np.repeat(np.arange(n_cl), n_pp)
        z = z_cl[cluster]
        icc = params["icc"]
        u_c = rng.normal(0, math.sqrt(icc), n_cl)
        e = rng.normal(0, math.sqrt(1 - icc), n)
        baseline = rng.standard_normal(n)
        y_true = params["effect"] * z + u_c[cluster] + e

    rel = params.get("reliability")
    if rel is not None and rel < 1:
        err_sd = math.sqrt(1.0 / rel - 1.0)
        y_obs = y_true + err_sd * rng.standard_normal(n)
    else:
        y_obs = y_true.copy()

    att = params.get("attrition")
    if att is not None:
        a0 = calibrate_dropout_intercept(att["rate_control"],
                                         att["slope_control"])
        a1 = calibrate_dropout_intercept(att["rate_treated"],
                                         att["slope_treated"])
        g0, g1 = att["slope_control"], att["slope_treated"]
        logit_p = (a0 + (a1 - a0) * z) + (g0 + (g1 - g0) * z) * baseline
        retained = rng.random(n) > _plogis(logit_p)
    else:
        retained = np.ones(n, dtype=bool)

    return {"treatment": z, "baseline": baseline, "y_true": y_true,
            "y_observed": y_obs, "retained": retained, "cluster": cluster}


# ---------------------------------------------------------------------
# Answer strategies. Each returns a dict with estimate/se/ci/p and the
# four-class failure record. Hard errors are recorded, not raised.
# ---------------------------------------------------------------------

_EMPTY = dict(estimate=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan,
              p=np.nan, fatal=False, nonconverged=False, degenerate=False,
              warned=False)


def _design_matrix(terms: dict, data: dict, mask: np.ndarray) -> np.ndarray:
    cols = [np.ones(int(mask.sum()))]
    for term in terms["fixed"]:
        if term not in data or data[term] is None:
            raise KeyError(f"unknown formula term '{term}'")
        cols.append(np.asarray(data[term], dtype=float)[mask])
    return np.column_stack(cols)


def _ols(y, X, alpha):
    n, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < k:
        return None
    resid = y - X @ beta
    dfree = n - k
    sigma2 = float(resid @ resid) / dfree
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return beta, se, dfree, sigma2, xtx_inv


def fit_linear_model(analysis, data) -> dict:
    out = dict(_EMPTY)
    try:
        mask = data["retained"]
        terms = analysis.terms
        y = np.asarray(data[terms["outcome"]], dtype=float)[mask]
        X = _design_matrix(terms, data, mask)
        res = _ols(y, X, analysis.alpha)
        if res is None:
            out["fatal"] = True
            return out
        beta, se, dfree, _, _ = res
        idx = 1 + terms["fixed"].index("treatment")
        est, s = float(beta[idx]), float(se[idx])
        tcrit = stats.t.ppf(1 - analysis.alpha / 2, dfree)
        out.update(estimate=est, se=s,
                   p=float(2 * stats.t.sf(abs(est / s), dfree)),
                   ci_lo=est - tcrit * s, ci_hi=est + tcrit * s)
    except Exception:
        out["fatal"] = True
    return out


def fit_lmm_random_intercept(analysis, data) -> dict:
    """Random-intercept LMM via the internal exact REML fitter (lmm.py),
    with Wald-z, Satterthwaite, or Kenward-Roger inference — validated
    against lmerTest/pbkrtest to numerical precision on shared datasets
    (tests/data/lmm_reference.json)."""
    from .lmm import fit_reml, infer_contrast

    out = dict(_EMPTY)
    try:
        mask = data["retained"]
        terms = analysis.terms
        y = np.asarray(data[terms["outcome"]], dtype=float)[mask]
        X = _design_matrix(terms, data, mask)
        groups = np.asarray(data[terms["random_intercept"]])[mask]
        fit = fit_reml(y, X, groups)
        out["nonconverged"] = not fit.converged
        out["degenerate"] = fit.degenerate
        out["warned"] = fit.degenerate  # boundary notice, lme4-style
        n_groups = len(np.unique(groups))
        idx = 1 + terms["fixed"].index("treatment")
        inf = infer_contrast(fit, y, X, idx, analysis.alpha,
                             analysis.inference,
                             fallback_df=max(n_groups - 2, 1))
        out.update(estimate=inf["estimate"], se=inf["se"], p=inf["p"],
                   ci_lo=inf["ci_lo"], ci_hi=inf["ci_hi"])
    except Exception:
        out = dict(_EMPTY)
        out["fatal"] = True
    return out


def fit_cluster_mean_ttest(analysis, data) -> dict:
    out = dict(_EMPTY)
    try:
        mask = data["retained"]
        cl = np.asarray(data["cluster"])[mask]
        y = np.asarray(data[analysis.terms["outcome"]], dtype=float)[mask]
        z = np.asarray(data["treatment"])[mask]
        ids = np.unique(cl)
        means = np.array([y[cl == c].mean() for c in ids])
        zc = np.array([z[cl == c][0] for c in ids])
        m1, m0 = means[zc == 1], means[zc == 0]
        n1, n0 = len(m1), len(m0)
        est = float(m1.mean() - m0.mean())
        sp2 = ((n1 - 1) * m1.var(ddof=1) + (n0 - 1) * m0.var(ddof=1)) / \
            (n1 + n0 - 2)
        se = math.sqrt(sp2 * (1 / n1 + 1 / n0))
        dfree = n1 + n0 - 2
        tcrit = stats.t.ppf(1 - analysis.alpha / 2, dfree)
        out.update(estimate=est, se=se,
                   p=float(2 * stats.t.sf(abs(est / se), dfree)),
                   ci_lo=est - tcrit * se, ci_hi=est + tcrit * se)
    except Exception:
        out["fatal"] = True
    return out


def fit_mi_baseline_adjusted(analysis, data,
                             rng: np.random.Generator) -> dict:
    """Proper normal-model MI on (treatment, baseline) + Rubin's rules
    with Barnard-Rubin degrees of freedom. Identifiable for the ITT
    estimand when dropout is MAR given the observed baseline."""
    out = dict(_EMPTY)
    try:
        m = analysis.m_imputations
        y = np.asarray(data[analysis.terms["outcome"]], dtype=float).copy()
        obs = np.asarray(data["retained"], dtype=bool)
        n = y.size
        X = np.column_stack([np.ones(n),
                             np.asarray(data["treatment"], dtype=float),
                             np.asarray(data["baseline"], dtype=float)])
        k = X.shape[1]
        Xo, yo = X[obs], y[obs]
        beta_hat, _, rank, _ = np.linalg.lstsq(Xo, yo, rcond=None)
        if rank < k:
            out["fatal"] = True
            return out
        resid = yo - Xo @ beta_hat
        dfr = obs.sum() - k
        rss = float(resid @ resid)
        xtx_inv = np.linalg.inv(Xo.T @ Xo)
        # N(0, V) draws use the LOWER Cholesky factor.
        L = np.linalg.cholesky(xtx_inv)

        # position of the treatment coefficient in the ANALYSIS model
        terms = analysis.terms
        idx = 1 + terms["fixed"].index("treatment")

        ests = np.empty(m)
        var_s = np.empty(m)
        n_mis = int((~obs).sum())
        for j in range(m):
            sigma2 = rss / rng.chisquare(dfr)
            beta = beta_hat + math.sqrt(sigma2) * (L @ rng.standard_normal(k))
            y_imp = y.copy()
            y_imp[~obs] = X[~obs] @ beta + \
                math.sqrt(sigma2) * rng.standard_normal(n_mis)
            data_j = dict(data)
            data_j[terms["outcome"]] = y_imp
            full_mask = np.ones(n, dtype=bool)
            Xa = _design_matrix(terms, data_j, full_mask)
            res = _ols(y_imp, Xa, analysis.alpha)
            beta_j, se_j, _, _, _ = res
            ests[j] = beta_j[idx]
            var_s[j] = se_j[idx] ** 2

        qbar = float(ests.mean())
        ubar = float(var_s.mean())
        b = float(ests.var(ddof=1))
        tt = ubar + (1 + 1 / m) * b
        df_com = n - k
        lam = (1 + 1 / m) * b / tt
        if lam <= 0:
            df_br = df_com
        else:
            df_old = (m - 1) / lam ** 2
            df_obs = ((df_com + 1) / (df_com + 3)) * df_com * (1 - lam)
            df_br = df_old * df_obs / (df_old + df_obs)
        se = math.sqrt(tt)
        tcrit = stats.t.ppf(1 - analysis.alpha / 2, df_br)
        out.update(estimate=qbar, se=se,
                   p=float(2 * stats.t.sf(abs(qbar / se), df_br)),
                   ci_lo=qbar - tcrit * se, ci_hi=qbar + tcrit * se)
    except Exception:
        out = dict(_EMPTY)
        out["fatal"] = True
    return out


def fit_analysis(analysis, data, rng: np.random.Generator) -> dict:
    if analysis.estimator == "linear_model":
        return fit_linear_model(analysis, data)
    if analysis.estimator == "lmm_random_intercept":
        return fit_lmm_random_intercept(analysis, data)
    if analysis.estimator == "cluster_mean_ttest":
        return fit_cluster_mean_ttest(analysis, data)
    return fit_mi_baseline_adjusted(analysis, data, rng)


def run_scenario(design, params: dict, sims: int,
                 rng: np.random.Generator) -> dict:
    """Run one scenario row; returns column arrays keyed by name."""
    analysis = design.answer_strategy
    theta = params["effect"]
    cols = {name: np.empty(sims) for name in
            ("estimate", "se", "ci_lo", "ci_hi", "p", "theta_obs",
             "n_analyzed", "attrition_realized")}
    for name in ("sig", "covered", "covered_obs", "fatal", "nonconverged",
                 "degenerate", "warned", "counted_failure"):
        cols[name] = np.zeros(sims, dtype=bool)

    for i in range(sims):
        data = draw_data(params, rng)
        fit = fit_analysis(analysis, data, rng)
        ret = data["retained"]
        z = data["treatment"]
        yt = data["y_true"]
        with np.errstate(invalid="ignore"):
            theta_obs = yt[ret & (z == 1)].mean() - yt[ret & (z == 0)].mean()
        counted = fit["fatal"] or fit["nonconverged"] or \
            (fit["degenerate"] and analysis.degenerate_counts)
        cols["estimate"][i] = fit["estimate"]
        cols["se"][i] = fit["se"]
        cols["ci_lo"][i] = fit["ci_lo"]
        cols["ci_hi"][i] = fit["ci_hi"]
        cols["p"][i] = fit["p"]
        cols["sig"][i] = (not math.isnan(fit["p"])) and \
            fit["p"] < analysis.alpha
        cols["covered"][i] = (not math.isnan(fit["ci_lo"])) and \
            fit["ci_lo"] <= theta <= fit["ci_hi"]
        cols["covered_obs"][i] = (not math.isnan(fit["ci_lo"])) and \
            fit["ci_lo"] <= theta_obs <= fit["ci_hi"]
        for name in ("fatal", "nonconverged", "degenerate", "warned"):
            cols[name][i] = fit[name]
        cols["counted_failure"][i] = counted
        cols["theta_obs"][i] = theta_obs
        cols["n_analyzed"][i] = int(ret.sum())
        cols["attrition_realized"][i] = 1.0 - ret.mean()
    cols["degenerate_counts"] = analysis.degenerate_counts
    return cols
