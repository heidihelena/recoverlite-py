"""Random-intercept linear mixed model: REML fit plus Wald-z,
Satterthwaite, and Kenward-Roger inference for a fixed-effect contrast.

Self-contained (numpy/scipy only). The model is
    y = X beta + u_g + e,   u_g ~ N(0, tau2), e ~ N(0, sigma2),
with Sigma block-diagonal: Sigma_j = sigma2 I + tau2 J for cluster j.
All block algebra is closed-form (Woodbury), so fits are fast and exact.

Satterthwaite follows the lmerTest algorithm: for a contrast c,
    df = 2 f(theta)^2 / (g' A g),
with f(theta) = c' V(theta) c, V = (X' Sigma^{-1} X)^{-1}, g the analytic
gradient of f in theta = (tau2, sigma2), and A the covariance of the
REML variance estimates (inverse numeric Hessian of the negative REML
log-likelihood).

Kenward-Roger follows Kenward & Roger (1997) as implemented by pbkrtest
for covariance structures LINEAR in the variance parameters (true here,
so the second-derivative term vanishes): the adjusted covariance
    Phi_A = Phi + 2 Phi [ sum_ij W_ij (Q_ij - P_i Phi P_j) ] Phi
and the KR denominator degrees of freedom m. Inference mirrors
lmerTest's summary(..., ddf="Kenward-Roger"): SE from Phi_A, t with m
degrees of freedom.

Validated against lmerTest/pbkrtest on shared datasets (see
tests/data/lmm_reference.json): estimates, SEs, dfs, and p-values agree
to numerical precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class LmmFit:
    beta: np.ndarray
    se_wald: np.ndarray          # from Phi = (X' Sigma^-1 X)^-1
    tau2: float
    sigma2: float
    converged: bool
    degenerate: bool             # tau2 at the zero boundary
    phi: np.ndarray              # vcov of beta (unadjusted)
    # per-cluster cached structure
    _groups: list


def _group_index(groups) -> list:
    groups = np.asarray(groups)
    ids = np.unique(groups)
    return [np.flatnonzero(groups == g) for g in ids]


def _gls_profile(lam: float, y: np.ndarray, X: np.ndarray, idx: list):
    """Profiled REML pieces at variance ratio lam = tau2/sigma2.

    Returns (XtRiX, XtRiy, rss_R, logdet_R, beta) where R = I + lam Z Z'
    (so Sigma = sigma2 R), all computed blockwise via Woodbury.
    """
    p = X.shape[1]
    XtRiX = np.zeros((p, p))
    XtRiy = np.zeros(p)
    logdet_R = 0.0
    # R_j^{-1} = I - (lam / (1 + m_j lam)) J
    for ix in idx:
        m = ix.size
        Xj, yj = X[ix], y[ix]
        w = lam / (1.0 + m * lam)
        sx = Xj.sum(axis=0)
        sy = yj.sum()
        XtRiX += Xj.T @ Xj - w * np.outer(sx, sx)
        XtRiy += Xj.T @ yj - w * sx * sy
        logdet_R += math.log1p(m * lam)
    beta = np.linalg.solve(XtRiX, XtRiy)
    rss = 0.0
    for ix in idx:
        m = ix.size
        rj = y[ix] - X[ix] @ beta
        w = lam / (1.0 + m * lam)
        rss += rj @ rj - w * rj.sum() ** 2
    return XtRiX, XtRiy, rss, logdet_R, beta


def _reml_profile_crit(lam: float, y, X, idx, n: int, p: int) -> float:
    """-2 profiled REML log-likelihood (up to a constant) at lam."""
    XtRiX, _, rss, logdet_R, _ = _gls_profile(lam, y, X, idx)
    sigma2 = rss / (n - p)
    sign, logdet_XtRiX = np.linalg.slogdet(XtRiX)
    if sign <= 0 or sigma2 <= 0:
        return np.inf
    return ((n - p) * math.log(sigma2) + logdet_R + logdet_XtRiX)


def reml_neg_loglik(tau2: float, sigma2: float, y, X, idx) -> float:
    """Negative REML log-likelihood (up to a constant) at (tau2, sigma2).

    -2 l_R = (n-p) log sigma2 + log|R| + log|X'R^-1 X| + rss_R / sigma2
    with R = I + (tau2/sigma2) Z Z'. Used for the numeric Hessian.
    """
    n, p = X.shape
    if sigma2 <= 0 or tau2 < 0:
        return np.inf
    lam = tau2 / sigma2
    XtRiX, _, rss, logdet_R, _ = _gls_profile(lam, y, X, idx)
    sign, logdet_XtRiX = np.linalg.slogdet(XtRiX)
    if sign <= 0:
        return np.inf
    return 0.5 * ((n - p) * math.log(sigma2) + logdet_R + logdet_XtRiX
                  + rss / sigma2)


def fit_reml(y, X, groups, boundary_tol: float = 1e-6) -> LmmFit:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    idx = _group_index(groups)

    crit = lambda u: _reml_profile_crit(math.exp(u), y, X, idx, n, p)
    res = optimize.minimize_scalar(crit, bounds=(-14.0, 10.0),
                                   method="bounded",
                                   options={"xatol": 1e-10})
    lam_hat = math.exp(res.x)
    crit0 = _reml_profile_crit(0.0, y, X, idx, n, p)
    degenerate = False
    if crit0 <= res.fun + 1e-8 or lam_hat < boundary_tol:
        lam_hat = 0.0
        degenerate = True

    XtRiX, _, rss, _, beta = _gls_profile(lam_hat, y, X, idx)
    sigma2 = rss / (n - p)
    tau2 = lam_hat * sigma2
    phi = sigma2 * np.linalg.inv(XtRiX)
    return LmmFit(beta=beta, se_wald=np.sqrt(np.diag(phi)), tau2=tau2,
                  sigma2=sigma2, converged=bool(res.success),
                  degenerate=degenerate, phi=phi, _groups=idx)


# ---------------------------------------------------------------------
# Shared small-sample machinery
# ---------------------------------------------------------------------

def _phi_of_theta(tau2, sigma2, X, idx):
    """Phi = (X' Si X)^-1 plus the P and Q matrices of the KR machinery.

    With Si = Sigma^-1 (blockwise a I - b J; a = 1/sigma2,
    b = tau2/(sigma2 d), d = sigma2 + m tau2) and derivative directions
    dSigma_tau = ZZ' (blockwise J), dSigma_sig = I:

        P_i  = X' Si dSigma_i Si X
        Q_ij = X' Si dSigma_i Si dSigma_j Si X

    Closed forms use Si 1 = (1/d) 1 and J Si J = (m/d) J per block.
    """
    p = X.shape[1]
    XtSiX = np.zeros((p, p))
    P_tau = np.zeros((p, p))
    P_sig = np.zeros((p, p))
    Q_tt = np.zeros((p, p))
    Q_ts = np.zeros((p, p))
    Q_ss = np.zeros((p, p))
    tr_JJ = tr_JI = tr_II = 0.0
    for ix in idx:
        m = ix.size
        Xj = X[ix]
        d = sigma2 + m * tau2
        a = 1.0 / sigma2
        b = tau2 / (sigma2 * d)
        sx = Xj.sum(axis=0)
        SiX = a * Xj - b * np.outer(np.ones(m), sx)          # Si X
        v = SiX.sum(axis=0)                                   # 1' Si X
        SiSiX = a * SiX - b * np.outer(np.ones(m), v)         # Si^2 X
        XtSiX += Xj.T @ SiX
        # P_tau = (Si X)' J (Si X) = v v'
        P_tau += np.outer(v, v)
        # P_sig = X' Si^2 X
        P_sig += SiX.T @ SiX
        # Q_tt = X' Si J Si J Si X = (1' Si 1) v v' = (m/d) v v'
        Q_tt += (m / d) * np.outer(v, v)
        # Q_ts = X' Si J Si^2 X = v (1' Si^2 X) = v v' / d
        Q_ts += np.outer(v, v) / d
        # Q_ss = X' Si^3 X = (Si X)' (Si^2 X)
        Q_ss += SiX.T @ SiSiX
        # scalar traces for the expected REML information:
        # tr(Si J Si J) = (m/d)^2; tr(Si J Si) = m/d^2;
        # tr(Si^2) = m a^2 - 2 a b m + b^2 m^2
        tr_JJ += (m / d) ** 2
        tr_JI += m / d ** 2
        tr_II += m * a * a - 2 * a * b * m + b * b * m * m
    phi = np.linalg.inv(XtSiX)
    return phi, (P_tau, P_sig), (Q_tt, Q_ts, Q_ss), (tr_JJ, tr_JI, tr_II)


def _theta_cov(tau2, sigma2, y, X, idx) -> np.ndarray:
    """Covariance of (tau2_hat, sigma2_hat): inverse numeric Hessian of
    the negative REML log-likelihood (central differences; one-sided in
    tau2 at the boundary)."""
    theta = np.array([tau2, sigma2], dtype=float)
    h = np.maximum(1e-4 * np.maximum(np.abs(theta), 1e-3), 1e-8)

    def f(t):
        return reml_neg_loglik(max(t[0], 0.0) if t[0] > -h[0] else np.inf,
                               t[1], y, X, idx) if t[0] >= 0 else np.inf

    # shift evaluation point off the boundary if needed
    t0 = theta.copy()
    if t0[0] < h[0]:
        t0[0] = h[0]
    H = np.zeros((2, 2))
    f0 = f(t0)
    for i in range(2):
        for j in range(i, 2):
            ei = np.zeros(2); ei[i] = h[i]
            ej = np.zeros(2); ej[j] = h[j]
            if i == j:
                H[i, i] = (f(t0 + ei) - 2 * f0 + f(t0 - ei)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (
                    f(t0 + ei + ej) - f(t0 + ei - ej)
                    - f(t0 - ei + ej) + f(t0 - ei - ej)
                ) / (4 * h[i] * h[j])
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full((2, 2), np.nan)


def _expected_information(phi, P, Q, trs) -> np.ndarray:
    """Expected REML information for theta = (tau2, sigma2):
    I_ij = 0.5 tr(P dSigma_i P dSigma_j) with the REML projection
    P = Si - Si X Phi X' Si, expanded into blockwise closed forms:
    tr(PAPB) = tr(S A S B) - 2 tr(Phi T_AB) + tr(Phi U'AU Phi U'BU),
    with T_AB = U'B S A U (equal to the Q matrices) and U'JU = P_tau,
    U'IU = P_sig. This is the information pbkrtest bases W on — the
    observed Hessian (used for Satterthwaite, matching lmerTest) differs
    in finite samples.
    """
    (P_tau, P_sig) = P
    (Q_tt, Q_ts, Q_ss) = Q
    (tr_JJ, tr_JI, tr_II) = trs
    i_tt = 0.5 * (tr_JJ - 2 * np.trace(phi @ Q_tt)
                  + np.trace(phi @ P_tau @ phi @ P_tau))
    i_ts = 0.5 * (tr_JI - 2 * np.trace(phi @ Q_ts)
                  + np.trace(phi @ P_tau @ phi @ P_sig))
    i_ss = 0.5 * (tr_II - 2 * np.trace(phi @ Q_ss)
                  + np.trace(phi @ P_sig @ phi @ P_sig))
    return np.array([[i_tt, i_ts], [i_ts, i_ss]])


def satterthwaite(fit: LmmFit, y, X, c: np.ndarray) -> tuple[float, float]:
    """(se, df) for contrast c'beta, lmerTest-style."""
    idx = fit._groups
    phi, (P_tau, P_sig), _, _ = _phi_of_theta(fit.tau2, fit.sigma2, X, idx)
    f_val = float(c @ phi @ c)
    # d(c' V c)/d theta_i = c' V (X' Si dSigma_i Si X) V c
    g = np.array([float(c @ phi @ P_tau @ phi @ c),
                  float(c @ phi @ P_sig @ phi @ c)])
    A = _theta_cov(fit.tau2, fit.sigma2, y, X, idx)
    denom = float(g @ A @ g)
    if not math.isfinite(denom) or denom <= 0:
        return math.sqrt(f_val), float("nan")
    return math.sqrt(f_val), 2.0 * f_val ** 2 / denom


def kenward_roger(fit: LmmFit, y, X,
                  c: np.ndarray) -> tuple[float, float]:
    """(se_adjusted, df) for contrast c'beta, per Kenward & Roger (1997)
    for covariance structures linear in the variance parameters,
    mirroring lmerTest's summary(..., ddf="Kenward-Roger")."""
    idx = fit._groups
    phi, (P1, P2), (Q11, Q12, Q22), trs = _phi_of_theta(
        fit.tau2, fit.sigma2, X, idx)
    # KR uses the EXPECTED REML information for W (as pbkrtest does);
    # the observed Hessian belongs to the Satterthwaite path.
    info = _expected_information(phi, (P1, P2), (Q11, Q12, Q22), trs)
    try:
        W = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return float(np.sqrt(c @ phi @ c)), float("nan")
    if not np.all(np.isfinite(W)):
        return float(np.sqrt(c @ phi @ c)), float("nan")
    P = [P1, P2]
    Q = [[Q11, Q12], [Q12.T, Q22]]

    # Adjusted covariance (R_ij = 0: Sigma linear in theta)
    M = np.zeros_like(phi)
    for i in range(2):
        for j in range(2):
            M += W[i, j] * (Q[i][j] - P[i] @ phi @ P[j])
    phi_a = phi + 2.0 * phi @ M @ phi

    # KR denominator df for q = 1
    q = 1
    Lmat = np.outer(c, c) / float(c @ phi @ c)   # Theta = c(c'Phi c)^-1 c'
    A1 = 0.0
    A2 = 0.0
    TP = [Lmat @ phi @ P[i] @ phi for i in range(2)]
    for i in range(2):
        for j in range(2):
            A1 += W[i, j] * np.trace(TP[i]) * np.trace(TP[j])
            A2 += W[i, j] * np.trace(TP[i] @ TP[j])
    B = (A1 + 6.0 * A2) / (2.0 * q)
    g_num = (q + 1) * A1 - (q + 4) * A2
    g_den = (q + 2) * A2
    gg = g_num / g_den if g_den != 0 else 0.0
    denom_c = 3 * q + 2 * (1 - gg)
    c1 = gg / denom_c
    c2 = (q - gg) / denom_c
    c3 = (q + 2 - gg) / denom_c
    E_star_inv = 1.0 - A2 / q
    if E_star_inv <= 0:
        return float(np.sqrt(c @ phi_a @ c)), float("nan")
    E_star = 1.0 / E_star_inv
    V_star = (2.0 / q) * (1 + c1 * B) / ((1 - c2 * B) ** 2 * (1 - c3 * B))
    rho = V_star / (2.0 * E_star ** 2)
    if q * rho <= 1:
        df = float("inf")
    else:
        df = 4.0 + (q + 2) / (q * rho - 1)
    return float(np.sqrt(c @ phi_a @ c)), df


def infer_contrast(fit: LmmFit, y, X, c_index: int, alpha: float,
                   inference: str, fallback_df: float) -> dict:
    """Estimate/SE/CI/p for one coefficient under the chosen inference."""
    p = X.shape[1]
    c = np.zeros(p)
    c[c_index] = 1.0
    est = float(fit.beta[c_index])

    if inference == "wald_z":
        se = float(fit.se_wald[c_index])
        crit = stats.norm.ppf(1 - alpha / 2)
        pval = 2 * stats.norm.sf(abs(est / se))
        return dict(estimate=est, se=se, p=float(pval),
                    ci_lo=est - crit * se, ci_hi=est + crit * se)

    if inference == "satterthwaite":
        se, df = satterthwaite(fit, y, X, c)
    else:  # kenward_roger
        se, df = kenward_roger(fit, y, X, c)
    if not math.isfinite(df) or df <= 0:
        df = fallback_df
    crit = stats.t.ppf(1 - alpha / 2, df)
    pval = 2 * stats.t.sf(abs(est / se), df)
    return dict(estimate=est, se=float(se), p=float(pval),
                ci_lo=est - crit * se, ci_hi=est + crit * se, df=float(df))
