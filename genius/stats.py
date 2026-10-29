"""Statistics used by the Validation room: rank IC, Newey-West t, one-sided p-values, Benjamini-Hochberg, AUC/KS/Brier, a small HMM."""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def spearman_by_month(pred: np.ndarray, y: np.ndarray, months, min_n: int = 30) -> np.ndarray:
    """Cross-sectional rank correlation each month (columns of N x T arrays). NaN where too few firms."""
    out = np.full(len(months), np.nan)
    for j, t in enumerate(months):
        a, b = pred[:, t], y[:, t]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() >= min_n:
            out[j] = st.spearmanr(a[ok], b[ok])[0]
    return out


def newey_west_t(x: np.ndarray, lags: int = 2) -> tuple[float, float, int]:
    """Mean, Newey-West t-statistic and n of a series (NaNs dropped)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), n
    m = x.mean()
    e = x - m
    s = (e @ e) / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(m), float(m / se), n


def p_one_sided(t: float, n: int) -> float:
    if np.isnan(t) or n < 2:
        return float("nan")
    return float(st.t.sf(t, df=n - 1))


def critical_t(conf: float, n: int) -> float:
    return float(st.t.ppf(conf, df=max(1, n - 1)))


def bh(pvals, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg: which hypotheses are rejected at false-discovery rate q (NaN p-values never rejected)."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.zeros(len(p), bool)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        k = np.max(np.where(passed)[0])
        out[order[:k + 1]] = True
    return out


def auc(score_bad: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve where higher score_bad means more likely y = 1."""
    ok = ~np.isnan(score_bad) & ~np.isnan(y)
    s, t = score_bad[ok], y[ok]
    if t.min() == t.max():
        return float("nan")
    r = st.rankdata(s)
    n1 = t.sum(); n0 = len(t) - n1
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks(score_bad, y):
    ok = ~np.isnan(score_bad) & ~np.isnan(y)
    s, t = score_bad[ok], y[ok]
    if t.min() == t.max():
        return float("nan")
    return float(st.ks_2samp(s[t == 1], s[t == 0]).statistic)


def brier(p, y):
    ok = ~np.isnan(p) & ~np.isnan(y)
    return float(np.mean((p[ok] - y[ok]) ** 2))


# --------------------------------------------------------------------------------------------- two-state Gaussian HMM (regime)
def hmm_fit(x: np.ndarray, iters: int = 200, seed: int = 0) -> dict:
    """Two-state Gaussian hidden Markov model fitted by EM on a 1-D series (NaNs skipped). State 1 = the lower-mean (stress) state."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    mu = np.array([np.percentile(x, 70), np.percentile(x, 15)])
    sd = np.array([x.std(), x.std()])
    A = np.array([[0.95, 0.05], [0.2, 0.8]])
    pi = np.array([0.8, 0.2])
    for _ in range(iters):
        B = st.norm.pdf(x[:, None], mu[None, :], sd[None, :]) + 1e-300
        a, c = _forward(pi, A, B)
        b = _backward(A, B, c)
        g = a * b
        g /= g.sum(1, keepdims=True)
        xi = (a[:-1, :, None] * A[None] * (B[1:] * b[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(0) / g[:-1].sum(0)[:, None]
        A /= A.sum(1, keepdims=True)
        pi = g[0]
        mu = (g * x[:, None]).sum(0) / g.sum(0)
        sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / g.sum(0)) + 1e-6
    if mu[0] < mu[1]:
        mu, sd = mu[::-1], sd[::-1]
        A = A[::-1, ::-1]; pi = pi[::-1]
    return dict(mu=mu, sd=sd, A=A, pi=pi)


def _forward(pi, A, B):
    T = len(B)
    a = np.zeros((T, 2)); c = np.zeros(T)
    a[0] = pi * B[0]; c[0] = a[0].sum(); a[0] /= c[0]
    for t in range(1, T):
        a[t] = (a[t - 1] @ A) * B[t]
        c[t] = a[t].sum(); a[t] /= c[t]
    return a, c


def _backward(A, B, c):
    T = len(B)
    b = np.ones((T, 2))
    for t in range(T - 2, -1, -1):
        b[t] = (A @ (B[t + 1] * b[t + 1])) / c[t + 1]
    return b


def hmm_filter(model: dict, x: np.ndarray) -> np.ndarray:
    """Probability of the stress state at each time using data up to that time only (filtered, never smoothed)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    p = model["pi"].copy()
    first = True
    for t, v in enumerate(x):
        if np.isnan(v):
            out[t] = p[1]
            continue
        lik = st.norm.pdf(v, model["mu"], model["sd"]) + 1e-300
        prior = p if first else p @ model["A"]
        post = prior * lik
        p = post / post.sum()
        first = False
        out[t] = p[1]
    return out
