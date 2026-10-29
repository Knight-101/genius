"""Second Opinion: an independent machine-learning ranking of the same firms, used only to say how confident we are in the Genius Score.

It never sets a rating, a limit or a rate. It can only make terms more cautious (a Low confidence holds the limit until a person reviews).

Target  : how deep each firm's cash stress gets over the next three months (lowest weeks of cash cover, capped at 8 weeks because
          above that a firm is not near stress), ranked within the month.
          It is never trained on the Genius Score, tier, rate or limit, so agreement means something.
Inputs  : the 54 measures as within-peer-group percentiles, Grit and its parts, direction of travel, steadiness, sector, MSME size,
          months of history, market regime (filtered two-state HMM fitted on the training years), peer trend, interactions, data reliability.
Models  : linear regression, ridge, lasso, elastic net, LightGBM. Trained on years 1-5, refitted every quarter on an expanding window with a
          three-month gap (the target needs three months to be known), the winner is the model with the highest average IC over years 6-8;
          it is then fixed and run once on years 9-10.
Checks  : IC by month (Spearman), Newey-West t with two lags (overlapping three-month targets), one-sided p-value, the same for the Genius
          Score and for the difference; per peer group, significance at the chosen confidence level with optional Benjamini-Hochberg.
Confidence (per firm): Second Opinion significant in the firm's peer group ->
          same fifth of the group as the Genius Score: High; one fifth apart: Medium; two or more fifths more worried: Low;
          two or more fifths more positive: Medium (noted, never an upgrade). Not significant: Scorecard only."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import stats as S

KEY_CHANGE = ["RC01", "RC05", "CR02", "PF01", "RC19", "FL06", "OW03", "PY03"]
MODELS = ["Linear regression (OLS)", "Ridge", "Lasso", "Elastic net", "LightGBM"]


def _model(name):
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    if name == "Linear regression (OLS)":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=10.0)
    if name == "Lasso":
        return Lasso(alpha=0.0005, max_iter=5000)
    if name == "Elastic net":
        return ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=150, learning_rate=0.07, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, random_state=7)


# ---------------------------------------------------------------------------------------------------------------- features
def features(pct: dict, raw: dict, grit: dict, firms: pd.DataFrame, band: np.ndarray, regime_p: np.ndarray, F: np.ndarray,
             zc: np.ndarray, hist: np.ndarray, dq: np.ndarray) -> tuple[dict, list]:
    """pct: measure id -> N x T within-group percentile (direction-aware, 0 worst .. 1 best). Returns (name -> N x T array, names)."""
    from .measures import lag, roll_std, div
    X = {}
    for m in C.MEASURES:
        X[f"p_{m}"] = pct[m]
    X["grit"] = grit["score"] / 100; X["grit_demonstrated"] = grit["demonstrated"] / 100; X["grit_capacity"] = grit["capacity"] / 100
    X["grit_setbacks"] = np.minimum(grit["n"], 10) / 10
    X["setback_now"] = (zc <= -1.3).astype(float)
    ms = np.full(zc.shape, 60.0)
    run = np.full(zc.shape[0], 60.0)
    for t in range(zc.shape[1]):
        run = np.where(zc[:, t] <= -1.3, 0.0, np.minimum(60.0, run + 1))
        ms[:, t] = run
    X["months_since_setback"] = ms / 60
    for m in KEY_CHANGE:
        X[f"d3_{m}"] = pct[m] - lag(pct[m], 3)
        X[f"d12_{m}"] = pct[m] - lag(pct[m], 12)
    rev = raw["rev"]
    g = div(rev, lag(rev, 1)) - 1
    X["vol_sales_6"] = roll_std(g, 6); X["vol_sales_12"] = roll_std(g, 12)
    for k, src in (("ocf", raw["ocf1"]), ("coll", raw["collections"])):
        r = div(src, np.where(rev > 0, rev, np.nan))
        X[f"vol_{k}_6"] = roll_std(r, 6); X[f"vol_{k}_12"] = roll_std(r, 12)
    gm = div(rev - raw["cogs"], np.where(rev > 0, rev, np.nan))
    X["vol_gm_6"] = roll_std(gm, 6); X["vol_gm_12"] = roll_std(gm, 12)
    N, T = rev.shape
    for s in C.SECTOR_LIST:
        X["sector_" + s.split(" · ")[1].replace(" ", "_")] = np.broadcast_to((firms["sector"].values == s)[:, None], (N, T)).astype(float)
    for b in C.SIZE_LIST:
        X["size_" + b] = (band == b).astype(float)
    X["history"] = np.minimum(hist, 60) / 60
    X["regime_stress"] = np.broadcast_to(regime_p[None, :], (N, T))
    X["peer_trend"] = F
    X["peer_trend_3"] = F - lag(F, 3)
    X["x_regime_cash"] = X["regime_stress"] * pct["CR02"]
    X["x_regime_debt"] = X["regime_stress"] * pct["FL06"]
    X["x_regime_overdue"] = X["regime_stress"] * pct["RC05"]
    X["x_trading_collections"] = (firms["family"].values == "trading")[:, None] * pct["RC03"]
    X["x_small_topcust"] = (band == "Micro") * pct["RC09"]
    X["x_setback_capacity"] = X["setback_now"] * X["grit_capacity"]
    X["data_confidence"] = np.broadcast_to((dq / 100)[:, None], (N, T))
    miss = np.stack([np.isnan(pct[m]) for m in C.MEASURES], 0).mean(0)
    X["missing_share"] = miss
    for m in C.MEASURES:
        if np.isnan(pct[m]).mean() > 0.20:
            X[f"miss_{m}"] = np.isnan(pct[m]).astype(float)
    names = list(X)
    return {k: np.asarray(v, np.float32) for k, v in X.items()}, names


def _matrix(X: dict, names: list, rows, cols, fill: dict | None = None):
    M = np.stack([X[n][rows, cols] for n in names], 1).astype(np.float32)
    if fill is None:
        fill = {i: (0.5 if n.startswith("p_") else float(np.nanmedian(M[:, i])) if np.isfinite(np.nanmedian(M[:, i])) else 0.0) for i, n in enumerate(names)}
    for i in range(M.shape[1]):
        c = M[:, i]
        c[np.isnan(c)] = fill[i]
    return M, fill


def target(severity: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Within-month percentile of next-quarter severity (0 = deepest stress, 1 = most cover)."""
    y = np.full(severity.shape, np.nan, np.float32)
    for t in range(severity.shape[1]):
        v = severity[:, t]
        ok = ~np.isnan(v) & alive[:, t]
        if ok.sum() > 10:
            r = pd.Series(v[ok]).rank(pct=True).to_numpy()
            y[ok, t] = r
    return y


# ---------------------------------------------------------------------------------------------------------------- walk-forward
def walk_forward(X: dict, names: list, y: np.ndarray, eligible: np.ndarray, months: range, train_until_first: int, models: list,
                 embargo: int = 3, refit_every: int = 3, sample_every: int = 3, log=print) -> dict:
    """Predictions (N x T) for each model over `months`, refitting every quarter on all eligible rows with month <= now - embargo."""
    N, T = y.shape
    preds = {m: np.full((N, T), np.nan, np.float32) for m in models}
    fitted = {}
    fills = None
    last_fit = None
    for t in months:
        if last_fit is None or t - last_fit >= refit_every:
            cut = t - embargo
            tr_months = np.arange(0, cut + 1)
            tr_months = tr_months[tr_months % sample_every == (cut % sample_every)]
            rr, cc = np.where(eligible[:, tr_months] & ~np.isnan(y[:, tr_months]))
            cc = tr_months[cc]
            Xtr, fills = _matrix(X, names, rr, cc)
            ytr = y[rr, cc]
            for m in models:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mdl = _model(m).fit(Xtr, ytr)
                fitted[m] = mdl
            last_fit = t
            log(f"  refit at month {t} on {len(ytr):,} rows")
        rows = np.where(eligible[:, t])[0]
        if len(rows) == 0:
            continue
        Xt, _ = _matrix(X, names, rows, np.full(len(rows), t), fills)
        for m in models:
            preds[m][rows, t] = fitted[m].predict(Xt)
    return dict(preds=preds, fitted=fitted, fills=fills)


def importance(model, names: list, top: int = 15) -> list:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_, float))
    else:
        return []
    imp = imp / max(imp.sum(), 1e-12)
    o = np.argsort(-imp)[:top]
    return [(names[i], float(imp[i])) for i in o]


# ---------------------------------------------------------------------------------------------------------------- statistics
def ic_table(pred: np.ndarray, sev: np.ndarray, months) -> dict:
    ic = S.spearman_by_month(pred, sev, list(months))
    m, t, n = S.newey_west_t(ic, lags=2)
    return dict(ic_series=ic, mean=m, t=t, n=n, p=S.p_one_sided(t, n), hit=float(np.nanmean(ic > 0)) if n else np.nan)


def group_ic(pred, sev, keys, months, min_n=20) -> pd.DataFrame:
    rows = []
    allk = pd.unique(np.concatenate([keys[:, t] for t in months]))
    for k in allk:
        if k is None:
            continue
        ics = []
        for t in months:
            sel = (keys[:, t] == k)
            a, b = pred[sel, t], sev[sel, t]
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() >= min_n:
                from scipy import stats as st
                ics.append(st.spearmanr(a[ok], b[ok])[0])
            else:
                ics.append(np.nan)
        m, tt, n = S.newey_west_t(np.array(ics), lags=2)
        rows.append(dict(group=k, ic=m, t=tt, n=n, p=S.p_one_sided(tt, n)))
    return pd.DataFrame(rows)


def significant_groups(gtab: pd.DataFrame, conf: float = 0.95, use_bh: bool = True) -> dict:
    """Peer groups where the Second Opinion ranks significantly better than chance (one-sided). With use_bh the
    Benjamini-Hochberg procedure controls false discoveries across groups at the same level."""
    p = gtab["p"].to_numpy(float)
    alpha = 1 - conf
    pos = gtab["ic"].to_numpy(float) > 0
    ok = (S.bh(p, alpha) if use_bh else (p <= alpha)) & pos
    return dict(zip(gtab["group"], ok))


def quintile_within(values: np.ndarray, keys: np.ndarray, higher_is_better=True) -> np.ndarray:
    """Fifth of the peer group at each month: 1 = weakest .. 5 = strongest."""
    from .peers import percentiles
    p = percentiles(values, keys, higher_is_better)
    return np.where(np.isnan(p), np.nan, np.minimum(5, np.floor(p * 5) + 1))


def confidence(q_score: np.ndarray, q_ml: np.ndarray, keys: np.ndarray, sig: dict) -> np.ndarray:
    N, T = q_score.shape
    out = np.full((N, T), None, object)
    sigm = np.vectorize(lambda k: bool(sig.get(k, False)) if k is not None else False, otypes=[bool])(keys)
    d = q_ml - q_score                       # negative: Second Opinion more worried
    ok = ~np.isnan(q_score) & ~np.isnan(q_ml)
    out = np.where(ok & ~sigm, "Scorecard only", out)
    out = np.where(ok & sigm & (d == 0), "High", out)
    out = np.where(ok & sigm & (np.abs(d) == 1), "Medium", out)
    out = np.where(ok & sigm & (d <= -2), "Low", out)
    out = np.where(ok & sigm & (d >= 2), "Medium", out)
    return out


def drivers_for(model, names, x_row: np.ndarray, fills: dict, top=3) -> list:
    """The inputs that most moved this firm's prediction (LightGBM: SHAP contributions; linear: coefficient x deviation)."""
    x = x_row.copy()
    for i in range(len(x)):
        if np.isnan(x[i]):
            x[i] = fills[i]
    if hasattr(model, "booster_"):
        contrib = model.booster_.predict(x[None, :], pred_contrib=True)[0][:-1]
    elif hasattr(model, "coef_"):
        base = np.array([fills[i] for i in range(len(x))])
        contrib = np.asarray(model.coef_) * (x - base)
    else:
        return []
    o = np.argsort(contrib)[:top]
    return [(names[i], float(contrib[i])) for i in o]
