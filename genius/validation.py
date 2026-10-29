"""Validation room: the method test on synthetic data. Pass marks are fixed here before anything is run.

Scorecard   : ranks risk (AUC for 90-day stress >= 0.70 in years 9-10); tiers in order every test year; tier stress rates inside their bands
              (with a 90% interval); IC with Newey-West t.
Grit        : adds information (with Grit, the 12-month default model ranks better than without, on the test years); finds true resilience
              (rank correlation with the hidden trait > 0); recovery after planted shocks.
Lending     : at the same approval rate, Genius loses less than a naive rule (approve the largest firms with positive net worth).
Second Opinion: IC, Newey-West t and one-sided p for the Genius Score, the Second Opinion and the difference; peer-group significance;
              agreement and outcomes by confidence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import stats as S
from .calibrate import wilson

PASS = dict(auc_stress=0.70)


def firm_facts(db, i: int, t: int) -> dict:
    """The inputs the policy needs for one firm at one month."""
    f, d = db.P.f, db.d
    from .measures import roll_sum, roll_mean
    def r12(x):
        return float(np.nansum(x[i, max(0, t - 11):t + 1]))
    return dict(cfads12=r12(d["ocf1"]) + r12(f["interest"]) - r12(f["capex"]) - r12(f["drawings"]),
                annual_sales=r12(f["rev"]), recv_young=float(f["recv_young"][i, t]), coll_month=float(np.nanmean(f["collections"][i, max(0, t - 2):t + 1])),
                std=float(f["std"][i, t]), fixed_month=float(np.nanmean((f["fixed"] + f["interest"])[i, max(0, t - 2):t + 1])),
                ebitda12=r12(d["ebitda"]), debt_service12=r12(f["interest"]) + r12(f["principal"]), total_liab=float(d["tl_all"][i, t]),
                equity=float(d["equity"][i, t]), invoices_eligible=float(f["recv_eligible"][i, t]))


def run(db, policy) -> dict:
    P, gs, cal = db.P, db.gs, db.cal
    T = P.T
    yr = C.year_of(np.arange(T))
    score = gs["score"].astype(float)
    s90, d12, sev = P.y["stress90"], P.y["default12"], P.y["severity"]
    out = {}
    split = {"Years 1-5 (calibrate)": (yr <= 5), "Years 6-8 (check)": (yr >= 6) & (yr <= 8), "Years 9-10 (test)": yr >= 9}
    # ---------------------------------------------------------------- scorecard ranking and calibration
    ic = {k: S.newey_west_t(S.spearman_by_month(score, sev, list(np.where(m)[0])), 2) for k, m in split.items()}
    out["ic"] = {k: dict(mean=v[0], t=v[1], n=v[2], p=S.p_one_sided(v[1], v[2])) for k, v in ic.items()}
    test = split["Years 9-10 (test)"]
    sc_t, y_t, d_t = score[:, test].ravel(), s90[:, test].ravel(), d12[:, test].ravel()
    out["auc_stress"] = S.auc(-sc_t, y_t); out["ks_stress"] = S.ks(-sc_t, y_t)
    out["auc_default"] = S.auc(-sc_t, d_t); out["ks_default"] = S.ks(-sc_t, d_t)
    out["brier_stress"] = S.brier(cal.stress(sc_t), y_t)
    q = np.full(score.shape, np.nan)
    for t in range(T):
        v = score[:, t]; ok = ~np.isnan(v)
        if ok.sum() > 50:
            q[ok, t] = pd.qcut(pd.Series(v[ok]).rank(method="first"), 5, labels=False) + 1
    qt, yt = q[:, test].ravel(), s90[:, test].ravel()
    out["quintile_stress"] = [float(np.nanmean(yt[qt == k])) for k in range(1, 6)]
    tiers = cal.tier(score)
    rows = []
    for name, m in split.items():
        tt = tiers[:, m].ravel(); ys = s90[:, m].ravel(); yd = d12[:, m].ravel()
        for tier in C.TIERS:
            sel = (tt == tier) & ~np.isnan(ys)
            k, n = float(np.nansum(ys[sel])), int(sel.sum())
            lo, hi = wilson(k, n)
            seld = (tt == tier) & ~np.isnan(yd)
            rows.append(dict(period=name, tier=tier, firm_months=n, stress_rate=k / n if n else np.nan, lo=lo, hi=hi,
                             default_rate=float(np.nanmean(yd[seld])) if seld.any() else np.nan))
    out["tiers"] = pd.DataFrame(rows)
    order_ok = []
    for y in (9, 10):
        m = yr == y
        tt = tiers[:, m].ravel(); ys = s90[:, m].ravel()
        rates = [np.nanmean(ys[(tt == t_) & ~np.isnan(ys)]) if ((tt == t_) & ~np.isnan(ys)).any() else np.nan for t_ in C.TIERS]
        r_ = [r for r in rates if not np.isnan(r)]
        order_ok.append(all(np.diff(r_) > 0))
    out["tiers_in_order"] = all(order_ok)
    bands = dict(C.TIER_STRESS_BANDS)
    lows = {"Safe": 0, "Moderate": .05, "Watch": .12, "Risky": .25}
    tt = out["tiers"][out["tiers"].period == "Years 9-10 (test)"]
    out["tiers_on_target"] = bool(all((r.lo <= bands[r.tier]) and (r.hi >= lows[r.tier]) for r in tt.itertuples() if r.firm_months > 0))
    # ---------------------------------------------------------------- Grit
    g = db.grit["score"].astype(float)
    r_true = P.firms["r_true"].values
    from scipy import stats as st
    gv = {}
    for label, t in (("end of year 8", 95), ("end of year 10", T - 1)):
        ok = ~np.isnan(g[:, t])
        gv[label] = dict(grit=float(st.spearmanr(g[ok, t], r_true[ok])[0]),
                         demonstrated=float(st.spearmanr(*_ok(db.grit["demonstrated"][:, t], r_true))[0]),
                         capacity=float(st.spearmanr(*_ok(db.grit["capacity"][:, t], r_true))[0]))
    out["grit_truth"] = gv
    # with vs without Grit: logistic model of 12-month default on the training years, ranked on the test years
    from sklearn.linear_model import LogisticRegression
    tr = (yr <= 5); te = test
    def xy(mask, with_grit):
        s = score[:, mask].ravel(); gg = g[:, mask].ravel(); y = d12[:, mask].ravel()
        ok = ~np.isnan(s) & ~np.isnan(y) & ~np.isnan(gg)
        X = np.c_[s[ok]] if not with_grit else np.c_[s[ok], gg[ok]]
        return X / 100, y[ok]
    res = {}
    for wg in (False, True):
        Xa, ya = xy(tr, wg); Xb, yb = xy(te, wg)
        mdl = LogisticRegression().fit(Xa, ya)
        res["with" if wg else "without"] = S.auc(mdl.predict_proba(Xb)[:, 1], yb)
    out["grit_adds"] = res
    # within the same score band, default by Grit band (test years)
    gb = np.vectorize(C.grit_band, otypes=[object])(np.round(g[:, test].ravel(), 1))
    sb = pd.cut(score[:, test].ravel(), [0, 50, 60, 70, 80, 101], labels=["under 50", "50-60", "60-70", "70-80", "80+"])
    dd = pd.DataFrame(dict(score_band=sb, grit=gb, default=d12[:, test].ravel())).dropna()
    out["grit_within_band"] = dd.pivot_table(index="score_band", columns="grit", values="default", aggfunc="mean", observed=False)
    out["grit_within_band_n"] = dd.pivot_table(index="score_band", columns="grit", values="default", aggfunc="count", observed=False)
    # recovery after planted shocks in the test years: months back to 95% of the pre-shock level, against Grit the month before
    hit = P.meta["shock_hit"]; path = P.meta["shock_path"]
    recs = []
    for i, t0 in zip(*np.where(hit[:, 96:T - 12])):
        t0 = t0 + 96
        after = path[i, t0:t0 + 12]
        k = np.argmax(after >= 0.95 * path[i, t0 - 1]) if (after >= 0.95 * path[i, t0 - 1]).any() else 12
        recs.append((g[i, t0 - 1], k, r_true[i]))
    rec = pd.DataFrame(recs, columns=["grit", "months", "r"]).dropna()
    out["grit_recovery"] = dict(n=len(rec), spearman=float(st.spearmanr(rec.grit, -rec.months)[0]) if len(rec) > 10 else np.nan,
                                by_band=rec.assign(band=rec.grit.map(C.grit_band)).groupby("band").months.mean().to_dict())
    # ML Grit: how much of resilience could the same inputs find if a model learned it directly (half the firms train, half test)
    out["grit_ml"] = _grit_oracle(db, 95)
    # ---------------------------------------------------------------- lending simulation
    out["lending"] = lending(db, policy)
    # ---------------------------------------------------------------- Second Opinion
    so = db.so
    out["so"] = dict(winner=so["winner"], selection={m: dict(mean=v["mean"], t=v["t"], p=v["p"]) for m, v in so["selection"].items()},
                     test_score=_strip(so["test_score"]), test_ml=_strip(so["test_ml"]), test_diff=_strip(so["test_diff"]),
                     importance=so["importance"])
    return out


def _ok(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    return a[ok], b[ok]


def _strip(d):
    return {k: v for k, v in d.items() if k not in ("ic_series", "series")}


def _grit_oracle(db, t):
    from sklearn.linear_model import Ridge
    from scipy import stats as st
    P = db.P
    cols = [db.grit["demonstrated"][:, t], db.grit["capacity"][:, t], db.grit["n"][:, t], db.grit["stress_test"][:, t]] + \
           [db.ratings[m][:, t] for m in C.CAPACITY]
    X = np.c_[tuple(np.nan_to_num(c, nan=np.nanmedian(c)) for c in cols)]
    y = P.firms["r_true"].values
    ok = P.alive[:, t] & ~np.isnan(db.grit["score"][:, t])
    rng = np.random.default_rng(3)
    half = rng.random(len(y)) < 0.5
    a, b = ok & half, ok & ~half
    import lightgbm as lgb
    out = {}
    for name, mdl in (("Ridge", Ridge(alpha=1.0)), ("LightGBM", lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=1))):
        mdl.fit(X[a], y[a])
        out[name] = float(st.spearmanr(mdl.predict(X[b]), y[b])[0])
    out["Rule-based Grit"] = float(st.spearmanr(db.grit["score"][b, t], y[b])[0])
    return out


def lending(db, policy) -> dict:
    """Each quarter-end in the test years: approve, lend the short-term limit for six months, count losses on defaults in the next six months."""
    P, cal = db.P, db.cal
    ev = P.meta["ev_default"] | ~P.alive
    score = db.gs["score"].astype(float)
    tiers = cal.tier(score)
    res = {"Genius": dict(lent=0.0, lost=0.0, income=0.0, n=0, approvals=0, candidates=0),
           "Naive rule": dict(lent=0.0, lost=0.0, income=0.0, n=0, approvals=0, candidates=0)}
    lgd = float(policy.v("lgd_unsecured")) / 100
    for t in range(96, P.T - 6, 3):
        cand = np.where(P.alive[:, t] & ~np.isnan(score[:, t]))[0]
        appr, exp_g, rate_g = [], {}, {}
        for i in cand:
            tier = tiers[i, t]
            ko = {k: bool(v[i, t]) for k, v in db.gs["ko"].items()}
            x = firm_facts(db, i, t)
            lim = policy.limits(x, tier, float(db.grit["score"][i, t]), ko)
            if lim["short"]["available"] and lim["short"]["limit"] > 0:
                appr.append(i); exp_g[i] = lim["short"]["limit"]
                rate_g[i] = policy.rate(tier, pd_score=100 * float(cal.default(score[i, t])))["rate"]
        k = len(appr)
        sales = np.array([firm_facts(db, i, t)["annual_sales"] for i in cand])
        eq = db.d["equity"][cand, t]
        naive_order = cand[np.argsort(-(sales * (eq > 0)))][:k]
        mod = policy.rate("Moderate")["rate"]
        for name, ids, expo, rates in (("Genius", appr, exp_g, rate_g),
                                       ("Naive rule", list(naive_order), None, None)):
            for i in ids:
                e = expo[i] if expo else 0.15 * firm_facts(db, i, t)["annual_sales"] * 0.8
                r = rates[i] if rates else mod
                bad = ev[i, t + 1:t + 7].any()
                res[name]["lent"] += e
                if bad:
                    res[name]["lost"] += e * lgd
                else:
                    cof = float(policy.v("repo_rate")) + float(policy.v("funding_spread")) + float(policy.v("operating_cost"))
                    res[name]["income"] += e * (r - cof) / 100 * 0.5
                res[name]["n"] += 1
            res[name]["approvals"] += len(ids); res[name]["candidates"] += len(cand)
    for name, r in res.items():
        r["loss_rate"] = r["lost"] / r["lent"] if r["lent"] else np.nan
        r["net_per_100"] = 100 * (r["income"] - r["lost"]) / r["lent"] if r["lent"] else np.nan
        r["approval_rate"] = r["approvals"] / r["candidates"] if r["candidates"] else np.nan
    return res
