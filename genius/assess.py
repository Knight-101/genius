"""One business, end to end: its books -> measures, Genius Score, tier, peers, Grit, Second Opinion, limits and rates, what-ifs.
Every step calls the same code the synthetic database used, so an uploaded file is judged exactly like the 5,000 firms it is compared with."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import grit as G
from . import measures as MS
from . import peers as PR
from . import scorecard as SC
from . import second_opinion as SO
from .policy import Policy


@dataclass
class Assessment:
    name: str
    sector: str
    family: str
    months: pd.PeriodIndex
    f: dict
    firm: pd.DataFrame
    meas: dict
    ratings: dict
    gs: dict
    band: str
    band_reason: str
    group: str
    group_level: str
    t_db: int
    peers: dict
    grit: dict
    tier: str
    p_stress: float
    p_default: float
    so: dict
    limits: dict
    pricing: dict
    drivers: dict
    inputs: dict
    info: dict = field(default_factory=dict)

    @property
    def score(self):
        return float(self.gs["score"][0, -1]) if not np.isnan(self.gs["score"][0, -1]) else None

    def m(self, mid):
        v = self.meas[mid][0, -1]
        return None if np.isnan(v) else float(v)

    def r(self, mid):
        v = self.ratings[mid][0, -1]
        return None if np.isnan(v) else float(v)


def db_month(db, months: pd.PeriodIndex) -> int:
    last = months[-1]
    if last < db.months[0]:
        return 0
    if last > db.months[-1]:
        return db.T - 1
    return int(db.months.get_loc(last))


def _group_at(db, sector: str, band: str, t: int):
    keys = db.keys[:, t]
    alive = db.P.alive[:, t]
    k1 = f"{sector} | {band}"
    n1 = int(((db.P.firms["sector"].values == sector) & (db.band[:, t] == band) & alive).sum())
    if n1 >= PR.MIN_PEERS:
        members = np.where(alive & (db.P.firms["sector"].values == sector) & (db.band[:, t] == band))[0]
        return k1, "sector and size", members
    m2 = np.where(alive & (db.P.firms["sector"].values == sector))[0]
    if len(m2) >= PR.MIN_PEERS:
        return "sector: " + sector, "sector only (widened)", m2
    m3 = np.where(alive & (db.band[:, t] == band))[0]
    return "size: " + band, "size only (widened)", m3


def pct_vs(values: np.ndarray, x: float, higher=True) -> float:
    v = values[~np.isnan(values)]
    if x is None or np.isnan(x) or len(v) == 0:
        return np.nan
    if not higher:
        v, x = -v, -x
    return float(((v < x).sum() + 0.5 * (v == x).sum()) / len(v))


def run(db, policy: Policy, f: dict, firm: pd.DataFrame, months: pd.PeriodIndex, name: str, sector: str, inputs: dict | None = None,
        info: dict | None = None, k5: bool = False) -> Assessment:
    inputs = dict(inputs or {})
    info = dict(info or {})
    fam = C.family(sector)
    meas = MS.compute(f, firm)
    ratings = MS.rate_all(meas, [fam])
    gs = SC.genius_score(ratings, meas, k5=np.array([k5]))
    # MSME band from the last completed financial year (Udyam category, if typed in, wins)
    to, inv = PR.fy_turnover_and_investment(f, months)
    band = C.msme_band(float(np.nan_to_num(inv[0, -1])), float(np.nan_to_num(to[0, -1])))
    reason = C.msme_reason(float(np.nan_to_num(inv[0, -1])), float(np.nan_to_num(to[0, -1])))
    if inputs.get("udyam") in C.SIZE_LIST:
        band = inputs["udyam"]
        reason = f"Udyam registration says {band.lower()}; that is used in place of the estimate from the books. " + reason
    band_use = band if band in C.SIZE_LIST else "Medium"
    t_db = db_month(db, months)
    gkey, glevel, members = _group_at(db, sector, band_use, t_db)
    # peers at the matching month of the database
    pe = {}
    for mid in C.MEASURES:
        vals = db.meas[mid][members, t_db].astype(float)
        x = meas[mid][0, -1]
        ok = vals[~np.isnan(vals)]
        pe[mid] = dict(median=float(np.median(ok)) if len(ok) else np.nan, p25=float(np.percentile(ok, 25)) if len(ok) else np.nan,
                       p75=float(np.percentile(ok, 75)) if len(ok) else np.nan, n=len(ok), pct=pct_vs(vals, float(x), C.M[mid]["dir"] == "H"))
    db_scores = db.gs["score"][members, t_db].astype(float)
    score = gs["score"][0, -1]
    areas_peer = {a: float(np.nanmedian(db.gs["areas"][a][members, t_db])) for a in C.AREA_LIST}
    areas_pct = {a: pct_vs(db.gs["areas"][a][members, t_db].astype(float), float(gs["areas"][a][0, -1])) for a in C.AREA_LIST}
    tiers_db = db.cal.tier(db_scores)
    mix = {t: float(np.mean(tiers_db[~pd.isna(tiers_db)] == t)) for t in C.TIERS}
    p_s = float(db.cal.stress(score)) if not np.isnan(score) else np.nan
    p_d = float(db.cal.default(score)) if not np.isnan(score) else np.nan
    tier = str(db.cal.tier(np.array([score]))[0]) if not np.isnan(score) else None
    peer_s = db.cal.stress(db_scores[~np.isnan(db_scores)])
    peers = dict(key=gkey, level=glevel, n=len(members), month=str(db.months[t_db]), measures=pe, score_pct=pct_vs(db_scores, float(score)),
                 score_median=float(np.nanmedian(db_scores)), rank=int(np.sum(db_scores > score) + 1) if not np.isnan(score) else None,
                 tier_mix=mix, areas_median=areas_peer, areas_pct=areas_pct, stress_median=float(np.median(peer_s)) if len(peer_s) else np.nan,
                 members=members)
    # Grit (market factor from the database's peer table at the same calendar months)
    d = MS.derived(f)
    sig = G.signals(f, d)
    if "grit_rev" in info:
        sig["sales"] = info["grit_rev"]; sig["rev"] = info["grit_rev"]; sig["rec"] = info["grit_coll"]
    cal_m = np.array([p.month for p in months])
    Fk = db.F_table.get(gkey)
    F = np.full((1, len(months)), np.nan)
    if Fk is not None:
        for j, p in enumerate(months):
            if db.months[0] <= p <= db.months[-1]:
                F[0, j] = Fk[db.months.get_loc(p)]
    stress = G.stress_survival(f, d, meas)
    cap = G.capacity(ratings, stress)
    integ = np.array([float(np.nanmean(firm[["dq_bal", "dq_cls", "dq_party", "dq_due"]].values))])
    alive = ~np.isnan(f["rev"])
    gr = G.compute(sig, cal_m, alive, integ, cap, F=F if not np.all(np.isnan(F)) else None, keep_eps=True)
    gr["stress_test"] = stress; gr["months"] = months
    for e in gr["episodes"] or []:
        for k in ("start", "trough", "end", "judged"):
            if e.get(k) is not None:
                e[k + "_m"] = str(months[e[k]])
    from .resilience import evidence
    recovery = evidence(gr, info.get("report", {}).get("quality_score", 0))
    info["recovery_policy"] = recovery
    grit_now = float(gr["score"][0, -1]) if not np.isnan(gr["score"][0, -1]) else None
    # The same fitted model scores uploads, peers, and what-if scenarios.
    model = getattr(db, "learned_model", None)
    if model is not None:
        gs = model.apply(gs, ratings, gr)
        score = float(gs["score"][0, -1])
        p_s = float(db.cal.stress(score)) if np.isfinite(score) else np.nan
        p_d = float(db.cal.default(score)) if np.isfinite(score) else np.nan
        tier = str(db.cal.tier(np.array([score]))[0]) if np.isfinite(score) else None
        peers["score_pct"] = pct_vs(db_scores, score)
        peers["rank"] = int(np.sum(db_scores > score) + 1) if np.isfinite(score) else None
        info["model"] = model.report
        info["model_weights"] = model.weights()
    # Second Opinion
    so = second_opinion(db, policy, f, meas, firm, months, sector, band_use, gkey, members, t_db, gr, d, score, inputs)
    # policy: limits and pricing
    x = facts_now(f, d)
    ko = {k: bool(v[0, -1]) for k, v in gs["ko"].items()}
    import datetime, os
    pctx = dict(sector=sector, size=band_use, regime="stress" if db.regime_p[t_db] > 0.5 else "calm",
                date=os.environ.get("GENIUS_QUOTE_DATE", datetime.date.today().isoformat()))
    info["policy_context"] = pctx
    info["lending_facts"] = x
    lim = policy.limits(x, tier, grit_now, ko, confidence=so.get("confidence"), ctx=pctx, pd_score=100 * p_d) if tier else {}
    pricing = {}
    if tier:
        for name_, months_ in (("short", 3), ("medium", int(policy.v("mt_tenor_months", dict(pctx, tier=tier, tenor="medium", secured="yes"))))):
            pricing[name_] = policy.rate(tier, pctx, pd_score=100 * p_d, tenor_months=months_,
                                         secured=(name_ == "medium"), grit=None, recovery=recovery)
        pricing["invoice"] = policy.rate(tier, pctx, pd_score=100 * p_d, tenor_months=3, invoice=True, recovery=recovery)
        pricing["peer_median"] = policy.rate("Moderate" if peers["score_median"] >= db.cal.cutoffs["Moderate"] else "Watch", pctx, pd_score=100 * float(db.cal.default(peers["score_median"])))
    rat_now = {m: float(ratings[m][0, -1]) for m in C.MEASURES}
    meas_now = {m: float(meas[m][0, -1]) for m in C.MEASURES}
    drv = SC.drivers(rat_now, meas_now)
    if model is not None:
        from .credit_model import FINANCIAL
        weights = dict(zip(FINANCIAL, model.theta[1:1 + len(FINANCIAL)]))
        for row in drv["all"]:
            row["impact"] = weights.get(row["id"], 0.0) * (row["rating"] - 60)
        drv["all"].sort(key=lambda row: row["impact"])
        drv["negative"] = [r for r in drv["all"] if r["impact"] < 0][:2]
        drv["positive"] = [r for r in reversed(drv["all"]) if r["impact"] > 0][:2]
    return Assessment(name, sector, fam, months, f, firm, meas, ratings, gs, band, reason, gkey, glevel, t_db, peers, gr, tier, p_s, p_d, so, lim,
                      pricing, drv, inputs, info)


def facts_now(f: dict, d: dict, t: int = -1) -> dict:
    s = lambda k, n=12: float(np.nansum(f[k][0, max(0, f[k].shape[1] + t - n + 1 if t < 0 else t - n + 1):(f[k].shape[1] + t + 1) if t < 0 else t + 1]))
    T = f["rev"].shape[1]
    tt = T + t if t < 0 else t
    sl = slice(max(0, tt - 11), tt + 1)
    sl3 = slice(max(0, tt - 2), tt + 1)
    cfads = float(np.nansum(d["ocf1"][0, sl]) + np.nansum(f["interest"][0, sl])
                  - np.nansum(f["capex"][0, sl]) - np.nansum(f["drawings"][0, sl]))
    return dict(cfads12=cfads, cash_cost_month=float(np.nanmean((f["cogs"] + f["fixed"] + f["var_opex"] + f["drawings"] + f["capex"])[0, sl3])), annual_sales=float(np.nansum(f["rev"][0, sl])), recv_young=float(f["recv_young"][0, tt]),
                coll_month=float(np.nanmean(f["collections"][0, sl3])), std=float(f["std"][0, tt]),
                fixed_month=float(np.nanmean((f["fixed"] + f["interest"])[0, sl3])), ebitda12=float(np.nansum(d["ebitda"][0, sl])),
                debt_service12=float(np.nansum(f["interest"][0, sl]) + np.nansum(f["principal"][0, sl])), total_liab=float(d["tl_all"][0, tt]),
                equity=float(d["equity"][0, tt]), invoices_eligible=float(f["recv_eligible"][0, tt]), cash=float(f["cash"][0, tt]),
                cc_limit=float(f["cc_limit"][0, tt]) if not np.isnan(f["cc_limit"][0, tt]) else None)


def second_opinion(db, policy, f, meas, firm, months, sector, band, gkey, members, t_db, gr, d, score, inputs) -> dict:
    """The Second Opinion for one business at its latest month, ranked against its peers in the database at the same month."""
    T = len(months)
    # percentiles of the firm against its peers at the matching database months (now, 3 and 12 months back)
    def pct_at(mid, j):
        tdb = max(0, t_db - (T - 1 - j))
        vals = db.meas[mid][members, tdb].astype(float)
        return pct_vs(vals, float(meas[mid][0, j]), C.M[mid]["dir"] == "H")
    pct = {}
    for mid in C.MEASURES:
        arr = np.full((1, T), np.nan, np.float32)
        for j in (T - 1, T - 4, T - 13):
            if j >= 0:
                arr[0, j] = pct_at(mid, j)
        pct[mid] = arr
    zc = G.raw_composite(gr["prep"]).astype(np.float32)
    hist = np.cumsum(~np.isnan(f["rev"]), axis=1)
    bandarr = np.full((1, T), band, object)
    reg = np.full(T, db.regime_p[t_db])
    Fk = db.F_table.get(gkey)
    F = np.full((1, T), np.nan)
    if Fk is not None:
        for j, p in enumerate(months):
            if db.months[0] <= p <= db.months[-1]:
                F[0, j] = Fk[db.months.get_loc(p)]
    dq = np.array([float(np.nanmean(firm[["dq_bal", "dq_cls", "dq_party", "dq_due"]].values))])
    X, names = SO.features(pct, {**f, "ocf1": d["ocf1"]}, gr, pd.DataFrame(dict(sector=[sector], family=[C.family(sector)])), bandarr, reg, F, zc, hist, dq)
    row = np.array([X[n][0, -1] if n in X else np.nan for n in db.names], np.float32)
    fills = db.so["fills"]
    x = row.copy()
    for i in range(len(x)):
        if np.isnan(x[i]):
            x[i] = fills[i]
    model = db.so["model"]
    pred = float(model.predict(x[None, :])[0])
    peer_pred = db.so["pred"][members, t_db].astype(float)
    peer_score = db.gs["score"][members, t_db].astype(float)
    q = lambda p: None if np.isnan(p) else int(min(5, np.floor(p * 5) + 1))
    q_ml = q(pct_vs(peer_pred, pred))
    q_sc = q(pct_vs(peer_score, float(score))) if not np.isnan(score) else None
    gtab = db.so["group_test"]
    sig = SO.significant_groups(gtab, float(policy.v("so_confidence")), bool(policy.v("so_bh")))
    row_g = gtab[gtab["group"] == gkey]
    significant = bool(sig.get(gkey, False))
    if q_ml is None or q_sc is None:
        conf = None
    elif not significant:
        conf = "Scorecard only"
    else:
        dd = q_ml - q_sc
        conf = "High" if dd == 0 else ("Medium" if abs(dd) == 1 else ("Low" if dd <= -2 else "Medium"))
    drivers = SO.drivers_for(model, db.names, row, fills, top=3)
    up = SO.drivers_for(model, db.names, row, fills, top=len(db.names))[::-1][:3]
    return dict(pred=pred, q_ml=q_ml, q_score=q_sc, significant=significant, confidence=conf,
                group_ic=None if row_g.empty else float(row_g["ic"].iloc[0]), group_t=None if row_g.empty else float(row_g["t"].iloc[0]),
                group_p=None if row_g.empty else float(row_g["p"].iloc[0]), drivers_down=drivers, drivers_up=up, model=db.so["winner"],
                agree_models=None)


# ---------------------------------------------------------------------------------------------------------------- what-ifs
def what_if(db, policy, a: Assessment, collect: float = 0.0, top_cust_to: float | None = None, drawings_share: float | None = None) -> dict:
    """Re-score the business with changes applied to its latest month, through the same engine."""
    f = {k: v.copy() for k, v in a.f.items()}
    if collect > 0:
        x = min(collect, float(f["ar_overdue"][0, -1]))
        f["recv"][0, -1] -= x; f["cash"][0, -1] += x; f["ar_overdue"][0, -1] -= x
        f["ar_90"][0, -1] = max(0.0, f["ar_90"][0, -1] - x)
        f["collections"][0, -1] += x; f["ar_paid"][0, -1] += x
        f["recv_young"][0, -1] = min(f["recv_young"][0, -1], f["recv"][0, -1])
        f["cash_low"][0, -1] += 0.5 * x
    if top_cust_to is not None and not np.isnan(f["top_cust"][0, -1]) and f["top_cust"][0, -1] > top_cust_to:
        old = f["top_cust"][0, -1]
        f["top_cust"][0, -1] = top_cust_to
        f["hhi_cust"][0, -1] = max(top_cust_to ** 2, f["hhi_cust"][0, -1] - old ** 2 + top_cust_to ** 2)
    if drawings_share is not None:
        d = MS.derived(f)
        ocf12 = float(MS.ocf(f, d, 12)[0, -1])
        dr12 = float(np.nansum(f["drawings"][0, -12:]))
        target = max(0.0, drawings_share * max(ocf12, 0.0))
        if dr12 > target and dr12 > 0:
            k = target / dr12
            saved = dr12 - target
            f["drawings"][0, -12:] *= k
            f["cash"][0, -12:] += np.linspace(saved / 12, saved, 12)
    revised = run(db, policy, f, a.firm, a.months, a.name, a.sector, a.inputs, a.info,
                  k5=bool(a.gs["ko"]["K5"][0, -1]))
    return dict(score=revised.score, tier=revised.tier, rate=revised.pricing.get("short", {}).get("rate"),
                areas={k: float(v[0, -1]) for k, v in revised.gs["areas"].items()},
                meas=revised.meas, ratings=revised.ratings)


def slowest_payers(a, b_books, k: int = 4) -> pd.DataFrame:
    """Overdue receivables by customer at the latest month, largest first (from the books)."""
    ag = b_books.ageing("receivable")
    od = ag[ag["days_past_due"] > 0]
    g = od.groupby("party").agg(overdue=("open", "sum"), invoices=("open", "size"), oldest=("days_past_due", "max")).sort_values("overdue", ascending=False)
    return g.reset_index()
