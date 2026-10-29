"""Builds everything the app needs from a seed: the synthetic database, measures, Genius Scores, Grit, peers, calibration, the Second
Opinion and the validation results. Deterministic: the same seed gives the same numbers every run."""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibrate as CAL
from . import config as C
from . import grit as G
from . import measures as MS
from . import peers as PR
from . import scorecard as SC
from . import second_opinion as SO
from . import stats as S
from . import synth
from .policy import Policy

ART = Path(__file__).resolve().parent.parent / "artifacts"
FIRST = ["Agarwal", "Bansal", "Chopra", "Desai", "Gupta", "Iyer", "Jain", "Kapoor", "Kulkarni", "Menon", "Mehta", "Nair", "Patel", "Pillai",
         "Rao", "Reddy", "Saxena", "Shah", "Sharma", "Singh", "Srinivasan", "Thakur", "Verma", "Yadav", "Bhat", "Chauhan", "Das", "Ghosh",
         "Joshi", "Khanna", "Malhotra", "Mishra", "Pandey", "Rathore", "Sethi", "Tiwari", "Varghese", "Kamath", "Hegde", "Shetty"]
SUFFIX = {"Trading · FMCG": ["Distributors", "Agencies", "Trading Co", "Sales Corporation"],
          "Trading · building materials": ["Hardware", "Building Supplies", "Cement Agencies", "Steel Traders"],
          "Trading · pharma": ["Pharma Distributors", "Medical Agencies", "Healthcare Traders"],
          "Trading · electronics": ["Electronics", "Electricals", "Digital Traders"],
          "Manufacturing · light engineering": ["Engineering Works", "Precision Components", "Fabricators"],
          "Manufacturing · textiles": ["Textiles", "Fabrics", "Garments"],
          "Manufacturing · food processing": ["Foods", "Agro Products", "Food Industries"],
          "Manufacturing · plastics": ["Polymers", "Plastics", "Packaging"],
          "Services · IT and ITeS": ["Infotech", "Software Services", "Digital Solutions"],
          "Services · logistics": ["Logistics", "Transport Co", "Freight Carriers"]}


@dataclass
class DB:
    P: object
    d: dict
    meas: dict
    ratings: dict
    gs: dict
    band: np.ndarray
    keys: np.ndarray
    level: np.ndarray
    pct: dict
    grit: dict
    F: np.ndarray
    F_table: dict
    cal: object
    regime_p: np.ndarray
    hmm: dict
    X: dict
    names: list
    so: dict
    val: dict
    firm_names: np.ndarray
    log: list = field(default_factory=list)
    learned_model: object = None

    @property
    def months(self):
        return self.P.months

    @property
    def T(self):
        return len(self.P.months)


def _names2(firms, seed):
    rng = np.random.default_rng(seed)
    out, seen = [], {}
    for s in firms["sector"]:
        nm = f"{FIRST[rng.integers(len(FIRST))]} {SUFFIX[s][rng.integers(len(SUFFIX[s]))]}"
        seen[nm] = seen.get(nm, 0) + 1
        out.append(nm if seen[nm] == 1 else f"{nm} ({seen[nm]})")
    return np.array(out, dtype=object)


def regime_series(f: dict, alive: np.ndarray) -> np.ndarray:
    """Peer-wide signal for the regime model: median across firms of the 3-month change in sales (log)."""
    from .measures import roll_sum, lag
    r3 = roll_sum(f["rev"], 3)
    g = np.log(np.where(r3 > 0, r3, np.nan) / np.where(lag(r3, 3) > 0, lag(r3, 3), np.nan))
    g = np.where(alive, g, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmedian(g, axis=0)


def build(n: int = 5000, seed: int = 20261028, log=print, policy: Policy | None = None) -> DB:
    t0 = time.time()
    msgs = []

    def say(m):
        msgs.append(f"{time.time() - t0:6.0f}s  {m}")
        log(msgs[-1])

    P = synth.simulate(n, seed)
    say(f"synthetic database: {n:,} firms x {P.T} months")
    fams = P.firms["family"].values
    d = MS.derived(P.f)
    meas = MS.compute(P.f, P.firms)
    ratings = MS.rate_all(meas, fams)
    gs = SC.genius_score(ratings, meas)
    say("54 measures and Genius Scores")
    band = PR.bands(P.f, P.months)
    keys, level = PR.group_keys(P.firms["sector"].values, band, P.alive)
    pct = {m: PR.percentiles(meas[m].astype(float), keys, C.M[m]["dir"] == "H") for m in C.MEASURES}
    pct["score"] = PR.percentiles(gs["score"].astype(float), keys, True)
    say("peer groups (sector x government MSME band) and percentiles")
    cal_months = np.array([p.month for p in P.months])
    sig = G.signals(P.f, d)
    prep = G.prepare(sig, cal_months)
    F, F_table = G.market_factor(prep, keys)
    stress = G.stress_survival(P.f, d, meas)
    cap = G.capacity(ratings, stress)
    integ = np.nanmean(np.stack([P.firms[c].values for c in ("dq_bal", "dq_cls", "dq_party", "dq_due")]), 0)
    grit = G.compute(sig, cal_months, P.alive, integ, cap, F=F, prep=prep, keep_eps=True)
    grit["stress_test"] = stress.astype(np.float32)
    grit["zc"] = G.raw_composite(prep).astype(np.float32)
    pct["grit"] = PR.percentiles(grit["score"].astype(float), keys, True)
    say("Grit recovery evidence for every firm and month")
    from . import credit_model
    model = credit_model.train(ratings, grit, P.y["default12"], P.alive & gs["published"], log=say)
    gs = model.apply(gs, ratings, grit)
    pct["score"] = PR.percentiles(gs["score"].astype(float), keys, True)
    yr = C.year_of(np.arange(P.T))
    train = (yr >= C.TRAIN_YEARS[0]) & (yr <= C.TRAIN_YEARS[1])
    cal = CAL.fit(gs["score"].astype(float), P.y["stress90"], P.y["default12"], train,
                  default_months_mask=train & (np.arange(P.T) <= 47),
                  stress_months_mask=train & (np.arange(P.T) <= 56))
    say(f"calibration on years 1-5: Safe from {cal.cutoffs['Safe']:.0f}, Moderate from {cal.cutoffs['Moderate']:.0f}, Watch from {cal.cutoffs['Watch']:.0f}")
    rs = regime_series(P.f, P.alive)
    hmm = S.hmm_fit(rs[train & ~np.isnan(rs)])
    regime_p = S.hmm_filter(hmm, rs)
    hist = np.cumsum(~np.isnan(P.f["rev"]), axis=1)
    dq = integ
    X, names = SO.features(pct, {**P.f, "ocf1": d["ocf1"]}, grit, P.firms, band, regime_p, F, grit["zc"], hist, dq)
    y = SO.target(P.y["severity"], P.alive)
    eligible = P.alive & ~np.isnan(gs["score"]) & (hist >= 12)
    say(f"Second Opinion inputs: {len(names)} features")
    valid_m = range(60, 93)          # labels look 3 months ahead, so selection stops before they reach the test years
    test_m = range(96, P.T)
    wf = SO.walk_forward(X, names, y, eligible, valid_m, 59, SO.MODELS, log=lambda m: None)
    sev = P.y["severity"]
    sel = {m: SO.ic_table(wf["preds"][m], sev, valid_m) for m in SO.MODELS}
    winner = max(SO.MODELS, key=lambda m: -np.inf if np.isnan(sel[m]["mean"]) else sel[m]["mean"])
    say(f"model choice on years 6-8: winner {winner} (IC {sel[winner]['mean']:.3f})")
    wf_test = SO.walk_forward(X, names, y, eligible, test_m, 95, [winner], log=lambda m: None)
    pred = wf["preds"][winner].copy()
    pred[:, 96:] = wf_test["preds"][winner][:, 96:]
    score_f = gs["score"].astype(float)
    res = dict(selection=sel, winner=winner, pred=pred, model=wf_test["fitted"][winner], fills=wf_test["fills"],
               importance=SO.importance(wf_test["fitted"][winner], names))
    res["test_score"] = SO.ic_table(score_f, sev, test_m)
    res["test_ml"] = SO.ic_table(pred, sev, test_m)
    res["test_diff"] = _diff(pred, score_f, sev, test_m)
    res["valid_score"] = SO.ic_table(score_f, sev, valid_m)
    res["group_valid"] = SO.group_ic(pred, sev, keys, valid_m)
    res["group_test"] = SO.group_ic(pred, sev, keys, test_m)
    res["q_score"] = SO.quintile_within(score_f, keys, True).astype(np.float32)
    res["q_ml"] = SO.quintile_within(pred.astype(float), keys, True).astype(np.float32)
    say("Second Opinion tested on years 9-10")
    pol = policy or Policy()
    pol.calibrated_pd = {t: round(100 * cal.tier_pd[t], 1) if not np.isnan(cal.tier_pd[t]) else np.nan for t in C.TIERS}
    db = DB(P, d, meas, ratings, gs, band, keys, level, pct, grit, F, F_table, cal, regime_p, hmm, X, names, res, {}, _names2(P.firms, 11), msgs)
    db.learned_model = model
    from . import validation as V
    db.val = V.run(db, pol)
    say("validation tables")
    slim(db)
    return db


def slim(db: DB) -> DB:
    """Keep what the app needs and store it compactly: facts as float32, inputs and percentiles of the latest month only."""
    T = db.T
    db.P.f = {k: v.astype(np.float32) for k, v in db.P.f.items()}
    db.d = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v) for k, v in db.d.items()}
    db.X = {k: v[:, T - 1].astype(np.float32) for k, v in db.X.items()}
    db.ratings = {k: v[:, T - 13:].astype(np.float32) for k, v in db.ratings.items()}
    db.pct = {k: (v if k in ("score", "grit") else v[:, T - 1:]) for k, v in db.pct.items()}
    for k in ("cover", "shock_path", "ev_stress"):
        db.P.meta.pop(k, None)
    db.grit.pop("prep", None)
    return db


def _diff(pred, score, sev, months):
    a = S.spearman_by_month(pred, sev, list(months))
    b = S.spearman_by_month(score, sev, list(months))
    m, t, n = S.newey_west_t(a - b, lags=2)
    return dict(series=a - b, mean=m, t=t, n=n, p=S.p_one_sided(t, n))


def save(db: DB, path: Path | None = None):
    path = Path(path or ART / "genius_db.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(db, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(path: Path | None = None) -> DB:
    with open(Path(path or ART / "genius_db.pkl"), "rb") as fh:
        return pickle.load(fh)
