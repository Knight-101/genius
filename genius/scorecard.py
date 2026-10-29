"""Genius Score: measure ratings -> eight area scores -> weighted score -> knock-out caps. Transparent, fixed, the same for every firm.

Rules (all from the specification):
  area score     = mean of the rated measures in the area; an area counts only if at least half its measures are available
  Genius Score   = sum(area weight x area score) / sum(weights of rated areas)
  published only if rated areas carry at least 70% of the weight and both liquidity and debt service are rated
  knock-outs     K1 cash runway critical (CR03 rating < 20)        -> capped at 44
                 K2 debt service cover below 1 (FL06 < 1.0)        -> capped at 59, no medium or long-term credit
                 K3 old receivables (RC08 rating < 20)             -> capped at 59
                 K4 statutory arrears (PY09 rating < 20)           -> capped at 59
                 K5 integrity failure (entries not balancing, balance-sheet gap, round-tripped sales) -> no score
  Under 12 months of history no score is published (every flow measure needs a full year)."""
from __future__ import annotations

import numpy as np

from . import config as C

CAPS = {"K1": 44, "K2": 59, "K3": 59, "K4": 59}
KO_TEXT = {"K1": "Cash runway is critical (under about 15 days)", "K2": "Operating profit does not cover interest and repayments",
           "K3": "Over a fifth of receivables are more than 90 days overdue", "K4": "Statutory dues are more than about 4 months in arrears",
           "K5": "The books fail an integrity check, so no score is published"}


def area_scores(ratings: dict) -> dict:
    out = {}
    for a, mids in C.BY_AREA.items():
        stack = np.stack([ratings[m] for m in mids], axis=0).astype(float)
        n_ok = (~np.isnan(stack)).sum(0)
        with np.errstate(all="ignore"):
            mean = np.nanmean(stack, axis=0)
        out[a] = np.where(n_ok >= np.ceil(len(mids) / 2), mean, np.nan)
    return out


def genius_score(ratings: dict, meas: dict, k5=None) -> dict:
    areas = area_scores(ratings)
    num = 0.0; den = 0.0
    for a, w in C.AREAS.items():
        s = areas[a]
        num = num + np.where(np.isnan(s), 0.0, w * s)
        den = den + np.where(np.isnan(s), 0.0, w)
    raw = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    published = (den >= 70) & ~np.isnan(areas["Liquidity"]) & ~np.isnan(areas["Debt service"])
    ko = {"K1": ratings["CR03"] < 20, "K2": meas["FL06"] < 1.0, "K3": ratings["RC08"] < 20, "K4": ratings["PY09"] < 20}
    score = raw.copy()
    for k, cap in CAPS.items():
        score = np.where(ko[k], np.minimum(score, cap), score)
    k5 = np.zeros(raw.shape, bool) if k5 is None else np.broadcast_to(np.asarray(k5, bool).reshape(-1, 1), raw.shape)
    ko["K5"] = k5
    score = np.where(published & ~k5, score, np.nan)
    return dict(score=score.astype(np.float32), raw=raw.astype(np.float32), areas={a: v.astype(np.float32) for a, v in areas.items()},
                ko={k: np.asarray(v, bool) for k, v in ko.items()}, published=published)


def drivers(ratings_at: dict, meas_at: dict, top: int = 2) -> dict:
    """Top positive and negative drivers for one firm at one month: impact = area weight x (rating - 60) / rated measures in the area."""
    rows = []
    for a, mids in C.BY_AREA.items():
        rated = [m for m in mids if ratings_at.get(m) is not None and not np.isnan(ratings_at[m])]
        for m in rated:
            imp = C.AREAS[a] * (ratings_at[m] - 60) / len(rated)
            rows.append(dict(id=m, name=C.M[m]["name"], area=a, rating=float(ratings_at[m]), value=float(meas_at[m]), impact=float(imp)))
    rows.sort(key=lambda r: r["impact"])
    neg = [r for r in rows if r["impact"] < 0][:top]
    pos = [r for r in reversed(rows) if r["impact"] > 0][:top]
    return dict(positive=pos, negative=neg, all=rows)
