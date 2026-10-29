"""Turning a Genius Score into a probability and a tier, fitted on the training years only (years 1-5).

  90-day stress probability: isotonic (never increasing in the score) fit of the stress rate on 5-point score bands.
  12-month default probability: the same, for default.
  Tier: Safe if the calibrated 90-day stress rate is under 5%, Moderate under 12%, Watch under 25%, else Risky.
  The default probability of each tier (used for pricing) is the training-years default rate of firms in that tier.
Years 6-8 check the cut-offs; years 9-10 are left untouched for the test."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from . import config as C

BAND = 5


@dataclass
class Calibration:
    centers: np.ndarray
    p_stress: np.ndarray
    p_default: np.ndarray
    counts: np.ndarray
    cutoffs: dict                 # tier -> lowest score in that tier
    tier_pd: dict                 # tier -> 12-month default rate (training years)
    tier_stress: dict             # tier -> 90-day stress rate (training years)

    def stress(self, score):
        return np.interp(score, self.centers, self.p_stress)

    def default(self, score):
        return np.interp(score, self.centers, self.p_default)

    def tier(self, score):
        s = np.asarray(score, float)
        out = np.full(s.shape, None, object)
        out = np.where(s >= self.cutoffs["Safe"], "Safe", np.where(s >= self.cutoffs["Moderate"], "Moderate",
                       np.where(s >= self.cutoffs["Watch"], "Watch", "Risky")))
        return np.where(np.isnan(s), None, out)


def fit(score: np.ndarray, stress: np.ndarray, default: np.ndarray, months_mask: np.ndarray,
        default_months_mask=None, stress_months_mask=None) -> Calibration:
    dm = months_mask if default_months_mask is None else default_months_mask
    sm = months_mask if stress_months_mask is None else stress_months_mask
    s = score[:, sm].ravel(); y = stress[:, sm].ravel(); d = default[:, dm].ravel()
    ok = ~np.isnan(s) & ~np.isnan(y)
    s, y = s[ok], y[ok]
    okd = ~np.isnan(score[:, dm].ravel()) & ~np.isnan(d)
    sd_, dd = score[:, dm].ravel()[okd], d[okd]
    edges = np.arange(0, 100 + BAND, BAND)
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.digitize(s, edges) - 1, 0, len(centers) - 1)
    cnt = np.bincount(idx, minlength=len(centers)).astype(float)
    rate = np.bincount(idx, weights=y, minlength=len(centers)) / np.maximum(cnt, 1)
    idd = np.clip(np.digitize(sd_, edges) - 1, 0, len(centers) - 1)
    cntd = np.bincount(idd, minlength=len(centers)).astype(float)
    rated = np.bincount(idd, weights=dd, minlength=len(centers)) / np.maximum(cntd, 1)
    use = cnt > 0
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(centers[use], rate[use], sample_weight=cnt[use])
    ps = iso.predict(centers)
    used = cntd > 0
    isod = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(centers[used], rated[used], sample_weight=cntd[used])
    pdf = isod.predict(centers)
    cut = {}
    for tier, lim in C.TIER_STRESS_BANDS[:-1]:
        ok_c = np.where(ps < lim)[0]
        cut[tier] = float(edges[ok_c[0]]) if len(ok_c) else 101.0
    cut["Risky"] = 0.0
    cal = Calibration(centers, ps, pdf, cnt, cut, {}, {})
    t_all = cal.tier(s)
    t_d = cal.tier(sd_)
    for tier in C.TIERS:
        m = t_all == tier
        cal.tier_stress[tier] = float(y[m].mean()) if m.any() else float("nan")
        md = t_d == tier
        cal.tier_pd[tier] = float(dd[md].mean()) if md.any() else float("nan")
    return cal


def wilson(k, n, z=1.645):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
