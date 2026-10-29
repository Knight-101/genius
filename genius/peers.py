"""Peer groups and peer statistics.

Group = sector x government MSME band, recomputed every month from the last completed financial year (turnover less exports;
plant and equipment at written-down value). At least 30 other firms are needed; otherwise the group widens to the sector, then to the
MSME band, and the screen says so. A firm is always left out of its own group (leave-one-out). Statistics are medians and quartiles;
a firm's percentile = (peers below + half the peers equal) / peers, so 0.80 means "better than 80% of peers" (direction-aware)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

MIN_PEERS = 30


def fy_turnover_and_investment(f: dict, months: pd.PeriodIndex):
    """Turnover (exports excluded) of the last completed April-March year, and plant at that year end; trailing 12 months before the
    first full year. Both N x T."""
    N, T = f["rev"].shape
    dom = f["rev"] - np.nan_to_num(f["exports"])
    to = np.full((N, T), np.nan); inv = np.full((N, T), np.nan)
    mar = np.array([p.month == 3 for p in months])
    last_to = np.full(N, np.nan); last_inv = np.full(N, np.nan)
    from .measures import roll_sum
    ttm = roll_sum(dom, 12)
    for t in range(T):
        if mar[t]:
            last_to = np.where(~np.isnan(ttm[:, t]), ttm[:, t], last_to)
            last_inv = np.where(~np.isnan(f["plant"][:, t]), f["plant"][:, t], last_inv)
        to[:, t] = np.where(np.isnan(last_to), ttm[:, t], last_to)
        inv[:, t] = np.where(np.isnan(last_inv), f["plant"][:, t], last_inv)
    return to, inv


def bands(f: dict, months) -> np.ndarray:
    to, inv = fy_turnover_and_investment(f, months)
    b = C.msme_band_vec(np.nan_to_num(inv), np.nan_to_num(to))
    return np.where(np.isnan(to), None, b)


def group_keys(sectors: np.ndarray, band: np.ndarray, alive: np.ndarray):
    """Per firm-month: the group used (sector x band, else sector, else band) and its label."""
    N, T = band.shape
    sec = np.broadcast_to(np.asarray(sectors, object)[:, None], (N, T))
    k1 = np.where(alive & (band != None), sec + " | " + band.astype(str), None)   # noqa: E711
    key = np.full((N, T), None, object)
    level = np.full((N, T), "", object)
    for t in range(T):
        col1 = pd.Series(k1[:, t])
        n1 = col1.map(col1.value_counts())
        ok1 = (n1 - 1 >= MIN_PEERS) & col1.notna()
        s_col = pd.Series(np.where(alive[:, t], sec[:, t], None))
        n2 = s_col.map(s_col.value_counts())
        b_col = pd.Series(np.where(alive[:, t] & (band[:, t] != None), band[:, t], None))   # noqa: E711
        n3 = b_col.map(b_col.value_counts())
        key[:, t] = np.where(ok1, col1, np.where((n2 - 1 >= MIN_PEERS) & s_col.notna(), "sector: " + s_col.astype(str),
                                                 np.where(b_col.notna(), "size: " + b_col.astype(str), None)))
        level[:, t] = np.where(ok1, "sector and size", np.where((n2 - 1 >= MIN_PEERS) & s_col.notna(), "sector only (widened)", "size only (widened)"))
    return key, level


def percentiles(values: np.ndarray, keys: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Leave-one-out percentile of each firm within its group at each month."""
    N, T = values.shape
    out = np.full((N, T), np.nan, np.float32)
    v = values if higher_is_better else -values
    for t in range(T):
        df = pd.DataFrame(dict(k=keys[:, t], v=v[:, t]))
        ok = df["k"].notna() & df["v"].notna()
        if not ok.any():
            continue
        sub = df[ok]
        r = sub.groupby("k")["v"].rank(method="average")
        n = sub.groupby("k")["v"].transform("count")
        out[sub.index.to_numpy(), t] = np.where(n > 1, (r - 1) / (n - 1), np.nan)
    return out


def group_stats(values: np.ndarray, keys: np.ndarray) -> dict:
    """{(key, t): (median, p25, p75, n)} for one measure."""
    N, T = values.shape
    out = {}
    for t in range(T):
        df = pd.DataFrame(dict(k=keys[:, t], v=values[:, t])).dropna()
        if df.empty:
            continue
        g = df.groupby("k")["v"]
        q = g.quantile([0.25, 0.5, 0.75]).unstack()
        cnt = g.count()
        for k in q.index:
            out[(k, t)] = (float(q.loc[k, 0.5]), float(q.loc[k, 0.25]), float(q.loc[k, 0.75]), int(cnt[k]))
    return out


def loo_median(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """For each firm-month, the median of its group's values with the firm itself left out."""
    N, T = values.shape
    out = np.full((N, T), np.nan)
    for t in range(T):
        df = pd.DataFrame(dict(k=keys[:, t], v=values[:, t]))
        df = df[df["k"].notna() & df["v"].notna()]
        for k, g in df.groupby("k"):
            idx = g.index.to_numpy()
            vals = g["v"].to_numpy()
            order = np.argsort(vals, kind="stable")
            s = vals[order]
            n = len(s)
            if n < 2:
                continue
            rank = np.empty(n, int); rank[order] = np.arange(n)
            m = n - 1
            def rem(j, r):
                return np.where(j < r, s[np.minimum(j, n - 1)], s[np.minimum(j + 1, n - 1)])
            if m % 2 == 1:
                med = rem(np.full(n, (m - 1) // 2), rank)
            else:
                med = 0.5 * (rem(np.full(n, m // 2 - 1), rank) + rem(np.full(n, m // 2), rank))
            out[idx, t] = med
    return out
