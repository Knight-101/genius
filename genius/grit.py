"""Grit: how a business behaves after a setback. It never marks down a bad patch, only a failure to recover from one.

Grit = w x Demonstrated + (1 - w) x Capacity,   w = n / (n + kappa),   n = setbacks judged so far (kappa about 2, drawn in secret)

Demonstrated recovery (method carried over from version 3, now run for every firm and month in one forward pass):
  0. Seasonality: from 24 months of history, a calendar pattern that holds in every whole year is removed. Offsets are re-estimated each
     September from the data up to then and used for the following year, so no month is adjusted with information from its future.
  1. Signals: sales, operating profit (EBITDA), cash received from customers. (Version 3 used operating cash flow in place of sales; the
     synthetic test showed it rises when sales fall, because receivables unwind, and so hides setbacks. It is still used for funding purity.)
  2. Robust baseline: median and MAD of the 12 months ending 3 months earlier (at least 8 months); z-scores clipped to +-6; composite
     z = weighted sum (about 0.4 / 0.3 / 0.3).
  3. Market adjustment: z_adj = z - beta x F, F = the peer group's median composite (leave-one-out), beta = Theil-Sen slope of the firm's
     composite on F over its history to date (re-estimated each September), shrunk towards 1.
  4. Episodes with hysteresis: start when z_adj <= -theta in two months running (a one-month blip is noise), end after two months
     above -theta/2, baseline frozen at the start.
     Depth D = min(3, |trough|); judging horizon H = clamp(round(4 + 1.5 D), 4, 9) months after the trough.
  5. An episode is judged once its horizon has passed: R = 100 (rho_w rho + (1 - rho_w) S); funding purity phi = operations / (operations
     + new borrowing); R' = R (0.25 + 0.75 phi); Q = response (costs flexed, collections tightened); I = book integrity;
     G = 0.45 R' + 0.25 Q + 0.30 I, capped at 40 if I < 40.
  6. Weathered storm (new): when the firm's peer group goes through a setback (F <= -theta, ending after two months above -theta/2) and the
     firm's own unadjusted composite never fell to -theta (no setback of its own) in that time, the firm is credited with a
     setback passed: G = 0.70 x 100 + 0.30 x I. Staying out of trouble in a storm counts the same as recovering from one.
  7. Demonstrated = weighted mean of judged G, weights D x 2^(-age/24).
Capacity: mean rating of cash cover, cash against the worst month, short-term share of debt, top-customer share, bills paid on time,
margin steadiness, drawings to operating cash flow, and a stress-test score (share of four preset shocks the cash survives).
Secret draws: theta, signal weights, kappa, horizon offset, beta shrinkage and the regain weight are drawn 25 times from a seed derived from
a secret; the score is the median and the band the 10th-90th percentile. Bands: Strong >= 75, Steady >= 55, Needs work >= 40, Weak."""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import config as C
from .measures import windows

SECRET = os.environ.get("GENIUS_GRIT_SECRET", "genius-prototype-secret")
K_DRAWS = 25
BASE_LEN, BASE_LAG, MIN_BASE = 12, 3, 8
SEASON_MIN = 24
SIGNALS = ("sales", "op", "rec")
SEASON_COLS = ("sales", "op", "rec", "rev", "opex", "ocf")
BASE_W = np.array([0.40, 0.30, 0.30])


def draws(seed_extra: str = "") -> list[dict]:
    h = hashlib.sha256((SECRET + seed_extra).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    out = []
    for _ in range(K_DRAWS):
        w = np.clip(BASE_W + rng.uniform(-0.07, 0.07, 3), 0.15, 0.6)
        out.append(dict(theta=rng.uniform(1.0, 1.6), w=w / w.sum(), kappa=rng.uniform(1.5, 2.5), h_off=int(rng.integers(-1, 2)),
                        kbeta=rng.uniform(4.0, 8.0), r_rho=rng.uniform(0.55, 0.65)))
    return out


# ---------------------------------------------------------------------------------------------------------------- seasonality
def seasonal_offsets(x: np.ndarray, u: int, cal_months: np.ndarray) -> np.ndarray:
    """Offsets per calendar month (N x 13, index 1..12) from data up to and including month u, whole years counted back from u.
    A month is seasonal only if it is on the same side of its year's mean in every usable year; offset = smallest such deviation."""
    N = x.shape[0]
    k = (u + 1) // 12
    off = np.zeros((N, 13))
    if k < 2:
        return off
    seg = x[:, u + 1 - 12 * k: u + 1].reshape(N, k, 12)
    cm = cal_months[u + 1 - 12 * k: u + 1].reshape(k, 12)[0]          # calendar month of each position (same in every block)
    usable = np.isnan(seg).sum(2) <= 2
    with np.errstate(all="ignore"):
        mu = np.nanmean(seg, axis=2, keepdims=True)
    dev = np.where(usable[:, :, None], seg - mu, np.nan)
    nval = (~np.isnan(dev)).sum(1)
    allpos = np.all(np.where(np.isnan(dev), True, dev > 0), axis=1) & (nval >= 2)
    allneg = np.all(np.where(np.isnan(dev), True, dev < 0), axis=1) & (nval >= 2)
    with np.errstate(all="ignore"):
        mn = np.nanmin(np.abs(dev), axis=1)
    o = np.where(allpos, mn, np.where(allneg, -mn, 0.0))
    o = o - o.mean(1, keepdims=True)
    off[:, cm] = o
    return off


def _epochs(T: int, cal_months: np.ndarray):
    """Update months (each September) and, for every month, which update applies (-1 = none yet)."""
    ups = [t for t in range(T) if cal_months[t] == 9]
    which = np.full(T, -1)
    for j, u in enumerate(ups):
        which[u + 1:] = j
    return ups, which


def _baseline(x: np.ndarray):
    """Median and scale of the 12 months ending 3 months before each month (N x T each); NaN with fewer than 8 months."""
    N, T = x.shape
    lagged = np.full((N, T), np.nan)
    lagged[:, BASE_LAG:] = x[:, :T - BASE_LAG]
    v = windows(lagged, BASE_LEN)
    cnt = (~np.isnan(v)).sum(2)
    with np.errstate(all="ignore"):
        med = np.nanmedian(v, axis=2)
        mad = np.nanmedian(np.abs(v - med[:, :, None]), axis=2) * 1.4826
        mabs = np.nanmean(np.abs(v), axis=2)
    scale = np.maximum(np.maximum(mad, 0.15 * mabs), 1.0)
    ok = cnt >= MIN_BASE
    return np.where(ok, med, np.nan), np.where(ok, scale, np.nan)


def prepare(sig: dict, cal_months: np.ndarray) -> dict:
    """Season-adjusted signals and baselines that every draw shares. Each month uses the offsets estimated at the latest September before it."""
    N, T = sig["rev"].shape
    ups, which = _epochs(T, cal_months)
    adj = {c: sig[c].astype(float).copy() for c in sig}
    med = {k: np.full((N, T), np.nan) for k in SIGNALS}
    scl = {k: np.full((N, T), np.nan) for k in SIGNALS}
    hist = np.cumsum(~np.isnan(sig["rev"]), axis=1)
    versions = [None] + ups
    for j, u in enumerate(versions):
        cols = np.where(which == j - 1)[0] if u is not None else np.where(which == -1)[0]
        if len(cols) == 0:
            continue
        X = {}
        for c in SEASON_COLS:
            if u is None:
                X[c] = sig[c].astype(float)
                continue
            off = seasonal_offsets(sig[c].astype(float), u, cal_months)
            enough = (hist[:, u] >= SEASON_MIN)[:, None]
            X[c] = sig[c] - np.where(enough, off[:, cal_months], 0.0)
        for c in SEASON_COLS:
            adj[c][:, cols] = X[c][:, cols]
        c0 = max(0, int(cols.min()) - BASE_LEN - BASE_LAG)
        for k in SIGNALS:
            m, s_ = _baseline(X[k][:, c0:int(cols.max()) + 1])
            med[k][:, cols] = m[:, cols - c0]
            scl[k][:, cols] = s_[:, cols - c0]
    pos = np.where(np.isnan(adj["ocf"]), 0.0, np.maximum(0.0, adj["ocf"]))
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(pos, 1)], 1)
    return dict(adj=adj, med=med, scl=scl, ups=ups, which=which, _cs_ocfpos=cs)


def raw_composite(prep: dict, w=BASE_W) -> np.ndarray:
    z = [np.clip((prep["adj"][k] - prep["med"][k]) / prep["scl"][k], -6, 6) for k in SIGNALS]
    return w[0] * z[0] + w[1] * z[1] + w[2] * z[2]


def theil_sen_by_epoch(zc: np.ndarray, F: np.ndarray, ups: list, chunk: int = 300):
    """Slope and number of points of the firm's composite on F using data up to each September (N x len(ups))."""
    N, T = zc.shape
    slopes = np.full((N, len(ups)), np.nan); counts = np.zeros((N, len(ups)))
    for j, u in enumerate(ups):
        x = F[:, :u + 1]; y = zc[:, :u + 1]
        ok = ~np.isnan(x) & ~np.isnan(y)
        counts[:, j] = ok.sum(1)
        for a in range(0, N, chunk):
            xs, ys, ms = x[a:a + chunk], y[a:a + chunk], ok[a:a + chunk]
            dx = xs[:, :, None] - xs[:, None, :]
            dy = ys[:, :, None] - ys[:, None, :]
            iu = np.triu_indices(u + 1, 1)
            dx, dy = dx[:, iu[0], iu[1]], dy[:, iu[0], iu[1]]
            mm = ms[:, iu[0]] & ms[:, iu[1]] & (np.abs(dx) > 1e-9)
            with np.errstate(all="ignore"):
                sl = np.where(mm, dy / np.where(mm, dx, 1), np.nan)
                slopes[a:a + chunk, j] = np.nanmedian(sl, axis=1)
    return slopes, counts


# ---------------------------------------------------------------------------------------------------------------- one draw, all firms
def _run_draw(p: dict, prep: dict, sig: dict, F, beta_m, integrity, alive, keep_eps=False):
    """Forward pass for one parameter draw. Returns per-month cumulative sums for the aggregate and (optionally) the episode list."""
    N, T = sig["rev"].shape
    th_in, th_out = p["theta"], 0.5 * p["theta"]
    zl = [np.clip((prep["adj"][k] - prep["med"][k]) / prep["scl"][k], -6, 6) for k in SIGNALS]
    adjv = [prep["adj"][k] for k in SIGNALS]
    zraw = p["w"][0] * zl[0] + p["w"][1] * zl[1] + p["w"][2] * zl[2]
    in_ep = np.zeros(N, bool); fmed = np.zeros((N, 3)); fscl = np.ones((N, 3))
    s0 = np.zeros(N, int); tr = np.zeros(N, int); zt = np.zeros(N); above = np.zeros(N, int)
    zpath = np.full((N, T), np.nan)
    pending = []                                            # (judge month, firm idx array, s, trough, zt, D, H, e, how)
    num = np.zeros((N, T)); den = np.zeros((N, T)); cnt = np.zeros((N, T))
    part_num = {k: np.zeros((N, T)) for k in ("gap_regained", "stayed_recovered", "funded_by_operations", "response")}
    rec_den = np.zeros((N, T)); rec_n = np.zeros((N, T)); weathered_n = np.zeros((N, T))
    # market setback state per firm (for weathered storms)
    m_in = np.zeros(N, bool); m_s = np.zeros(N, int); m_ab = np.zeros(N, int); m_low = np.zeros(N); own_overlap = np.zeros(N, bool); fmin = np.zeros(N)
    eps = [] if keep_eps else None
    I = integrity
    prev_z = np.zeros(N)
    for i in range(T):
        live = ~np.isnan(zraw[:, i])
        # current composite: frozen baseline inside an episode
        zf = sum(p["w"][k] * np.clip((adjv[k][:, i] - fmed[:, k]) / fscl[:, k], -6, 6) for k in range(3))
        zc = np.where(in_ep, zf, zraw[:, i])
        fi = F[:, i] if F is not None else np.full(N, np.nan)
        b = beta_m[:, i] if beta_m is not None else np.ones(N)
        zc = np.where(np.isnan(fi), zc, zc - b * fi)
        valid = live | (in_ep & ~np.isnan(zf))
        zc = np.where(valid & alive[:, i], zc, np.nan)
        zpath[:, i] = zc
        # enter: two months running at or below -theta; the episode starts at the first of them, baseline frozen there
        enter = ~in_ep & (zc <= -th_in) & (prev_z <= -th_in) & (i >= 1)
        if enter.any():
            fmed[enter] = np.stack([prep["med"][k][enter, i - 1] for k in SIGNALS], 1)
            fscl[enter] = np.stack([prep["scl"][k][enter, i - 1] for k in SIGNALS], 1)
            s0[enter] = i - 1; tr[enter] = np.where(prev_z[enter] < zc[enter], i - 1, i); zt[enter] = np.minimum(zc[enter], prev_z[enter]); above[enter] = 0
            in_ep |= enter
        prev_z = np.where(in_ep, 0.0, zc)
        cont = in_ep & ~enter & ~np.isnan(zc)
        newlow = cont & (zc < zt - 0.3)
        tr = np.where(newlow, i, tr); zt = np.where(newlow, zc, zt); above = np.where(newlow, 0, above)
        D = np.minimum(3.0, -zt)
        H = np.clip(np.round(4 + 1.5 * D).astype(int) + p["h_off"], 4, 9)
        above = np.where(cont & ~newlow, np.where(zc > -th_out, above + 1, 0), above)
        rec = cont & (above >= 2)
        tout = cont & ~rec & ((i >= tr + H) | (i >= s0 + 12))
        close = rec | tout
        if close.any():
            idx = np.where(close)[0]
            e = np.where(rec[idx], i - 1, i)
            jm = tr[idx] + H[idx]
            pending.append((jm, idx, s0[idx].copy(), tr[idx].copy(), zt[idx].copy(), D[idx].copy(), H[idx].copy(), e, rec[idx]))
            in_ep[idx] = False
        own_overlap |= in_ep
        # market setbacks seen through the firm's own peer series
        if F is not None:
            fv = fi
            m_enter = ~m_in & (fv <= -th_in)
            m_s = np.where(m_enter, i, m_s); fmin = np.where(m_enter, fv, fmin)
            m_low = np.where(m_enter, np.where(np.isnan(zraw[:, i]), 0, zraw[:, i]), m_low)
            own_overlap = np.where(m_enter, in_ep, own_overlap)
            m_ab = np.where(m_enter, 0, m_ab)
            m_in |= m_enter
            mc = m_in & ~m_enter & ~np.isnan(fv)
            fmin = np.where(mc, np.minimum(fmin, fv), fmin)
            m_low = np.where(m_in & ~np.isnan(zraw[:, i]), np.minimum(m_low, zraw[:, i]), m_low)
            m_ab = np.where(mc, np.where(fv > -th_out, m_ab + 1, 0), m_ab)
            m_end = mc & ((m_ab >= 2) | (i >= m_s + 12))
            if m_end.any():
                w_ok = m_end & ~own_overlap & (m_low > -th_in) & alive[:, i]
                ww = np.where(w_ok)[0]
                if len(ww):
                    G = 0.70 * 100 + 0.30 * I[ww]
                    G = np.where(I[ww] < 40, np.minimum(G, 40), G)
                    Dm = np.minimum(3.0, -fmin[ww])
                    wt = Dm * 2.0 ** (i / 24)
                    num[ww, i] += wt * G; den[ww, i] += wt; cnt[ww, i] += 1
                    weathered_n[ww, i] += 1
                    if keep_eps:
                        for k_, f_ in enumerate(ww):
                            eps.append(dict(firm=int(f_), kind="weathered", start=int(m_s[f_]), trough=int(m_s[f_]), end=i, judged=i, depth=float(Dm[k_]),
                                            G=float(G[k_]), outcome="weathered the sector's setback"))
                m_in &= ~m_end
        # judge episodes whose horizon has passed
        still = []
        for rec_ in pending:
            jm, idx, s, t_, z_, D_, H_, e_, recd = rec_
            due = jm <= i
            if not due.any():
                still.append(rec_)
                continue
            if (~due).any():
                still.append(tuple(a[~due] if isinstance(a, np.ndarray) else a for a in rec_))
            sel = np.where(due)[0]
            G, parts = _judge(idx[sel], s[sel], t_[sel], z_[sel], H_[sel], e_[sel], zpath, sig, prep, p, I, th_out, i)
            wt = D_[sel] * 2.0 ** (e_[sel] / 24)
            np.add.at(num, (idx[sel], i), wt * G); np.add.at(den, (idx[sel], i), wt); np.add.at(cnt, (idx[sel], i), 1)
            np.add.at(rec_den, (idx[sel], i), wt); np.add.at(rec_n, (idx[sel], i), 1)
            for part_name, part_value in parts.items():
                value = part_value / 100 if part_name == "response" else part_value
                np.add.at(part_num[part_name], (idx[sel], i), wt * value)
            if keep_eps:
                for k_ in range(len(sel)):
                    eps.append(dict(firm=int(idx[sel][k_]), kind="setback", start=int(s[sel][k_]), trough=int(t_[sel][k_]), end=int(e_[sel][k_]), judged=i,
                                    depth=float(D_[sel][k_]), G=float(G[k_]), outcome="recovered" if recd[sel][k_] else "did not recover in time",
                                    **{k: float(v[k_]) for k, v in parts.items()}))
        pending = still
    if keep_eps:      # still unfolding at the end
        for i_ in np.where(in_ep)[0]:
            eps.append(dict(firm=int(i_), kind="setback", start=int(s0[i_]), trough=int(tr[i_]), end=None, judged=None, depth=float(min(3, -zt[i_])), G=None,
                            outcome="still unfolding"))
        for rec_ in pending:
            jm, idx, s, t_, z_, D_, H_, e_, recd = rec_
            for k_ in range(len(idx)):
                eps.append(dict(firm=int(idx[k_]), kind="setback", start=int(s[k_]), trough=int(t_[k_]), end=int(e_[k_]), judged=None, depth=float(D_[k_]), G=None,
                                outcome="being judged (horizon not over)"))
    recovery = {k: np.cumsum(v, 1) / np.where(np.cumsum(rec_den, 1) > 0, np.cumsum(rec_den, 1), np.nan) for k, v in part_num.items()}
    recovery["recovery_n"] = np.cumsum(rec_n, 1)
    recovery["weathered_n"] = np.cumsum(weathered_n, 1)
    return np.cumsum(num, 1), np.cumsum(den, 1), np.cumsum(cnt, 1), zpath, eps, recovery


def _gather(x, rows, cols):
    """x[rows, cols] with NaN where cols fall outside the array."""
    T = x.shape[1]
    ok = (cols >= 0) & (cols < T)
    return np.where(ok, x[rows[:, None] if cols.ndim == 2 else rows, np.clip(cols, 0, T - 1)], np.nan)


def _judge(idx, s, t, zt, H, e, zpath, sig, prep, p, I, th_out, now):
    """Score a batch of closed episodes whose horizon has passed (vectorised)."""
    m = len(idx)
    L = 10
    k = np.arange(L)[None, :]
    b = np.minimum(now, t + H)
    cols = t[:, None] + k
    inwin = k <= (b - t)[:, None]
    win = np.where(inwin, _gather(zpath, idx, cols), np.nan)
    nxt = np.concatenate([win[:, 1:], np.full((m, 1), np.nan)], 1)
    with np.errstate(all="ignore"):
        pair = np.where(inwin, np.nanmean(np.stack([win, np.where(np.concatenate([inwin[:, 1:], np.zeros((m, 1), bool)], 1), nxt, np.nan)]), 0), np.nan)
        best = np.nanmax(np.where(np.isnan(pair), -np.inf, pair), 1)
    best = np.where(np.isfinite(best), best, zt)
    rho = np.clip((best - zt) / np.maximum(1e-9, -zt), 0, 1)
    back = inwin & (win > -th_out)
    has = back.any(1)
    j0 = t + np.argmax(back, 1)
    c3 = j0[:, None] + np.arange(1, 4)[None, :]
    ok3 = c3 <= now
    z3 = np.where(ok3, _gather(zpath, idx, c3), np.nan)
    n3 = ok3.sum(1)
    with np.errstate(all="ignore"):
        frac = np.nanmean(np.where(np.isnan(z3), np.nan, (z3 > -th_out).astype(float)), 1)
    S = np.where(~has, 0.0, np.where(n3 == 0, 0.5, np.nan_to_num(frac, nan=0.5)))
    # funding purity
    cs = prep["_cs_ocfpos"]
    ops = cs[idx, b + 1] - cs[idx, s]
    debt = sig["debt"]
    d1 = debt[idx, b]; d0 = debt[idx, np.maximum(0, s - 1)]
    newf = np.where(np.isnan(d1) | np.isnan(d0), 0.0, np.maximum(0.0, d1 - d0))
    phi = np.where(ops + newf <= 0, 1.0, ops / np.where(ops + newf > 0, ops + newf, 1))
    # response
    rev, opx = prep["adj"]["rev"], prep["adj"]["opex"]
    b0 = np.maximum(0, s - 6)
    c6 = b0[:, None] + np.arange(6)[None, :]
    in6 = c6 < s[:, None]
    with np.errstate(all="ignore"):
        base_rev = np.nanmedian(np.where(in6, _gather(rev, idx, c6), np.nan), 1)
        base_opx = np.nanmedian(np.where(in6, _gather(opx, idx, c6), np.nan), 1)
        c13 = s[:, None] + np.arange(13)[None, :]
        in13 = c13 <= t[:, None]
        rev_dip = np.nanmean(np.where(in13, _gather(rev, idx, c13), np.nan), 1)
        c4 = t[:, None] + np.arange(4)[None, :]
        in4 = c4 <= b[:, None]
        opx_after = np.nanmean(np.where(in4, _gather(opx, idx, c4), np.nan), 1)
    shortfall = np.where(np.isnan(base_rev), 0.0, np.maximum(0.0, base_rev - rev_dip))
    cut = np.where(np.isnan(base_opx), 0.0, np.maximum(0.0, base_opx - opx_after))
    adj_cost = np.where(shortfall > 0, np.clip(cut / np.where(shortfall > 0, shortfall, 1), 0, 1), 0.5)
    recv, rv = sig["recv"], sig["rev"]
    with np.errstate(all="ignore"):
        r0 = np.where(rv[idx, s] > 0, recv[idx, s] / rv[idx, s], np.nan)
        r1 = np.where(rv[idx, b] > 0, recv[idx, b] / rv[idx, b], np.nan)
    adj_coll = np.where(np.isnan(r0) | np.isnan(r1) | (r0 <= 0), 0.5, np.clip(0.5 + (r0 - r1) / np.where(r0 > 0, r0, 1), 0, 1))
    Q = 100 * (0.5 * adj_cost + 0.5 * adj_coll)
    R = 100 * (p["r_rho"] * rho + (1 - p["r_rho"]) * S)
    Rp = R * (0.25 + 0.75 * phi)
    Ii = I[idx]
    G = 0.45 * Rp + 0.25 * Q + 0.30 * Ii
    G = np.where(Ii < 40, np.minimum(G, 40.0), G)
    return G, dict(gap_regained=rho, stayed_recovered=S, funded_by_operations=phi, response=Q)


# ---------------------------------------------------------------------------------------------------------------- capacity
def stress_survival(f: dict, d: dict, meas: dict) -> np.ndarray:
    """Share (0-100) of four preset shocks over three months that leave at least one month of fixed costs in cash and unused credit."""
    from .measures import roll_mean, roll_sum, ocf as _ocf
    rev3 = roll_mean(f["rev"], 3)
    cm = 1 - np.where(rev3 > 0, (roll_mean(f["cogs"], 3) + roll_mean(f["var_opex"], 3)) / np.where(rev3 > 0, rev3, 1), np.nan)
    avail = f["cash"] + d["headroom"]
    base = (_ocf(f, d, 3) - roll_sum(f["principal"], 3) - roll_sum(f["drawings"], 3)) / 3
    floor = d["wf"] * 4.345
    losses = [0.2 * rev3 * cm * 3, rev3, np.nan_to_num(f["top_cust"]) * rev3 * cm * 3, 0.1 * (roll_mean(f["cogs"], 3) + roll_mean(d["opex"], 3)) * 3]
    ok = sum(((avail + 3 * base - L) > floor).astype(float) for L in losses)
    return np.where(np.isnan(rev3) | np.isnan(base), np.nan, 100 * ok / 4)


def capacity(ratings: dict, stress_score: np.ndarray) -> np.ndarray:
    stack = np.stack([ratings[m].astype(float) for m in C.CAPACITY] + [stress_score], 0)
    with np.errstate(all="ignore"):
        cap = np.nanmean(stack, 0)
    return np.where((~np.isnan(stack)).sum(0) >= 4, cap, np.nan)


# ---------------------------------------------------------------------------------------------------------------- public
def signals(f: dict, d: dict) -> dict:
    return dict(sales=f["rev"], ocf=d["ocf1"], op=d["ebitda"], rec=f["collections"], rev=f["rev"], opex=d["opex"], recv=f["recv"], debt=d["debt"])


def market_factor(prep: dict, keys: np.ndarray) -> tuple[np.ndarray, dict]:
    """F per firm-month (leave-one-out peer median of the base composite) and the plain group medians {key: array(T)} for outside firms."""
    from .peers import loo_median
    zc = raw_composite(prep)
    F = loo_median(zc, keys)
    T = zc.shape[1]
    table = {}
    for t in range(T):
        col = keys[:, t]
        ok = (col != None) & ~np.isnan(zc[:, t])   # noqa: E711
        if not ok.any():
            continue
        import pandas as pd
        med = pd.Series(zc[ok, t]).groupby(pd.Series(col[ok])).median()
        for k, v in med.items():
            table.setdefault(k, np.full(T, np.nan))[t] = v
    return F, table


def compute(sig: dict, cal_months: np.ndarray, alive: np.ndarray, integrity: np.ndarray, cap: np.ndarray,
            F: np.ndarray | None = None, keep_eps: bool = False, prep: dict | None = None) -> dict:
    """Grit for every firm and month. Returns score/lo/hi/n/demonstrated (N x T) and, if asked, the episodes of the typical draw."""
    prep = prep or prepare(sig, cal_months)
    N, T = sig["rev"].shape
    D = draws()
    beta_m = None
    if F is not None:
        zc = raw_composite(prep)
        ups, which = prep["ups"], prep["which"]
        slopes, counts = theil_sen_by_epoch(zc, F, ups)
    scores = np.full((K_DRAWS, N, T), np.nan, np.float32)
    dems = np.full((K_DRAWS, N, T), np.nan, np.float32)
    ns = np.zeros((K_DRAWS, N, T), np.float32)
    med_theta = float(np.median([p["theta"] for p in D]))
    rep = int(np.argmin([abs(p["theta"] - med_theta) for p in D]))
    episodes = None
    recovery_draws = {}
    for k, p in enumerate(D):
        if F is not None:
            b = np.ones((N, T))
            for j in range(len(ups)):
                cols = np.where(which == j)[0]
                n_ = counts[:, j]
                bj = np.where((n_ >= 6) & ~np.isnan(slopes[:, j]), np.clip(slopes[:, j], 0, 2), 1.0)
                bj = (n_ * bj + p["kbeta"]) / (n_ + p["kbeta"])
                b[:, cols] = bj[:, None]
            beta_m = b
        num, den, cnt, zpath, eps, recovery = _run_draw(p, prep, sig, F, beta_m, integrity, alive, keep_eps=keep_eps and k == rep)
        for part_name, values in recovery.items():
            recovery_draws.setdefault(part_name, []).append(values.astype(np.float32))
        if eps is not None:
            episodes = eps
        with np.errstate(all="ignore"):
            dem = np.where(den > 0, num / den, np.nan)
        wb = cnt / (cnt + p["kappa"])
        g = np.where(np.isnan(dem), cap, np.where(np.isnan(cap), dem, wb * dem + (1 - wb) * cap))
        hist = np.cumsum(~np.isnan(sig["rev"]), axis=1)
        g = np.where((hist >= 12) & alive, g, np.nan)
        scores[k] = g; dems[k] = dem; ns[k] = cnt
    with np.errstate(all="ignore"):
        out = dict(score=np.nanmedian(scores, 0).astype(np.float32), lo=np.nanpercentile(scores, 10, axis=0).astype(np.float32),
                   hi=np.nanpercentile(scores, 90, axis=0).astype(np.float32), n=np.median(ns, 0).astype(np.float32),
                   demonstrated=np.nanmedian(dems, 0).astype(np.float32), capacity=cap.astype(np.float32))
    out["recovery_features"] = {k: np.nanmedian(np.stack(v), 0).astype(np.float32) for k, v in recovery_draws.items() if k not in ("recovery_n", "weathered_n")}
    out["recovery_n"] = np.nanmedian(np.stack(recovery_draws["recovery_n"]), 0).astype(np.float32)
    out["weathered_n"] = np.nanmedian(np.stack(recovery_draws["weathered_n"]), 0).astype(np.float32)
    out["episodes"] = episodes
    out["prep"] = prep
    return out
