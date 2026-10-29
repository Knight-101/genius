"""Synthetic peer database: 5,000 Indian MSMEs, 120 months each (Oct 2016 - Sep 2026), simulated from a hidden cash process.

Nothing here uses a Genius formula. Firms have hidden traits (true resilience r, collection discipline, payment discipline, cash habit,
owner drawings habit, cost flexibility); a two-state market regime (normal / stress) and a forced downturn in the test years push them;
idiosyncratic shocks hit them, some recoverable and some permanent (a lost key customer, which Grit cannot see). Receivables and
payables are tracked as monthly cohorts, so ageing, days late and on-time shares come out of the process rather than being drawn.
Cash shortfalls are resolved the way small firms do it: cash-credit first, then delaying statutory dues, suppliers and finally loan
instalments, sometimes with the owner putting money in. Outcomes (90-day stress, 12-month default, severity) are read off that
process with 5% label noise.

Output: Panel with .f (dict of N x T arrays, the same "monthly facts" an uploaded ledger produces), .firms (one row per firm, hidden
traits included), .y (outcomes) and .alive (observed months)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import config as C

AGES = 13            # receivable / payable cohorts tracked for 13 months
DAYS = 30.4375

SEASON = {  # Jan..Dec
    "Trading · FMCG": [.97, .95, 1.0, .97, .96, .95, .97, 1.0, 1.03, 1.12, 1.10, .98],
    "Trading · building materials": [1.05, 1.08, 1.15, 1.08, 1.05, .90, .82, .85, .92, 1.0, 1.0, 1.10],
    "Trading · pharma": [.98, .96, 1.0, .97, .97, 1.0, 1.08, 1.08, 1.06, .99, .97, .94],
    "Trading · electronics": [.92, .90, .95, 1.10, 1.14, .98, .92, .93, .96, 1.15, 1.13, .92],
    "Manufacturing · light engineering": [.98, 1.0, 1.15, .92, .96, .98, .97, .98, 1.0, 1.0, .99, 1.07],
    "Manufacturing · textiles": [.90, .90, .98, .97, .95, .98, 1.03, 1.12, 1.15, 1.13, 1.0, .89],
    "Manufacturing · food processing": [1.0, .96, 1.02, .96, .95, .97, .98, 1.0, 1.04, 1.12, 1.08, .98],
    "Manufacturing · plastics": [1.0, 1.0, 1.05, .98, .99, .98, .97, .99, 1.0, 1.03, 1.01, 1.0],
    "Services · IT and ITeS": [1.0, 1.0, 1.08, .98, .99, 1.0, 1.0, 1.0, 1.0, 1.01, 1.0, .94],
    "Services · logistics": [.98, .97, 1.04, 1.0, .99, .92, .92, .97, 1.02, 1.10, 1.09, 1.0],
}
SENS = {"Trading · FMCG": .5, "Trading · building materials": 1.3, "Trading · pharma": .4, "Trading · electronics": 1.0,
        "Manufacturing · light engineering": 1.2, "Manufacturing · textiles": 1.1, "Manufacturing · food processing": .6,
        "Manufacturing · plastics": 1.0, "Services · IT and ITeS": .8, "Services · logistics": 1.1}
GM = {"Trading · FMCG": (.09, .02), "Trading · building materials": (.14, .03), "Trading · pharma": (.16, .03), "Trading · electronics": (.11, .025),
      "Manufacturing · light engineering": (.27, .05), "Manufacturing · textiles": (.22, .05), "Manufacturing · food processing": (.24, .05),
      "Manufacturing · plastics": (.25, .05), "Services · IT and ITeS": (.45, .08), "Services · logistics": (.28, .06)}
EBITDA = {"trading": (.02, .07), "manufacturing": (.04, .15), "services": (.05, .20)}
PLANT = {"trading": (.005, .04), "manufacturing": (.10, .45), "Services · IT and ITeS": (.02, .08), "Services · logistics": (.25, .60)}


@dataclass
class Panel:
    f: dict                      # monthly facts, each N x T
    firms: pd.DataFrame          # one row per firm, hidden traits included
    y: dict                      # outcomes, each N x T
    alive: np.ndarray            # N x T, month observed
    months: pd.PeriodIndex
    regime: np.ndarray           # market stress flag per month (truth; models never see it directly)
    meta: dict = field(default_factory=dict)

    @property
    def N(self):
        return self.alive.shape[0]

    @property
    def T(self):
        return self.alive.shape[1]

    def firm(self, i: int) -> "Panel":
        """One firm as its own one-row panel (used to run the single-firm path on a database firm)."""
        sl = slice(i, i + 1)
        return Panel({k: v[sl] for k, v in self.f.items()}, self.firms.iloc[sl].reset_index(drop=True), {k: v[sl] for k, v in self.y.items()},
                     self.alive[sl], self.months, self.regime, dict(self.meta))


def _cohort(terms, mean_days, sd_days, k_ontime=7.0, k_mse=45.0):
    """For one month's invoices (issue day uniform in the month, days-to-pay ~ Normal(mean, sd)), per month of age 0..12:
    share of the cohort paid (the remainder is paid at 12), share of that money paid within terms+7 days, its mean days late,
    and share paid within 45 days of the bill (Section 43B(h))."""
    n = len(mean_days)
    u = ((np.arange(6) + 0.5) / 6 * DAYS)[None, :, None]
    a = np.arange(AGES)[None, None, :]
    mu = np.asarray(mean_days, float)[:, None, None]
    sd = np.maximum(np.asarray(sd_days, float), 1.0)[:, None, None]
    lo = np.where(a == 0, -1e9, a * DAYS - u)
    hi = np.where(a == AGES - 1, 1e9, (a + 1) * DAYS - u)
    al, be = (lo - mu) / sd, (hi - mu) / sd
    Pa, Pb = ndtr(al), ndtr(be)
    mass = np.clip(Pb - Pa, 0, 1)
    t = np.asarray(terms, float)[:, None, None]

    def within(limit):
        top = np.minimum(hi, limit)
        return np.clip(ndtr((top - mu) / sd) - Pa, 0, None) * (top > lo)
    phi = lambda z: np.exp(-0.5 * np.clip(z, -30, 30) ** 2) / np.sqrt(2 * np.pi)
    with np.errstate(all="ignore"):
        tmean = mu + sd * (phi(al) - phi(be)) / np.where(mass > 1e-12, mass, 1)
    m = mass.mean(1)
    on = within(t + k_ontime).mean(1)
    i45 = within(np.broadcast_to(np.asarray(k_mse, float).reshape(-1, 1, 1) if np.ndim(k_mse) else np.full_like(t, k_mse), t.shape)).mean(1)
    late = (mass * (tmean - t)).mean(1)
    safe = np.where(m > 1e-12, m, 1)
    return dict(p=m / m.sum(1, keepdims=True), ontime=np.clip(on / safe, 0, 1), late=late / safe, in45=np.clip(i45 / safe, 0, 1))


def simulate(n: int = 5000, seed: int = 20261028, burn: int = 24) -> Panel:
    rng = np.random.default_rng(seed)
    T = C.T_MONTHS
    TT = T + burn
    sectors = np.array(C.SECTOR_LIST)[rng.integers(0, len(C.SECTOR_LIST), n)]
    fam = np.array([C.family(s) for s in sectors])
    has_stock = np.array([C.SECTORS[s][1] for s in sectors])

    # --- size: aim at today's MSME mix (about 40% micro, 45% small, 15% medium)
    u = rng.random(n)
    lo, hi = np.where(u < .40, 1.2e7, np.where(u < .85, 1.1e8, 1.05e9)), np.where(u < .40, 9e7, np.where(u < .85, 8.5e8, 4.2e9))
    annual_now = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    g = np.clip(rng.normal(0.08, 0.05, n), -0.04, 0.22)                    # nominal sales growth a year
    annual0 = annual_now / (1 + g) ** ((TT - 1) / 12)
    noise_sd = rng.uniform(0.04, 0.10, n)
    season = np.array([SEASON[s] for s in sectors])
    amp = rng.uniform(0.5, 1.5, n)
    season = 1 + amp[:, None] * (season - 1)
    sens_s = np.array([SENS[s] for s in sectors])
    sens_f = sens_s * rng.uniform(0.7, 1.3, n)

    # --- hidden traits
    coll = rng.beta(2.5, 1.8, n)                                            # collection discipline
    pdisc = rng.beta(3, 1.8, n)                                             # payment discipline
    r = rng.beta(2, 2, n)                                                   # true resilience (Grit's target)
    owner = np.clip(0.6 * rng.beta(2, 2, n) + 0.4 * r + rng.normal(0, .05, n), 0, 1)   # owner restraint, partly part of resilience
    cash_weeks = np.exp(rng.normal(np.log(14), 0.5, n) + 0.8 * (r - 0.5))    # cash habit, weeks of fixed costs (resilient owners keep more)
    flex = np.clip(0.05 + 0.75 * r + rng.normal(0, 0.08, n), 0, 0.9)        # cost flexibility in a setback
    k_rec = np.clip(0.02 + 0.40 * r + rng.normal(0, 0.04, n), 0.01, 0.5)    # recovery speed per month
    infuse = 0.35 * r                                                      # chance the owner puts money in when cash runs short

    gm0 = np.array([rng.normal(*GM[s]) for s in sectors]).clip(0.04, 0.75)
    e_lo = np.array([EBITDA[f][0] for f in fam]); e_hi = np.array([EBITDA[f][1] for f in fam])
    ebitda_t = rng.uniform(e_lo, e_hi)
    var_share = np.where(fam == "trading", rng.uniform(.015, .04, n), rng.uniform(.03, .06, n))
    gm_sd = rng.uniform(0.003, 0.02, n)

    terms_c = np.clip(np.array([C.SECTORS[s][2] for s in sectors]) + rng.normal(0, 8, n), 7, 90)
    terms_s = np.clip(np.array([C.SECTORS[s][3] for s in sectors]) + rng.normal(0, 8, n), 7, 90)
    late0 = 0.5 + 24 * (1 - coll) ** 1.6 + rng.gamma(1.5, 1.6, n)
    plate0 = 0.0 + 14 * (1 - pdisc) ** 2 + rng.gamma(1.0, 1.0, n) - 2
    top_c = np.where(fam == "services", rng.uniform(.10, .55, n), rng.uniform(.05, .45, n))
    n_c = rng.integers(8, 60, n)
    top_s = rng.uniform(.10, .60, n)
    n_s = rng.integers(4, 25, n)
    mse_share = rng.uniform(0, .6, n)
    stock_days = np.where(fam == "trading", rng.uniform(30, 90, n), np.where(fam == "manufacturing", rng.uniform(50, 120, n), 0.0))
    p_lo = np.array([PLANT.get(s, PLANT.get(f, (0, 0)))[0] for s, f in zip(sectors, fam)])
    p_hi = np.array([PLANT.get(s, PLANT.get(f, (0, 0)))[1] for s, f in zip(sectors, fam)])
    plant_ratio = rng.uniform(p_lo, p_hi)
    fixed_share = np.maximum(0.02, gm0 - var_share - ebitda_t - plant_ratio * 0.15)   # EBITDA also pays for depreciation
    export_share = np.where(((fam == "manufacturing") & (rng.random(n) < .3)) | ((sectors == "Services · IT and ITeS") & (rng.random(n) < .4)),
                            rng.uniform(.05, .5, n), 0.0)
    has_cc = rng.random(n) < .8
    cc_limit_ratio = np.where(has_cc, rng.uniform(.06, .22, n), 0.0)
    cc_rate = rng.uniform(.10, .125, n)
    has_tl = rng.random(n) < .6
    tl_rate = rng.uniform(.10, .135, n)
    tl_ratio = np.where(has_tl, rng.uniform(.3, 1.0, n), 0.0)              # loan sized so the instalment is at most half of EBITDA at the start
    draw_habit = np.clip(1.15 - 0.85 * owner + rng.normal(0, .08, n), 0.2, 1.3)
    swing = rng.uniform(.01, .06, n)
    gm_drift = rng.normal(0, .003, n)                                     # structural margin drift a year
    # data quality (constant per firm)
    dq_bal = np.where(rng.random(n) < .92, 100.0, rng.uniform(94, 99.9, n))
    dq_cls = np.clip(rng.normal(98, 2, n), 85, 100)
    dq_party = np.clip(rng.normal(96, 4, n), 70, 100)
    dq_due = np.clip(rng.normal(88, 12, n), 30, 100)
    checks = np.where(rng.random(n) < .9, 7, np.where(rng.random(n) < .8, 6, 5)).astype(float)
    entry = np.where(rng.random(n) < .75, 0, rng.integers(1, 48, n))

    # --- market regime: two-state chain (stress about 15% of months) plus a forced downturn
    stress = np.zeros(TT, bool)
    st = False
    for t in range(TT):
        st = (rng.random() < .03) if not st else (rng.random() > .17)
        stress[t] = st
    m0, m1 = C.DOWNTURN
    stress[burn + m0: burn + m1 + 1] = True
    depth = np.where(stress, 0.10, 0.0)
    depth[burn + m0: burn + m1 + 1] = 0.22
    sec_idx = {s: i for i, s in enumerate(C.SECTOR_LIST)}
    X = np.ones((len(C.SECTOR_LIST), TT))
    for j, s in enumerate(C.SECTOR_LIST):
        x = 1.0
        for t in range(TT):
            target = 1 - SENS[s] * depth[t]
            x = x + 0.45 * (target - x)
            X[j, t] = x
    si = np.array([sec_idx[s] for s in sectors])

    cal_month = np.array([(p.month) for p in pd.period_range(pd.Period(C.START, "M") - burn, periods=TT, freq="M")])

    # --- state
    rev_level = annual0 / 12
    fixed_base = fixed_share * rev_level
    plant = plant_ratio * annual0
    other_fa = 0.04 * annual0
    annuity = (tl_rate / 12) / (1 - (1 + tl_rate / 12) ** -60)
    tl_emi = tl_ratio * 0.5 * np.maximum(ebitda_t * rev_level, 0)
    tl = tl_emi / annuity
    cc_lim = cc_limit_ratio * annual0
    std = 0.4 * cc_lim
    inv = stock_days / DAYS * rev_level * (1 - gm0)
    co_c = _cohort(terms_c, terms_c + late0, 0.25 * (terms_c + late0) + 4)
    mse_lim = 45 + np.maximum(0, terms_s - 40)        # micro and small suppliers are kept on terms of 40 days or less
    co_s = _cohort(terms_s, terms_s + plate0, 0.15 * (terms_s + plate0) + 3, k_mse=mse_lim)
    sched_c, sched_s = co_c["p"], co_s["p"]
    # cohorts: amount outstanding by age, and each cohort's remaining schedule (share of original still to be paid at each future age)
    ar_amt = np.zeros((n, AGES)); ar_orig = np.zeros((n, AGES)); ar_sched = np.zeros((n, AGES, AGES))
    ap_amt = np.zeros((n, AGES)); ap_orig = np.zeros((n, AGES)); ap_sched = np.zeros((n, AGES, AGES))
    ar_on = np.zeros((n, AGES, AGES)); ar_late = np.zeros((n, AGES, AGES))
    ap_on = np.zeros((n, AGES, AGES)); ap_late = np.zeros((n, AGES, AGES)); ap_in45 = np.zeros((n, AGES, AGES))
    for a in range(AGES):                     # steady-state backlog
        ar_orig[:, a] = rev_level; ar_sched[:, a, :] = sched_c; ar_on[:, a, :] = co_c["ontime"]; ar_late[:, a, :] = co_c["late"]
        ar_amt[:, a] = rev_level * (1 - sched_c[:, :a + 1].sum(1))
        ap_orig[:, a] = rev_level * (1 - gm0); ap_sched[:, a, :] = sched_s; ap_on[:, a, :] = co_s["ontime"]; ap_late[:, a, :] = co_s["late"]; ap_in45[:, a, :] = co_s["in45"]
        ap_amt[:, a] = rev_level * (1 - gm0) * (1 - sched_s[:, :a + 1].sum(1))
    wf0 = (fixed_base + tl * tl_rate / 12) / 4.345
    cash = cash_weeks * wf0
    stat_due = 0.12 * 0.6 * fixed_base
    tax_due = np.zeros(n)
    pool = np.zeros((n, 4))                    # supplier payments pushed back, by months overdue (0, 1, 2, 3+)
    emi_arrears = np.zeros(n)
    stat_arrears = np.zeros(n)
    shock = np.ones(n); perm = np.ones(n)
    prev_cover = cash_weeks.copy()
    net_hist = np.zeros((n, 3))
    oca_ratio = rng.uniform(.01, .04, n); ol_ratio = rng.uniform(.005, .025, n)
    oca = oca_ratio * 3 * rev_level; oth = ol_ratio * 3 * rev_level
    arrears_exp = np.zeros(n)
    hits = np.zeros((n, TT), bool); perms = np.zeros((n, TT), bool); shock_path = np.ones((n, TT))

    keys = ["rev", "exports", "cogs", "fixed", "payroll", "var_opex", "dep", "interest", "tax", "net", "invoiced", "billed", "collections",
            "supplier_paid", "drawings", "capital_in", "principal", "new_lt", "capex", "cash_in", "cash_out",
            "cash", "cash_low", "recv", "inv", "oca", "pay", "std", "ltd", "oth_liab", "stat_due", "tax_due", "plant", "fa_net", "cc_limit",
            "ar_overdue", "ar_90", "ar_paid", "ar_paid_late_w", "ar_paid_ontime", "ap_overdue", "ap_30", "ap_90", "ap_paid", "ap_paid_late_w",
            "ap_paid_ontime", "mse_paid", "mse_paid_intime", "stat_accrual", "top_cust", "hhi_cust", "top_supp", "emi_arrears", "stat_arrears",
            "recv_young", "recv_eligible"]
    F = {k: np.full((n, TT), np.nan) for k in keys}
    ages = np.arange(AGES)
    ar_past = np.clip(((ages + 1) * DAYS - terms_c[:, None]) / DAYS, 0, 1)
    ar_past90 = np.clip(((ages + 1) * DAYS - terms_c[:, None] - 90) / DAYS, 0, 1)
    ap_past = np.clip(((ages + 1) * DAYS - terms_s[:, None]) / DAYS, 0, 1)
    ap_past30 = np.clip(((ages + 1) * DAYS - terms_s[:, None] - 30) / DAYS, 0, 1)
    ap_past90 = np.clip(((ages + 1) * DAYS - terms_s[:, None] - 90) / DAYS, 0, 1)

    for t in range(TT):
        stress_t = float(stress[t])
        # shocks
        hz = 0.02 * (1 + 1.5 * stress_t)
        hit = rng.random(n) < hz
        permanent = hit & (rng.random(n) < .2)
        d = rng.uniform(.20, .55, n)
        shock = np.where(hit & ~permanent, shock * (1 - d), shock)
        drop = top_c * rng.uniform(.5, 1.0, n)
        perm = np.where(permanent, perm * (1 - drop), perm)
        top_c = np.where(permanent, np.maximum(.04, top_c * rng.uniform(.3, .7, n)), top_c)
        shock = shock + k_rec * (1 - shock)
        hits[:, t] = hit & ~permanent; perms[:, t] = permanent; shock_path[:, t] = shock * perm
        mkt = X[si, t] ** (sens_f / sens_s)
        level = rev_level * (1 + g) ** (t / 12)
        rev = level * season[np.arange(n), cal_month[t] - 1] * mkt * shock * perm * np.exp(rng.normal(0, noise_sd))
        gm = gm0 + gm_drift * t / 12 - 0.012 * stress_t * sens_f - 0.03 * (1 - shock) + rng.normal(0, gm_sd)
        cogs = rev * (1 - np.clip(gm, 0.01, 0.9))
        fb = fixed_base * (1 + np.maximum(0.03, 0.7 * g + 0.02)) ** (t / 12)     # running costs grow with the business
        cut = flex * np.clip(1 - shock * perm ** 0.3, 0, 1) * 0.8
        fixed = fb * (1 - cut)
        payroll = 0.6 * fixed
        var_opex = var_share * rev
        dep = plant * 0.15 / 12 + other_fa * 0.05 / 12
        interest = tl * tl_rate / 12 + std * cc_rate / 12
        pretax = rev - cogs - fixed - var_opex - dep - interest
        tax = 0.25 * np.maximum(0, pretax)
        net = pretax - tax
        # receivables: new cohort with a schedule set by today's lateness
        late_now = late0 + 14 * stress_t * sens_f + 25 * (1 - shock) * (1 - coll) + rng.gamma(1, 1.5, n)
        co = _cohort(terms_c, terms_c + late_now, 0.25 * (terms_c + late_now) + 4)
        ar_amt = np.roll(ar_amt, 1, axis=1); ar_orig = np.roll(ar_orig, 1, axis=1); ar_sched = np.roll(ar_sched, 1, axis=1)
        ar_on = np.roll(ar_on, 1, axis=1); ar_late = np.roll(ar_late, 1, axis=1)
        tail = ar_amt[:, 0].copy()                                           # 13-month-old balance: collected now
        ar_amt[:, 0] = rev; ar_orig[:, 0] = rev; ar_sched[:, 0, :] = co["p"]; ar_on[:, 0, :] = co["ontime"]; ar_late[:, 0, :] = co["late"]
        due_c = ar_orig * ar_sched[:, ages, ages]                            # scheduled this month at each age
        due_c = np.minimum(due_c, ar_amt)
        coll_by_age = due_c.copy(); coll_by_age[:, -1] += tail
        collections = coll_by_age.sum(1)
        ar_amt = ar_amt - due_c
        ar_paid_late_w = (coll_by_age * ar_late[:, ages, ages]).sum(1)
        ar_paid_ontime = (coll_by_age * ar_on[:, ages, ages]).sum(1)
        # stock and purchases
        if True:
            target_inv = stock_days / DAYS * cogs * (1 + 0.35 * (1 - shock))
            d_inv = np.maximum(0.5 * (target_inv - inv), -cogs)
            inv = inv + d_inv
        billed = np.maximum(0.0, cogs + d_inv)
        pressure = np.clip((4 - prev_cover) / 4, 0, 1)
        plate = plate0 + 30 * pressure * (1 - 0.5 * pdisc) + 6 * stress_t + rng.gamma(1, 1, n)
        cs_ = _cohort(terms_s, terms_s + plate, 0.15 * (terms_s + plate) + 3, k_mse=mse_lim)
        ap_amt = np.roll(ap_amt, 1, axis=1); ap_orig = np.roll(ap_orig, 1, axis=1); ap_sched = np.roll(ap_sched, 1, axis=1)
        ap_on = np.roll(ap_on, 1, axis=1); ap_late = np.roll(ap_late, 1, axis=1); ap_in45 = np.roll(ap_in45, 1, axis=1)
        tail_s = ap_amt[:, 0].copy()
        ap_amt[:, 0] = billed; ap_orig[:, 0] = billed; ap_sched[:, 0, :] = cs_["p"]; ap_on[:, 0, :] = cs_["ontime"]; ap_late[:, 0, :] = cs_["late"]; ap_in45[:, 0, :] = cs_["in45"]
        due_core = np.minimum(ap_orig * ap_sched[:, ages, ages], ap_amt)   # leaves the cohorts this month
        due_s = due_core.copy()
        due_s[:, -1] += tail_s                                               # plus any 13-month-old remainder
        # statutory and tax
        stat_acc = 0.12 * payroll
        tax_due = tax_due + tax
        q_end = cal_month[t] in (6, 9, 12, 3)
        tax_pay = np.where(q_end, tax_due, 0.0)
        stat_pay = stat_due.copy()
        # owner drawings and other flows
        net_hist = np.roll(net_hist, 1, axis=1); net_hist[:, 0] = net
        draw = np.maximum(0.02 * fb, draw_habit * np.maximum(0, net_hist.mean(1))) * (1 - 0.6 * pressure * owner)
        capex = np.where(rng.random(n) < .25, dep * 4 * rng.uniform(.6, 1.2, n), 0.0) * (1 - 0.7 * pressure)
        plant = plant + 0.8 * capex - plant * 0.15 / 12
        other_fa = other_fa + 0.2 * capex - other_fa * 0.05 / 12
        principal = np.minimum(tl, np.maximum(0, tl_emi - tl * tl_rate / 12))
        new_oca = oca_ratio * 3 * rev; new_ol = ol_ratio * 3 * rev
        d_wc_other = (new_oca - oca) - (new_ol - oth)
        oca, oth = new_oca, new_ol
        # new term loan: firms in a setback borrow more often when resilience is low
        want_loan = (shock * perm < .85) & (rng.random(n) < (0.20 * (1 - r))) & (tl < 0.4 * level * 12)
        new_lt = np.where(want_loan, 4 * fixed, 0.0)
        tl_emi = np.where(want_loan, (tl + new_lt) * (tl_rate / 12) / (1 - (1 + tl_rate / 12) ** -60), tl_emi)
        tl = tl + new_lt
        pool_pay = pool.sum(1)                                              # deferred supplier payments are cleared first
        sup_due = due_s.sum(1)
        fixed_cash = fixed - stat_acc                                       # statutory deductions are withheld, paid next month
        out_planned = sup_due + pool_pay + fixed_cash + var_opex + interest + tax_pay + stat_pay + draw + capex + principal + d_wc_other + arrears_exp
        cash_pre = cash + collections + new_lt - out_planned
        buffer = cash_weeks * (fixed + interest) / 4.345
        draw_cc = np.where(cash_pre < buffer, np.minimum(np.maximum(0, cc_lim - std), buffer - cash_pre), 0.0)
        repay_cc = np.where(cash_pre > 1.5 * buffer, np.minimum(std, cash_pre - 1.2 * buffer), 0.0)
        std = std + draw_cc - repay_cc
        cash_pre = cash_pre + draw_cc - repay_cc
        # shortfall: the owner may put money in and cuts drawings; then statutory dues, suppliers and loan principal are delayed;
        # anything left becomes unpaid expenses carried to next month
        short = np.maximum(0, -cash_pre)
        cap_in = np.where((short > 0) & (rng.random(n) < infuse), short * rng.uniform(.5, 1.0, n), 0.0)
        short = np.maximum(0, short - cap_in)
        cut_draw = np.minimum(short, draw); short -= cut_draw; draw = draw - cut_draw
        defer_stat = np.minimum(short, stat_pay); short -= defer_stat
        defer_pool = np.minimum(short, pool_pay); short -= defer_pool
        defer_sup = np.minimum(short, sup_due); short -= defer_sup
        miss_emi = np.minimum(short, principal); short -= miss_emi
        paid_arrears = arrears_exp.copy()
        arrears_exp = short.copy()                                          # unpaid running costs, settled next month
        cash = cash_pre + cap_in + cut_draw + defer_stat + defer_pool + defer_sup + miss_emi + short
        stat_arrears = np.where(defer_stat > 0, stat_arrears + 1, 0.0)
        stat_due = stat_acc + defer_stat
        emi_arrears = np.where(miss_emi > 0.5 * principal + 1, emi_arrears + 1, np.maximum(0, emi_arrears - 1))
        paid_pool = pool_pay - defer_pool
        keep_frac = np.where(pool_pay > 0, defer_pool / np.maximum(pool_pay, 1e-9), 0.0)
        pool = pool * keep_frac[:, None]
        pool = np.concatenate([np.zeros((n, 1)), pool[:, :2], pool[:, 2:].sum(1, keepdims=True)], axis=1)   # age by a month
        pool[:, 0] += defer_sup
        sup_share = np.where(sup_due > 0, 1 - defer_sup / np.maximum(sup_due, 1e-9), 1.0)
        paid_by_age = due_s * sup_share[:, None]
        principal_paid = principal - miss_emi
        tl = tl - principal_paid
        tax_due = np.where(q_end, 0.0, tax_due)
        supplier_paid = paid_by_age.sum(1) + paid_pool
        ap_paid_late_w = (paid_by_age * ap_late[:, ages, ages]).sum(1) + paid_pool * 60
        ap_paid_ontime = (paid_by_age * ap_on[:, ages, ages]).sum(1)
        mse_paid = mse_share * supplier_paid
        mse_intime = mse_share * (paid_by_age * ap_in45[:, ages, ages]).sum(1)
        ap_amt = ap_amt - due_core
        ap_amt[:, -1] = np.maximum(0, ap_amt[:, -1])
        pay_bal = ap_amt.sum(1) + pool.sum(1)
        cash_out = out_planned - cut_draw - defer_stat - defer_pool - defer_sup - miss_emi - (arrears_exp) + repay_cc
        cash_in = collections + new_lt + draw_cc + cap_in
        F_low = np.maximum(0.0, np.minimum(cash, cash - (cash_in - cash_out)) - swing * cash_out * np.exp(rng.normal(0, .2, n)))
        wf = (fixed + interest) / 4.345
        cover = (F_low + np.maximum(0, cc_lim - std)) / np.maximum(wf, 1)
        prev_cover = cover
        # concentration drifts slowly
        top_c = np.clip(top_c * np.exp(rng.normal(0, .02, n)), .03, .8)
        hhi = top_c ** 2 + (1 - top_c) ** 2 / n_c
        rec = dict(rev=rev, exports=export_share * rev, cogs=cogs, fixed=fixed, payroll=payroll, var_opex=var_opex, dep=dep, interest=interest, tax=tax,
                   net=net, invoiced=rev, billed=billed, collections=collections, supplier_paid=supplier_paid, drawings=draw, capital_in=cap_in,
                   principal=principal_paid, new_lt=new_lt, capex=capex, cash_in=cash_in, cash_out=cash_out,
                   cash=cash, cash_low=F_low, recv=ar_amt.sum(1), inv=inv, oca=oca, pay=pay_bal, std=std, ltd=tl, oth_liab=oth + arrears_exp,
                   stat_due=stat_due, tax_due=tax_due, plant=plant, fa_net=plant + other_fa, cc_limit=np.where(has_cc, cc_lim, np.nan),
                   ar_overdue=(ar_amt * ar_past).sum(1), ar_90=(ar_amt * ar_past90).sum(1), ar_paid=collections, ar_paid_late_w=ar_paid_late_w,
                   ar_paid_ontime=ar_paid_ontime, ap_overdue=(ap_amt * ap_past).sum(1) + pool.sum(1), ap_30=(ap_amt * ap_past30).sum(1) + pool[:, 1:].sum(1),
                   ap_90=(ap_amt * ap_past90).sum(1) + pool[:, 3], ap_paid=supplier_paid, ap_paid_late_w=ap_paid_late_w, ap_paid_ontime=ap_paid_ontime,
                   mse_paid=mse_paid, mse_paid_intime=mse_intime, stat_accrual=stat_acc, top_cust=top_c, hhi_cust=hhi, top_supp=top_s,
                   emi_arrears=emi_arrears, stat_arrears=stat_arrears, recv_young=ar_amt[:, :2].sum(1),
                   recv_eligible=(ar_amt[:, :3] * (1 - ar_past[:, :3])).sum(1))
        for k, v in rec.items():
            F[k][:, t] = v
        cc_lim = cc_lim * (1.06) ** (1 / 12) * np.where((pressure > .5) & (stress_t > 0), 0.96, 1.0)   # banks trim limits of strained firms in a downturn

    F = {k: v[:, burn:] for k, v in F.items()}
    stress = stress[burn:]
    months = pd.period_range(C.START, periods=T, freq="M")

    # --- outcomes from the hidden process
    wf = (F["fixed"] + F["interest"]) / 4.345
    cover = (F["cash_low"] + np.nan_to_num(np.maximum(0, F["cc_limit"] - F["std"]))) / np.maximum(wf, 1)
    ap30s = F["ap_30"] / np.maximum(F["pay"], 1)
    ap90s = F["ap_90"] / np.maximum(F["pay"], 1)
    ev_stress = (cover < 4) | (ap30s >= 0.25) | (F["emi_arrears"] >= 1) | (F["stat_arrears"] >= 2)
    ev_default = (F["emi_arrears"] >= 2) | (ap90s >= 0.10) | (F["stat_arrears"] >= 3)
    alive = np.zeros((n, T), bool)
    for i in range(n):
        alive[i, entry[i]:] = True
    # exit six months after a default that does not clear
    run = np.zeros(n)
    for t in range(T):
        run = np.where(ev_default[:, t] & alive[:, t], run + 1, 0)
        gone = run >= 6
        if gone.any():
            alive[gone, t + 1:] = False
    for k in F:
        F[k] = np.where(alive, F[k], np.nan)
    y = _outcomes(ev_stress, ev_default, cover, alive, rng)
    firms = pd.DataFrame(dict(firm_id=[f"F{i:05d}" for i in range(n)], sector=sectors, family=fam, has_stock=has_stock, entry=entry,
                              r_true=r, coll=coll, pdisc=pdisc, owner=owner, cash_weeks=cash_weeks, flex=flex, k_rec=k_rec,
                              growth=g, terms_c=terms_c, terms_s=terms_s, mse_share=mse_share,
                              dq_bal=dq_bal, dq_cls=dq_cls, dq_party=dq_party, dq_due=dq_due, checks=checks))
    meta = dict(seed=seed, n=n, ev_stress=ev_stress, ev_default=ev_default, cover=np.where(alive, cover, np.nan),
                shock_hit=hits[:, burn:], shock_perm=perms[:, burn:], shock_path=shock_path[:, burn:])
    return Panel(F, firms, y, alive, months, stress, meta)


def _outcomes(ev_s, ev_d, cover, alive, rng, noise=0.05):
    n, T = ev_s.shape
    s90 = np.full((n, T), np.nan); d12 = np.full((n, T), np.nan); sev = np.full((n, T), np.nan)
    for t in range(T):
        if t + 3 < T:
            w = slice(t + 1, t + 4)
            obs = alive[:, w].all(1) | (~alive[:, w]).any(1)            # an exit counts as stress
            s90[:, t] = np.where(alive[:, t], (ev_s[:, w] | ~alive[:, w]).any(1), np.nan)
            sev[:, t] = np.where(alive[:, t], np.nanmin(np.where(alive[:, w], cover[:, w], -5.0), axis=1), np.nan)
        if t + 12 < T:
            w = slice(t + 1, t + 13)
            d12[:, t] = np.where(alive[:, t], (ev_d[:, w] | ~alive[:, w]).any(1), np.nan)
    for arr in (s90, d12):                      # 5% of events go unrecorded and an equal number of false events appear
        ok = ~np.isnan(arr)
        base = float(np.nanmean(arr))
        u = rng.random(arr.shape)
        to0 = ok & (arr == 1) & (u < noise)
        to1 = ok & (arr == 0) & (u < noise * base / max(1e-9, 1 - base))
        arr[to0] = 0; arr[to1] = 1
    # severity: lowest weeks of cover next quarter, capped at 8 weeks (beyond that a firm is not near stress, so ranking there adds nothing)
    return dict(stress90=s90, default12=d12, severity=np.clip(sev, -5, 8))
