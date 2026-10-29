"""The 54 measures, computed from monthly facts.

Input: a dict of N x T arrays (the same schema for the synthetic database and for an uploaded ledger, see facts.py), plus per-firm
attributes (family, data-quality figures). Output: dict measure id -> N x T array of values (NaN = not available).
Every measure is defined here once, so the database, peers, the Second Opinion and an uploaded file can never disagree.
Conventions: flows are totalled over the 12 months ending at month t; balances are month-end values; percentages are 0-100.
Where a ratio has no meaningful denominator the value is set to the end of the scale it deserves (for example no borrowing gives
interest cover 99, drawings taken while the business produced no cash gives 999): a business is never rewarded by a gap."""
from __future__ import annotations

import numpy as np

from . import config as C

BIG = 99.0
WORST_PCT = 999.0
WEEK = 4.345


# ------------------------------------------------------------------------------------------------ window helpers (axis 1 = months)
def roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sum over the w months ending at t; NaN unless all w months are present."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(np.where(ok, x, 0.0), axis=1)], axis=1)
    cn = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(ok, axis=1)], axis=1)
    out = np.full(x.shape, np.nan)
    if x.shape[1] >= w:
        s = cs[:, w:] - cs[:, :-w]
        n = cn[:, w:] - cn[:, :-w]
        out[:, w - 1:] = np.where(n == w, s, np.nan)
    return out


def roll_mean(x, w):
    return roll_sum(x, w) / w


def lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if k < x.shape[1]:
        out[:, k:] = x[:, :-k] if k > 0 else x
    return out


def windows(x: np.ndarray, w: int) -> np.ndarray:
    """N x T x w view of the last w months (NaN-padded at the start)."""
    pad = np.concatenate([np.full((x.shape[0], w - 1), np.nan), x], axis=1)
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(pad, w, axis=1)


def roll_std(x, w):
    v = windows(np.asarray(x, float), w)
    out = np.nanstd(v, axis=2)
    full = (~np.isnan(v)).sum(2) == w
    return np.where(full, out, np.nan)


def roll_min(x, w):
    v = windows(np.asarray(x, float), w)
    full = (~np.isnan(v)).sum(2) == w
    with np.errstate(all="ignore"):
        return np.where(full, np.nanmin(np.where(np.isnan(v), np.inf, v), axis=2), np.nan)


def roll_median(x, w):
    v = windows(np.asarray(x, float), w)
    full = (~np.isnan(v)).sum(2) == w
    with np.errstate(all="ignore"):
        return np.where(full, np.nanmedian(v, axis=2), np.nan)


def div(a, b, zero=np.nan):
    with np.errstate(all="ignore"):
        return np.where(np.abs(b) > 1e-9, a / b, zero)


# ------------------------------------------------------------------------------------------------ derived monthly quantities
def derived(f: dict) -> dict:
    d = {}
    d["opex"] = f["fixed"] + f["var_opex"]
    d["gp"] = f["rev"] - f["cogs"]
    d["ebitda"] = d["gp"] - d["opex"] + (f["other_inc"] if "other_inc" in f else 0.0)
    d["ebit"] = d["ebitda"] - f["dep"]
    d["ca"] = f["cash"] + f["recv"] + f["inv"] + f["oca"]
    d["cl"] = f["pay"] + f["std"] + f["oth_liab"] + f["stat_due"] + f["tax_due"]
    d["debt"] = f["std"] + f["ltd"]
    d["tl_all"] = d["cl"] + f["ltd"]
    d["ta"] = d["ca"] + f["fa_net"]
    d["equity"] = d["ta"] - d["tl_all"]
    d["wc"] = f["recv"] + f["inv"] + f["oca"] - f["pay"] - f["oth_liab"] - f["stat_due"] - f["tax_due"]
    d["headroom"] = np.where(np.isnan(f["cc_limit"]), 0.0, np.maximum(0.0, f["cc_limit"] - f["std"]))
    d["wf"] = roll_mean(f["fixed"] + f["interest"], 3) / WEEK                     # weekly fixed costs, last 3 months
    d["costbase"] = roll_mean(f["cogs"] + d["opex"] + f["interest"], 3)
    d["ocf1"] = f["net"] + f["dep"] - (d["wc"] - lag(d["wc"], 1))
    return d


def ocf(f, d, w):
    return roll_sum(f["net"], w) + roll_sum(f["dep"], w) - (d["wc"] - lag(d["wc"], w))


# ------------------------------------------------------------------------------------------------ the 54 measures
def compute(f: dict, firms) -> dict:
    """firms: DataFrame-like with columns family, dq_bal, dq_cls, dq_party, dq_due, checks (one row per firm)."""
    d = derived(f)
    N, T = f["rev"].shape
    r12 = roll_sum(f["rev"], 12)
    pos_rev = np.where(r12 > 0, r12, np.nan)
    ocf12 = ocf(f, d, 12)
    net12 = roll_sum(f["net"], 12)
    out = {}

    # Liquidity
    out["LQ01"] = div(d["ca"], d["cl"], BIG)
    out["LQ03"] = div(f["cash"], d["cl"], BIG)
    out["CR02"] = div(f["cash"] + d["headroom"], d["wf"])
    out["CR03"] = div(f["cash"], d["costbase"] / 30.44)
    out["CG02"] = 100 * div(ocf12, pos_rev)
    out["CG03"] = np.where(net12 > 0, 100 * div(ocf12, net12), np.nan)
    out["FL11"] = np.where(np.isnan(f["cc_limit"]), np.nan, 100 * div(f["std"], f["cc_limit"]))
    proj = (ocf(f, d, 3) - roll_sum(f["principal"], 3) - roll_sum(f["drawings"], 3) - roll_sum(f["capex"], 3)) / 3
    avail0 = f["cash"] + d["headroom"] - 0.5 * f["ap_overdue"]
    out["EW1"] = div(avail0 + np.minimum(0.0, 3 * proj), d["wf"])
    out["TB5"] = roll_median(div(f["cash_low"], d["wf"]), 3)

    # Collections and working capital
    inv12 = roll_sum(f["invoiced"], 12)
    out["RC01"] = div(f["recv"], inv12) * 365
    out["RC03"] = div(roll_sum(f["ar_paid_late_w"], 12), roll_sum(f["ar_paid"], 12))
    out["RC05"] = 100 * div(f["ar_overdue"], f["recv"], 0.0)
    out["RC08"] = 100 * div(f["ar_90"], f["recv"], 0.0)
    out["RC09"] = 100 * f["top_cust"]
    out["RC11"] = f["hhi_cust"]
    out["RC14"] = 100 * (div(f["recv"], lag(f["recv"], 12)) - 1) - 100 * (div(inv12, lag(inv12, 12)) - 1)
    beg = lag(f["recv"], 12)
    out["RC16"] = 100 * div(beg + inv12 - f["recv"], beg + inv12 - (f["recv"] - f["ar_overdue"]))
    stock = np.array([fm != "services" for fm in firms["family"]])[:, None]
    cogs12 = roll_sum(f["cogs"], 12)
    in01 = np.where(stock, div(f["inv"], cogs12) * 365, np.nan)
    py01 = div(f["pay"], roll_sum(f["billed"], 12)) * 365
    out["WC01"] = np.where(stock, in01, 0.0) + out["RC01"] - py01
    late3 = div(roll_sum(f["ar_paid_late_w"], 3), roll_sum(f["ar_paid"], 3))
    out["TB1"] = late3 - lag(late3, 3)

    # Payment discipline
    out["PY02"] = div(roll_sum(f["ap_paid_late_w"], 12), roll_sum(f["ap_paid"], 12))
    out["PY03"] = 100 * div(roll_sum(f["ap_paid_ontime"], 12), roll_sum(f["ap_paid"], 12))
    out["PY05"] = 100 * div(f["ap_overdue"], f["pay"], 0.0)
    out["PY06"] = 100 * f["top_supp"]
    acc3 = roll_mean(f["stat_accrual"], 3)
    out["PY09"] = np.where(acc3 > 0, div(np.maximum(f["stat_due"], 0), acc3), np.nan)
    plate3 = div(roll_sum(f["ap_paid_late_w"], 3), roll_sum(f["ap_paid"], 3))
    out["TB2"] = plate3 - lag(plate3, 3)
    mp = roll_sum(f["mse_paid"], 12)
    out["MS1"] = np.where(mp > 0, 100 * div(roll_sum(f["mse_paid_intime"], 12), mp), np.nan)

    # Debt service and funding
    debt, eq = d["debt"], d["equity"]
    out["FL01"] = np.where(debt <= 1, 0.0, np.where(eq > 0, div(debt, eq), BIG))
    out["FL02"] = np.where(eq > 0, div(d["tl_all"], eq), BIG)
    e12 = roll_sum(d["ebitda"], 12)
    nd = np.maximum(debt - f["cash"], 0.0)
    out["FL04"] = np.where(nd <= 1, 0.0, np.where(e12 > 0, div(nd, e12), BIG))
    i12 = roll_sum(f["interest"], 12)
    out["FL05"] = np.where(i12 <= 1, BIG, div(roll_sum(d["ebit"], 12), i12))
    ds12 = i12 + roll_sum(f["principal"], 12)
    out["FL06"] = np.where(ds12 <= 1, BIG, div(e12, ds12))
    out["FL07"] = np.where(debt <= 1, np.nan, 100 * div(f["std"], debt))
    dprev = lag(debt, 12)
    out["FL08"] = np.where(dprev <= 1, np.where(debt <= 1, 0.0, 100.0), 100 * (div(debt, dprev) - 1))
    up = (f["std"] - lag(f["std"], 1)) > np.maximum(1.0, 0.005 * np.abs(lag(f["std"], 1)))
    out["TB3"] = roll_sum(np.where(np.isnan(lag(f["std"], 1)), np.nan, up.astype(float)), 12)

    # Profitability
    out["PF01"] = 100 * div(roll_sum(d["gp"], 12), pos_rev)
    out["PF02"] = 100 * div(e12, pos_rev)
    out["PF04"] = 100 * div(net12, pos_rev)
    out["PF05"] = 100 * div(roll_sum(d["opex"], 12), pos_rev)
    out["PF10"] = 100 * div(i12, pos_rev)
    cm = div(r12 - cogs12 - roll_sum(f["var_opex"], 12), pos_rev)
    fixed12 = roll_sum(f["fixed"], 12) + roll_sum(f["dep"], 12) + i12
    be = div(fixed12, cm)
    out["PF12"] = np.where(cm > 0, 100 * div(r12 - be, pos_rev), -100.0)

    # Stability and stock
    out["RC19"] = 100 * (div(r12, lag(r12, 12)) - 1)
    out["RS01"] = 100 * div(roll_std(f["rev"], 12), roll_mean(f["rev"], 12))
    gm_m = 100 * div(d["gp"], np.where(f["rev"] > 0, f["rev"], np.nan))
    out["RS02"] = roll_std(gm_m, 12)
    out["RS03"] = 100 * roll_mean(np.where(np.isnan(f["net"]), np.nan, (f["net"] > 0).astype(float)), 12)
    worst = -roll_min(f["cash_in"] - f["cash_out"], 12)
    out["RS04"] = np.where(worst <= 0, 24.0, np.minimum(24.0, div(f["cash"], worst)))
    out["IN01"] = in01
    out["IN04"] = np.where(stock, 100 * (div(f["inv"], lag(f["inv"], 12)) - 1) - 100 * (div(cogs12, lag(cogs12, 12)) - 1), np.nan)

    # Owner behaviour
    dr12 = roll_sum(f["drawings"], 12)
    out["OW02"] = np.where(dr12 <= 1, 0.0, np.where(net12 > 0, 100 * div(dr12, net12), WORST_PCT))
    out["OW03"] = np.where(dr12 <= 1, 0.0, np.where(ocf12 > 0, 100 * div(dr12, ocf12), WORST_PCT))

    # Data integrity (per file / firm, the same at every month)
    one = np.where(np.isnan(f["rev"]), np.nan, 1.0)
    for mid, col in (("DQ01", "dq_bal"), ("DQ02", "dq_cls"), ("DQ03", "dq_party"), ("DQ04", "dq_due")):
        out[mid] = one * np.asarray(firms[col], float)[:, None]
    out["IC1"] = one * (100 * np.asarray(firms["checks"], float) / 7)[:, None]

    # a measure needs 12 months of history at the firm; earlier months stay empty
    hist = np.cumsum(~np.isnan(f["rev"]), axis=1)
    for k in out:
        out[k] = np.where(hist >= 12, out[k], np.nan)
        out[k] = np.where(np.isinf(out[k]), np.nan, out[k]).astype(np.float32)
    assert set(out) == set(C.MEASURES), set(C.MEASURES) ^ set(out)
    return out


# ------------------------------------------------------------------------------------------------ rating a value 0-100
def rate(values: np.ndarray, mid: str, fams) -> np.ndarray:
    """Piecewise-linear through the four boundaries at 20/40/60/80; one band-width beyond each end to 0 and 100; no cliff edges."""
    v = np.asarray(values, float)
    out = np.full(v.shape, np.nan, np.float32)
    fams = np.asarray(fams)
    for fam in np.unique(fams):
        b = C.bounds_for(mid, fam)
        rows = fams == fam
        if b is None:
            continue
        b = np.array(b, float)
        x = v[rows]
        if C.M[mid]["dir"] == "L":
            x, b = -x, -b
        xs = np.concatenate([[b[0] - (b[1] - b[0])], b, [b[3] + (b[3] - b[2])]])
        ys = np.array([0, 20, 40, 60, 80, 100], float)
        out[rows] = np.where(np.isnan(x), np.nan, np.interp(x, xs, ys))
    return out


def rate_all(meas: dict, fams) -> dict:
    return {m: rate(meas[m], m, fams) for m in C.MEASURES}
