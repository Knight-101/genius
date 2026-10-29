"""Early warning: a 13-week cash projection built from the invoices themselves, and one alert.

The alert (one rule, so it can be tested)
  "Cash cover will drop below 4 weeks within 6 weeks."  Cash cover = money available / weekly fixed costs.
  Money available = cash and bank, plus any unused cash-credit limit the owner typed in. Weekly fixed costs = payroll, rent, insurance, utilities,
  professional fees and interest over the last 13 weeks, divided by 13 (the same cost base as the Cash cover measure on the Health tab).

How each week of the projection is built (everything from the books up to the as-of date; nothing from the future)
  1. Customers' open invoices. Each is expected on its due date plus that customer's usual lateness (median days late over the last 180 days).
     An invoice already later than that is assumed to take as long again as it already has (at least one more week): a customer who has turned slow
     is not assumed to pay next week.
  2. New sales. The last 8 weeks' invoicing run-rate, collected after the usual days-to-pay (value-weighted median over 180 days).
  3. Suppliers' open bills. Each is paid on its due date plus our usual lateness to that supplier; a bill already past that is paid within a week.
  4. New purchases. The last 8 weeks' billing run-rate, paid after the usual days-to-pay.
  5. Everything else that moves cash (payroll, rent, taxes, drawings, loan instalments): the average week of the last 13. Borrowing drawn or repaid on a
     cash-credit or overdraft account is left out, because it is a choice, not a flow.
The drivers list names the invoices that move the answer: overdue customer invoices not expected before the breach, and bills that fall due before it.
Limits: weekly, not daily; no seasonality in the run-rates; a planned big payment the books do not know about is missed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 13          # weeks projected
ALERT_WEEKS = 6       # the alert looks this far ahead
THRESHOLD = 4.0       # weeks of fixed costs
LOOKBACK = 180        # days of payment history used for lateness
PERSIST = 1           # projected weeks in a row below the threshold needed to fire
RECENT = 0            # days; when set, a customer's recent lateness counts if worse than its habit. Off: it made results worse in the back-test
RUN_RATE_WEEKS = 8
FIXED_SUBS = ("payroll", "rent", "insurance", "utilities", "professional_fees", "finance_cost")
CASH = ("cash", "bank")


def _entries(b):
    """Per entry touching cash: date, cash delta, and what kind of flow it is (cached on the Books object)."""
    if getattr(b, "_al_cache", None) is not None:
        return b._al_cache
    p = b.p[~b.p["is_opening"]]
    is_cash = p["subtype"].isin(CASH)
    cash_e = p.loc[is_cash, "entry_no"].unique()
    q = p[p["entry_no"].isin(cash_e)]
    g = q.groupby("entry_no")
    delta = q[q["subtype"].isin(CASH)].groupby("entry_no")["net"].sum()
    subs = g["subtype"].agg(lambda s: set(s))
    date = g["date"].min()
    kind = []
    for e in delta.index:
        s = subs[e]
        if "receivable" in s:
            kind.append("ar")
        elif "payable" in s:
            kind.append("ap")
        elif "short_term_debt" in s:
            kind.append("cc")
        elif s & {"long_term_debt", "capital"} and delta[e] > 0:
            kind.append("fin_in")
        elif s <= set(CASH):
            kind.append("transfer")
        else:
            kind.append("other")
    out = pd.DataFrame(dict(date=date.reindex(delta.index), delta=delta, kind=kind))
    b._al_cache = out
    return out


def weekly_fixed(b, d) -> float:
    d = pd.Timestamp(d)
    p = b.p[(~b.p["is_opening"]) & (b.p["date"] > d - pd.Timedelta(days=91)) & (b.p["date"] <= d)]
    x = p[(p["account_type"] == "expense") & p["subtype"].isin(FIXED_SUBS)]["net"].sum()
    return float(x) / 13.0


def cash_at(b, d) -> float:
    p = b.p[(b.p["date"] <= d) & b.p["subtype"].isin(CASH)]
    return float(p["net"].sum())


def st_debt_at(b, d) -> float:
    p = b.p[(b.p["date"] <= d) & (b.p["subtype"] == "short_term_debt")]
    return float(-p["net"].sum())


def _lateness(inv: pd.DataFrame, d) -> tuple[dict, float, float]:
    """(median days late per party, overall median days late, value-weighted median days to pay) over the lookback."""
    x = inv[(inv["open"].abs() < 0.5) & inv["last_payment"].notna() & (inv["last_payment"] > d - pd.Timedelta(days=LOOKBACK))
            & (inv["amount"] > 0) & ~inv["is_opening"].astype(bool)]
    if x.empty:
        return {}, 0.0, 30.0
    per = x.groupby("party")["days_late"].median()
    if RECENT:
        r = x[x["last_payment"] > d - pd.Timedelta(days=RECENT)].groupby("party")["days_late"].median()
        per = pd.concat([per, r], axis=1).max(axis=1)
    per = per.to_dict()
    o = x.sort_values("days_to_pay")
    cw = o["amount"].cumsum() / o["amount"].sum()
    dtp = float(o.loc[cw >= 0.5, "days_to_pay"].iloc[0])
    return per, float(x["days_late"].median()), dtp


def _run_rate(inv: pd.DataFrame, d) -> float:
    x = inv[(inv["raised"] > d - pd.Timedelta(weeks=RUN_RATE_WEEKS)) & (inv["raised"] <= d) & (inv["amount"] > 0) & ~inv["is_opening"].astype(bool)]
    return float(x["amount"].sum()) / RUN_RATE_WEEKS


def project(b, asof=None, cc_limit: float | None = None, threshold: float | None = None, alert_weeks: int | None = None, horizon: int = HORIZON) -> dict:
    threshold = THRESHOLD if threshold is None else threshold
    alert_weeks = ALERT_WEEKS if alert_weeks is None else alert_weeks
    d = pd.Timestamp(asof) if asof is not None else b.last
    d = d.normalize()
    F = weekly_fixed(b, d)
    cash0 = cash_at(b, d)
    headroom = max(0.0, cc_limit - st_debt_at(b, d)) if cc_limit else 0.0
    avail0 = cash0 + headroom
    weeks = [d + pd.Timedelta(weeks=k) for k in range(1, horizon + 1)]
    edges = [d] + weeks

    def bucket(dates: pd.Series, amounts: pd.Series) -> np.ndarray:
        out = np.zeros(horizon)
        for dt_, a in zip(dates, amounts):
            if pd.isna(dt_) or dt_ <= d or dt_ > edges[-1]:
                continue
            k = int(np.ceil((dt_ - d).days / 7.0)) - 1
            out[min(max(k, 0), horizon - 1)] += a
        return out

    rows = {}
    drivers = {}
    for kind, sign in (("receivable", +1), ("payable", -1)):
        inv = b.invoices(kind, d)
        per, overall, dtp = _lateness(inv, d)
        op = inv[(inv["open"] > 0.5) & (inv["amount"] > 0)].copy()
        op["usual_late"] = op["party"].map(per).fillna(overall).clip(lower=0)
        op["dpd"] = (d - op["due"]).dt.days
        exp = op["due"] + pd.to_timedelta(op["usual_late"], unit="D")
        late_now = exp <= d
        if kind == "receivable":
            exp = exp.where(~late_now, d + pd.to_timedelta(np.maximum(7, op["dpd"].clip(lower=0)), unit="D"))
        else:
            exp = exp.where(~late_now, d + pd.Timedelta(days=7))
        op["expected"] = exp
        rows[f"{kind}_open"] = sign * bucket(op["expected"], op["open"])
        # new invoices at the run-rate, settled after the usual days to pay
        rr = _run_rate(inv, d)
        new = np.zeros(horizon)
        for j in range(horizon):
            when = d + pd.Timedelta(days=7 * j + 3.5 + dtp)
            k = int(np.ceil((when - d).days / 7.0)) - 1
            if 0 <= k < horizon:
                new[k] += rr
        rows[f"{kind}_new"] = sign * new
        drivers[kind] = op
    ent = _entries(b)
    recent = ent[(ent["date"] > d - pd.Timedelta(days=91)) & (ent["date"] <= d)]
    other_w = float(recent.loc[recent["kind"].isin(["other"]), "delta"].sum()) / 13.0
    rows["other"] = np.full(horizon, other_w)
    tbl = pd.DataFrame(rows)
    tbl.insert(0, "week_ending", weeks)
    tbl["net"] = tbl.drop(columns="week_ending").sum(axis=1)
    tbl["available"] = avail0 + tbl["net"].cumsum()
    tbl["cover"] = tbl["available"] / F if F > 0 else np.nan
    cover_now = avail0 / F if F > 0 else None
    win = tbl.head(alert_weeks)
    lowv = (win["cover"] < threshold).to_numpy()
    ok = np.array([lowv[i:i + PERSIST].all() and i + PERSIST <= len(lowv) for i in range(len(lowv))])
    breach = win[ok]
    fired = bool(len(breach)) and F > 0
    k_low = int(tbl["cover"].idxmin()) if F > 0 else None
    alert = dict(fired=fired, threshold=threshold, alert_weeks=alert_weeks, cover_now=cover_now,
                 breach_week=int(breach.index[0]) + 1 if fired else None, breach_date=breach["week_ending"].iloc[0].date().isoformat() if fired else None,
                 lowest_cover=float(tbl["cover"].min()) if F > 0 else None, lowest_week=None if k_low is None else k_low + 1,
                 lowest_date=None if k_low is None else tbl["week_ending"].iloc[k_low].date().isoformat())
    if F <= 0:
        alert["message"] = "No fixed costs found in the last 13 weeks, so cash cover cannot be measured."
    elif fired:
        cv = float(breach["cover"].iloc[0])
        level = (f"about {cv:.1f} weeks" if cv >= 0 else f"cash runs out, about {abs(cv):.1f} weeks of fixed costs short")
        alert["message"] = (f"Cash cover will drop below {threshold:g} weeks within {alert_weeks} weeks: {level} in week {alert['breach_week']} "
                            f"(week ending {alert['breach_date']}).")
    else:
        alert["message"] = (f"No alert: cash cover stays above {threshold:g} weeks for the next {alert_weeks} weeks "
                            f"(lowest {alert['lowest_cover']:.1f} weeks in week {alert['lowest_week']} of {horizon}).")
    # what moves the answer: customers' overdue invoices not expected before the low point, and bills due before it
    cut = d + pd.Timedelta(weeks=alert["breach_week"] if fired else (alert["lowest_week"] or alert_weeks))
    ar = drivers["receivable"]
    ar_drv = ar[(ar["dpd"] > 0)].assign(expected_after_low=lambda x: x["expected"] > cut).sort_values("open", ascending=False)
    ap = drivers["payable"]
    ap_drv = ap[ap["expected"] <= cut].sort_values("expected")
    return dict(asof=d.date().isoformat(), weekly_fixed=F, cash=cash0, headroom=headroom, available=avail0, table=tbl, alert=alert,
                overdue_customers=ar_drv[["party", "invoice_ref", "raised", "due", "open", "dpd", "usual_late", "expected", "expected_after_low"]].head(15),
                bills_before_low=ap_drv[["party", "invoice_ref", "due", "open", "expected"]].head(15),
                overdue_total=float(ar_drv["open"].sum()), overdue_after_low=float(ar_drv.loc[ar_drv["expected_after_low"], "open"].sum()))


def realised_cover(b, d, cc_limit: float | None = None) -> float | None:
    """Actual cash cover at date d from the books (used by the back-test as the truth)."""
    d = pd.Timestamp(d)
    F = weekly_fixed(b, d)
    if F <= 0:
        return None
    head = max(0.0, cc_limit - st_debt_at(b, d)) if cc_limit else 0.0
    return (cash_at(b, d) + head) / F
