"""The 45-day squeeze: India's Section 43B(h) rule for payments to micro and small suppliers, seen from a trader's books.

Why this matters for a distributor
  Section 43B(h) of the Income-tax Act (from FY 2023-24) allows a buyer to deduct a sum owed to a micro or small enterprise only in the year it is
  actually paid, unless it is paid within the time limit in Section 15 of the MSMED Act: the agreed date, capped at 45 days from acceptance, or 15 days
  when there is no written agreement. Late payment also carries compound interest at three times the RBI bank rate with monthly rests (Section 16),
  and that interest is never deductible (Section 23).
  The protection runs one way. Traders are not covered as suppliers (Udyam registration gives them priority-sector lending only), so a distributor must
  pay its small manufacturers within 45 days while the retailers who owe it money face no such clock. Only the ledger shows both sides of that squeeze.

What Genius measures (all from the payables and receivables ledgers)
  - Which suppliers count: the owner ticks the suppliers registered on Udyam as Micro or Small manufacturers or service providers, with the time limit
    for each (45 days with a written agreement, 15 without). Genius never guesses this. Sample defaults live in config/msme_suppliers.csv.
  - Days are counted from the bill date (a stand-in for the date of acceptance) to the date the bill was fully paid.
  - Today: open bills already past their limit, and bills that will pass it within 30 days.
  - Year-end test: for each 31 March in the books, bills still outstanding on that date and paid after their limit (or not paid yet). That amount is
    disallowed for that year and deducted only when paid; the tax effect is shown at the owner's tax rate (default 25.17%, a company under Section 115BAA).
  - Section 16 interest: what the late-paid and still-late bills would carry at 3 x bank rate, compounded monthly. Suppliers rarely claim it, but it is a
    legal liability and is shown as such.
  - The squeeze: the days customers actually take to pay (value-weighted) against the 45-day clock, and the money needed to fund that gap.

Honest limits: the bill date stands in for the date of acceptance; whether trading purchases held in closing stock fall under 43B(h) is debated, so the
year-end figure is the most a tax officer could disallow, not a tax computation. This is a planning aid, not tax advice.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MSE_FILE = ROOT / "config" / "msme_suppliers.csv"
BANK_RATE = 5.75          # RBI Bank Rate (%), after the 7 October 2026 MPC decision. Update when the RBI changes it.
BANK_RATE_ASOF = "7 Oct 2026"
DEFAULT_TAX = 25.17       # company tax under Section 115BAA incl. surcharge and cess
LIMIT_AGREED, LIMIT_NONE = 45, 15


def load_mse_list(path: Path | str = MSE_FILE) -> dict[str, int]:
    """{supplier name: time limit in days} from the CSV (columns: supplier, udyam_category, written_agreement)."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = (r.get("supplier") or "").strip()
            cat = (r.get("udyam_category") or "").strip().lower()
            if not name or cat not in ("micro", "small"):
                continue
            agreed = (r.get("written_agreement") or "yes").strip().lower() in ("yes", "y", "true", "1")
            out[name] = LIMIT_AGREED if agreed else LIMIT_NONE
    return out


def suppliers_in_books(c) -> list[str]:
    ap = c.ap_inv
    ap = ap[~ap["is_opening"].astype(bool) & (ap["amount"] > 0)]
    return ap.groupby("party")["amount"].sum().sort_values(ascending=False).index.tolist()


def _fy_ends(first: pd.Timestamp, last: pd.Timestamp) -> list[pd.Timestamp]:
    ends = []
    for y in range(first.year, last.year + 1):
        d = pd.Timestamp(year=y, month=3, day=31)
        if first <= d <= last:
            ends.append(d)
    return ends


def _interest(amount: float, days_late: float, bank_rate: float) -> float:
    """Section 16: compound interest with monthly rests at 3 x bank rate, for the days beyond the limit."""
    if days_late <= 0:
        return 0.0
    r = 3 * bank_rate / 100 / 12
    return float(amount * ((1 + r) ** (days_late / 30.4375) - 1))


def analyse(c, mse: dict[str, int] | None = None, tax_rate: float = DEFAULT_TAX, bank_rate: float = BANK_RATE, asof=None) -> dict:
    """Everything the 45-day tab shows. mse: {supplier: limit days}; None reads config/msme_suppliers.csv."""
    if c.cur != "INR":
        return dict(status="na", note="Section 43B(h) is an Indian rule; it applies only to books in INR.")
    mse = load_mse_list() if mse is None else dict(mse)
    names = set(suppliers_in_books(c))
    mse = {k: v for k, v in mse.items() if k in names}
    if not mse:
        return dict(status="need_input", note="Tick the suppliers registered on Udyam as Micro or Small (manufacturers or service providers). Genius does not guess this.",
                    suppliers=sorted(names))
    d = pd.Timestamp(asof) if asof is not None else c.b.last
    inv = c.b.invoices("payable", d)
    inv = inv[~inv["is_opening"].astype(bool) & (inv["amount"] > 0) & inv["party"].isin(mse)].copy()
    inv["limit"] = inv["party"].map(mse)
    inv["limit_date"] = inv["raised"] + pd.to_timedelta(inv["limit"], unit="D")
    paid = inv["open"].abs() < 0.5
    inv["paid_on"] = np.where(paid, inv["last_payment"], pd.NaT)
    inv["paid_on"] = pd.to_datetime(inv["paid_on"])
    end_or_today = inv["paid_on"].fillna(d)
    inv["days_beyond"] = (end_or_today - inv["limit_date"]).dt.days.clip(lower=0)
    inv["late"] = inv["days_beyond"] > 0

    # ---- today
    op = inv[~paid].copy()
    op["age"] = (d - op["raised"]).dt.days
    past = op[op["age"] > op["limit"]]
    soon = op[(op["age"] <= op["limit"]) & (op["age"] > op["limit"] - 30)]

    # ---- trailing twelve months of settled bills
    ttm = inv[paid & (inv["paid_on"] > d - pd.Timedelta(days=365))]
    on_time_share = float((~ttm["late"]).mean()) if len(ttm) else None
    late_value_share = float(ttm.loc[ttm["late"], "amount"].sum() / ttm["amount"].sum()) if len(ttm) and ttm["amount"].sum() > 0 else None
    avg_days_mse = float(np.average((ttm["paid_on"] - ttm["raised"]).dt.days, weights=ttm["amount"])) if len(ttm) else None

    # ---- year-end test at each 31 March in the books
    fy = []
    full = c.b.invoices("payable")      # latest knowledge of when each bill was finally paid
    full = full[~full["is_opening"].astype(bool) & (full["amount"] > 0) & full["party"].isin(mse)].copy()
    full["limit_date"] = full["raised"] + pd.to_timedelta(full["party"].map(mse), unit="D")
    full_paid = full["open"].abs() < 0.5
    for ye in _fy_ends(c.b.first, d):
        at = c.b.invoices("payable", ye)
        at = at[~at["is_opening"].astype(bool) & (at["amount"] > 0) & at["party"].isin(mse) & (at["open"] > 0.5)]
        if at.empty:
            fy.append(dict(fy_end=ye.date().isoformat(), outstanding=0.0, disallowed=0.0, tax_effect=0.0, bills=0))
            continue
        j = at[["party", "invoice_ref", "open"]].merge(full[["party", "invoice_ref", "limit_date", "last_payment", "open"]].rename(columns={"open": "open_now"}),
                                                       on=["party", "invoice_ref"], how="left")
        settled_late = (j["open_now"].abs() < 0.5) & (j["last_payment"] > j["limit_date"])
        unpaid_past = (j["open_now"].abs() >= 0.5) & (d > j["limit_date"])
        dis = j[settled_late | unpaid_past]
        fy.append(dict(fy_end=ye.date().isoformat(), outstanding=float(at["open"].sum()), disallowed=float(dis["open"].sum()),
                       tax_effect=float(dis["open"].sum() * tax_rate / 100), bills=int(len(dis))))

    # ---- what-if today were 31 March
    at_risk_now = float(past["open"].sum())

    # ---- Section 16 interest (TTM late-paid + currently late)
    rows = pd.concat([ttm[ttm["late"]], inv[~paid & inv["late"]]])
    s16 = float(sum(_interest(a, b, bank_rate) for a, b in zip(rows["amount"], rows["days_beyond"])))

    # ---- the squeeze: customers' clock against the 45-day clock
    ar = c.b.invoices("receivable", d)
    ar = ar[~ar["is_opening"].astype(bool) & (ar["amount"] > 0) & (ar["open"].abs() < 0.5) & (ar["last_payment"] > d - pd.Timedelta(days=365))]
    cust_days = float(np.average(ar["days_to_pay"], weights=ar["amount"])) if len(ar) else None
    mse_bills_ttm = inv[inv["raised"] > d - pd.Timedelta(days=365)]["amount"].sum()
    gap_days = None if cust_days is None else max(0.0, cust_days - LIMIT_AGREED)
    gap_money = None if gap_days is None else float(mse_bills_ttm / 365 * gap_days)

    # ---- monthly picture for the chart
    m = inv[paid].copy()
    m["month"] = m["paid_on"].dt.to_period("M")
    monthly = m.groupby("month").apply(lambda g: pd.Series(dict(paid=g["amount"].sum(), late=g.loc[g["late"], "amount"].sum())), include_groups=False)

    by_sup = []
    for sname, lim in sorted(mse.items()):
        g = ttm[ttm["party"] == sname]
        o = op[op["party"] == sname]
        by_sup.append(dict(supplier=sname, limit=lim, bills_ttm=int(len(g)), paid_on_time=None if g.empty else float((~g["late"]).mean()),
                           median_days=None if g.empty else float((g["paid_on"] - g["raised"]).dt.days.median()),
                           open=float(o["open"].sum()), open_past_limit=float(o.loc[o["age"] > o["limit"], "open"].sum())))

    past_list = past.sort_values("age", ascending=False)[["party", "invoice_ref", "raised", "limit_date", "open", "age", "limit"]].head(15)
    return dict(status="ok", asof=d.date().isoformat(), suppliers=mse, tax_rate=tax_rate, bank_rate=bank_rate,
                past_limit=float(past["open"].sum()), past_limit_n=int(len(past)), due_30=float(soon["open"].sum()), due_30_n=int(len(soon)),
                on_time_share=on_time_share, late_value_share=late_value_share, avg_days_mse=avg_days_mse,
                fy=fy, at_risk_now=at_risk_now, tax_at_risk_now=at_risk_now * tax_rate / 100, s16_interest=s16,
                customer_days=cust_days, gap_days=gap_days, gap_money=gap_money, monthly=monthly, by_supplier=by_sup, past_list=past_list,
                note="Planning aid, not tax advice. Bill date stands in for the date of acceptance.")
