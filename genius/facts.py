"""From an uploaded ledger to the monthly facts every other part of Genius reads (the same schema the synthetic database uses).

Each fact is defined the way the books record it: flows are monthly totals of postings, balances are month-end positions, receivable and
payable timing comes from invoice-level matching (by reference when the file has references, otherwise oldest-first per party), the
intra-month cash low from the daily bank balance. Customers whose payments go straight back out (round-tripping) are found first and
taken out of the sales and collections that Grit reads (version 3's red-team check)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .books import Books
from .ctx import Ctx

PLANT_EXCLUDE = ("land", "building", "premises", "furniture", "fixture", "office", "interior")
EXPORT_WORDS = ("export",)
CIRCULAR_DAYS, CIRCULAR_SHARE = 30, 0.5


def circular_flows(c: Ctx) -> pd.DataFrame:
    """Customers whose money flows straight back out (version 3, unchanged): within 30 days, to the same party name through a non-trade
    account, or in the same amount (within 1%) to a loan, advance or other non-trade asset. Flagged when at least half came back."""
    p = c.b.p[~c.b.p["is_opening"]]
    party = p["party"].fillna("")
    rec_in = p[(p["subtype"] == "receivable") & (p["credit"] > 0) & (party != "")]
    if rec_in.empty:
        return pd.DataFrame(columns=["party", "received", "sent_back", "share", "rule"])
    cash_out_entries = set(p.loc[p["subtype"].isin(("cash", "bank")) & (p["credit"] > 0), "entry_no"])
    out = p[p["entry_no"].isin(cash_out_entries) & (p["debit"] > 0) & ~p["subtype"].isin(("receivable", "payable", "cash", "bank"))
            & (p["account_type"] == "asset") & ~p["subtype"].isin(("fixed_asset", "inventory", "input_tax"))]
    rows, used = [], set()
    for name, g in rec_in.groupby("party"):
        back, rule = 0.0, set()
        for i, d, a, nm in zip(out.index, out["date"], out["debit"], out["party"].fillna("")):
            if i in used:
                continue
            win = g[(g["date"] <= d) & (g["date"] >= d - pd.Timedelta(days=CIRCULAR_DAYS))]
            if win.empty:
                continue
            if nm == name:
                back += a; used.add(i); rule.add("name")
            elif (abs(win["credit"] - a) <= 0.01 * a).any():
                back += a; used.add(i); rule.add("amount")
        got = float(g["credit"].sum())
        if got > 0 and back > 0:
            rows.append(dict(party=name, received=got, sent_back=back, share=back / got, rule="+".join(sorted(rule))))
    return pd.DataFrame(rows, columns=["party", "received", "sent_back", "share", "rule"])


def from_books(c: Ctx, cc_limit: float | None = None, mse: dict | None = None) -> tuple[dict, dict]:
    """(facts: dict of 1 x T arrays, info) for one business. mse = {supplier name: time limit in days} for Section 43B(h)."""
    b, months = c.b, c.months
    T = len(months)
    idx = months
    def ser(s):
        return s.reindex(idx).fillna(0.0).to_numpy(float)
    p = b.p
    inc = c.inc
    f = {}
    rev_acc = p[(p["account_type"] == "income") & (p["subtype"] != "other_income") & ~p["is_opening"]]
    exp_mask = rev_acc["account"].str.lower().str.contains("|".join(EXPORT_WORDS), na=False)
    f["rev"] = ser(inc["Revenue"])
    f["exports"] = ser((-rev_acc[exp_mask]["net"]).groupby(rev_acc[exp_mask]["month"]).sum()) if exp_mask.any() else np.zeros(T)
    f["cogs"] = ser(inc["Cost of goods sold"])
    opex = ser(inc["Operating expenses"])
    f["fixed"] = ser(c.sub_cost("payroll", "rent", "insurance", "utilities", "professional_fees"))
    f["payroll"] = ser(c.sub_cost("payroll"))
    f["var_opex"] = opex - f["fixed"]
    f["dep"] = ser(inc["Depreciation"]); f["interest"] = ser(inc["Finance cost"]); f["tax"] = ser(inc["Tax"])
    f["other_inc"] = ser(inc["Other income"]); f["net"] = ser(inc["Net profit"])
    f["invoiced"] = ser(c.invoiced()); f["billed"] = ser(c.billed())
    f["collections"] = ser(c.sub_flow("receivable", "credit")); f["supplier_paid"] = ser(c.sub_flow("payable", "debit"))
    f["drawings"] = ser(c.sub_flow("drawings", "debit")); f["capital_in"] = ser(c.sub_flow("capital", "credit"))
    f["principal"] = ser(c.sub_flow("long_term_debt", "debit")); f["new_lt"] = ser(c.sub_flow("long_term_debt", "credit"))
    f["capex"] = ser(c.sub_flow("fixed_asset", "debit"))
    cf = b.cash_flows_monthly().reindex(idx).fillna(0.0)
    f["cash_in"], f["cash_out"] = cf["cash_in"].to_numpy(float), cf["cash_out"].to_numpy(float)
    bs = c.bs
    f["cash"] = bs["cash"].to_numpy(float); f["recv"] = bs["receivables"].to_numpy(float); f["inv"] = bs["inventory"].to_numpy(float)
    f["oca"] = bs["other_current_assets"].to_numpy(float); f["pay"] = bs["payables"].to_numpy(float)
    f["std"] = bs["short_term_debt"].to_numpy(float); f["ltd"] = bs["long_term_debt"].to_numpy(float)
    stat = np.array([max(0.0, -c.sub_bal("statutory", "tax_payable", t=t)) for t in idx])
    f["stat_due"] = stat
    f["oth_liab"] = bs["other_liabilities"].to_numpy(float) - stat
    f["tax_due"] = np.zeros(T)
    f["fa_net"] = bs["fixed_assets_net"].to_numpy(float)
    # plant, machinery and equipment for the MSME test (land, buildings, furniture and fittings left out), net of depreciation pro rata
    fa_acc = b._acct[b._acct["subtype"] == "fixed_asset"].index
    keep = [a for a in fa_acc if not any(w in a.lower() for w in PLANT_EXCLUDE)]
    bm = b.balances_at_month_end()
    gross_all = bm.reindex(fa_acc).sum().reindex(idx).to_numpy(float) if len(fa_acc) else np.zeros(T)
    gross_keep = bm.reindex(keep).sum().reindex(idx).to_numpy(float) if keep else np.zeros(T)
    with np.errstate(all="ignore"):
        f["plant"] = np.where(gross_all > 0, f["fa_net"] * gross_keep / gross_all, 0.0)
    f["cc_limit"] = np.full(T, float(cc_limit) if cc_limit else np.nan)
    # receivables and payables timing
    for kind, pre in (("receivable", "ar"), ("payable", "ap")):
        overdue, o30, o90, paid, late_w, ontime, young, elig = (np.zeros(T) for _ in range(8))
        inv_all = b.invoices(kind)
        settled = inv_all[(inv_all["open"].abs() < 0.5) & inv_all["last_payment"].notna() & (inv_all["amount"] > 0) & ~inv_all["is_opening"].astype(bool)]
        pm = settled["last_payment"].dt.to_period("M")
        for j, t in enumerate(idx):
            a = c.ageing(kind, t)
            dpd = a["days_past_due"]
            overdue[j] = a.loc[dpd > 0, "open"].sum()
            o30[j] = a.loc[dpd > 30, "open"].sum()
            o90[j] = a.loc[dpd > 90, "open"].sum()
            if kind == "receivable":
                age = (c.date(t) - a["raised"]).dt.days
                young[j] = a.loc[age <= 60, "open"].sum()
                elig[j] = a.loc[(age <= 90) & (dpd <= 0), "open"].sum()
            x = settled[pm == t]
            paid[j] = x["amount"].sum()
            late_w[j] = (x["amount"] * x["days_late"]).sum()
            ontime[j] = x.loc[x["days_late"] <= 7, "amount"].sum()
        f[f"{pre}_overdue"], f[f"{pre}_paid"], f[f"{pre}_paid_late_w"], f[f"{pre}_paid_ontime"] = overdue, paid, late_w, ontime
        if kind == "receivable":
            f["ar_90"], f["recv_young"], f["recv_eligible"] = o90, young, elig
        else:
            f["ap_30"], f["ap_90"] = o30, o90
    # Section 43B(h): bills of micro and small suppliers paid within their limit
    mse = mse or {}
    ap = b.invoices("payable")
    ap = ap[(ap["open"].abs() < 0.5) & ap["last_payment"].notna() & (ap["amount"] > 0) & ~ap["is_opening"].astype(bool) & ap["party"].isin(mse)].copy()
    ap["lim"] = ap["party"].map(mse)
    ap["intime"] = (ap["last_payment"] - ap["raised"]).dt.days <= ap["lim"]
    pm = ap["last_payment"].dt.to_period("M")
    f["mse_paid"] = ser(ap.groupby(pm)["amount"].sum()) if len(ap) else np.zeros(T)
    f["mse_paid_intime"] = ser(ap[ap["intime"]].groupby(pm[ap["intime"]])["amount"].sum()) if len(ap) else np.zeros(T)
    f["stat_accrual"] = ser(c.sub_flow("statutory", "credit")) + ser(c.sub_flow("tax_payable", "credit"))
    top_c, hhi, top_s = (np.full(T, np.nan) for _ in range(3))
    for j, t in enumerate(idx):
        try:
            sh = c.party_shares("receivable", t)
            top_c[j] = float(sh.iloc[0]); hhi[j] = float((sh ** 2).sum())
        except Exception:
            pass
        try:
            top_s[j] = float(c.party_shares("payable", t).iloc[0])
        except Exception:
            pass
    f["top_cust"], f["hhi_cust"], f["top_supp"] = top_c, hhi, top_s
    cs = b.cash_series()
    low = cs.groupby(cs.index.to_period("M")).min().reindex(idx)
    f["cash_low"] = np.maximum(0.0, low.fillna(0.0).to_numpy(float))
    f["emi_arrears"] = np.full(T, np.nan); f["stat_arrears"] = np.full(T, np.nan)
    # round-tripped customers: out of the sales and collections Grit reads, and an integrity knock-out if one is a top-5 customer
    circ = circular_flows(c)
    flagged = sorted(circ.loc[circ["share"] >= CIRCULAR_SHARE, "party"]) if len(circ) else []
    grit_rev, grit_coll = f["rev"].copy(), f["collections"].copy()
    top5 = False
    if flagged:
        q = p[~p["is_opening"]]
        r_in = q[(q["subtype"] == "receivable") & (q["credit"] > 0) & q["party"].isin(flagged)].groupby("month")["credit"].sum()
        sale_e = set(q.loc[(q["subtype"] == "receivable") & (q["debit"] > 0) & q["party"].isin(flagged), "entry_no"])
        fake = -q[q["entry_no"].isin(sale_e) & (q["subtype"] == "revenue")].groupby("month")["net"].sum()
        grit_coll -= ser(r_in); grit_rev -= ser(fake)
        try:
            top5 = bool(set(c.party_shares("receivable", c.last).head(5).index) & set(flagged))
        except Exception:
            top5 = False
    out = {k: np.asarray(v, float).reshape(1, -1) for k, v in f.items()}
    info = dict(months=idx, circular=circ, flagged=flagged, top5_flagged=top5, grit_rev=grit_rev.reshape(1, -1), grit_coll=grit_coll.reshape(1, -1))
    return out, info


def firm_row(sector: str, family: str, report: dict, checks: list) -> pd.DataFrame:
    ch = report.get("checks", {})
    return pd.DataFrame([dict(sector=sector, family=family, dq_bal=float(ch.get("balanced_entries_pct", 100)),
                              dq_cls=float(ch.get("classified_value_pct", 100)), dq_party=float(ch.get("party_coverage_pct", 100)),
                              dq_due=float(ch.get("due_date_coverage_pct", 100)), checks=float(sum(1 for x in checks if x["passed"])))])
