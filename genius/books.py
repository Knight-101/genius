"""Books engine: turns parsed postings into a trial balance, statements, invoice-level receivable/payable
history and integrity checks. Everything downstream (metrics, ratings, forecast, chat) reads from here.

Sign convention: net = debit - credit. Revenue/liability/equity accounts therefore carry negative net.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OPEX_EXCLUDE = {"cogs", "depreciation", "finance_cost", "tax_expense"}
CASH_SUBTYPES = {"cash", "bank"}
FIXED_ASSET = {"fixed_asset", "accumulated_depreciation"}
DEBT_SHORT = {"short_term_debt"}
DEBT_LONG = {"long_term_debt"}
BUCKETS = [(-10**6, 0, "Not yet due"), (1, 30, "1-30 days"), (31, 60, "31-60 days"), (61, 90, "61-90 days"), (91, 10**6, "90+ days")]
DEFAULT_TERMS = 30


def _bucket(days_past_due: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= days_past_due <= hi:
            return name
    return "Not yet due"


class Books:
    def __init__(self, postings: pd.DataFrame, currency: str = "INR"):
        p = postings.copy()
        p["date"] = pd.to_datetime(p["date"])
        for c in ("due_date",):
            p[c] = pd.to_datetime(p[c], errors="coerce")
        for c in ("party", "invoice_ref", "narration", "voucher_type", "subtype", "account_type"):
            if c not in p:
                p[c] = None
        p["net"] = p["debit"].astype(float) - p["credit"].astype(float)
        vt = p["voucher_type"].fillna("").str.lower()
        nar = p["narration"].fillna("").str.lower()
        p["is_opening"] = (p["entry_no"].astype(str) == "OPENING-BAL") | vt.str.contains("opening") | nar.str.startswith("opening balance")
        p["month"] = p["date"].dt.to_period("M")
        p["_o"] = ~p["is_opening"]  # opening lines sort first on the same date
        self.p = p.sort_values(["date", "_o", "entry_no"], kind="stable").drop(columns="_o").reset_index(drop=True)
        self.currency = currency
        self.first = self.p["date"].min()
        self.last = self.p["date"].max()
        self.months = pd.period_range(self.first, self.last, freq="M")
        self._acct = (self.p.drop_duplicates("account")[["account", "account_type", "subtype"]].set_index("account"))

    # ------------------------------------------------------------------ scoping
    def asof(self, date) -> "Books":
        """Books as they stood at a past date (used for replays and back-tests)."""
        d = pd.Timestamp(date)
        return Books(self.p[self.p["date"] <= d].drop(columns=["net", "is_opening", "month"]), self.currency)

    # ------------------------------------------------------------------ balances
    def monthly_net(self) -> pd.DataFrame:
        m = self.p.pivot_table(index="account", columns="month", values="net", aggfunc="sum", fill_value=0.0)
        return m.reindex(columns=self.months, fill_value=0.0)

    def balances_at_month_end(self) -> pd.DataFrame:
        """Cumulative balance (debit - credit) per account at each month end."""
        return self.monthly_net().cumsum(axis=1)

    def trial_balance(self, asof=None) -> pd.DataFrame:
        d = pd.Timestamp(asof) if asof is not None else self.last
        g = self.p[self.p["date"] <= d].groupby("account")[["debit", "credit"]].sum()
        g["balance"] = g["debit"] - g["credit"]
        return g.join(self._acct)

    # ------------------------------------------------------------------ income statement
    def income_statement(self, freq: str = "M") -> pd.DataFrame:
        pl = self.p[self.p["account_type"].isin(["income", "expense"]) & ~self.p["is_opening"]].copy()
        per = pl["date"].dt.to_period(freq)
        pl["per"] = per
        sub = pl["subtype"].fillna("")
        typ = pl["account_type"]
        amt = -pl["net"]  # income positive, expense negative
        rows = {}
        rows["Revenue"] = amt[(typ == "income") & (sub != "other_income")]
        rows["Cost of goods sold"] = -amt[(typ == "expense") & (sub == "cogs")]
        rows["Operating expenses"] = -amt[(typ == "expense") & ~sub.isin(OPEX_EXCLUDE)]
        rows["Depreciation"] = -amt[(typ == "expense") & (sub == "depreciation")]
        rows["Finance cost"] = -amt[(typ == "expense") & (sub == "finance_cost")]
        rows["Tax"] = -amt[(typ == "expense") & (sub == "tax_expense")]
        rows["Other income"] = amt[(typ == "income") & (sub == "other_income")]
        out = pd.DataFrame({k: v.groupby(pl.loc[v.index, "per"]).sum() for k, v in rows.items()}).fillna(0.0)
        out = out.reindex(pd.period_range(self.first, self.last, freq=freq), fill_value=0.0)
        out["Gross profit"] = out["Revenue"] - out["Cost of goods sold"]
        out["EBITDA"] = out["Gross profit"] - out["Operating expenses"] + out["Other income"]
        out["EBIT"] = out["EBITDA"] - out["Depreciation"]
        out["Net profit"] = out["EBIT"] - out["Finance cost"] - out["Tax"]
        order = ["Revenue", "Cost of goods sold", "Gross profit", "Operating expenses", "Other income", "EBITDA", "Depreciation", "EBIT",
                 "Finance cost", "Tax", "Net profit"]
        return out[order]

    # ------------------------------------------------------------------ balance sheet
    def balance_sheet(self, asof=None) -> dict:
        tb = self.trial_balance(asof)
        sub = tb["subtype"].fillna("")
        typ = tb["account_type"].fillna("")
        b = tb["balance"]

        def s(mask):
            return float(b[mask].sum())

        cash = s(sub.isin(CASH_SUBTYPES))
        ar = s(sub == "receivable")
        inv = s(sub == "inventory")
        other_ca = s((typ == "asset") & ~sub.isin(CASH_SUBTYPES | {"receivable", "inventory"} | FIXED_ASSET))
        fa_gross = s(sub == "fixed_asset")
        acc_dep = s(sub == "accumulated_depreciation")
        liab = -b[typ == "liability"]
        ap = float(-b[sub == "payable"].sum())
        std = float(-b[sub.isin(DEBT_SHORT)].sum())
        ltd = float(-b[sub.isin(DEBT_LONG)].sum())
        other_l = float(liab.sum()) - ap - std - ltd
        eq_book = float(-b[typ == "equity"].sum())
        profit = float(-b[typ.isin(["income", "expense"])].sum())
        d = {
            "cash": cash, "receivables": ar, "inventory": inv, "other_current_assets": other_ca,
            "fixed_assets_gross": fa_gross, "accumulated_depreciation": acc_dep, "fixed_assets_net": fa_gross + acc_dep,
            "payables": ap, "short_term_debt": std, "long_term_debt": ltd, "other_liabilities": other_l,
            "equity_booked": eq_book, "retained_profit": profit,
        }
        d["total_assets"] = cash + ar + inv + other_ca + d["fixed_assets_net"]
        d["total_liabilities"] = ap + std + ltd + other_l
        d["total_equity"] = eq_book + profit
        d["gap"] = d["total_assets"] - d["total_liabilities"] - d["total_equity"]
        return d

    def balance_sheet_monthly(self) -> pd.DataFrame:
        rows = {m: self.balance_sheet(m.end_time.normalize()) for m in self.months}
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------ cash
    def cash_series(self, freq: str = "D") -> pd.Series:
        c = self.p[self.p["subtype"].isin(CASH_SUBTYPES)]
        s = c.groupby("date")["net"].sum().cumsum()
        idx = pd.date_range(self.first, self.last, freq="D")
        s = s.reindex(idx).ffill().fillna(0.0)
        if freq != "D":
            s = s.resample("ME").last()
        return s

    def cash_flows_monthly(self) -> pd.DataFrame:
        """Direct method: money in / out of bank by month, excluding opening balances and bank sweeps with debt."""
        c = self.p[self.p["subtype"].isin(CASH_SUBTYPES) & ~self.p["is_opening"]]
        g = c.groupby("month")[["debit", "credit"]].sum().reindex(self.months, fill_value=0.0)
        g.columns = ["cash_in", "cash_out"]
        g["net"] = g["cash_in"] - g["cash_out"]
        return g

    # ------------------------------------------------------------------ invoices (receivable / payable)
    def _match(self, kind: str):
        """Match settlements to invoices once (by reference when the file has them, otherwise oldest-first per party).
        Returns (invoices, allocations); positions at any past date are then derived from the allocations."""
        if not hasattr(self, "_mcache"):
            self._mcache = {}
        if kind in self._mcache:
            return self._mcache[kind]
        ledger = self.p[self.p["subtype"] == kind].copy()
        empty_inv = pd.DataFrame(columns=["inv_id", "party", "invoice_ref", "raised", "due", "amount", "is_opening"])
        empty_al = pd.DataFrame(columns=["inv_id", "date", "amount"])
        if ledger.empty:
            self._mcache[kind] = (empty_inv, empty_al)
            return self._mcache[kind]
        ledger["party"] = ledger["party"].fillna(ledger["account"])
        up = "debit" if kind == "receivable" else "credit"
        down = "credit" if kind == "receivable" else "debit"
        if not hasattr(self, "match_mode"):
            self.match_mode = {}
        if ledger["invoice_ref"].notna().mean() > 0.8:
            res = self._by_ref(ledger, up, down)
            self.match_mode[kind] = "reference"
        else:
            res = self._fifo(ledger, up, down)
            self.match_mode[kind] = "oldest_first"
        self._mcache[kind] = res
        return res

    def invoices(self, kind: str = "receivable", asof=None) -> pd.DataFrame:
        """Invoice-level history as it stood at `asof` (default: latest)."""
        assert kind in ("receivable", "payable")
        d = pd.Timestamp(asof) if asof is not None else self.last
        inv, al = self._match(kind)
        cols = ["party", "invoice_ref", "raised", "due", "amount", "settled", "open", "last_payment", "days_to_pay", "days_late", "is_opening"]
        if inv.empty:
            return pd.DataFrame(columns=cols)
        x = inv[inv["raised"] <= d].copy()
        a = al[al["date"] <= d]
        g = a.groupby("inv_id")["amount"].agg(["sum"]).rename(columns={"sum": "settled"})
        lp = a.groupby("inv_id")["date"].max().rename("last_payment")
        x = x.join(g, on="inv_id").join(lp, on="inv_id")
        x["settled"] = x["settled"].fillna(0.0)
        x["open"] = (x["amount"] - x["settled"]).round(2)
        miss = x["due"].isna()
        x.loc[miss, "due"] = x.loc[miss, "raised"] + pd.Timedelta(days=DEFAULT_TERMS)
        paid = x["open"].abs() < 0.5
        x["days_to_pay"] = np.where(paid, (x["last_payment"] - x["raised"]).dt.days, np.nan)
        x["days_late"] = np.where(paid, (x["last_payment"] - x["due"]).dt.days, np.nan)
        return x.sort_values(["raised", "invoice_ref"]).reset_index(drop=True)[cols]

    @staticmethod
    def _by_ref(ledger, up, down):
        ledger = ledger.copy()
        ledger["invoice_ref"] = ledger["invoice_ref"].fillna("UNREFERENCED")
        inv = (ledger[ledger[up] > 0].groupby(["party", "invoice_ref"], sort=False)
               .agg(raised=("date", "min"), due=("due_date", "min"), amount=(up, "sum"), is_opening=("is_opening", "all")).reset_index())
        inv["inv_id"] = np.arange(len(inv))
        st = ledger[ledger[down] > 0][["party", "invoice_ref", "date", down]].rename(columns={down: "amount"})
        st = st.merge(inv[["party", "invoice_ref", "inv_id"]], on=["party", "invoice_ref"], how="left")
        orphan = st[st["inv_id"].isna()]
        if len(orphan):  # settlement with no invoice: keep as unapplied credit so the ledger still ties
            o = (orphan.groupby(["party", "invoice_ref"], sort=False).agg(raised=("date", "min"), amount=("amount", "sum")).reset_index())
            o["amount"] = -o["amount"]
            o["due"] = pd.NaT
            o["is_opening"] = False
            o["inv_id"] = np.arange(len(inv), len(inv) + len(o))
            inv = pd.concat([inv, o], ignore_index=True)
        al = st.dropna(subset=["inv_id"])[["inv_id", "date", "amount"]].astype({"inv_id": int})
        return inv[["inv_id", "party", "invoice_ref", "raised", "due", "amount", "is_opening"]], al

    @staticmethod
    def _fifo(ledger, up, down):
        inv_rows, al_rows = [], []
        for party, g in ledger.groupby("party", sort=False):
            g = g.assign(_o=~g["is_opening"]).sort_values(["date", "_o", "entry_no"], kind="stable")
            queue = []
            for r in g.itertuples():
                if getattr(r, up) > 0:
                    queue.append(dict(inv_id=len(inv_rows) + len(queue), party=party, invoice_ref=r.invoice_ref or f"{r.entry_no}", raised=r.date,
                                      due=r.due_date, amount=getattr(r, up), left=getattr(r, up), is_opening=bool(r.is_opening)))
                amt = getattr(r, down)
                for q in queue:
                    if amt <= 0.004:
                        break
                    take = min(q["left"], amt)
                    if take > 0:
                        q["left"] -= take
                        al_rows.append((q["inv_id"], r.date, take))
                        amt -= take
                if amt > 0.004:
                    queue.append(dict(inv_id=len(inv_rows) + len(queue), party=party, invoice_ref="UNAPPLIED", raised=r.date, due=pd.NaT,
                                      amount=-amt, left=0.0, is_opening=False))
                queue = [q for q in queue] 
            inv_rows.extend(queue)
        inv = pd.DataFrame(inv_rows)
        inv = inv.drop(columns="left") if len(inv) else inv
        return inv, pd.DataFrame(al_rows, columns=["inv_id", "date", "amount"])

    def ageing(self, kind: str = "receivable", asof=None) -> pd.DataFrame:
        d = pd.Timestamp(asof) if asof is not None else self.last
        inv = self.invoices(kind, d)
        op = inv[inv["open"].abs() >= 0.5].copy()
        op["days_past_due"] = (d - op["due"]).dt.days
        op["bucket"] = op["days_past_due"].map(_bucket)
        return op

    def ageing_summary(self, kind: str = "receivable", asof=None) -> pd.DataFrame:
        a = self.ageing(kind, asof)
        names = [n for _, _, n in BUCKETS]
        t = a.pivot_table(index="party", columns="bucket", values="open", aggfunc="sum", fill_value=0.0).reindex(columns=names, fill_value=0.0)
        t["Total"] = t.sum(axis=1)
        return t.sort_values("Total", ascending=False)

    def party_payment_history(self, party: str, kind: str = "receivable") -> pd.DataFrame:
        inv = self.invoices(kind)
        return inv[inv["party"].str.lower() == party.lower()].reset_index(drop=True)

    def party_share(self, kind: str = "receivable", window_days: int = 365) -> pd.Series:
        """Share of invoiced value by party over the trailing window (gross of tax; the books carry tax inside the receivable)."""
        inv = self.invoices(kind)
        inv = inv[~inv["is_opening"].astype(bool)]
        inv = inv[inv["raised"] > self.last - pd.Timedelta(days=window_days)]
        s = inv.groupby("party")["amount"].sum().sort_values(ascending=False)
        return s / s.sum() if s.sum() else s

    # ------------------------------------------------------------------ integrity checks
    def checks(self) -> list[dict]:
        out = []

        def add(name, ok, detail):
            out.append({"check": name, "passed": bool(ok), "detail": detail})

        tb = self.trial_balance()
        add("Trial balance nets to zero", abs(tb["balance"].sum()) < 1.0, f"difference {tb['balance'].sum():,.2f}")
        g = self.p.groupby("entry_no")[["debit", "credit"]].sum()
        bad = int(((g["debit"] - g["credit"]).abs() > 0.51).sum())
        add("Every entry balances", bad == 0, f"{bad} of {len(g)} entries out of balance")
        gaps = self.balance_sheet_monthly()["gap"].abs()
        add("Assets = liabilities + equity at every month end", gaps.max() < 1.0, f"largest gap {gaps.max():,.2f}")
        unk = self.p[self.p["account_type"].isna()]
        add("All accounts classified", unk.empty, f"{unk['account'].nunique()} unclassified accounts")
        cash = self.cash_series()
        add("Cash never negative (bank accounts)", (cash >= -1.0).all(), f"lowest balance {cash.min():,.0f}")
        isg = self.income_statement("M")
        bs0 = self.balance_sheet(self.first - pd.Timedelta(days=1))["retained_profit"] if False else 0.0
        add("Profit & loss ties to retained profit", abs(isg["Net profit"].sum() - self.balance_sheet()["retained_profit"]) < 1.0,
            f"P&L {isg['Net profit'].sum():,.0f} vs balance sheet {self.balance_sheet()['retained_profit']:,.0f}")
        ar_ledger = self.balance_sheet()["receivables"]
        inv = self.invoices("receivable")
        add("Receivable ledger ties to open invoices", abs(ar_ledger - inv["open"].sum()) < 1.0, f"ledger {ar_ledger:,.0f} vs invoices {inv['open'].sum():,.0f}")
        return out
