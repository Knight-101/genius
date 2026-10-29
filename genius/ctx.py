"""Calculation context: precomputed monthly tables and helpers shared by every metric."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .books import Books


class NoData(Exception):
    """The books do not contain enough to compute this metric."""


class NotApplicable(NoData):
    """The metric does not apply to this business (for example stock metrics for a services firm)."""


class NeedInput(Exception):
    """The owner must supply one fact (for example headcount)."""

    def __init__(self, field_name: str, why: str = ""):
        super().__init__(field_name)
        self.field_name = field_name
        self.why = why


@dataclass
class Profile:
    """What the owner confirms on the Upload screen. Sector comes from the one list in config.SECTORS."""
    sector: str = "Trading · building materials"


class Ctx:
    def __init__(self, books: Books, profile: Profile | None = None, inputs: dict | None = None, quality: dict | None = None):
        self.b = books
        self.profile = profile or Profile()
        self.inputs = dict(inputs or {})
        self.quality = quality or {}
        self.months = books.months
        self.first = self.months[0]
        self.last = self.months[-1]
        self.inc = books.income_statement("M")
        self.bs = books.balance_sheet_monthly()
        self.bs.index = self.months
        p = books.p
        self._p = p
        self.cur = books.currency
        self.ar_inv = books.invoices("receivable")
        self.ap_inv = books.invoices("payable")
        self._age_cache: dict = {}
        self._mflows: dict = {}

    # ------------------------------------------------------------------ windows
    def avail(self, t, w) -> bool:
        return (t - (w - 1)) >= self.first

    def win(self, s: pd.Series, t, w) -> float:
        """Sum of a monthly flow series over the w months ending at t. NoData if the history is too short."""
        if not self.avail(t, w):
            raise NoData(f"needs {w} months of history; the books start {self.first}")
        return float(s.reindex(pd.period_range(t - (w - 1), t, freq="M")).fillna(0.0).sum())

    def days(self, t, w) -> int:
        end = t.end_time.normalize()
        start = (t - (w - 1)).start_time
        return int((end - start).days + 1)

    def date(self, t) -> pd.Timestamp:
        return t.end_time.normalize()

    def bal(self, key: str, t) -> float:
        return float(self.bs.loc[t, key])

    # ------------------------------------------------------------------ P&L flows by subtype (positive = cost)
    def sub_cost(self, *subs: str) -> pd.Series:
        k = ("c",) + subs
        if k not in self._mflows:
            p = self._p
            m = p[(p["account_type"] == "expense") & p["subtype"].isin(subs) & ~p["is_opening"]]
            self._mflows[k] = m.groupby("month")["net"].sum().reindex(self.months, fill_value=0.0)
        return self._mflows[k]

    def sub_flow(self, subtype: str, side: str, include_opening: bool = False) -> pd.Series:
        """Monthly sum of debit or credit on accounts of one subtype."""
        k = ("f", subtype, side, include_opening)
        if k not in self._mflows:
            p = self._p
            m = p[(p["subtype"] == subtype) & (include_opening | ~p["is_opening"])]
            self._mflows[k] = m.groupby("month")[side].sum().reindex(self.months, fill_value=0.0)
        return self._mflows[k]

    @property
    def revenue(self): return self.inc["Revenue"]
    @property
    def cogs(self): return self.inc["Cost of goods sold"]
    @property
    def opex(self): return self.inc["Operating expenses"]
    @property
    def gp(self): return self.inc["Gross profit"]
    @property
    def ebitda(self): return self.inc["EBITDA"]
    @property
    def ebit(self): return self.inc["EBIT"]
    @property
    def fin(self): return self.inc["Finance cost"]
    @property
    def net(self): return self.inc["Net profit"]

    # ------------------------------------------------------------------ invoices and ageing
    def invoiced(self) -> pd.Series:
        """Gross value invoiced to customers per month (opening balances excluded)."""
        k = ("inv",)
        if k not in self._mflows:
            a = self.ar_inv[~self.ar_inv["is_opening"].astype(bool) & (self.ar_inv["amount"] > 0)]
            self._mflows[k] = a.groupby(a["raised"].dt.to_period("M"))["amount"].sum().reindex(self.months, fill_value=0.0)
        return self._mflows[k]

    def billed(self) -> pd.Series:
        """Gross value billed by suppliers per month (opening balances excluded)."""
        k = ("bill",)
        if k not in self._mflows:
            a = self.ap_inv[~self.ap_inv["is_opening"].astype(bool) & (self.ap_inv["amount"] > 0)]
            self._mflows[k] = a.groupby(a["raised"].dt.to_period("M"))["amount"].sum().reindex(self.months, fill_value=0.0)
        return self._mflows[k]

    def ageing(self, kind: str, t) -> pd.DataFrame:
        k = (kind, t)
        if k not in self._age_cache:
            self._age_cache[k] = self.b.ageing(kind, self.date(t))
        return self._age_cache[k]

    def paid_in_window(self, kind: str, t, w) -> pd.DataFrame:
        if not self.avail(t, w):
            raise NoData(f"needs {w} months of history")
        inv = self.ar_inv if kind == "receivable" else self.ap_inv
        start = (t - (w - 1)).start_time
        end = self.date(t) + pd.Timedelta(days=1)
        x = inv[(inv["open"].abs() < 0.5) & inv["last_payment"].notna() & (inv["last_payment"] >= start) & (inv["last_payment"] < end)
                & ~inv["is_opening"].astype(bool) & (inv["amount"] > 0)]
        if x.empty:
            raise NoData("no invoices were settled in this period")
        return x

    def party_shares(self, kind: str, t, window_days: int = 365) -> pd.Series:
        inv = self.ar_inv if kind == "receivable" else self.ap_inv
        d = self.date(t)
        x = inv[~inv["is_opening"].astype(bool) & (inv["amount"] > 0) & (inv["raised"] <= d) & (inv["raised"] > d - pd.Timedelta(days=window_days))]
        s = x.groupby("party")["amount"].sum().sort_values(ascending=False)
        if s.sum() <= 0:
            raise NoData("no invoices in the trailing year")
        return s / s.sum()

    # ------------------------------------------------------------------ owner inputs
    def need(self, key: str, why: str = ""):
        v = self.inputs.get(key)
        if v in (None, "", 0):
            raise NeedInput(key, why)
        return float(v)

    def has(self, key: str) -> bool:
        return self.inputs.get(key) not in (None, "", 0)

    # ------------------------------------------------------------------ balances by subtype
    def sub_bal(self, *subs: str, t=None) -> float:
        """Closing balance (debit - credit) of all accounts of the given subtypes at month t."""
        if "_bm" not in self._mflows:
            bm = self.b.balances_at_month_end()
            self._mflows["_bm"] = bm.join(self.b._acct["subtype"]).groupby("subtype").sum()
        bm = self._mflows["_bm"]
        rows = [x for x in subs if x in bm.index]
        return float(bm.loc[rows, t].sum()) if rows else 0.0
