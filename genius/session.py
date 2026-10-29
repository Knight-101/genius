"""One user's session: read a file, confirm details, assess, and keep what the screens need."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import alerts, assess, config as C, facts
from .ask import Genie
from .books import Books
from .ctx import Ctx, Profile
from .ingest import ingest

SECTOR_HINTS = [
    ("Trading · building materials", r"cement|steel|hardware|tile|sanitary|plumb|building material|paint|build"),
    ("Trading · FMCG", r"fmcg|grocery|kirana|provision|consumer goods|soap|beverage"),
    ("Trading · pharma", r"pharma|medical|medicine|drug|chemist|healthcare"),
    ("Trading · electronics", r"electronic|mobile|appliance|electrical goods|digital"),
    ("Manufacturing · food processing", r"food|agro|spice|snack|flour|dairy|oil mill|packaged"),
    ("Manufacturing · textiles", r"textile|fabric|garment|yarn|weav|apparel"),
    ("Manufacturing · plastics", r"plastic|polymer|moulding|packaging film"),
    ("Manufacturing · light engineering", r"engineering|fabricat|machin|precision|casting|forging"),
    ("Services · logistics", r"freight|logistic|transport|truck|cargo|courier|fleet"),
    ("Services · IT and ITeS", r"software|it services|cloud|hosting|consult|digital solutions|infotech"),
]


def suggest_sector(postings: pd.DataFrame) -> str:
    text = " ".join(postings["account"].dropna().unique().tolist() + postings["party"].dropna().unique().tolist()[:200]).lower()
    best, hits = "Trading · building materials", 0
    for s, pat in SECTOR_HINTS:
        n = len(re.findall(pat, text))
        if n > hits:
            best, hits = s, n
    return best


@dataclass
class Session:
    path: str | None = None
    res: object = None
    books: Books | None = None
    ctx: Ctx | None = None
    sector: str | None = None
    company: str = "Your business"
    a: object = None
    genie: Genie | None = None
    ew: dict | None = None
    inputs: dict = field(default_factory=dict)

    def read(self, path: str, company: str | None = None) -> dict:
        self.path = path
        self.res = ingest(path)
        self.books = Books(self.res.postings, self.res.currency)
        self.sector = suggest_sector(self.res.postings)
        self.company = company or _company_from(path)
        rep = self.res.report
        return dict(report=rep, sector=self.sector, suppliers=self.suppliers(), checks=self.books.checks())

    def suppliers(self) -> list:
        inv = self.books.invoices("payable")
        inv = inv[inv["amount"] > 0]
        return list(inv.groupby("party")["amount"].sum().sort_values(ascending=False).index)

    def run(self, db, policy, sector: str, cc_limit: float | None = None, mse: list | None = None, udyam: str | None = None) -> object:
        self.sector = sector
        self.inputs = dict(cc_limit=cc_limit, mse=list(mse or []), udyam=udyam)
        self.ctx = Ctx(self.books, Profile(sector), inputs={"cc_limit": cc_limit}, quality=self.res.report["checks"])
        f, info = facts.from_books(self.ctx, cc_limit=cc_limit, mse={m: 45 for m in (mse or [])})
        checks = self.books.checks()
        firm = facts.firm_row(sector, C.family(sector), self.res.report, checks)
        bal_ok = float(self.res.report["checks"].get("balanced_entries_pct", 100)) >= 95
        bs_ok = next((c["passed"] for c in checks if c["check"].startswith("Assets")), True)
        k5 = (not bal_ok) or (not bs_ok) or bool(info.get("top5_flagged"))
        info["report"] = self.res.report
        info["checks"] = checks
        try:
            self.ew = alerts.project(self.books, cc_limit=cc_limit)
            info["early_warning"] = self.ew
        except Exception:
            self.ew = None
        self.a = assess.run(db, policy, f, firm, info["months"], self.company, sector, dict(cc_limit=cc_limit, udyam=udyam), info, k5=k5)
        self.genie = Genie(self.a, db, policy, self.books)
        return self.a


def _company_from(path: str) -> str:
    import os
    base = os.path.basename(path)
    for k, v in {"Sharma": "Sharma Traders Pvt Ltd", "Kaveri": "Kaveri Foods Pvt Ltd", "Swift": "Swift Freight Carriers"}.items():
        if base.startswith(k):
            return v
    return re.sub(r"[_\-]+", " ", os.path.splitext(base)[0]).strip().title()
