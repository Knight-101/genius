"""Lender policy: every number that sets a limit or a rate is a setting with a default, and any setting can be overridden for a situation.

Override rules say "use this value when ..." (tier, loan length, sector, MSME size, secured or not, market regime, from a date). The most
specific rule that matches wins; on a tie the newer rule wins. Every change is saved as a numbered policy version with who, when and why,
and every decision records the version it used. Business owners see only the outcome.

Pricing: rate = cost of funds (repo + funding spread) + operating cost + capital charge (capital ratio x (target ROE - cost of funds))
       + expected loss (probability of default x loss given default) + margin (+ term premium beyond one year).
Probability of default is read from the firm's calibrated score (smooth, no jump at tier edges) and scaled to the lender's own tier figures.
Limits: short-term = lowest of (share of annual sales, share of young receivables, multiple of monthly collections) x tier factor, less
short-term debt already drawn; shock-absorber line = lower of a multiple of monthly fixed costs and a share of sales; medium-term = the
largest loan that keeps debt cover and outside-liabilities-to-net-worth within limits; long-term = referral to the partner lender.
Grit only ever improves terms (until it is calibrated on real recoveries it is never the sole reason for a refusal). The Second Opinion
can only make terms more cautious."""
from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass, field

import numpy as np

DEFAULTS = {
    # pricing
    "repo_rate": 5.50, "funding_spread": 3.50, "operating_cost": 2.00, "capital_ratio": 15.0, "target_roe": 18.0,
    "lgd_unsecured": 60.0, "lgd_secured": 35.0, "lgd_invoice": 30.0, "margin": 2.00, "term_premium": 0.50,
    "pd_Safe": None, "pd_Moderate": None, "pd_Watch": None, "pd_Risky": None,          # None = use the calibrated figure
    "pricing_mode": "score",                                                          # "score" (smooth) or "tier"
    "grit_long_concession": 0.0, "grit_policy_enabled": False,
    "recovery_demo_enabled": True, "recovery_max_bps": 50.0,
    "cashflow_stress_pct": 20.0, "collateral_ltv_pct": 70.0, "term_max_months": 60,
    # fees
    "processing_fee": 1.5, "fee_mode": "annual_on_limit", "reservation_fee": 1.0, "zoho_share_fee": 1.0, "zoho_share_reservation": 0.25,
    # short-term limit
    "st_sales_pct": 15.0, "st_recv_pct": 75.0, "st_recv_age_days": 60, "st_coll_mult": 2.0,
    "tier_factor_Safe": 1.0, "tier_factor_Moderate": 0.8, "tier_factor_Watch": 0.4, "tier_factor_Risky": 0.0, "st_max_days": 180,
    # shock-absorber line
    "shock_fixed_mult": 1.5, "shock_sales_pct": 10.0, "shock_max_days": 90, "shock_trigger_weeks": 4.0, "shock_window_weeks": 6,
    "shock_grit_full": 55,
    # medium term
    "mt_dscr_min": 1.5, "mt_tol_tnw_max": 3.0, "mt_tenor_months": 36, "mt_grit_bonus": 55, "mt_without_grit_factor": 1.0,
    # invoice financing
    "inv_advance_pct": 80.0, "inv_max_age_days": 90,
    # validity and review
    "limit_validity_months": 12, "renewal_window_days": 30, "upgrade_buffer": 2,
    # Second Opinion
    "so_confidence": 0.95, "so_bh": True, "so_low_limit_cut": 50.0,
}
LABELS = {
    "repo_rate": "RBI repo rate (%)", "funding_spread": "Funding spread over repo (%)", "operating_cost": "Operating cost (% a year)",
    "capital_ratio": "Capital ratio (%)", "target_roe": "Target return on equity (%)", "lgd_unsecured": "Loss if default, unsecured (%)",
    "lgd_secured": "Loss if default, secured (%)", "lgd_invoice": "Loss if default, invoice financing (%)", "margin": "Margin (%)",
    "term_premium": "Term premium beyond 1 year (%)", "pd_Safe": "Default probability, Safe (%)", "pd_Moderate": "Default probability, Moderate (%)",
    "pd_Watch": "Default probability, Watch (%)", "pd_Risky": "Default probability, Risky (%)", "pricing_mode": "Pricing by score or by tier",
    "recovery_demo_enabled": "Enable assumed recovery pricing (synthetic demo)",
    "recovery_max_bps": "Maximum recovery concession (basis points; demo assumption)",
    "grit_long_concession": "Legacy standalone Grit concession (disabled by default)",
    "grit_policy_enabled": "Enable legacy standalone Grit terms (off for learned score)",
    "cashflow_stress_pct": "Cash-flow stress haircut (%)", "collateral_ltv_pct": "Maximum collateral advance (%)",
    "term_max_months": "Longest medium-term loan (months)", "processing_fee": "Processing fee (%)",
    "fee_mode": "Fee charged once a year on the limit or on each drawdown", "reservation_fee": "Shock-line reservation fee (% a year)",
    "zoho_share_fee": "Zoho share of disbursal (%)", "zoho_share_reservation": "Zoho share of reservation fee (% a year)",
    "st_sales_pct": "Short-term: share of annual sales (%)", "st_recv_pct": "Short-term: share of young receivables (%)",
    "st_recv_age_days": "Short-term: receivables counted up to (days old)", "st_coll_mult": "Short-term: multiple of monthly collections",
    "tier_factor_Safe": "Tier factor, Safe", "tier_factor_Moderate": "Tier factor, Moderate", "tier_factor_Watch": "Tier factor, Watch",
    "tier_factor_Risky": "Tier factor, Risky", "st_max_days": "Short-term: longest drawdown (days)",
    "shock_fixed_mult": "Shock line: months of fixed costs", "shock_sales_pct": "Shock line: share of annual sales (%)",
    "shock_max_days": "Shock line: longest use (days)", "shock_trigger_weeks": "Shock line opens below (weeks of cover)",
    "shock_window_weeks": "Shock line looks ahead (weeks)", "shock_grit_full": "Shock line at full size from Grit",
    "mt_dscr_min": "Medium term: debt cover after loan (x)", "mt_tol_tnw_max": "Medium term: liabilities to net worth at most (x)",
    "mt_tenor_months": "Medium term: tenor (months)", "mt_grit_bonus": "Medium term at full size from Grit",
    "mt_without_grit_factor": "Medium term size below that Grit", "inv_advance_pct": "Invoice financing: advance (%)",
    "inv_max_age_days": "Invoice financing: invoices up to (days old)", "limit_validity_months": "Limit valid for (months)",
    "renewal_window_days": "Renewal starts (days before expiry)", "upgrade_buffer": "Upgrade buffer (score points)",
    "so_confidence": "Second Opinion: confidence level", "so_bh": "Second Opinion: correct for many peer groups (Benjamini-Hochberg)",
    "so_low_limit_cut": "Second Opinion: limit cut on Low confidence (%)",
}
GROUPS = {
    "Pricing": ["repo_rate", "funding_spread", "operating_cost", "capital_ratio", "target_roe", "lgd_unsecured", "lgd_secured", "lgd_invoice",
                "margin", "term_premium", "pricing_mode", "recovery_demo_enabled", "recovery_max_bps"],
    "Default probability by tier": ["pd_Safe", "pd_Moderate", "pd_Watch", "pd_Risky"],
    "Fees": ["processing_fee", "fee_mode", "reservation_fee", "zoho_share_fee", "zoho_share_reservation"],
    "Short-term limit": ["st_sales_pct", "st_recv_pct", "st_recv_age_days", "st_coll_mult", "tier_factor_Safe", "tier_factor_Moderate",
                         "tier_factor_Watch", "tier_factor_Risky", "st_max_days"],
    "Shock-absorber line": ["shock_fixed_mult", "shock_sales_pct", "shock_max_days", "shock_trigger_weeks", "shock_window_weeks", "shock_grit_full"],
    "Medium term": ["cashflow_stress_pct", "collateral_ltv_pct", "term_max_months", "mt_dscr_min", "mt_tol_tnw_max", "mt_tenor_months", "mt_grit_bonus", "mt_without_grit_factor"],
    "Invoice financing": ["inv_advance_pct", "inv_max_age_days"],
    "Validity and review": ["limit_validity_months", "renewal_window_days", "upgrade_buffer"],
    "Second Opinion": ["so_confidence", "so_bh", "so_low_limit_cut"],
}
GUARD = {"recovery_max_bps": (0, 100), "so_low_limit_cut": (0, 100), "tier_factor_Safe": (0, 1.5), "tier_factor_Moderate": (0, 1.5), "tier_factor_Watch": (0, 1.5), "tier_factor_Risky": (0, 1.5),
         "lgd_unsecured": (30, 100), "lgd_secured": (10, 100), "lgd_invoice": (10, 100), "capital_ratio": (15, 40), "so_confidence": (0.80, 0.999),
         "processing_fee": (0, 4), "margin": (0, 15), "cashflow_stress_pct": (0, 80),
         "collateral_ltv_pct": (0, 90), "mt_dscr_min": (1, 5), "mt_tenor_months": (6, 60),
         "term_max_months": (6, 120), "repo_rate": (0, 30), "funding_spread": (0, 30),
         "operating_cost": (0, 30), "target_roe": (0, 50), "reservation_fee": (0, 10),
         "pd_Safe": (0, 100), "pd_Moderate": (0, 100), "pd_Watch": (0, 100), "pd_Risky": (0, 100)}
CONDITIONS = ("tier", "tenor", "sector", "size", "secured", "regime", "from_date")


@dataclass
class Rule:
    param: str
    value: object
    when: dict
    note: str = ""
    added: int = 0

    def matches(self, ctx: dict) -> bool:
        for k, v in self.when.items():
            if k == "from_date":
                if str(ctx.get("date", "9999")) < str(v):
                    return False
            elif ctx.get(k) != v:
                return False
        return True

    def label(self):
        cond = ", ".join(f"{k} = {v}" for k, v in self.when.items()) or "always"
        return f"{self.param} = {self.value} when {cond}"


@dataclass
class Policy:
    base: dict = field(default_factory=lambda: dict(DEFAULTS))
    rules: list = field(default_factory=list)
    versions: list = field(default_factory=list)
    calibrated_pd: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.versions:
            self._save("Genius defaults", "system", "Initial policy")

    # ---------------------------------------------------------------------------------------------- settings
    def _save(self, who, by, why):
        self.versions.append(dict(version=len(self.versions) + 1, at=dt.datetime.now().isoformat(timespec="seconds"), by=by, why=why,
                                  base=copy.deepcopy(self.base), rules=[copy.deepcopy(r.__dict__) for r in self.rules]))

    @property
    def version(self) -> int:
        return len(self.versions)

    def set(self, changes: dict, by: str = "lender", why: str = "") -> list:
        """Change base settings. Values outside sensible guard-rails are refused with a reason."""
        refused, applied = [], 0
        for k, v in changes.items():
            if k not in DEFAULTS:
                refused.append(f"{k}: unknown setting")
                continue
            bad_v = self.check(k, v)
            if bad_v:
                refused.append(bad_v)
                continue
            self.base[k] = v
            applied += 1
        if applied:
            self._save("settings", by, why or "settings changed")
        return refused

    @staticmethod
    def check(param, value):
        """None if the value is allowed, else the reason it is refused."""
        if param in GUARD and value is not None and not isinstance(value, str):
            lo, hi = GUARD[param]
            if not (lo <= float(value) <= hi):
                return f"{LABELS.get(param, param)} must be between {lo} and {hi}"
        return None

    def add_rule(self, param, value, when: dict, note="", by="lender", why=""):
        if param not in DEFAULTS:
            raise ValueError(f"unknown setting {param}")
        bad_v = self.check(param, value)
        if bad_v:
            raise ValueError(bad_v)
        if "from_date" in when:
            import re as _re
            if not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(when["from_date"])):
                raise ValueError("From date must be written YYYY-MM-DD")
        bad = set(when) - set(CONDITIONS)
        if bad:
            raise ValueError(f"unknown condition {bad}")
        self.rules.append(Rule(param, value, dict(when), note, added=1 + max([r.added for r in self.rules], default=-1)))
        self._save("rule", by, why or f"override {param}")

    def remove_rule(self, i: int, by="lender", why=""):
        self.rules.pop(i)
        self._save("rule removed", by, why or "override removed")

    def reset(self, by="lender"):
        self.base = dict(DEFAULTS); self.rules = []
        self._save("reset", by, "back to Genius defaults")

    def get(self, param: str, ctx: dict | None = None):
        """(value, source) for one setting in a situation."""
        ctx = ctx or {}
        hits = [r for r in self.rules if r.param == param and r.matches(ctx)]
        if hits:
            r = max(hits, key=lambda r: (len(r.when), r.added))
            return r.value, r.label()
        v = self.base[param]
        if v is None and param.startswith("pd_"):
            return self.calibrated_pd.get(param[3:], np.nan), "calibrated on the synthetic training years"
        return v, "base setting"

    def v(self, param, ctx=None):
        return self.get(param, ctx)[0]

    # ---------------------------------------------------------------------------------------------- pricing
    def rate(self, tier: str, ctx: dict | None = None, pd_score: float | None = None, tenor_months: int = 3, secured: bool = False,
             invoice: bool = False, grit: float | None = None, recovery: dict | None = None) -> dict:
        ctx = dict(ctx or {}, tier=tier, tenor="short" if tenor_months <= 12 else ("medium" if tenor_months <= 36 else "long"),
                   secured="yes" if (secured or invoice) else "no")
        g = lambda k: float(self.v(k, ctx))
        cof = g("repo_rate") + g("funding_spread")
        cap = g("capital_ratio") / 100 * max(0.0, g("target_roe") - cof)
        tier_pd = float(self.v(f"pd_{tier}", ctx))
        if self.v("pricing_mode", ctx) == "score" and pd_score is not None:
            pd_ = float(np.clip(pd_score, 0, 100))
            # Only an explicit lender override rescales a firm's calibrated probability.
            value, source = self.get(f"pd_{tier}", ctx)
            ref = self.calibrated_pd.get(tier, 0)
            overridden = self.base[f"pd_{tier}"] is not None or source.startswith(f"pd_{tier} =")
            if overridden and ref > 0:
                pd_ = float(np.clip(pd_ * float(value) / ref, 0, 100))
        else:
            pd_ = tier_pd
        lgd = g("lgd_invoice") if invoice else (g("lgd_secured") if secured else g("lgd_unsecured"))
        el = pd_ * lgd / 100
        prem = g("term_premium") if tenor_months > 12 else 0.0
        conc = g("grit_long_concession") if (self.v("grit_policy_enabled", ctx) and tenor_months > 12 and grit is not None and grit >= 75) else 0.0
        from .resilience import concession
        recovery_discount = concession(recovery, g("recovery_max_bps"), g("margin")) if self.v("recovery_demo_enabled", ctx) and tier != "Risky" else 0.0
        # Use one recovery policy only; legacy terms cannot stack on top.
        if recovery_discount: conc = 0.0
        before = cof + g("operating_cost") + cap + el + g("margin") + prem
        total = before - conc - recovery_discount
        return dict(rate=total, offered=tier != "Risky", cost_of_funds=cof, operating_cost=g("operating_cost"), capital_charge=cap, expected_loss=el, pd=pd_, lgd=lgd,
                    margin=g("margin"), term_premium=prem, grit_concession=conc, recovery_concession=recovery_discount,
                    rate_before_recovery=before, recovery_evidence=recovery, recovery_policy_demo=True)

    def apr(self, amount: float, rate: float, days: int, ctx=None, facility_limit=None, fee_paid=False) -> dict:
        """Bullet-draw estimate. Annual facility fees are charged in full, never prorated as cash paid."""
        if not (np.isfinite(amount) and amount > 0 and np.isfinite(rate) and rate >= 0 and days > 0):
            raise ValueError("A positive amount and duration and a nonnegative finite rate are required")
        limit = amount if facility_limit is None else float(facility_limit)
        if not np.isfinite(limit) or limit < amount:
            raise ValueError("Facility limit must cover the drawdown")
        interest = amount * rate / 100 * days / 365
        fee_pct = float(self.v("processing_fee", ctx))
        annual = self.v("fee_mode", ctx) == "annual_on_limit"
        fee = 0.0 if fee_paid and annual else (limit if annual else amount) * fee_pct / 100
        note = ("Annual facility fee charged in full on the requested limit, payable upfront; not prorated by days."
                if annual else "One-time processing fee on this drawdown, payable upfront.")
        net = amount - fee
        effective = ((amount + interest) / net) ** (365 / days) - 1 if net > 0 else np.nan
        return dict(interest=interest, fee=fee, total=interest + fee, apr=effective * 100,
                    net_disbursed=net, note=note)

    # ---------------------------------------------------------------------------------------------- limits
    def limits(self, x: dict, tier: str, grit: float | None, ko: dict, confidence: str | None = None, ctx: dict | None = None, pd_score: float | None = None) -> dict:
        """x: the firm's facts at the as-of month (annual_sales, recv_young, coll_month, std, fixed_month, ebitda12, debt_service12,
        total_liab, equity, invoices_eligible). Returns every product with its limit, rate inputs and the reasons."""
        ctx = dict(ctx or {}, tier=tier)
        g = lambda k: float(self.v(k, ctx))
        out = {}
        reasons = []
        cut = 1.0
        if confidence == "Low":
            cut = float(np.clip(1 - g("so_low_limit_cut") / 100, 0.0, 1.0))
            reasons.append(f"Second Opinion is more worried than the scorecard: limits held at {100 * cut:.0f}% until a person reviews")
        # short term
        a = g("st_sales_pct") / 100 * x["annual_sales"]
        b = g("st_recv_pct") / 100 * x["recv_young"]
        c = g("st_coll_mult") * x["coll_month"]
        base = min(a, b, c)
        tf = g(f"tier_factor_{tier}")
        st_lim = max(0.0, base * tf - x["std"]) * cut
        st_ok = tier != "Risky" and not ko.get("K1") and st_lim > 0
        why = [f"lowest of {a / 1e5:,.1f} lakh (sales), {b / 1e5:,.1f} lakh (young receivables), {c / 1e5:,.1f} lakh (collections)",
               f"x tier factor {tf:g}", f"less {x['std'] / 1e5:,.1f} lakh already drawn on cash credit"]
        bridge = bool(ko.get("K1")) and grit is not None and grit >= g("shock_grit_full")
        if ko.get("K1"):
            why.append("cash runway is critical: " + ("an invoice-linked bridge is offered because Grit shows the business recovers" if bridge else
                                                       "no short-term limit; an invoice-linked bridge needs Grit Steady or better"))
        if not st_ok and not ko.get("K1"):
            why.append("Not offered in the Risky tier" if tier == "Risky" else "Nothing left after the cash credit already drawn")
        out["short"] = dict(available=st_ok, limit=st_lim if st_ok else 0.0, why=why, bridge=bridge, max_days=int(g("st_max_days")))
        # shock-absorber line
        sh = min(g("shock_fixed_mult") * x["fixed_month"], g("shock_sales_pct") / 100 * x["annual_sales"])
        full = not self.v("grit_policy_enabled", ctx) or (grit is not None and grit >= g("shock_grit_full"))
        sh_lim = (sh if full else 0.5 * sh) * cut
        sh_ok = tier in ("Safe", "Moderate") and not ko.get("K1") and not ko.get("K2")
        sh_why = [f"lower of {g('shock_fixed_mult'):g} months of fixed costs and {g('shock_sales_pct'):g}% of sales",
                  "line sized from business cash needs and lender policy" if full else "legacy Grit policy reduces this line"]
        if not sh_ok:
            sh_why.append("Needs the Moderate tier or better, with no cash-runway or debt-cover red flag")
        out["shock"] = dict(available=sh_ok, limit=sh_lim if sh_ok else 0.0, full_size=full, max_days=int(g("shock_max_days")),
                            fee_year=(sh_lim if sh_ok else 0) * g("reservation_fee") / 100, why=sh_why)
        # medium term
        n = int(g("mt_tenor_months"))
        r = self.rate(tier, ctx, pd_score=pd_score, tenor_months=n, secured=True, grit=grit)["rate"] / 1200
        ann = r / (1 - (1 + r) ** -n) if r > 0 else 1 / n
        cfads = x.get("cfads12", x["ebitda12"]) * (1 - g("cashflow_stress_pct") / 100)
        emi_max = (cfads / g("mt_dscr_min") - x["debt_service12"]) / 12
        by_cover = max(0.0, emi_max / ann)
        by_tol = max(0.0, g("mt_tol_tnw_max") * x["equity"] - x["total_liab"]) if x["equity"] > 0 else 0.0
        mt = min(by_cover, by_tol)
        gfull = not self.v("grit_policy_enabled", ctx) or (grit is not None and grit >= g("mt_grit_bonus"))
        mt = mt * (1.0 if gfull else g("mt_without_grit_factor")) * cut
        mt_ok = tier != "Risky" and not ko.get("K2") and mt > 0
        cover_after = cfads / (x["debt_service12"] + 12 * mt * ann) if mt > 0 else np.nan
        mt_extra = [] if mt_ok else ["Not offered: " + ("Risky tier" if tier == "Risky" else ("debt cover below 1x" if ko.get("K2") else "no room within debt cover and net worth"))]
        out["medium"] = dict(available=mt_ok, limit=mt if mt_ok else 0.0, tenor_months=n, cover_after=cover_after, secured=True,
                             why=[f"largest loan keeping debt cover at {g('mt_dscr_min'):g}x after the new instalments: {by_cover / 1e5:,.1f} lakh",
                                  f"and liabilities within {g('mt_tol_tnw_max'):g}x net worth: {by_tol / 1e5:,.1f} lakh",
                                  "no separate Grit size bonus; financial repayment checks set the amount" if gfull else "legacy standalone Grit adjustment",
                                  "secured: charge on receivables and stock, plus a personal or CGTMSE guarantee"] + mt_extra)
        # invoice financing
        inv = g("inv_advance_pct") / 100 * x.get("invoices_eligible", 0.0) * cut
        inv_ok = tier != "Risky" and inv > 0
        out["invoice"] = dict(available=inv_ok, limit=inv if inv_ok else 0.0, secured=True,
                              why=[f"{g('inv_advance_pct'):g}% of customer invoices up to {g('inv_max_age_days'):g} days old and not overdue"] +
                                  ([] if inv_ok else ["Not offered: " + ("Risky tier" if tier == "Risky" else "no eligible invoices")]))
        out["long"] = dict(available=False, referral=tier in ("Safe", "Moderate") and not ko.get("K2"), why=["a partner lender appraises the project"])
        if tier == "Watch":
            reasons.append("Watch tier: any loan needs security or a guarantee")
        if cut < 1:
            for k_ in ("short", "shock", "medium", "invoice"):
                if out[k_].get("available"):
                    out[k_]["why"].append(f"held at {100 * cut:.0f}% until a credit reviewer at the lender has looked at the file")
        out["reasons"] = reasons
        out["validity_months"] = int(g("limit_validity_months"))
        out["policy_version"] = self.version
        return out

    def early_payment(self, bills: list, rate: float, ctx=None) -> list:
        """bills: [{supplier, amount, discount_pct, days_early, msme}] -> net gain of paying early with the short-term limit."""
        out = []
        per_draw = self.v("fee_mode", ctx) == "per_drawdown"
        for b in bills:
            pay = b["amount"] * (1 - b["discount_pct"] / 100)
            disc = b["amount"] * b["discount_pct"] / 100
            interest = pay * rate / 100 * b["days_early"] / 365
            fee = pay * float(self.v("processing_fee", ctx)) / 100 if per_draw else 0.0
            net = disc - interest - fee
            out.append(dict(b, discount=disc, interest=interest, fee=fee, net=net, take=net > 0,
                            annualised=disc / pay * 365 / max(1, b["days_early"]) * 100))
        return out
