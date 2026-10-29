"""One request -> one consistent, conditional borrowing illustration.

Rates, affordability, limits, fees and repayment cash flows share the same policy context.
No automatic disbursement, credit approval, live bank feed or regulatory KFS is implied.
"""
import math

import numpy as np
from scipy.optimize import brentq

PRODUCTS = {"Short term": "short", "Shock-absorber line": "shock",
            "Invoice financing": "invoice", "Medium term": "medium"}


def context(a, product="short", months=3, secured=False):
    ctx = dict(a.info.get("policy_context", {}))
    ctx.update(tier=a.tier, sector=a.sector, size=a.band,
               tenor="short" if months <= 12 else ("medium" if months <= 36 else "long"),
               secured="yes" if secured or product == "invoice" else "no")
    return ctx


def payment(principal, annual_rate, months):
    r = annual_rate / 1200
    return principal / months if abs(r) < 1e-12 else principal * r / (1 - (1 + r) ** -months)


def principal_for_payment(emi, annual_rate, months):
    return max(0.0, emi / payment(1.0, annual_rate, months))


def effective_cost(net, flows):
    """Effective annual cash-flow IRR. flows are (days from disbursement, positive payment)."""
    if net <= 0 or not flows:
        return None
    def npv(log_growth):
        return sum(v * math.exp(-log_growth * d / 365) for d, v in flows) - net
    try:
        return 100 * math.expm1(brentq(npv, -0.99, 30))
    except (ValueError, OverflowError):
        return None


def quote(a, pol, requested, days, product, months=36, secured=False, collateral_value=None):
    base = dict(status="invalid", product=product, requested=requested, supported=0.0, reasons=[], schedule=[], policy_version=pol.version)
    try:
        amount = float(requested)
        duration = float(months if product == "Medium term" else days)
        if not math.isfinite(amount) or not 0 < amount <= 1e12:
            raise ValueError("Enter a positive amount up to ₹1 lakh crore.")
        if not math.isfinite(duration) or duration != int(duration):
            raise ValueError("Enter a whole-number duration.")
    except (TypeError, ValueError) as exc:
        base["reasons"] = [str(exc) if isinstance(exc, ValueError) else "Enter an amount and duration."]
        return base
    if product not in PRODUCTS:
        base["reasons"] = ["Choose a supported borrowing product."]; return base
    if a is None or a.score is None or not a.tier:
        base["reasons"] = ["Assess at least 12 months of reliable books before requesting a quote."]; return base
    key = PRODUCTS[product]
    n = int(duration) if key == "medium" else 3
    secured = bool(secured)
    ctx = context(a, key, n, secured)
    g = lambda k: float(pol.v(k, ctx))
    if key == "medium":
        if not 6 <= duration <= g("term_max_months"):
            base["reasons"] = [f"Choose 6 to {int(g('term_max_months'))} months."]; return base
        if not secured:
            base.update(status="review", reasons=["This medium-term product requires collateral. Select security and enter its estimated value for a conditional quote."])
            return base
    else:
        max_days = g("shock_max_days") if key == "shock" else (g("inv_max_age_days") if key == "invoice" else g("st_max_days"))
        if not 7 <= duration <= max_days:
            base["reasons"] = [f"Choose 7 to {int(max_days)} days for this product."]; return base
    collateral_cap = float("inf")
    if secured:
        try:
            value = float(collateral_value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError
        except (TypeError, ValueError):
            base["reasons"] = ["Enter a positive estimated collateral value. The lender must verify ownership and valuation."]; return base
        collateral_cap = value * g("collateral_ltv_pct") / 100
    x = dict(a.info["lending_facts"])
    ko = {k: bool(v[0, -1]) for k, v in a.gs["ko"].items()}
    limits = pol.limits(x, a.tier, None, ko, confidence=a.so.get("confidence"), ctx=ctx, pd_score=100 * a.p_default)
    if not limits[key]["available"]:
        base.update(status="review", reasons=list(limits[key]["why"]))
        return base
    rate = pol.rate(a.tier, ctx, pd_score=100 * a.p_default, tenor_months=n,
                    secured=secured, invoice=key == "invoice", grit=None, recovery=a.info.get("recovery_policy"))
    if not rate["offered"] or not math.isfinite(rate["rate"]) or rate["rate"] < 0:
        base.update(status="review", reasons=["A credit reviewer must assess this request before a rate can be shown."]); return base
    annual = rate["rate"]
    haircut = 1 - g("cashflow_stress_pct") / 100
    cfads = max(0.0, x["cfads12"] * haircut)
    service = max(0.0, x["debt_service12"])
    cut = 1 - g("so_low_limit_cut") / 100 if a.so.get("confidence") == "Low" else 1.0
    reasons = []
    if key == "medium":
        emi_budget = max(0.0, cfads / g("mt_dscr_min") - service) / 12
        renewal_rate = g("processing_fee") / 100 if pol.v("fee_mode", ctx) == "annual_on_limit" and n > 12 else 0.0
        affordability = max(0.0, 12 * emi_budget / (12 * payment(1.0, annual, n) + renewal_rate)) * cut
        leverage = max(0.0, g("mt_tol_tnw_max") * x["equity"] - x["total_liab"]) * cut
        cap = min(affordability, leverage, collateral_cap)
        reasons += [f"Repayment capacity uses cash flow after working-capital needs, capex and drawings, with a {g('cashflow_stress_pct'):g}% haircut.",
                    f"Debt-service cover must remain at least {g('mt_dscr_min'):g}×, including existing payments."]
    else:
        d = int(duration)
        # Cash already held above one month of fixed costs plus stressed collection surplus.
        reserve = max(0.0, x.get("cash", 0) - x["fixed_month"])
        cash_in = x["coll_month"] * haircut * d / 30.44
        committed = x.get("cash_cost_month", x["fixed_month"]) * d / 30.44 + service * d / 365
        resources = max(0.0, reserve + cash_in - committed)
        affordability = resources / (1 + annual / 100 * d / 365) * cut
        cap = min(limits[key]["limit"], affordability, collateral_cap)
        reasons += [f"Checks collections after a {g('cashflow_stress_pct'):g}% haircut, cash above a fixed-cost reserve, operating commitments and existing debt payments.",
                    "A lender must verify invoice eligibility, existing pledges and the purpose of funds before approval."]
    cap = max(0.0, float(cap))
    # Round down so rounding never pushes a quote over its calculated capacity.
    supported = amount if amount <= cap else math.floor(cap / 100) * 100
    if supported <= 0:
        base.update(status="review", max_supported=cap, reasons=["No positive amount fits the stressed repayment and policy checks."] + reasons)
        return base
    if supported < amount:
        reasons.insert(0, "The requested amount does not fit. The illustration below is a smaller alternative, not approval of the original request.")
    fee_pct = g("processing_fee")
    annual_fee = pol.v("fee_mode", ctx) == "annual_on_limit"
    upfront_fee = supported * fee_pct / 100
    reservation = supported * g("reservation_fee") / 100 if key == "shock" else 0.0
    schedule = []
    if key == "medium":
        emi = payment(supported, annual, n)
        balance = supported; total_interest = 0.0; fees = upfront_fee
        flows = []
        for m in range(1, n + 1):
            interest = balance * annual / 1200
            principal = min(balance, emi - interest)
            balance = max(0.0, balance - principal)
            renewal_fee = supported * fee_pct / 100 if annual_fee and m % 12 == 0 and m < n else 0.0
            fees += renewal_fee; total_interest += interest
            cash_payment = principal + interest + renewal_fee
            flows.append((m * 365 / 12, cash_payment))
            schedule.append(dict(Period=f"Month {m}", Principal=round(principal, 2), Interest=round(interest, 2),
                                 Fees=round(renewal_fee, 2), Payment=round(cash_payment, 2), Balance=round(balance, 2)))
        renewal = supported * fee_pct / 100 if annual_fee and n > 12 else 0.0
        cover = cfads / (service + 12 * emi + renewal) if service + 12 * emi + renewal > 0 else None
        net = supported - upfront_fee
        eac = effective_cost(net, flows)
    else:
        costs = pol.apr(supported, annual, int(duration), ctx, facility_limit=supported)
        total_interest = costs["interest"]; fees = costs["fee"] + reservation
        net = supported - fees; emi = None; cover = None
        due = supported + total_interest
        eac = effective_cost(net, [(int(duration), due)])
        schedule.append(dict(Period=f"Day {int(duration)}", Principal=round(supported, 2), Interest=round(total_interest, 2),
                             Fees=0.0, Payment=round(due, 2), Balance=0.0))
    if net <= 0:
        base.update(status="review", reasons=["Fees exceed the available disbursement; a lender must review this policy."]); return base
    schedule.insert(0, dict(Period="Disbursement", Principal=0.0, Interest=0.0, Fees=round(upfront_fee + reservation, 2),
                            Payment=0.0, Balance=round(supported, 2)))
    return dict(base, status="fits" if supported >= amount else "reduced", requested=amount, supported=supported,
                max_supported=cap, days=None if key == "medium" else int(duration), months=n if key == "medium" else None,
                rate=annual, rate_build=rate, interest=total_interest, fees=fees, upfront_fees=upfront_fee + reservation,
                reservation_fee=reservation, net_disbursed=net, total_cost=total_interest + fees,
                total_repayment=supported + total_interest + fees, emi=emi, cover_after=cover,
                annual_effective_cost=eac, reasons=reasons, schedule=schedule, context=ctx,
                fee_note=("Full annual fee on the requested facility, upfront and at each anniversary while open."
                          if annual_fee else "One-time processing fee on the requested borrowing."),
                conditions=["Synthetic-data prototype; conditional illustration, not a loan offer or regulatory Key Facts Statement.",
                            "Quoted interest excludes taxes and third-party costs, which a lender must disclose.",
                            "Existing loan rates change only under the agreed contract and reset terms."])
