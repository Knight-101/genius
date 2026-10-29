"""One place for every definition the model shares: sectors, the government MSME bands, the 54-measure catalogue with its ranges,
area weights, tiers and Grit bands. Synthetic data, uploaded books, peers, scoring, the Second Opinion and the app all read from here,
so a definition can never differ between two parts of the system."""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------------- calendar of the synthetic world
# 120 months, October 2016 to September 2026. Year 1 = Oct 2016 - Sep 2017, ..., year 10 = Oct 2025 - Sep 2026.
T_MONTHS = 120
START = "2016-10"
TRAIN_YEARS = (1, 5)      # model fitting and calibration
VALID_YEARS = (6, 8)      # cut-off checks and choosing the Second Opinion model
TEST_YEARS = (9, 10)      # opened once; includes the forced downturn
DOWNTURN = (102, 113)     # month index (Apr 2025 - Mar 2026): every sector is pushed into stress


def year_of(month_index):
    return np.asarray(month_index) // 12 + 1


# ---------------------------------------------------------------------------------------------------- sectors (one list, used everywhere)
SECTORS = {
    # name: (family, stock held?, typical customer terms in days, typical supplier terms in days)
    "Trading · FMCG": ("trading", True, 21, 30),
    "Trading · building materials": ("trading", True, 30, 45),
    "Trading · pharma": ("trading", True, 45, 60),
    "Trading · electronics": ("trading", True, 30, 45),
    "Manufacturing · light engineering": ("manufacturing", True, 60, 60),
    "Manufacturing · textiles": ("manufacturing", True, 60, 45),
    "Manufacturing · food processing": ("manufacturing", True, 30, 30),
    "Manufacturing · plastics": ("manufacturing", True, 45, 45),
    "Services · IT and ITeS": ("services", False, 45, 30),
    "Services · logistics": ("services", False, 45, 30),
}
SECTOR_LIST = list(SECTORS)


def family(sector: str) -> str:
    return SECTORS[sector][0]


# ---------------------------------------------------------------------------------------------------- government MSME definition
# Notification S.O. 1364(E), in force from 1 April 2025: investment in plant, machinery or equipment AND turnover (exports excluded).
# Exceeding either limit puts the enterprise in the next band up. Above medium is outside Genius's scope.
MSME_LIMITS = [("Micro", 2.5e7, 10e7), ("Small", 25e7, 100e7), ("Medium", 125e7, 500e7)]
SIZE_LIST = ["Micro", "Small", "Medium"]


def msme_band(investment: float, turnover: float) -> str:
    """Band from the government rule: the lowest band whose BOTH limits hold."""
    for name, inv_lim, to_lim in MSME_LIMITS:
        if investment <= inv_lim and turnover <= to_lim:
            return name
    return "Above medium"


def msme_band_vec(investment: np.ndarray, turnover: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(turnover), "Above medium", dtype=object)
    for name, inv_lim, to_lim in reversed(MSME_LIMITS):
        out = np.where((investment <= inv_lim) & (turnover <= to_lim), name, out)
    return out


def msme_reason(investment: float, turnover: float) -> str:
    band = msme_band(investment, turnover)
    cr = lambda x: f"₹{x / 1e7:,.2f} crore"
    if band == "Above medium":
        return f"Turnover {cr(turnover)} and plant and equipment {cr(investment)} are above the medium limits, so the business is outside MSME scope."
    i = SIZE_LIST.index(band)
    parts = []
    if i > 0:
        prev, pinv, pto = MSME_LIMITS[i - 1]
        over = []
        if turnover > pto:
            over.append(f"turnover {cr(turnover)} is above the {prev.lower()} limit of {cr(pto)}")
        if investment > pinv:
            over.append(f"plant and equipment {cr(investment)} is above the {prev.lower()} limit of {cr(pinv)}")
        parts.append(" and ".join(over).capitalize() + f", so the business is {band.lower()}")
    else:
        parts.append(f"Turnover {cr(turnover)} and plant and equipment {cr(investment)} are both within the micro limits")
    return "; ".join(parts) + " (government MSME definition, exceeding either limit moves a firm up)."


# ---------------------------------------------------------------------------------------------------- measure catalogue
# id: (name, area, unit, direction H/L, boundaries worst->best at scores 20/40/60/80, plain description)
# Family overrides replace the boundaries; None means "does not apply to this family".
AREAS = {"Liquidity": 22, "Collections": 18, "Debt service": 18, "Payment discipline": 12, "Profitability": 12,
         "Stability": 10, "Owner behaviour": 4, "Data integrity": 4}
AREA_LIST = list(AREAS)

M = {}


def _m(mid, name, area, unit, d, b, desc, fam=None, bank_norm=False):
    M[mid] = dict(id=mid, name=name, area=area, unit=unit, dir=d, bounds=tuple(b), desc=desc, fam=fam or {}, bank_norm=bank_norm)


# Liquidity
_m("LQ01", "Current ratio", "Liquidity", "x", "H", (1.0, 1.17, 1.33, 2.0), "Short-term assets for every 1 of short-term liabilities", bank_norm=True)
_m("LQ03", "Cash ratio", "Liquidity", "x", "H", (0.1, 0.2, 0.4, 0.8), "Cash and bank for every 1 of short-term liabilities")
_m("CR02", "Cash cover", "Liquidity", "weeks", "H", (2, 4, 8, 13), "Weeks of fixed costs that cash plus unused cash-credit can pay",
   fam={"trading": (4, 8, 16, 26), "manufacturing": (3, 6, 12, 20)})
_m("CR03", "Cash runway", "Liquidity", "days", "H", (15, 30, 60, 90), "Days cash lasts if every cost keeps running",
   fam={"trading": (2, 5, 10, 20), "manufacturing": (3, 7, 15, 30)})
_m("CG02", "Operating cash-flow margin", "Liquidity", "%", "H", (-5, 0, 4, 10), "Cash the business produced per 100 of sales")
_m("CG03", "Cash conversion of profit", "Liquidity", "%", "H", (0, 40, 70, 100), "How much of profit turned into cash")
_m("FL11", "Cash-credit limit used", "Liquidity", "%", "L", (95, 85, 70, 50), "Share of the cash-credit limit in use (needs the limit)")
_m("EW1", "Cash-trend low point", "Liquidity", "weeks", "H", (2, 4, 6, 10), "Cash cover in three months if the last quarter's cash burn simply continued (a trend, not the invoice-level 13-week forecast)",
   fam={"trading": (4, 8, 12, 20), "manufacturing": (3, 6, 9, 15)})
_m("TB5", "Intra-month cash low", "Liquidity", "weeks", "H", (0.5, 1, 2, 4), "Lowest balance inside the month, in weeks of fixed costs",
   fam={"trading": (1, 2, 4, 8), "manufacturing": (0.75, 1.5, 3, 6)})
# Collections and working capital
_m("RC01", "Days sales outstanding", "Collections", "days", "L", (85, 65, 48, 33), "Days of sales customers still owe",
   fam={"manufacturing": (100, 80, 62, 45), "services": (100, 80, 60, 45)})
_m("RC03", "Days beyond terms", "Collections", "days", "L", (30, 15, 7, 0), "How many days late customers pay, on average")
_m("RC05", "Overdue share", "Collections", "%", "L", (50, 35, 20, 10), "Share of what customers owe that is past due")
_m("RC08", "Overdue 90+ days", "Collections", "%", "L", (20, 12, 6, 2), "Share of what customers owe that is over 90 days past due")
_m("RC09", "Top customer share", "Collections", "%", "L", (50, 35, 25, 15), "Largest customer's share of sales")
_m("RC11", "Customer concentration", "Collections", "index", "L", (0.40, 0.25, 0.15, 0.08), "Herfindahl index of customer sales shares")
_m("RC14", "Receivables vs sales growth", "Collections", "pp", "L", (15, 8, 3, 0), "Receivables growth minus sales growth over 12 months")
_m("RC16", "Collection effectiveness", "Collections", "%", "H", (60, 75, 85, 92), "Share of collectable money actually collected")
_m("WC01", "Cash conversion cycle", "Collections", "days", "L", (100, 78, 55, 32), "Stock days plus customer days minus supplier days",
   fam={"manufacturing": (140, 110, 85, 60), "services": (100, 80, 60, 45)})
_m("TB1", "Customer lateness trend", "Collections", "days", "L", (10, 5, 2, 0), "Change in customers' days late, last 3 months vs the 3 before")
# Payment discipline
_m("PY02", "Days beyond supplier terms", "Payment discipline", "days", "L", (30, 20, 10, 3), "How late bills are paid against their due dates")
_m("PY03", "Bills paid on time", "Payment discipline", "%", "H", (40, 60, 75, 90), "Value of bills paid within 7 days of the due date")
_m("PY05", "Overdue payables", "Payment discipline", "%", "L", (40, 25, 12, 4), "Share of what you owe suppliers that is past due")
_m("PY06", "Top supplier share", "Payment discipline", "%", "L", (60, 45, 30, 20), "Largest supplier's share of purchases")
_m("PY09", "Statutory dues outstanding", "Payment discipline", "months", "L", (4, 3, 2, 1.3), "Unpaid statutory dues in months of normal build-up")
_m("TB2", "Supplier stretching trend", "Payment discipline", "days", "L", (10, 5, 2, 0), "Change in days late to suppliers, last 3 months vs the 3 before")
_m("MS1", "Micro and small suppliers paid in time", "Payment discipline", "%", "H", (50, 70, 85, 95),
   "Value paid to micro and small suppliers within the 45-day limit (Section 43B(h))")
# Debt service and funding
_m("FL01", "Debt to equity", "Debt service", "x", "L", (3, 2, 1.2, 0.6), "Borrowings for every 1 of owner's money", bank_norm=True)
_m("FL02", "Outside liabilities to net worth", "Debt service", "x", "L", (5, 4, 3, 2), "Everything owed to others for every 1 of net worth", bank_norm=True)
_m("FL04", "Net debt to EBITDA", "Debt service", "x", "L", (6, 4, 3, 1.5), "Years of operating profit needed to repay net borrowings")
_m("FL05", "Interest cover", "Debt service", "x", "H", (1.5, 2.0, 2.5, 4), "Times operating profit covers interest", bank_norm=True)
_m("FL06", "Debt service cover", "Debt service", "x", "H", (1.0, 1.25, 1.5, 2.0), "Operating profit for every 1 of interest and repayments", bank_norm=True)
_m("FL07", "Short-term share of debt", "Debt service", "%", "L", (90, 75, 55, 35), "Share of borrowing due or renewable within a year",
   fam={"trading": (100, 95, 85, 65)})
_m("FL08", "Borrowing growth", "Debt service", "%", "L", (60, 35, 15, 0), "Growth in borrowings over 12 months")
_m("TB3", "New-borrowing frequency", "Debt service", "count", "L", (9, 6, 4, 2), "Months in the last 12 with new short-term borrowing")
# Profitability
_m("PF01", "Gross margin", "Profitability", "%", "H", (6, 10, 15, 22), "Profit left after the cost of goods sold",
   fam={"manufacturing": (12, 20, 28, 38), "services": (25, 35, 50, 65)})
_m("PF02", "EBITDA margin", "Profitability", "%", "H", (1.5, 3, 5, 8), "Operating profit before depreciation and interest",
   fam={"manufacturing": (4, 8, 12, 18), "services": (5, 10, 15, 22)})
_m("PF04", "Net margin", "Profitability", "%", "H", (0, 1.5, 3, 5), "Final profit per 100 of sales",
   fam={"manufacturing": (0, 3, 6, 10), "services": (0, 5, 9, 14)})
_m("PF05", "Running-cost ratio", "Profitability", "%", "L", (16, 12, 9, 6), "Operating expenses per 100 of sales",
   fam={"manufacturing": (35, 28, 20, 14), "services": (45, 35, 27, 20)})
_m("PF10", "Interest-cost ratio", "Profitability", "%", "L", (4, 2.5, 1.5, 0.7), "Interest per 100 of sales")
_m("PF12", "Margin of safety", "Profitability", "%", "H", (0, 10, 25, 40), "How far sales can fall before the business loses money")
# Stability and stock
_m("RC19", "Sales growth", "Stability", "%", "H", (-10, 0, 5, 12), "Sales against the same 12 months a year earlier")
_m("RS01", "Sales steadiness", "Stability", "%", "L", (60, 40, 25, 12), "How much monthly sales swing (coefficient of variation)")
_m("RS02", "Margin steadiness", "Stability", "pp", "L", (6, 4, 2.5, 1), "How much the gross margin moves month to month")
_m("RS03", "Profitable months", "Stability", "%", "H", (50, 65, 80, 92), "Share of the last 12 months with a profit")
_m("RS04", "Cash against worst month", "Stability", "months", "H", (0.5, 1, 2, 4), "Today's cash against the worst month of cash outflow")
_m("IN01", "Stock days", "Stability", "days", "L", (160, 120, 80, 50), "Days of cost of sales held in stock",
   fam={"trading": (110, 80, 55, 35), "services": None})
_m("IN04", "Stock vs cost-of-sales growth", "Stability", "pp", "L", (20, 10, 3, 0), "Stock growth minus cost-of-sales growth", fam={"services": None})
# Owner behaviour
_m("OW02", "Drawings to profit", "Owner behaviour", "%", "L", (100, 85, 65, 40), "Share of profit the owner takes out")
_m("OW03", "Drawings to operating cash flow", "Owner behaviour", "%", "L", (100, 85, 65, 40), "Owner withdrawals against the cash the business produced")
# Data integrity
_m("DQ01", "Entries that balance", "Data integrity", "%", "H", (90, 95, 98, 99.5), "Share of journal entries where debits equal credits")
_m("DQ02", "Value classified", "Data integrity", "%", "H", (80, 90, 95, 99), "Share of money in recognised accounts")
_m("DQ03", "Parties named", "Data integrity", "%", "H", (70, 85, 95, 99), "Share of customer and supplier lines with a name")
_m("DQ04", "Invoices with due dates", "Data integrity", "%", "H", (50, 70, 85, 95), "Share of invoices that carry a due date")
_m("IC1", "Integrity checks passed", "Data integrity", "%", "H", (57, 71, 86, 100), "Share of the 7 integrity checks passed")

MEASURES = list(M)
assert len(MEASURES) == 54, len(MEASURES)
BY_AREA = {a: [m for m in MEASURES if M[m]["area"] == a] for a in AREA_LIST}
CAPACITY = ["CR02", "RS04", "FL07", "RC09", "PY03", "RS02", "OW03"]   # Grit's capacity part (plus the stress test)


def bounds_for(mid: str, fam: str):
    """Boundaries for a family; None when the measure does not apply (for example stock days for a services firm)."""
    f = M[mid]["fam"]
    if fam in f:
        return f[fam]
    return M[mid]["bounds"]


# ---------------------------------------------------------------------------------------------------- tiers and bands
TIERS = ["Safe", "Moderate", "Watch", "Risky"]
TIER_STRESS_BANDS = [("Safe", 0.05), ("Moderate", 0.12), ("Watch", 0.25), ("Risky", 1.01)]   # calibrated 90-day stress rate below this
GRIT_BANDS = [(75, "Strong"), (55, "Steady"), (40, "Needs work"), (0, "Weak")]
RATING_BANDS = [(80, "Excellent"), (60, "Good"), (40, "Fair"), (20, "Poor"), (0, "Critical")]


def grit_band(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "Not enough history"
    return next(b for lo, b in GRIT_BANDS if s >= lo)


def rating_band(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "Not rated"
    return next(b for lo, b in RATING_BANDS if s >= lo)
