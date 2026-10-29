"""Plain-English number formatting shared by every screen."""
from __future__ import annotations

SYMBOL = {"INR": "₹", "SGD": "S$", "USD": "$", "EUR": "€", "GBP": "£", "AED": "AED ", "MYR": "RM", "IDR": "Rp "}


def money(x: float | None, cur: str = "INR", short: bool = True) -> str:
    if x is None:
        return "-"
    sym = SYMBOL.get(cur, cur + " ")
    sign = "-" if x < 0 else ""
    a = abs(x)
    if not short:
        return f"{sign}{sym}{a:,.0f}"
    if cur == "INR":
        if a >= 1e9:
            return f"{sign}{sym}{a / 1e7:,.0f} cr"
        if a >= 1e7:
            return f"{sign}{sym}{a / 1e7:.2f} cr"
        if a >= 1e5:
            return f"{sign}{sym}{a / 1e5:.1f} lakh"
        return f"{sign}{sym}{a:,.0f}"
    if a >= 1e6:
        return f"{sign}{sym}{a / 1e6:.2f}m"
    if a >= 1e3:
        return f"{sign}{sym}{a / 1e3:.0f}k"
    return f"{sign}{sym}{a:,.0f}"


def value(v: float | None, unit: str, cur: str = "INR") -> str:
    if v is None:
        return "-"
    if unit == "money":
        return money(v, cur)
    if unit == "%":
        return f"{v:.1f}%"
    if unit == "pp":
        return f"{v:+.1f} pts"
    if unit == "days":
        return f"{v:.0f} days"
    if unit == "weeks":
        return f"{v:.1f} weeks"
    if unit == "months":
        return f"{v:.1f} months"
    if unit == "x":
        return f"{v:.2f}x"
    if unit == "count":
        return f"{v:.0f}"
    return f"{v:.2f}"


def pct(x: float, d: int = 0) -> str:
    return f"{x:.{d}f}%"
