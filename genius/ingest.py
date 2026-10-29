"""Ledger / journal ingestion.

Reads Excel, CSV or Word-table ledgers, finds the header row, maps columns with an
editable alias dictionary (config/header_aliases.csv), parses dates and amounts in the
formats small businesses actually use, rebuilds journal postings, classifies accounts with
an editable keyword dictionary (config/account_keywords.csv) plus structural inference,
and returns a quality report. Nothing is guessed silently: low-confidence choices are
listed in report["needs_confirmation"].
"""
from __future__ import annotations

import csv
import datetime as dt
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CFG = Path(__file__).resolve().parent.parent / "config"

# ----------------------------------------------------------------------------- helpers
def norm(s: Any) -> str:
    s = str(s).lower().replace("'", "").replace("\u2019", "").replace("&", " and ").replace("a/c", " ac ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


_ALIAS: dict[str, str] | None = None
_KEYWORDS: list[tuple[str, str, str]] | None = None


def load_aliases(path: Path | None = None) -> dict[str, str]:
    global _ALIAS
    if _ALIAS is None or path is not None:
        out: dict[str, str] = {}
        with open(path or CFG / "header_aliases.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.setdefault(norm(r["alias"]), r["field"])
        if path is None:
            _ALIAS = out
        else:
            return out
    return _ALIAS


CUSTOM_KW = CFG / "custom_keywords.csv"


def valid_types() -> dict[str, list[str]]:
    """account_type -> subtypes known to the built-in dictionary (custom keywords must use these)."""
    out: dict[str, set] = {}
    with open(CFG / "account_keywords.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["account_type"], set()).add(r["subtype"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def custom_keywords() -> list[tuple[str, str, str]]:
    if not CUSTOM_KW.exists():
        return []
    with open(CUSTOM_KW, encoding="utf-8") as f:
        return [(r["keyword"].strip(), r["account_type"].strip(), r["subtype"].strip()) for r in csv.DictReader(f) if (r.get("keyword") or "").strip()]


def load_keywords(path: Path | None = None) -> list[tuple[str, str, str]]:
    """Built-in dictionary plus the owner's custom keywords. Longest phrase wins; on equal length a custom keyword wins."""
    global _KEYWORDS
    if _KEYWORDS is None or path is not None:
        rows = []
        with open(path or CFG / "account_keywords.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((norm(r["keyword"]), r["account_type"], r["subtype"], 1))
        if path is None:
            rows += [(norm(k), t, st, 0) for k, t, st in custom_keywords() if norm(k)]
        rows.sort(key=lambda x: (-len(x[0]), x[3], x[0]))
        rows = [(k, t, st) for k, t, st, _ in rows]
        if path is None:
            _KEYWORDS = rows
        else:
            return rows
    return _KEYWORDS


def add_keywords(rows: list[tuple[str, str, str]]) -> dict:
    """Append keywords to the custom dictionary. Returns {'added': n, 'updated': n, 'rejected': [(row, reason)]}."""
    vt = valid_types()
    cur = {norm(k): (k, t, s) for k, t, s in custom_keywords()}
    added = updated = 0
    rejected = []
    for k, t, st in rows:
        k, t, st = (k or "").strip(), (t or "").strip().lower(), (st or "").strip().lower()
        if not norm(k):
            rejected.append(((k, t, st), "empty keyword")); continue
        if t not in vt:
            rejected.append(((k, t, st), f"account type must be one of {', '.join(vt)}")); continue
        if st not in vt[t]:
            rejected.append(((k, t, st), f"for {t} the subtype must be one of {', '.join(vt[t])}")); continue
        if norm(k) in cur:
            updated += cur[norm(k)][1:] != (t, st)
        else:
            added += 1
        cur[norm(k)] = (k, t, st)
    with open(CUSTOM_KW, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["keyword", "account_type", "subtype"])
        for k, t, st in cur.values():
            w.writerow([k, t, st])
    reload_config()
    return {"added": added, "updated": updated, "rejected": rejected}


def import_keywords(path: str | Path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        rows = [(r.get("keyword", ""), r.get("account_type", ""), r.get("subtype", "")) for r in rd]
    return add_keywords(rows)


def remove_keyword(keyword: str) -> None:
    keep = [(k, t, s) for k, t, s in custom_keywords() if norm(k) != norm(keyword)]
    with open(CUSTOM_KW, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["keyword", "account_type", "subtype"])
        for r in keep:
            w.writerow(r)
    reload_config()


def reload_config() -> None:
    global _ALIAS, _KEYWORDS
    _ALIAS = None
    _KEYWORDS = None


# ----------------------------------------------------------------------------- header matching
def match_header(cell: Any) -> tuple[str, float] | None:
    n = norm(cell)
    if not n or len(n) > 60:
        return None
    aliases = load_aliases()
    if n in aliases:
        return aliases[n], 1.0
    toks = set(n.split())
    best = None
    for a, f in aliases.items():
        at = a.split()
        if len(at) >= 2 and set(at) <= toks:
            if best is None or len(at) > best[2]:
                best = (f, 0.88, len(at))
    if best:
        return best[0], best[1]
    close = difflib.get_close_matches(n, aliases.keys(), n=1, cutoff=0.87)
    if close:
        return aliases[close[0]], round(difflib.SequenceMatcher(None, n, close[0]).ratio(), 2)
    return None


def row_header_score(row: list[Any]) -> int:
    fields = set()
    for c in row:
        if c is None or (isinstance(c, float) and np.isnan(c)):
            continue
        if isinstance(c, (int, float, dt.datetime, dt.date)):
            continue
        m = match_header(c)
        if m and m[1] >= 0.85:
            fields.add(m[0])
    return len(fields)


def detect_header(rows: list[list[Any]], limit: int = 60) -> int | None:
    best_i, best = None, 0
    for i, r in enumerate(rows[:limit]):
        s = row_header_score(r)
        if s > best:
            best_i, best = i, s
    return best_i if best >= 3 else None


# ----------------------------------------------------------------------------- value parsing
_CURR = re.compile(r"(₹|\brs\.?|\binr\b|s\$|\bsgd\b|\busd\b|\$|€|\baed\b|\bsar\b|\bmyr\b|rm\b)", re.I)


def parse_amount(v: Any) -> tuple[float | None, str | None]:
    """Return (signed value, side) where side is 'D', 'C' or None."""
    if v is None:
        return None, None
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        if isinstance(v, float) and np.isnan(v):
            return None, None
        return float(v), None
    s = str(v).strip()
    if not s or s.lower() in {"-", "–", "—", "nil", "null", "na", "n/a", "none"}:
        return None, None
    side = None
    m = re.search(r"(?:^|\s)(dr|cr|debit|credit)\.?\s*$", s, re.I) or re.match(r"^\s*(dr|cr)\.?\s", s, re.I)
    if m:
        side = "D" if m.group(1).lower().startswith("d") else "C"
        s = (s[: m.start()] + " " + s[m.end():]).strip()
    neg = (s.startswith("(") and s.endswith(")")) or s.startswith("-") or s.endswith("-")
    s = _CURR.sub("", s)
    s = re.sub(r"[^\d.]", "", s)
    if not s or s == ".":
        return None, side
    try:
        val = float(s)
    except ValueError:
        return None, side
    return (-val if neg else val), side


def _to_ts(v: Any) -> pd.Timestamp | None:
    if isinstance(v, (dt.datetime, dt.date, pd.Timestamp)):
        return pd.Timestamp(v)
    return None


def parse_dates(values: list[Any]) -> tuple[list[pd.Timestamp | None], dict]:
    """Parse a column of dates, deciding day-first vs month-first from the data."""
    num_pat = re.compile(r"^\s*(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})\s*$")
    day_ev = mon_ev = amb = 0
    for v in values:
        if isinstance(v, str):
            m = num_pat.match(v)
            if m and len(m.group(1)) <= 2 and len(m.group(3)) in (2, 4):
                a, b = int(m.group(1)), int(m.group(2))
                if a > 12:
                    day_ev += 1
                elif b > 12:
                    mon_ev += 1
                else:
                    amb += 1
    dayfirst = day_ev >= mon_ev
    notes = {"dayfirst": dayfirst, "ambiguous_dates": amb if (day_ev == 0 and mon_ev == 0) else 0}
    out: list[pd.Timestamp | None] = []
    for v in values:
        ts = _to_ts(v)
        if ts is not None:
            out.append(ts)
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
            continue
        if isinstance(v, (int, float)) and 20000 < float(v) < 80000:
            out.append(pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v)))
            continue
        s = str(v).strip()
        if not s:
            out.append(None)
            continue
        m = num_pat.match(s)
        try:
            if m:
                g1, g2, g3 = m.group(1), m.group(2), m.group(3)
                if len(g1) == 4:
                    y, mo, d = int(g1), int(g2), int(g3)
                else:
                    a, b, y = int(g1), int(g2), int(g3)
                    if y < 100:
                        y += 2000
                    d, mo = (a, b) if dayfirst else (b, a)
                out.append(pd.Timestamp(year=y, month=mo, day=d))
            else:
                ts = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
                out.append(None if pd.isna(ts) else pd.Timestamp(ts))
        except (ValueError, OverflowError):
            out.append(None)
    return out, notes


OPENING_TEXT = {"opening balance", "to opening balance", "by opening balance", "opening bal", "balance b f", "balance b d", "brought forward"}
_SKIP_TEXT = {"total", "grand total", "sub total", "subtotal", "closing balance",
              "balance c d", "balance c f", "c f", "carried forward", "net total", "totals",
              "by closing balance", "by balance c d", "to balance c d", "closing bal", "to closing balance"}


def is_skip_text(s: Any) -> bool:
    n = norm(s)
    if not n:
        return False
    return n in _SKIP_TEXT or n.startswith("total ") or n.startswith("grand total") or n.startswith("sub total")


# ----------------------------------------------------------------------------- readers
def _clean(v: Any) -> Any:
    if isinstance(v, str):
        v = v.replace(" ", " ").strip()
        return v or None
    return v


def read_sheets(path: str | Path) -> list[tuple[str, list[list[Any]]]]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(p, data_only=True)
        sheets = []
        for ws in wb.worksheets:
            if ws.sheet_state != "visible":
                continue
            rows = [[_clean(c) for c in r] for r in ws.iter_rows(values_only=True)]
            sheets.append((ws.title, rows))
        return sheets
    if ext == ".xls":
        xl = pd.read_excel(p, sheet_name=None, header=None)
        return [(n, [[_clean(c) if not (isinstance(c, float) and np.isnan(c)) else None for c in r]
                     for r in df.values.tolist()]) for n, df in xl.items()]
    if ext in (".csv", ".tsv", ".txt"):
        raw = p.read_bytes()
        text = None
        for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        assert text is not None
        sample = text[:5000]
        delim = ","
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            if sample.count("\t") > sample.count(","):
                delim = "\t"
        rows = [[_clean(c) for c in r] for r in csv.reader(text.splitlines(), delimiter=delim)]
        return [(p.stem, rows)]
    if ext == ".docx":
        import docx
        d = docx.Document(str(p))
        sheets = []
        for i, t in enumerate(d.tables, 1):
            rows = [[_clean(c.text) for c in r.cells] for r in t.rows]
            sheets.append((f"table{i}", rows))
        return sheets
    raise ValueError(f"Unsupported file type '{ext}'. Use .xlsx, .csv or .docx (tables).")


# ----------------------------------------------------------------------------- core
BLOCK_RE = re.compile(r"^\s*(?:ledger(?:\s+account)?|account|a/c|account name)\s*[:\-]\s*(.+?)\s*$", re.I)
GROUP_RE = re.compile(r"[|(,;]\s*(?:under|group|parent)\s*[:\-]?\s*([^|)]+)\)?\s*$", re.I)


def find_block_marker(row: list[Any]) -> tuple[str, str | None] | None:
    first = next((c for c in row if c not in (None, "")), None)
    if not isinstance(first, str):
        return None
    others = [c for c in row if c not in (None, "")]
    if len(others) > 2:
        return None
    m = BLOCK_RE.match(first)
    if not m:
        return None
    body = m.group(1)
    group = None
    g = GROUP_RE.search(body)
    if g:
        group = g.group(1).strip()
        body = body[: g.start()].strip()
    return body.strip(), group


def map_columns(header: list[Any], sample_rows: list[list[Any]], block_mode: bool) -> tuple[dict[int, str], list[dict]]:
    cand: list[tuple[int, str, float, str]] = []
    for j, h in enumerate(header):
        if h in (None, "") or isinstance(h, (int, float)):
            continue
        m = match_header(h)
        if m:
            cand.append((j, m[0], m[1], str(h)))
    mapping: dict[int, str] = {}
    info: list[dict] = []
    by_field: dict[str, list[tuple[int, float, str]]] = {}
    for j, f, sc, h in cand:
        by_field.setdefault(f, []).append((j, sc, h))
    for f, items in by_field.items():
        items.sort(key=lambda x: (-x[1], x[0]))
        j, sc, h = items[0]
        mapping[j] = f
        for j2, sc2, h2 in items[1:]:
            if f == "account":
                mapping[j2] = "counter_account"
                info.append({"column": h2, "field": "counter_account", "confidence": sc2, "note": "second account-like column"})
            else:
                info.append({"column": h2, "field": f, "confidence": sc2, "note": "duplicate, ignored"})
    # 'Type' ambiguity: Dr/Cr marker versus voucher type, decided from the values.
    for j, f in list(mapping.items()):
        if f == "drcr":
            vals = [norm(r[j]) for r in sample_rows if j < len(r) and r[j] not in (None, "")]
            if vals and sum(v in {"dr", "cr", "debit", "credit", "d", "c"} for v in vals) / len(vals) < 0.8:
                mapping[j] = "voucher_type"
    if block_mode:
        for j, f in list(mapping.items()):
            if f == "account":
                mapping[j] = "counter_account"
    for j, f in mapping.items():
        h = header[j]
        sc = next((c[2] for c in cand if c[0] == j), 1.0)
        info.append({"column": str(h), "field": f, "confidence": sc, "note": ""})
    return mapping, info


@dataclass
class IngestResult:
    postings: pd.DataFrame
    report: dict = field(default_factory=dict)

    @property
    def currency(self) -> str:
        return self.report.get("currency", "INR")

    def summary(self) -> str:
        r = self.report
        lines = [f"File: {r.get('file')}  layout: {r.get('layout')}  quality: {r.get('quality_score')}/100",
                 f"Postings: {len(self.postings)}  entries: {r['checks'].get('entries')}  "
                 f"period: {r['checks'].get('first_date')} to {r['checks'].get('last_date')}"]
        for k, v in r["checks"].items():
            lines.append(f"  {k}: {v}")
        for w in r["warnings"]:
            lines.append(f"  ! {w}")
        if r["needs_confirmation"]:
            lines.append(f"  needs confirmation: {r['needs_confirmation']}")
        return "\n".join(lines)


CONTROL_PHRASES = {"sundry debtors", "debtors", "trade receivables", "trade receivable", "accounts receivable",
                   "a r", "receivables", "customers control", "customer control", "book debts", "bills receivable",
                   "accounts receivable control", "sundry debtors control", "sundry creditors", "creditors",
                   "trade payables", "trade payable", "accounts payable", "a p", "payables", "suppliers control",
                   "vendors control", "bills payable", "accounts payable control", "sundry creditors control",
                   "trade creditors", "msme creditors", "msme payables"}


def _best_kw(text: str | None):
    if not text:
        return None
    n = f" {norm(text)} "
    for kw, t, s in load_keywords():
        if f" {kw} " in n:
            return kw, t, s
    return None


REVENUE_HEAD = ("sales", "revenue", "turnover")
REVENUE_TAIL = ("income", "revenue", "receipts", "sales")
EXPENSE_WORDS = ("expense", "expenses", "charges", "cost", "costs", "promotion", "paid", "allowed", "commission", "incentive", "discount", "tax", "payable", "receivable", "a c", "ac")


def _income_cue(name: str):
    """A name that starts with 'Sales', 'Revenue' or 'Turnover', or ends with 'Income' or 'Revenue', is an income account whatever other
    words it carries ('Sales - Building Materials' is not a building; 'Freight Income' is not freight paid). The longest income keyword
    in the name decides revenue or other income; with none, it is revenue."""
    n = norm(name)
    words = n.split()
    if n.startswith(("direct cost", "direct expense", "cost of goods", "cost of sales", "cost of revenue", "cost of services")):
        return n.split()[0] + " " + n.split()[1], "expense", "cogs"
    tail_income = words[-1] in ("income", "revenue", "receipts") and "tax" not in words
    if not tail_income and (any(w in words for w in EXPENSE_WORDS) or words[0] not in REVENUE_HEAD):
        return None
    padded = f" {n} "
    for kw, t, s in load_keywords():
        if t == "income" and f" {kw} " in padded:
            return kw, t, s
    return words[0] if words[0] in REVENUE_HEAD else words[-1], "income", "revenue"


def classify_name(name: str, group: str | None = None) -> tuple[str | None, str | None, float, str]:
    """Longest keyword wins; between a group label and an account name the name wins ties. Income cues at the start or end of a name
    come first (see _income_cue)."""
    cue = _income_cue(name) if name else None
    if cue:
        return cue[1], cue[2], 0.9, f"name cue:{cue[0]}"
    g, a = _best_kw(group), _best_kw(name)
    pick = None
    if a and (not g or len(a[0]) >= len(g[0])):
        pick = ("keyword", a)
    elif g:
        pick = ("group", g)
    if pick:
        how, (kw, t, s) = pick
        conf = 0.95 if how == "group" else 0.9
        if len(kw) <= 4 and len(kw.split()) == 1:
            conf -= 0.1
        return t, s, conf, f"{how}:{kw}"
    n = norm(name)
    if n:
        pool = [k for k, _, _ in load_keywords()]
        close = difflib.get_close_matches(n, pool, n=1, cutoff=0.88)
        if close:
            t, s = next((t, s) for k, t, s in load_keywords() if k == close[0])
            return t, s, 0.6, f"fuzzy:{close[0]}"
    return None, None, 0.0, "unknown"


def infer_party_accounts(post: pd.DataFrame, cls: dict[str, dict]) -> None:
    """Classify accounts keywords could not (usually customer or supplier names) from how they are used:
    a name debited alongside sales is a customer, a name credited alongside purchases or expenses is a supplier."""
    unknown = [a for a, c in cls.items() if c["type"] is None]
    if not unknown:
        return
    sub = post["account"].map(lambda a: cls.get(a, {}).get("subtype"))
    work = post.assign(_sub=sub)
    sales_like = {"revenue"}
    cost_like = {"cogs", "inventory", "input_tax", "fixed_asset", "repairs", "freight_out", "utilities", "rent",
                 "professional_fees", "marketing", "admin", "insurance", "payroll"}
    by_entry = {k: g for k, g in work.groupby("entry_no")}
    for acct in unknown:
        votes = {"receivable": 0, "payable": 0}
        cash_uses = {"in": 0, "out": 0}
        for eno in work.loc[work.account == acct, "entry_no"].dropna().unique():
            ent = by_entry[eno]
            others = set(ent.loc[ent.account != acct, "_sub"].dropna())
            mine = ent[ent.account == acct]
            d, c = mine.debit.sum(), mine.credit.sum()
            if d > c and others & sales_like:
                votes["receivable"] += 1
            elif c > d and others & cost_like:
                votes["payable"] += 1
            elif c > d and others & {"bank", "cash"}:
                cash_uses["in"] += 1
            elif d > c and others & {"bank", "cash"}:
                cash_uses["out"] += 1
        if votes["receivable"] > votes["payable"] and votes["receivable"] > 0:
            cls[acct].update(type="asset", subtype="receivable", confidence=0.7, method="structure:debited with sales")
        elif votes["payable"] > votes["receivable"]:
            cls[acct].update(type="liability", subtype="payable", confidence=0.7, method="structure:credited with purchases or costs")


def detect_currency(sheets) -> tuple[str, bool]:
    """Currency from symbols and local statutory words in the file. Returns (code, certain)."""
    import re
    from collections import Counter
    cnt = Counter()
    for _, rows in sheets:
        txt = " ".join(str(c) for r in rows[:600] for c in r if c not in (None, ""))
        for k in re.findall(r"SGD|S\$|INR|₹|\bRs\.?|\bCPF\b|\bIRAS\b|\bTDS\b|\bESI\b|\bMYR\b|\bUSD\b", txt):
            cnt[k] += 1
    sgd = cnt["SGD"] * 5 + cnt["S$"] + cnt["CPF"] * 3 + cnt["IRAS"] * 3
    inr = cnt["INR"] * 5 + cnt["₹"] + cnt["Rs"] + cnt["Rs."] + cnt["TDS"] * 3 + cnt["ESI"] * 3
    if sgd > inr and sgd > 0:
        return "SGD", True
    if inr > 0:
        return "INR", True
    return "INR", False


def ingest(path: str | Path, overrides: dict[str, tuple[str, str]] | None = None) -> IngestResult:
    p = Path(path)
    warnings: list[str] = []
    needs: list[str] = []
    sheet_reports: list[dict] = []
    recs: list[dict] = []
    layout = "journal"
    groups_by_account: dict[str, str] = {}
    date_notes: dict = {}

    sheets = read_sheets(p)
    cur_code, cur_certain = detect_currency(sheets)
    for sname, rows in sheets:
        if not rows:
            continue
        markers = [(i, find_block_marker(r)) for i, r in enumerate(rows)]
        markers = [(i, m) for i, m in markers if m]
        block_mode = len(markers) >= 2
        h_idx = detect_header(rows)
        if h_idx is None:
            sheet_reports.append({"sheet": sname, "used": False, "reason": "no header row recognised"})
            continue
        header = rows[h_idx]
        width = max(len(r) for r in rows)
        header = list(header) + [None] * (width - len(header))
        sample = [r for r in rows[h_idx + 1: h_idx + 40] if any(c not in (None, "") for c in r)]
        mapping, minfo = map_columns(header, sample, block_mode)
        fields = set(mapping.values())
        have_amount = ({"debit", "credit"} & fields) or "amount" in fields
        if not have_amount or "date" not in fields:
            sheet_reports.append({"sheet": sname, "used": False, "reason": "needs a date and amount/debit/credit column", "mapping": minfo})
            continue
        sheet_account = None
        if "account" not in fields and not block_mode:
            if "counter_account" in fields and "party" not in fields:
                pass
            sheet_account = sname
            layout = "sheet_per_account"
        if block_mode:
            layout = "ledger_blocks"
        cur_acct, cur_group = None, None
        n_before = len(recs)
        marker_at = dict(markers)
        above = [i for i in marker_at if i < h_idx]
        if block_mode and above:  # first block's marker sits above the detected header row
            cur_acct, cur_group = marker_at[max(above)]
            if cur_group:
                groups_by_account[cur_acct] = cur_group
        for i in range(h_idx, len(rows)):
            r = list(rows[i]) + [None] * (width - len(rows[i]))
            if i in marker_at:
                cur_acct, cur_group = marker_at[i]
                if cur_group:
                    groups_by_account[cur_acct] = cur_group
                continue
            if i == h_idx or row_header_score(r) >= max(3, row_header_score(header) - 1) and i != h_idx:
                continue
            if not any(c not in (None, "") for c in r):
                continue
            rec = {f: r[j] for j, f in mapping.items()}
            if block_mode:
                if cur_acct is None:
                    continue
                rec["account"] = cur_acct
                texts = [rec.get("counter_account"), rec.get("narration")]
                if any(norm(t) in OPENING_TEXT for t in texts):
                    rec["_opening"] = True
                    has_amt = any(rec.get(k) not in (None, "") for k in ("debit", "credit", "amount"))
                    if not has_amt and rec.get("balance") not in (None, ""):
                        rec["amount"] = rec["balance"]
                elif any(is_skip_text(t) for t in texts):
                    continue
            else:
                if sheet_account and not rec.get("account"):
                    rec["account"] = sheet_account
                if is_skip_text(rec.get("account")) or is_skip_text(rec.get("narration")) and not rec.get("account"):
                    continue
                if not rec.get("account") and not any(rec.get(k) not in (None, "") for k in ("debit", "credit", "amount")):
                    continue
            rec["_sheet"], rec["_row"] = sname, i + 1
            recs.append(rec)
        sheet_reports.append({"sheet": sname, "used": True, "header_row": h_idx + 1, "rows": len(recs) - n_before,
                              "layout": "ledger_blocks" if block_mode else layout, "mapping": minfo})
        for m in minfo:
            if m["confidence"] < 0.9 and m["note"] == "":
                needs.append(f"column '{m['column']}' read as {m['field']} (confidence {m['confidence']})")

    if not recs:
        raise ValueError("No ledger rows could be read. Check that the file has a header row with date, account and amount columns.")

    df = pd.DataFrame(recs)
    for col in ["date", "entry_no", "account", "debit", "credit", "amount", "drcr", "party", "invoice_ref", "due_date",
                "narration", "voucher_type", "counter_account"]:
        if col not in df:
            df[col] = None

    # amounts -> debit / credit
    deb, cre, assumed_sign = [], [], 0
    for _, r in df.iterrows():
        d, ds = parse_amount(r["debit"])
        c, cs = parse_amount(r["credit"])
        a, asd = parse_amount(r["amount"])
        dv = cv = None
        if d is not None or c is not None:
            dv, cv = (abs(d) if d else None), (abs(c) if c else None)
            if (d is not None and d < 0 and not c):
                cv, dv = abs(d), None
        elif a is not None:
            side = asd
            if side is None and r["drcr"] not in (None, ""):
                side = "D" if norm(r["drcr"]).startswith("d") else "C"
            if side is None:
                assumed_sign += 1
                side = "D" if a >= 0 else "C"
            dv, cv = (abs(a), None) if side == "D" else (None, abs(a))
        deb.append(dv or 0.0)
        cre.append(cv or 0.0)
    df["debit"], df["credit"] = deb, cre
    if assumed_sign:
        warnings.append(f"{assumed_sign} amounts had no Dr/Cr marker; positive read as debit, negative as credit")
    df = df[(df.debit != 0) | (df.credit != 0)].copy()

    # dates
    dates, date_notes = parse_dates(df["date"].tolist())
    df["date"] = pd.to_datetime(pd.Series(dates, index=df.index))
    dues, _ = parse_dates(df["due_date"].tolist())
    df["due_date"] = pd.to_datetime(pd.Series(dues, index=df.index))
    if "_opening" in df:
        op = df["_opening"].fillna(False).astype(bool)
        if op.any():
            first_tx = df.loc[~op, "date"].min()
            df.loc[op, "date"] = first_tx
            df.loc[op, "entry_no"] = "OPENING-BAL"
            df.loc[op, "voucher_type"] = "Opening balance"
            warnings.append("opening balances read per account; invoice-level due dates for them are not available")
    if date_notes.get("ambiguous_dates"):
        warnings.append(f"{date_notes['ambiguous_dates']} dates could be day-first or month-first; read as day-first")
        needs.append("date format (day-first assumed)")
    for col in ("entry_no",):
        df[col] = df[col].map(lambda v: None if v in (None, "") or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
    blank = df["date"].isna()
    if blank.any():
        df["date"] = df["date"].ffill()
        warnings.append(f"{int(blank.sum())} lines had no date and took the date of the line above")
    if df["entry_no"].isna().any() and df["entry_no"].notna().any():
        df["entry_no"] = df["entry_no"].ffill()
    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        warnings.append(f"{bad} lines had unreadable dates and were dropped")
        df = df[df["date"].notna()]

    NULLISH = {"", "nan", "none", "null", "nil", "n/a", "na", "-", "–", "—", "#n/a"}
    for col in ("account", "party", "invoice_ref", "narration", "voucher_type", "counter_account"):
        df[col] = df[col].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip().lower() in NULLISH else str(v).strip())
    df = df[df["account"].notna()].copy()
    if df["entry_no"].notna().any():  # voucher type / narration are often shown on the first line of an entry only
        df["voucher_type"] = df.groupby("entry_no", sort=False)["voucher_type"].transform(lambda s: s.ffill().bfill())

    # entry grouping when no entry number
    if df["entry_no"].isna().all():
        warnings.append("no entry number column; entries rebuilt by running balance (less reliable)")
        run, n, ids = 0.0, 1, []
        for d, c in zip(df.debit, df.credit):
            ids.append(f"AUTO-{n:05d}")
            run += d - c
            if abs(run) < 0.01:
                n += 1
                run = 0.0
        df["entry_no"] = ids
        needs.append("entry numbers rebuilt")

    # classification
    accts = df["account"].unique().tolist()
    cls: dict[str, dict] = {}
    for a in accts:
        t, s, conf, how = classify_name(a, groups_by_account.get(a))
        cls[a] = {"type": t, "subtype": s, "confidence": conf, "method": how}
    infer_party_accounts(df, cls)
    for a, v in (overrides or {}).items():
        if a in cls:
            cls[a].update(type=v[0], subtype=v[1], confidence=1.0, method="user")
    df["account_type"] = df["account"].map(lambda a: cls[a]["type"])
    df["subtype"] = df["account"].map(lambda a: cls[a]["subtype"])
    df["classify_method"] = df["account"].map(lambda a: cls[a]["method"])

    # party from party-named sub-ledgers
    def is_control(a: str) -> bool:
        return norm(a) in CONTROL_PHRASES or cls[a]["subtype"] in ("receivable", "payable") and cls[a]["method"].startswith("keyword") and cls[a]["method"].split(":", 1)[1] == norm(a)
    sub_party = df["subtype"].isin(["receivable", "payable"]) & df["party"].isna() & ~df["account"].map(is_control)
    df.loc[sub_party, "party"] = df.loc[sub_party, "account"]

    unclassified = sorted(a for a, c in cls.items() if c["type"] is None)
    low = sorted(a for a, c in cls.items() if c["type"] and c["confidence"] < 0.7)
    if unclassified:
        needs.append(f"{len(unclassified)} accounts not recognised: {unclassified[:8]}")
    if low:
        needs.append(f"{len(low)} accounts matched loosely: {low[:8]}")

    # checks
    g = df.groupby("entry_no")[["debit", "credit"]].sum()
    diff = (g.debit - g.credit).abs()
    balanced = float((diff < 0.51).mean())
    first, last = df["date"].min(), df["date"].max()
    months = (last.year - first.year) * 12 + last.month - first.month + 1
    ar_ap = df[df.subtype.isin(["receivable", "payable"])]
    party_cov = float(ar_ap.party.notna().mean()) if len(ar_ap) else 1.0
    inv = ar_ap[((ar_ap.subtype == "receivable") & (ar_ap.debit > 0)) | ((ar_ap.subtype == "payable") & (ar_ap.credit > 0))]
    due_cov = float(inv.due_date.notna().mean()) if len(inv) else 1.0
    value = df.debit + df.credit
    class_cov = float(value[df.account_type.notna()].sum() / value.sum()) if value.sum() else 0.0
    total_d, total_c = df.debit.sum(), df.credit.sum()
    checks = {
        "entries": int(g.shape[0]), "first_date": str(first.date()), "last_date": str(last.date()), "months": int(months),
        "balanced_entries_pct": round(balanced * 100, 2),
        "total_debit": round(float(total_d), 2), "total_credit": round(float(total_c), 2),
        "party_coverage_pct": round(party_cov * 100, 1), "due_date_coverage_pct": round(due_cov * 100, 1),
        "classified_value_pct": round(class_cov * 100, 1),
        "accounts": len(accts), "unclassified_accounts": len(unclassified),
    }
    if balanced < 0.999:
        warnings.append(f"{round((1 - balanced) * 100, 1)}% of entries do not balance; scores are blocked until fixed or confirmed")
    if months < 12:
        warnings.append("under 12 months of history: Grit and seasonality will show 'insufficient data'")
    if due_cov < 0.95:
        warnings.append("some invoices have no due date; a default term will be assumed for ageing")
    score = 100.0
    score -= (1 - balanced) * 60
    score -= (1 - class_cov) * 25
    score -= (1 - party_cov) * 5
    score -= (1 - due_cov) * 10
    score -= min(10, len(needs) * 2)
    if months < 12:
        score -= 10
    report = {
        "file": p.name, "layout": layout, "sheets": sheet_reports, "date_format": date_notes,
        "unclassified_list": unclassified,
        "accounts": {a: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()} for a, c in cls.items()},
        "checks": checks, "warnings": warnings, "needs_confirmation": needs,
        "quality_score": int(max(0, round(score))),
        "currency": cur_code,
    }
    if not cur_certain:
        warnings.append("No currency symbol found in the file; assuming INR. Change it on the confirm screen if wrong.")
        needs.append("currency")
    cols = ["date", "entry_no", "account", "debit", "credit", "party", "invoice_ref", "due_date", "narration",
            "voucher_type", "counter_account", "account_type", "subtype", "classify_method", "_sheet", "_row"]
    out = df[cols].rename(columns={"_sheet": "source_sheet", "_row": "source_row"}).sort_values(["date", "entry_no"], kind="stable").reset_index(drop=True)
    return IngestResult(out, report)


if __name__ == "__main__":
    import sys
    res = ingest(sys.argv[1])
    print(res.summary())
