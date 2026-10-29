"""HTML for every owner screen. One small design system (colours, type, cards, pills), so the app looks like the agreed mockups."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from . import config as C
from . import fmt

ACC = "#0B5D5A"
TAKE = "<span class='pill safe'>Take it</span>"
SKIP = "<span class='pill grey'>Skip</span>"
TIER_CLS = {"Safe": "safe", "Moderate": "mod", "Watch": "watch", "Risky": "risk"}
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.gradio-container{font-family:'Inter',system-ui,sans-serif !important;background:#F4F6F7 !important;max-width:1240px !important}
.g{font-family:'Inter',system-ui,sans-serif;color:#13212B;font-size:15px;line-height:1.5;font-variant-numeric:tabular-nums}
.g a{color:#0B5D5A}
.g h1{margin:0;font-size:26px;line-height:1.2;font-weight:700;letter-spacing:-0.02em}
.g h2{margin:0 0 10px;font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#5E6B75}
.g .sub{margin:6px 0 0;font-size:15.5px;color:#4A5863;max-width:820px}
.g .card{background:#fff;border:1px solid #E2E7EA;border-radius:14px;padding:20px;box-sizing:border-box;min-width:0}
.g .grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr))}
.g .grid2{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr))}
.g .muted{color:#5E6B75}.g .note{font-size:12.5px;color:#5E6B75}
.g .pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:12.5px;font-weight:600;white-space:nowrap}
.g .safe{background:#E3F2EA;color:#17633F}.g .mod{background:#E4EDF8;color:#24528C}.g .watch{background:#FBEFD9;color:#8A5208}
.g .risk{background:#F8E3E2;color:#9A2A27}.g .grey{background:#EEF1F3;color:#4A5863}
.g .kpi{font-size:32px;font-weight:700;letter-spacing:-0.02em;line-height:1.1}
.gradio-container input,.gradio-container textarea,.gradio-container select,.gradio-container button{font-family:'Inter',system-ui,sans-serif !important}
.g .standing-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;min-width:0;container-type:inline-size}
.g .standing-stats .soft{min-width:0}
.g .standing-stats h2{font-size:11px;letter-spacing:.04em;overflow-wrap:anywhere}
.g .standing-stats .kpi{font-size:28px;font-size:clamp(20px,7cqi,28px);white-space:nowrap;margin:6px 0}
.g .standing-stats .note{overflow-wrap:anywhere}
@media(max-width:420px){.g .card{padding:16px}.g .standing-stats{gap:10px}.g .standing-stats .soft{padding:12px}}
.g .soft{background:#F7F9FA;border-radius:12px;padding:14px}
.g table.t{width:100%;border-collapse:collapse;font-size:14px}
.g table.t th{text-align:left;font-size:12px;font-weight:600;color:#5E6B75;padding:8px 10px;border-bottom:1px solid #E2E7EA;white-space:nowrap}
.g table.t td{padding:10px;border-bottom:1px solid #EEF1F3;vertical-align:top}
.g table.t tr:last-child td{border-bottom:0}
.g .row{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-top:1px solid #EEF1F3;font-size:14px}
.g .up{color:#17633F;font-weight:600}.g .dn{color:#9A2A27;font-weight:600}
.g .banner{display:flex;gap:10px;padding:12px 16px;border-radius:12px;background:#FBEFD9;border:1px solid #EFD7A8;color:#5E3A05;font-size:14px}
.g .ok{background:#F3F8F8;border:1px solid #CFE0DF;border-radius:12px;padding:12px 16px;font-size:14px}
.g .pos{position:relative;height:8px;border-radius:999px;background:#EAF0F1;min-width:120px}
.g .pos .med{position:absolute;top:-4px;width:2px;height:16px;background:#7D8A93}
.g .pos .you{position:absolute;top:-3px;width:14px;height:14px;margin-left:-7px;border-radius:50%;border:2px solid #fff;box-sizing:border-box}
.g .hdr{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;background:#fff;border:1px solid #E2E7EA;border-radius:14px;padding:12px 18px}
.g .logo{width:30px;height:30px;border-radius:8px;background:#0B5D5A;display:inline-flex;align-items:center;justify-content:center}
.g .scale{position:relative;display:flex;height:10px;border-radius:999px;overflow:hidden}
.g .drv{display:flex;gap:10px;align-items:flex-start;padding:10px 0;border-top:1px solid #EEF1F3;font-size:14px}
.g .bar{height:10px;border-radius:999px}
.g table.t,.g table.t th,.g table.t td{border-left:0 !important;border-right:0 !important;border-top:0 !important;background:transparent}
.g table.t,.g table.t tr,.g table.t tbody,.g table.t thead{border:0 !important;margin:0}
.g table.t th{border-bottom:1px solid #E2E7EA !important}.g table.t td{border-bottom:1px solid #EEF1F3 !important}
.g table.t tr:last-child td{border-bottom:0 !important}
.gradio-container table, .gradio-container .table-wrap *{font-family:'Inter',system-ui,sans-serif !important;font-size:13px}
.g .legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:#5E6B75;margin-top:8px}
.g .legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
"""


def esc(x):
    return html.escape(str(x))


def money(x, short=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return fmt.money(float(x), "INR", short)


def owner_reason(r):
    """Lender-side reasons rewritten for the owner: the hold is explained, the model behind it is not named."""
    if "Second Opinion" in r:
        import re
        m = re.search(r"(\d+)%", r)
        return f"Limits are held at {m.group(1) if m else 50}% until a credit reviewer at the lender has looked at the file."
    return r


def tier_pill(t, big=False):
    if not t:
        return '<span class="pill grey">Not rated</span>'
    st = ' style="font-size:14px;padding:5px 12px"' if big else ""
    return f'<span class="pill {TIER_CLS.get(t, "grey")}"{st}>{esc(t)}</span>'


def grit_pill(s):
    b = C.grit_band(s)
    cls = {"Strong": "safe", "Steady": "mod", "Needs work": "watch", "Weak": "risk"}.get(b, "grey")
    return f'<span class="pill {cls}">{esc(b)}</span>'


def rating_pill(r):
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return '<span class="pill grey">n/a</span>'
    cls = "safe" if r >= 70 else ("mod" if r >= 55 else ("watch" if r >= 35 else "risk"))
    return f'<span class="pill {cls}">{r:.0f}</span>'


def val(mid, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    u = C.M[mid]["unit"]
    if u == "index":
        return f"{v:.2f}"
    if v in (99.0, 999.0) and mid in ("FL05", "FL06", "FL01", "FL02", "FL04", "OW02", "OW03", "LQ01", "LQ03"):
        return {99.0: "no borrowing" if mid.startswith("FL") else "very high", 999.0: "more than the business produced"}[v]
    return fmt.value(float(v), u, "INR")


ICON_UP = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#17633F" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M6 11l6-6 6 6"/></svg>'
ICON_DN = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#9A2A27" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M6 13l6 6 6-6"/></svg>'
ICON_OK = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#17633F" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>'
ICON_WARN = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8A5208" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.5v.5"/></svg>'


def wrap(inner):
    return f'<div class="g">{inner}</div>'


def header(sess=None, presenter=False):
    right = '<span class="muted" style="font-size:13px">No books loaded yet</span>'
    if sess is not None and getattr(sess, "a", None) is not None:
        a = sess.a
        right = (f'<span style="font-weight:600;font-size:13px">{esc(a.name)}</span><span class="muted" style="font-size:13px">Books to '
                 f'{a.months[-1].strftime("%b %Y")}</span>{tier_pill(a.tier)}')
    return wrap(f'''<div class="hdr"><div style="display:flex;align-items:center;gap:10px">
<span class="logo"><svg width="20" height="20" viewBox="0 0 30 30"><path d="M6 19l4.5-4.5 3.5 3.5 6-7" stroke="#fff" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></span>
<span style="font-size:19px;font-weight:700">Genius</span><span class="muted" style="font-size:13px">Credit standing for your business · prototype on a synthetic peer database</span></div>
<div style="display:flex;align-items:center;gap:10px">{right}</div></div>''')


def empty(msg="Load books on the Upload tab first."):
    return wrap(f'<div class="card"><p class="muted" style="margin:0">{esc(msg)}</p></div>')


# ------------------------------------------------------------------------------------------------ upload
def upload_check(r: dict, name: str):
    rep = r["report"]; ch = rep["checks"]
    q = rep["quality_score"]
    cls = "safe" if q >= 85 else ("watch" if q >= 60 else "risk")
    rows = []
    passed = sum(1 for c in r["checks"] if c["passed"])
    rows.append((passed == 7, f"{passed} of 7 integrity checks passed", "; ".join(c["check"] for c in r["checks"] if not c["passed"]) or "Every entry balances and the statements tie"))
    un = ch.get("unclassified_accounts", 0)
    rows.append((un == 0, f"{ch.get('accounts', 0) - un} of {ch.get('accounts', 0)} accounts recognised", ", ".join(rep.get("unclassified_list", [])[:6]) or "All accounts classified"))
    rows.append((ch.get("party_coverage_pct", 0) >= 90, f"Customer and supplier named on {ch.get('party_coverage_pct', 0):.0f}% of lines", "Needed for collections and payment measures"))
    rows.append((ch.get("due_date_coverage_pct", 0) >= 90, f"Due dates on {ch.get('due_date_coverage_pct', 0):.0f}% of invoices", "Needed for overdue and on-time measures"))
    rows.append((ch.get("months", 0) >= 24, f"{ch.get('months', 0)} months of history", "12 needed for a score; 24 lets Grit remove seasonal patterns"))
    items = "".join(f'<div class="drv">{ICON_OK if ok else ICON_WARN}<div><b>{esc(t)}</b><div class="note">{esc(d)}</div></div></div>' for ok, t, d in rows)
    warn = "".join(f"<li>{esc(w)}</li>" for w in rep.get("warnings", [])[:4])
    return wrap(f'''<div class="card"><h2>Your books</h2>
<div class="ok" style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap"><div><b>{esc(name)}</b><div class="note">{esc(rep.get("file"))} · {ch.get("months")} months ·
{ch.get("entries", 0):,} entries · layout: {esc(rep.get("layout"))}</div></div><div style="display:flex;align-items:center;gap:8px"><span class="note">Data confidence</span>
<span class="kpi" style="font-size:26px">{q}</span><span class="pill {cls}">{"High" if q >= 85 else ("Fair" if q >= 60 else "Low")}</span></div></div>
<div style="margin-top:8px">{items}</div>{f'<ul class="note">{warn}</ul>' if warn else ""}</div>''')


# ------------------------------------------------------------------------------------------------ credit standing
def standing(a, db, pol):
    if a.score is None:
        why = "the books fail an integrity check" if a.gs["ko"]["K5"][0, -1] else "too few areas could be rated (at least 12 months and liquidity and debt-service data are needed)"
        return wrap(f'<div class="card"><h1>No score published</h1><p class="sub">Genius did not publish a score because {why}. Fix the items on the Upload tab and read the file again.</p></div>')
    c = db.cal.cutoffs
    sc = a.score
    seg = [("Risky", c["Watch"], "#E9B8B6"), ("Watch", c["Moderate"] - c["Watch"], "#F2D49E"), ("Moderate", c["Safe"] - c["Moderate"], "#9FBDE3"), ("Safe", 100 - c["Safe"], "#9CD2B5")]
    scale = "".join(f'<div style="width:{w}%;background:{col}"></div>' for _, w, col in seg)
    rng = {"Risky": f"under {c['Watch']:.0f}", "Watch": f"{c['Watch']:.0f}-{c['Moderate']:.0f}", "Moderate": f"{c['Moderate']:.0f}-{c['Safe']:.0f}", "Safe": f"{c['Safe']:.0f}+"}
    labels = "".join(f'<span style="{"color:#13212B;font-weight:700" if n == a.tier else ""}"><i style="background:{col}"></i>{n} {rng[n]}</span>' for n, w, col in seg)
    ko_on = [k for k in ("K1", "K2", "K3", "K4") if a.gs["ko"][k][0, -1]]
    from .scorecard import KO_TEXT
    ko_html = (f'<p class="note" style="margin:8px 0 0;display:flex;gap:6px;align-items:center">{ICON_OK}No red flags: all knock-out checks are clear</p>' if not ko_on else
               "".join(f'<p class="note" style="margin:8px 0 0;display:flex;gap:6px;align-items:center;color:#9A2A27">{ICON_WARN}{esc(KO_TEXT[k])}</p>' for k in ko_on))
    g = a.grit; gs_ = float(g["score"][0, -1]) if not np.isnan(g["score"][0, -1]) else None
    p = a.peers
    pr = a.pricing.get("short", {})
    peer_rate = a.pricing.get("peer_median", {}).get("rate")
    rate_txt = (f'{pr["rate"]:.2f}%' if pr and a.tier != "Risky" else "Not offered")
    save = (peer_rate - pr["rate"]) / 100 * 1e6 if pr and peer_rate else None
    q = a.info.get("report", {}).get("quality_score", "-")
    L = a.limits
    def row(prod, label, sub, rate_key):
        x = L.get(prod, {})
        if prod == "long":
            st = '<span class="pill grey">Partner referral</span>' if x.get("referral") else '<span class="pill grey">Not offered</span>'
            return f'<tr><td><b>{label}</b><div class="note">{sub}</div></td><td>{st}</td><td class="muted">Lender appraises</td><td class="muted">-</td></tr>'
        ok = x.get("available")
        st = '<span class="pill safe">Available</span>' if ok else '<span class="pill grey">Not available</span>'
        r = a.pricing.get(rate_key, {}).get("rate")
        return (f'<tr><td><b>{label}</b><div class="note">{sub}</div></td><td>{st}</td><td>{money(x.get("limit")) if ok else "-"}</td>'
                f'<td>{f"{r:.2f}%" if ok and r else "-"}</td></tr>')
    qual = (row("short", "Short term", "7-180 days, repaid from collections", "short") + row("shock", "Shock-absorber line", "Standby, opens in a cash dip", "short") +
            row("medium", "Medium term", "1-3 years, secured", "medium") + row("invoice", "Invoice financing", "Against young, not-overdue invoices", "invoice") +
            row("long", "Long term", "3 years and more", None))
    drv = a.drivers
    d_html = "".join(f'<div class="drv">{ICON_UP}<div><b>{esc(r["name"])}</b> is a strength: {esc(val(r["id"], r["value"]))} (peers {esc(val(r["id"], a.peers["measures"][r["id"]]["median"]))})</div></div>' for r in drv["positive"])
    d_html += "".join(f'<div class="drv">{ICON_DN}<div><b>{esc(r["name"])}</b> holds you back: {esc(val(r["id"], r["value"]))} (peers {esc(val(r["id"], a.peers["measures"][r["id"]]["median"]))})</div></div>' for r in drv["negative"])
    nxt = {"Risky": "Watch", "Watch": "Moderate", "Moderate": "Safe"}.get(a.tier)
    sub = f"You are in the {a.tier} tier." + (f" {c[nxt] - sc:.0f} more points would move you to {nxt}." if nxt else " That is the top tier.")
    return wrap(f'''<div style="display:flex;flex-direction:column;gap:18px">
<div><h1>Your credit standing</h1><p class="sub">{sub} Indicative only: a partner lender decides and shows its Key Fact Statement before you accept.</p></div>
<div class="card grid2" style="gap:24px"><div><h2>Genius Score</h2>
<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap"><span style="font-size:60px;font-weight:700;letter-spacing:-0.03em;line-height:1">{sc:.0f}</span><span class="muted" style="font-size:18px">/ 100</span>{tier_pill(a.tier, True)}</div>
<div style="margin-top:22px"><div class="scale">{scale}</div><div style="position:relative;height:0"><div style="position:absolute;left:{sc}%;top:-16px;width:4px;height:22px;margin-left:-2px;background:#13212B;border-radius:2px"></div></div>
<div class="legend" style="margin-top:12px">{labels}</div></div>
<p style="margin:16px 0 0;font-size:14px">About <b>{100 * a.p_stress:.0f} in 100</b> businesses with a similar score hit a cash squeeze within 90 days.</p>{ko_html}</div>
<div class="standing-stats">
<div class="soft"><h2 style="margin-bottom:4px">Grit Score</h2><div class="kpi">{f"{gs_:.0f}" if gs_ is not None else "—"}</div>
<div class="note">{esc(C.grit_band(gs_)) if gs_ is not None else "Not enough history"} · resilience / 100</div><div class="note">{int(g["n"][0, -1])} assessed setbacks</div></div>
<div class="soft"><h2 style="margin-bottom:4px">Among {p["n"]} peers</h2><div class="kpi">Top {max(1, round(100 * (1 - p["score_pct"]))):.0f}%</div><div class="note">Rank {p["rank"]} of {p["n"]}</div></div>
<div class="soft"><h2 style="margin-bottom:4px">Indicative rate</h2><div class="kpi">{rate_txt}</div><div class="note">{f"Peer median {peer_rate:.2f}%" if peer_rate else ""}{f"<br>{money(save)} less/year on ₹10 lakh" if save and save > 0 and a.tier != "Risky" else ""}</div></div>
<div class="soft"><h2 style="margin-bottom:4px">Data confidence</h2><div class="kpi">{q}</div><div class="note">{len(a.months)} months of books</div></div></div></div>
<div class="card"><h2>Two scores, two questions</h2><p style="margin:0;font-size:14px"><b>Genius:</b> can the business repay from its current finances? <b>Grit:</b> how does it handle a setback?</p><p class="note">Grit combines demonstrated recovery with resilience capacity; its weights remain demo assumptions. {"No completed setback yet: this Grit score is a capacity estimate, not proven recovery." if g["n"][0, -1] < 1 else "More assessed setbacks give demonstrated recovery more weight."} The scores are kept separate. Only completed recovery evidence can qualify for a capped, assumed pricing concession; cash buffers and debt cover are not rewarded twice. Loan amounts still have to pass the financial checks.</p></div>
<div class="grid2"><div class="card"><h2>Find borrowing that fits</h2><p>Open <b>Borrow</b>, enter the amount you need and choose a repayment period. Genius calculates a conditional quote using your cash flows and the lender policy.</p><p class="note">You will see the supported amount, interest, fees and repayment plan for that request.</p>
{"".join(f'<p class="note" style="margin:8px 0 0">{esc(owner_reason(r_))}</p>' for r_ in L.get("reasons", []))}</div>
<div class="card"><h2>What drives your score</h2>{d_html}</div></div>
<p class="note" style="margin:0">Peer figures come from a synthetic database of 5,000 firms in this prototype. Genius Score version 5 · factors learned on synthetic defaults · lender policy version {L.get("policy_version", "-")}.</p></div>''')


# ------------------------------------------------------------------------------------------------ peers
def peers(a, db=None):
    p = a.peers
    downturn = ""
    if db is not None and db.regime_p[a.t_db] > 0.5:
        downturn = " The synthetic market is in a downturn at that month, so peer figures are weaker than usual."
    if a.score is None:
        return empty("No score published, so there is no peer position to show.")
    mix = p["tier_mix"]
    cols = {"Safe": "#9CD2B5", "Moderate": "#9FBDE3", "Watch": "#F2D49E", "Risky": "#E9B8B6"}
    bar = "".join(f'<div style="width:{100 * mix[t]:.1f}%;background:{cols[t]};{"box-shadow:inset 0 0 0 3px #13212B" if t == a.tier else ""}"></div>' for t in C.TIERS)
    lab = "".join(f'<span style="{"color:#13212B;font-weight:700" if t == a.tier else ""}"><i style="background:{cols[t]}"></i>{t} {100 * mix[t]:.0f}%{" · you" if t == a.tier else ""}</span>' for t in C.TIERS)
    rows = ""
    for ar in C.AREA_LIST:
        s = float(a.gs["areas"][ar][0, -1]); m = p["areas_median"][ar]; pc = p["areas_pct"][ar]
        col = "#17633F" if (not np.isnan(s) and s >= m) else "#B26A0A"
        rows += (f'<tr><td><b>{ar}</b></td><td>{"-" if np.isnan(s) else f"{s:.0f}"}</td><td class="muted">{m:.0f}</td><td>{"-" if np.isnan(pc) else f"{100 * pc:.0f}%"}</td>'
                 f'<td style="width:34%"><div class="pos"><div class="med" style="left:{m:.0f}%"></div>{"" if np.isnan(s) else f"<div class=you style=left:{s:.0f}%;background:{col}></div>"}</div></td></tr>')
    ms = [(m_, v) for m_, v in p["measures"].items() if not np.isnan(v["pct"]) and not m_.startswith("DQ") and m_ != "IC1"]
    worst = sorted(ms, key=lambda x: x[1]["pct"])[:4]; best = sorted(ms, key=lambda x: -x[1]["pct"])[:4]
    def gl(lst):
        return "".join(f'<div class="drv" style="justify-content:space-between"><div><b>{esc(C.M[m_]["name"])}</b><div class="note">You {esc(val(m_, a.m(m_)))} · peers {esc(val(m_, v["median"]))} · better than {100 * v["pct"]:.0f}%</div></div></div>' for m_, v in lst)
    g = a.grit; gs_ = float(g["score"][0, -1]) if not np.isnan(g["score"][0, -1]) else None
    rate = a.pricing.get("short", {}).get("rate"); pr = a.pricing.get("peer_median", {}).get("rate")
    return wrap(f'''<div style="display:flex;flex-direction:column;gap:18px"><div><h1>How you compare</h1><p class="sub">Against {p["n"]} {esc(p["key"].replace(" | ", ", ").lower())} businesses
({esc(p["level"])}), grouped by the government MSME definition, at {pd.Period(p["month"]).strftime("%b %Y")}.{downturn}</p></div>
<div class="grid2"><div class="card"><h2>Your position</h2><div style="display:flex;align-items:baseline;gap:12px"><span class="kpi" style="font-size:40px">Top {max(1, round(100 * (1 - p["score_pct"]))):.0f}%</span><span class="muted">rank {p["rank"]} of {p["n"]}</span></div>
<div class="grid" style="grid-template-columns:repeat(2,minmax(0,1fr));margin-top:16px"><div class="soft"><div class="note">Your Genius Score</div><b style="font-size:22px">{a.score:.0f}</b><div class="note">Peer median {p["score_median"]:.0f}</div></div>
<div class="soft"><div class="note">Cash squeeze within 90 days</div><b style="font-size:22px">{100 * a.p_stress:.0f} in 100</b><div class="note">Peer group {100 * p["stress_median"]:.0f} in 100</div></div></div></div>
<div class="card"><h2>Where the group sits</h2><div style="display:flex;height:30px;border-radius:8px;overflow:hidden;gap:2px">{bar}</div><div class="legend">{lab}</div>
{"" if not rate or not pr or a.tier == "Risky" else f'<div class="row" style="margin-top:14px"><span>Your short-term rate</span><b>{rate:.2f}%</b></div><div class="row"><span>Peer-median business</span><b>{pr:.2f}%</b></div>'}</div></div>
<div class="card"><h2>Area by area</h2><table class="t"><tr><th>Area</th><th>You</th><th>Peer median</th><th>Better than</th><th>Position (0-100)</th></tr>{rows}</table></div>
<div class="grid"><div class="card"><h2>Furthest behind peers</h2>{gl(worst)}</div><div class="card"><h2>Furthest ahead of peers</h2>{gl(best)}</div>
<div class="card"><h2>Grit</h2><div style="display:flex;align-items:center;gap:10px"><span class="kpi">{f"{gs_:.0f}" if gs_ is not None else "-"}</span>{grit_pill(gs_) if gs_ is not None else ""}</div>
<p class="note">Grit stays separate from Genius. Only completed recovery evidence can qualify for the assumed demo pricing concession; financial capacity is not rewarded twice.</p></div></div>
<div class="ok">Group of {p["n"]} (minimum 30; {esc(p["level"])}). Your business is never part of its own group. In this prototype the peers are synthetic; in production they would be
anonymised, consenting Zoho Books customers.</div></div>''')


# ------------------------------------------------------------------------------------------------ borrow
def borrow(a, pol):
    if a is None or a.score is None:
        return empty("Assess your books first, then tell us how much you need.")
    return wrap(f'''<div class="card"><h1>How much do you need?</h1>
<p class="sub">Enter an amount and choose how you want to repay it. Genius checks your books, existing debt and stressed cash flows, then calculates a quote for your request.</p>
<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin-top:18px">
<div class="soft"><h2>Short term</h2><p class="note">7–180 days · working capital repaid from collections</p></div>
<div class="soft"><h2>Invoice financing</h2><p class="note">7–90 days · supported by eligible receivables</p></div>
<div class="soft"><h2>Medium term</h2><p class="note">6–60 months · monthly instalments, collateral required</p></div>
<div class="soft"><h2>Shock-absorber line</h2><p class="note">7–90 days · standby funding with an annual reservation fee</p></div></div>
<p class="note" style="margin-top:14px">Your standing: {esc(a.tier)}. Quotes are conditional illustrations using synthetic risk calibration; a lender must verify the books, funding purpose and any security.</p></div>''')


def quote_prompt():
    return empty("Enter your amount and duration, then select Calculate my quote.")


def quote_result(q):
    if q["status"] in ("invalid", "review"):
        title = "Check your request" if q["status"] == "invalid" else "This request needs review"
        return wrap(f'<div class="card"><h2>{title}</h2><p>{"<br>".join(esc(owner_reason(r)) for r in q["reasons"])}</p><p class="note">No borrowing offer has been issued. Try a different product or amount, or ask the lender to review the file.</p></div>')
    reduced = q["status"] == "reduced"
    tag = '<span class="pill watch">Smaller alternative</span>' if reduced else '<span class="pill safe">Fits the current checks</span>'
    title = "A smaller amount fits" if reduced else "Your borrowing illustration"
    due = f'{q["months"]} monthly payments' if q["months"] else f'repaid on day {q["days"]}'
    eac = q.get("annual_effective_cost")
    emi = f'<div class="row"><span>Monthly principal + interest</span><b>{money(q["emi"], False)}</b></div>' if q.get("emi") else ''
    cover = f'<div class="row"><span>Stressed debt-service cover after borrowing</span><b>{q["cover_after"]:.2f}×</b></div>' if q.get("cover_after") else ''
    renewal = '<p class="note">Annual facility renewals shown in the schedule are additional to the monthly principal-and-interest instalment.</p>' if q["months"] and 'annual' in q['fee_note'].lower() else ''
    rate_rows = ''.join(f'<div class="row"><span>{label}</span><b>{q["rate_build"][key]:.2f}%</b></div>' for label, key in (
        ("Funding cost", "cost_of_funds"), ("Operating cost", "operating_cost"), ("Capital charge", "capital_charge"),
        ("Expected loss", "expected_loss"), ("Margin", "margin"), ("Term premium", "term_premium")))
    discount = q['rate_build'].get('recovery_concession', 0)
    rate_rows += f'<div class="row"><span>Recovery concession (demo assumption)</span><b>−{discount:.3f}%</b></div>'
    recovery_note = ('Completed recovery history earns this assumed margin concession.' if discount else
                     'No recovery concession applies to this request.')
    return wrap(f'''<div style="display:flex;flex-direction:column;gap:16px"><div class="card">
<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap"><h1>{title}</h1>{tag}</div>
<p class="sub">{esc(q["product"])} · {due}</p>
<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin-top:18px">
<div class="soft"><h2>You requested</h2><div class="kpi">{money(q["requested"])}</div></div>
<div class="soft"><h2>{"Alternative amount" if reduced else "Amount assessed"}</h2><div class="kpi">{money(q["supported"])}</div></div>
<div class="soft"><h2>Annual interest rate</h2><div class="kpi">{q["rate"]:.2f}%</div></div></div>
<div class="row"><span>Net cash received after upfront fees</span><b>{money(q["net_disbursed"], False)}</b></div>
{emi}<div class="row"><span>Total interest over this borrowing</span><b>{money(q["interest"], False)}</b></div>
<div class="row"><span>All facility / processing / reservation fees</span><b>{money(q["fees"], False)}</b></div>
<div class="row"><span>Interest + fees</span><b>{money(q["total_cost"], False)}</b></div>
<div class="row"><span>Principal + interest + all fees</span><b>{money(q["total_repayment"], False)}</b></div>
<div class="row"><span>Effective annual cost, from cash flows</span><b>{f"{eac:.2f}%" if eac is not None else "Not available"}</b></div>{cover}
<p class="note">{esc(q["fee_note"])} Fees are shown separately from interest. The effective annual cost is an estimate, not a regulatory KFS APR.</p>{renewal}</div>
<div class="grid2"><div class="card"><h2>Why this amount?</h2>{''.join(f'<p class="note">{esc(owner_reason(r))}</p>' for r in q['reasons'])}</div>
<div class="card"><h2>How the annual rate is built</h2>{rate_rows}<div class="row"><span>Total</span><b>{q["rate"]:.2f}%</b></div>
<p class="note">The calibrated 12-month default probability is {q['rate_build']['pd']:.2f}%; the assumed loss after default is {q['rate_build']['lgd']:.0f}%. These are synthetic-data assumptions. {recovery_note} Genius Score and calibrated default probability do not change. This policy has not been validated on real loans.</p></div></div>
<div class="banner">{esc(q["conditions"][0])} {esc(q["conditions"][1])}</div>
<p class="note">Policy version {q['policy_version']}. {esc(q['conditions'][2])}</p></div>''')


def calc(a, pol, amount, days, product, months=36, secured=False, collateral_value=None):
    if amount is None or amount == 0:
        return quote_prompt()
    from .lending import quote
    return quote_result(quote(a, pol, amount, days, product, months, secured, collateral_value))


def learned_model_summary(db):
    model = getattr(db, "learned_model", None)
    if model is None:
        return empty("No learned model has been trained.")
    r = model.report
    rows = ''.join(f'<tr><td>{esc(w["factor"])}</td><td>{esc(w["area"])}</td><td>{w["coefficient"]:.4f}</td><td>{w["relative_weight_pct"]:.2f}%</td></tr>' for w in model.weights() if w['coefficient'] > 1e-6)
    fv, cv = r['financial_validation'], r['combined_validation']
    ft, ct = r['financial_test'], r['combined_test']
    return wrap(f'''<div class="card"><h1>How Genius learned its factors</h1>
<p class="sub">{esc(r['data_source'])}. {r['train_observations']:,} training firm-month observations. The probability model learns nonnegative risk coefficients; data quality gates publication.</p>
<p><b>Genius and Grit remain separate.</b> Genius uses financial factors only. The combined model below is an exploratory comparison, not the deployed score. A separate demo policy offers up to 50 basis points of margin concession for recent, completed recovery evidence, scaled by evidence strength. This assumed policy is not validated by the AUC/Brier results below; it does not change PD or add a limit bonus.</p>
<table class="t"><tr><th>Model</th><th>Validation AUC</th><th>Validation Brier</th><th>Final test AUC</th><th>Final test Brier</th></tr>
<tr><td>Financial only</td><td>{fv['auc']:.4f}</td><td>{fv['brier']:.4f}</td><td>{ft['auc']:.4f}</td><td>{ft['brier']:.4f}</td></tr>
<tr><td>Financial + recovery</td><td>{cv['auc']:.4f}</td><td>{cv['brier']:.4f}</td><td>{ct['auc']:.4f}</td><td>{ct['brier']:.4f}</td></tr></table>
<p class="note">Training features end at month 47 so their 12-month outcomes are known by month 59. Validation features end at 83 (labels known by 95); final test features are 96–107. Final test results do not select the model. Repeated firm-months are not independent borrowers.</p>
<p class="note">The 0–100 display uses the model's log-odds on a fixed training scale, followed by financial caps. Default and cash-stress probabilities are calibrated separately. Measure rating thresholds remain explicit assumptions.</p>
<table class="t"><tr><th>Factor</th><th>Area</th><th>Risk coefficient</th><th>Relative coefficient weight</th></tr>{rows}</table>
<p class="note">Relative weights normalise coefficient magnitudes, not causal importance or percentage changes in default probability. Financial indicators use their existing 0–100 ratings. Recovery features are evidence-weighted and neutral without completed episodes.</p></div>''')


def rate_chart(a, pol):
    import plotly.graph_objects as go
    if a is None or a.tier is None:
        return go.Figure()
    xs = [1, 3, 6, 12, 18, 24, 36]
    pd_ = 100 * a.p_default
    gs_ = float(a.grit["score"][0, -1]) if not np.isnan(a.grit["score"][0, -1]) else None
    mine = [pol.rate(a.tier, a.info.get("policy_context"), pd_score=pd_, tenor_months=m, secured=m > 12, grit=None, recovery=a.info.get("recovery_policy"))["rate"] for m in xs]
    peer = [pol.rate("Moderate", a.info.get("policy_context"), tenor_months=m, secured=m > 12)["rate"] for m in xs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=peer, name="Moderate-tier business", line=dict(color="#9AA6AE", dash="dash", width=2.5), mode="lines"))
    fig.add_trace(go.Scatter(x=xs, y=mine, name="You", line=dict(color=ACC, width=3.5), mode="lines+markers"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white", font=dict(family="Inter, sans-serif", size=12),
                      title=dict(text="Your rate by loan length (months)", font=dict(size=13)), yaxis_title="% a year", legend=dict(orientation="h", y=-0.25))
    return fig


def ew_chart(ew, threshold=4.0):
    import plotly.graph_objects as go
    fig = go.Figure()
    if not ew:
        return fig
    t = ew["table"]
    fig.add_trace(go.Scatter(x=list(range(1, len(t) + 1)), y=t["cover"], line=dict(color=ACC, width=3.5), name="Cash cover (weeks of fixed costs)"))
    fig.add_hline(y=threshold, line=dict(color="#9A2A27", dash="dash"), annotation_text=f"{threshold:g}-week line")
    fig.add_vrect(x0=0.5, x1=ew["alert"]["alert_weeks"] + 0.5, fillcolor="#F1F5F6", line_width=0, layer="below")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white", font=dict(family="Inter, sans-serif", size=12),
                      title=dict(text="Early warning: next 13 weeks", font=dict(size=13)), xaxis_title="week", showlegend=False)
    return fig


def ew_text(ew):
    if not ew:
        return empty("The 13-week early warning needs invoice-level data.")
    al = ew["alert"]
    pill = '<span class="pill risk">Alert</span>' if al["fired"] else '<span class="pill safe">No alert</span>'
    top = ew["overdue_customers"].head(3)
    top = ew["overdue_customers"].groupby("party")["open"].sum().sort_values(ascending=False).head(3)
    chase = "; ".join(f"{k} {money(v)}" for k, v in top.items()) if len(top) else ""
    return wrap(f'''<div class="card"><div style="display:flex;justify-content:space-between;align-items:center"><h2 style="margin:0">Shock-absorber trigger</h2>{pill}</div>
<p style="font-size:14px;margin:10px 0 0">{esc(al["message"])}</p>{f'<p class="note">Chasing the largest overdue invoices lifts the low point: {esc(chase)}.</p>' if chase else ""}
<p class="note">A forecast below 4 weeks within 6 weeks flags a possible standby-funding need. This prototype simulates the trigger; it does not open a credit line or disburse funds.</p></div>''')


# ------------------------------------------------------------------------------------------------ improve
def improve(a, res, chosen):
    if a is None or a.score is None:
        return empty()
    base_rate = a.pricing.get("short", {}).get("rate")
    if not chosen or res is None:
        body = '<p class="muted" style="margin:0">Tick one or more changes to see your new score, tier and rate. Each set of changes is re-scored through the same engine, applied to your books.</p>'
    else:
        r = res
        body = (f'<div style="display:flex;align-items:center;gap:14px"><span style="font-size:22px;color:#5E6B75;font-weight:600">{a.score:.0f}</span><span>→</span>'
                f'<span style="font-size:48px;font-weight:700;line-height:1">{r["score"]:.0f}</span>{tier_pill(r["tier"], True)}</div>'
                f'<div class="soft" style="padding:4px 16px;margin-top:12px"><div class="row" style="border-top:0"><span>New short-term rate</span><b>{"not offered" if r["tier"] == "Risky" else format(r["rate"], ".2f") + "%"}</b></div>'
                f'<div class="row"><span>Saving per ₹10 lakh a year</span><b class="up">{money(max(0, (base_rate - r["rate"]) / 100 * 1e6)) if base_rate and r["tier"] != "Risky" else "-"}</b></div></div>'
                '<p class="note" style="margin:8px 0 0">Your tier updates at the next quarterly review, once the change shows in your books.</p>')
    return wrap(f'<div class="card" style="background:#F3F8F8;border-color:#CFE0DF"><h2>Result</h2>{body}</div>')


def early_pay(rows, rate):
    if not rows:
        return empty("No supplier bills to test.")
    tr = "".join(f'<tr><td><b>{esc(r["supplier"])}</b>{"<div class=note>Micro or small supplier</div>" if r.get("msme") else ""}</td><td>{money(r["amount"], False)}</td>'
                 f'<td>{r["discount_pct"]:g}% for {r["days_early"]:g} days early</td><td>{money(r["discount"], False)}</td><td>{money(r["interest"] + r["fee"], False)}</td>'
                 f'<td class="{"up" if r["net"] > 0 else "dn"}">{"+" if r["net"] > 0 else ""}{money(r["net"], False)}</td><td>{TAKE if r["take"] else SKIP}</td></tr>' for r in rows)
    net = sum(r["net"] for r in rows if r["take"])
    return wrap(f'''<div class="card"><div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap"><h2 style="margin:0">Early-payment discounts</h2>
<span>Net gain if financed: <b class="up">{money(net, False)}</b></span></div><p class="note">Discount earned against the cost of borrowing at your {rate:.2f}% to pay early (fee charged once a year
on the limit, so each drawdown costs interest only).</p><table class="t"><tr><th>Supplier</th><th>Bill</th><th>Offer</th><th>Discount</th><th>Cost</th><th>Net</th><th></th></tr>{tr}</table></div>''')


# ------------------------------------------------------------------------------------------------ details
def details_area(a, area):
    if a is None or a.score is None:
        return empty()
    rows = ""
    for m in C.BY_AREA[area]:
        v, r = a.m(m), a.r(m)
        pm = a.peers["measures"][m]["median"]
        rows += f'<tr><td><b>{esc(C.M[m]["name"])}</b><div class="note">{esc(C.M[m]["desc"])}</div></td><td>{esc(val(m, v))}</td><td class="muted">{esc(val(m, pm))}</td><td>{rating_pill(r)}</td></tr>'
    s = float(a.gs["areas"][area][0, -1])
    pts = "".join(f'<tr><td>{ar} <span class="muted">{w}%</span></td><td>{rating_pill(float(a.gs["areas"][ar][0, -1]))}</td><td class="muted">{0 if np.isnan(a.gs["areas"][ar][0, -1]) else w * float(a.gs["areas"][ar][0, -1]) / 100:.1f}</td></tr>' for ar, w in C.AREAS.items())
    return wrap(f'''<div class="grid2" style="grid-template-columns:minmax(min(100%,260px),1fr) minmax(0,2.2fr)"><div class="card"><h2>Area · weight · points</h2><table class="t">{pts}
<tr><td><b>Genius Score</b></td><td></td><td><b>{a.score:.1f}</b></td></tr></table><p class="note">Knock-out checks: {"none apply" if not any(a.gs["ko"][k][0, -1] for k in ("K1", "K2", "K3", "K4")) else ", ".join(k for k in ("K1", "K2", "K3", "K4") if a.gs["ko"][k][0, -1])}</p></div>
<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap"><b style="font-size:19px">{area}</b><span>{rating_pill(s)} <span class="note">peer median {a.peers["areas_median"][area]:.0f} · {C.AREAS[area]}% of your score</span></span></div>
<table class="t" style="margin-top:8px"><tr><th>Measure</th><th>You</th><th>Peer median</th><th>Rating</th></tr>{rows}</table></div></div>''')


def measure_card(a, mid, books=None):
    if a is None or a.score is None:
        return empty()
    from .ask import FORMULA, IMPROVE
    m = C.M[mid]; v, r = a.m(mid), a.r(mid)
    b = C.bounds_for(mid, a.family)
    p = a.peers["measures"][mid]
    scale = "" if b is None else (f'<div class="note" style="margin-top:6px">Rated 20 / 40 / 60 / 80 at {" / ".join(f"{x:g}" for x in b)} ({"higher" if m["dir"] == "H" else "lower"} is better)'
                                  f'{" · published bank norm" if m["bank_norm"] else ""}</div>')
    hist = a.meas[mid][0]
    pts = [(a.months[j].strftime("%b %y"), hist[j]) for j in range(len(hist) - 1, max(-1, len(hist) - 13), -3) if not np.isnan(hist[j])][::-1]
    trend = " · ".join(f"{t}: {val(mid, float(x))}" for t, x in pts)
    extra = ""
    if books is not None and mid in ("RC05", "RC08", "RC01", "RC03", "RC16"):
        ag = books.ageing("receivable")
        od = ag[ag["days_past_due"] > 0].groupby("party").agg(overdue=("open", "sum"), invoices=("open", "size"), oldest=("days_past_due", "max")).sort_values("overdue", ascending=False).head(6)
        extra = "<h2 style='margin-top:14px'>Where the overdue money sits</h2><table class='t'><tr><th>Customer</th><th>Overdue</th><th>Invoices</th><th>Oldest</th></tr>" + \
                "".join(f"<tr><td><b>{esc(i)}</b></td><td>{money(rw.overdue)}</td><td>{int(rw.invoices)}</td><td>{rw.oldest:.0f} days late</td></tr>" for i, rw in od.iterrows()) + "</table>"
    return wrap(f'''<div class="card"><h2>Measure · {esc(m["area"])}</h2><b style="font-size:19px">{esc(m["name"])}</b>
<div class="grid" style="margin-top:12px"><div class="soft"><div class="note">How it is worked out</div><div style="font-size:14px">{esc(FORMULA.get(mid, m["desc"]))}</div>
<div style="font-size:18px;font-weight:700;margin-top:6px">{esc(val(mid, v))}</div></div>
<div class="soft"><div class="note">Rating</div><div style="display:flex;gap:8px;align-items:center"><span class="kpi" style="font-size:28px">{"-" if r is None else f"{r:.0f}"}</span><span class="note">of 100</span></div>{scale}</div>
<div class="soft"><div class="note">Peers ({p["n"]})</div><div style="font-size:14px">median {esc(val(mid, p["median"]))}<br>middle half {esc(val(mid, p["p25"]))} to {esc(val(mid, p["p75"]))}</div>
<div class="note">{"" if np.isnan(p["pct"]) else f"Better than {100 * p['pct']:.0f}% of peers"}</div></div></div>
<p class="note" style="margin-top:10px">Last year: {esc(trend) or "not enough history"}</p>{extra}
<div class="ok" style="margin-top:10px">{esc(IMPROVE.get(mid, ""))}</div></div>''')


def resilience_lab():
    from .resilience_demo import reference_quotes
    rows = []
    for r in reference_quotes():
        q = r['quote']
        rows.append(f'<tr><td>{esc(r["profile"])}</td><td>{r["genius"]:.0f}</td><td>{r["grit"]:.0f}</td><td>{r["concession_bps"]:.1f} bps</td><td>{q["rate"]:.2f}%</td><td>{money(q["interest"],False)}</td></tr>')
    return wrap('<div class="card"><h2>Synthetic resilience example</h2>'
        '<p>Same current finances, same Genius Score, same ₹5 lakh request for 60 days. Two constructed operating histories face identical two-month sales shocks of 45%; recovery speed, cost flexibility, collections response and extra borrowing differ.</p>'
        '<div style="overflow-x:auto"><table class="t"><tr><th>History</th><th>Genius</th><th>Grit</th><th>Concession</th><th>Rate</th><th>Interest</th></tr>' + ''.join(rows) + '</table></div>'
        '<p class="note">Both requests support ₹5 lakh under the same financial checks. Fees are ₹7,500 in each case. Reference policy: 1% PD, 60% LGD; rate before concession 14.95%. These are constructed scenarios, not the uploaded company’s history or empirical evidence that Grit predicts defaults.</p>'
        '<p class="note">Assumed concession = up to 50 bps × recovery strength above 50/100 × n/(n+2) × recency decay (24-month half-life), capped at available lender margin. Recovery strength averages gap regained, sustained recovery, operating funding and management response. No completed evidence means zero concession; capacity indicators are excluded. Real use needs lender agreement and validation on real outcomes.</p></div>')
