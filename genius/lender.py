"""Lender screens: the applications list (filters, saved views, export), one firm in detail, policy preview and the Validation room.
Everything is computed from the database and the lender's current policy, so changing a setting changes these tables."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from . import second_opinion as SO
from . import validation as V
from .ui import esc, money, tier_pill, grit_pill, wrap, ICON_OK, ICON_WARN

YES = "<span class='pill safe'>Yes</span>"
NO = "<span class='pill grey'>No</span>"
CHOSEN = " <span class='pill safe'>chosen</span>"
CONF_CLS = {"High": "safe", "Medium": "mod", "Low": "watch", "Scorecard only": "grey"}
QWORD = {1: "weakest fifth", 2: "second fifth", 3: "middle fifth", 4: "fourth fifth", 5: "strongest fifth"}


def conf_pill(c):
    if not c:
        return '<span class="pill grey">n/a</span>'
    return f'<span class="pill {CONF_CLS.get(c, "grey")}">{esc(c if c == "Scorecard only" else c + " confidence")}</span>'


def feature_label(n: str) -> str:
    if n.startswith("p_") and n[2:] in C.M:
        return f"{C.M[n[2:]]['name']} (rank in peer group)"
    if n.startswith("d3_") and n[3:] in C.M:
        return f"{C.M[n[3:]]['name']}: change in peer rank over 3 months"
    if n.startswith("d12_") and n[4:] in C.M:
        return f"{C.M[n[4:]]['name']}: change in peer rank over 12 months"
    if n.startswith("miss_") and n[5:] in C.M:
        return f"{C.M[n[5:]]['name']} missing"
    if n.startswith("sector_"):
        return "Sector: " + n[7:].replace("_", " ")
    if n.startswith("size_"):
        return "Size: " + n[5:]
    fixed = {"grit": "Grit", "grit_demonstrated": "Grit: demonstrated part", "grit_capacity": "Grit: capacity part", "grit_setbacks": "Number of setbacks judged",
             "setback_now": "In a setback now", "months_since_setback": "Months since last setback", "vol_sales_6": "Sales volatility (6 months)",
             "vol_sales_12": "Sales volatility (12 months)", "vol_ocf_6": "Cash-flow volatility (6 months)", "vol_ocf_12": "Cash-flow volatility (12 months)",
             "vol_coll_6": "Collections volatility (6 months)", "vol_coll_12": "Collections volatility (12 months)", "vol_gm_6": "Gross-margin volatility (6 months)",
             "vol_gm_12": "Gross-margin volatility (12 months)", "history": "Months of history", "regime_stress": "Market stress (regime model)",
             "peer_trend": "Peer-group sales trend", "peer_trend_3": "Peer-group trend, 3-month change", "x_regime_cash": "Market stress x cash runway",
             "x_regime_debt": "Market stress x debt cover", "x_regime_overdue": "Market stress x overdue receivables",
             "x_trading_collections": "Trading firm x collection speed", "x_small_topcust": "Micro firm x top-customer share",
             "x_setback_capacity": "In setback x capacity", "data_confidence": "Data confidence", "missing_share": "Share of measures missing"}
    return fixed.get(n, n)


# ------------------------------------------------------------------------------------------------ portfolio
def portfolio(db, pol, t: int | None = None) -> pd.DataFrame:
    """Every live firm at month t with both rankings, confidence, limit and rate under the current policy."""
    t = db.T - 1 if t is None else t
    sc = db.gs["score"][:, t].astype(float)
    ok = db.P.alive[:, t] & ~np.isnan(sc)
    idx = np.where(ok)[0]
    tiers = db.cal.tier(sc)
    sig = SO.significant_groups(db.so["group_test"], float(pol.v("so_confidence")), bool(pol.v("so_bh")))
    qs, qm = db.so["q_score"][:, t:t + 1].astype(float), db.so["q_ml"][:, t:t + 1].astype(float)
    conf = SO.confidence(qs, qm, db.keys[:, t:t + 1], sig)[:, 0]
    g = db.grit["score"][:, t].astype(float)
    regime = "stress" if db.regime_p[t] > 0.5 else "calm"
    rows = []
    for i in idx:
        tier = tiers[i]
        ko = {k: bool(v[i, t]) for k, v in db.gs["ko"].items()}
        x = V.firm_facts(db, i, t)
        gg = None if np.isnan(g[i]) else float(g[i])
        ctx = dict(sector=db.P.firms["sector"].iat[i], size=db.band[i, t], regime=regime, date=str(db.months[t].end_time.date()))
        lim = pol.limits(x, tier, None, ko, confidence=conf[i], ctx=ctx, pd_score=100 * float(db.cal.default(sc[i])))
        from .resilience import evidence
        quality = float(np.mean(db.P.firms.iloc[i][['dq_bal','dq_cls','dq_party','dq_due']].astype(float)))
        rec = evidence(db.grit, quality, now=t, firm_index=i)
        r = pol.rate(tier, ctx, pd_score=100 * float(db.cal.default(sc[i])), recovery=rec)
        rows.append(dict(ID=int(i), Firm=db.firm_names[i], Sector=db.P.firms["sector"].iat[i], Size=db.band[i, t], Score=round(sc[i], 1), Tier=tier,
                         Grit=None if gg is None else round(gg), **{"Grit band": C.grit_band(gg) if gg is not None else "-"},
                         **{"Score 5th": "-" if np.isnan(qs[i, 0]) else str(int(qs[i, 0])), "ML 5th": "-" if np.isnan(qm[i, 0]) else str(int(qm[i, 0]))},
                         Confidence=conf[i] or "-", **{"ST limit (₹ lakh)": round(lim["short"]["limit"] / 1e5, 1),
                                                       "MT limit (₹ lakh)": round(lim["medium"]["limit"] / 1e5, 1),
                                                       "Rate %": round(r["rate"], 2) if r["offered"] else None, "Annual sales (₹ lakh)": round(x["annual_sales"] / 1e5, 1)},
                         **{"Peer group": db.keys[i, t]}))
    return pd.DataFrame(rows)


def filter_df(df, tiers=None, confs=None, sector="All", size="All", min_score=0, grit_bands=None, text=""):
    m = pd.Series(True, index=df.index)
    if tiers:
        m &= df["Tier"].isin(tiers)
    if confs:
        m &= df["Confidence"].isin(confs)
    if sector and sector != "All":
        m &= df["Sector"] == sector
    if size and size != "All":
        m &= df["Size"] == size
    if min_score:
        m &= df["Score"] >= float(min_score)
    if grit_bands:
        m &= df["Grit band"].isin(grit_bands)
    if text:
        m &= df["Firm"].str.contains(text, case=False, regex=False)
    return df[m]


def portfolio_summary(df: pd.DataFrame, full: pd.DataFrame) -> str:
    n = len(df)
    lent = df["ST limit (₹ lakh)"].sum() + df["MT limit (₹ lakh)"].sum()
    offered = (df["ST limit (₹ lakh)"] > 0).sum()
    rate = df.loc[df["ST limit (₹ lakh)"] > 0, "Rate %"].mean()
    mix = df["Tier"].value_counts()
    low = (df["Confidence"] == "Low").sum()
    tiles = "".join(f'<div class="soft"><div class="note">{k}</div><b style="font-size:22px">{v}</b></div>' for k, v in (
        ("Firms shown", f"{n:,} of {len(full):,}"), ("Offered short-term credit", f"{offered:,}"), ("Total limits", money(lent * 1e5)),
        ("Average short-term rate", "-" if np.isnan(rate) else f"{rate:.2f}%"), ("Low confidence: review", f"{low:,}")))
    bar = " ".join(f'{tier_pill(t)} {mix.get(t, 0):,}' for t in C.TIERS)
    return wrap(f'<div class="card"><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">{tiles}</div>'
                f'<p class="note" style="margin:10px 0 0">{bar}</p></div>')


# ------------------------------------------------------------------------------------------------ one firm
def firm_detail(db, pol, i: int, notes: list | None = None, t: int | None = None) -> str:
    t = db.T - 1 if t is None else t
    sc = float(db.gs["score"][i, t])
    tier = db.cal.tier(np.array([sc]))[0]
    key = db.keys[i, t]
    sig = SO.significant_groups(db.so["group_test"], float(pol.v("so_confidence")), bool(pol.v("so_bh")))
    qs, qm = db.so["q_score"][i, t], db.so["q_ml"][i, t]
    conf = SO.confidence(np.array([[qs]], float), np.array([[qm]], float), np.array([[key]], object), sig)[0, 0]
    gt = db.so["group_test"]
    gr = gt[gt["group"] == key]
    g = float(db.grit["score"][i, t]) if not np.isnan(db.grit["score"][i, t]) else None
    regime = "stress" if db.regime_p[t] > 0.5 else "calm"
    x = V.firm_facts(db, i, t)
    ko = {k: bool(v[i, t]) for k, v in db.gs["ko"].items()}
    ctx = dict(sector=db.P.firms["sector"].iat[i], size=db.band[i, t], regime=regime, date=str(db.months[t].end_time.date()))
    lim = pol.limits(x, tier, g, ko, confidence=conf, ctx=ctx)
    r = pol.rate(tier, ctx, pd_score=100 * float(db.cal.default(sc)))
    row = np.array([db.X[n][i] for n in db.names], np.float32)
    down = SO.drivers_for(db.so["model"], db.names, row, db.so["fills"], top=4)
    up = SO.drivers_for(db.so["model"], db.names, row, db.so["fills"], top=len(db.names))[::-1][:4]
    drv = "".join(f'<div class="drv"><span class="dn">−</span><div>{esc(feature_label(n))}</div></div>' for n, v in down if v < 0)
    drv += "".join(f'<div class="drv"><span class="up">+</span><div>{esc(feature_label(n))}</div></div>' for n, v in up if v > 0)
    sdr = []
    from .scorecard import drivers as sdrivers
    k0 = db.ratings[C.MEASURES[0]].shape[1] - (db.T - t)
    rat = {m: float(db.ratings[m][i, k0]) for m in C.MEASURES}
    mea = {m: float(db.meas[m][i, t]) for m in C.MEASURES}
    sd = sdrivers(rat, mea)
    sdr = "".join(f'<div class="drv"><span class="up">+</span><div>{esc(d["name"])}</div></div>' for d in sd["positive"]) + \
          "".join(f'<div class="drv"><span class="dn">−</span><div>{esc(d["name"])}</div></div>' for d in sd["negative"])
    hist = ""
    for k in range(7, -1, -1):
        tt = t - 3 * k
        if tt < 0 or np.isnan(db.gs["score"][i, tt]):
            continue
        s_ = float(db.gs["score"][i, tt])
        a_, b_ = db.so["q_score"][i, tt], db.so["q_ml"][i, tt]
        hist += (f'<tr><td>{db.months[tt].strftime("%b %Y")}</td><td>{s_:.0f}</td><td>{tier_pill(db.cal.tier(np.array([s_]))[0])}</td>'
                 f'<td>{"-" if np.isnan(a_) else int(a_)}</td><td>{"-" if np.isnan(b_) else int(b_)}</td>'
                 f'<td>{"-" if np.isnan(db.grit["score"][i, tt]) else format(float(db.grit["score"][i, tt]), ".0f")}</td></tr>')
    prods = "".join(f'<div class="row"><span>{lab}</span><b>{money(lim[p]["limit"]) if lim[p]["available"] else "not offered"}</b></div>'
                    for p, lab in (("short", "Short term"), ("shock", "Shock-absorber line"), ("medium", "Medium term (secured)"), ("invoice", "Invoice financing")))
    note_html = "".join(f'<div class="drv"><div><b>{esc(n["by"])}</b> <span class="note">{esc(n["at"])}</span><div>{esc(n["text"])}</div></div></div>'
                        for n in (notes or [])) or '<p class="note">No reviewer notes yet.</p>'
    gtxt = ("not tested (group too small)" if gr.empty else
            f'IC {gr["ic"].iloc[0]:.2f}, t {gr["t"].iloc[0]:.1f}, one-sided p {gr["p"].iloc[0]:.2g} → {"significant" if sig.get(key) else "not significant"} '
            f'at {100 * float(pol.v("so_confidence")):.0f}%{" with Benjamini-Hochberg" if pol.v("so_bh") else ""}')
    reasons = "".join(f'<p class="note" style="margin:6px 0 0">{esc(x_)}</p>' for x_ in lim.get("reasons", []))
    return wrap(f'''<div style="display:flex;flex-direction:column;gap:16px">
<div class="card" style="display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:center"><div><h1 style="font-size:22px">{esc(db.firm_names[i])}</h1>
<div class="note">{esc(db.P.firms["sector"].iat[i])} · {esc(db.band[i, t])} · peer group {esc(key)} · synthetic firm #{i}</div></div>
<div style="display:flex;gap:10px;align-items:center"><span class="kpi">{sc:.0f}</span>{tier_pill(tier, True)}{conf_pill(conf)}</div></div>
<div class="grid2"><div class="card"><h2>Two rankings inside the peer group</h2>
<div class="grid" style="grid-template-columns:repeat(2,minmax(0,1fr))"><div class="soft"><div class="note">Genius Score (rules)</div><b style="font-size:20px">{"-" if np.isnan(qs) else QWORD[int(qs)]}</b></div>
<div class="soft"><div class="note">Second Opinion ({esc(db.so["winner"])})</div><b style="font-size:20px">{"-" if np.isnan(qm) else QWORD[int(qm)]}</b></div></div>
<p class="note">Peer-group test (years 9-10): {gtxt}.</p><p style="font-size:14px;margin:6px 0 0"><b>Confidence: {esc(conf or "n/a")}.</b>
{ {"High": "Both methods put the firm in the same fifth.", "Medium": "The methods are one fifth apart, or the Second Opinion is more positive.",
   "Low": "The Second Opinion is at least two fifths more worried: limits are held back until a person reviews.",
   "Scorecard only": "The Second Opinion is not reliable in this peer group, so the scorecard decides alone."}.get(conf, "")}</p></div>
<div class="card"><h2>Terms under policy version {pol.version}</h2>{prods}<div class="row"><span>Short-term rate</span><b>{f"{r['rate']:.2f}%" if r["offered"] else "not offered"}</b></div>
<div class="row"><span>Grit</span><span>{"-" if g is None else f"{g:.0f}"} {grit_pill(g) if g is not None else ""}</span></div>{reasons}</div></div>
<div class="grid2"><div class="card"><h2>What moves the Second Opinion</h2>{drv or '<p class="note">No strong drivers.</p>'}</div>
<div class="card"><h2>What drives the Genius Score</h2>{sdr}</div></div>
<div class="grid2"><div class="card"><h2>Last 8 quarters</h2><table class="t"><tr><th>Month</th><th>Score</th><th>Tier</th><th>Score fifth</th><th>Second Opinion fifth</th><th>Grit</th></tr>{hist}</table></div>
<div class="card"><h2>Reviewer notes</h2>{note_html}</div></div></div>''')


# ------------------------------------------------------------------------------------------------ policy preview
def rate_table(pol, ctx=None) -> pd.DataFrame:
    rows = []
    for tier in C.TIERS:
        r = {"Tier": tier}
        for lab, m, sec, inv in (("3 months", 3, False, False), ("Invoice", 3, False, True), ("12 months", 12, False, False), ("36 months, secured", 36, True, False)):
            x = pol.rate(tier, ctx, tenor_months=m, secured=sec, invoice=inv)
            r[lab] = round(x["rate"], 2) if x["offered"] else "not offered"
        rows.append(r)
    return pd.DataFrame(rows)


def rate_build(pol, tier="Moderate") -> str:
    x = pol.rate(tier)
    parts = [("Cost of funds (repo + spread)", x["cost_of_funds"]), ("Operating cost", x["operating_cost"]), ("Capital charge", x["capital_charge"]),
             ("Expected loss (default x loss)", x["expected_loss"]), ("Margin", x["margin"])]
    rows = "".join(f'<div class="row"><span>{k}</span><b>{v:.2f}%</b></div>' for k, v in parts)
    return wrap(f'<div class="card"><h2>How a {tier} 3-month rate is built</h2>{rows}<div class="row"><span><b>Rate</b></span><b style="font-size:18px">{x["rate"]:.2f}%</b></div>'
                f'<p class="note">Default probability {x["pd"]:.2f}% x loss if default {x["lgd"]:.0f}%.</p></div>')


def impact(before: pd.DataFrame, after: pd.DataFrame) -> str:
    def s(df):
        off = df["ST limit (₹ lakh)"] > 0
        return dict(n=int(off.sum()), lim=float(df["ST limit (₹ lakh)"].sum() + df["MT limit (₹ lakh)"].sum()) * 1e5,
                    rate=float(df.loc[off, "Rate %"].mean()) if off.any() else np.nan, low=int((df["Confidence"] == "Low").sum()))
    a, b = s(before), s(after)
    def row(lab, x, y, f):
        d = "" if x == y else (' <span class="up">▲</span>' if y > x else ' <span class="dn">▼</span>')
        return f'<tr><td>{lab}</td><td>{f(x)}</td><td><b>{f(y)}</b>{d}</td></tr>'
    return wrap(f'''<div class="card"><h2>Portfolio impact (all {len(after):,} live synthetic firms)</h2><table class="t"><tr><th></th><th>Saved policy</th><th>Draft</th></tr>
{row("Firms offered short-term credit", a["n"], b["n"], lambda v: f"{v:,}")}{row("Total short and medium limits", a["lim"], b["lim"], money)}
{row("Average short-term rate", a["rate"], b["rate"], lambda v: "-" if np.isnan(v) else f"{v:.2f}%")}{row("Held for review (Low confidence)", a["low"], b["low"], lambda v: f"{v:,}")}</table></div>''')


# ------------------------------------------------------------------------------------------------ validation room
def _check(ok, title, detail):
    return f'<div class="drv">{ICON_OK if ok else ICON_WARN}<div><b>{esc(title)}</b><div class="note">{detail}</div></div></div>'


def val_scorecard(db, pol) -> str:
    v = db.val
    test = v["ic"]["Years 9-10 (test)"]
    L = v["lending"]
    gv = v["grit_truth"]["end of year 10"]
    checks = (_check(v["auc_stress"] >= 0.70, f"Ranks risk: AUC {v['auc_stress']:.2f} for a 90-day cash squeeze (pass mark 0.70)",
                     f"Default within 12 months: AUC {v['auc_default']:.2f}. KS {v['ks_stress']:.2f}. Brier score {v['brier_stress']:.3f}. Test years 9-10, never used to set the cut-offs.") +
              _check(v["tiers_in_order"], "Tiers in order every test year", "Safe has fewer squeezes than Moderate, Moderate fewer than Watch, Watch fewer than Risky.") +
              _check(v["tiers_on_target"], "Tier stress rates inside their target bands (90% interval)", "Targets: Safe under 5%, Moderate 5-12%, Watch 12-25%, Risky over 25%.") +
              _check(test["t"] > 2, f"Information coefficient {test['mean']:.2f} (Newey-West t {test['t']:.1f}, one-sided p {test['p']:.1g})",
                     "Rank correlation each month between the score and how severe the next quarter turned out, averaged over the test months.") +
              _check(L["Genius"]["loss_rate"] < L["Naive rule"]["loss_rate"], f"Loses less than a naive rule: {100 * L['Genius']['loss_rate']:.2f}% vs {100 * L['Naive rule']['loss_rate']:.2f}% of money lent",
                     f"Same number of approvals ({100 * L['Genius']['approval_rate']:.0f}%). Net margin per ₹100 lent over six months: ₹{L['Genius']['net_per_100']:.2f} vs ₹{L['Naive rule']['net_per_100']:.2f}. "
                     "Naive rule: approve the largest firms with positive net worth.") +
              _check(gv["grit"] > 0, f"Grit finds hidden resilience: rank correlation {gv['grit']:.2f} (demonstrated part {gv['demonstrated']:.2f})",
                     "Each synthetic firm has a hidden resilience trait the model never sees. Modest but positive, reported as found."))
    tt = v["tiers"][v["tiers"].period == "Years 9-10 (test)"]
    trows = "".join(f'<tr><td>{tier_pill(r.tier)}</td><td>{r.firm_months:,}</td><td><b>{100 * r.stress_rate:.1f}%</b></td><td class="muted">{100 * r.lo:.1f}-{100 * r.hi:.1f}%</td>'
                    f'<td>{100 * r.default_rate:.2f}%</td><td class="muted">{ {"Safe": "under 5%", "Moderate": "5-12%", "Watch": "12-25%", "Risky": "over 25%"}[r.tier]}</td></tr>' for r in tt.itertuples())
    q = v["quintile_stress"]
    mx = max(q)
    qbars = "".join(f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0"><span style="width:110px" class="note">{["Weakest fifth", "Second", "Middle", "Fourth", "Strongest fifth"][k]}</span>'
                    f'<div class="bar" style="width:{60 * q[k] / mx:.0f}%;background:{"#C0504D" if k == 0 else "#0B5D5A"}"></div><b>{100 * q[k]:.1f}%</b></div>' for k in range(5))
    gw = v["grit_within_band"]
    cols = [c for c in ["Strong", "Steady", "Needs work", "Weak"] if c in gw.columns]
    gwr = "".join(f'<tr><td>{i_}</td>' + "".join(f'<td>{"-" if pd.isna(gw.loc[i_, c]) or v["grit_within_band_n"].loc[i_, c] < 200 else f"{100 * gw.loc[i_, c]:.1f}%"}</td>' for c in cols) + "</tr>" for i_ in gw.index)
    rec = v["grit_recovery"]
    gml = v["grit_ml"]
    ic_rows = "".join(f'<tr><td>{k}</td><td>{x["mean"]:.3f}</td><td>{x["t"]:.1f}</td><td>{x["p"]:.1g}</td><td>{x["n"]}</td></tr>' for k, x in v["ic"].items())
    return wrap(f'''<div style="display:flex;flex-direction:column;gap:16px"><div class="banner">{ICON_WARN}<div><b>Method test on synthetic data.</b> These results show the method works as
designed on a simulated world of 5,000 firms over 10 years. They are not evidence about real Zoho customers; a pilot on real, consenting books is the real test.</div></div>
<div class="card"><h2>Pass marks, set before the run</h2>{checks}</div>
<div class="grid2"><div class="card"><h2>Tiers in the test years (9-10)</h2><table class="t"><tr><th>Tier</th><th>Firm-months</th><th>Squeeze in 90 days</th><th>90% interval</th><th>Default in 12 months</th><th>Target</th></tr>{trows}</table>
<p class="note">Cut-offs from years 1-5: Safe ≥ {db.cal.cutoffs["Safe"]:.0f}, Moderate ≥ {db.cal.cutoffs["Moderate"]:.0f}, Watch ≥ {db.cal.cutoffs["Watch"]:.0f}.</p></div>
<div class="card"><h2>Cash squeeze within 90 days, by fifth of the Genius Score</h2>{qbars}<p class="note">Test years. Fifths are ranked within each month.</p></div></div>
<div class="grid2"><div class="card"><h2>Grit inside the same score band: 12-month default rate</h2><table class="t"><tr><th>Score</th>{"".join(f"<th>{c}</th>" for c in cols)}</tr>{gwr}</table>
<p class="note">Adds to the score for default ranking: AUC {v["grit_adds"]["without"]:.4f} without, {v["grit_adds"]["with"]:.4f} with, a negligible gain. Cells under 200 firm-months hidden.</p></div>
<div class="card"><h2>Grit and recovery after planted shocks</h2><div class="row"><span>Shocks followed</span><b>{rec["n"]:,}</b></div>
{"".join(f'<div class="row"><span>Months to recover, Grit {k}</span><b>{m_:.1f}</b></div>' for k, m_ in sorted(rec["by_band"].items(), key=lambda z: ["Strong", "Steady", "Needs work", "Weak"].index(z[0]) if z[0] in ["Strong", "Steady", "Needs work", "Weak"] else 9))}
<div class="row"><span>Rank correlation, Grit vs speed of recovery</span><b>{rec["spearman"]:.2f}</b></div>
<p class="note">Grit challenger: a model trained on half the firms to find the hidden trait directly reaches {gml["LightGBM"]:.2f} (LightGBM) and {gml["Ridge"]:.2f} (Ridge) against {gml["Rule-based Grit"]:.2f}
for the rule-based Grit. The rules leave signal on the table; they stay because owners can follow them. Genius and Grit are separate. An assumed demo policy permits a capped pricing concession for completed recovery evidence, with no automatic size bonus. These historical metrics do not validate that concession.</p></div></div>
<div class="card"><h2>Information coefficient of the Genius Score</h2><table class="t"><tr><th>Period</th><th>Mean IC</th><th>Newey-West t</th><th>One-sided p</th><th>Months</th></tr>{ic_rows}</table></div></div>''')


def ml_view(db, pol, conf_level: float, use_bh: bool, t: int | None = None) -> dict:
    """Recompute peer-group significance and every firm's confidence at the chosen level."""
    t = db.T - 1 if t is None else t
    gt = db.so["group_test"].copy()
    sig = SO.significant_groups(gt, conf_level, use_bh)
    gt["significant"] = gt["group"].map(sig).fillna(False)
    qs, qm = db.so["q_score"][:, t:t + 1].astype(float), db.so["q_ml"][:, t:t + 1].astype(float)
    conf = SO.confidence(qs, qm, db.keys[:, t:t + 1], sig)[:, 0]
    alive = db.P.alive[:, t] & ~np.isnan(qs[:, 0]) & ~np.isnan(qm[:, 0])
    mix = pd.Series(conf[alive]).value_counts().reindex(["High", "Medium", "Low", "Scorecard only"]).fillna(0).astype(int)
    agree = float(np.mean(qs[alive, 0] == qm[alive, 0]))
    within1 = float(np.mean(np.abs(qs[alive, 0] - qm[alive, 0]) <= 1))
    return dict(gt=gt, sig=sig, conf=conf, alive=alive, mix=mix, agree=agree, within1=within1, qs=qs[:, 0], qm=qm[:, 0], t=t)


def val_ml(db, pol, conf_level: float, use_bh: bool) -> str:
    v = db.val["so"]
    mv = ml_view(db, pol, conf_level, use_bh)
    from . import stats as S
    crit = S.critical_t(conf_level, v["test_ml"]["n"])
    def strow(name, x, diff=False):
        sig = x["t"] > crit and (x["mean"] > 0)
        return (f'<tr><td><b>{name}</b></td><td>{x["mean"]:+.3f}</td><td>{x["t"]:.1f}</td><td>{x["p"]:.2g}</td><td>{x["n"]}</td>'
                f'<td>{YES if sig else NO}</td></tr>')
    sel = "".join(f'<tr><td>{m}{CHOSEN if m == v["winner"] else ""}</td><td>{x["mean"]:.3f}</td><td>{x["t"]:.1f}</td></tr>'
                  for m, x in sorted(v["selection"].items(), key=lambda z: -z[1]["mean"]))
    imp = "".join(f'<div style="display:flex;align-items:center;gap:10px;margin:5px 0"><span style="width:52%;font-size:13px">{esc(feature_label(n))}</span>'
                  f'<div class="bar" style="width:{40 * w / v["importance"][0][1]:.0f}%;background:#0B5D5A"></div><span class="note">{100 * w:.1f}%</span></div>' for n, w in v["importance"][:12])
    gt = mv["gt"].sort_values("ic")
    grows = "".join(f'<tr><td>{esc(r.group)}</td><td>{r.ic:+.3f}</td><td>{r.t:.1f}</td><td>{r.p:.2g}</td>'
                    f'<td>{YES if r.significant else NO}</td></tr>' for r in gt.itertuples())
    mix = mv["mix"]
    tot = max(1, mix.sum())
    tiles = "".join(f'<div class="soft"><div>{conf_pill(k)}</div><b style="font-size:24px">{n:,}</b><div class="note">{100 * n / tot:.0f}% of live firms</div></div>' for k, n in mix.items())
    n_sig = int(mv["gt"]["significant"].sum())
    return wrap(f'''<div style="display:flex;flex-direction:column;gap:16px"><div class="banner">{ICON_WARN}<div><b>Second Opinion check on synthetic data.</b> A machine-learning model
ranks the same firms independently. It never changes a score; where it is reliable and more worried, it holds limits back for a person to review.</div></div>
<div class="card"><h2>Summary at {100 * conf_level:.1f}% confidence{", Benjamini-Hochberg on" if use_bh else ""}</h2>
<p style="font-size:15px;margin:0 0 12px">The Second Opinion ({esc(v["winner"])}, chosen on years 6-8) ranks next-quarter stress with IC <b>{v["test_ml"]["mean"]:.2f}</b> against <b>{v["test_score"]["mean"]:.2f}</b>
for the Genius Score in the test years. It is significant in <b>{n_sig} of {len(mv["gt"])}</b> peer groups. The two methods put a firm in the same fifth of its peer group
{100 * mv["agree"]:.0f}% of the time and within one fifth {100 * mv["within1"]:.0f}% of the time.</p><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">{tiles}</div></div>
<div class="grid2"><div class="card"><h2>Statistics, test years 9-10</h2><table class="t"><tr><th></th><th>Mean IC</th><th>Newey-West t</th><th>One-sided p</th><th>Months</th><th>Significant</th></tr>
{strow("Genius Score", v["test_score"])}{strow("Second Opinion", v["test_ml"])}{strow("Difference (ML - score)", v["test_diff"], True)}</table>
<p class="note">One-sided test that the IC is above zero. Critical t at this level: {crit:.2f}. Newey-West with 2 lags because overlapping quarters make months correlated.</p></div>
<div class="card"><h2>Model choice on years 6-8 (validation)</h2><table class="t"><tr><th>Model</th><th>Mean IC</th><th>t</th></tr>{sel}</table>
<p class="note">Trained on years 1-5, retrained every quarter walking forward with a 3-month gap. The test years were used once.</p></div></div>
<div class="grid2"><div class="card"><h2>Peer groups, weakest first</h2><div style="max-height:420px;overflow:auto"><table class="t"><tr><th>Group</th><th>IC</th><th>t</th><th>p</th><th>Significant</th></tr>{grows}</table></div></div>
<div class="card"><h2>What the Second Opinion leans on</h2>{imp}</div></div></div>''')


def review_queue(db, pol, conf_level, use_bh) -> pd.DataFrame:
    mv = ml_view(db, pol, conf_level, use_bh)
    t = mv["t"]
    idx = np.where(mv["alive"] & (mv["conf"] == "Low"))[0]
    df = pd.DataFrame(dict(ID=idx, Firm=db.firm_names[idx], Sector=db.P.firms["sector"].values[idx], Size=db.band[idx, t],
                           Score=np.round(db.gs["score"][idx, t].astype(float), 1), Tier=db.cal.tier(db.gs["score"][idx, t].astype(float)),
                           **{"Score fifth": mv["qs"][idx].astype(int), "Second Opinion fifth": mv["qm"][idx].astype(int)}))
    return df.sort_values("Score", ascending=False)


def grid5(db, pol, conf_level, use_bh) -> str:
    mv = ml_view(db, pol, conf_level, use_bh)
    a = mv["alive"]
    tab = pd.crosstab(mv["qs"][a].astype(int), mv["qm"][a].astype(int)).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    mx = tab.values.max()
    cells = ""
    for q1 in range(5, 0, -1):
        cells += f'<tr><td class="note">Score fifth {q1}</td>'
        for q2 in range(1, 6):
            n = tab.loc[q1, q2]
            col = "#E3F2EA" if q1 == q2 else ("#FBEFD9" if q2 - q1 <= -2 else "#F7F9FA")
            cells += f'<td style="text-align:center;background:{col};opacity:{0.45 + 0.55 * n / mx:.2f}"><b>{n}</b></td>'
        cells += "</tr>"
    return wrap(f'<div class="card"><h2>Score fifth against Second Opinion fifth (live firms, latest month)</h2><table class="t">{cells}'
                f'<tr><td></td>{"".join(f"<td class=note style=text-align:center>ML {k}</td>" for k in range(1, 6))}</tr></table>'
                '<p class="note">Green diagonal: the methods agree. Amber: the Second Opinion is two or more fifths more worried (Low confidence where the group is significant).</p></div>')
