"""Genius prototype app (Gradio). Owner screens, Ask Genius, the lender's screens and the Validation room, all on one synthetic database.

    from genius import app
    app.launch()          # builds or loads the database, then serves the app with a public link
"""
from __future__ import annotations

import copy
import datetime as dt
import os
import tempfile
import time
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from . import assess, config as C, lender as LD, lending, pipeline, ui
from .policy import CONDITIONS, DEFAULTS, GROUPS, LABELS, Policy
from .session import Session

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {
    "Sharma Traders · building materials (Tally day book, CSV)": ("Sharma_Traders_daybook_FY24-FY26.csv", 2.5e7, ["Rajan Polymers", "Sundaram Industries", "Lakshmi Plastics"]),
    "Sharma Traders · same books as a ledger-by-account workbook": ("Sharma_Traders_ledger_by_account.xlsx", 2.5e7, ["Rajan Polymers", "Sundaram Industries", "Lakshmi Plastics"]),
    "Kaveri Foods · food processing (Zoho Books journal)": ("Kaveri_Foods_journal_FY24-FY26.xlsx", 1e7, ["Cauvery Agro Producers", "Deccan Packaging", "Malnad Spices"]),
    "Swift Freight · logistics (messy export)": ("Swift_Freight_journal_messy.xlsx", 3e6, ["Rathore Fleet Hire", "Toll and Tyre Services"]),
}
DB = None
BASE_POL = None
PORT = {}          # policy fingerprint -> portfolio table (shared across sessions; same policy, same table)


# ------------------------------------------------------------------------------------------------ data
def get_db(log=print, rebuild=False):
    global DB, BASE_POL
    if DB is None:
        p = Path(os.environ.get("GENIUS_DB", pipeline.ART / "genius_db.pkl"))
        if p.exists() and not rebuild:
            t = time.time(); cached = pipeline.load(p)
            from .credit_model import VERSION, PREVIOUS_VERSION
            model = getattr(cached, "learned_model", None)
            if getattr(model, "version", None) == PREVIOUS_VERSION and not model.with_recovery:
                # Compatible financial coefficients/data: migrate metadata, do not re-fit or change outcomes.
                model.version = VERSION
                model.report.update(scores_separate=True, recovery_selected=False,
                    selection_rule="Genius is financial-only; Grit is separate. Combined results are exploratory only")
                pipeline.save(cached, p)
            if getattr(model, "version", None) != VERSION:
                log("Cached database belongs to an older scoring version; rebuilding Genius v5.")
                DB = pipeline.build(5000, log=log)
                pipeline.save(DB, p)
            else:
                DB = cached
                log(f"Loaded the synthetic database in {time.time() - t:.0f}s")
        else:
            log("Building the synthetic database (5,000 firms x 120 months). About 5-10 minutes the first time.")
            DB = pipeline.build(5000, log=log)
            pipeline.save(DB, p)
        BASE_POL = new_policy()
    return DB


def new_policy() -> Policy:
    p = Policy()
    p.calibrated_pd = {t: round(100 * DB.cal.tier_pd[t], 1) for t in C.TIERS}
    return p


def fingerprint(pol: Policy):
    return repr((sorted(pol.base.items(), key=lambda kv: kv[0]), [(r.param, r.value, sorted(r.when.items())) for r in pol.rules]))


def port(pol: Policy) -> pd.DataFrame:
    k = fingerprint(pol)
    if k not in PORT:
        if len(PORT) > 12:
            PORT.pop(next(iter(PORT)))
        PORT[k] = LD.portfolio(DB, pol)
    return PORT[k]


# ------------------------------------------------------------------------------------------------ owner flow
def do_read(sample, upload, company, st):
    st = st or {}
    path = upload if upload else (str(ROOT / "samples" / SAMPLES[sample][0]) if sample else None)
    if not path:
        raise gr.Error("Pick a sample or upload a file.")
    s = Session()
    try:
        r = s.read(path, company or None)
    except Exception as e:  # noqa
        raise gr.Error(f"Could not read that file: {e}")
    st["sess"] = s
    st["read"] = r
    cc, mse = (SAMPLES[sample][1], SAMPLES[sample][2]) if (sample and not upload) else (None, [])
    sups = r["suppliers"][:25]
    return (st, ui.upload_check(r, s.company), gr.update(value=r["sector"]), gr.update(value=cc),
            gr.update(choices=sups, value=[m for m in mse if m in sups]), gr.update(visible=True), ui.quote_prompt(), schedule_empty())


def do_run(sector, udyam, cc, mse, consent, st):
    st = st or {}
    s = st.get("sess")
    if s is None:
        raise gr.Error("Read a file first.")
    if not consent:
        raise gr.Error("Please acknowledge the prototype notice. Live lending would require consent-based verification and a lender decision.")
    pol = st.setdefault("pol", new_policy())
    months = int(st.get("read", {}).get("report", {}).get("checks", {}).get("months", 0) or 0)
    if months and months < 12:
        raise gr.Error(f"The books cover {months} months. Genius needs at least 12 months to publish a credit standing.")
    try:
        s.run(DB, pol, sector, cc_limit=float(cc) if cc else None, mse=mse or [], udyam=None if udyam == "Estimate from the books" else udyam)
    except gr.Error:
        raise
    except Exception as e:  # noqa
        raise gr.Error("Genius could not assess these books: it needs at least 12 months of sales, with customer and supplier "
                       f"invoices. Check the items flagged on the Upload tab. (Detail: {type(e).__name__})")
    st["levers"] = levers(s)
    return [st] + owner_outputs(st) + [gr.update(selected="standing")]


def owner_outputs(st):
    s = st["sess"]; a = s.a; pol = st["pol"]
    lev = st.get("levers") or {}
    measures = [(f'{C.M[m]["name"]} ({C.M[m]["area"]})', m) for m in C.MEASURES]
    tree = s.genie.tree() if s.genie else {}
    return [ui.header(s), ui.standing(a, DB, pol), ui.peers(a, DB), ui.borrow(a, pol), ui.quote_prompt(), ui.rate_chart(a, pol),
            ui.ew_chart(s.ew), ui.ew_text(s.ew), gr.update(choices=list(lev), value=[]), ui.improve(a, None, []), bills_default(s),
            ui.early_pay(pol.early_payment(bills_rows(bills_default(s)), a.pricing.get("short", {}).get("rate", 15)), a.pricing.get("short", {}).get("rate", 15)) if a.tier else ui.empty(),
            ui.details_area(a, C.AREA_LIST[0]), gr.update(choices=measures, value="RC05"), ui.measure_card(a, "RC05", s.books),
            gr.update(choices=list(tree), value=None), [], "", schedule_empty()]


def levers(s):
    a = s.a
    if a.score is None:
        return {}
    out = {}
    od = float(a.f["ar_overdue"][0, -1])
    if od > 0:
        out[f"Collect the {ui.money(od)} overdue from customers"] = dict(collect=od)
    tc = a.m("RC09")
    if tc is not None and tc > 20:
        out["Bring your biggest customer below 20% of sales"] = dict(top_cust_to=0.2)
    dr = a.m("OW03")
    if dr is not None and dr > 50:
        out["Keep drawings under half of operating cash flow"] = dict(drawings_share=0.5)
    return out


def do_improve(chosen, st):
    s = st.get("sess") if st else None
    if s is None or s.a is None:
        return ui.empty()
    if not chosen:
        return ui.improve(s.a, None, [])
    kw = {}
    for c in chosen:
        kw.update(st["levers"][c])
    r = assess.what_if(DB, st["pol"], s.a, **kw)
    return ui.improve(s.a, r, chosen)


def bills_default(s) -> pd.DataFrame:
    try:
        inv = s.books.invoices("payable")
        op = inv[(inv["open"] > 0.5) & (inv["amount"] > 0)]
        g = op.groupby("party")["open"].sum().sort_values(ascending=False).head(5)
        if g.empty:
            g = inv.groupby("party")["amount"].sum().sort_values(ascending=False).head(5) / 6
    except Exception:
        return pd.DataFrame(columns=["Supplier", "Bill (₹)", "Discount %", "Days early", "Micro or small"])
    mse = set((s.inputs or {}).get("mse", []))
    return pd.DataFrame({"Supplier": g.index, "Bill (₹)": g.values.round(0), "Discount %": 2.0, "Days early": 30, "Micro or small": [p in mse for p in g.index]})


def bills_rows(df) -> list:
    out = []
    for r in pd.DataFrame(df).itertuples(index=False):
        try:
            out.append(dict(supplier=str(r[0]), amount=float(r[1]), discount_pct=float(r[2]), days_early=float(r[3]), msme=str(r[4]).lower() in ("true", "1", "yes")))
        except (TypeError, ValueError):
            continue
    return [b for b in out if b["amount"] > 0]


def do_bills(df, st):
    s = st.get("sess") if st else None
    if s is None or s.a is None or not s.a.tier:
        return ui.empty()
    r = s.a.pricing["short"]["rate"]
    return ui.early_pay(st["pol"].early_payment(bills_rows(df), r), r)


def schedule_empty():
    return pd.DataFrame(columns=["Period", "Principal", "Interest", "Fees", "Payment", "Balance"])


def do_calc(amount, days, product, months, secured, collateral, st):
    s = st.get("sess") if st else None
    if s is None or s.a is None:
        return ui.empty(), schedule_empty()
    q = lending.quote(s.a, st["pol"], amount, days, product, months, secured, collateral)
    return ui.quote_result(q), pd.DataFrame(q["schedule"]) if q["schedule"] else schedule_empty()


def borrow_fields(product, secured):
    medium = product == "Medium term"
    maxd = 90 if product in ("Invoice financing", "Shock-absorber line") else 180
    return (gr.update(visible=not medium, maximum=maxd, value=min(60, maxd)),
            gr.update(visible=medium), gr.update(visible=bool(secured)), ui.quote_prompt(), schedule_empty())


def clear_quote():
    return ui.quote_prompt(), schedule_empty()


def do_area(area, st):
    s = st.get("sess") if st else None
    return ui.details_area(s.a if s else None, area)


def do_measure(mid, st):
    s = st.get("sess") if st else None
    return ui.measure_card(s.a if s else None, mid, s.books if s else None) if mid else ui.empty()


# ------------------------------------------------------------------------------------------------ Ask Genius
def _node(st, path):
    t = st["sess"].genie.tree()
    for p in path:
        t = t.get(p, {}) if p else {}
    return t


def ask_level(level, v1, v2, v3, st):
    s = st.get("sess") if st else None
    if s is None or s.genie is None:
        return gr.update(), gr.update(), gr.update()
    path = [v1, v2, v3][:level]
    node = _node(st, path) if all(path) else {}
    subs = [k for k in node if k != "_q"]
    qs = [(s.genie.by_id[q].text, q) for q in node.get("_q", [])]
    upd = [gr.update(), gr.update()]
    if level == 1:
        upd = [gr.update(choices=subs, value=None, visible=bool(subs)), gr.update(choices=[], value=None, visible=False)]
    elif level == 2:
        upd = [gr.update(), gr.update(choices=subs, value=None, visible=bool(subs))]
    return upd[0], upd[1], gr.update(choices=qs, value=None)


def _say(st, hist, user_text, ans):
    hist = list(hist or [])
    hist.append({"role": "user", "content": user_text})
    body = ans.text
    if ans.understood and ans.understood.lower().strip("?") != user_text.lower().strip("?"):
        body = f"*Answering: {ans.understood}*\n\n" + body
    if ans.sources:
        body += "\n\n<span style='font-size:12px;color:#5E6B75'>From: " + ", ".join(map(str, ans.sources)) + "</span>"
    hist.append({"role": "assistant", "content": body})
    opts = [(t, i) for i, t in ans.options] if ans.status == "did_you_mean" else []
    if not opts:
        g = st["sess"].genie
        for f in ans.followups[:3]:
            hit = next((q.id for q in g.bank if q.text == f), None)
            opts.append((f, hit or f))
    return hist, gr.update(choices=opts, value=None, visible=bool(opts))


def ask_pick(qid, hist, st):
    s = st.get("sess") if st else None
    if not qid or s is None or s.genie is None:
        return hist, gr.update()
    g = s.genie
    ans = g.answer(qid) if qid in g.by_id else g.ask(qid)
    return _say(st, hist, g.by_id[qid].text if qid in g.by_id else qid, ans)


def ask_typed(text, hist, st):
    s = st.get("sess") if st else None
    if s is None or s.genie is None:
        hist = list(hist or []) + [{"role": "user", "content": text}, {"role": "assistant", "content": "Load books on the Upload tab first, then ask me anything about them."}]
        return hist, gr.update(), ""
    h, o = _say(st, hist, text, s.genie.ask(text))
    return h, o, ""


# ------------------------------------------------------------------------------------------------ lender: applications and firm detail
SHOW = ["ID", "Firm", "Sector", "Size", "Score", "Tier", "Grit", "Confidence", "Score 5th", "ML 5th", "ST limit (₹ lakh)",
        "MT limit (₹ lakh)", "Rate %"]


def apps_view(tiers, confs, sector, size, min_score, gbands, text, page, st):
    pol = (st or {}).get("pol") or BASE_POL
    full = port(pol)
    df = LD.filter_df(full, tiers, confs, sector, size, min_score, gbands, text).sort_values("Score", ascending=False)
    pages = max(1, int(np.ceil(len(df) / 50)))
    page = int(min(max(1, page or 1), pages))
    return LD.portfolio_summary(df, full), df[SHOW].iloc[(page - 1) * 50: page * 50], gr.update(maximum=pages, value=page, label=f"Page (of {pages})")


def apps_export(tiers, confs, sector, size, min_score, gbands, text, st):
    pol = (st or {}).get("pol") or BASE_POL
    df = LD.filter_df(port(pol), tiers, confs, sector, size, min_score, gbands, text).sort_values("Score", ascending=False)
    p = Path(tempfile.gettempdir()) / f"genius_applications_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(p, index=False)
    return gr.update(value=str(p), visible=True)


def view_save(name, tiers, confs, sector, size, min_score, gbands, text, st):
    st = st or {}
    if not name:
        raise gr.Error("Give the view a name.")
    v = st.setdefault("views", dict(DEFAULT_VIEWS))
    v[name] = dict(tiers=tiers, confs=confs, sector=sector, size=size, min_score=min_score, gbands=gbands, text=text)
    return st, gr.update(choices=list(v), value=name)


DEFAULT_VIEWS = {
    "All live firms": dict(tiers=[], confs=[], sector="All", size="All", min_score=0, gbands=[], text=""),
    "Review queue (Low confidence)": dict(tiers=[], confs=["Low"], sector="All", size="All", min_score=0, gbands=[], text=""),
    "Safe and Moderate, High confidence": dict(tiers=["Safe", "Moderate"], confs=["High"], sector="All", size="All", min_score=0, gbands=[], text=""),
    "Watch tier with Strong Grit": dict(tiers=["Watch"], confs=[], sector="All", size="All", min_score=0, gbands=["Strong"], text=""),
}


def view_load(name, st):
    v = ((st or {}).get("views") or DEFAULT_VIEWS).get(name)
    if not v:
        return [gr.update()] * 7
    return [gr.update(value=v["tiers"]), gr.update(value=v["confs"]), gr.update(value=v["sector"]), gr.update(value=v["size"]),
            gr.update(value=v["min_score"]), gr.update(value=v["gbands"]), gr.update(value=v["text"])]


def pick_firm(df, evt: gr.SelectData, st):
    try:
        row = evt.index[0]
        i = int(pd.DataFrame(df).iloc[row]["ID"])
    except Exception:
        return gr.update(), gr.update(), gr.update()
    return i, firm_html(i, st), gr.update(selected="firm")


def firm_html(i, st):
    if i is None or i == "":
        return ui.wrap('<div class="card"><p class="muted" style="margin:0">Click a firm in the Applications list, or type its ID.</p></div>')
    i = int(i)
    if not (0 <= i < len(DB.firm_names)) or not DB.P.alive[i, -1] or np.isnan(DB.gs["score"][i, -1]):
        return ui.empty("That firm is not live or has no score at the latest month.")
    pol = (st or {}).get("pol") or BASE_POL
    return LD.firm_detail(DB, pol, i, ((st or {}).get("notes") or {}).get(i, []))


def add_note(i, text, who, st):
    st = st or {}
    if i is None or not text:
        raise gr.Error("Pick a firm and write a note.")
    st.setdefault("notes", {}).setdefault(int(i), []).append(dict(by=who or "Reviewer", at=dt.datetime.now().strftime("%d %b %Y %H:%M"), text=text))
    return st, firm_html(i, st), ""


# ------------------------------------------------------------------------------------------------ lender: policy
def policy_table(pol) -> pd.DataFrame:
    rows = []
    for g, keys in GROUPS.items():
        for k in keys:
            v, src = pol.get(k, {})
            rows.append({"Group": g, "Setting": LABELS[k], "Key": k, "Value": "calibrated" if (k.startswith("pd_") and pol.base[k] is None) else v,
                         "Genius default": DEFAULTS[k] if DEFAULTS[k] is not None else "calibrated", "In use": f"{v:.2f}" if isinstance(v, float) else str(v)})
    return pd.DataFrame(rows)


def _parse(k, x):
    if isinstance(x, str):
        x = x.strip()
        if k.startswith("pd_") and x.lower() in ("calibrated", "", "none"):
            return None
        if k in ("pricing_mode", "fee_mode"):
            return x
        if k in ("so_bh", "grit_policy_enabled"):
            return x.lower() in ("true", "1", "yes", "on")
    if k in ("so_bh", "grit_policy_enabled"):
        return bool(x)
    if k in ("pricing_mode", "fee_mode"):
        return str(x)
    return float(x)


def draft_from(df, st):
    pol = copy.deepcopy(st.get("pol") or new_policy())
    changes, bad = {}, []
    for r in pd.DataFrame(df).itertuples(index=False):
        k, v = r[2], r[3]
        try:
            nv = _parse(k, v)
        except (TypeError, ValueError):
            bad.append(f"{LABELS.get(k, k)}: '{v}' is not a number")
            continue
        if nv != pol.base[k]:
            changes[k] = nv
    if k_ := [k for k in ("pricing_mode",) if k in changes and changes[k] not in ("score", "tier")]:
        bad.append("Pricing by score or by tier must be 'score' or 'tier'"); changes.pop("pricing_mode")
    if "fee_mode" in changes and changes["fee_mode"] not in ("annual_on_limit", "per_drawdown"):
        bad.append("Fee mode must be 'annual_on_limit' or 'per_drawdown'"); changes.pop("fee_mode")
    bad += pol.set(changes, why="draft") if changes else []
    return pol, changes, bad


def pol_preview(df, st):
    st = st or {}
    st.setdefault("pol", new_policy())
    draft, changes, bad = draft_from(df, st)
    msg = ("<b>Draft changes:</b> " + ", ".join(f"{LABELS[k]} → {v}" for k, v in changes.items()) if changes else "No changes from the saved policy.")
    if bad:
        msg += "<br><b>Refused by guard-rails:</b> " + "; ".join(bad)
    s = st.get("sess")
    firm = ""
    if s is not None and s.a is not None and s.a.tier:
        a = s.a
        r0 = st["pol"].rate(a.tier, a.info.get("policy_context"), pd_score=100 * a.p_default)["rate"]; r1 = draft.rate(a.tier, a.info.get("policy_context"), pd_score=100 * a.p_default)["rate"]
        firm = f'<div class="row"><span>{ui.esc(a.name)}: short-term rate</span><b>{r0:.2f}% → {r1:.2f}%</b></div>'
    return (ui.wrap(f'<div class="ok">{msg}</div>{firm}'), LD.rate_table(draft), LD.rate_build(draft), LD.impact(port(st["pol"]), port(draft)))


def pol_save(df, why, who, st):
    st = st or {}
    st.setdefault("pol", new_policy())
    draft, changes, bad = draft_from(df, st)
    if not changes:
        raise gr.Error("Nothing to save: no setting differs from the saved policy." + (" " + "; ".join(bad) if bad else ""))
    pol = st["pol"]
    v0 = pol.version
    refused = pol.set(changes, by=who or "lender", why=why or "settings changed")
    st["pol"] = pol
    refused = list(dict.fromkeys(refused + bad))
    if pol.version == v0:
        raise gr.Error("Nothing saved. " + "; ".join(refused))
    return [st, policy_table(pol), versions(pol), ui.wrap(f'<div class="ok">Saved as policy version {pol.version}.' + (f' Refused: {"; ".join(refused)}' if refused else "") + "</div>")] + refresh_owner(st)


def pol_reset(st):
    st = st or {}
    st.setdefault("pol", new_policy()).reset()
    return [st, policy_table(st["pol"]), versions(st["pol"]), rules_df(st["pol"]), ui.wrap('<div class="ok">Back to Genius defaults (saved as a new version).</div>')] + refresh_owner(st)


def rules_df(pol):
    return pd.DataFrame([dict(No=i + 1, Setting=LABELS[r.param], Value=r.value, When=", ".join(f"{k} = {v}" for k, v in r.when.items()) or "always", Note=r.note)
                         for i, r in enumerate(pol.rules)], columns=["No", "Setting", "Value", "When", "Note"])


def versions(pol):
    return pd.DataFrame([dict(Version=v["version"], When=v["at"].replace("T", " "), By=v["by"], Why=v["why"], Overrides=len(v["rules"])) for v in pol.versions][::-1])


def rule_add(setting, value, tier, tenor, sector, size, secured, regime, from_date, note, st):
    st = st or {}
    pol = st.setdefault("pol", new_policy())
    k = next((k for k, l in LABELS.items() if l == setting), None)
    if k is None:
        raise gr.Error("Pick a setting.")
    try:
        v = _parse(k, value)
    except (TypeError, ValueError):
        raise gr.Error(f"'{value}' is not a valid value for {setting}.")
    when = {c: x for c, x in dict(tier=tier, tenor=tenor, sector=sector, size=size, secured=secured, regime=regime, from_date=from_date).items() if x and x != "Any"}
    try:
        pol.add_rule(k, v, when, note=note or "")
    except ValueError as e:
        raise gr.Error(str(e))
    return [st, rules_df(pol), versions(pol), ui.wrap(f'<div class="ok">Override added (policy version {pol.version}). The most specific matching rule wins.</div>')] + refresh_owner(st)


def rule_remove(no, st):
    st = st or {}
    pol = st.setdefault("pol", new_policy())
    try:
        k = float(no)
    except (TypeError, ValueError):
        k = -1
    if k != int(k) or not (1 <= int(k) <= len(pol.rules)):
        raise gr.Error(f"Type the number of an override from 1 to {len(pol.rules)}." if pol.rules else "There are no overrides to remove.")
    pol.remove_rule(int(k) - 1)
    return [st, rules_df(pol), versions(pol), ui.wrap(f'<div class="ok">Override removed (policy version {pol.version}).</div>')] + refresh_owner(st)


def refresh_owner(st):
    """Owner screens re-run under the new policy (only the policy step changes; the score does not)."""
    s = st.get("sess")
    if s is None or s.a is None:
        return [gr.update()] * 4 + [ui.quote_prompt(), schedule_empty()]
    s.run(DB, st["pol"], s.sector, s.inputs.get("cc_limit"), s.inputs.get("mse"), s.inputs.get("udyam"))
    a = s.a
    return [ui.header(s), ui.standing(a, DB, st["pol"]), ui.borrow(a, st["pol"]), ui.rate_chart(a, st["pol"]), ui.quote_prompt(), schedule_empty()]


# ------------------------------------------------------------------------------------------------ validation
def val_ml(conf, bh):
    conf = float(conf)
    return LD.val_ml(DB, BASE_POL, conf, bool(bh)), LD.review_queue(DB, BASE_POL, conf, bool(bh)).head(300), LD.grid5(DB, BASE_POL, conf, bool(bh))


# ------------------------------------------------------------------------------------------------ layout
def build_ui():
    get_db()
    pol0 = BASE_POL
    with gr.Blocks(title="Genius · credit standing for MSMEs") as demo:
        st = gr.State({})
        hdr = gr.HTML(ui.header())
        with gr.Tabs() as tabs:
            # ------------------------------------------------ owner
            with gr.Tab("1 · Upload", id="upload"):
                gr.HTML(ui.wrap('<div><h1>Upload your books</h1><p class="sub">A Tally day book, a Zoho Books journal, or a ledger-by-account workbook. Genius reads it, checks it, '
                                'and compares your business with peers of the same sector and government MSME size. Or try one of the three sample businesses.</p></div>'))
                with gr.Row():
                    with gr.Column(scale=3):
                        sample = gr.Radio(list(SAMPLES), value=list(SAMPLES)[0], label="Sample businesses (made-up books)")
                        upload = gr.File(label="…or upload your own books (CSV or Excel)", type="filepath", file_types=[".csv", ".xlsx", ".xls"])
                        company = gr.Textbox(label="Business name (optional)", placeholder="Read from the file name if left empty")
                        read_btn = gr.Button("Read the books", variant="primary")
                    with gr.Column(scale=4):
                        check = gr.HTML(ui.empty("Pick a sample or upload a file, then press Read the books. Reading takes 10-40 seconds."))
                with gr.Group(visible=False) as confirm:
                    gr.HTML(ui.wrap('<h2 style="margin:12px 0 0">Confirm a few details</h2>'))
                    with gr.Row():
                        sector = gr.Dropdown(C.SECTOR_LIST, label="Sector (suggested from your accounts)")
                        udyam = gr.Dropdown(["Estimate from the books"] + list(C.SIZE_LIST), value="Estimate from the books", label="Udyam MSME category")
                        cc = gr.Number(label="Cash-credit / overdraft limit (₹)", precision=0)
                    mse = gr.CheckboxGroup([], label="Which suppliers are micro or small enterprises? (45-day payment rule, section 43B(h))")
                    consent = gr.Checkbox(label="I understand these are prototype estimates. Live lending requires verified records, explicit sharing consent and lender approval.", value=True)
                    run_btn = gr.Button("Get my credit standing", variant="primary")
            with gr.Tab("2 · Credit standing", id="standing"):
                standing = gr.HTML(ui.empty())
            with gr.Tab("3 · Peers", id="peers"):
                peers = gr.HTML(ui.empty())
            with gr.Tab("4 · Borrow", id="borrow"):
                borrow = gr.HTML(ui.empty())
                with gr.Row():
                    product = gr.Dropdown(list(lending.PRODUCTS), value="Short term", label="What kind of borrowing?")
                    amount = gr.Number(value=None, label="How much do you need? (₹)", precision=0, minimum=1)
                with gr.Row():
                    days = gr.Slider(7, 180, value=60, step=1, label="Repay after how many days?")
                    months = gr.Slider(6, 60, value=36, step=1, label="Repayment period (months)", visible=False)
                    secured = gr.Checkbox(value=False, label="I can offer collateral (required for medium term)")
                    collateral = gr.Number(value=None, label="Estimated collateral value (₹)", precision=0, minimum=1, visible=False)
                quote_btn = gr.Button("Calculate my quote", variant="primary", size="lg")
                calc = gr.HTML(ui.quote_prompt())
                repayments = gr.Dataframe(schedule_empty(), label="Repayment plan (₹) · fees paid at disbursement are deducted from net cash received", interactive=False, wrap=True)
                with gr.Accordion("Demo: how resilience can affect credit pricing", open=False):
                    gr.HTML(ui.resilience_lab())
                with gr.Accordion("How loan length affects indicative pricing", open=False):
                    rchart = gr.Plot(label="Rate by loan length")
                with gr.Row():
                    ewc = gr.Plot(label="Early warning")
                    ewt = gr.HTML(ui.empty())
            with gr.Tab("5 · Improve your rate", id="improve"):
                gr.HTML(ui.wrap('<div><h1>Improve your rate</h1><p class="sub">Changes that would lift your Genius Score, each re-scored on your own books.</p></div>'))
                with gr.Row():
                    lev = gr.CheckboxGroup([], label="Try a change")
                    imp = gr.HTML(ui.empty())
                gr.HTML(ui.wrap('<h2 style="margin:12px 0 0">Pay suppliers early for a discount? Edit the table and press Recalculate.</h2>'))
                bills = gr.Dataframe(interactive=True, wrap=True)
                bills_btn = gr.Button("Recalculate")
                ep = gr.HTML(ui.empty())
            with gr.Tab("6 · Details", id="details"):
                with gr.Row():
                    area = gr.Radio(C.AREA_LIST, value=C.AREA_LIST[0], label="Area")
                det = gr.HTML(ui.empty())
                mdd = gr.Dropdown([], label="Any of the 54 measures")
                mcard = gr.HTML(ui.empty())
                with gr.Accordion("Learned factors, weights and validation", open=False):
                    gr.HTML(ui.learned_model_summary(DB))
            with gr.Tab("7 · Ask Genius", id="ask"):
                gr.HTML(ui.wrap('<div><h1>Ask Genius</h1><p class="sub">Click through the topics, or type a question. Answers come only from your books and Genius\'s own numbers; '
                                'no external AI service is called for chat answers.</p></div>'))
                with gr.Row():
                    with gr.Column(scale=2):
                        t1 = gr.Dropdown([], label="Topic")
                        t2 = gr.Dropdown([], label="Sub-topic", visible=False)
                        t3 = gr.Dropdown([], label="Detail", visible=False)
                        qs = gr.Radio([], label="Questions")
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(label="Genius", height=430)
                        opts = gr.Radio([], label="Did you mean / ask next", visible=False)
                        txt = gr.Textbox(placeholder="Type a question, e.g. why is my collection score low?", show_label=False)
            # ------------------------------------------------ lender
            with gr.Tab("Lender", id="lender"):
                with gr.Tabs() as ltabs:
                    with gr.Tab("Applications", id="apps"):
                        with gr.Row():
                            views = gr.Dropdown(list(DEFAULT_VIEWS), value="All live firms", label="Saved view")
                            vname = gr.Textbox(label="Save current filters as", placeholder="e.g. Pharma, small, review")
                            vsave = gr.Button("Save view")
                        with gr.Row():
                            f_tier = gr.CheckboxGroup(list(C.TIERS), label="Tier")
                            f_conf = gr.CheckboxGroup(["High", "Medium", "Low", "Scorecard only"], label="Confidence")
                            f_grit = gr.CheckboxGroup(["Strong", "Steady", "Needs work", "Weak"], label="Grit band")
                        with gr.Row():
                            f_sec = gr.Dropdown(["All"] + list(C.SECTOR_LIST), value="All", label="Sector")
                            f_size = gr.Dropdown(["All"] + list(C.SIZE_LIST), value="All", label="Size")
                            f_min = gr.Slider(0, 100, value=0, step=1, label="Minimum Genius Score")
                            f_txt = gr.Textbox(label="Firm name contains")
                        summ = gr.HTML()
                        apps = gr.Dataframe(interactive=False, max_height=520)
                        with gr.Row():
                            page = gr.Number(value=1, precision=0, label="Page", minimum=1)
                            exp_btn = gr.Button("Export filtered list (CSV)")
                            exp_file = gr.File(visible=False, label="Download")
                        gr.HTML(ui.wrap('<p class="note">Click a row to open the firm. Synthetic firms, latest month (Sep 2026). 5th = fifth of the peer group (5 strongest). ST / MT = short / medium term. Limits and rates follow your current policy.</p>'))
                    with gr.Tab("Firm detail", id="firm"):
                        with gr.Row():
                            fid = gr.Number(label="Firm ID", precision=0)
                            fopen = gr.Button("Open")
                        fdet = gr.HTML(firm_html(None, {}))
                        with gr.Row():
                            note_who = gr.Textbox(label="Reviewer", value="Credit reviewer")
                            note_txt = gr.Textbox(label="Note", placeholder="e.g. Called the owner: large customer paid on 3 Oct; keep limit at 50% until GST check.", scale=3)
                            note_btn = gr.Button("Add note")
                    with gr.Tab("Policy", id="policy"):
                        gr.HTML(ui.wrap('<div><h1>Lender policy</h1><p class="sub">Every number that sets a limit or a rate is yours to set. Edit the Value column, preview the effect, '
                                        'then save as a new version. Overrides apply a different value in a situation; the most specific matching rule wins.</p></div>'))
                        ptab = gr.Dataframe(policy_table(pol0), interactive=True, wrap=True, max_height=560)
                        with gr.Row():
                            why = gr.Textbox(label="Reason for the change", placeholder="e.g. repo cut by 25 bp")
                            who = gr.Textbox(label="Changed by", value="Partner lender: credit head")
                        with gr.Row():
                            prev_btn = gr.Button("Preview")
                            save_btn = gr.Button("Save as new version", variant="primary")
                            reset_btn = gr.Button("Reset to Genius defaults")
                        pmsg = gr.HTML()
                        with gr.Row():
                            with gr.Column():
                                rtab = gr.Dataframe(LD.rate_table(pol0), label="Rates by tier and loan (draft)", interactive=False)
                                rbuild = gr.HTML(LD.rate_build(pol0))
                            with gr.Column():
                                pimpact = gr.HTML()
                        gr.HTML(ui.wrap('<h2 style="margin:16px 0 0">Overrides</h2>'))
                        with gr.Row():
                            r_set = gr.Dropdown([LABELS[k] for k in DEFAULTS], label="Setting")
                            r_val = gr.Textbox(label="Value")
                            r_note = gr.Textbox(label="Note")
                        with gr.Row():
                            r_tier = gr.Dropdown(["Any"] + list(C.TIERS), value="Any", label="Tier")
                            r_tenor = gr.Dropdown(["Any", "short", "medium", "long"], value="Any", label="Loan length")
                            r_sec = gr.Dropdown(["Any"] + list(C.SECTOR_LIST), value="Any", label="Sector")
                            r_size = gr.Dropdown(["Any"] + list(C.SIZE_LIST), value="Any", label="Size")
                            r_secd = gr.Dropdown(["Any", "yes", "no"], value="Any", label="Secured")
                            r_reg = gr.Dropdown(["Any", "calm", "stress"], value="Any", label="Market regime")
                            r_from = gr.Textbox(label="From date (YYYY-MM-DD)")
                        with gr.Row():
                            r_add = gr.Button("Add override")
                            r_no = gr.Number(label="Override no. to remove", precision=0)
                            r_rm = gr.Button("Remove")
                        rules = gr.Dataframe(rules_df(pol0), interactive=False)
                        gr.HTML(ui.wrap('<h2 style="margin:16px 0 0">Version log</h2>'))
                        vlog = gr.Dataframe(versions(pol0), interactive=False)
            # ------------------------------------------------ validation room
            with gr.Tab("Validation room", id="validation"):
                with gr.Tabs():
                    with gr.Tab("Learned score"):
                        gr.HTML(ui.learned_model_summary(DB))
                    with gr.Tab("Score and lending checks"):
                        gr.HTML(LD.val_scorecard(DB, pol0))
                    with gr.Tab("ML check (Second Opinion)"):
                        with gr.Row():
                            conf = gr.Slider(0.80, 0.99, value=0.95, step=0.01, label="Confidence level for significance")
                            bh = gr.Checkbox(value=True, label="Correct for testing many peer groups (Benjamini-Hochberg)")
                        v0 = val_ml(0.95, True)
                        mlh = gr.HTML(v0[0])
                        gr.HTML(ui.wrap('<h2 style="margin:12px 0 0">Review queue: Second Opinion two or more fifths more worried, in groups where it is significant</h2>'))
                        rq = gr.Dataframe(v0[1], interactive=False, max_height=360)
                        with gr.Accordion("Score fifth against Second Opinion fifth (5 x 5)", open=False):
                            g5 = gr.HTML(v0[2])

        # ------------------------------------------------ wiring
        owner_out = [hdr, standing, peers, borrow, calc, rchart, ewc, ewt, lev, imp, bills, ep, det, mdd, mcard, t1, chat, txt, repayments]
        read_btn.click(do_read, [sample, upload, company, st], [st, check, sector, cc, mse, confirm, calc, repayments])
        run_btn.click(do_run, [sector, udyam, cc, mse, consent, st], [st] + owner_out + [tabs])
        quote_btn.click(do_calc, [amount, days, product, months, secured, collateral, st], [calc, repayments], api_name="quote")
        product.change(borrow_fields, [product, secured], [days, months, collateral, calc, repayments])
        secured.change(lambda x: (gr.update(visible=bool(x)), ui.quote_prompt(), schedule_empty()), secured, [collateral, calc, repayments])
        for c in (amount, days, months, collateral):
            c.input(clear_quote, [], [calc, repayments])
        lev.change(do_improve, [lev, st], imp)
        bills_btn.click(do_bills, [bills, st], ep)
        area.change(do_area, [area, st], det)
        mdd.change(do_measure, [mdd, st], mcard)
        t1.change(lambda a, b, c, s: ask_level(1, a, b, c, s), [t1, t2, t3, st], [t2, t3, qs])
        t2.change(lambda a, b, c, s: ask_level(2, a, b, c, s), [t1, t2, t3, st], [t2, t3, qs])
        t3.change(lambda a, b, c, s: ask_level(3, a, b, c, s), [t1, t2, t3, st], [t2, t3, qs])
        qs.select(lambda q, h, s: ask_pick(q, h, s), [qs, chat, st], [chat, opts])
        opts.select(lambda q, h, s: ask_pick(q, h, s), [opts, chat, st], [chat, opts])
        txt.submit(ask_typed, [txt, chat, st], [chat, opts, txt])

        filt = [f_tier, f_conf, f_sec, f_size, f_min, f_grit, f_txt]
        for c in filt:
            c.change(lambda *x: apps_view(*x[:-2], 1, x[-1]), filt + [page, st], [summ, apps, page])
        page.submit(apps_view, filt + [page, st], [summ, apps, page])
        exp_btn.click(apps_export, filt + [st], exp_file)
        vsave.click(view_save, [vname] + filt + [st], [st, views])
        views.change(view_load, [views, st], filt)
        apps.select(pick_firm, [apps, st], [fid, fdet, ltabs])
        fopen.click(firm_html, [fid, st], fdet)
        note_btn.click(add_note, [fid, note_txt, note_who, st], [st, fdet, note_txt])

        own4 = [hdr, standing, borrow, rchart, calc, repayments]
        prev_btn.click(pol_preview, [ptab, st], [pmsg, rtab, rbuild, pimpact])
        save_btn.click(pol_save, [ptab, why, who, st], [st, ptab, vlog, pmsg] + own4).then(apps_view, filt + [page, st], [summ, apps, page])
        reset_btn.click(pol_reset, [st], [st, ptab, vlog, rules, pmsg] + own4).then(apps_view, filt + [page, st], [summ, apps, page])
        r_add.click(rule_add, [r_set, r_val, r_tier, r_tenor, r_sec, r_size, r_secd, r_reg, r_from, r_note, st], [st, rules, vlog, pmsg] + own4).then(
            apps_view, filt + [page, st], [summ, apps, page])
        r_rm.click(rule_remove, [r_no, st], [st, rules, vlog, pmsg] + own4).then(apps_view, filt + [page, st], [summ, apps, page])

        conf.release(val_ml, [conf, bh], [mlh, rq, g5])
        bh.change(val_ml, [conf, bh], [mlh, rq, g5])
        demo.load(apps_view, filt + [page, st], [summ, apps, page])
    return demo


def launch(share=True, **kw):
    demo = build_ui()
    theme = gr.themes.Soft(primary_hue=gr.themes.colors.teal, font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    # The original CSS uses light surfaces. Select the matching native Gradio mode,
    # without overriding its colours or hover states.
    kw.setdefault("js", """() => {
        const url = new URL(window.location.href);
        if (!url.searchParams.has('__theme')) {
            url.searchParams.set('__theme', 'light');
            window.location.replace(url.toString());
        }
    }""")
    kw.setdefault("inline", False)
    demo.queue(default_concurrency_limit=4)
    demo.launch(share=share, css=ui.CSS, theme=theme, show_error=True, **kw)
    if demo.share_url:
        # The tunnel is already running. Serve the installed frontend assets through
        # it instead of the CDN template, whose component CSS can fail to preload.
        demo.share = False
    return demo


if __name__ == "__main__":
    launch(share=False, server_name="0.0.0.0", server_port=7860)
