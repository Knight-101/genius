"""Ask Genius: answers from the business's own books and Genius's own numbers, with no language model and no API.

How it works
  1. A question bank is generated from the assessment: every number Genius shows has questions behind it (what it is, how it is worked
     out, my value, why it is rated so, how I compare, how to improve it, what if it improves, what data it needs), plus questions on
     tiers, knock-outs, Grit, peers, borrowing, fees, customers, suppliers, cash and data. Each question sits in a topic tree.
  2. A menu question always has an answer: its answer function is wired to the engine and tested over the whole bank.
  3. A typed question is matched to the bank: words are normalised, misspellings corrected against the bank's vocabulary, synonyms
     expanded, and names of customers, suppliers and measures picked out. Matching is by keyword overlap (BM25). A confident match is
     answered; a close one gets "Did you mean ...?"; otherwise Genius says it cannot tell and shows the nearest topics.
  4. Numbers in answers come only from the assessment. Hidden Grit parameters and lender-only checks are never revealed."""
from __future__ import annotations

import difflib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import fmt

SYN = {
    "dso": "days sales outstanding collect", "receivable": "owed customers", "receivables": "owed customers", "debtors": "owed customers",
    "creditors": "suppliers owe", "payables": "suppliers owe", "overdue": "late overdue past due", "late": "late overdue",
    "loan": "borrow credit limit", "borrow": "borrow credit limit loan", "credit": "credit borrow", "limit": "limit borrow how much",
    "rate": "rate interest price", "interest": "interest rate", "apr": "apr annual percentage rate fee", "fees": "fee processing",
    "score": "score genius", "rating": "score tier rating", "tier": "tier band", "grade": "tier", "safe": "tier", "risky": "tier",
    "cash": "cash bank liquidity", "runway": "cash runway days", "money": "cash", "profit": "profit margin", "margin": "margin profit",
    "sales": "sales revenue turnover", "revenue": "sales revenue", "turnover": "sales turnover", "customer": "customer customers",
    "client": "customer", "clients": "customer customers", "supplier": "supplier suppliers", "vendor": "supplier", "vendors": "supplier suppliers",
    "grit": "grit resilience bounce", "resilience": "grit", "bounce": "grit recover", "peers": "peers compare similar", "compare": "compare peers",
    "competitors": "peers", "improve": "improve better raise increase", "better": "improve", "increase": "improve", "fix": "improve",
    "why": "why reason", "how": "how", "what": "what", "msme": "size msme micro small medium", "size": "size msme",
    "drawings": "drawings owner withdraw", "owner": "owner drawings", "debt": "debt borrowings loan", "emi": "loan repayment instalment",
    "stock": "stock inventory", "inventory": "stock inventory", "gst": "statutory tax", "statutory": "statutory dues", "pf": "statutory",
    "43b": "section 43b micro small suppliers", "udyam": "msme size udyam", "knockout": "knock out", "red": "knock out", "flag": "knock out",
    "shock": "shock absorber line", "standby": "shock absorber line", "invoice": "invoice financing", "factoring": "invoice financing",
    "expiry": "expiry renewal valid", "renew": "renewal expiry", "valid": "valid expiry", "collateral": "security collateral secured",
    "security": "security collateral", "privacy": "privacy consent data", "consent": "consent privacy",
    "others": "peers compare", "other": "peers", "similar": "peers", "expires": "expiry", "expire": "expiry", "expired": "expiry",
    "run": "runway squeeze", "out": "", "y": "why", "wat": "what", "hw": "how", "u": "you", "ur": "my", "pls": "", "plz": "",
}
STOP = set("a an the of to for in on at is are was were be my me i do does did can could should would will what's whats it its this that "
           "and or with by from about please tell show give much many".split())


def toks(s: str) -> list:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9%\. ]", " ", s)
    s = re.sub(r"\by\b", "why", s); s = re.sub(r"\bu\b", "you", s); s = re.sub(r"\bur\b", "your", s)
    return [w for w in s.split() if w not in STOP and len(w) > 1]


@dataclass
class Q:
    id: str
    path: list
    text: str
    kind: str
    arg: object = None
    keywords: str = ""


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)
    followups: list = field(default_factory=list)
    status: str = "ok"            # ok | did_you_mean | cant_tell
    understood: str | None = None
    options: list = field(default_factory=list)


class Genie:
    def __init__(self, a, db, policy, books=None, owner_view: bool = True):
        self.a, self.db, self.pol, self.b = a, db, policy, books
        self.owner = owner_view
        self.cur = "INR"
        self.bank = self._bank()
        self.by_id = {q.id: q for q in self.bank}
        self._index()

    # ---------------------------------------------------------------------------------------------- the question bank
    def _bank(self) -> list:
        a = self.a
        B = []
        add = lambda *x: B.append(Q(*x))
        T = "My credit standing"
        add("score_what", [T, "Genius Score"], "What is my Genius Score?", "score", None, "score number genius")
        add("score_why", [T, "Genius Score"], "Why is my score what it is?", "score_why", None, "why score low high drivers reason")
        add("score_how", [T, "Genius Score"], "How is the Genius Score worked out?", "score_how", None, "how calculated method weights areas")
        add("tier_what", [T, "Tier"], "What tier am I in and what does it mean?", "tier", None, "tier safe moderate watch risky band")
        add("tier_next", [T, "Tier"], "What would move me up a tier?", "tier_next", None, "move up next tier improve reach safe")
        add("tier_prob", [T, "Tier"], "How likely is a cash squeeze for firms like mine?", "prob", None, "probability stress squeeze default chance")
        for k in ("K1", "K2", "K3", "K4", "K5"):
            add(f"ko_{k}", [T, "Knock-out checks"], f"Does the '{_ko_name(k)}' check apply to me?", "ko", k, "knock out red flag cap " + _ko_name(k))
        add("ko_all", [T, "Knock-out checks"], "Are any red flags capping my score?", "ko_all", None, "knock out red flag cap")
        for area in C.AREA_LIST:
            add(f"area_{area}", [T, "Areas", area], f"What is my {area.lower()} score?", "area", area, area.lower() + " area score")
            add(f"area_why_{area}", [T, "Areas", area], f"Why is my {area.lower()} score at this level?", "area_why", area, area.lower() + " why low high")
            add(f"area_imp_{area}", [T, "Areas", area], f"How do I improve {area.lower()}?", "area_imp", area, area.lower() + " improve")
        M = "Every measure"
        for mid in C.MEASURES:
            m = C.M[mid]; nm = m["name"]; ar = m["area"]
            kw = f"{nm.lower()} {mid.lower()} {m['desc'].lower()}"
            add(f"m_what_{mid}", [M, ar, nm], f"What is {nm.lower()}?", "m_what", mid, kw + " what mean definition")
            add(f"m_val_{mid}", [M, ar, nm], f"What is my {nm.lower()}?", "m_val", mid, kw + " my value number")
            add(f"m_how_{mid}", [M, ar, nm], f"How is {nm.lower()} worked out?", "m_how", mid, kw + " how calculated formula worked out")
            add(f"m_why_{mid}", [M, ar, nm], f"Why is my {nm.lower()} rated the way it is?", "m_why", mid, kw + " why rated rating score")
            add(f"m_peer_{mid}", [M, ar, nm], f"How does my {nm.lower()} compare with peers?", "m_peer", mid, kw + " compare peers median")
            add(f"m_imp_{mid}", [M, ar, nm], f"How do I improve my {nm.lower()}?", "m_imp", mid, kw + " improve better fix")
            add(f"m_trend_{mid}", [M, ar, nm], f"How has my {nm.lower()} changed over the last year?", "m_trend", mid, kw + " trend change last year history")
            add(f"m_data_{mid}", [M, ar, nm], f"What data does {nm.lower()} use?", "m_data", mid, kw + " data needs uses missing")
        G_ = "My Grit"
        add("grit_what", [G_, "About Grit"], "What is Grit?", "grit_what", None, "grit what mean resilience bounce back")
        add("grit_my", [G_, "My Grit"], "What is my Grit?", "grit_my", None, "grit my score band")
        add("grit_why", [G_, "My Grit"], "Why is my Grit at this level?", "grit_why", None, "grit why reason setbacks")
        add("grit_setbacks", [G_, "Setbacks"], "Which setbacks did Genius find in my books?", "grit_eps", None, "grit setbacks episodes dips found")
        add("grit_imp", [G_, "My Grit"], "How do I improve my Grit?", "grit_imp", None, "grit improve")
        add("grit_use", [G_, "About Grit"], "How does Grit affect my borrowing?", "grit_use", None, "grit borrowing loan terms limit")
        add("grit_noshock", [G_, "About Grit"], "I never had a setback. Am I penalised?", "grit_noshock", None, "no setback never trouble penalised capacity")
        P_ = "Compared with peers"
        add("peer_group", [P_, "My peer group"], "Who am I compared with?", "peer_group", None, "peer group compared who similar")
        add("peer_rank", [P_, "My position"], "Where do I rank among peers?", "peer_rank", None, "rank position percentile peers")
        add("peer_behind", [P_, "Gaps and strengths"], "Where am I furthest behind peers?", "peer_behind", None, "behind weakest peers gaps")
        add("peer_ahead", [P_, "Gaps and strengths"], "Where am I furthest ahead of peers?", "peer_ahead", None, "ahead strongest strengths peers")
        add("peer_rate", [P_, "My position"], "How does my rate compare with peers?", "peer_rate", None, "rate compare peers cheaper")
        Bo = "Borrowing"
        for prod, lab in (("short", "Short-term credit"), ("shock", "Shock-absorber line"), ("medium", "Medium-term loan"), ("invoice", "Invoice financing"), ("long", "Long-term loan")):
            add(f"b_lim_{prod}", [Bo, lab], f"How much {lab.lower()} can I get?", "b_lim", prod,
                lab.lower() + (" how much limit borrow loan credit can i get eligible" if prod == "short" else " limit"))
            add(f"b_why_{prod}", [Bo, lab], f"How is my {lab.lower()} worked out?", "b_why", prod, lab.lower() + " why how limit calculated")
            add(f"b_rate_{prod}", [Bo, lab], f"What rate would I pay on {lab.lower()}?", "b_rate", prod,
                lab.lower() + (" rate interest price will i get pay charged" if prod == "short" else " rate"))
        add("b_rate_build", [Bo, "Rates and fees"], "How is my interest rate worked out?", "rate_build", None, "rate how worked out cost funds margin expected loss")
        add("b_apr", [Bo, "Rates and fees"], "What is the annual percentage rate on a typical drawdown?", "apr", None, "apr annual percentage rate fee kfs")
        add("b_fee", [Bo, "Rates and fees"], "What fees are there?", "fees", None, "fee processing reservation charges")
        add("b_repay", [Bo, "Repayment"], "How would I repay?", "repay", None, "repay repayment collections sweep instalment")
        add("b_expiry", [Bo, "Repayment"], "How long is my limit valid?", "expiry", None, "expiry valid renewal review")
        add("b_security", [Bo, "Security"], "Do I need to give security or collateral?", "security", None, "security collateral guarantee secured")
        add("b_who", [Bo, "Who decides"], "Who decides my loan?", "who", None, "who decides lender approval kfs")
        Im = "Improve my rate"
        add("imp_best", [Im, "Best moves"], "What are the best moves to improve my rate?", "imp_best", None, "improve rate best moves levers")
        add("imp_collect", [Im, "Collect overdue"], "What if I collect my overdue invoices?", "imp_collect", None, "what if collect overdue receivables")
        add("imp_conc", [Im, "Customer concentration"], "What if my biggest customer were below 20% of sales?", "imp_conc", None, "what if top customer concentration")
        add("imp_draw", [Im, "Owner drawings"], "What if I kept drawings under half of cash flow?", "imp_draw", None, "what if drawings owner")
        Cu = "Customers and suppliers"
        add("cust_owe", [Cu, "Who owes me"], "Who owes me the most?", "cust_owe", None, "who owes most customers receivables")
        add("cust_overdue", [Cu, "Who owes me"], "Which customers are overdue?", "cust_overdue", None, "overdue customers late list")
        add("cust_top", [Cu, "Concentration"], "Who is my biggest customer?", "cust_top", None, "biggest customer top largest")
        add("sup_top", [Cu, "Suppliers"], "Who is my biggest supplier?", "sup_top", None, "biggest supplier top largest")
        add("sup_43b", [Cu, "Suppliers"], "Am I paying micro and small suppliers on time (Section 43B(h))?", "sup_43b", None, "43b micro small suppliers 45 days msme")
        for party in self._parties("receivable")[:40]:
            add(f"party_{party}", [Cu, "Customers", party], f"How does {party} pay?", "party", party, f"{party.lower()} customer pay late owe")
            add(f"partyif_{party}", [Cu, "Customers", party], f"What if {party} pays this month?", "party_if", party, f"{party.lower()} what if pays")
        for party in self._parties("payable")[:20]:
            add(f"sparty_{party}", [Cu, "Suppliers", party], f"How do I pay {party}?", "sparty", party, f"{party.lower()} supplier pay")
        Ca = "Cash and early warning"
        add("cash_now", [Ca, "Cash"], "How much cash cover do I have?", "cash_now", None, "cash cover weeks bank")
        add("cash_low", [Ca, "Early warning"], "Is a cash squeeze coming?", "cash_warn", None, "cash squeeze coming early warning forecast low point run out short next weeks")
        add("cash_runway", [Ca, "Cash"], "How long would my cash last?", "cash_runway", None, "runway cash last days")
        Da = "My data"
        add("data_quality", [Da, "Data quality"], "How good is the data I uploaded?", "data_q", None, "data quality confidence checks")
        add("data_missing", [Da, "Data quality"], "What is missing from my books?", "data_missing", None, "missing data fields upload")
        add("data_size", [Da, "Size and sector"], "Why am I in this MSME size band?", "data_size", None, "msme size micro small medium band why udyam")
        add("data_sector", [Da, "Size and sector"], "Which sector am I in?", "data_sector", None, "sector industry")
        Ab = "About Genius"
        add("about_how", [Ab, "How it works"], "How does Genius work?", "about_how", None, "how genius works method")
        add("about_privacy", [Ab, "Privacy"], "Who can see my data?", "about_privacy", None, "privacy consent data share who see")
        add("about_synthetic", [Ab, "How it works"], "Are the peer figures real?", "about_synth", None, "peers real synthetic prototype")
        add("about_ml", [Ab, "How it works"], "Does Genius use AI or machine learning?", "about_ml", None, "ai machine learning ml model")
        return B

    def _parties(self, kind):
        if self.b is None:
            return []
        try:
            inv = self.b.invoices(kind)
            inv = inv[inv["amount"] > 0]
            return list(inv.groupby("party")["amount"].sum().sort_values(ascending=False).index)
        except Exception:
            return []

    def tree(self) -> dict:
        t = {}
        for q in self.bank:
            node = t
            for p in q.path:
                node = node.setdefault(p, {})
            node.setdefault("_q", []).append(q.id)
        return t

    # ---------------------------------------------------------------------------------------------- matching typed questions
    def _index(self):
        self.docs = []
        df = Counter()
        for q in self.bank:
            words = toks(q.text + " " + q.keywords + " " + " ".join(q.path))
            self.docs.append(words)
            df.update(set(words))
        self.df = df
        self.N = len(self.bank)
        self.avgdl = np.mean([len(d) for d in self.docs])
        self.vocab = sorted(df)

    def _expand(self, words):
        out = []
        for w in words:
            if w not in self.df and len(w) > 3:
                c = difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.86)
                if c:
                    w = c[0]
            out.append(w)
            if w in SYN:
                out.extend(SYN[w].split())
        return out

    def match(self, text: str, k: int = 3):
        qw = self._expand(toks(text))
        scores = np.zeros(self.N)
        for w in qw:
            if w not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
            for i, d in enumerate(self.docs):
                tf = d.count(w)
                if tf:
                    scores[i] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(d) / self.avgdl))
        # named things in the question pull hard towards their questions
        low = text.lower()
        for i, q in enumerate(self.bank):
            if q.kind in ("party", "party_if", "sparty") and q.arg.lower() in low:
                scores[i] += 8
            if q.kind.startswith("m_") and C.M[q.arg]["name"].lower() in low:
                scores[i] += 6
        mine = bool(re.search(r"\b(my|mine|i|me|we|our)\b", low))
        exp = set(self._expand(toks(text)))
        why = "why" in exp
        for i, q in enumerate(self.bank):
            if q.kind.startswith("area") and q.arg.lower()[:8] in " ".join(exp) + " " + low:
                scores[i] += 4 + (2 if why and q.kind == "area_why" else 0)
            if why and ("why" in q.kind or q.kind in ("grit_why", "score_why")):
                scores[i] += 1.5
        specific = any(ar.lower()[:8] in " ".join(exp) + " " + low for ar in C.AREA_LIST) or \
            any(C.M[m]["name"].lower() in low for m in C.MEASURES)
        if specific:
            for i, q in enumerate(self.bank):
                if q.kind in ("score", "score_why", "score_how", "tier", "tier_next"):
                    scores[i] -= 8
        for i, q in enumerate(self.bank):
            if mine and q.kind in ("m_val", "area", "score", "grit_my", "b_lim", "cash_now"):
                scores[i] += 1.5
            if not mine and q.kind in ("m_what", "grit_what"):
                scores[i] += 1.0
            if q.kind in ("party", "party_if", "sparty"):
                name_toks = [w for w in toks(q.arg) if len(w) > 3]
                if not any(w in low for w in name_toks):
                    scores[i] -= 6
        if re.search(r"\bwhat if\b|\bif i\b", low):
            for i, q in enumerate(self.bank):
                if q.kind in ("imp_collect", "imp_conc", "imp_draw", "party_if"):
                    scores[i] += 2
        order = np.argsort(-scores)[:k]
        return [(self.bank[i], float(scores[i])) for i in order]

    def ask(self, text: str) -> Answer:
        m = re.search(r"what if (.+?) (?:pays|paid|pay)\b", (text or "").lower())
        if m:
            name = m.group(1).strip()
            parties = self._parties("receivable")
            hit = difflib.get_close_matches(name, [p.lower() for p in parties], n=1, cutoff=0.6)
            if not hit and name not in ("i", "we", "my customers", "customers", "everyone", "all customers"):
                return Answer(f"I can't find a customer called '{m.group(1).strip()}' in your books.", status="cant_tell",
                              followups=["Who owes me the most?", "Which customers are overdue?"])
            if hit:
                real = next(p for p in parties if p.lower() == hit[0])
                ans = self.answer(f"partyif_{real}")
                ans.understood = f"What if {real} pays this month?"
                return ans
        if not text or not text.strip():
            return Answer("Ask me anything about your credit standing, Grit, peers, borrowing, customers, cash or your data.", status="cant_tell",
                          followups=["What is my Genius Score?", "How much short-term credit can I get?", "Who owes me the most?"])
        if text.strip() in self.by_id:
            return self.answer(text.strip())
        top = self.match(text)
        (q1, s1), (q2, s2) = top[0], top[1]
        content = [w for w in self._expand(toks(text)) if w in self.df and self.df[w] < 0.25 * self.N and w in self.docs[self.bank.index(q1)]]
        if s1 < 4.0 or not content:
            nearest = sorted({q.path[0] for q, s in top if s > 0})
            return Answer("I can't tell yet what you're asking. I answer from your books and Genius's own numbers: your score and tier, every measure, "
                          "Grit, peers, borrowing and rates, customers and suppliers, cash, and your data." +
                          (f" Closest topics: {', '.join(nearest)}." if nearest else ""), status="cant_tell",
                          followups=[q.text for q, _ in top if _ > 0][:3] or ["What is my Genius Score?"])
        if s2 > 0.95 * s1 and q1.kind != q2.kind and q1.arg != q2.arg:
            return Answer("Did you mean one of these?", status="did_you_mean", options=[(q.id, q.text) for q, _ in top])
        ans = self.answer(q1.id)
        ans.understood = q1.text
        return ans

    # ---------------------------------------------------------------------------------------------- answers
    def answer(self, qid: str) -> Answer:
        q = self.by_id[qid]
        fn = getattr(self, "_" + q.kind)
        return fn(q.arg) if q.arg is not None else fn()

    def money(self, x):
        return fmt.money(x, self.cur)

    def _val(self, mid, v=None):
        v = self.a.m(mid) if v is None else v
        return "not available" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.value(v, C.M[mid]["unit"] if C.M[mid]["unit"] != "index" else "x", self.cur).replace("x", "") if C.M[mid]["unit"] == "index" else fmt.value(v, C.M[mid]["unit"], self.cur)

    def _score(self):
        a = self.a
        if a.score is None:
            return Answer("No Genius Score is published for these books: " + self._no_score_reason(), ["Upload check"])
        return Answer(f"Your Genius Score is {a.score:.0f} out of 100, which puts you in the {a.tier} tier. About {100 * a.p_stress:.0f} in 100 businesses "
                      f"with a similar score hit a cash squeeze within 90 days in the training years.", ["Genius Score, learned model version 5"],
                      ["Why is my score what it is?", "What would move me up a tier?", "How is the Genius Score worked out?"])

    def _no_score_reason(self):
        a = self.a
        if a.gs["ko"]["K5"][0, -1]:
            return scorecard_ko_text("K5")
        n = int(np.sum(~np.isnan(a.f["rev"][0])))
        if n < 12:
            return f"there are {n} months of history; a score needs 12."
        return "too few areas could be rated from the data in the file (liquidity and debt service must both be rated)."

    def _score_why(self):
        d = self.a.drivers
        pos = "; ".join(f"{r['name'].lower()} is strong ({self._val(r['id'], r['value'])}, rated {r['rating']:.0f})" for r in d["positive"])
        neg = "; ".join(f"{r['name'].lower()} is weak ({self._val(r['id'], r['value'])}, rated {r['rating']:.0f})" for r in d["negative"])
        return Answer(f"Helping most: {pos or 'nothing stands out'}. Holding you back: {neg or 'nothing stands out'}. Each measure is rated 0-100 against fixed "
                      "ranges; the displayed drivers use the learned credit coefficients. Data quality is a publication gate.", ["Details tab: every measure"],
                      ["What would move me up a tier?", "Where am I furthest behind peers?"])

    def _score_how(self):
        return Answer("Genius rates financial measures against explicit ranges; a monotone model learns their risk coefficients from synthetic defaults. "
                      "Grit remains a separate resilience score and is not blended into Genius. The score maps financial log-odds to 0–100, with financial caps and data-quality gates. "
                      "A separate, assumed recovery policy can reduce lender margin for completed recovery evidence; it does not change Genius or calibrated default probability. "
                      "See Details for coefficients and synthetic held-out results.", ["Learned factors and validation"],
                      ["What tier am I in and what does it mean?", "How does Grit affect my borrowing?"])

    def _tier(self):
        a, c = self.a, self.db.cal.cutoffs
        if a.tier is None:
            return self._score()
        ranges = f"Safe from {c['Safe']:.0f}, Moderate from {c['Moderate']:.0f}, Watch from {c['Watch']:.0f}, Risky below that"
        return Answer(f"You are {a.tier}. Tiers are set by how often similar-scoring firms hit a cash squeeze within 90 days: under 5% is Safe, 5-12% Moderate, "
                      f"12-25% Watch, over 25% Risky. On today's calibration that means {ranges}. Your score {a.score:.0f} corresponds to about "
                      f"{100 * a.p_stress:.0f}%.", ["Calibration on synthetic years 1-5"], ["What would move me up a tier?"])

    def _tier_next(self):
        a, c = self.a, self.db.cal.cutoffs
        if a.tier in (None, "Safe"):
            return Answer("You are already in the top tier." if a.tier == "Safe" else self._score().text, followups=["How do I improve my Grit?"])
        nxt = {"Risky": "Watch", "Watch": "Moderate", "Moderate": "Safe"}[a.tier]
        gap = c[nxt] - a.score
        neg = self.a.drivers["negative"]
        tips = " and ".join(f"{r['name'].lower()}" for r in neg)
        return Answer(f"You need about {gap:.0f} more points to reach {nxt} (from {c[nxt]:.0f}). The biggest drags on your score are {tips}. "
                      "Improve your rate shows the effect of each change, re-scored through the same engine.", ["Improve your rate tab"],
                      ["What are the best moves to improve my rate?", "What if I collect my overdue invoices?"])

    def _prob(self):
        a = self.a
        return Answer(f"For businesses with a score like yours, about {100 * a.p_stress:.0f} in 100 hit a cash squeeze within 90 days and about "
                      f"{100 * a.p_default:.1f} in 100 fell 90 days behind on a loan or trade payment within a year, in the synthetic training years. "
                      "These are prototype figures; real ones come from Zoho customers' repayment history.", ["Calibration"], ["What tier am I in and what does it mean?"])

    def _ko(self, k):
        on = bool(self.a.gs["ko"][k][0, -1])
        from .scorecard import CAPS
        cap = f" and caps your score at {CAPS[k]}" if k in CAPS else ""
        how = {"K1": "Build cash or unused cash-credit to cover at least two weeks of total costs.", "K2": "Bring operating profit above interest plus repayments, or restructure a loan.",
               "K3": "Collect or provide for invoices more than 90 days overdue.", "K4": "Clear statutory dues so no more than about 4 months build up.",
               "K5": "Fix entries that do not balance and check customers whose money flows straight back out."}[k]
        return Answer(f"{scorecard_ko_text(k)}. This check {'applies to you' + cap if on else 'does not apply to you'}. {how if on else ''}".strip(),
                      ["Knock-out rules, specification 6.3"], ["Are any red flags capping my score?"])

    def _ko_all(self):
        on = [k for k in ("K1", "K2", "K3", "K4", "K5") if self.a.gs["ko"][k][0, -1]]
        if not on:
            return Answer("No red flags: none of the knock-out checks applies, so nothing caps your score.", ["Knock-out rules"])
        return Answer("These red flags apply: " + "; ".join(scorecard_ko_text(k) for k in on) + ".", ["Knock-out rules"], [f"Does the '{_ko_name(k)}' check apply to me?" for k in on])

    def _area(self, area):
        a = self.a
        s = float(a.gs["areas"][area][0, -1])
        pm, pp = a.peers["areas_median"][area], a.peers["areas_pct"][area]
        if np.isnan(s):
            return Answer(f"Your {area.lower()} area could not be rated: fewer than half its measures are available from these books.", ["Details tab"])
        return Answer(f"Your {area.lower()} score is {s:.0f} out of 100 ({C.rating_band(s).lower()}), against a peer median of {pm:.0f}; better than "
                      f"{100 * pp:.0f}% of peers. It carries {C.AREAS[area]}% of your Genius Score.", ["Details tab", "Peers tab"],
                      [f"Why is my {area.lower()} score at this level?", f"How do I improve {area.lower()}?"])

    def _area_why(self, area):
        a = self.a
        rows = [(m, a.r(m)) for m in C.BY_AREA[area] if a.r(m) is not None]
        rows.sort(key=lambda x: x[1])
        lo = "; ".join(f"{C.M[m]['name'].lower()} {self._val(m)} (rated {r:.0f})" for m, r in rows[:2])
        hi = "; ".join(f"{C.M[m]['name'].lower()} {self._val(m)} (rated {r:.0f})" for m, r in rows[-2:][::-1])
        return Answer(f"In {area.lower()}, the weakest measures are {lo}; the strongest are {hi}. The area score is the average of its rated measures.",
                      ["Details tab"], [f"How do I improve {area.lower()}?"])

    def _area_imp(self, area):
        a = self.a
        rows = sorted([(m, a.r(m)) for m in C.BY_AREA[area] if a.r(m) is not None], key=lambda x: x[1])[:2]
        tips = " ".join(IMPROVE.get(m, "") for m, _ in rows)
        return Answer(f"Start with the weakest measures in {area.lower()}: {', '.join(C.M[m]['name'].lower() for m, _ in rows)}. {tips}", ["Details tab"],
                      [f"How do I improve my {C.M[m]['name'].lower()}?" for m, _ in rows])

    def _m_what(self, mid):
        m = C.M[mid]
        b = C.bounds_for(mid, self.a.family)
        rng = "" if b is None else f" It is rated 20, 40, 60 and 80 at {', '.join(_fmt_b(x) for x in b)} ({'higher' if m['dir'] == 'H' else 'lower'} is better)."
        return Answer(f"{m['name']}: {m['desc'].lower()}. It is part of the {m['area'].lower()} area.{rng}" + (" This is a published bank norm." if m["bank_norm"] else ""),
                      ["Measure library, specification section 5"], [f"What is my {m['name'].lower()}?", f"How is {m['name'].lower()} worked out?"])

    def _m_val(self, mid):
        a = self.a
        v, r = a.m(mid), a.r(mid)
        if v is None:
            return self._m_data(mid)
        return Answer(f"Your {C.M[mid]['name'].lower()} is {self._val(mid)}" + (f", rated {r:.0f} out of 100 ({C.rating_band(r).lower()})." if r is not None else ", not rated for your sector."),
                      [f"Your books to {a.months[-1].strftime('%b %Y')}"], [f"How does my {C.M[mid]['name'].lower()} compare with peers?", f"How do I improve my {C.M[mid]['name'].lower()}?"])

    def _m_how(self, mid):
        return Answer(f"{C.M[mid]['name']}: {FORMULA.get(mid, C.M[mid]['desc'])}. Flows are totalled over the last 12 months and balances read at the month end. "
                      f"Your value: {self._val(mid)}.", ["Measure library"], [f"Why is my {C.M[mid]['name'].lower()} rated the way it is?"])

    def _m_why(self, mid):
        a = self.a
        v, r = a.m(mid), a.r(mid)
        b = C.bounds_for(mid, a.family)
        if v is None or r is None or b is None:
            return self._m_val(mid)
        nxt = next((x for x, s in zip(b, (20, 40, 60, 80)) if s > r), None)
        step = "" if nxt is None else f" It would reach the next band at {_fmt_b(nxt)}."
        return Answer(f"Your {C.M[mid]['name'].lower()} is {self._val(mid)}. Against the ranges {', '.join(_fmt_b(x) for x in b)} (scores 20/40/60/80, "
                      f"{'higher' if C.M[mid]['dir'] == 'H' else 'lower'} is better) that rates {r:.0f}.{step}", ["Measure library"], [f"How do I improve my {C.M[mid]['name'].lower()}?"])

    def _m_peer(self, mid):
        p = self.a.peers["measures"][mid]
        if np.isnan(p["median"]):
            return Answer("Peers don't have enough data on this measure.", ["Peers tab"])
        pct = "" if np.isnan(p["pct"]) else f" You are better than {100 * p['pct']:.0f}% of them."
        return Answer(f"Your {C.M[mid]['name'].lower()} is {self._val(mid)}. Your {self.a.peers['n']} peers: median {self._val(mid, p['median'])}, middle half "
                      f"{self._val(mid, p['p25'])} to {self._val(mid, p['p75'])}.{pct}", [f"Peer group: {self.a.peers['key']}"], [f"How do I improve my {C.M[mid]['name'].lower()}?"])

    def _m_imp(self, mid):
        return Answer(IMPROVE.get(mid, "Look at the entries behind this measure on the Details tab.") + " Improve your rate re-scores the business with a change applied.",
                      ["Genius guidance"], ["What are the best moves to improve my rate?"])

    def _m_trend(self, mid):
        a = self.a
        s = a.meas[mid][0]
        vals = [(a.months[j], s[j]) for j in range(len(s) - 1, max(-1, len(s) - 13), -3) if not np.isnan(s[j])][::-1]
        if len(vals) < 2:
            return Answer("Not enough history to show a trend.", ["Details tab"])
        txt = ", ".join(f"{p.strftime('%b %Y')}: {self._val(mid, float(v))}" for p, v in vals)
        return Answer(f"{C.M[mid]['name']} over the last year: {txt}.", ["Your books"], [f"Why is my {C.M[mid]['name'].lower()} rated the way it is?"])

    def _m_data(self, mid):
        need = DATA_NEED.get(mid, "your ledger")
        v = self.a.m(mid)
        state = "available from your books" if v is not None else "not available from this file"
        return Answer(f"{C.M[mid]['name']} uses {need}. It is {state}.", ["Upload check"], ["What is missing from my books?"])

    def _grit_what(self):
        return Answer("Grit measures how a business behaves after a setback: how much of the lost ground came back, whether it stayed back, whether the "
                      "recovery came from the business itself rather than new borrowing, how the owner responded, and whether the books stayed clean. "
                      "It never marks down a bad patch, only a failure to recover. With few setbacks to judge, it leans on capacity: cash buffers, "
                      "customer spread, paying on time and a stress test.", ["Grit method"], ["What is my Grit?", "How does Grit affect my borrowing?"])

    def _grit_my(self):
        g = self.a.grit
        s = float(g["score"][0, -1]) if not np.isnan(g["score"][0, -1]) else None
        if s is None:
            return Answer("Grit needs at least 12 months of history.", ["Grit"])
        lo, hi = float(g["lo"][0, -1]), float(g["hi"][0, -1])
        n = int(g["n"][0, -1])
        return Answer(f"Your Grit is {s:.0f} ({C.grit_band(s)}), likely range {lo:.0f}-{hi:.0f}. It is based on {n} setback{'s' if n != 1 else ''} judged so far "
                      f"and your capacity ({float(g['capacity'][0, -1]):.0f}).", ["Grit"], ["Why is my Grit at this level?", "Which setbacks did Genius find in my books?"])

    def _grit_why(self):
        g = self.a.grit
        n = int(g["n"][0, -1]); dem = g["demonstrated"][0, -1]; cap = g["capacity"][0, -1]
        part = f"Recovery from past setbacks scores {dem:.0f}; " if not np.isnan(dem) else "No setback has been judged yet, so Grit rests on capacity; "
        return Answer(part + f"capacity scores {cap:.0f}. The more setbacks are judged, the more weight goes to how you actually recovered.", ["Grit"],
                      ["How do I improve my Grit?"])

    def _grit_eps(self):
        eps = [e for e in (self.a.grit["episodes"] or [])]
        if not eps:
            return Answer("No setbacks were found in your books: Genius has not seen a sharp drop in sales, operating profit and collections together, "
                          "after removing your seasonal pattern and what the whole sector went through.", ["Grit"], ["I never had a setback. Am I penalised?"])
        parts = []
        for e in eps[:5]:
            if e["kind"] == "weathered":
                parts.append(f"{e.get('start_m', '')}: your sector had a setback and you stayed steady (counted as a setback passed)")
            else:
                ex = ""
                if e.get("gap_regained") is not None:
                    ex = f", {100 * e['gap_regained']:.0f}% of the lost ground regained, {100 * e['funded_by_operations']:.0f}% funded by the business itself"
                parts.append(f"{e.get('start_m', '')} to {e.get('end_m', 'now')}: {e['outcome']}{ex}")
        return Answer("Setbacks found: " + "; ".join(parts) + ".", ["Grit"], ["Why is my Grit at this level?"])

    def _grit_imp(self):
        return Answer("Keep a cash buffer, spread sales across more customers, pay suppliers on time and keep drawings within what the business generates. "
                      "If a setback comes, recover from operations (collect faster, trim costs) rather than new borrowing: Grit credits that most.", ["Grit"])

    def _grit_use(self):
        r = self.a.pricing.get('short', {})
        discount = r.get('recovery_concession', 0) * 100
        return Answer("Genius and Grit are separate. A demo lender policy can offer up to 50 basis points off its margin for recent, completed recovery evidence, "
                      "scaled by recovery strength, evidence count and recency. Grit capacity is excluded to avoid counting financial factors twice. "
                      f"Your current short-term recovery concession is {discount:.1f} basis points. "
                      "This is an unvalidated policy assumption, not a learned reduction in default probability. No automatic amount bonus applies; repayment and financial checks still bind. "
                      "Enter an amount and duration on Borrow for the request-specific result.",
                      ["Separate Grit and demo lender policy"], ["How much shock-absorber line can I get?"])

    def _grit_noshock(self):
        return Answer("No. A business that never got into trouble is neither marked up nor down for that: its Grit rests on capacity (buffers, customer spread, "
                      "paying on time, a stress test). And when your whole sector has a setback and you stay steady, Genius counts that as a setback passed.", ["Grit"])

    def _peer_group(self):
        p = self.a.peers
        return Answer(f"You are compared with {p['n']} {p['key'].replace(' | ', ', ').lower()} businesses ({p['level']}), at {p['month']}. Groups need at "
                      "least 30 firms; your own business is never in its own group. In this prototype the peers are synthetic.", ["Peers tab"], ["Where do I rank among peers?"])

    def _peer_rank(self):
        p = self.a.peers
        if p["rank"] is None:
            return self._score()
        return Answer(f"You rank {p['rank']} of {p['n']} on Genius Score, better than {100 * p['score_pct']:.0f}% of peers (peer median {p['score_median']:.0f}).", ["Peers tab"],
                      ["Where am I furthest behind peers?", "Where am I furthest ahead of peers?"])

    def _gaps(self, worst=True):
        rows = [(m, v["pct"]) for m, v in self.a.peers["measures"].items() if not np.isnan(v["pct"]) and not m.startswith("DQ") and m != "IC1"]
        rows.sort(key=lambda x: x[1], reverse=not worst)
        return rows[:3]

    def _peer_behind(self):
        txt = "; ".join(f"{C.M[m]['name'].lower()} {self._val(m)} (peers {self._val(m, self.a.peers['measures'][m]['median'])}, better than {100 * p:.0f}%)" for m, p in self._gaps(True))
        return Answer("Furthest behind: " + txt + ".", ["Peers tab"], [f"How do I improve my {C.M[m]['name'].lower()}?" for m, _ in self._gaps(True)][:2])

    def _peer_ahead(self):
        txt = "; ".join(f"{C.M[m]['name'].lower()} {self._val(m)} (peers {self._val(m, self.a.peers['measures'][m]['median'])}, better than {100 * p:.0f}%)" for m, p in self._gaps(False))
        return Answer("Furthest ahead: " + txt + ".", ["Peers tab"])

    def _peer_rate(self):
        pr = self.a.pricing
        if not pr:
            return self._score()
        mine, peer = pr["short"]["rate"], pr["peer_median"]["rate"]
        return Answer(f"Your indicative short-term rate is {mine:.2f}% against about {peer:.2f}% for a peer-median business: {fmt.money((peer - mine) / 100 * 1e6, self.cur)} "
                      "a year less per ₹10 lakh." if mine < peer else f"Your indicative short-term rate is {mine:.2f}% against about {peer:.2f}% for a peer-median business.", ["Pricing"])

    def _b_lim(self, prod):
        L = self.a.limits
        if not L:
            return self._score()
        x = L[prod]
        if prod == "long":
            return Answer("Long-term loans (over 3 years) are referred to the partner lender, who appraises the project." + (" You qualify for a referral." if x.get("referral") else ""), ["Borrow tab"])
        if not x["available"]:
            return Answer(f"{PROD[prod]} is not available on these books. " + "; ".join(x["why"]) + ".", ["Borrow tab"], ["What would move me up a tier?"])
        return Answer(f"Up to {self.money(x['limit'])} of {PROD[prod].lower()}." + (f" Up to {x.get('max_days')} days per drawdown." if x.get("max_days") else "") +
                      (f" Over {x['tenor_months']} months, with debt cover after the loan of {x['cover_after']:.1f}x." if prod == "medium" else ""),
                      ["Borrow tab", f"Lender policy version {L.get('policy_version')}"], [f"How is my {PROD[prod].lower()} worked out?", f"What rate would I pay on {PROD[prod].lower()}?"])

    def _b_why(self, prod):
        L = self.a.limits
        if not L:
            return self._score()
        return Answer(f"{PROD[prod]}: " + "; ".join(L[prod]["why"]) + "." + (" " + " ".join(L["reasons"]) if L.get("reasons") and self.owner is False else ""), ["Lender policy"])

    def _b_rate(self, prod):
        pr = self.a.pricing
        key = {"short": "short", "shock": "short", "medium": "medium", "invoice": "invoice", "long": None}[prod]
        if key is None:
            return Answer("The partner lender sets the rate on long-term loans after appraising the project.", ["Borrow tab"])
        if not pr:
            return self._score()
        r = pr[key]
        return Answer(f"Indicative rate on {PROD[prod].lower()}: {r['rate']:.2f}% a year." + (" Secured, so the expected loss is lower." if key in ("medium", "invoice") else ""),
                      ["Pricing"], ["How is my interest rate worked out?"])

    def _rate_build(self):
        pr = self.a.pricing
        if not pr:
            return self._score()
        r = pr["short"]
        return Answer(f"{r['cost_of_funds']:.2f}% cost of funds + {r['operating_cost']:.2f}% operating cost + {r['capital_charge']:.2f}% capital charge + "
                      f"{r['expected_loss']:.2f}% expected loss ({r['pd']:.2f}% default probability x {r['lgd']:.0f}% loss if default) + {r['margin']:.2f}% margin = "
                      f"{r['rate']:.2f}%. The lender can change every input; the final rate is in its Key Fact Statement.", ["Lender policy"], ["What is the annual percentage rate on a typical drawdown?"])

    def _apr(self):
        pr = self.a.pricing
        if not pr:
            return self._score()
        x = self.pol.apr(15e5, pr["short"]["rate"], 75, self.a.info.get("policy_context"))
        return Answer(f"On ₹15 lakh for 75 days at {pr['short']['rate']:.2f}%: interest {self.money(x['interest'])}, fee {self.money(x['fee'])}, total "
                      f"{self.money(x['total'])}, an estimated effective annual cost of {x['apr']:.1f}%. {x['note']} This is a cost illustration, not a KFS APR or an approved amount.", ["Cash-flow cost illustration"], ["What fees are there?"])

    def _fees(self):
        p = self.pol
        return Answer(f"Processing fee {p.v('processing_fee')}% ({'once a year on the limit' if p.v('fee_mode') == 'annual_on_limit' else 'on each drawdown'}); "
                      f"shock-absorber line reservation fee {p.v('reservation_fee')}% a year of the line. No other charges are added by Genius.", ["Lender policy"])

    def _repay(self):
        return Answer("Short-term drawdowns and the shock-absorber line are repaid from customer collections: a share of each payment you receive goes straight to "
                      "the lender until the drawdown is cleared (never through an account Zoho controls). Medium-term loans have fixed monthly instalments.", ["Borrow tab"])

    def _expiry(self):
        n = self.pol.v("limit_validity_months")
        return Answer(f"Limits are valid for {n} months and reviewed every quarter. Renewal starts {self.pol.v('renewal_window_days')} days before expiry with fresh books. "
                      "A red flag suspends the limit at once; an upgrade needs a lasting improvement.", ["Lender policy"])

    def _security(self):
        return Answer("Short-term credit and the shock-absorber line are unsecured and repaid from collections. Medium-term loans are secured by a charge on receivables "
                      "and stock plus a personal or CGTMSE guarantee. Invoice financing is secured by the invoices. In the Watch tier any loan needs security.", ["Lender policy"])

    def _who(self):
        return Answer("A partner lender regulated by the RBI decides and lends. Genius recommends a limit and rate; the lender shows its Key Fact Statement before you "
                      "accept, and nothing is shared without your consent.", ["About Genius"])

    def _imp_best(self):
        res = self.what_ifs()
        lines = [f"{r['label']}: score {self.a.score:.0f} -> {r['score']:.0f} ({r['tier']}), rate {r['rate']:.2f}%" for r in res]
        return Answer("Re-scored through the engine: " + "; ".join(lines) + ".", ["Improve your rate tab"], ["What if I collect my overdue invoices?"])

    def what_ifs(self):
        from .assess import what_if
        a = self.a
        out = []
        od = float(a.f["ar_overdue"][0, -1])
        if od > 0:
            r = what_if(self.db, self.pol, a, collect=od)
            out.append(dict(label=f"Collect the {self.money(od)} overdue", **{k: r[k] for k in ("score", "tier", "rate")}))
        tc = a.m("RC09")
        if tc is not None and tc > 20:
            r = what_if(self.db, self.pol, a, top_cust_to=0.2)
            out.append(dict(label="Biggest customer below 20% of sales", **{k: r[k] for k in ("score", "tier", "rate")}))
        dr = a.m("OW03")
        if dr is not None and dr > 50:
            r = what_if(self.db, self.pol, a, drawings_share=0.5)
            out.append(dict(label="Drawings under half of operating cash flow", **{k: r[k] for k in ("score", "tier", "rate")}))
        return out

    def _imp_collect(self):
        from .assess import what_if
        od = float(self.a.f["ar_overdue"][0, -1])
        if od <= 0:
            return Answer("Nothing is overdue from customers right now.", ["Receivables"])
        r = what_if(self.db, self.pol, self.a, collect=od)
        return Answer(f"Collecting the {self.money(od)} overdue would move your Genius Score from {self.a.score:.0f} to {r['score']:.0f} ({r['tier']}) and your "
                      f"short-term rate to about {r['rate']:.2f}%.", ["Re-scored with the payment applied"], ["Who owes me the most?"])

    def _imp_conc(self):
        from .assess import what_if
        r = what_if(self.db, self.pol, self.a, top_cust_to=0.2)
        return Answer(f"With your biggest customer at 20% of sales, the score would be {r['score']:.0f} ({r['tier']}).", ["Re-scored"])

    def _imp_draw(self):
        from .assess import what_if
        r = what_if(self.db, self.pol, self.a, drawings_share=0.5)
        return Answer(f"Keeping drawings under half of operating cash flow would give a score of {r['score']:.0f} ({r['tier']}).", ["Re-scored"])

    def _cust_owe(self):
        if self.b is None:
            return Answer("Customer detail is not available.", ["Upload"])
        s = self.b.ageing_summary("receivable").head(5)
        txt = "; ".join(f"{p}: {self.money(v)}" for p, v in s["Total"].items())
        return Answer(f"Largest amounts owed to you: {txt}.", ["Receivables ageing from your books"], ["Which customers are overdue?"])

    def _cust_overdue(self):
        if self.b is None:
            return Answer("Customer detail is not available.", ["Upload"])
        a = self.b.ageing("receivable")
        od = a[a["days_past_due"] > 0].groupby("party").agg(v=("open", "sum"), d=("days_past_due", "max")).sort_values("v", ascending=False).head(5)
        if od.empty:
            return Answer("No customer is overdue.", ["Receivables ageing"])
        txt = "; ".join(f"{p}: {self.money(r.v)}, oldest {r.d:.0f} days past due" for p, r in od.iterrows())
        return Answer(f"Overdue now: {txt}. Total overdue {self.money(float(a.loc[a['days_past_due'] > 0, 'open'].sum()))}.", ["Receivables ageing"],
                      [f"What if {od.index[0]} pays this month?"])

    def _cust_top(self):
        a = self.a
        if self.b is None:
            return Answer(f"Your biggest customer is {self._val('RC09')} of sales.", ["Measures"])
        sh = self.b.party_share("receivable").head(3)
        return Answer("Biggest customers by invoiced value over the last year: " + "; ".join(f"{p} {100 * v:.0f}%" for p, v in sh.items()) + ".", ["Invoices"])

    def _sup_top(self):
        if self.b is None:
            return Answer(f"Your biggest supplier is {self._val('PY06')} of purchases.", ["Measures"])
        sh = self.b.party_share("payable").head(3)
        return Answer("Biggest suppliers by billed value over the last year: " + "; ".join(f"{p} {100 * v:.0f}%" for p, v in sh.items()) + ".", ["Bills"])

    def _sup_43b(self):
        v = self.a.m("MS1")
        if v is None:
            return Answer("Tick the suppliers registered on Udyam as micro or small on the Upload screen and Genius will check the 45-day rule.", ["Section 43B(h)"])
        return Answer(f"{v:.0f}% of what you paid micro and small suppliers in the last year was paid within their time limit (45 days with a written agreement, "
                      "15 without). Late payment loses the tax deduction for that year and carries interest at three times the RBI bank rate.", ["Section 43B(h)"])

    def _party(self, name):
        if self.b is None:
            return Answer("Customer detail is not available.", ["Upload"])
        h = self.b.party_payment_history(name)
        paid = h[h["open"].abs() < 0.5]
        op = h[h["open"].abs() >= 0.5]
        late = float(np.average(paid["days_late"], weights=paid["amount"])) if len(paid) else np.nan
        recent = paid[paid["last_payment"] > self.b.last - np.timedelta64(180, "D")]
        lr = float(np.average(recent["days_late"], weights=recent["amount"])) if len(recent) else np.nan
        txt = f"{name} has paid {len(paid)} invoices"
        if not np.isnan(late):
            txt += f", on average {late:.0f} days {'late' if late > 0 else 'early'}"
        if not np.isnan(lr):
            txt += f" ({lr:.0f} days over the last six months)"
        txt += f". Open now: {self.money(float(op['open'].sum()))} across {len(op)} invoices."
        return Answer(txt, ["Invoices and receipts in your books"], [f"What if {name} pays this month?"])

    def _party_if(self, name):
        from .assess import what_if
        if self.b is None:
            return Answer("Customer detail is not available.", ["Upload"])
        ag = self.b.ageing("receivable")
        od = float(ag.loc[(ag["party"] == name) & (ag["days_past_due"] > 0), "open"].sum())
        if od <= 0:
            return Answer(f"{name} has nothing overdue, so paying early would not change your score.", ["Receivables ageing"])
        r = what_if(self.db, self.pol, self.a, collect=od)
        return Answer(f"If {name} pays its {self.money(od)} overdue, your Genius Score moves from {self.a.score:.0f} to {r['score']:.0f} ({r['tier']}).",
                      ["Re-scored with the payment applied"], ["What if I collect my overdue invoices?"])

    def _sparty(self, name):
        if self.b is None:
            return Answer("Supplier detail is not available.", ["Upload"])
        h = self.b.party_payment_history(name, "payable")
        paid = h[h["open"].abs() < 0.5]
        late = float(np.average(paid["days_late"], weights=paid["amount"])) if len(paid) else np.nan
        return Answer(f"You have paid {name} {len(paid)} bills, on average {late:.0f} days {'after' if late > 0 else 'before'} the due date." if not np.isnan(late) else f"No settled bills from {name}.",
                      ["Bills and payments in your books"])

    def _cash_now(self):
        return Answer(f"Cash and unused cash credit cover {self._val('CR02')} of fixed costs (rated {self.a.r('CR02') or 0:.0f}).", ["Cash cover"], ["Is a cash squeeze coming?"])

    def _cash_warn(self):
        ew = self.a.info.get("early_warning")
        if ew:
            extra = ""
            v = self.a.m("EW1")
            if v is not None and v < 4 and not ew["alert"]["fired"]:
                extra = (f" Note the cash-trend measure in your score is weaker ({self._val('EW1')}): it assumes last quarter's cash burn simply continues, "
                         "while this forecast uses the actual due dates of your open invoices and bills.")
            return Answer(ew["alert"]["message"] + extra, ["13-week early warning from your invoices"], ["How much cash cover do I have?"])
        return Answer(f"If last quarter's cash burn continued, cash cover in three months would be {self._val('EW1')} of fixed costs.", ["Early warning"])

    def _cash_runway(self):
        return Answer(f"If every cost kept running and nothing came in, cash would last {self._val('CR03')}.", ["Cash runway"])

    def _data_q(self):
        rep = self.a.info.get("report", {})
        q = rep.get("quality_score")
        return Answer(f"Data confidence {q}/100." if q is not None else "Data confidence is shown on the Upload tab.", ["Upload check"], ["What is missing from my books?"])

    def _data_missing(self):
        miss = [C.M[m]["name"] for m in C.MEASURES if self.a.m(m) is None and C.bounds_for(m, self.a.family) is not None]
        return Answer("Not available from this file: " + (", ".join(miss) if miss else "nothing important") + ". Adding the cash-credit limit, Udyam suppliers and "
                      "due dates on every invoice fills most gaps.", ["Upload check"])

    def _data_size(self):
        return Answer(self.a.band_reason, ["Government MSME definition, S.O. 1364(E)"])

    def _data_sector(self):
        return Answer(f"Your sector is {self.a.sector}. You can change it on the Upload tab; it decides your peer group and some measure ranges.", ["Upload"])

    def _about_how(self):
        return Answer("Genius reads your ledger, checks it, works out 54 measures, rates them into a Genius Score and tier, compares you with similar businesses, "
                      "measures how you recover from setbacks (Grit), and turns that into limits and indicative rates under a partner lender's policy.", ["About Genius"])

    def _about_privacy(self):
        return Answer("Only with your consent, for the credit check you asked for. Data stays in India. A lender sees your file only if you request a loan. "
                      "You can withdraw consent at any time.", ["Consent"])

    def _about_synth(self):
        return Answer("Not yet. In this prototype the 5,000 peers are synthetic; in production they would be anonymised, consenting Zoho Books customers.", ["Prototype note"])

    def _about_ml(self):
        return Answer("Your score is set by written-down rules. A separate machine-learning model, the Second Opinion, checks the same firms for the lender and can "
                      "only make terms more cautious. Ask Genius itself uses no AI model: it matches your question to a bank of questions and answers from your numbers.",
                      ["About Genius"])


# ---------------------------------------------------------------------------------------------------------------- helpers and text
PROD = {"short": "Short-term credit", "shock": "Shock-absorber line", "medium": "Medium-term loan", "invoice": "Invoice financing", "long": "Long-term loan"}


def _ko_name(k):
    return {"K1": "Cash runway critical", "K2": "Debt service below 1", "K3": "Old receivables", "K4": "Statutory arrears", "K5": "Integrity failure"}[k]


def scorecard_ko_text(k):
    from .scorecard import KO_TEXT
    return KO_TEXT[k]


def _fmt_b(x):
    return f"{x:g}"


FORMULA = {
    "LQ01": "current assets divided by current liabilities", "LQ03": "cash and bank divided by current liabilities",
    "CR02": "(cash + unused cash-credit limit) divided by weekly fixed costs (pay, rent, insurance, utilities, professional fees and interest over the last 3 months / 13)",
    "CR03": "cash divided by daily total costs (cost of sales + running costs + interest, last 3 months)", "CG02": "operating cash flow divided by sales x 100",
    "CG03": "operating cash flow divided by net profit x 100", "FL11": "short-term debt divided by the cash-credit limit x 100",
    "EW1": "cash plus unused credit, less half of overdue bills, plus three months of projected operating cash flow after loan repayments and drawings, divided by weekly fixed costs",
    "TB5": "lowest bank balance inside each month divided by weekly fixed costs, median of the last 3 months",
    "RC01": "receivables divided by invoiced sales over 12 months x 365", "RC03": "value-weighted average days late of invoices paid in the last 12 months",
    "RC05": "past-due receivables divided by all receivables x 100", "RC08": "receivables over 90 days past due divided by all receivables x 100",
    "RC09": "largest customer's share of invoiced sales over 12 months", "RC11": "sum of squared customer shares of sales",
    "RC14": "growth in receivables minus growth in invoiced sales over 12 months", "RC16": "(opening receivables + invoiced - closing) / (opening + invoiced - not yet due) x 100",
    "WC01": "stock days + days sales outstanding - supplier days", "TB1": "days late in the last 3 months minus the 3 months before",
    "PY02": "value-weighted average days late of bills paid in the last 12 months", "PY03": "value of bills paid within 7 days of the due date / value paid x 100",
    "PY05": "past-due payables divided by all payables x 100", "PY06": "largest supplier's share of purchases",
    "PY09": "statutory dues outstanding divided by their normal monthly build-up", "TB2": "days late to suppliers, last 3 months minus the 3 before",
    "MS1": "value paid to micro and small suppliers within their limit / value paid to them x 100",
    "FL01": "borrowings divided by owner's equity", "FL02": "all liabilities divided by owner's equity", "FL04": "(borrowings - cash) / EBITDA over 12 months",
    "FL05": "operating profit (EBIT) / interest over 12 months", "FL06": "EBITDA / (interest + loan principal repaid) over 12 months",
    "FL07": "short-term debt / all debt x 100", "FL08": "borrowings now / 12 months ago - 1, x 100", "TB3": "months in the last 12 when short-term debt rose",
    "PF01": "gross profit / sales x 100", "PF02": "EBITDA / sales x 100", "PF04": "net profit / sales x 100", "PF05": "running costs / sales x 100",
    "PF10": "interest / sales x 100", "PF12": "(sales - break-even sales) / sales x 100", "RC19": "sales over 12 months vs the 12 months before",
    "RS01": "standard deviation / mean of monthly sales x 100", "RS02": "standard deviation of monthly gross margin (points)",
    "RS03": "share of the last 12 months with a profit", "RS04": "cash divided by the worst month of net cash outflow in 12 months",
    "IN01": "stock / cost of sales over 12 months x 365", "IN04": "growth in stock minus growth in cost of sales",
    "OW02": "drawings / net profit x 100", "OW03": "drawings / operating cash flow x 100",
    "DQ01": "share of journal entries where debits equal credits", "DQ02": "share of money in accounts Genius recognised",
    "DQ03": "share of customer and supplier lines with a name", "DQ04": "share of invoices with a due date", "IC1": "integrity checks passed out of 7",
}
IMPROVE = {
    "CR02": "Collect overdue invoices and agree a cash-credit limit that matches your cycle.", "CR03": "Hold more cash against total costs; collect faster.",
    "LQ01": "Fund stock and receivables with long-term money rather than short-term borrowing.", "LQ03": "Keep more of current liabilities covered by cash.",
    "CG02": "Turn sales into cash faster: shorter customer terms, less stock.", "CG03": "Profit that stays in receivables is not cash: collect it.",
    "EW1": "Plan large payments around expected collections; clear overdue bills in a planned way.", "TB5": "Spread big supplier payments across the month.",
    "RC01": "Invoice promptly, follow up on due dates, offer small discounts for early payment.", "RC03": "Chase invoices on the due date; stop credit to habitual late payers.",
    "RC05": "Collect the overdue invoices, largest first.", "RC08": "Settle or provide for invoices more than 90 days overdue.",
    "RC09": "Grow sales with other customers so no single customer dominates.", "RC11": "Widen your customer base.",
    "RC14": "Make sure receivables grow no faster than sales.", "RC16": "Collect more of what is due each month.", "WC01": "Cut stock days and collect faster.",
    "TB1": "Customers are paying later than before: follow up sooner.", "PY02": "Pay bills by their due dates.", "PY03": "Pay bills within a week of the due date.",
    "PY05": "Clear overdue bills.", "PY06": "Add a second source for your largest supplies.", "PY09": "Pay PF, ESI, TDS and GST on time every month.",
    "TB2": "You are stretching suppliers more than before: plan payments.", "MS1": "Pay micro and small suppliers within 45 days to keep the tax deduction.",
    "FL01": "Retain profits or bring in equity rather than borrow.", "FL02": "Reduce outside liabilities relative to net worth.", "FL04": "Repay debt from operating profit.",
    "FL05": "Raise operating profit or reduce interest.", "FL06": "Raise operating profit or lengthen loan tenors.", "FL07": "Fund long-term needs with term loans, not cash credit.",
    "FL08": "Grow borrowing no faster than the business.", "TB3": "Avoid topping up short-term debt every month.", "PF01": "Review pricing and purchase costs.",
    "PF02": "Control running costs as sales grow.", "PF04": "Watch interest, depreciation and tax as well as margins.", "PF05": "Trim running costs.",
    "PF10": "Refinance expensive debt.", "PF12": "Lower fixed costs or raise contribution margin.", "RC19": "Grow sales steadily.", "RS01": "Smooth sales across the year.",
    "RS02": "Keep pricing discipline month to month.", "RS03": "Avoid loss-making months.", "RS04": "Hold a buffer for your worst month.", "IN01": "Reduce slow-moving stock.",
    "IN04": "Keep stock growth in line with sales.", "OW02": "Take drawings within profit.", "OW03": "Take drawings within the cash the business produces.",
    "DQ01": "Fix entries that do not balance.", "DQ02": "Name accounts clearly or teach Genius new account names.", "DQ03": "Record the customer or supplier on every line.",
    "DQ04": "Put a due date on every invoice.", "IC1": "Fix the failed integrity checks shown on the Upload tab.",
}
DATA_NEED = {m: "invoice-level customer and supplier detail (names, invoice numbers, due dates)" for m in
             ("RC03", "RC05", "RC08", "RC09", "RC11", "RC16", "TB1", "PY02", "PY03", "PY05", "PY06", "TB2", "MS1")}
DATA_NEED.update({"FL11": "the cash-credit limit you type in", "MS1": "invoice detail and the suppliers you tick as micro or small (Udyam)",
                  "CR02": "cash, the cash-credit limit (if typed in) and fixed costs", "TB5": "dated bank entries"})
