"""Genius prototype tests. Run with:  python -m pytest -q tests
The leakage test builds a small world twice: once as simulated, once with everything after a cut-off month scrambled.
Anything decided at or before the cut-off must come out identical; if it doesn't, future data leaked into it."""
import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from genius import config as C, measures as MS, pipeline, scorecard as SC, synth  # noqa: E402
from genius.policy import Policy  # noqa: E402

N_SMALL = 400


@pytest.fixture(scope="module")
def small():
    return synth.simulate(N_SMALL, seed=7)


@pytest.fixture(scope="module")
def db():
    return pipeline.load() if (pipeline.ART / "genius_db.pkl").exists() else pipeline.build(5000, log=lambda m: None)


# ------------------------------------------------------------------------------------------------ synthetic world
def test_accounting_identity(small):
    f = small.f
    d = MS.derived(f)
    gap = d["ta"] - d["tl_all"] - d["equity"]
    assert np.nanmax(np.abs(gap) / np.maximum(1, np.abs(d["ta"]))) < 1e-6


def test_deterministic():
    a, b = synth.simulate(150, seed=3), synth.simulate(150, seed=3)
    for k in ("rev", "cash", "recv"):
        np.testing.assert_array_equal(a.f[k], b.f[k])
    np.testing.assert_array_equal(a.y["stress90"], b.y["stress90"])


def test_base_rates_sensible(small):
    s = np.nanmean(small.y["stress90"]); d = np.nanmean(small.y["default12"])
    assert 0.05 < s < 0.35 and 0.005 < d < 0.12


# ------------------------------------------------------------------------------------------------ scorecard
def test_54_measures_and_weights():
    assert len(C.MEASURES) == 54
    assert abs(sum(C.AREAS.values()) - 100) < 1e-9


def test_score_range_and_knockouts(small):
    meas = MS.compute(small.f, small.firms)
    gs = SC.genius_score(MS.rate_all(meas, small.firms["family"].values), meas)
    s = gs["score"][~np.isnan(gs["score"])]
    assert s.min() >= 0 and s.max() <= 100
    k1 = gs["ko"]["K1"] & ~np.isnan(gs["score"])
    assert np.all(gs["score"][k1] <= 44 + 1e-6)


def test_msme_definition():
    assert C.msme_band(2e7, 9e7) == "Micro"
    assert C.msme_band(3e7, 9e7) == "Small"          # investment over 2.5 cr moves it up
    assert C.msme_band(2e7, 11e7) == "Small"         # turnover over 10 cr moves it up
    assert C.msme_band(130e7, 50e7) not in ("Micro", "Small", "Medium")


# ------------------------------------------------------------------------------------------------ policy
def test_policy_guardrails_and_overrides():
    p = Policy(); p.calibrated_pd = {"Safe": 0.4, "Moderate": 1.0, "Watch": 3.8, "Risky": 17.4}
    assert p.set({"lgd_unsecured": 5}) and p.base["lgd_unsecured"] == 60      # refused below the 30% floor
    base = p.rate("Watch")["rate"]
    p.add_rule("margin", 3.0, {"tier": "Watch"})
    p.add_rule("margin", 4.0, {"tier": "Watch", "sector": "Services · logistics"})
    assert abs(p.rate("Watch")["rate"] - base - 1.0) < 1e-9
    assert abs(p.rate("Watch", {"sector": "Services · logistics"})["rate"] - base - 2.0) < 1e-9   # most specific wins
    assert p.rate("Risky")["offered"] is False
    assert p.version == 3


def test_second_opinion_only_cuts():
    p = Policy(); p.calibrated_pd = {"Safe": 0.4, "Moderate": 1.0, "Watch": 3.8, "Risky": 17.4}
    x = dict(annual_sales=1e8, recv_young=2e7, coll_month=8e6, std=0, fixed_month=1e6, ebitda12=1.2e7, debt_service12=2e6,
             total_liab=3e7, equity=4e7, invoices_eligible=1e7)
    ko = {k: False for k in ("K1", "K2", "K3", "K4", "K5")}
    base = p.limits(x, "Safe", 60, ko)
    for c in ("High", "Medium", "Scorecard only"):
        assert p.limits(x, "Safe", 60, ko, confidence=c)["short"]["limit"] == base["short"]["limit"]
    assert p.limits(x, "Safe", 60, ko, confidence="Low")["short"]["limit"] == pytest.approx(0.5 * base["short"]["limit"])


def test_grit_only_improves_terms():
    p = Policy(); p.calibrated_pd = {"Safe": 0.4, "Moderate": 1.0, "Watch": 3.8, "Risky": 17.4}
    x = dict(annual_sales=1e8, recv_young=2e7, coll_month=8e6, std=0, fixed_month=1e6, ebitda12=1.2e7, debt_service12=2e6,
             total_liab=3e7, equity=4e7, invoices_eligible=1e7)
    ko = {k: False for k in ("K1", "K2", "K3", "K4", "K5")}
    lo, hi, none = p.limits(x, "Moderate", 10, ko), p.limits(x, "Moderate", 90, ko), p.limits(x, "Moderate", None, ko)
    assert lo["short"]["available"] == hi["short"]["available"] == none["short"]["available"]      # never the reason for a refusal
    assert hi["shock"]["limit"] >= lo["shock"]["limit"] and hi["medium"]["limit"] >= lo["medium"]["limit"]


# ------------------------------------------------------------------------------------------------ no look-ahead
def test_no_future_data_leaks(monkeypatch):
    """Rebuild a small world with every fact after month 100 and every outcome label from month 93 scrambled.
    Scores, Grit and the market regime up to month 100 must not change. The Second Opinion's model is chosen at the end of the
    checking years (month 95, using labels up to month 92, which look 3 months ahead) and refitted each quarter on labels up to
    3 months back, so its predictions up to month 95 must not change either; nor may the tier cut-offs (years 1-5)."""
    CUT_F, CUT_Y = 100, 93
    P0 = synth.simulate(N_SMALL, seed=11)
    P1 = copy.deepcopy(P0)
    rng = np.random.default_rng(0)
    for k, v in P1.f.items():
        v[:, CUT_F + 1:] = v[:, CUT_F + 1:] * rng.uniform(0.3, 1.7, size=v[:, CUT_F + 1:].shape)
    for k in ("stress90", "default12", "severity"):
        y = P1.y[k]
        y[:, CUT_Y:] = rng.permutation(y[:, CUT_Y:], axis=0)
    calls = iter([P0, P1])
    monkeypatch.setattr(synth, "simulate", lambda *a, **k: next(calls))
    q = lambda m: None
    a, b = pipeline.build(N_SMALL, log=q), pipeline.build(N_SMALL, log=q)
    np.testing.assert_allclose(a.gs["score"][:, :CUT_F + 1], b.gs["score"][:, :CUT_F + 1], equal_nan=True)
    np.testing.assert_allclose(a.grit["score"][:, :CUT_F + 1], b.grit["score"][:, :CUT_F + 1], equal_nan=True)
    np.testing.assert_allclose(a.regime_p[:CUT_F + 1], b.regime_p[:CUT_F + 1])
    assert a.cal.cutoffs == b.cal.cutoffs
    assert a.so["winner"] == b.so["winner"]
    np.testing.assert_allclose(a.so["pred"][:, 60:96], b.so["pred"][:, 60:96], rtol=1e-5, atol=1e-6, equal_nan=True)


# ------------------------------------------------------------------------------------------------ the full database and the app
def test_validation_pass_marks(db):
    v = db.val
    assert v["auc_stress"] >= 0.70 and v["tiers_in_order"] and v["tiers_on_target"]
    assert v["lending"]["Genius"]["loss_rate"] < v["lending"]["Naive rule"]["loss_rate"]
    assert v["so"]["test_ml"]["p"] < 0.05


@pytest.fixture(scope="module")
def sessions(db):
    from genius.session import Session
    from genius import app
    app.DB = db; app.BASE_POL = app.new_policy()
    out = {}
    for key, (fn, cc, mse) in app.SAMPLES.items():
        if "ledger" in key:
            continue
        s = Session(); r = s.read(str(ROOT / "samples" / fn)); s.run(db, app.new_policy(), r["sector"], cc, mse)
        out[fn.split("_")[0]] = s
    return out


def test_demo_businesses(sessions):
    assert sessions["Sharma"].a.tier == "Watch"
    assert sessions["Kaveri"].a.tier == "Safe"
    assert sessions["Swift"].a.tier == "Risky" and not sessions["Swift"].a.limits["short"]["available"]


def test_every_menu_question_answers(sessions):
    for s in sessions.values():
        g = s.genie
        for q in g.bank:
            ans = g.answer(q.id)
            assert ans.text and "nan" not in ans.text.lower().split(), (q.id, ans.text)


def test_owner_never_sees_second_opinion(sessions):
    from genius import ui, app
    s = sessions["Sharma"]
    html = ui.standing(s.a, app.DB, app.BASE_POL) + ui.peers(s.a, app.DB) + ui.borrow(s.a, app.BASE_POL)
    for w in ("Second Opinion", "confidence: Low", "LightGBM"):
        assert w not in html
    for q in s.genie.bank:
        t = s.genie.answer(q.id).text
        assert "Low confidence" not in t and "secret" not in t.lower()


def test_guardrails_cover_overrides_and_the_hold():
    p = Policy(); p.calibrated_pd = {"Safe": 0.4, "Moderate": 1.0, "Watch": 3.8, "Risky": 17.4}
    with pytest.raises(ValueError):
        p.add_rule("lgd_unsecured", 5.0, {})                       # same guard-rails as base settings
    with pytest.raises(ValueError):
        p.add_rule("margin", 1.0, {"from_date": "yesterday"})
    assert p.set({"so_low_limit_cut": -50}) and p.base["so_low_limit_cut"] == 50
    p.add_rule("margin", 1.0, {"tier": "Safe"}); p.add_rule("margin", 2.0, {"tier": "Safe"})
    p.remove_rule(0); p.add_rule("margin", 3.0, {"tier": "Safe"})
    assert p.v("margin", {"tier": "Safe"}) == 3.0                   # newer rule wins a tie, even after a removal
