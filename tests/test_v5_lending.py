"""V5 regressions: requests, financial consistency, fees, and evidence gates."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from genius import config as C, credit_model as CM, lending, scorecard
from genius.policy import Policy


@pytest.fixture
def pol():
    p = Policy(); p.calibrated_pd = {"Safe": .4, "Moderate": 1., "Watch": 3.8, "Risky": 17.4}
    return p


@pytest.fixture
def business():
    facts = dict(annual_sales=10000000, recv_young=2000000, coll_month=800000, std=200000,
                 fixed_month=300000, cash_cost_month=300000, ebitda12=1200000, cfads12=1200000,
                 debt_service12=200000, total_liab=2000000, equity=5000000,
                 invoices_eligible=1000000, cash=1500000)
    return SimpleNamespace(score=65, tier="Moderate", p_default=.01, sector="Trading · FMCG", band="Micro",
                           gs={"ko": {k: np.array([[False]]) for k in ("K1", "K2", "K3", "K4", "K5")}},
                           so={"confidence": "High"}, info={"lending_facts": facts,
                           "policy_context": {"sector": "Trading · FMCG", "size": "Micro", "regime": "calm", "date": "2026-10-29"}})


def test_requested_amount_and_repayment_reconcile(business, pol):
    q = lending.quote(business, pol, 500000, 60, "Short term")
    assert q["status"] == "fits" and q["supported"] == 500000
    assert q["interest"] == pytest.approx(500000 * q["rate"] / 100 * 60 / 365)
    assert q["fees"] == 7500  # full annual fee, not a prorated share
    assert q["net_disbursed"] == 492500
    assert q["total_cost"] == pytest.approx(q["fees"] + q["interest"])
    assert sum(r["Principal"] for r in q["schedule"]) == 500000


def test_oversized_request_is_alternative_not_original_approval(business, pol):
    q = lending.quote(business, pol, 5000000, 60, "Short term")
    assert q["status"] == "reduced"
    assert q["requested"] == 5000000 and 0 < q["supported"] < q["requested"]
    assert q["interest"] == pytest.approx(q["supported"] * q["rate"] / 100 * 60 / 365)


@pytest.mark.parametrize("amount", [None, 0, -1, np.nan, np.inf, "bad"])
def test_bad_amounts_never_create_quotes(business, pol, amount):
    q = lending.quote(business, pol, amount, 60, "Short term")
    assert q["status"] == "invalid" and not q["schedule"]


@pytest.mark.parametrize("days", [0, -1, 6, 181, 60.5, np.inf])
def test_bad_duration_rejected(business, pol, days):
    assert lending.quote(business, pol, 500000, days, "Short term")["status"] == "invalid"


def test_medium_security_and_affordability(business, pol):
    assert lending.quote(business, pol, 500000, 60, "Medium term", 36)["status"] == "review"
    q = lending.quote(business, pol, 5000000, 60, "Medium term", 36, True, 10000000)
    assert q["status"] == "reduced" and q["cover_after"] >= pol.v("mt_dscr_min")
    assert q["emi"] == pytest.approx(lending.payment(q["supported"], q["rate"], 36))
    assert sum(r["Principal"] for r in q["schedule"]) == pytest.approx(q["supported"], abs=.1)
    assert q["fees"] == pytest.approx(q["supported"] * .015 * 3)
    assert q["schedule"][-1]["Balance"] == 0
    assert q["annual_effective_cost"] > q["rate"]


def test_collateral_limits_actual_quote(business, pol):
    q = lending.quote(business, pol, 500000, 60, "Medium term", 36, True, 400000)
    assert q["supported"] <= 280000


def test_context_overrides_price_and_fee_together(business, pol):
    ctx = {"sector": business.sector, "size": "Micro", "regime": "calm", "tenor": "short", "from_date": "2026-10-29"}
    before = lending.quote(business, pol, 500000, 60, "Short term")
    pol.add_rule("margin", 3, ctx); pol.add_rule("processing_fee", 3, ctx)
    after = lending.quote(business, pol, 500000, 60, "Short term")
    assert after["rate"] == pytest.approx(before["rate"] + 1)
    assert after["fees"] == 15000 and after["policy_version"] == pol.version


def test_full_annual_fee_uses_facility_not_draw(pol):
    q = pol.apr(200000, 14.95, 90, facility_limit=1000000)
    assert q["fee"] == 15000
    assert q["interest"] == pytest.approx(7372.602739726028)
    assert q["net_disbursed"] == 185000


def test_no_duplicate_grit_reward(pol):
    low = pol.rate("Moderate", pd_score=1, tenor_months=36, grit=10)
    high = pol.rate("Moderate", pd_score=1, tenor_months=36, grit=90)
    assert low["rate"] == high["rate"] and high["grit_concession"] == 0


def test_more_debt_and_less_cashflow_cannot_increase_amount(business, pol):
    q = lending.quote(business, pol, 5000000, 60, "Medium term", 36, True, 10000000)
    business.info["lending_facts"]["debt_service12"] += 100000
    business.info["lending_facts"]["cfads12"] *= .8
    r = lending.quote(business, pol, 5000000, 60, "Medium term", 36, True, 10000000)
    assert r["supported"] <= q["supported"]


def test_low_cash_no_fake_positive_offer(business, pol):
    business.info["lending_facts"].update(cash=0, coll_month=100000, cash_cost_month=300000)
    q = lending.quote(business, pol, 500000, 60, "Short term")
    assert q["status"] == "review" and q["supported"] == 0 and not q["schedule"]


def test_recovery_missing_is_neutral_and_capacity_is_not_repeated():
    r = {m: np.array([[70.]]) for m in C.MEASURES}
    g = {"score": np.array([[100.]]), "capacity": np.array([[100.]])}
    a = CM.features(r, g)
    g["score"][:] = 0; g["capacity"][:] = 0
    b = CM.features(r, g)
    np.testing.assert_array_equal(a, b)
    assert np.all(a[..., len(CM.FINANCIAL):] == 0)
    assert all(C.M[m]["area"] != "Data integrity" for m in CM.FINANCIAL)


def test_learned_model_keeps_caps_and_quality_gate():
    r = {m: np.array([[70.]]) for m in C.MEASURES}
    meas = {m: np.array([[70.]]) for m in C.MEASURES}; meas["FL06"][:] = 1.6
    model = CM.CreditModel(np.zeros(len(CM.FINANCIAL) + len(CM.RECOVERY)),
                           np.r_[-4., np.ones(len(CM.FINANCIAL)) * .01], False, -5., 0.)
    r["CR03"][:] = 0
    gs = model.apply(scorecard.genius_score(r, meas), r, {})
    assert gs["score"][0, 0] <= 44
    for m in C.BY_AREA["Data integrity"]:
        r[m][:] = 0
    assert np.isnan(model.apply(scorecard.genius_score(r, meas), r, {})["score"][0, 0])


def test_zero_interest_payment():
    assert lending.payment(120000, 0, 12) == 10000
    assert lending.principal_for_payment(10000, 0, 12) == 120000


def test_obsolete_cache_cannot_restore_fixed_weight_score(monkeypatch, tmp_path):
    from genius import app
    cache = tmp_path / "old_database.pkl"; cache.touch()
    fresh = SimpleNamespace(learned_model=SimpleNamespace(version=CM.VERSION),
                            cal=SimpleNamespace(tier_pd={t: .01 for t in C.TIERS}))
    saved = []
    monkeypatch.setenv("GENIUS_DB", str(cache))
    monkeypatch.setattr(app, "DB", None)
    monkeypatch.setattr(app, "BASE_POL", None)
    monkeypatch.setattr(app.pipeline, "load", lambda p: SimpleNamespace(learned_model=None))
    monkeypatch.setattr(app.pipeline, "build", lambda *a, **kw: fresh)
    monkeypatch.setattr(app.pipeline, "save", lambda db, path: saved.append(db))
    assert app.get_db(log=lambda s: None) is fresh
    assert saved == [fresh]
