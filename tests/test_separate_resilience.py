"""Separate scores: constructed recovery histories, honest policy and no duplicate reward."""
import copy
import numpy as np
import pytest
from genius import resilience, resilience_demo
from genius.policy import Policy


def test_history_drives_different_grit_without_changing_present_financial_rating():
    rows = resilience_demo.reference_quotes()
    fast, slow = rows
    assert fast['genius'] == slow['genius'] == 65
    assert fast['grit'] > slow['grit']
    assert fast['n'] >= 1 and slow['n'] >= 1
    assert fast['quote']['supported'] == slow['quote']['supported'] == 500000
    assert fast['quote']['rate_build']['pd'] == slow['quote']['rate_build']['pd'] == 1
    assert fast['quote']['rate'] < slow['quote']['rate']
    assert fast['quote']['fees'] == slow['quote']['fees'] == 7500
    assert fast['quote']['interest'] < slow['quote']['interest']


def test_capacity_never_creates_an_additional_price_reward():
    g = copy.deepcopy(resilience_demo.profiles()[0]['grit'])
    before = resilience.evidence(g,100)
    g['score'][:] = 100; g['capacity'][:] = 100
    after = resilience.evidence(g,100)
    assert after == before
    assert resilience.concession(after) == resilience.concession(before)


def test_no_episodes_no_concession_and_low_quality_blocks_it():
    g = copy.deepcopy(resilience_demo.profiles()[0]['grit'])
    assert resilience.concession(resilience.evidence(g,84)) == 0
    g['episodes'] = []
    assert resilience.concession(resilience.evidence(g,100)) == 0


def test_stale_or_unresolved_recovery_has_no_concession():
    g = copy.deepcopy(resilience_demo.profiles()[0]['grit'])
    for e in g['episodes']: e['judged'] = 0
    assert resilience.concession(resilience.evidence(g,100)) == 0
    g = copy.deepcopy(resilience_demo.profiles()[0]['grit'])
    g['episodes'].append(dict(kind='setback',firm=0,start=64,judged=None))
    assert resilience.concession(resilience.evidence(g,100)) == 0


def test_discount_is_bounded_by_margin_and_explicitly_switchable():
    p = Policy(); rec = resilience_demo.profiles()[0]['evidence']
    p.base['margin'] = .01
    r = p.rate('Moderate',pd_score=1,recovery=rec)
    assert 0 < r['recovery_concession'] <= .01
    assert r['rate'] == pytest.approx(r['rate_before_recovery']-r['recovery_concession'])
    p.base['recovery_demo_enabled'] = False
    off = p.rate('Moderate',pd_score=1,recovery=rec)
    assert off['recovery_concession'] == 0
    assert off['rate'] == pytest.approx(r['rate_before_recovery'])
    assert p.rate('Risky',pd_score=20,recovery=rec)['recovery_concession'] == 0


def test_no_reward_for_unobserved_future_episode():
    g = copy.deepcopy(resilience_demo.profiles()[0]['grit'])
    # A historical quote cannot use the later setback's assessment.
    record = resilience.evidence(g,100,now=46)
    assert not record['eligible']
    assert resilience.concession(record) == 0
