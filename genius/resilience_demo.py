"""Constructed operating histories for the defence; never overwrite uploaded books.

Both histories face identical 45% demand shocks. Recovery speed, collection response,
cost flexibility and extra borrowing differ. No score or default labels are fabricated.
Current financial underwriting inputs are held fixed in the comparison UI to isolate
an assumed lender policy; this is a scenario, not evidence of incremental prediction.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from . import grit as G
from .resilience import evidence


@lru_cache(maxsize=1)
def profiles():
    T = 66
    months = pd.period_range('2021-01', periods=T, freq='M')
    noise = np.random.default_rng(49).normal(0, .025, T)
    output = []
    for name, speed, flex, collection_response in (
        ('Fast recovery', .65, .8, .9), ('Slow recovery', .10, .15, .2)):
        shock = 1.; debt = 300000.; rows = []
        for t in range(T):
            if t in (24, 45): shock *= .55
            elif t not in (25, 46): shock += speed * (1 - shock)
            sales = 1e6 * (1 + noise[t]) * shock
            costs = 130000 * (1 - flex * (1 - shock))
            profit = .25 * sales - costs
            received = sales * (1 - (1 - collection_response) * .35 * (1 - shock))
            operating = received - .75 * sales - costs
            debt += max(0, -operating)
            rows.append(dict(profile=name, month=str(months[t]), sales=sales, operating_profit=profit,
                             receipts=received, operating_costs=costs, operating_cash_flow=operating,
                             receivables=1e6 * (.8 + (1 - collection_response) * (1 - shock)), debt=debt,
                             recovery_speed=speed, cost_flexibility=flex, collection_response=collection_response))
        df = pd.DataFrame(rows)
        arr = lambda k: df[k].to_numpy()[None, :]
        sig = dict(sales=arr('sales'), rev=arr('sales'), op=arr('operating_profit'), rec=arr('receipts'),
                   opex=arr('operating_costs'), ocf=arr('operating_cash_flow'), recv=arr('receivables'), debt=arr('debt'))
        g = G.compute(sig, np.array([m.month for m in months]), np.ones((1,T),bool),
                      np.array([100.]), np.full((1,T),60.), keep_eps=True)
        g['months'] = months
        output.append(dict(name=name, grit=g, evidence=evidence(g,100), history=df))
    return output


def reference_quotes():
    """An isolated pair: identical present finances and loan request, different history."""
    from types import SimpleNamespace
    from .policy import Policy
    from .lending import quote
    pol = Policy(); pol.calibrated_pd = {'Moderate':1.0}
    facts = dict(annual_sales=10000000, recv_young=2000000, coll_month=800000, std=200000,
                 fixed_month=300000, cash_cost_month=300000, ebitda12=1200000, cfads12=1200000,
                 debt_service12=200000, total_liab=2000000, equity=5000000,
                 invoices_eligible=1000000, cash=1500000)
    out = []
    for p in profiles():
        a = SimpleNamespace(score=65., tier='Moderate', p_default=.01, sector='Trading · FMCG', band='Micro',
            gs={'ko':{k:np.array([[False]]) for k in ('K1','K2','K3','K4','K5')}}, so={'confidence':'High'},
            info={'lending_facts':dict(facts), 'recovery_policy':p['evidence'],
                  'policy_context':{'sector':'Trading · FMCG','size':'Micro','regime':'calm','date':'2026-10-29'}})
        q = quote(a,pol,500000,60,'Short term')
        out.append(dict(profile=p['name'], genius=a.score, grit=float(p['grit']['score'][0,-1]),
            recovery_index=p['evidence']['recovery_index'], n=p['evidence']['completed_recoveries'],
            concession_bps=100*q['rate_build']['recovery_concession'], quote=q))
    return out
