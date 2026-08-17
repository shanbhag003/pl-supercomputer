"""Market correction layer.

Opening-day bookmaker odds are set before any match is played, but AFTER every
transfer, manager change and injury the market knows about. So the gap between
what our model expects from those fixtures and what the market prices is an
estimate of everything our model is blind to.

We fit one net strength adjustment per team (delta), then apply a fraction of
it. delta > 0 means the market rates the team higher than we do.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ratings import score_matrix

FD2US = {'Man City': 'Manchester City', 'Man United': 'Manchester United',
         'Man Utd': 'Manchester United', 'Newcastle': 'Newcastle United',
         'Spurs': 'Tottenham', 'Wolves': 'Wolverhampton Wanderers',
         "Nott'm Forest": 'Nottingham Forest', 'West Brom': 'West Bromwich Albion'}


def opening_odds(path, days=4, div='E0'):
    """1X2 and over/under 2.5 for the opening round of a season."""
    d = pd.read_csv(path, encoding='latin-1')
    if 'Div' in d.columns:
        d = d[d['Div'] == div]
    d = d.dropna(subset=['HomeTeam']).copy()
    d['dt'] = pd.to_datetime(d['Date'], dayfirst=True, format='mixed')
    d = d[d['dt'] <= d['dt'].min() + pd.Timedelta(days=days)]

    h = 'AvgH' if 'AvgH' in d.columns else 'B365H'
    dr = 'AvgD' if 'AvgD' in d.columns else 'B365D'
    a = 'AvgA' if 'AvgA' in d.columns else 'B365A'
    ov = 'Avg>2.5' if 'Avg>2.5' in d.columns else 'B365>2.5'
    un = 'Avg<2.5' if 'Avg<2.5' in d.columns else 'B365<2.5'

    out = []
    for _, r in d.iterrows():
        if pd.isna(r.get(h)) or pd.isna(r.get(a)):
            continue
        p = np.array([1 / r[h], 1 / r[dr], 1 / r[a]])
        p = p / p.sum()
        po = np.nan
        if ov in d.columns and not pd.isna(r.get(ov)) and not pd.isna(r.get(un)):
            po = (1 / r[ov]) / (1 / r[ov] + 1 / r[un])
        out.append(dict(home=FD2US.get(r.HomeTeam, r.HomeTeam),
                        away=FD2US.get(r.AwayTeam, r.AwayTeam),
                        pH=p[0], pD=p[1], pA=p[2], pOver=po))
    return pd.DataFrame(out)


def _probs(lh, la, rho):
    M = score_matrix(lh, la, rho, 10)
    k = np.arange(11)
    tot = k[:, None] + k[None, :]
    return (np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum(),
            M[tot >= 3].sum())


def fit_market_delta(model, odds, teams, ridge=8.0, w_over=0.5):
    """Per-team net strength adjustment implied by the opening odds."""
    idx = {t: i for i, t in enumerate(teams)}
    rows = [r for _, r in odds.iterrows()
            if r.home in idx and r.away in idx]
    if len(rows) < 5:
        return {t: 0.0 for t in teams}, dict(n=len(rows), rmse=np.nan)

    att = np.array([model['att'][t] for t in teams])
    dfn = np.array([model['dfn'][t] for t in teams])
    hi = np.array([idx[r.home] for r in rows])
    ai = np.array([idx[r.away] for r in rows])
    mkt = np.array([[r.pH, r.pD, r.pA] for r in rows])
    ovr = np.array([r.pOver for r in rows])

    def loss(delta, ret_rmse=False):
        A = att + delta / 2
        D = dfn + delta / 2
        tot, se = 0.0, []
        for j in range(len(rows)):
            lh = np.clip(np.exp(model['mu'] + model['gamma'] + A[hi[j]] - D[ai[j]]), .05, 8)
            la = np.clip(np.exp(model['mu'] + A[ai[j]] - D[hi[j]]), .05, 8)
            pH, pD, pA, pO = _probs(lh, la, model['rho'])
            e = ((pH - mkt[j, 0]) ** 2 + (pD - mkt[j, 1]) ** 2 + (pA - mkt[j, 2]) ** 2)
            if not np.isnan(ovr[j]):
                e += w_over * (pO - ovr[j]) ** 2
            tot += e
            se.append(e)
        if ret_rmse:
            return np.sqrt(np.mean(se))
        return tot + ridge * np.sum(delta ** 2)

    res = minimize(loss, np.zeros(len(teams)), method='L-BFGS-B',
                   options=dict(maxiter=4000, ftol=1e-12, gtol=1e-10))
    delta = res.x
    delta = delta - delta.mean()          # only relative strength is identified
    return (dict(zip(teams, delta)),
            dict(n=len(rows), rmse=float(loss(delta, ret_rmse=True))))


def apply_delta(models, delta, weight):
    """Shift a bootstrap ensemble by `weight` x the market adjustment."""
    if weight == 0:
        return models
    for m in models:
        for t in m['teams']:
            d = delta.get(t, 0.0) * weight
            m['att'][t] += d / 2
            m['dfn'][t] += d / 2
    return models
