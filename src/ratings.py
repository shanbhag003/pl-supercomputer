"""Dixon-Coles style attack/defence ratings with exponential time decay.

Stage 1: weighted quasi-Poisson MLE on blended (xG, goals) for attack/defence/home.
Stage 2: low-score dependence parameter rho estimated on actual scorelines.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def blend_target(df, w_xg=0.7):
    """Blend non-penalty xG with actual goals. w_xg=1 -> pure xG, 0 -> pure goals."""
    h = w_xg * df['hnpxg'].values + (1 - w_xg) * df['hg'].values
    a = w_xg * df['anpxg'].values + (1 - w_xg) * df['ag'].values
    return h, a


PROMOTED_ATT = -0.255   # empirical, 33 promoted teams 2015-2025
PROMOTED_DEF = -0.192
PROMOTED_SD = 0.285


def fit_ratings(df, ref_date, xi=0.0045, w_xg=0.7, max_years=4.0,
                ridge=0.0, prior_att=None, prior_dfn=None, extra_teams=()):
    """Fit ratings using only matches strictly before ref_date.

    ridge      : L2 shrinkage strength toward prior (0 = plain MLE)
    prior_att/dfn : dict team -> prior value. Missing teams default to the
                 empirical promoted-team prior.
    extra_teams: teams to include with no match data (pure prior).
    """
    ref = pd.Timestamp(ref_date)
    d = df[(df['date'] < ref) &
           (df['date'] >= ref - pd.Timedelta(days=365.25 * max_years))].copy()
    if len(d) < 50:
        raise ValueError('not enough history')

    days = (ref - d['date']).dt.days.values
    w = np.exp(-xi * days)

    teams = sorted(set(d['home']) | set(d['away']) | set(extra_teams))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    hi = d['home'].map(idx).values
    ai = d['away'].map(idx).values
    yh, ya = blend_target(d, w_xg)

    pa_ = np.array([(prior_att or {}).get(t, PROMOTED_ATT) for t in teams])
    pd_ = np.array([(prior_dfn or {}).get(t, PROMOTED_DEF) for t in teams])

    # params: [att(T-1 free), def(T), mu, gamma]
    def unpack(p):
        att = np.zeros(T)
        att[:T - 1] = p[:T - 1]
        att[T - 1] = -att[:T - 1].sum()          # sum-to-zero constraint
        dfn = p[T - 1:2 * T - 1]
        mu, gamma = p[-2], p[-1]
        return att, dfn, mu, gamma

    def nll(p):
        att, dfn, mu, gamma = unpack(p)
        lh = np.exp(mu + gamma + att[hi] - dfn[ai])
        la = np.exp(mu + att[ai] - dfn[hi])
        lh = np.clip(lh, 1e-6, 12); la = np.clip(la, 1e-6, 12)
        ll = np.sum(w * (lh - yh * np.log(lh) + la - ya * np.log(la)))
        if ridge > 0:
            ll += ridge * (np.sum((att - pa_) ** 2) + np.sum((dfn - pd_) ** 2))
        return ll

    p0 = np.concatenate([np.zeros(T - 1), np.zeros(T), [np.log(1.3)], [0.2]])
    res = minimize(nll, p0, method='L-BFGS-B',
                   options=dict(maxiter=3000, ftol=1e-10))
    att, dfn, mu, gamma = unpack(res.x)
    if ridge == 0:
        dfn = dfn - dfn.mean()  # identifiability: mu absorbs the level

    # --- stage 2: rho from actual scorelines ---
    lh = np.exp(mu + gamma + att[hi] - dfn[ai])
    la = np.exp(mu + att[ai] - dfn[hi])
    gh, ga = d['hg'].values, d['ag'].values

    def tau(h, a, lh, la, rho):
        t = np.ones_like(lh, dtype=float)
        m = (h == 0) & (a == 0); t[m] = 1 - lh[m] * la[m] * rho
        m = (h == 0) & (a == 1); t[m] = 1 + lh[m] * rho
        m = (h == 1) & (a == 0); t[m] = 1 + la[m] * rho
        m = (h == 1) & (a == 1); t[m] = 1 - rho
        return np.clip(t, 1e-6, None)

    def nll_rho(r):
        return -np.sum(w * np.log(tau(gh, ga, lh, la, r[0])))

    rr = minimize(nll_rho, [-0.05], method='L-BFGS-B', bounds=[(-0.25, 0.25)])
    rho = float(rr.x[0])

    return dict(teams=teams, att=dict(zip(teams, att)), dfn=dict(zip(teams, dfn)),
                mu=float(mu), gamma=float(gamma), rho=rho,
                n_matches=len(d), ref_date=ref)


def score_matrix(lh, la, rho, maxg=10):
    """Full scoreline probability matrix with Dixon-Coles low-score correction."""
    from math import factorial
    k = np.arange(maxg + 1)
    fac = np.array([factorial(int(i)) for i in k], dtype=float)
    ph = np.exp(-lh) * lh ** k / fac
    pa = np.exp(-la) * la ** k / fac
    M = np.outer(ph, pa)
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()


def lambdas(model, home, away, att_ov=None, dfn_ov=None):
    att = att_ov if att_ov is not None else model['att']
    dfn = dfn_ov if dfn_ov is not None else model['dfn']
    lh = np.exp(model['mu'] + model['gamma'] + att[home] - dfn[away])
    la = np.exp(model['mu'] + att[away] - dfn[home])
    return float(np.clip(lh, .05, 8)), float(np.clip(la, .05, 8))


def outcome_probs(model, home, away, **kw):
    lh, la = lambdas(model, home, away, **kw)
    M = score_matrix(lh, la, model['rho'])
    return dict(H=float(np.tril(M, -1).sum()), D=float(np.trace(M)),
                A=float(np.triu(M, 1).sum()), lh=lh, la=la)
