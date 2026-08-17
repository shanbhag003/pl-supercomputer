"""Manager effects, fitted alongside player values.

Club fixed effects are included so a manager's coefficient is identified only
from what changed WITHIN a club when the manager changed. Without them a
manager at one club for the whole window would simply absorb that club's
quality.

  team xG = club FE + manager + Σ player shares + home + season
"""
import glob, json, sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

sys.path.insert(0, '/home/claude/pl/src')

PMC = '/home/claude/pl/data/processed/pm_merged.parquet'
MGR = '/home/claude/pl/data/processed/managers.csv'


def match_managers(mt):
    m = pd.read_csv(MGR, parse_dates=['start', 'end'])
    by = {c: g.sort_values('start') for c, g in m.groupby('club')}

    def who(club, date):
        g = by.get(club)
        if g is None:
            return 'UNKNOWN'
        s = g[(g.start <= date) & (g.end >= date)]
        if len(s):
            return s.iloc[-1].manager
        p = g[g.start <= date]
        return p.iloc[-1].manager if len(p) else 'UNKNOWN'

    return ([who(r.home, r.date) for r in mt.itertuples()],
            [who(r.away, r.date) for r in mt.itertuples()])


def build_and_fit(cutoff=None, lam_p=1.0, lam_m=2.0, min_mgr_matches=15,
                  verbose=True):
    mt = pd.read_parquet('/home/claude/pl/data/processed/matches.parquet')
    mt = mt[mt.season >= 2014].reset_index(drop=True)
    mt['mid'] = np.arange(len(mt))
    if cutoff is not None:
        mt = mt[mt.date < pd.Timestamp(cutoff)].reset_index(drop=True)
    h, a = match_managers(mt)
    mt['mgr_h'], mt['mgr_a'] = h, a

    pm = pd.read_parquet(PMC)
    pm = pm[pm.mid.isin(set(mt.mid))]

    km = pm.groupby('pid').mins.sum()
    big = set(km[km >= 900].index)
    pm = pm.assign(key=np.where(pm.pid.isin(big), pm.pid, 'POOL'))

    players = sorted(set(pm.key)); pidx = {p: i for i, p in enumerate(players)}
    P = len(players)
    cnt = pd.concat([mt.mgr_h, mt.mgr_a]).value_counts()
    mgrs = sorted(cnt[cnt >= min_mgr_matches].index)
    midx = {m: i for i, m in enumerate(mgrs)}; M = len(mgrs)
    clubs = sorted(set(mt.home) | set(mt.away))
    cidx = {c: i for i, c in enumerate(clubs)}; C = len(clubs)
    seasons = sorted(mt.season.unique()); sidx = {s: i for i, s in enumerate(seasons)}
    S = len(seasons)

    # columns: P att | P def | M mgr-att | M mgr-def | C club-att | C club-def | home | S
    O_PD, O_MA, O_MD = 0, 2 * P, 2 * P + M
    O_CA, O_CD = 2 * P + 2 * M, 2 * P + 2 * M + C
    O_H, O_S = 2 * P + 2 * M + 2 * C, 2 * P + 2 * M + 2 * C + 1
    ncol = O_S + S

    grp = {k: v for k, v in pm.groupby(['mid', 'on_home'])}
    rows, cols, vals, y = [], [], [], []
    for r in mt.itertuples():
        for ah in (True, False):
            atk, dfd = grp.get((r.mid, ah)), grp.get((r.mid, not ah))
            if atk is None or dfd is None:
                continue
            i = len(y)
            for k, sh in zip(atk.key, atk.share):
                rows.append(i); cols.append(pidx[k]); vals.append(sh)
            for k, sh in zip(dfd.key, dfd.share):
                rows.append(i); cols.append(P + pidx[k]); vals.append(sh)
            am, dm = (r.mgr_h, r.mgr_a) if ah else (r.mgr_a, r.mgr_h)
            ac, dc = (r.home, r.away) if ah else (r.away, r.home)
            if am in midx:
                rows.append(i); cols.append(O_MA + midx[am]); vals.append(1.0)
            if dm in midx:
                rows.append(i); cols.append(O_MD + midx[dm]); vals.append(1.0)
            rows.append(i); cols.append(O_CA + cidx[ac]); vals.append(1.0)
            rows.append(i); cols.append(O_CD + cidx[dc]); vals.append(1.0)
            if ah:
                rows.append(i); cols.append(O_H); vals.append(1.0)
            rows.append(i); cols.append(O_S + sidx[r.season]); vals.append(1.0)
            y.append(r.hnpxg if ah else r.anpxg)

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(y), ncol))
    y = np.asarray(y)
    if verbose:
        print(f'design {X.shape}  ({P} players, {M} managers, {C} clubs)')

    pen_idx = np.concatenate([np.arange(0, 2 * P), np.arange(O_MA, O_MA + 2 * M)])
    pen_val = np.concatenate([np.full(2 * P, lam_p), np.full(2 * M, lam_m)])

    def solve(Xs, ys):
        Pen = sparse.csr_matrix((pen_val, (np.arange(len(pen_idx)), pen_idx)),
                                shape=(len(pen_idx), ncol))
        return lsqr(sparse.vstack([Xs, Pen]).tocsr(),
                    np.concatenate([ys, np.zeros(len(pen_idx))]),
                    atol=1e-10, btol=1e-10, iter_lim=5000)[0]

    b = solve(X, y)
    res = dict(
        att=dict(zip(players, b[:P])), dfn=dict(zip(players, -b[P:2 * P])),
        known=big, managers=mgrs,
        mgr_att=dict(zip(mgrs, b[O_MA:O_MA + M])),
        mgr_dfn=dict(zip(mgrs, -b[O_MD:O_MD + M])),
        home=float(b[O_H]), X=X, y=y, solve=solve)
    return res


def cv_lambda(X, y, ncol, pen_idx, lams_p, lams_m, folds=4, seed=0):
    rng = np.random.default_rng(seed); f = rng.integers(0, folds, X.shape[0])
    out = []
    for lp in lams_p:
        for lm in lams_m:
            pv = np.concatenate([np.full(pen_idx[0], lp), np.full(pen_idx[1], lm)])
            out.append((lp, lm))
    return out
