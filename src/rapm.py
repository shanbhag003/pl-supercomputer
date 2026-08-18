"""Regularized Adjusted Plus-Minus for Premier League players.

One row per team-match. The attacking side's players contribute an attack
coefficient, the defending side's players a defence coefficient, weighted by
share of minutes played. Ridge shrinks thin-sample players toward the league
average, so five appearances buys you a coefficient near zero rather than noise.

  team xG  =  intercept + home + season + Σ att_p·share_p + Σ def_q·share_q

Players below the minutes threshold are pooled into a per-club residual bucket
so their minutes are still controlled for.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

PM = f'{ROOT}/data/processed/player_matches.parquet'
MT = f'{ROOT}/data/processed/matches.parquet'


def build(min_total_minutes=900):
    pm = pd.read_parquet(PM)
    mt = pd.read_parquet(MT)[['date', 'season', 'home', 'away', 'hnpxg', 'anpxg']]

    pm = pm.merge(mt[['date', 'home', 'away']], left_on=['date', 'h_team', 'a_team'],
                  right_on=['date', 'home', 'away'], how='inner')
    pm['team'] = np.where(pm.name.notna(), None, None)

    # which side was the player on?
    tot = pm.groupby(['date', 'h_team', 'a_team']).mins.sum()
    keep = pm.groupby('pid').mins.sum()
    small = set(keep[keep < min_total_minutes].index)
    pm['key'] = pm.pid.where(~pm.pid.isin(small), 'POOL')

    return pm, mt


def fit(alpha=None, min_total_minutes=900, verbose=True):
    pm = pd.read_parquet(PM)
    mt = pd.read_parquet(MT)[['date', 'season', 'home', 'away', 'hnpxg', 'anpxg']]
    mt = mt[mt.season >= 2014].reset_index(drop=True)
    mt['mid'] = np.arange(len(mt))

    pm = pm.merge(mt[['date', 'home', 'away', 'mid', 'season']],
                  left_on=['date', 'h_team', 'a_team'],
                  right_on=['date', 'home', 'away'], how='inner',
                  suffixes=('', '_m'))
    pm['side'] = np.where(pm.h_team == pm.home, 'H', 'A')
    # a player's team in that match: infer from which club he appears for most
    # (Understat rows are per player, so use his season club via majority vote)
    pm['is_home'] = pm['side'] == 'H'

    # Understat player-match rows do not name the player's club, so infer it:
    # a player belongs to whichever of the two clubs he appears with most often.
    club = {}
    for pid, g in pm.groupby('pid'):
        c = pd.concat([g.loc[g.is_home, 'h_team'], g.loc[~g.is_home, 'a_team']])
        club[pid] = None
    # instead use per season-club majority from the league files
    import json, glob
    pcl = {}
    for f in sorted(glob.glob(f'{ROOT}/data/understat/EPL_*.json')):
        s = int(f.split('_')[-1].split('.')[0])
        for p in json.load(open(f))['players']:
            pcl[(p['id'], s)] = p['team_title']
    pm['club'] = [pcl.get((p, s)) for p, s in zip(pm.pid, pm.season)]
    pm = pm.dropna(subset=['club'])
    pm['on_home'] = pm.club == pm.home
    pm = pm[(pm.club == pm.home) | (pm.club == pm.away)]

    tot = pm.groupby(['mid', 'on_home']).mins.transform('sum')
    pm['share'] = pm.mins / tot.clip(lower=1)

    kmin = pm.groupby('pid').mins.sum()
    big = set(kmin[kmin >= min_total_minutes].index)
    pm['key'] = np.where(pm.pid.isin(big), pm.pid, 'POOL')

    players = sorted(set(pm.key))
    pidx = {p: i for i, p in enumerate(players)}
    P = len(players)
    seasons = sorted(mt.season.unique())
    sidx = {s: i for i, s in enumerate(seasons)}
    S = len(seasons)

    # two rows per match: home attacking, away attacking
    rows, cols, vals, y = [], [], [], []
    ncol = 2 * P + 1 + S          # att, def, home dummy, season dummies

    grp = {k: v for k, v in pm.groupby(['mid', 'on_home'])}
    for _, m in mt.iterrows():
        for att_home in (True, False):
            r = len(y)
            atk = grp.get((m.mid, att_home))
            dfd = grp.get((m.mid, not att_home))
            if atk is None or dfd is None:
                continue
            for k, sh in zip(atk.key, atk.share):
                rows.append(r); cols.append(pidx[k]); vals.append(sh)
            for k, sh in zip(dfd.key, dfd.share):
                rows.append(r); cols.append(P + pidx[k]); vals.append(sh)
            if att_home:
                rows.append(r); cols.append(2 * P); vals.append(1.0)
            rows.append(r); cols.append(2 * P + 1 + sidx[m.season]); vals.append(1.0)
            y.append(m.hnpxg if att_home else m.anpxg)

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(y), ncol))
    y = np.asarray(y)
    if verbose:
        print(f'design: {X.shape[0]} team-matches x {X.shape[1]} params '
              f'({P} players)')
    return X, y, players, P, S, seasons


def solve(X, y, damp):
    return lsqr(X, y, damp=damp, atol=1e-10, btol=1e-10, iter_lim=3000)[0]


def cv(X, y, damps, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.integers(0, folds, X.shape[0])
    out = []
    for d in damps:
        err = []
        for k in range(folds):
            tr, te = f != k, f == k
            b = solve(X[tr], y[tr], d)
            err.append(np.mean((X[te] @ b - y[te]) ** 2))
        out.append((d, float(np.mean(err))))
        print(f'  damp {d:<7} cv MSE {out[-1][1]:.5f}', flush=True)
    return out
