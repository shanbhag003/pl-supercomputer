"""Player values: plus-minus (RAPM) with an informed prior.

Each team-match is one observation: the side's game-state adjusted npxG
(gamestate.py, as the live ratings use), explained by who was on the pitch.
Every player has an attack value (adds to his side's xG) and a defence value
(adds to the opponent's), weighted by his share of his side's minutes:

    xG = level + home + sum(att_p * share_p) + sum(def_q * share_q)

Plain RAPM shrinks every value toward zero, league average. In football that
is a heavy pull: substitutions are few and the same players share the pitch,
so on-pitch results alone separate players poorly.

The informed prior (as in basketball's box-score-prior RAPM) shrinks each
player toward what his own numbers suggest instead. Stage 1 fits plain RAPM.
Stage 2 regresses the stage-1 values on per-90 stats (xG, xA, xGChain,
xGBuildup, position), weighted by minutes. Stage 3 refits RAPM shrunk toward
that prediction. All three stages see training data only.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

ROSTERS = f'{ROOT}/data/understat/rosters'
STATS = ['xG', 'xA', 'xGChain', 'xGBuildup']
PSEUDO_MIN = 900          # per-90 rates are shrunk toward position means by this


def training_targets(beta=0.1):
    """Per match: game-state adjusted npxG for each side (as the live ratings
    use), with its match_id and season."""
    import gamestate as G
    st = pd.read_parquet(f'{ROOT}/data/processed/gamestate_matches.parquet')
    h, a = G.adjusted_npxg(st, beta)
    t = st[['match_id', 'date', 'home', 'away']].assign(h_y=h, a_y=a)
    mt = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')[['date', 'home', 'away', 'season']]
    return t.merge(mt, on=['date', 'home', 'away'])


def load_rosters():
    r = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{ROSTERS}/EPL_*.parquet'))],
                  ignore_index=True)
    r['match_id'] = r.match_id.astype(str)
    r['pos'] = r.position.str[0].map({'G': 'GK', 'D': 'D', 'M': 'M', 'F': 'F', 'A': 'M',
                                      'S': 'F'}).fillna('M')
    return r[r.time > 0]


def team_matches(rosters, targets):
    """targets: match_id, season, h_y, a_y. Returns long team-match rows
    (match_id, season, side, home, y) and lineups (match_id, side, player, share)."""
    t = targets.copy()
    t['match_id'] = t.match_id.astype(str)
    rows = pd.concat([
        t.assign(side='h', home=1, y=t.h_y)[['match_id', 'season', 'side', 'home', 'y']],
        t.assign(side='a', home=0, y=t.a_y)[['match_id', 'season', 'side', 'home', 'y']]],
        ignore_index=True)
    lu = rosters[['match_id', 'h_a', 'player_id', 'time', 'pos'] + STATS].copy()
    lu['share'] = lu.time / lu.groupby(['match_id', 'h_a']).time.transform('sum')
    return rows, lu.rename(columns={'h_a': 'side'})


def design(rows, lu, players):
    """Sparse X: [att (P) | def (P) | home | season dummies]."""
    pidx = {p: i for i, p in enumerate(players)}
    P = len(players)
    seasons = sorted(rows.season.unique())
    sidx = {s: i for i, s in enumerate(seasons)}
    key = {(m, s): i for i, (m, s) in enumerate(zip(rows.match_id, rows.side))}
    other = {'h': 'a', 'a': 'h'}
    r_, c_, v_ = [], [], []
    for m, s, p, sh in zip(lu.match_id, lu.side, lu.player_id, lu.share):
        if p not in pidx:
            continue
        i = key.get((m, s))                    # this side attacking: att
        if i is not None:
            r_.append(i); c_.append(pidx[p]); v_.append(sh)
        j = key.get((m, other[s]))             # opponent attacking: def
        if j is not None:
            r_.append(j); c_.append(P + pidx[p]); v_.append(sh)
    n = len(rows)
    r_ += list(range(n)); c_ += [2 * P] * n; v_ += list(rows.home.astype(float))
    r_ += list(range(n)); c_ += [2 * P + 1 + sidx[s] for s in rows.season]; v_ += [1.0] * n
    X = sparse.csr_matrix((v_, (r_, c_)), shape=(n, 2 * P + 1 + len(seasons)))
    return X, seasons


def solve(X, y, damp, P, prior=None):
    """Ridge on the 2P player columns only (toward prior if given); level,
    home and season terms unpenalised. Returns coefficients."""
    b0 = np.zeros(X.shape[1])
    if prior is not None:
        b0[:2 * P] = prior
    resid = y - X @ b0
    # penalise only player columns: augment with damp * I on those columns
    pen = sparse.csr_matrix((np.full(2 * P, damp), (np.arange(2 * P), np.arange(2 * P))),
                            shape=(2 * P, X.shape[1]))
    A = sparse.vstack([X, pen]).tocsr()
    rhs = np.concatenate([resid, np.zeros(2 * P)])
    d = lsqr(A, rhs, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    return b0 + d


def player_stats(lu):
    """Per-90 stats per player, shrunk toward his position's mean by
    PSEUDO_MIN minutes, plus his modal position and total minutes."""
    g = lu.groupby('player_id')
    mins = g.time.sum()
    pos = g.pos.agg(lambda s: s.value_counts().index[0])
    tot = g[STATS].sum()
    rate = tot.div(mins, axis=0) * 90
    pos_mean = (tot.groupby(pos).sum().div(mins.groupby(pos).sum(), axis=0) * 90)
    pm = pos_mean.loc[pos.values].set_index(pos.index)
    shrunk = (rate.mul(mins, axis=0) + pm.mul(PSEUDO_MIN, axis=0)).div(mins + PSEUDO_MIN, axis=0)
    out = shrunk.add_suffix('90')
    out['mins'] = mins
    out['pos'] = pos
    return out


def stat_features(st):
    F = st[[c + '90' for c in STATS]].copy()
    for p in ('D', 'M', 'F'):
        F[f'is_{p}'] = (st.pos == p).astype(float)
    F['const'] = 1.0
    return F


def fit_prior_map(st, att, dfn, min_mins=900):
    """Weighted least squares of stage-1 values on stats, players with enough
    minutes for their value to mean something. Returns (w_att, w_def)."""
    keep = st.index[(st.mins >= min_mins) & st.index.isin(att.index)]
    F = stat_features(st.loc[keep]).values
    w = np.sqrt(st.loc[keep, 'mins'].values)
    wa = np.linalg.lstsq(F * w[:, None], att.loc[keep].values * w, rcond=None)[0]
    wd = np.linalg.lstsq(F * w[:, None], dfn.loc[keep].values * w, rcond=None)[0]
    return wa, wd


def fit(rows, lu, damp, informed=False, damp_prior=None, min_mins=0):
    """Fit player values on the given (training) rows. Returns dict with
    att, def (Series by player), level terms, and the prior map if informed."""
    lu = lu[lu.match_id.isin(set(rows.match_id))]
    mins = lu.groupby('player_id').time.sum()
    players = sorted(mins.index[mins >= min_mins])
    P = len(players)
    X, seasons = design(rows, lu, players)
    y = rows.y.values
    b = solve(X, y, damp, P)
    att = pd.Series(b[:P], index=players)
    dfn = pd.Series(b[P:2 * P], index=players)
    out = dict(att=att, dfn=dfn, home=b[2 * P], season=dict(zip(seasons, b[2 * P + 1:])),
               players=players)
    if informed:
        st = player_stats(lu)
        wa, wd = fit_prior_map(st, att, dfn)
        F = stat_features(st.loc[players]).values
        prior = np.concatenate([F @ wa, F @ wd])
        b = solve(X, y, damp_prior or damp, P, prior=prior)
        out.update(att=pd.Series(b[:P], index=players), dfn=pd.Series(b[P:2 * P], index=players),
                   home=b[2 * P], season=dict(zip(seasons, b[2 * P + 1:])),
                   prior_map=(wa, wd))
    return out


def predict(model, rows, lu, prior_for_unknown=None):
    """Predicted team-match target from lineups, using the last training
    season's level. Unknown players get 0, or prior_for_unknown(player) ->
    (att, def) if given."""
    att, dfn = model['att'].to_dict(), model['dfn'].to_dict()
    if prior_for_unknown is not None:
        for p in set(lu.player_id) - set(att):
            a, d = prior_for_unknown(p)
            att[p], dfn[p] = a, d
    lu = lu[lu.match_id.isin(set(rows.match_id))]
    a_part = (lu.player_id.map(att).fillna(0) * lu.share).groupby([lu.match_id, lu.side]).sum()
    d_part = (lu.player_id.map(dfn).fillna(0) * lu.share).groupby([lu.match_id, lu.side]).sum()
    other = rows.side.map({'h': 'a', 'a': 'h'})
    level = model['season'][max(model['season'])]
    pa = a_part.reindex(list(zip(rows.match_id, rows.side))).fillna(0).values
    pd_ = d_part.reindex(list(zip(rows.match_id, other))).fillna(0).values
    return level + model['home'] * rows.home.values + pa + pd_
