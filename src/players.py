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
    use), with its match_id and season.

    Built straight from the downloaded shot files, season taken from each
    file's name, so a newly finished season is picked up with nothing else to
    regenerate by hand (matches.parquet and the cached state table are not
    rebuilt automatically)."""
    import gamestate as G
    files = sorted(glob.glob(f'{ROOT}/data/understat/shots/EPL_*.parquet'))
    shots = pd.concat([pd.read_parquet(f).assign(season=int(f.split('_')[-1].split('.')[0]))
                       for f in files], ignore_index=True)
    season_of = shots.groupby('match_id').season.first()
    st = G.match_table(shots.drop(columns='season'))
    h, a = G.adjusted_npxg(st, beta)
    return st[['match_id', 'date', 'home', 'away']].assign(
        h_y=h, a_y=a, season=st.match_id.map(season_of).values)


def load_rosters(league='EPL'):
    parts = []
    for f in sorted(glob.glob(f'{ROSTERS}/{league}_*.parquet')):
        parts.append(pd.read_parquet(f).assign(season=int(f.split('_')[-1].split('.')[0])))
    r = pd.concat(parts, ignore_index=True)
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


FOREIGN = ['Bundesliga', 'La_liga', 'Serie_A', 'Ligue_1']


def foreign_targets(league, beta=0.1):
    """Game-state adjusted npxG per match for a foreign league, from its shots."""
    import gamestate as G
    files = sorted(glob.glob(f'{ROOT}/data/understat/shots/{league}_*.parquet'))
    shots = pd.concat([pd.read_parquet(f).assign(season=int(f.split('_')[-1].split('.')[0]))
                       for f in files], ignore_index=True)
    season_of = shots.groupby('match_id').season.first()
    st = G.match_table(shots.drop(columns='season'))
    h, a = G.adjusted_npxg(st, beta)
    return st[['match_id']].assign(h_y=h, a_y=a, season=st.match_id.map(season_of).values)


def foreign_values(league, before, years=4, damp=0.5, rosters=None, targets=None):
    """Informed plus-minus for one foreign league, fitted on the `years`
    seasons before `before` - nothing a transfer window could not know.
    Returns {player: net value} (attack minus def, in that league's units)."""
    ro = rosters if rosters is not None else load_rosters(league)
    tg = targets if targets is not None else foreign_targets(league)
    keep = set(range(before - years, before))
    tg = tg[tg.season.isin(keep)]
    rows, lu = team_matches(ro[ro.season.isin(keep)], tg)
    m = fit(rows, lu, damp, informed=True)
    mins = lu.groupby('player_id').time.sum()
    net = (m['att'] - m['dfn'])
    return net[mins.reindex(net.index).fillna(0) >= 900].to_dict()


FOREIGN_CACHE = f'{ROOT}/data/processed/foreign_values.parquet'


def foreign_value_table(windows, rebuild=False):
    """Foreign values for every league and transfer window (season the player
    would join for), cached. Where a player qualifies in two leagues the one
    he played more recently in wins. Columns: window, player_id, league, fv."""
    import os
    if os.path.exists(FOREIGN_CACHE) and not rebuild:
        t = pd.read_parquet(FOREIGN_CACHE)
        if set(windows) <= set(t.window):
            return t
    out = []
    for lg in FOREIGN:
        ro, tg = load_rosters(lg), foreign_targets(lg)
        for w in windows:
            before = ro[ro.season < w]
            last = before.groupby('player_id').season.max()
            v = foreign_values(lg, w, rosters=before, targets=tg)
            out.append(pd.DataFrame({'window': w, 'player_id': list(v), 'league': lg,
                                     'fv': list(v.values()),
                                     'last': [last.get(p, 0) for p in v]}))
    t = pd.concat(out, ignore_index=True)
    t = (t.sort_values('last').drop_duplicates(['window', 'player_id'], keep='last')
          .drop(columns='last'))
    t.to_parquet(FOREIGN_CACHE, index=False)
    return t


def newcomer_model_foreign(players_seasons, net, before, ftab):
    """newcomer_model plus the player's own value in a foreign league as of
    the window he joined in (ftab, from foreign_value_table). Learned from
    earlier newcomers only, weighted by minutes:

        value = a + b*club_prev_xgd + c*promoted + d*fv + e*has_fv

    Returns f(club_prev_xgd or None, fv or None) -> value."""
    ps = players_seasons
    first = ps.groupby('player_id').season.min()
    new = ps[(ps.season == ps.player_id.map(first)) & (ps.season > ps.season.min())
             & (ps.season < before)].copy()
    new['value'] = new.player_id.map(net)
    new = new.dropna(subset=['value'])
    prev = {s: club_strength(s - 1) for s in new.season.unique()}
    new['cx'] = [prev[s].get(c, np.nan) for c, s in zip(new.club, new.season)]
    fv = ftab.set_index(['window', 'player_id']).fv
    new['fv'] = [fv.get((s, p), np.nan) for s, p in zip(new.season, new.player_id)]

    def feats(cx, f):
        cx, f = np.asarray(cx, float), np.asarray(f, float)
        pro, has = ~np.isfinite(cx), np.isfinite(f)
        return np.column_stack([np.ones(len(cx)), np.where(pro, 0, cx), pro,
                                np.where(has, f, 0), has]).astype(float)
    X = feats(new.cx, new.fv)
    w = np.sqrt(new.mins.values)
    coef = np.linalg.lstsq(X * w[:, None], new.value.values * w, rcond=None)[0]
    return lambda cx, f: float((feats([np.nan if cx is None else cx],
                                      [np.nan if f is None else f]) @ coef)[0])


def club_strength(season):
    """Each club's non-penalty xG difference per game in the given season,
    from matches.parquet. Clubs absent that season (promoted) are missing."""
    mt = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
    mt = mt[mt.season == season]
    x = pd.concat([pd.DataFrame({'club': mt.home, 'xgd': mt.hnpxg - mt.anpxg}),
                   pd.DataFrame({'club': mt.away, 'xgd': mt.anpxg - mt.hnpxg})])
    return x.groupby('club').xgd.mean().to_dict()


def newcomer_model(players_seasons, net, before):
    """How good are players arriving with no Premier League record?

    players_seasons: player_id, season, club, mins (one row per player-season).
    net: {player: value} fitted on seasons before `before`. Learns, from
    newcomers who debuted in earlier seasons, value = a + b * (buying club's
    xG difference per game the season before), weighted by minutes; promoted
    clubs (no previous season) get their own minutes-weighted mean.
    Returns a function (club, club_prev_xgd or None) -> value."""
    ps = players_seasons
    first = ps.groupby('player_id').season.min()
    new = ps[(ps.season == ps.player_id.map(first)) & (ps.season > ps.season.min())
             & (ps.season < before)].copy()
    new['value'] = new.player_id.map(net)
    new = new.dropna(subset=['value'])
    prev = {s: club_strength(s - 1) for s in new.season.unique()}
    new['cx'] = [prev[s].get(c, np.nan) for c, s in zip(new.club, new.season)]
    pro = new[new.cx.isna()]
    est = new[new.cx.notna()]
    w = est.mins.values
    A = np.column_stack([np.ones(len(est)), est.cx.values]) * np.sqrt(w)[:, None]
    a, b = np.linalg.lstsq(A, est.value.values * np.sqrt(w), rcond=None)[0]
    promoted = float(np.average(pro.value, weights=pro.mins)) if len(pro) else float(a)
    return lambda cx: promoted if cx is None or not np.isfinite(cx) else float(a + b * cx)


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
