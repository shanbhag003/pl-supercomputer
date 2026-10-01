"""Season-level test: does the squad layer forecast better with informed
player values?

The squad layer turns a change in squad quality into a rating shift: each
club's minutes-weighted player values this season against last season,
divided by league xG (squad_live.build), applied at SQUAD_W. This replays it
with player values from players.py, fitted before each season:

  plain1    RAPM shrunk toward zero, damp 1.0 (the live penalty)
  plain05   the same at damp 0.5, the best plain setting in backtest_players
  informed  RAPM shrunk toward stat-based predictions, damp 0.5

at squad weights 0, 0.25, 0.5 (live) and 1. As in backtest_squad.py, minute
shares come from the season being forecast - perfect knowledge of who played,
the optimistic version of the live minutes allocation - and promoted clubs
get no shift.

Replayed exactly as backtest_ranges.py (live settings, game-state adjusted xG,
live drift, common random numbers), pre-season plus GW 5/10/19/28, 2019/20 to
2025/26, and scored against final tables.

    python src/backtest_squad_players.py run
    python src/backtest_squad_players.py report
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, time, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, f'{ROOT}/src')
from simulate import bootstrap_models, fixture_grids, simulate, positions
import backtest_midseason as bm
from backtest_ranges import crps
import players as PL
warnings.filterwarnings('ignore')

OUT = f'{ROOT}/data/processed/squad_players_bt.parquet'
SEASONS = list(range(2019, 2026))
CHECKPOINTS = [0, 5, 10, 19, 28]
WEIGHTS = [0.25, 0.5, 1.0]
VALUES = {'plain1': dict(damp=1.0, informed=False),
          'plain05': dict(damp=0.5, informed=False),
          'informed': dict(damp=0.5, informed=True)}


def player_values(season, rows, lu):
    """Net quality per player (attack minus def, since def adds to the
    opponent's xG), fitted on seasons before this one."""
    tr = rows[rows.season < season]
    out = {}
    for name, kw in VALUES.items():
        m = PL.fit(tr, lu, kw['damp'], informed=kw['informed'])
        out[name] = (m['att'] - m['dfn']).to_dict()
    return out


def squad_delta(season, net, lu_season, base):
    """squad_live.build's delta: change in minutes-weighted net value."""
    def rating(s):
        x = lu_season[s]
        v = x.player_id.map(net).fillna(0.0)
        w = x.time / x.groupby('club').time.transform('sum')
        return (w * v).groupby(x.club).sum()
    new, old = rating(season), rating(season - 1)
    return {c: float((new[c] - old[c]) / base) if c in old.index else 0.0 for c in new.index}


def cell(args):
    season, gw = args
    t0 = time.time()
    import backtest_gamestate as bg
    df = bg.matches_with(0.1)                     # live xG for the ratings
    fin = bm.final_table(season)
    d, played, rest, teams, st, cutoff = bm.season_state(season, gw)
    start = {t: (v[0], v[1], v[2]) for t, v in st.items()}
    g = np.mean([v[3] for v in st.values()])
    fixtures = list(zip(rest.home, rest.away))
    prev = set(df[df.season == season - 1].home)
    pa = {t: 0.0 for t in prev}; pdf_ = {t: 0.0 for t in prev}
    mods = bootstrap_models(df[df.date < cutoff], cutoff, teams, pa, pdf_,
                            B=bm.B, seed=7, xi=bm.LIVE['xi'], **bm.BASE)
    dr = bm.drift_at(g, bm.LIVE['d0'], bm.LIVE['H'], bm.LIVE['floor'])
    z = np.random.default_rng(1000 * season + gw).normal(size=(bm.B, len(teams), 2))

    # player values (fitted before the season) and each season's minutes
    rows, lu = PL.team_matches(PL.load_rosters(), bg_targets())
    meta = bg_targets()[['match_id', 'season', 'home', 'away']]
    meta = meta.assign(match_id=meta.match_id.astype(str))
    lux = lu.merge(meta, on='match_id')
    lux['club'] = np.where(lux.side == 'h', lux.home, lux.away)
    lu_season = {s: lux[lux.season == s] for s in (season - 1, season)}
    base = float(df[df.season == season - 1].hnpxg.mean())
    vals = player_values(season, rows, lu)
    deltas = {k: squad_delta(season, v, lu_season, base) for k, v in vals.items()}

    variants = [('none', 0.0)] + [(k, w) for k in VALUES for w in WEIGHTS]
    ap = fin.loc[teams, 'pts'].values
    apos = fin.loc[teams, 'pos'].values
    out = []
    for name, w in variants:
        ms = []
        for b, m in enumerate(mods):
            att, dfn = dict(m['att']), dict(m['dfn'])
            for i, t in enumerate(teams):
                s = deltas[name].get(t, 0.0) * w if name != 'none' else 0.0
                att[t] += s / 2 + dr * z[b, i, 0]
                dfn[t] += s / 2 + dr * z[b, i, 1]
            ms.append(dict(m, att=att, dfn=dfn))
        pts, gd, gf = simulate(fixture_grids(ms, fixtures, teams), fixtures, teams,
                               N=bm.N, start=start, seed=11)
        pos = positions(pts, gd, gf, seed=12)
        for i, t in enumerate(teams):
            p = pts[:, i]
            out.append(dict(season=season, gw=gw, values=name, weight=w, team=t,
                            xpts=p.mean(), act=ap[i], act_pos=apos[i],
                            pit=(p < ap[i]).mean() + 0.5 * (p == ap[i]).mean(),
                            crps=crps(p, ap[i]), p_title=(pos[:, i] == 1).mean(),
                            p_top4=(pos[:, i] <= 4).mean(), p_releg=(pos[:, i] >= 18).mean(),
                            delta=deltas[name].get(t, 0.0) if name != 'none' else 0.0))
    print(f'  {season} GW{gw}: {time.time() - t0:.0f}s', flush=True)
    return out


def bg_targets():
    import backtest_players as bp
    return bp.targets()


def run():
    cells = [(s, g) for s in SEASONS for g in CHECKPOINTS]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as p:
        rows = sum(p.map(cell, cells), [])
    pd.DataFrame(rows).to_parquet(OUT)


def report():
    r = pd.read_parquet(OUT)
    r['abserr'] = (r.act - r.xpts).abs()
    r['b_title'] = (r.p_title - (r.act_pos == 1)) ** 2
    r['b_top4'] = (r.p_top4 - (r.act_pos <= 4)) ** 2
    r['b_releg'] = (r.p_releg - (r.act_pos >= 18)) ** 2
    c = r.groupby(['values', 'weight', 'season', 'gw']).agg(
        mae=('abserr', 'mean'), crps=('crps', 'mean'), b_title=('b_title', 'sum'),
        b_top4=('b_top4', 'sum'), b_releg=('b_releg', 'sum'))
    pd.set_option('display.width', 200)
    for label, sel in (('pre-season only (7 forecasts)', c.index.get_level_values('gw') == 0),
                       ('all checkpoints (35 forecasts)', slice(None))):
        print(f'\n=== {label} ===')
        print(c[sel].groupby(['values', 'weight']).mean().round(4).to_string())

    live = c.xs(('plain1', 0.5), level=('values', 'weight'))
    rng = np.random.default_rng(0)
    print('\n=== vs live (plain1 at weight 0.5), per forecast ===')
    for k, w in [('none', 0.0)] + [(k, w) for k in VALUES for w in WEIGHTS]:
        if (k, w) == ('plain1', 0.5):
            continue
        x = c.xs((k, w), level=('values', 'weight'))
        for met in ('mae', 'b_title'):
            dlt = (x[met] - live[met]).values
            bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f'  {k:<9} w={w:<5} {met:<8} {dlt.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  '
                  f'better in {int((dlt < 0).sum())}/{len(dlt)}')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    run() if what == 'run' else report()
