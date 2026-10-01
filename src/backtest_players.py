"""Do informed priors make player values better at predicting teams?

For each season 2019/20-2025/26, fit player values on every earlier season,
then predict each team-match's game-state adjusted xG from the actual lineups.
That is what the squad layer relies on: given who plays, how good is the side?

  none      everyone at zero - level and home advantage only
  plain     RAPM shrunk toward zero (as squad_live uses)
  plain900  the same, but players under 900 training minutes count as zero
  informed  RAPM shrunk toward each player's stat-based prediction

Players with no minutes before the season count as zero in every model: using
their test-season numbers would leak the answer. The ridge strength (damp) is
chosen on 2017/18-2018/19 and held fixed.

Scored at club-season level (mean xG for and against per club over the
season, error vs actual) and team-match level, with the season's level offset
removed so promoted-era scoring shifts do not count.

    python scripts/pull_understat_rosters.py 2014 2026   # once
    python src/backtest_players.py
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
import players as PL
import gamestate as G
warnings.filterwarnings('ignore')

OUT = f'{ROOT}/data/processed/players_bt.parquet'
TUNE, TEST = [2017, 2018], list(range(2019, 2026))
DAMPS = [0.25, 0.5, 1.0, 2.0, 4.0]
BETA = 0.1


def targets():
    return PL.training_targets(BETA)


def job(args):
    season, model, damp = args
    t0 = time.time()
    rows, lu = PL.team_matches(PL.load_rosters(), targets())
    tr, te = rows[rows.season < season], rows[rows.season == season].reset_index(drop=True)
    if model == 'none':
        lvl = tr[tr.season == season - 1].y.mean()
        home = tr[tr.home == 1].y.mean() - tr[tr.home == 0].y.mean()
        pred = lvl + home * (te.home.values - 0.5)
    else:
        m = PL.fit(tr, lu, damp, informed=(model == 'informed'),
                   min_mins=900 if model == 'plain900' else 0)
        pred = PL.predict(m, te, lu)
    out = te.assign(pred=pred, model=model, damp=damp)
    sides = lu[lu.match_id.isin(set(te.match_id))][['match_id', 'side']].drop_duplicates()
    print(f'  {season} {model} damp={damp}: {time.time() - t0:.0f}s', flush=True)
    return out


def run():
    jobs = [(s, 'none', 0.0) for s in TUNE + TEST]
    jobs += [(s, m, d) for s in TUNE + TEST for m in ('plain', 'plain900', 'informed') for d in DAMPS]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as p:
        out = p.map(job, jobs)
    pd.concat(out, ignore_index=True).to_parquet(OUT)


def club_table(r):
    """Per club-season: mean actual and predicted xG for and against."""
    team = r.copy()
    m = targets()[['match_id', 'home', 'away']].rename(columns={'home': 'hteam', 'away': 'ateam'})
    m['match_id'] = m.match_id.astype(str)
    team = team.merge(m, on='match_id')
    team['club'] = np.where(team.side == 'h', team.hteam, team.ateam)
    team['opp'] = np.where(team.side == 'h', team.ateam, team.hteam)
    # remove the season's level offset, which no player model can know
    team['resid'] = team.y - team.pred
    team['pred_adj'] = team.pred + team.groupby(['model', 'damp', 'season']).resid.transform('mean')
    f = team.groupby(['model', 'damp', 'season', 'club'])[['y', 'pred_adj']].mean()
    a = team.groupby(['model', 'damp', 'season', 'opp'])[['y', 'pred_adj']].mean()
    a.index = a.index.set_names('club', level='opp')
    return f.join(a, lsuffix='_for', rsuffix='_ag')


def report():
    r = pd.read_parquet(OUT)
    c = club_table(r)
    c['err_for'] = (c.y_for - c.pred_adj_for).abs()
    c['err_ag'] = (c.y_ag - c.pred_adj_ag).abs()
    s = c.groupby(['model', 'damp', 'season'])[['err_for', 'err_ag']].mean()
    s['err'] = s.mean(1)
    tune = s[s.index.get_level_values('season').isin(TUNE)].groupby(['model', 'damp']).err.mean()
    test = s[s.index.get_level_values('season').isin(TEST)].groupby(['model', 'damp']).err.mean()
    pd.set_option('display.width', 200)
    print('\n=== club-season mean xG error per match (for+against)/2, by damp ===')
    print(pd.DataFrame({'tune 2017-18': tune, 'test 2019-25': test}).round(4).to_string())
    best = {m: tune.xs(m, level='model').idxmin() for m in ('plain', 'plain900', 'informed')}
    print('\ndamp chosen on tuning seasons:', best)
    rows = {'none': s.xs(('none', 0.0), level=('model', 'damp'))}
    for m, d in best.items():
        rows[m] = s.xs((m, d), level=('model', 'damp'))
    t = pd.DataFrame({m: v.err for m, v in rows.items()})
    t = t[t.index.isin(TEST)]
    print('\n=== held-out, chosen damp: error by season ===')
    print(t.round(4).to_string())
    print('mean', t.mean().round(4).to_dict())
    d = t.informed - t.plain
    print(f'informed vs plain: {d.mean():+.4f} per club-match, better in {int((d < 0).sum())}/{len(d)} seasons')
    # paired over club-seasons
    ci = c.reset_index()
    pick = lambda m: ci[(ci.model == m) & (ci.damp == best[m]) & ci.season.isin(TEST)] \
        .set_index(['season', 'club'])[['err_for', 'err_ag']].mean(1)
    dd = (pick('informed') - pick('plain')).values
    rng = np.random.default_rng(0)
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'paired over {len(dd)} club-seasons: {dd.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    run() if what == 'run' else report()
