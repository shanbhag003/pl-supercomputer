"""Does game-state adjusted xG improve match predictions?

Walk-forward exactly as backtest.py (weekly refits, never sees a result before
predicting it), 2019/20-2025/26. The only change is the xG each past match
feeds into the ratings: every non-penalty shot rescaled by gamestate.factors
(beta). beta=0 is the shots summed with no adjustment; 'orig' is the npxG
column already in matches.parquet, which should reproduce VALIDATION.md s.1.

    python scripts/pull_understat_shots.py 2014 2025   # once
    python src/backtest_gamestate.py run
    python src/backtest_gamestate.py report
    python src/backtest_gamestate.py season   # season-table check, ~10 min
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
from ratings import fit_ratings, outcome_probs
import gamestate as G
warnings.filterwarnings('ignore')

TABLE = f'{ROOT}/data/processed/gamestate_matches.parquet'
OUT = f'{ROOT}/data/processed/gamestate_bt.parquet'
SEASONS = list(range(2019, 2026))
BETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)


def state_table():
    if os.path.exists(TABLE):
        return pd.read_parquet(TABLE)
    m = G.match_table(G.load_shots())
    print(f'state table: {len(m)} matches, '
          f'{len(m.attrs["unreconciled"])} with unreconciled scores dropped')
    m.to_parquet(TABLE)
    return m


def matches_with(beta):
    """matches.parquet with hnpxg/anpxg replaced by adjusted xG. Matches with no
    usable shot data keep their original npxG."""
    mt = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
    if beta == 'orig':
        return mt
    if beta == 0:                      # shots summed, unadjusted - apply() skips 0
        st = state_table()
        h, a = G.adjusted_npxg(st, 0.0)
        adj = st[['date', 'home', 'away']].assign(h_adj=h, a_adj=a)
        mt = mt.merge(adj, on=['date', 'home', 'away'], how='left')
        mt['hnpxg'] = mt.h_adj.fillna(mt.hnpxg)
        mt['anpxg'] = mt.a_adj.fillna(mt.anpxg)
        return mt.drop(columns=['h_adj', 'a_adj'])
    return G.apply(mt, beta)[0]        # exactly what update.py runs


def run_season(args):
    """backtest.run_season, on a given version of the match data."""
    beta, season = args
    t0 = time.time()
    df = matches_with(beta)
    d = df[df.season == season].sort_values('date')
    prev = set(df[df.season == season - 1].home)
    pa = {t: 0.0 for t in prev}; pdf_ = {t: 0.0 for t in prev}
    teams = sorted(set(d.home))
    out, model, last = [], None, None
    for date, chunk in d.groupby('date'):
        if last is None or (date - last).days >= 7:
            model = fit_ratings(df, date, prior_att=pa, prior_dfn=pdf_,
                                extra_teams=teams, **CFG)
            last = date
        for m in chunk.itertuples():
            o = outcome_probs(model, m.home, m.away)
            res = 0 if m.hg > m.ag else (1 if m.hg == m.ag else 2)
            out.append(dict(beta=str(beta), season=season, date=date, home=m.home,
                            away=m.away, res=res, pH=o['H'], pD=o['D'], pA=o['A']))
    print(f'  beta={beta} {season}: {time.time() - t0:.0f}s', flush=True)
    return out


def run():
    state_table()                     # build the cache once, before forking
    jobs = [(b, s) for b in ['orig'] + BETAS for s in SEASONS]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as p:
        rows = sum(p.map(run_season, jobs), [])
    pd.DataFrame(rows).to_parquet(OUT)


def report():
    r = pd.read_parquet(OUT)
    P = r[['pH', 'pD', 'pA']].values
    y = r.res.values
    cp, co = np.cumsum(P, 1), np.cumsum(np.eye(3)[y], 1)
    r['rps'] = ((cp - co) ** 2).sum(1) / 2
    r['ll'] = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))
    r['hit'] = P.argmax(1) == y
    pd.set_option('display.width', 200)
    order = ['orig'] + [str(b) for b in BETAS]
    print(f'\n=== {r[r.beta == "orig"].shape[0]} matches per variant, '
          f'{r.season.min()}-{r.season.max()} ===')
    print(r.groupby('beta')[['rps', 'll', 'hit']].mean().loc[order].round(5).to_string())
    print('\n=== RPS by season ===')
    print(r.pivot_table(index='season', columns='beta', values='rps')[order].round(5).to_string())

    base = r[r.beta == '0.0'].set_index(['date', 'home', 'away']).rps
    rng = np.random.default_rng(0)
    print('\n=== paired vs beta=0 (shots, unadjusted) ===')
    for b in order:
        if b == '0.0':
            continue
        v = r[r.beta == b].set_index(['date', 'home', 'away']).rps
        d = (v - base.loc[v.index]).values
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        by = pd.Series(d).groupby(r[r.beta == b].season.values).mean()
        print(f'  {b:<5} {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  '
              f'better in {int((by < 0).sum())}/{len(by)} seasons  '
              f'2019-21 {by.loc[2019:2021].mean():+.5f}  2022-25 {by.loc[2022:].mean():+.5f}')


SEASON_OUT = f'{ROOT}/data/processed/gamestate_season_bt.parquet'


def _init_season(beta):
    """Pool initializer: point the mid-season harness at this version of the data."""
    import backtest_midseason as bm, backtest_ranges as br
    d = matches_with(beta)
    bm.df = d; br.df = d
    br.SIGMAS = [0.0]


def _cell(args):
    import backtest_ranges as br
    return br.cell(args)


def season_level():
    """The season-table replay of backtest_midseason / backtest_ranges (live
    settings, 8 seasons x GW 5/10/19/28), before and after the adjustment."""
    import backtest_midseason as bm, backtest_ranges as br
    state_table()
    out = []
    for beta in ['orig', 0.1]:
        with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init_season,
                  initargs=(beta,)) as p:
            rows = sum(p.map(_cell, [(s, g) for s in bm.SEASONS for g in bm.CHECKPOINTS]), [])
        out.append(pd.DataFrame(rows).assign(beta=str(beta)))
    r = pd.concat(out)
    r.to_parquet(SEASON_OUT)
    cov = lambda x, q: ((x.pit > (1 - q) / 2) & (x.pit < (1 + q) / 2)).mean()
    for b, g in r.groupby('beta'):
        _, c = br.summarise(g.drop(columns='beta'))
        print(b, c.mean().round(4).to_dict(), f' 7th-93rd coverage {cov(g, .86):.3f}')
    a = r[r.beta == 'orig'].set_index(['season', 'gw', 'team'])
    b = r[r.beta == '0.1'].set_index(['season', 'gw', 'team'])
    print('CRPS diff by season:', (b.crps - a.crps).groupby('season').mean().round(3).to_dict())
    print('MAE diff by season:', ((b.act - b.xpts).abs() - (a.act - a.xpts).abs())
          .groupby('season').mean().round(3).to_dict())


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    {'run': run, 'season': season_level}.get(what, report)()
