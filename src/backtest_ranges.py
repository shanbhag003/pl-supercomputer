"""Mid-season ranges: does an in-season random walk fix the coverage?

backtest_midseason.py found the published 10th-90th percentile points ranges
contain the final total only ~72% of the time mid-season. Holding the drift at
its pre-season size fixes coverage but worsens title and relegation Brier, so
more uniform noise is not the answer.

The hypothesis: in the simulation a club's strength is frozen from the
checkpoint to May, so nothing that happens after today - injuries, sackings,
form - can move it. fixture_grids_walk adds a weekly random walk to every
rating. Its spread grows with the weeks remaining, so it widens early-season
ranges most and late ones least, and it moves a club's whole run-in together.

Prior for sigma: the measured summer-to-summer change (0.17 SD/year), if it
accumulated as a random walk, would be ~0.024 per week.

Same replay as backtest_midseason.py (live settings, 8 seasons x GW 5/10/19/28,
common random numbers). Decision metric is CRPS of final points - a proper
score for the whole distribution, so it cannot be gamed by just widening -
alongside coverage and the event Brier scores.

    python src/backtest_ranges.py run
    python src/backtest_ranges.py report
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
from simulate import bootstrap_models, fixture_grids_walk, simulate, positions
from backtest_midseason import (df, season_state, final_table, drift_at,
                                SEASONS, CHECKPOINTS, B, N, BASE, LIVE)
warnings.filterwarnings('ignore')

OUT = f'{ROOT}/data/processed/ranges_bt.parquet'
SIGMAS = [0.0, 0.01, 0.015, 0.02, 0.025, 0.03]


def crps(samples, y):
    """CRPS from Monte Carlo samples: E|X - y| - E|X - X'| / 2."""
    x = np.sort(samples)
    n = len(x)
    exx = 2 * np.sum(x * (2 * np.arange(1, n + 1) - n - 1)) / n ** 2
    return np.abs(x - y).mean() - exx / 2


def cell(args):
    season, gw = args
    t0 = time.time()
    fin = final_table(season)
    d, played, rest, teams, st, cutoff = season_state(season, gw)
    start = {t: (v[0], v[1], v[2]) for t, v in st.items()}
    g = np.mean([v[3] for v in st.values()])
    fixtures = list(zip(rest.home, rest.away))
    weeks = ((rest.date - cutoff).dt.days // 7).values
    prev = set(df[df.season == season - 1].home)
    pa = {t: 0.0 for t in prev}; pdf_ = {t: 0.0 for t in prev}
    mods = bootstrap_models(df[df.date < cutoff], cutoff, teams, pa, pdf_,
                            B=B, seed=7, xi=LIVE['xi'], **BASE)
    dr = drift_at(g, LIVE['d0'], LIVE['H'], LIVE['floor'])
    z = np.random.default_rng(1000 * season + gw).normal(size=(B, len(teams), 2))
    ms = []
    for b, m in enumerate(mods):                  # live drift, as in update.py
        att, dfn = dict(m['att']), dict(m['dfn'])
        for i, t in enumerate(teams):
            att[t] += dr * z[b, i, 0]; dfn[t] += dr * z[b, i, 1]
        ms.append(dict(m, att=att, dfn=dfn))

    ap = fin.loc[teams, 'pts'].values
    apos = fin.loc[teams, 'pos'].values
    now = np.array([st[t][0] for t in teams])
    rows = []
    for sg in SIGMAS:
        cum = fixture_grids_walk(ms, fixtures, weeks, teams, sg, seed=season * 100 + gw)
        pts, gd, gf = simulate(cum, fixtures, teams, N=N, start=start, seed=11)
        pos = positions(pts, gd, gf, seed=12)
        for i, t in enumerate(teams):
            p = pts[:, i]
            rows.append(dict(season=season, gw=gw, sigma=sg, team=t, games=g,
                             pts_now=now[i], xpts=p.mean(), act=ap[i], act_pos=apos[i],
                             lo10=np.percentile(p, 10), hi90=np.percentile(p, 90),
                             pit=(p < ap[i]).mean() + 0.5 * (p == ap[i]).mean(),
                             crps=crps(p, ap[i]),
                             p_title=(pos[:, i] == 1).mean(),
                             p_top4=(pos[:, i] <= 4).mean(),
                             p_releg=(pos[:, i] >= 18).mean()))
    print(f'  {season} GW{gw}: {time.time() - t0:.0f}s', flush=True)
    return rows


def run():
    cells = [(s, g) for s in SEASONS for g in CHECKPOINTS]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as p:
        rows = sum(p.map(cell, cells), [])
    pd.DataFrame(rows).to_parquet(OUT)


def summarise(r):
    """Per-cell metrics, then averaged: Brier is summed over the 20 clubs of a
    season, like backtest_midseason.py."""
    r = r.assign(cov80=(r.pit > .1) & (r.pit < .9), cov50=(r.pit > .25) & (r.pit < .75),
                 width=r.hi90 - r.lo10, abserr=(r.act - r.xpts).abs(),
                 b_title=(r.p_title - (r.act_pos == 1)) ** 2,
                 b_top4=(r.p_top4 - (r.act_pos <= 4)) ** 2,
                 b_releg=(r.p_releg - (r.act_pos >= 18)) ** 2)
    c = r.groupby(['sigma', 'season', 'gw']).agg(
        crps=('crps', 'mean'), mae=('abserr', 'mean'), cov80=('cov80', 'mean'),
        cov50=('cov50', 'mean'), width=('width', 'mean'),
        b_title=('b_title', 'sum'), b_top4=('b_top4', 'sum'), b_releg=('b_releg', 'sum'))
    return r, c


def report():
    r, c = summarise(pd.read_parquet(OUT))
    pd.set_option('display.width', 200)
    print(f'\n=== all 32 cells ({r.season.nunique()} seasons x {r.gw.nunique()} checkpoints) ===')
    print(c.groupby('sigma').mean().round(4).to_string())

    print('\n=== 80% coverage by checkpoint ===')
    print(c.groupby(['sigma', 'gw']).cov80.mean().unstack().round(3).to_string())
    print('\n=== CRPS by checkpoint ===')
    print(c.groupby(['sigma', 'gw']).crps.mean().unstack().round(3).to_string())

    base = c.xs(0.0, level='sigma')
    rng = np.random.default_rng(0)
    print('\n=== vs sigma=0 (live), per cell: CRPS diff, cells better, split halves ===')
    for sg in [s for s in r.sigma.unique() if s > 0]:
        dlt = (c.xs(sg, level='sigma').crps - base.crps)
        bs = [dlt.sample(len(dlt), replace=True, random_state=int(rng.integers(1e9))).mean()
              for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        early = dlt[dlt.index.get_level_values('season') <= 2021].mean()
        late = dlt[dlt.index.get_level_values('season') >= 2022].mean()
        print(f'  sigma {sg:<6} {dlt.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  '
              f'better in {int((dlt < 0).sum())}/{len(dlt)} cells  '
              f'2018-21 {early:+.4f}  2022-25 {late:+.4f}')

    print('\n=== diagnosis at sigma=0: where do the misses fall? ===')
    l = r[r.sigma == 0]
    print(f'  finished below the range (PIT < 0.1): {(l.pit < .1).mean():.1%}   '
          f'finished above it (PIT > 0.9): {(l.pit > .9).mean():.1%}   (10% each if calibrated)')
    l = l.assign(strength=pd.qcut(l.groupby(['season', 'gw']).xpts.rank(ascending=False),
                                  [0, .25, .75, 1], labels=['top 5', 'middle 10', 'bottom 5']))
    print(l.groupby('strength', observed=True).apply(lambda x: pd.Series(dict(
        cov80=((x.pit > .1) & (x.pit < .9)).mean(), below=(x.pit < .1).mean(),
        above=(x.pit > .9).mean(), bias=(x.act - x.xpts).mean()))).round(3).to_string())


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    run() if what == 'run' else report()
