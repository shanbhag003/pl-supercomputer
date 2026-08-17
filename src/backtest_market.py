"""Does the market correction actually improve pre-season forecasts?

For each season: fit the model as of the day before it started, extract the
market adjustment from opening-day odds, apply it at several weights, simulate,
and score against what really happened. Resumable — writes after each season.
"""
import sys, os, copy, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/claude/pl/src')
warnings.filterwarnings('ignore')

from ratings import fit_ratings
from simulate import bootstrap_models, fixture_grids, simulate, positions, summarise
from market import opening_odds, fit_market_delta
from preseason_bt import actual_table

CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)
DRIFT, B, N = 0.16, 60, 20000
MKT_RIDGE = 0.10
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT = '/home/claude/pl/data/processed/market_bt.csv'

df = pd.read_parquet('/home/claude/pl/data/processed/matches.parquet')


def season_code(s):
    return f'{str(s)[2:]}{str(s + 1)[2:]}'


def run_season(season):
    d = df[df.season == season].sort_values('date')
    ref = d.date.min() - pd.Timedelta(days=1)
    teams = sorted(set(d.home))
    prev = set(df[df.season == season - 1].home)
    pa = {t: 0.0 for t in prev}
    pdf = {t: 0.0 for t in prev}

    base = fit_ratings(df, ref, prior_att=pa, prior_dfn=pdf,
                       extra_teams=teams, **CFG)
    odds = opening_odds(f'/home/claude/pl/data/raw/E0_{season_code(season)}.csv')
    delta, info = fit_market_delta(base, odds, teams, ridge=MKT_RIDGE)

    mods0 = bootstrap_models(df, ref, teams, pa, pdf, B=B, seed=7, **CFG)
    rng = np.random.default_rng(77)
    noise = {t: (rng.normal(0, DRIFT), rng.normal(0, DRIFT))
             for m in mods0 for t in m['teams']}
    fx = list(zip(d.home, d.away))
    act = actual_table(season)

    out = []
    for w in WEIGHTS:
        mods = copy.deepcopy(mods0)
        r2 = np.random.default_rng(99)
        for m in mods:
            for t in m['teams']:
                shift = delta.get(t, 0.0) * w
                m['att'][t] += shift / 2 + r2.normal(0, DRIFT)
                m['dfn'][t] += shift / 2 + r2.normal(0, DRIFT)
        cum = fixture_grids(mods, fx, teams)
        pts, gd, gf = simulate(cum, fx, teams, N=N, seed=11)
        pos = positions(pts, gd, gf, seed=12)
        s = summarise(pts, pos, teams).merge(act, on='team')
        ti = {t: i for i, t in enumerate(teams)}
        s['pct'] = [(pts[:, ti[t]] < a).mean() for t, a in zip(s.team, s.act_pts)]
        err = (s.act_pts - s.xPts)
        champ = s.loc[s.act_pos == 1]
        out.append(dict(
            season=season, weight=w, mkt_rmse=info['rmse'],
            mae=err.abs().mean(),
            rankcorr=s[['rank', 'act_pos']].corr(method='spearman').iloc[0, 1],
            cover80=((s.pct > .10) & (s.pct < .90)).mean(),
            champ_rank=int(champ['rank'].iloc[0]),
            champ_prob=float(champ.title.iloc[0]),
            rel_caught=int((s[s.act_pos >= 18]['rank'] >= 18).sum()),
        ))
        print(f'  w={w:<5} MAE {out[-1]["mae"]:.2f}  rank {out[-1]["rankcorr"]:.3f}  '
              f'cover {out[-1]["cover80"]:.0%}  champ rank {out[-1]["champ_rank"]}',
              flush=True)
    return pd.DataFrame(out), delta


if __name__ == '__main__':
    todo = [int(x) for x in sys.argv[1:]]
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    rows = done.to_dict('records')
    for s in todo:
        if len(done) and s in set(done.season):
            print(f'{s} already done')
            continue
        print(f'--- {s}', flush=True)
        r, _ = run_season(s)
        rows += r.to_dict('records')
        pd.DataFrame(rows).to_csv(OUT, index=False)
