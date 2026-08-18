"""Does the manager layer improve pre-season forecasts, on top of squads?

Baseline here is squad-on (weight 0.5), matching what is live. We then add the
manager change delta at several weights.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, copy, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, f'{ROOT}/src')
warnings.filterwarnings('ignore')

from simulate import bootstrap_models, fixture_grids, simulate, positions, summarise
from preseason_bt import actual_table
from manager_model import build_and_fit, match_managers
import backtest_squad as bs

CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)
DRIFT, B, N, SQUAD_W = 0.16, 60, 20000, 0.5
WEIGHTS = [0.0, 0.25, 0.5, 1.0]
OUT = f'{ROOT}/data/processed/mgr_bt_within.csv'


def manager_delta(season, ref):
    """Change in manager quality between the end of last season and now."""
    fitr = build_and_fit(cutoff=ref, verbose=False)
    net = {m: fitr['mgr_att'][m] + fitr['mgr_dfn'][m] for m in fitr['managers']}

    mt = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
    prev = mt[mt.season == season - 1]
    cur = mt[mt.season == season]
    # manager at the END of last season vs at the START of this one
    last_h, last_a = match_managers(prev.tail(60))
    p = prev.tail(60).assign(mh=last_h, ma=last_a)
    end_mgr = {}
    for r in p.itertuples():
        end_mgr[r.home] = r.mh
        end_mgr[r.away] = r.ma
    first_h, first_a = match_managers(cur.head(30))
    c = cur.head(30).assign(mh=first_h, ma=first_a)
    start_mgr = {}
    for r in c.itertuples():
        start_mgr.setdefault(r.home, r.mh)
        start_mgr.setdefault(r.away, r.ma)

    base = float(prev.hnpxg.mean())
    delta, changed, skipped = {}, [], []
    for club, new in start_mgr.items():
        old = end_mgr.get(club)
        if old is None or old == new:
            delta[club] = 0.0
            continue
        # Only score a change when BOTH managers have a measured PL record.
        # Otherwise the "delta" is just minus the outgoing manager, which tests
        # the unknown-manager prior rather than manager effects themselves.
        if new not in net or old not in net:
            delta[club] = 0.0
            skipped.append((club, old, new))
            continue
        d = (net[new] - net[old]) / base
        delta[club] = d
        changed.append((club, old, new, round(d, 3)))
    return delta, changed, skipped


def run(season):
    mt = bs.mt
    d = mt[mt.season == season].sort_values('date')
    ref = d.date.min() - pd.Timedelta(days=1)
    teams = sorted(set(d.home))
    prev = set(mt[mt.season == season - 1].home)
    pa = {t: 0.0 for t in prev}; pdf = {t: 0.0 for t in prev}

    att, dfn, known = bs.fit_rapm(ref)
    cur_s = bs.squad_rating(season, att, dfn, known)
    old_s = bs.squad_rating(season - 1, att, dfn, known)
    base = float(mt[mt.season == season - 1].hnpxg.mean())
    sq = {t: (((cur_s.loc[t, 'A'] + cur_s.loc[t, 'D'])
               - (old_s.loc[t, 'A'] + old_s.loc[t, 'D'])) / base)
          if (t in cur_s.index and t in old_s.index) else 0.0 for t in teams}

    mg, changed, skipped = manager_delta(season, ref)
    print(f'   scored {len(changed)}, skipped {len(skipped)} (new manager had no '
          f'PL record) -> {changed}', flush=True)

    mods0 = bootstrap_models(mt, ref, teams, pa, pdf, B=B, seed=7, **CFG)
    fx = list(zip(d.home, d.away))
    act = actual_table(season)
    out = []
    for w in WEIGHTS:
        mods = copy.deepcopy(mods0)
        r2 = np.random.default_rng(99)
        for m in mods:
            for t in m['teams']:
                s = sq.get(t, 0.0) * SQUAD_W + mg.get(t, 0.0) * w
                m['att'][t] += s / 2 + r2.normal(0, DRIFT)
                m['dfn'][t] += s / 2 + r2.normal(0, DRIFT)
        cum = fixture_grids(mods, fx, teams)
        pts, gd, gf = simulate(cum, fx, teams, N=N, seed=11)
        pos = positions(pts, gd, gf, seed=12)
        s = summarise(pts, pos, teams).merge(act, on='team')
        ti = {t: i for i, t in enumerate(teams)}
        s['pct'] = [(pts[:, ti[t]] < a).mean() for t, a in zip(s.team, s.act_pts)]
        err = s.act_pts - s.xPts
        out.append(dict(season=season, mgr_weight=w, mae=err.abs().mean(),
                        rankcorr=s[['rank', 'act_pos']].corr(method='spearman').iloc[0, 1],
                        cover80=((s.pct > .10) & (s.pct < .90)).mean(),
                        n_changes=len(changed)))
        print(f'  mgr_w={w:<5} MAE {out[-1]["mae"]:.2f}  rank {out[-1]["rankcorr"]:.3f}  '
              f'cover {out[-1]["cover80"]:.0%}', flush=True)
    return pd.DataFrame(out)


if __name__ == '__main__':
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    rows = done.to_dict('records')
    for s in [int(x) for x in sys.argv[1:]]:
        if len(done) and s in set(done.season):
            print(f'{s} done'); continue
        print(f'--- {s}', flush=True)
        rows += run(s).to_dict('records')
        pd.DataFrame(rows).to_csv(OUT, index=False)
