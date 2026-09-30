"""Mid-season backtest: how good is the forecast at GW 5, 10, 19 and 28?

The pre-season backtest scores one forecast per season. The live model makes
thirty-eight, and every one after the first mixes last season's ratings with
this season's evidence. How fast it should trust the new evidence is set by
three numbers that have never been tested in-season:

  xi     time decay of match weights (tuned on match RPS, not on tables)
  boost  extra weight on current-season matches (new; 1.0 = live behaviour)
  drift  extra rating noise, decaying as games are played. Live:
         max(0.04, 0.16 * (1 - games/12))

Each season 2018/19-2025/26 is replayed as the live pipeline would have seen
it at each checkpoint: same priors, same ridge, same bootstrap, actual points
as the starting table, remaining fixtures simulated. Scored against the final
table.

Out of scope: the squad and manager layers. The player-match data they need is
not in the repository, and neither layer changes during a season anyway, so the
in-season settings tested here act on the same club-ratings core.

Baselines, so the numbers mean something:
  ppg      points now + points-per-game x games left
  frozen   pre-season ratings, never updated, actual points as the start
  market   points now + de-vigged pre-closing odds for every remaining match
           (collected days before each one). Not achievable - they are set with
           information the checkpoint could not have. A ceiling, not a rival.

    python src/backtest_midseason.py run      # resumable, uses all cores
    python src/backtest_midseason.py report
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
warnings.filterwarnings('ignore')

df = pd.read_parquet(f'{ROOT}/data/processed/matches.parquet')
OUT = f'{ROOT}/data/processed/midseason_bt.csv'

SEASONS = list(range(2018, 2026))      # every season with 4 years of history
CHECKPOINTS = [5, 10, 19, 28]
B, N = 40, 10000
BASE = dict(w_xg=0.7, ridge=2.0)
FITS = [dict(xi=xi, boost=b) for xi in (0.003, 0.0045, 0.007, 0.01)
        for b in (1.0, 2.0, 3.0)]
# (start, games to decay to the floor, floor). H=None means no decay.
DRIFTS = [(0.16, 12, 0.04),            # live
          (0.16, 6, 0.04), (0.16, 24, 0.04), (0.16, 38, 0.04),
          (0.16, 12, 0.0), (0.16, 12, 0.08), (0.16, None, 0.16)]
LIVE = dict(xi=0.0045, boost=1.0, d0=0.16, H=12, floor=0.04)
KEY = ['season', 'gw', 'xi', 'boost']


def drift_at(g, d0, H, floor):
    return d0 if H is None else max(floor, d0 * (1 - g / H))


def season_state(season, gw):
    """Matches played before the checkpoint, the rest, and the table then."""
    d = df[df.season == season].sort_values('date', kind='stable')
    cutoff = d.date.iloc[gw * 10]           # first match not yet played
    played, rest = d[d.date < cutoff], d[d.date >= cutoff]
    teams = sorted(set(d.home))
    st = {t: [0, 0, 0, 0] for t in teams}   # pts, gf, ga, played
    for r in played.itertuples():
        st[r.home][1] += r.hg; st[r.home][2] += r.ag; st[r.home][3] += 1
        st[r.away][1] += r.ag; st[r.away][2] += r.hg; st[r.away][3] += 1
        if r.hg > r.ag: st[r.home][0] += 3
        elif r.ag > r.hg: st[r.away][0] += 3
        else: st[r.home][0] += 1; st[r.away][0] += 1
    return d, played, rest, teams, st, cutoff


def final_table(season):
    d = df[df.season == season]
    st = {}
    for r in d.itertuples():
        for t in (r.home, r.away):
            st.setdefault(t, [0, 0, 0])
        st[r.home][1] += r.hg - r.ag; st[r.away][1] += r.ag - r.hg
        st[r.home][2] += r.hg; st[r.away][2] += r.ag
        if r.hg > r.ag: st[r.home][0] += 3
        elif r.ag > r.hg: st[r.away][0] += 3
        else: st[r.home][0] += 1; st[r.away][0] += 1
    a = pd.DataFrame([(k, *v) for k, v in st.items()], columns=['team', 'pts', 'gd', 'gf'])
    a = a.sort_values(['pts', 'gd', 'gf'], ascending=False).reset_index(drop=True)
    a['pos'] = np.arange(1, len(a) + 1)
    return a.set_index('team')


def score(pts, pos, teams, fin):
    """Point-forecast accuracy, interval calibration and event Brier scores."""
    ap = fin.loc[teams, 'pts'].values
    apos = fin.loc[teams, 'pos'].values
    xp = pts.mean(0)
    err = ap - xp
    # mid-PIT: points are discrete, so count ties as half below
    pit = (pts < ap).mean(0) + 0.5 * (pts == ap).mean(0)

    def brier(p, y):
        return float(((p - y) ** 2).sum())
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                rankcorr=float(pd.Series(xp).corr(pd.Series(ap), method='spearman')),
                cover80=float(((pit > .1) & (pit < .9)).mean()),
                b_title=brier((pos == 1).mean(0), apos == 1),
                b_top4=brier((pos <= 4).mean(0), apos <= 4),
                b_releg=brier((pos >= 18).mean(0), apos >= 18))


def run_sims(mods, z, teams, fixtures, start, dr, fin):
    """Perturb every bootstrap model by dr * z (common random numbers across
    variants), simulate the rest of the season, score it."""
    ms = []
    for b, m in enumerate(mods):
        att, dfn = dict(m['att']), dict(m['dfn'])
        for i, t in enumerate(teams):
            att[t] += dr * z[b, i, 0]
            dfn[t] += dr * z[b, i, 1]
        ms.append(dict(m, att=att, dfn=dfn))
    cum = fixture_grids(ms, fixtures, teams)
    pts, gd, gf = simulate(cum, fixtures, teams, N=N, start=start, seed=11)
    pos = positions(pts, gd, gf, seed=12)
    return score(pts, pos, teams, fin)


def task(args):
    season, fit = args
    t0 = time.time()
    fin = final_table(season)
    prev = set(df[df.season == season - 1].home)
    pa = {t: 0.0 for t in prev}; pdf_ = {t: 0.0 for t in prev}
    rows = []
    frozen = fit == 'frozen'
    if frozen:        # fitted once, the day before the season, never updated
        d0 = df[df.season == season].date.min()
        teams = sorted(set(df[df.season == season].home))
        mods0 = bootstrap_models(df, d0, teams, pa, pdf_, B=B, seed=7,
                                 xi=LIVE['xi'], **BASE)
    for gw in CHECKPOINTS:
        d, played, rest, teams, st, cutoff = season_state(season, gw)
        start = {t: (v[0], v[1], v[2]) for t, v in st.items()}
        g = np.mean([v[3] for v in st.values()])
        fixtures = list(zip(rest.home, rest.away))
        z = np.random.default_rng(1000 * season + gw).normal(size=(B, len(teams), 2))
        if frozen:
            m = run_sims(mods0, z, teams, fixtures, start, 0.16, fin)
            rows.append(dict(season=season, gw=gw, xi=np.nan, boost=np.nan,
                             d0=0.16, H=None, floor=0.16, variant='frozen',
                             games=g, **m))
            continue
        # the live pipeline fits on everything before today, current season included
        mods = bootstrap_models(df[df.date < cutoff], cutoff, teams, pa, pdf_,
                                B=B, seed=7, xi=fit['xi'], boost=fit['boost'],
                                boost_from=d.date.min(), **BASE)
        for d0, H, floor in DRIFTS:
            m = run_sims(mods, z, teams, fixtures, start, drift_at(g, d0, H, floor), fin)
            rows.append(dict(season=season, gw=gw, xi=fit['xi'], boost=fit['boost'],
                             d0=d0, H=H, floor=floor, variant='model',
                             games=g, **m))
    print(f'  {season} {fit}: {time.time() - t0:.0f}s', flush=True)
    return rows


def baselines():
    """PPG extrapolation and the pre-match-odds reference. No simulation needed,
    so only the point-forecast metrics are reported."""
    rows = []
    for season in SEASONS:
        fin = final_table(season)
        for gw in CHECKPOINTS:
            d, played, rest, teams, st, _ = season_state(season, gw)
            ap = fin.loc[teams, 'pts'].values
            now = np.array([st[t][0] for t in teams], float)
            p = np.array([st[t][3] for t in teams], float)
            ppg = now + now / np.maximum(p, 1) * (38 - p)
            r = rest.dropna(subset=['AvgH', 'AvgD', 'AvgA'])
            inv = 1 / r[['AvgH', 'AvgD', 'AvgA']].values
            pr = inv / inv.sum(1, keepdims=True)
            mk = dict(zip(teams, now))
            for (h, a), (ph, pd_, pa_) in zip(zip(r.home, r.away), pr):
                mk[h] += 3 * ph + pd_; mk[a] += 3 * pa_ + pd_
            mk = np.array([mk[t] for t in teams])
            for name, xp in (('ppg', ppg), ('market', mk)):
                err = ap - xp
                rows.append(dict(season=season, gw=gw, variant=name,
                                 mae=np.abs(err).mean(),
                                 rmse=np.sqrt((err ** 2).mean()),
                                 rankcorr=pd.Series(xp).corr(pd.Series(ap),
                                                             method='spearman')))
    return pd.DataFrame(rows)


def run():
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    have = set()
    if len(done):
        m = done[done.variant == 'model']
        have = {(int(s), float(x), float(b)) for s, x, b in zip(m.season, m.xi, m.boost)}
        have |= {(int(s), 'frozen') for s in done[done.variant == 'frozen'].season}
    jobs = [(s, 'frozen') for s in SEASONS if (s, 'frozen') not in have]
    jobs += [(s, f) for s in SEASONS for f in FITS
             if (s, f['xi'], f['boost']) not in have]
    print(f'{len(jobs)} jobs to run', flush=True)
    rows = done.to_dict('records')
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for r in pool.imap_unordered(task, jobs):
            rows += r
            pd.DataFrame(rows).to_csv(OUT, index=False)   # resumable


def report():
    r = pd.read_csv(OUT)
    r['H'] = r.H.fillna(-1)          # -1 = no decay
    mets = ['mae', 'rankcorr', 'cover80', 'b_title', 'b_top4', 'b_releg']
    live = r[(r.variant == 'model') & (r.xi == LIVE['xi']) & (r.boost == LIVE['boost'])
             & (r.d0 == LIVE['d0']) & (r.H == LIVE['H']) & (r.floor == LIVE['floor'])]
    frz = r[r.variant == 'frozen']
    bl = baselines()
    pd.set_option('display.width', 200)

    print('\n=== Live settings vs baselines, mean over 8 seasons ===')
    t = pd.concat([
        live.groupby('gw')[mets].mean().assign(model='live'),
        frz.groupby('gw')[mets].mean().assign(model='frozen pre-season'),
        bl[bl.variant == 'ppg'].groupby('gw')[['mae', 'rankcorr']].mean().assign(model='points-per-game'),
        bl[bl.variant == 'market'].groupby('gw')[['mae', 'rankcorr']].mean().assign(model='pre-match odds (ceiling)'),
    ]).reset_index().set_index(['gw', 'model']).sort_index()
    print(t.round(3).to_string())

    # every variant against live, on the same season x checkpoint cells
    m = r[r.variant == 'model']
    cols = ['xi', 'boost', 'H', 'floor']
    lv = live.set_index(['season', 'gw'])[mets]
    out = []
    for k, g in m.groupby(cols):
        g = g.set_index(['season', 'gw'])[mets]
        dlt = g - lv.loc[g.index]
        out.append(dict(zip(cols, k), mae=g.mae.mean(), d_mae=dlt.mae.mean(),
                        mae_wins=int((dlt.mae < 0).sum()), cells=len(dlt),
                        seasons_better=int((dlt.mae.groupby('season').mean() < 0).sum()),
                        rankcorr=g.rankcorr.mean(), cover80=g.cover80.mean(),
                        b_title=g.b_title.mean(), b_top4=g.b_top4.mean(),
                        b_releg=g.b_releg.mean()))
    v = pd.DataFrame(out).sort_values('mae')
    print('\n=== All variants vs live (d_mae < 0 is better; wins out of season x checkpoint cells) ===')
    print(v.head(15).round(4).to_string(index=False))
    print('\n--- drift schedules only (xi, boost at live values) ---')
    print(v[(v.xi == LIVE['xi']) & (v.boost == LIVE['boost'])].round(4).to_string(index=False))

    best = v.iloc[0]
    bb = m[(m.xi == best.xi) & (m.boost == best.boost) & (m.H == best.H) & (m.floor == best.floor)]
    print(f'\n=== Best variant by checkpoint: xi={best.xi} boost={best.boost} '
          f'H={best.H} floor={best.floor} ===')
    cmp = pd.concat([live.groupby('gw')[mets].mean().assign(model='live'),
                     bb.groupby('gw')[mets].mean().assign(model='best')])
    print(cmp.reset_index().set_index(['gw', 'model']).sort_index().round(3).to_string())
    print('\nper-season MAE (all checkpoints), live vs best:')
    ps = pd.DataFrame({'live': live.groupby('season').mae.mean(),
                       'best': bb.groupby('season').mae.mean()})
    ps['diff'] = ps.best - ps.live
    print(ps.round(3).to_string())


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    run() if what == 'run' else report()
