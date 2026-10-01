"""Expected FPL points vs real FPL points, and vs FPL's own expected points.

Every player-gameweek of 2021/22-2025/26 is predicted from data before the
gameweek (fpl_model.components), with scale factors and the bonus model
learned only from earlier seasons (2020/21 is calibration only). Scored per
player-gameweek - double gameweeks summed, blanks absent - over every listed
player, including those who did not play: a pre-gameweek forecast cannot
know who will be benched.

FPL's own xP (from the vaastav archive) is the benchmark for 2021/22-
2024/25; the archive's 2025/26 xP is broken (mean 0.33, correlation 0.21)
and is not used. FPL's xP also sees injury and suspension flags, which this
backtest does not have.

    python src/backtest_fpl.py build     # components, ~10 min, cached
    python src/backtest_fpl.py report
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, time, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, f'{ROOT}/src')
import fpl as F
import fpl_model as FM
import players as PL
warnings.filterwarnings('ignore')

LABELS = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
CACHE = f'{ROOT}/data/processed/fpl_components.parquet'
XP_SEASONS = [2021, 2022, 2023, 2024]


def build():
    t0 = time.time()
    ro = PL.load_rosters()
    hist = pd.concat([F.load_history(l) for l in LABELS], ignore_index=True)
    maps = {int(l[:4]): F.match_players(hist[hist.season == int(l[:4])], ro, F.web_names(l))
            for l in LABELS}
    lam = pd.read_parquet(f'{ROOT}/data/processed/fpl_team_lambdas.parquet')
    c = FM.components(hist, ro, lam, maps)
    c['person'] = c.person.astype(str)
    c.to_parquet(CACHE)
    print(f'{len(c)} player-fixtures in {time.time() - t0:.0f}s; '
          f'fixtures without a team prediction dropped: {len(hist) - len(c)}')


def per_gw(c, pts):
    """One row per player-gameweek: model and FPL expectations, actual points."""
    d = c.assign(model=pts.xpts.values)
    return d.groupby(['season', 'gw', 'element']).agg(
        model=('model', 'sum'), actual=('points', 'sum'), xP=('xP', 'first'),
        pos=('pos', 'first'), minutes=('minutes', 'sum'), person=('person', 'first'),
        n_fix=('points', 'size'), p_play=('p60', 'first'), psub=('psub', 'first')).reset_index()


def add_baselines(g, all_c):
    """Fair benchmarks from points already scored before each gameweek:
      form   mean points over the person's last 5 gameweeks (FPL's own xP is
             essentially this, scaled by fixture)
      ppm    points per appearance over the last 10, x chance he plays,
             x fixtures this gameweek
    """
    h = all_c.groupby(['person', 'season', 'gw']).agg(
        pts=('points', 'sum'), played=('minutes', lambda m: (m > 0).sum()),
        nfix=('points', 'size')).reset_index().sort_values(['person', 'season', 'gw'])
    grp = h.groupby('person')
    h['form'] = grp.pts.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    tot = grp.pts.transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    app = grp.played.transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    h['ppm'] = (tot / app.replace(0, np.nan)).fillna(0)
    g = g.merge(h[['person', 'season', 'gw', 'form', 'ppm']], on=['person', 'season', 'gw'], how='left')
    g['form'] = g.form.fillna(0) * g.n_fix
    g['ppm'] = g.ppm * (g.p_play.fillna(0.25) + g.psub.fillna(0.15)) * g.n_fix
    return g


def rank_quality(g, col, n=10):
    """Per gameweek: mean actual points of the top-n by col, and of the #1."""
    rows = []
    for (s, gw), x in g.groupby(['season', 'gw']):
        x = x.dropna(subset=[col])
        if len(x) < n:
            continue
        top = x.nlargest(n, col)
        rows.append(dict(season=s, gw=gw, topn=top.actual.mean(), captain=top.actual.iloc[0],
                         best=x.nlargest(n, 'actual').actual.mean()))
    return pd.DataFrame(rows)


def report():
    c = pd.read_parquet(CACHE)
    out = []
    for s in sorted(c.season.unique()):
        if s == c.season.min():
            continue                                      # calibration only
        cal = FM.calibrate(c[c.season < s])
        test = c[c.season == s]
        out.append(per_gw(test, FM.points(test, cal)))
        print(f"{s}: calibration from {sorted(c[c.season < s].season.unique())} - "
              f"goals x{cal['g']:.2f}, assists x{cal['a']:.2f}, clean sheets x{cal['cs']:.2f}")
    g = add_baselines(pd.concat(out, ignore_index=True), c)
    rng = np.random.default_rng(0)

    def ci(d):
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    pd.set_option('display.width', 200)
    print('\n=== every player-gameweek (benched players included): correlation with actual points ===')
    print(g.groupby('season').apply(lambda x: pd.Series(dict(
        n=len(x), model=x[['model', 'actual']].corr().iloc[0, 1],
        form=x[['form', 'actual']].corr().iloc[0, 1], ppm=x[['ppm', 'actual']].corr().iloc[0, 1],
        mae_model=(x.model - x.actual).abs().mean(), mae_form=(x.form - x.actual).abs().mean(),
        mean_model=x.model.mean(), mean_actual=x.actual.mean()))).round(3).to_string())

    print('\n=== picking players: mean actual points per gameweek ===')
    R = {k: rank_quality(g, k) for k in ('model', 'form', 'ppm')}
    t = pd.DataFrame({f'{k} top-10': v.groupby('season').topn.mean() for k, v in R.items()})
    for k, v in R.items():
        t[f'{k} captain'] = v.groupby('season').captain.mean()
    t['hindsight top-10'] = R['model'].groupby('season').best.mean()
    print(t.round(2).to_string())
    for k in ('form', 'ppm'):
        b = R['model'].merge(R[k], on=['season', 'gw'], suffixes=('_m', '_b'))
        d, dc = (b.topn_m - b.topn_b).values, (b.captain_m - b.captain_b).values
        lo, hi = ci(d); clo, chi = ci(dc)
        print(f'model minus {k}, {len(d)} gameweeks: top-10 {d.mean():+.3f} pts/player [{lo:+.3f}, {hi:+.3f}], '
              f'captain {dc.mean():+.3f} pts/GW [{clo:+.3f}, {chi:+.3f}], '
              f'better captain in {int((dc > 0).sum())} GWs, worse in {int((dc < 0).sum())}')

    print("\n=== FPL's archived xP is NOT a fair benchmark (recorded with hindsight) ===")
    x = g[g.season.isin(XP_SEASONS)]
    st = x[(x.p_play >= 0.9) & (x.minutes >= 60)]
    print(f"  starters who then scored 10+: FPL xP {st[st.actual >= 10].xP.mean():.2f}, model {st[st.actual >= 10].model.mean():.2f}; "
          f"who scored 2-3: FPL xP {st[st.actual.between(2, 3)].xP.mean():.2f}, model {st[st.actual.between(2, 3)].model.mean():.2f}")


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    build() if what == 'build' else report()
