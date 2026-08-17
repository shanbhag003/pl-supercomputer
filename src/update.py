"""Weekly pipeline. Run after every gameweek.

  1. refresh data (Understat xG + football-data.co.uk results/odds)
  2. rebuild the current table from matches actually played
  3. refit ratings, simulate the remaining fixtures 20,000 times
  4. validate: score our own past predictions against what happened
  5. render the PNG and write a changelog explaining what moved

Designed to run headless in GitHub Actions.
"""
import os, sys, json, time, warnings, datetime as dt
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
warnings.filterwarnings('ignore')

from ratings import fit_ratings
from simulate import bootstrap_models, fixture_grids, simulate, positions
import render as R
import gate, mailer

SEASON = 2026                     # Understat label for 2026/27
FD_CODE = '2627'
CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)
BASE_DRIFT, B, N = 0.16, 80, 20000
UA = {'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'}

DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(f'{DATA}/raw', exist_ok=True)
os.makedirs(f'{DATA}/understat', exist_ok=True)
os.makedirs(OUT, exist_ok=True)

FIX2US = {'Man Utd': 'Manchester United', 'Man City': 'Manchester City',
          'Newcastle': 'Newcastle United', 'Spurs': 'Tottenham',
          "Nott'm Forest": 'Nottingham Forest'}
FD2US = dict(FIX2US, **{'Man United': 'Manchester United',
                        'Tottenham': 'Tottenham', 'Wolves': 'Wolverhampton Wanderers'})


def log(m):
    print(f'[{dt.datetime.now():%H:%M:%S}] {m}', flush=True)


# ---------------------------------------------------------------- data refresh
def refresh():
    """Pull current-season data. Returns (understat_ok, n_played)."""
    for code in [FD_CODE]:
        for div in ['E0', 'E1']:
            try:
                r = requests.get(
                    f'https://www.football-data.co.uk/mmz4281/{code}/{div}.csv',
                    timeout=40)
                if r.ok and len(r.content) > 200:
                    open(f'{DATA}/raw/{div}_{code}.csv', 'wb').write(r.content)
            except Exception as e:
                log(f'  {div}_{code} fetch failed: {e}')
    ok = False
    try:
        r = requests.get(f'https://understat.com/getLeagueData/EPL/{SEASON}',
                         headers=UA, timeout=40)
        if r.ok:
            json.dump(r.json(), open(f'{DATA}/understat/EPL_{SEASON}.json', 'w'))
            ok = True
    except Exception as e:
        log(f'  understat fetch failed: {e}')
    return ok


def current_season_matches():
    """Played 2026/27 matches with xG where available, goals otherwise."""
    rows = []
    p = f'{DATA}/understat/EPL_{SEASON}.json'
    if os.path.exists(p):
        j = json.load(open(p))
        hist = {}
        for t in (j['teams'].values() if isinstance(j['teams'], dict) else j['teams']):
            for g in t['history']:
                hist[(t['title'], g['date'][:10], g['h_a'])] = g
        for m in j['dates']:
            if not m.get('isResult'):
                continue
            d = m['datetime'][:10]
            gh, ga = hist.get((m['h']['title'], d, 'h')), hist.get((m['a']['title'], d, 'a'))
            rows.append(dict(season=SEASON, date=d, home=m['h']['title'],
                             away=m['a']['title'],
                             hg=int(m['goals']['h']), ag=int(m['goals']['a']),
                             hxg=float(m['xG']['h']), axg=float(m['xG']['a']),
                             hnpxg=float(gh['npxG']) if gh else float(m['xG']['h']),
                             anpxg=float(ga['npxG']) if ga else float(m['xG']['a'])))
    if not rows:   # fallback: goals only, no xG yet
        f = f'{DATA}/raw/E0_{FD_CODE}.csv'
        if os.path.exists(f):
            o = pd.read_csv(f, encoding='latin-1')
            if 'Div' in o.columns:              # guard: file can contain other divisions
                o = o[o['Div'] == 'E0']
            o = o.dropna(subset=['FTHG'])
            for _, r in o.iterrows():
                h = FD2US.get(r.HomeTeam, r.HomeTeam); a = FD2US.get(r.AwayTeam, r.AwayTeam)
                rows.append(dict(season=SEASON,
                                 date=pd.to_datetime(r.Date, dayfirst=True).strftime('%Y-%m-%d'),
                                 home=h, away=a, hg=int(r.FTHG), ag=int(r.FTAG),
                                 hxg=np.nan, axg=np.nan,
                                 hnpxg=float(r.FTHG), anpxg=float(r.FTAG)))
    d = pd.DataFrame(rows)
    if len(d):
        d['date'] = pd.to_datetime(d['date'])
    return d


# ---------------------------------------------------------------------- table
def build_table(played, teams):
    st = {t: dict(P=0, W=0, D=0, L=0, GF=0, GA=0, Pts=0) for t in teams}
    for _, r in played.iterrows():
        if r.home not in st or r.away not in st:
            continue
        st[r.home]['P'] += 1; st[r.away]['P'] += 1
        st[r.home]['GF'] += r.hg; st[r.home]['GA'] += r.ag
        st[r.away]['GF'] += r.ag; st[r.away]['GA'] += r.hg
        if r.hg > r.ag:
            st[r.home]['W'] += 1; st[r.away]['L'] += 1; st[r.home]['Pts'] += 3
        elif r.ag > r.hg:
            st[r.away]['W'] += 1; st[r.home]['L'] += 1; st[r.away]['Pts'] += 3
        else:
            st[r.home]['D'] += 1; st[r.away]['D'] += 1
            st[r.home]['Pts'] += 1; st[r.away]['Pts'] += 1
    return st


# ------------------------------------------------------------------ validation
def validate(played, teams):
    """Score our own previously published match predictions."""
    f = f'{OUT}/match_predictions.csv'
    if not os.path.exists(f) or not len(played):
        return None
    mp = pd.read_csv(f)
    mp['date'] = pd.to_datetime(mp['date'])
    m = played.merge(mp, on=['date', 'home', 'away'], how='inner')
    if not len(m):
        return None
    res = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    P = m[['pH', 'pD', 'pA']].values
    cp = np.cumsum(P, 1)
    co = np.cumsum(np.eye(3)[res], 1)
    rps = ((cp - co) ** 2).sum(1) / 2
    ll = -np.log(np.clip(P[np.arange(len(P)), res], 1e-9, 1))
    out = dict(n=int(len(m)), rps=float(rps.mean()), logloss=float(ll.mean()),
               hit=float((P.argmax(1) == res).mean()))
    if 'bH' in m.columns and m.bH.notna().any():
        Q = m[['bH', 'bD', 'bA']].values
        ok = ~np.isnan(Q).any(1)
        cq = np.cumsum(Q[ok], 1)
        out['bookie_rps'] = float((((cq - co[ok]) ** 2).sum(1) / 2).mean())
        out['bookie_n'] = int(ok.sum())
    return out


# ------------------------------------------------------------------------ main
def main():
    log('refreshing data')
    us_ok = refresh()

    hist = pd.read_parquet(f'{DATA}/processed/matches.parquet')
    cur = current_season_matches()
    n_played = len(cur)
    log(f'2026/27 matches played: {n_played}  (understat xG: {us_ok})')

    fx = pd.read_csv(f'{DATA}/raw/fixtures_2627.csv')
    fx['home'] = fx['Home Team'].map(lambda x: FIX2US.get(x, x))
    fx['away'] = fx['Away Team'].map(lambda x: FIX2US.get(x, x))
    teams = sorted(set(fx.home))

    # CRITICAL: drop anything that is not a 2026/27 PL fixture before it can
    # reach the training set. The feed has served other divisions.
    if n_played:
        bad = sorted((set(cur.home) | set(cur.away)) - set(teams))
        if bad:
            log(f'  WARNING: dropping {len(bad)} non-PL teams from feed: {bad[:5]}')
            cur = cur[cur.home.isin(teams) & cur.away.isin(teams)]
            n_played = len(cur)
            log(f'  2026/27 PL matches after filter: {n_played}')
    full = pd.concat([hist, cur], ignore_index=True) if n_played else hist
    done = {(r.home, r.away) for _, r in cur.iterrows()} if n_played else set()
    played_mask = fx.apply(lambda r: (r.home, r.away) in done, axis=1)
    remaining = fx[~played_mask]

    force = os.environ.get('FORCE_RUN', '').lower() in ('1', 'true', 'yes')
    lastgw = gate.last_published(f'{OUT}/status.json')
    ok, gw, why = gate.decide(fx, done, last_published_gw=lastgw)
    log(f'gate: {why}')
    if not ok and not force:
        log('nothing to publish - exiting')
        return
    if force and not ok:
        log('FORCE_RUN set - publishing anyway')
        gw = int(fx.loc[played_mask, 'Round Number'].max()) if played_mask.any() else 0
    fixtures = list(zip(remaining.home, remaining.away))
    log(f'gameweek {gw} complete, {len(fixtures)} fixtures remaining')

    table = build_table(cur, teams) if n_played else {t: dict(P=0, GF=0, GA=0, Pts=0) for t in teams}
    start = {t: (v['Pts'], v['GF'], v['GA']) for t, v in table.items()}

    # drift shrinks as we learn what the summer actually did
    gpt = np.mean([v['P'] for v in table.values()]) if n_played else 0
    drift = max(0.04, BASE_DRIFT * (1 - gpt / 12))
    log(f'avg games played {gpt:.1f} -> drift {drift:.3f}')

    prev_pl = set(hist[hist.season == 2025].home)
    pa = {t: 0.0 for t in prev_pl}; pdf = {t: 0.0 for t in prev_pl}

    log(f'fitting {B} bootstrap models')
    ref = dt.date.today().isoformat()
    mods = bootstrap_models(full, ref, teams, pa, pdf, B=B, seed=7, **CFG)
    rng = np.random.default_rng(99)
    if drift > 0:
        for m in mods:
            for t in m['teams']:
                m['att'][t] += rng.normal(0, drift)
                m['dfn'][t] += rng.normal(0, drift)

    if fixtures:
        cum = fixture_grids(mods, fixtures, teams)
        pts, gd, gf = simulate(cum, fixtures, teams, N=N, start=start, seed=11)
    else:
        pts = np.array([[start[t][0] for t in teams]] * N)
        gd = np.array([[start[t][1] - start[t][2] for t in teams]] * N)
        gf = np.array([[start[t][1] for t in teams]] * N)
    pos = positions(pts, gd, gf, seed=12)
    log(f'{N:,} simulations done')

    rows = []
    for i, t in enumerate(teams):
        p = pos[:, i]
        rows.append(dict(team=t, played=table[t]['P'], pts_now=table[t]['Pts'],
                         xPts=pts[:, i].mean(),
                         lo=np.percentile(pts[:, i], 10),
                         hi=np.percentile(pts[:, i], 90),
                         title=(p == 1).mean(), top4=(p <= 4).mean(),
                         top6=(p <= 6).mean(), releg=(p >= 18).mean()))
    pred = pd.DataFrame(rows).sort_values('xPts', ascending=False).reset_index(drop=True)
    pred.insert(0, 'pos', np.arange(1, len(pred) + 1))

    # ---- what changed since last run ----
    changes = []
    prev_f = f'{OUT}/prediction_latest.csv'
    if os.path.exists(prev_f):
        old = pd.read_csv(prev_f)[['team', 'xPts', 'title', 'releg']]
        old.columns = ['team', 'xPts_prev', 'title_prev', 'releg_prev']
        c = pred.merge(old, on='team', how='left')
        c['d_xPts'] = c.xPts - c.xPts_prev
        c['d_title'] = 100 * (c.title - c.title_prev)
        c['d_releg'] = 100 * (c.releg - c.releg_prev)
        changes = c.reindex(c.d_xPts.abs().sort_values(ascending=False).index).head(5)
        changes = changes[['team', 'd_xPts', 'd_title', 'd_releg']].round(2).to_dict('records')

    # ---- next-gameweek match predictions, for future self-scoring ----
    if fixtures:
        nxt = remaining[remaining['Round Number'] == remaining['Round Number'].min()]
        mp = []
        for _, r in nxt.iterrows():
            ps = []
            for m in mods[:40]:
                from ratings import score_matrix
                lh = np.clip(np.exp(m['mu'] + m['gamma'] + m['att'][r.home] - m['dfn'][r.away]), .05, 8)
                la = np.clip(np.exp(m['mu'] + m['att'][r.away] - m['dfn'][r.home]), .05, 8)
                M = score_matrix(lh, la, m['rho'], 10)
                ps.append([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
            p = np.mean(ps, 0)
            mp.append(dict(date=pd.to_datetime(r.Date, dayfirst=True).strftime('%Y-%m-%d'),
                           gw=int(r['Round Number']), home=r.home, away=r.away,
                           pH=p[0], pD=p[1], pA=p[2]))
        mpd = pd.DataFrame(mp)
        f = f'{OUT}/match_predictions.csv'
        if os.path.exists(f):
            mpd = pd.concat([pd.read_csv(f), mpd]).drop_duplicates(
                subset=['date', 'home', 'away'], keep='first')
        mpd.to_csv(f, index=False)

    scorecard = validate(cur, teams)
    log(f'validation: {scorecard}')

    # ---- write everything ----
    pred.to_csv(f'{OUT}/prediction_latest.csv', index=False)
    pred.to_csv(f'{OUT}/history/prediction_gw{gw:02d}.csv'
                if os.path.isdir(f'{OUT}/history') else f'{OUT}/prediction_gw{gw:02d}.csv',
                index=False)
    pm = pd.DataFrame({t: np.bincount(pos[:, i], minlength=22)[1:21] / N
                       for i, t in enumerate(teams)}).T
    pm.columns = range(1, 21)
    pm.loc[pred.team].to_csv(f'{OUT}/position_matrix.csv')

    label = ('PRE-SEASON' if gw == 0 else f'AFTER GAMEWEEK {gw}')
    R.render(pred, f'{OUT}/table_wide.png',
             f'{label}  ·  {dt.date.today():%d %b %Y}', n_sims=N)
    R.render_mobile(pred, f'{OUT}/table.png', label, n_sims=N)

    status = dict(updated=dt.datetime.now().isoformat(timespec='seconds'),
                  gameweek=gw, matches_played=n_played, drift=round(drift, 3),
                  n_sims=N, gate=why,
                  xg_source='understat' if us_ok else 'goals-only fallback',
                  validation=scorecard, biggest_moves=changes)
    json.dump(status, open(f'{OUT}/status.json', 'w'), indent=2)

    body = mailer.build_body(pred, status, label)
    open(f'{OUT}/email_body.txt', 'w').write(body)
    try:
        mailer.send(f'PL Supercomputer — {label}', body,
                    f'{OUT}/table.png', [f'{OUT}/prediction_latest.csv'])
    except Exception as e:
        log(f'email failed: {e}')
    log('done -> outputs/table.png')


if __name__ == '__main__':
    main()
