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

from ratings import fit_ratings, score_matrix
from simulate import bootstrap_models, fixture_grids, simulate, positions
import render as R
import gate, mailer
import pickle

SEASON = 2026                     # Understat label for 2026/27
FD_CODE = '2627'
CFG = dict(xi=0.0045, w_xg=0.7, ridge=2.0)
BASE_DRIFT, B, N = 0.16, 80, 20000
SQUAD_W = 0.5          # backtested on 7 seasons; see VALIDATION.md
CLOSE_CALL = 0.04      # two leading outcomes this close => call it level


def call_outcome(pH, pD, pA):
    """Index of the outcome to publish: 0 home, 1 draw, 2 away.

    Plain argmax over the three outcomes can essentially never return a draw,
    because in a Dixon-Coles grid the draw is almost never the single largest
    of the three. That capped the hit rate near 76% and, worse, made the model
    "call" 36/28/36 fixtures for the home side purely on tie-break order.

    So when the two leading outcomes are within CLOSE_CALL of each other, the
    model cannot separate them and we publish a draw instead of pretending.
    At 4 percentage points this produces draws for about 20% of fixtures,
    against a real Premier League draw rate near 24%.

    Every consumer of a prediction must use this one function, or the tick on
    the fixtures page will disagree with the hit rate in the accuracy panel.
    """
    p = [float(pH), float(pD), float(pA)]
    order = sorted(range(3), key=lambda i: -p[i])
    return 1 if p[order[0]] - p[order[1]] < CLOSE_CALL else order[0]
MGR_W = 0.25           # half the tested optimum; within-PL moves only
UA = {'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'}

DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(f'{DATA}/raw', exist_ok=True)
os.makedirs(f'{DATA}/understat', exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}/history', exist_ok=True)

FIX2US = {'Man Utd': 'Manchester United', 'Man City': 'Manchester City',
          'Newcastle': 'Newcastle United', 'Spurs': 'Tottenham',
          "Nott'm Forest": 'Nottingham Forest'}
FD2US = dict(FIX2US, **{'Man United': 'Manchester United',
                        'Tottenham': 'Tottenham', 'Wolves': 'Wolverhampton Wanderers'})


def explain(team, played, gw, mods, pred):
    """A plain-English reason for a club's movement, from what actually happened."""
    if not len(played):
        return ''
    d = played.copy()
    d = d[(d.home == team) | (d.away == team)].sort_values('date')
    if not len(d):
        return 'did not play'
    m = d.iloc[-1]
    home = m.home == team
    opp = m.away if home else m.home
    gf, ga = (m.hg, m.ag) if home else (m.ag, m.hg)
    xgf, xga = (m.hnpxg, m.anpxg) if home else (m.anpxg, m.hnpxg)
    where = 'at home to' if home else 'away at'
    won, lost = gf > ga, gf < ga
    if won:
        head = f'beat {opp} {int(gf)}-{int(ga)} {"at home" if home else "away"}'
    elif lost:
        head = f'lost {int(gf)}-{int(ga)} {where} {opp}'
    else:
        head = f'drew {int(gf)}-{int(ga)} {where} {opp}'

    # did the performance match the scoreline? phrasing has to fit the result
    edge = (xgf - xga) - (gf - ga)
    if edge > 1.0:
        tail = (', and deserved more' if won else
                ', but played far better than the score')
    elif edge < -1.0:
        tail = (', and rode their luck' if won or not lost else
                ', though it could have been worse')
    else:
        tail = ''
    return head + tail


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
                # This host answers a missing file with HTTP 300 and an HTML
                # "Multiple Choices" page rather than a 404, so status alone is
                # not enough - check it actually looks like the CSV we expect.
                body = r.content[:400].lstrip()
                looks_html = body[:1] == b'<' or b'<html' in body.lower()
                header_ok = body.lstrip(b'\xef\xbb\xbf').startswith(b'Div,')
                if r.status_code == 200 and not looks_html and header_ok:
                    open(f'{DATA}/raw/{div}_{code}.csv', 'wb').write(r.content)
                else:
                    log(f'  {div}_{code}: not a valid CSV yet '
                        f'(HTTP {r.status_code}) - skipping')
            except Exception as e:
                log(f'  {div}_{code} fetch failed: {e}')
    ok = False
    cache = f'{DATA}/understat/EPL_{SEASON}.json'
    try:
        r = requests.get(f'https://understat.com/getLeagueData/EPL/{SEASON}',
                         headers=UA, timeout=40)
        if r.ok:
            j = r.json()
            # Results only ever accumulate within a season, so a payload with
            # fewer results than we already hold is a fault, not an update.
            # Writing it would wipe the table - and refresh mode, which skips
            # the gate, would then publish a season reset to zero.
            n_new = len([m for m in j.get('dates', []) if m.get('isResult')])
            n_old = 0
            if os.path.exists(cache):
                try:
                    n_old = len([m for m in json.load(open(cache)).get('dates', [])
                                 if m.get('isResult')])
                except Exception:
                    n_old = 0
            if n_new >= n_old:
                with open(cache, 'w') as fh:
                    json.dump(j, fh)
                ok = True
            else:
                log(f'  understat returned {n_new} results but the cache holds '
                    f'{n_old} - keeping the cache')
                ok = os.path.exists(cache)
    except Exception as e:
        log(f'  understat fetch failed: {type(e).__name__}: {e}')
        ok = os.path.exists(cache)
    return ok


def current_season_matches():
    """Played 2026/27 matches with xG where available, goals otherwise.

    xG comes from understat; any match understat has not posted yet is filled
    from the football-data results feed on a per-match basis, so a round is
    never partially represented.
    """
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
    # Understat lags the results feed, and it does not lag uniformly - it can
    # hold 9 of a 10-match round. The old code only consulted football-data
    # when understat had NOTHING, so a single missing fixture was dropped
    # outright: absent from the training set and from the standings table.
    # Publish runs were shielded by the gate (which reads `done` from this
    # function), but REFRESH_ONLY skips the gate and would rebuild the site
    # from a partial round. So fill per match instead of all-or-nothing.
    # Goals stand in for npxG, which is what ratings.py actually consumes;
    # hxg/axg stay NaN and are never read by the fit.
    have = {(r['home'], r['away']) for r in rows}
    f = f'{DATA}/raw/E0_{FD_CODE}.csv'
    if os.path.exists(f):
        try:
            o = pd.read_csv(f, encoding='latin-1')
        except Exception as e:
            log(f'  {f} is not parseable ({e}) - treating as no matches yet')
            o = pd.DataFrame()
        # football-data ships this CSV with a UTF-8 BOM. Reading it as latin-1
        # never raises, but it decodes the BOM into mojibake, so the first
        # column arrives as 'i>>?Div' rather than 'Div' and the column check
        # below rejected the whole file. Normalise before validating.
        o.columns = [str(c).replace('\ufeff', '').replace('ï»¿', '').strip()
                     for c in o.columns]
        # Reject anything that is not the CSV we expect. A stale or
        # HTML-ish file can still parse into a junk DataFrame.
        need = {'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'}
        if not need.issubset(set(o.columns)):
            log(f'  {f} lacks the expected columns - ignoring it')
            o = pd.DataFrame(columns=sorted(need))
        o = o[o['Div'] == 'E0'].dropna(subset=['FTHG'])
        filled = 0
        for _, r in o.iterrows():
            h = FD2US.get(r.HomeTeam, r.HomeTeam); a = FD2US.get(r.AwayTeam, r.AwayTeam)
            if (h, a) in have:
                continue
            rows.append(dict(season=SEASON,
                             date=pd.to_datetime(r.Date, dayfirst=True).strftime('%Y-%m-%d'),
                             home=h, away=a, hg=int(r.FTHG), ag=int(r.FTAG),
                             hxg=np.nan, axg=np.nan,
                             hnpxg=float(r.FTHG), anpxg=float(r.FTAG)))
            have.add((h, a))
            filled += 1
            log(f'  {h} v {a}: no understat xG yet, using goals')
        if filled:
            log(f'  filled {filled} match(es) from football-data')
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
    # Match on the fixture, not the date: a postponed game is replayed on a
    # different day, and merging on date would silently drop it from scoring -
    # losing exactly the matches most likely to have surprised the model.
    mp = mp.sort_values('date').drop_duplicates(['home', 'away'], keep='first')
    m = played.merge(mp.drop(columns=['date']), on=['home', 'away'], how='inner')
    if not len(m):
        return None
    res = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    P = m[['pH', 'pD', 'pA']].values
    cp = np.cumsum(P, 1)
    co = np.cumsum(np.eye(3)[res], 1)
    rps = ((cp - co) ** 2).sum(1) / 2
    ll = -np.log(np.clip(P[np.arange(len(P)), res], 1e-9, 1))
    # Same close-call rule as the fixtures page, or the tick beside a match
    # would contradict the hit rate reported here.
    srt = np.sort(P, 1)
    call = np.where(srt[:, 2] - srt[:, 1] < CLOSE_CALL, 1, P.argmax(1))
    out = dict(n=int(len(m)), rps=float(rps.mean()), logloss=float(ll.mean()),
               hit=float((call == res).mean()))
    # Exact-scoreline accuracy, once predictions carry a scoreline. Expect
    # roughly one in nine: the most likely single result in a football match
    # is usually 1-1 or 1-0 and rarely carries more than ~12% probability.
    if 'sc_h' in m.columns and m.sc_h.notna().any():
        s = m[m.sc_h.notna() & m.sc_a.notna()]
        if len(s):
            exact = (s.sc_h.astype(int) == s.hg) & (s.sc_a.astype(int) == s.ag)
            out['hit_score'] = float(exact.mean())
            out['n_score'] = int(len(s))
    # argmax over three outcomes can essentially never return a draw in a
    # Dixon-Coles model, so `hit` is structurally capped near 76%. Publish the
    # count so the ceiling is visible rather than hidden.
    out['drawn'] = int((res == 1).sum())
    out['picked_draw'] = int((call == 1).sum())
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
    # Team news lands Thursday/Friday, but a gameweek publish happens Tuesday.
    # Refresh mode re-reads squads and rebuilds the site without emailing.
    refresh_only = os.environ.get('REFRESH_ONLY', '').lower() in ('1', 'true', 'yes')
    lastgw = gate.last_published(f'{OUT}/status.json')
    ok, gw, why = gate.decide(fx, done, last_published_gw=lastgw)
    log(f'gate: {why}')
    if refresh_only:
        gw = max(lastgw, 0)
        log(f'REFRESH_ONLY - rebuilding site for GW{gw} with current squad data')
    elif not ok and not force:
        log('nothing to publish - exiting')
        return
    if force and not ok:
        log('FORCE_RUN set - publishing anyway')
        gw = int(fx.loc[played_mask, 'Round Number'].max()) if played_mask.any() else 0
    fixtures = list(zip(remaining.home, remaining.away))
    log(f'gameweek {gw} complete, {len(fixtures)} fixtures remaining')

    # one code path: build_table already returns zeros for an unplayed season
    table = build_table(cur if n_played else cur.iloc[0:0], teams)
    start = {t: (v['Pts'], v['GF'], v['GA']) for t, v in table.items()}

    # drift shrinks as we learn what the summer actually did
    gpt = np.mean([v['P'] for v in table.values()]) if n_played else 0
    drift = max(0.04, BASE_DRIFT * (1 - gpt / 12))
    log(f'avg games played {gpt:.1f} -> drift {drift:.3f}')

    # ---- squad layer: RAPM player values x current FPL rosters ----
    squad_delta, squad_note = {}, 'off'
    try:
        import squad_live
        rl = pickle.load(open(f'{DATA}/processed/rapm_live.pkl', 'rb'))
        base = float(hist[hist.season == 2025].hnpxg.mean())
        att_x, dfn_x, known_x = dict(rl['att']), dict(rl['dfn']), set(rl['known'])
        eu_idx, n_eu = None, 0
        try:    # European values for players with no Premier League record
            euv = pickle.load(open(f'{DATA}/processed/eu_values.pkl', 'rb'))
            eu_idx = squad_live.european_index()
            for pid_, v in euv.items():
                if pid_ not in known_x:
                    att_x[pid_] = v / 2; dfn_x[pid_] = v / 2; known_x.add(pid_)
                    n_eu += 1
        except Exception as e:
            log(f'  european values unavailable: {type(e).__name__}: {e}')
        sd, _, _, sq = squad_live.build(att_x, dfn_x, known_x, base,
                                        remaining=max(38 - int(gpt), 1),
                                        eu_index=eu_idx)
        squad_delta = dict(sd)   # club names already normalised in squad_live
        unmatched = int(((sq.how == 'none') & (sq.last_mins.notna())).sum())
        squad_note = (f'on (weight {SQUAD_W}, {len(squad_delta)} clubs, '
                      f'{unmatched} unmatched with PL minutes, '
                      f'{n_eu} players valued from European leagues)')
        if unmatched > 10:
            log(f'  WARNING: {unmatched} players with PL minutes did not match')
    except Exception as e:
        log(f'  squad layer unavailable: {type(e).__name__}: {e}')
    log(f'squad layer: {squad_note}')

    # ---- manager layer: only scored when BOTH managers have a PL record ----
    mgr_delta, mgr_note, mgr_changes = {}, 'off', []
    try:
        import managers as mgr_mod
        mf = pickle.load(open(f'{DATA}/processed/manager_fit.pkl', 'rb'))
        net = {m: mf['mgr_att'][m] + mf['mgr_dfn'][m] for m in mf['managers']}
        try:
            spells = mgr_mod.scrape()          # refresh: catches in-season sackings
            spells.to_csv(f'{DATA}/processed/managers.csv', index=False)
            src = 'live'
        except Exception:
            spells = pd.read_csv(f'{DATA}/processed/managers.csv',
                                 parse_dates=['start', 'end'])
            src = 'cached'

        def boss(club, when):
            g = spells[(spells.club == club) & (spells.start <= when)
                       & (spells.end >= when)]
            if len(g):
                return g.sort_values('start').iloc[-1].manager
            g = spells[(spells.club == club) & (spells.start <= when)]
            return g.sort_values('start').iloc[-1].manager if len(g) else None

        now = pd.Timestamp(dt.date.today())
        last_end = hist[hist.season == 2025].date.max()
        base_x = float(hist[hist.season == 2025].hnpxg.mean())
        for t in teams:
            new, old = boss(t, now), boss(t, last_end)
            if new and old and new != old and new in net and old in net:
                mgr_delta[t] = (net[new] - net[old]) / base_x
                mgr_changes.append(f'{t}: {old} -> {new} ({mgr_delta[t]:+.2f})')
        mgr_note = f'on (weight {MGR_W}, spells {src}, {len(mgr_delta)} scored)'
    except Exception as e:
        log(f'  manager layer unavailable: {type(e).__name__}: {e}')
    log(f'manager layer: {mgr_note}')
    for c in mgr_changes:
        log(f'    {c}')

    prev_pl = set(hist[hist.season == 2025].home)
    pa = {t: 0.0 for t in prev_pl}; pdf = {t: 0.0 for t in prev_pl}

    log(f'fitting {B} bootstrap models')
    ref = dt.date.today().isoformat()
    mods = bootstrap_models(full, ref, teams, pa, pdf, B=B, seed=7, **CFG)
    rng = np.random.default_rng(99)
    for m in mods:
        for t in m['teams']:
            shift = (squad_delta.get(t, 0.0) * SQUAD_W
                     + mgr_delta.get(t, 0.0) * MGR_W)
            m['att'][t] += shift / 2 + rng.normal(0, drift)
            m['dfn'][t] += shift / 2 + rng.normal(0, drift)

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

    # ---- what changed since the last PUBLISHED gameweek ----
    # Comparing against the last run would report Monte Carlo noise as news:
    # rerunning the identical model with a different seed moves expected points
    # by up to 0.3. So compare against the previous gameweek's saved file, and
    # only report moves that clear the noise floor.
    NOISE = 0.4
    changes = []
    prev_f = f'{OUT}/history/prediction_gw{gw - 1:02d}.csv'
    if gw > 0 and os.path.exists(prev_f):
        old = pd.read_csv(prev_f)[['team', 'xPts', 'title', 'releg']]
        old.columns = ['team', 'xPts_prev', 'title_prev', 'releg_prev']
        c = pred.merge(old, on='team', how='left')
        c['d_xPts'] = c.xPts - c.xPts_prev
        c['d_title'] = 100 * (c.title - c.title_prev)
        c['d_releg'] = 100 * (c.releg - c.releg_prev)
        c = c[c.d_xPts.abs() >= NOISE]
        c = c.reindex(c.d_xPts.abs().sort_values(ascending=False).index).head(5)
        for _, r in c.iterrows():
            changes.append(dict(team=r.team, d_xPts=round(r.d_xPts, 2),
                                d_title=round(r.d_title, 2),
                                d_releg=round(r.d_releg, 2),
                                reason=explain(r.team, cur, gw, mods, pred)))

    # ---- match predictions for the next TWO gameweeks, for self-scoring ----
    # Two rounds rather than one, so a missed scheduler run can never orphan a
    # gameweek: the round after next was already predicted and stored, and can
    # still be scored honestly.
    #
    # A prediction FREEZES AT KICKOFF and is never revised afterwards. Before
    # kickoff it is rewritten on every run, so late team news is reflected.
    # This is the single rule that keeps the public scorecard meaningful: what
    # gets marked is what stood when the ball was kicked, not a tidied-up
    # version written once the result was known. The `ko > now` test below is
    # what enforces it, and it also stops a run that fires mid-match from
    # inventing a prediction for a game already in progress.
    if fixtures:
        f = f'{OUT}/match_predictions.csv'
        prev = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
        rounds = sorted(remaining['Round Number'].unique())[:2]
        nxt = remaining[remaining['Round Number'].isin(rounds)]
        now_utc = pd.Timestamp.now(tz='UTC')
        log(f'predicting gameweeks {rounds} ({len(nxt)} fixtures)')

        mp, frozen = [], 0
        for _, r in nxt.iterrows():
            ko = pd.to_datetime(r.Date, dayfirst=True).tz_localize('UTC')
            if ko <= now_utc:
                frozen += 1
                continue
            Ms, lhs, las = [], [], []
            for m in mods[:40]:
                lh = np.clip(np.exp(m['mu'] + m['gamma'] + m['att'][r.home] - m['dfn'][r.away]), .05, 8)
                la = np.clip(np.exp(m['mu'] + m['att'][r.away] - m['dfn'][r.home]), .05, 8)
                Ms.append(score_matrix(lh, la, m['rho'], 10))
                lhs.append(lh); las.append(la)
            M = np.mean(Ms, 0)
            pH = float(np.tril(M, -1).sum())
            pD = float(np.trace(M))
            pA = float(np.triu(M, 1).sum())

            # Modal scoreline CONDITIONAL on the modal outcome. Taking the
            # single most likely cell of the whole grid would routinely give
            # 1-1 even when a home win is the most likely result, so the page
            # would read "home win, most likely 1-1" - incoherent, and the
            # first thing anyone would screenshot.
            oc = call_outcome(pH, pD, pA)
            keep = np.zeros_like(M, dtype=bool)
            if oc == 0:
                keep[np.tril_indices_from(M, -1)] = True
            elif oc == 1:
                np.fill_diagonal(keep, True)
            else:
                keep[np.triu_indices_from(M, 1)] = True
            sh, sa = np.unravel_index(np.argmax(np.where(keep, M, -1.0)), M.shape)

            mp.append(dict(date=ko.strftime('%Y-%m-%d'), ko=ko.isoformat(),
                           gw=int(r['Round Number']), home=r.home, away=r.away,
                           pH=pH, pD=pD, pA=pA, outcome='HDA'[oc],
                           sc_h=int(sh), sc_a=int(sa),
                           p_score=float(M[sh, sa]),
                           xg_h=float(np.mean(lhs)), xg_a=float(np.mean(las))))

        # New rows first, so drop_duplicates(keep='first') lets a refreshed
        # pre-kickoff prediction win while anything already frozen survives.
        mpd = pd.concat([pd.DataFrame(mp), prev], ignore_index=True) \
            if len(mp) else prev
        if len(mpd):
            mpd = (mpd.drop_duplicates(subset=['home', 'away'], keep='first')
                      .sort_values(['gw', 'date', 'home']))
            mpd.to_csv(f, index=False)
        log(f'  {len(mp)} predictions written, {frozen} already frozen at kickoff')

    scorecard = validate(cur, teams)

    # ---- fixture-by-fixture payload for the site ----
    # Every stored prediction, joined to the actual result where one exists, so
    # the page can show the call and the outcome side by side instead of just
    # asserting an accuracy figure. `outcome` is derived here rather than read
    # from the file, so rows written before scorelines existed still work.
    match_rows = []
    fmp = f'{OUT}/match_predictions.csv'
    if os.path.exists(fmp):
        allp = pd.read_csv(fmp)
        act = (cur[['home', 'away', 'hg', 'ag']] if n_played
               else pd.DataFrame(columns=['home', 'away', 'hg', 'ag']))
        j = allp.merge(act, on=['home', 'away'], how='left')
        # Predictions stored before the `ko` column existed have no kickoff
        # time, so their cards would render without one. The kickoff is public
        # fact from the fixture list, not part of the forecast, so filling it in
        # afterwards is not hindsight - unlike the prediction itself, which is
        # never touched.
        kos = {(r.home, r.away): pd.to_datetime(r.Date, dayfirst=True)
                                   .tz_localize('UTC').isoformat()
               for _, r in fx.iterrows()}
        for _, r in j.sort_values(['gw', 'date', 'home']).iterrows():
            probs = [float(r.pH), float(r.pD), float(r.pA)]
            oc = 'HDA'[call_outcome(*probs)]
            row = dict(gw=int(r.gw), home=r.home, away=r.away,
                       ko=(r.ko if isinstance(r.get('ko'), str)
                           else kos.get((r.home, r.away))),
                       pH=round(probs[0], 4), pD=round(probs[1], 4),
                       pA=round(probs[2], 4), outcome=oc,
                       conf=round(max(probs), 4))
            if pd.notna(r.get('sc_h')) and pd.notna(r.get('sc_a')):
                row.update(sc_h=int(r.sc_h), sc_a=int(r.sc_a),
                           p_score=round(float(r.p_score), 4))
            # The modal scoreline is only the tallest bar in a very flat
            # distribution - it carries ~14% at best. Expected goals say what
            # the model actually thinks: 2.3-0.6 conveys "possibly a rout",
            # which "2-0" cannot. Already computed and stored in the CSV; this
            # just carries it through to the page.
            if pd.notna(r.get('xg_h')) and pd.notna(r.get('xg_a')):
                row.update(xg_h=round(float(r.xg_h), 2),
                           xg_a=round(float(r.xg_a), 2))
            if pd.notna(r.get('hg')):
                hg, ag = int(r.hg), int(r.ag)
                row.update(res_h=hg, res_a=ag,
                           actual='H' if hg > ag else ('D' if hg == ag else 'A'))
                row['ok_outcome'] = bool(row['actual'] == oc)
                if 'sc_h' in row:
                    row['ok_score'] = bool(row['sc_h'] == hg and row['sc_a'] == ag)
            match_rows.append(row)
        log(f'  fixture payload: {len(match_rows)} rows, '
            f'{sum(1 for r in match_rows if "res_h" in r)} settled')
    log(f'validation: {scorecard}')

    # ---- write everything ----
    pred.to_csv(f'{OUT}/prediction_latest.csv', index=False)
    if not refresh_only:      # history is the record of each gameweek, not each refresh
        pred.to_csv(f'{OUT}/history/prediction_gw{gw:02d}.csv'
                    if os.path.isdir(f'{OUT}/history')
                    else f'{OUT}/prediction_gw{gw:02d}.csv', index=False)
    pm = pd.DataFrame({t: np.bincount(pos[:, i], minlength=22)[1:21] / N
                       for i, t in enumerate(teams)}).T
    pm.columns = range(1, 21)
    pm.loc[pred.team].to_csv(f'{OUT}/position_matrix.csv')

    label = ('PRE-SEASON' if gw == 0 else f'AFTER GAMEWEEK {gw}')
    if refresh_only:
        label += '  ·  TEAM NEWS UPDATE'
    R.render(pred, f'{OUT}/table_wide.png',
             f'{label}  ·  {dt.date.today():%d %b %Y}', n_sims=N)
    R.render_mobile(pred, f'{OUT}/table.png', label, n_sims=N)
    story = None
    try:                      # 9:16 version for Instagram / WhatsApp Status
        import render_story
        story = render_story.render_story(pred, f'{OUT}/story.png', label, n_sims=N)
    except Exception as e:
        log(f'  story image skipped: {type(e).__name__}: {e}')

    status = dict(updated=dt.datetime.now().isoformat(timespec='seconds'),
                  gameweek=gw, matches_played=n_played, drift=round(drift, 3),
                  n_sims=N, gate=why,
                  xg_source='understat' if us_ok else 'goals-only fallback',
                  squad_layer=squad_note, manager_layer=mgr_note,
                  manager_changes=mgr_changes,
                  squad_delta={k: round(v, 3) for k, v in
                               sorted(squad_delta.items(), key=lambda kv: kv[1])},
                  validation=scorecard, biggest_moves=changes)
    with open(f'{OUT}/status.json', 'w') as fh:
        json.dump(status, fh, indent=2)

    # ---- next three fixtures per club, with our own win probability ----
    nextfx = {t: [] for t in teams}
    try:
        from ratings import score_matrix as _sm
        sub = mods[:24]
        for _, r in remaining.sort_values(
                pd.to_datetime(remaining['Date'], dayfirst=True).name
                if False else 'Round Number').iterrows():
            for side, team, opp in [('H', r.home, r.away), ('A', r.away, r.home)]:
                if len(nextfx[team]) >= 3:
                    continue
                ps = []
                for mm in sub:
                    lh = np.clip(np.exp(mm['mu'] + mm['gamma']
                                        + mm['att'][r.home] - mm['dfn'][r.away]), .05, 8)
                    la = np.clip(np.exp(mm['mu'] + mm['att'][r.away]
                                        - mm['dfn'][r.home]), .05, 8)
                    M = _sm(lh, la, mm['rho'], 10)
                    hw, dr = np.tril(M, -1).sum(), np.trace(M)
                    ps.append(hw if side == 'H' else 1 - hw - dr)
                nextfx[team].append(dict(opp=opp, side=side,
                                         win=round(float(np.mean(ps)), 3)))
    except Exception as e:
        log(f'  next fixtures failed: {type(e).__name__}: {e}')

    # ---- official FPL fixture difficulty, for the Actual tab ----
    fdr = {t: [] for t in teams}
    try:
        import squad_live as _sl
        b = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/',
                         headers={'User-Agent': 'Mozilla/5.0'}, timeout=40).json()
        fl = requests.get('https://fantasy.premierleague.com/api/fixtures/',
                          headers={'User-Agent': 'Mozilla/5.0'}, timeout=40).json()
        fid = {t['id']: _sl.FPL2US.get(t['name'], t['name']) for t in b['teams']}
        for f in sorted([x for x in fl if not x['finished'] and x['event']],
                        key=lambda x: (x['event'], x['id'])):
            h_, a_ = fid.get(f['team_h']), fid.get(f['team_a'])
            for me, opp, side, dif in [(h_, a_, 'H', f['team_h_difficulty']),
                                       (a_, h_, 'A', f['team_a_difficulty'])]:
                if me in fdr and opp and len(fdr[me]) < 3:
                    fdr[me].append(dict(opp=opp, side=side, fdr=int(dif)))
        log(f'  FPL difficulty ratings loaded for {sum(1 for v in fdr.values() if v)} clubs')
    except Exception as e:
        log(f'  FPL fixture difficulty unavailable: {type(e).__name__}: {e}')

    # ---- data for the public web page ----
    try:
        web = os.path.join(ROOT, 'docs')
        os.makedirs(web, exist_ok=True)
        nxt = remaining[remaining['Round Number'] == remaining['Round Number'].min()] \
            if len(remaining) else None
        next_gw, next_settle = None, None
        if nxt is not None and len(nxt):
            ko = pd.to_datetime(nxt['Date'], dayfirst=True, utc=True)
            next_gw = int(nxt['Round Number'].iloc[0])
            # same window the gate uses, imported so the two cannot drift
            next_settle = (ko.max() + pd.Timedelta(
                hours=gate.MATCH_HOURS + gate.SETTLE_HOURS)).isoformat()
        pm_web = pd.DataFrame(
            {t: np.bincount(pos[:, i], minlength=22)[1:21] / N
             for i, t in enumerate(teams)}).T.loc[pred.team]
        payload = dict(
            updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(
                timespec='seconds'),
            gameweek=gw, label=label, n_sims=N, refresh_only=refresh_only,
            matches_played=n_played,
            next_gameweek=next_gw, next_update_utc=next_settle,
            validation=scorecard, moves=changes,
            actual=[dict(pos=i + 1, team=t,
                         P=v['P'], W=v['W'], D=v['D'], L=v['L'],
                         GF=int(v['GF']), GA=int(v['GA']),
                         GD=int(v['GF'] - v['GA']), Pts=v['Pts'])
                    for i, (t, v) in enumerate(sorted(
                        table.items(),
                        key=lambda kv: (-kv[1]['Pts'],
                                        -(kv[1]['GF'] - kv[1]['GA']),
                                        -kv[1]['GF'], kv[0])))],
            teams=[dict(pos=int(r['pos']), team=r.team, xPts=round(r.xPts, 1),
                        pts_now=int(r.pts_now), played=int(r.played),
                        lo=int(r.lo), hi=int(r.hi),
                        title=round(r.title, 4), top4=round(r.top4, 4),
                        top6=round(r.top6, 4), releg=round(r.releg, 4),
                        next3=nextfx.get(r.team, []),
                        next3_fdr=fdr.get(r.team, []))
                   for _, r in pred.iterrows()],
            matches=match_rows,
            position_matrix={t: [round(v, 4) for v in pm_web.loc[t].tolist()]
                             for t in pm_web.index},
        )
        # write to a temp file then move it into place, so a crash mid-write
        # cannot leave the site serving truncated JSON
        tmp = f'{web}/data.json.tmp'
        with open(tmp, 'w') as fh:
            json.dump(payload, fh)
        os.replace(tmp, f'{web}/data.json')
        log(f'web data written -> docs/data.json')
    except Exception as e:
        log(f'  web data failed: {type(e).__name__}: {e}')

    body = mailer.build_body(pred, status, label)
    html = mailer.build_html(pred, status, label)
    with open(f'{OUT}/email_body.txt', 'w') as fh:
        fh.write(body)
    with open(f'{OUT}/email_body.html', 'w') as fh:
        fh.write(html)
    lead = pred.iloc[0]
    subj = (f'{label.title()} — {lead.team} favourites '
            f'({100 * lead.title:.0f}%)')
    if refresh_only:
        log('refresh only - no email sent')
        return
    try:
        extras = [f'{OUT}/prediction_latest.csv']
        if story:
            extras.insert(0, story)          # attached alongside the main table
        mailer.send(subj, body, f'{OUT}/table.png', extras, html=html)
    except Exception as e:
        log(f'email failed: {type(e).__name__}: {e}')
    log('done -> outputs/table.png')


if __name__ == '__main__':
    main()
