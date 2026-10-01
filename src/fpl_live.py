"""Live FPL expected points for the next five gameweeks.

Called by update.py after the match predictions, with the same bootstrap
models, so FPL numbers and fixture predictions always agree. Uses:

  FPL API          players, prices, injury flags, ownership, FPL's own
                   pre-deadline expected points (ep_next), fixtures, and each
                   finished gameweek's minutes / saves / defensive points
  Understat        this season's lineups - each player's share of chances
  committed files  fpl_calibration.json (scale factors and bonus model from
                   the backtest), fpl_prev_season.parquet (last season's final
                   appearances, so the minutes model works from GW1)

Points model as backtested in VALIDATION.md section 18, plus what the backtest
could not have: injury and suspension flags, and 2025/26 defensive-
contribution points (2 for 10+ CBIT as a defender, 12+ CBIRT otherwise).
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json, os, re
import datetime as dt
import numpy as np
import pandas as pd
import requests
import fpl as F
import fpl_model as FM
import players as PL

API = 'https://fantasy.premierleague.com/api'
UA = {'User-Agent': 'Mozilla/5.0'}
CAL = f'{ROOT}/data/processed/fpl_calibration.json'
PREV = f'{ROOT}/data/processed/fpl_prev_season.parquet'
ROWS = f'{ROOT}/data/raw/fpl_season_rows.parquet'      # finished GWs, cached
OUT = f'{ROOT}/outputs'
SEASON = 2026
HORIZON = 5
POS = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
DC_THRESHOLD = {'DEF': 10, 'MID': 12, 'FWD': 12}
DC_PRIOR_K = 3.0
# fixtures an injured / suspended player is assumed to miss when FPL's news
# gives no return date (as squad_live.MISSED)
MISSED = {'i': 6, 's': 2, 'd': 0}


def get(path):
    r = requests.get(f'{API}/{path}', headers=UA, timeout=40)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- offline
def build_prev_season(label='2025-26'):
    """Last season's final appearances per FPL code, and defensive-
    contribution hit rates, from the vaastav archive. Run once a season."""
    h = F.load_history(label).sort_values('kickoff')
    raw = pd.read_csv(f'{F.HIST}/players_raw_{label}.csv', encoding='utf-8', low_memory=False)
    code = dict(zip(raw.id, raw.code))
    h['code'] = h.element.map(code)
    h['dc_hit'] = [dc >= DC_THRESHOLD.get(p, 99) for dc, p in zip(h.dc, h.pos)]
    tail = h.groupby('code').tail(FM.N_RECENT)[['code', 'kickoff', 'minutes', 'saves', 'yellow']]
    a60 = h[h.minutes >= 60]
    dc = a60.groupby('code').agg(dc_hits=('dc_hit', 'sum'), dc_apps=('dc_hit', 'size'), pos=('pos', 'last'))
    out = tail.merge(dc, left_on='code', right_index=True, how='left')
    out.to_parquet(PREV, index=False)
    return out


def save_calibration(cal):
    with open(CAL, 'w') as fh:
        json.dump({k: ({p: list(map(float, v)) for p, v in cal[k].items()} if k == 'bonus' else float(cal[k]))
                   for k in cal}, fh, indent=1)


def load_calibration():
    c = json.load(open(CAL))
    c['bonus'] = {p: np.array(v) for p, v in c['bonus'].items()}
    return c


# ------------------------------------------------------------------- live
def players_table(boot):
    teams = {t['id']: F.FPL2US.get(t['name'], t['name']) for t in boot['teams']}
    return pd.DataFrame([dict(
        element=e['id'], code=e['code'], web=e['web_name'],
        name=f"{e['first_name']} {e['second_name']}", club=teams[e['team']],
        pos=POS[e['element_type']], price=e['now_cost'] / 10,
        own=float(e['selected_by_percent'] or 0), status=e['status'],
        chance=e['chance_of_playing_next_round'], news=e['news'] or '',
        ep_next=float(e['ep_next'] or 0), pen=e['penalties_order'])
        for e in boot['elements']])


def season_rows(boot, log=print):
    """This season's per-gameweek minutes, saves, yellows and defensive hits
    for every player, from FPL's event/live endpoint. Finished gameweeks are
    cached, so each is fetched once."""
    have = pd.read_parquet(ROWS) if os.path.exists(ROWS) else pd.DataFrame()
    done = set(have.gw) if len(have) else set()
    new = []
    for ev in boot['events']:
        if not ev['finished'] or ev['id'] in done:
            continue
        lv = get(f"event/{ev['id']}/live/")
        for x in lv['elements']:
            if not x['explain']:
                continue                       # blank gameweek for his club
            st = x['stats']
            dc = sum(s['points'] for f in x['explain'] for s in f['stats']
                     if s['identifier'] == 'defensive_contribution')
            new.append(dict(element=x['id'], gw=ev['id'], minutes=st['minutes'],
                            saves=st['saves'], yellow=st['yellow_cards'], dc_hit=dc > 0,
                            points=st['total_points']))
    if new:
        log(f'  FPL: cached {len({r["gw"] for r in new})} finished gameweek(s)')
        have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
        os.makedirs(os.path.dirname(ROWS), exist_ok=True)
        have.to_parquet(ROWS, index=False)
    return have


def return_date(news, today):
    """'Hamstring injury - Expected back 18 Oct' -> that date, else None."""
    m = re.search(r'[Ee]xpected back (\d{1,2}) ([A-Z][a-z]{2})', news or '')
    if not m:
        return None
    try:
        d = dt.datetime.strptime(f'{m.group(1)} {m.group(2)} {today.year}', '%d %b %Y').date()
        return d if d >= today - dt.timedelta(days=60) else d.replace(year=today.year + 1)
    except ValueError:
        return None


def availability(p, k, kickoff, today):
    """Chance he is available for his k-th upcoming fixture (0 = next)."""
    if p.status in ('u', 'n'):
        return 0.0
    if p.status == 'a':
        return 1.0
    back = return_date(p.news, today)
    if back is not None:
        return 1.0 if kickoff.date() >= back else 0.0
    if k == 0 and p.chance is not None and not pd.isna(p.chance):
        return float(p.chance) / 100
    return 0.0 if k < MISSED.get(p.status, 0) else 1.0


def match_understat(pl, ro):
    """FPL element -> Understat id: this season's lineups first (players.py
    rules, via fpl.match_players), then last season's by name for anyone who
    has not played yet."""
    df = pl.assign(season=SEASON)
    web = dict(zip(pl.element, pl.web))
    m = F.match_players(df, ro, web)
    last = ro[ro.season == SEASON - 1].groupby('player_id').player.last()
    toks = {pid: tuple(F.norm(n)) for pid, n in last.items()}
    for r in pl[~pl.element.isin(m)].itertuples():
        t = set(F.norm(r.name)); w = tuple(F.norm(r.web))
        hits = [pid for pid, u in toks.items() if u and (set(u) <= t or u == w)]
        if len(hits) == 1:
            m[r.element] = hits[0]
    return m


def run(lam_fn, log=print, today=None):
    """Expected points for every player over the next HORIZON gameweeks.
    lam_fn(home, away) -> (home goals, away goals) expected, from the live
    models. Writes outputs/fpl_predictions.csv and the benchmark log, and
    returns the site payload."""
    today = today or dt.date.today()
    boot = get('bootstrap-static/')
    nxt = next((e for e in boot['events'] if e['is_next']), None)
    if nxt is None:
        log('  FPL: no upcoming gameweek - skipped')
        return None
    gws = list(range(nxt['id'], min(nxt['id'] + HORIZON, 39)))
    fx = [f for f in get('fixtures/?future=1') if f['event'] in gws]
    tname = {t['id']: F.FPL2US.get(t['name'], t['name']) for t in boot['teams']}
    pl = players_table(boot)

    # minutes model: last season's tail, then this season's gameweeks
    prev = pd.read_parquet(PREV) if os.path.exists(PREV) else pd.DataFrame()
    cur = season_rows(boot, log)
    code_of = dict(zip(pl.element, pl.code))
    rows = []
    if len(prev):
        rows.append(prev.assign(person=prev.code, order=0)[['person', 'order', 'minutes', 'saves', 'yellow']])
    if len(cur):
        rows.append(cur.assign(person=cur.element.map(code_of), order=cur.gw)
                    [['person', 'order', 'minutes', 'saves', 'yellow']])
    hist = pd.concat(rows, ignore_index=True).dropna(subset=['person']).sort_values(['order'], kind='stable')
    mm = FM.minutes_model(hist)

    # defensive-contribution hit rate per 60+ appearance, this season + half of last
    pri = {}
    if len(prev):
        p0 = prev.drop_duplicates('code')
        for pos, g in p0.groupby('pos'):
            pri[pos] = g.dc_hits.sum() / max(g.dc_apps.sum(), 1)
        prev_dc = p0.set_index('code')[['dc_hits', 'dc_apps']]
    else:
        prev_dc = pd.DataFrame(columns=['dc_hits', 'dc_apps'])
    cur60 = cur[cur.minutes >= 60] if len(cur) else cur
    cur_dc = (cur60.assign(code=cur60.element.map(code_of)).groupby('code').dc_hit.agg(['sum', 'size'])
              if len(cur60) else pd.DataFrame(columns=['sum', 'size']))

    def dc_rate(code, pos):
        if pos == 'GKP':
            return 0.0
        h = cur_dc['sum'].get(code, 0) + 0.5 * prev_dc.dc_hits.get(code, 0)
        n = cur_dc['size'].get(code, 0) + 0.5 * prev_dc.dc_apps.get(code, 0)
        return (h + pri.get(pos, 0.1) * DC_PRIOR_K) / (n + DC_PRIOR_K)

    # chance shares from Understat lineups
    ro = PL.load_rosters()
    ro = ro[ro.season >= SEASON - 1]
    us = match_understat(pl, ro)
    pos_us = {us[e]: p for e, p in zip(pl.element, pl.pos) if e in us}
    sh = FM.shares(FM.roster_frame(ro), pd.Timestamp(today), pos_us)
    cal = load_calibration()

    lam = {}
    recs = []
    for f in fx:
        h, a = tname[f['team_h']], tname[f['team_a']]
        if (h, a) not in lam:
            lam[(h, a)] = lam_fn(h, a)
        lh, la = lam[(h, a)]
        ko = pd.Timestamp(f['kickoff_time']) if f['kickoff_time'] else pd.Timestamp(today)
        for club, opp, home, lf, lg in ((h, a, True, lh, la), (a, h, False, la, lh)):
            recs.append(dict(gw=f['event'], club=club, opp=opp, home=home, lam_for=lf,
                             lam_ag=lg, kickoff=ko))
    fixt = pd.DataFrame(recs).sort_values(['kickoff'])
    fixt['k'] = fixt.groupby('club').cumcount()

    c = fixt.merge(pl, on='club')
    c['avail'] = [availability(p, k, ko, today) for p, k, ko in
                  zip(c.itertuples(), c.k, c.kickoff)]
    m = mm.reindex(c.code)
    for col in ('p60', 'psub', 'emins', 'saves90', 'yrate'):
        c[col] = m[col].values
    c['yrate'] = c.yrate.fillna(0.0)
    sg = [sh.get(us.get(e), (FM.PRIOR_G[p], FM.PRIOR_A[p])) for e, p in zip(c.element, c.pos)]
    c['share_g'], c['share_a'] = [s[0] for s in sg], [s[1] for s in sg]
    c = FM.fill_new(c)
    for col in ('p60', 'psub', 'emins'):
        c[col] = c[col] * c.avail
    pts = FM.points(c, cal)
    c = pd.concat([c.reset_index(drop=True), pts.reset_index(drop=True)], axis=1)
    c['dc_pts'] = [2 * dc_rate(cd, p) * p60 for cd, p, p60 in zip(c.code, c.pos, c.p60)]
    c['xpts'] = c.xpts + c.dc_pts

    os.makedirs(OUT, exist_ok=True)
    keep = ['gw', 'element', 'web', 'club', 'pos', 'opp', 'home', 'kickoff', 'price', 'own',
            'status', 'avail', 'p60', 'eg', 'ea', 'pcs', 'ebonus', 'dc_pts', 'xpts', 'ep_next']
    c[keep].to_csv(f'{OUT}/fpl_predictions.csv', index=False, float_format='%.4g')

    # FPL's own pre-deadline expected points for the next gameweek, beside
    # ours. Rewritten every run until the deadline; once the next gameweek
    # moves on, these rows are never touched again.
    g1 = c[c.gw == gws[0]].groupby('element').agg(model=('xpts', 'sum'), fpl=('ep_next', 'first')).reset_index()
    bf = f'{OUT}/fpl_benchmark.csv'
    old = pd.read_csv(bf) if os.path.exists(bf) else pd.DataFrame(columns=['gw'])
    old = old[old.gw != gws[0]]
    bench = pd.concat([old, g1.assign(gw=gws[0], recorded=dt.datetime.now(dt.timezone.utc).isoformat(
        timespec='seconds'))], ignore_index=True)
    bench[['gw', 'element']] = bench[['gw', 'element']].astype(int)
    bench[['gw', 'element', 'model', 'fpl', 'recorded']].to_csv(bf, index=False, float_format='%.3f')

    per = c.groupby(['element', 'gw']).agg(x=('xpts', 'sum'), opp=('opp', lambda s: '/'.join(s)),
                                           home=('home', 'first')).reset_index()
    tot = per.pivot(index='element', columns='gw', values='x').reindex(columns=gws).fillna(0)
    first = c[c.gw == gws[0]].groupby('element').agg(eg=('eg', 'sum'), ea=('ea', 'sum'), pcs=('pcs', 'max'))
    meta = pl.set_index('element')
    out = []
    for e in tot.index:
        p = meta.loc[e]
        x1, x3, x5 = tot.loc[e].iloc[:1].sum(), tot.loc[e].iloc[:3].sum(), tot.loc[e].sum()
        if x5 < 0.5 and p.own < 1:
            continue                           # unlikely to play and nobody owns him
        fixtures = per[per.element == e].sort_values('gw')
        out.append(dict(
            id=int(e), name=p.web, club=p.club, pos=p.pos, price=p.price, own=p.own,
            status=p.status, news=p.news, x1=round(x1, 2), x3=round(x3, 2), x5=round(x5, 2),
            ep_next=p.ep_next, eg=round(first.eg.get(e, 0), 2), ea=round(first.ea.get(e, 0), 2),
            cs=round(first.pcs.get(e, 0), 3),
            next=[dict(gw=int(r.gw), opp=r.opp, home=bool(r.home), x=round(r.x, 2))
                  for r in fixtures.itertuples()]))
    out.sort(key=lambda r: -r['x1'])
    matched = sum(1 for e in pl.element if e in us)
    log(f'  FPL: {len(out)} players over GW{gws[0]}-{gws[-1]}; '
        f'{matched}/{len(pl)} linked to Understat; top pick {out[0]["name"]} {out[0]["x1"]:.1f}')
    payload = dict(gws=gws, deadline=nxt['deadline_time'], players=out,
                   benchmark_note="FPL's own pre-deadline expected points are recorded "
                                  "beside these for an honest comparison.")
    # plain Python numbers only: numpy types would make the site's JSON write
    # fail, and that would take the whole data.json down with it
    return json.loads(json.dumps(payload, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
