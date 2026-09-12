"""Pull everything Understat exposes for the top five European leagues into one Excel file.

Usage
-----
    python scripts/pull_top5_players.py                    # current season, all five leagues
    python scripts/pull_top5_players.py --season 2024
    python scripts/pull_top5_players.py --season 2023 2024 2025
    python scripts/pull_top5_players.py --leagues EPL La_liga
    python scripts/pull_top5_players.py --min-minutes 450
    python scripts/pull_top5_players.py --out data/understat.xlsx

Understat serves this as JSON from getLeagueData/{league}/{season}. One request
per league-season returns three blocks: players, teams (with per-match history)
and dates (the fixture list). All three are written out.

Seasons are named by their STARTING year: 2025 means 2025/26.

A note on what is and is not here. Understat is an expected-goals site, built
from shot events. Every PLAYER field is attacking or disciplinary - there are no
tackles, interceptions, clearances or saves, and a goalkeeper's row is zeros.
Defensive numbers exist only at TEAM level (xGA, npxGA, deep_allowed, PPDA), and
those are captured in full below. For player-level defensive or goalkeeping data
you need a different source; FBref is the usual free one.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

LEAGUES = {
    'EPL': 'Premier League',
    'La_liga': 'La Liga',
    'Bundesliga': 'Bundesliga',
    'Serie_A': 'Serie A',
    'Ligue_1': 'Ligue 1',
}

# X-Requested-With is not optional. Without it the endpoint does not answer.
HDR = {'User-Agent': 'Mozilla/5.0', 'X-Requested-With': 'XMLHttpRequest'}
BASE = 'https://understat.com/getLeagueData'

PLAYER_NUM = ['games', 'time', 'goals', 'xG', 'assists', 'xA', 'shots',
              'key_passes', 'yellow_cards', 'red_cards', 'npg', 'npxG',
              'xGChain', 'xGBuildup']


def season_label(y):
    return f'{y}/{str(y + 1)[-2:]}'


def fetch(league, season, session, retries=3):
    """The whole league-season payload: players, teams, dates."""
    last = None
    for attempt in range(retries):
        try:
            r = session.get(f'{BASE}/{league}/{season}', headers=HDR, timeout=40)
            r.raise_for_status()
            j = r.json()
            if not j.get('players'):
                raise ValueError(f'no players in payload (keys: {list(j)})')
            return j
        except Exception as e:                        # noqa: BLE001
            last = e
            if attempt < retries - 1:
                wait = 3 * (attempt + 1)
                print(f'    retry {attempt + 1}/{retries - 1} in {wait}s ({e})',
                      flush=True)
                time.sleep(wait)

    # Last resort: the old HTML route, which only carries players.
    try:
        r = session.get(f'https://understat.com/league/{league}/{season}',
                        headers=HDR, timeout=40)
        r.raise_for_status()
        m = re.search(r"playersData\s*=\s*JSON\.parse\('(.*?)'\)", r.text, re.S)
        if m:
            print('    (JSON endpoint failed, fell back to the HTML page - '
                  'players only, no team data)', flush=True)
            return {'players': json.loads(
                m.group(1).encode('utf8').decode('unicode_escape'))}
    except Exception:                                 # noqa: BLE001
        pass
    raise RuntimeError(f'{league} {season}: {last}')


# ── players ──────────────────────────────────────────────────────────────────

def players_frame(raw, league, season):
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    df = df.rename(columns={'player_name': 'player', 'team_title': 'team',
                            'time': 'minutes'})
    # Everything arrives as a string. Convert AFTER the rename, and against the
    # renamed column: looking for 'time' here left minutes as text, and the
    # per-90 divide then failed with "unsupported operand: str / int".
    for c in [('minutes' if c == 'time' else c) for c in PLAYER_NUM]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Understat concatenates clubs for anyone who moved mid-season, so
    # team_title arrives as "Everton,Manchester City". Left alone that invents
    # phantom clubs - the Premier League came out with 23 "teams".
    df['teams_all'] = df['team']
    df['team'] = df['team'].astype(str).str.split(',').str[-1].str.strip()
    df['transferred'] = df['teams_all'].astype(str).str.contains(',')

    df.insert(0, 'league', LEAGUES.get(league, league))
    df.insert(1, 'season', season_label(season))

    # NaN rather than pd.NA: NAType has no __round__, so .round() would raise
    # on any player with zero minutes.
    nineties = (df['minutes'] / 90).replace(0, float('nan'))
    for src, dst in (('goals', 'goals_90'), ('assists', 'assists_90'),
                     ('xG', 'xG_90'), ('xA', 'xA_90'), ('npxG', 'npxG_90'),
                     ('shots', 'shots_90'), ('key_passes', 'key_passes_90'),
                     ('xGChain', 'xGChain_90'), ('xGBuildup', 'xGBuildup_90')):
        if src in df.columns:
            df[dst] = (df[src] / nineties).round(3)
    if {'npxG', 'xA'}.issubset(df.columns):
        df['npxG_xA_90'] = ((df['npxG'] + df['xA']) / nineties).round(3)
    if {'goals', 'xG'}.issubset(df.columns):
        df['goals_minus_xG'] = (df['goals'] - df['xG']).round(2)
    if {'assists', 'xA'}.issubset(df.columns):
        df['assists_minus_xA'] = (df['assists'] - df['xA']).round(2)
    if {'goals', 'shots'}.issubset(df.columns):
        df['shot_conversion'] = (df['goals'] / df['shots'].replace(0, float('nan'))).round(3)
    if {'xG', 'shots'}.issubset(df.columns):
        df['xG_per_shot'] = (df['xG'] / df['shots'].replace(0, float('nan'))).round(3)

    order = ['league', 'season', 'player', 'team', 'position', 'games',
             'minutes', 'goals', 'assists', 'xG', 'npxG', 'xA', 'npg',
             'shots', 'key_passes', 'xGChain', 'xGBuildup',
             'yellow_cards', 'red_cards',
             'goals_90', 'assists_90', 'xG_90', 'npxG_90', 'xA_90',
             'npxG_xA_90', 'shots_90', 'key_passes_90', 'xGChain_90',
             'xGBuildup_90', 'goals_minus_xG', 'assists_minus_xA',
             'shot_conversion', 'xG_per_shot', 'transferred', 'teams_all', 'id']
    df = df[[c for c in order if c in df.columns]]
    return df.sort_values(['minutes', 'npxG'], ascending=False, na_position='last')


# ── teams ────────────────────────────────────────────────────────────────────

def team_matches_frame(raw_teams, league, season):
    """One row per team per match. This is where the defensive data lives."""
    rows = []
    for t in raw_teams.values():
        for h in t.get('history', []):
            r = dict(h)
            # ppda arrives as {'att': passes, 'def': defensive actions}. The
            # metric everyone quotes is att/def: passes the opposition were
            # allowed per defensive action, so LOWER means a more intense press.
            for key, out in (('ppda', 'ppda'), ('ppda_allowed', 'ppda_allowed')):
                v = r.pop(key, None)
                if isinstance(v, dict):
                    att, dfn = v.get('att', 0), v.get('def', 0)
                    r[out] = round(att / dfn, 2) if dfn else None
                    r[f'{out}_passes'] = att
                    r[f'{out}_actions'] = dfn
                else:
                    r[out] = v
            r['team'] = t['title']
            r['venue'] = 'home' if r.pop('h_a', '') == 'h' else 'away'
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, 'league', LEAGUES.get(league, league))
    df.insert(1, 'season', season_label(season))
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['xG_diff'] = (df['xG'] - df['xGA']).round(3)
    df['pts_minus_xpts'] = (df['pts'] - df['xpts']).round(3)
    order = ['league', 'season', 'date', 'team', 'venue', 'result',
             'scored', 'missed', 'pts', 'xpts', 'pts_minus_xpts',
             'xG', 'xGA', 'xG_diff', 'npxG', 'npxGA', 'npxGD',
             'deep', 'deep_allowed', 'ppda', 'ppda_allowed',
             'ppda_passes', 'ppda_actions',
             'ppda_allowed_passes', 'ppda_allowed_actions',
             'wins', 'draws', 'loses']
    df = df[[c for c in order if c in df.columns]]
    return df.sort_values(['league', 'date', 'team'])


def team_season_frame(tm):
    """Season totals per club, rolled up from the per-match rows."""
    if tm.empty:
        return tm
    g = (tm.groupby(['league', 'season', 'team'], as_index=False)
           .agg(matches=('date', 'count'), wins=('wins', 'sum'),
                draws=('draws', 'sum'), losses=('loses', 'sum'),
                pts=('pts', 'sum'), xpts=('xpts', 'sum'),
                scored=('scored', 'sum'), conceded=('missed', 'sum'),
                xG=('xG', 'sum'), xGA=('xGA', 'sum'),
                npxG=('npxG', 'sum'), npxGA=('npxGA', 'sum'),
                deep=('deep', 'sum'), deep_allowed=('deep_allowed', 'sum'),
                ppda=('ppda', 'mean'), ppda_allowed=('ppda_allowed', 'mean')))
    g['npxGD'] = (g['npxG'] - g['npxGA']).round(2)
    g['xG_diff'] = (g['xG'] - g['xGA']).round(2)
    g['pts_minus_xpts'] = (g['pts'] - g['xpts']).round(2)
    g['goals_minus_xG'] = (g['scored'] - g['xG']).round(2)
    g['conceded_minus_xGA'] = (g['conceded'] - g['xGA']).round(2)
    m = g['matches'].replace(0, float('nan'))
    for c in ('xG', 'xGA', 'npxG', 'npxGA', 'deep', 'deep_allowed'):
        g[f'{c}_per_match'] = (g[c] / m).round(3)
    for c in ('xpts', 'xG', 'xGA', 'npxG', 'npxGA', 'ppda', 'ppda_allowed'):
        g[c] = g[c].round(2)
    g = g.sort_values(['league', 'pts', 'npxGD'], ascending=[True, False, False])
    g.insert(3, 'rank_by_pts', g.groupby(['league', 'season']).cumcount() + 1)
    return g


def matches_frame(dates, league, season):
    """The fixture list, with Understat's own result probabilities."""
    rows = []
    for d in dates:
        f = d.get('forecast') or {}
        rows.append(dict(
            date=d.get('datetime'), home=(d.get('h') or {}).get('title'),
            away=(d.get('a') or {}).get('title'), played=bool(d.get('isResult')),
            home_goals=(d.get('goals') or {}).get('h'),
            away_goals=(d.get('goals') or {}).get('a'),
            home_xG=(d.get('xG') or {}).get('h'),
            away_xG=(d.get('xG') or {}).get('a'),
            p_home=f.get('w'), p_draw=f.get('d'), p_away=f.get('l'),
            match_id=d.get('id')))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ('home_goals', 'away_goals', 'home_xG', 'away_xG',
              'p_home', 'p_draw', 'p_away'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.insert(0, 'league', LEAGUES.get(league, league))
    df.insert(1, 'season', season_label(season))
    return df.sort_values(['league', 'date'])


# ── output ───────────────────────────────────────────────────────────────────

def write_sheet(writer, name, df):
    if df is None or df.empty:
        return
    sheet = name[:31]
    df.to_excel(writer, sheet_name=sheet, index=False)
    ws = writer.sheets[sheet]
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))
    for i, col in enumerate(df.columns):
        longest = df[col].astype(str).str.len().max() if len(df) else 0
        ws.set_column(i, i, min(max(len(str(col)) + 2, int(longest or 0) + 2), 38))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    now = datetime.now(timezone.utc)
    ap.add_argument('--season', type=int, nargs='+',
                    default=[now.year - (1 if now.month < 7 else 0)],
                    help='starting year(s), e.g. 2025 for 2025/26')
    ap.add_argument('--leagues', nargs='+', default=list(LEAGUES),
                    choices=list(LEAGUES))
    ap.add_argument('--out', default='understat_top5.xlsx')
    ap.add_argument('--min-minutes', type=int, default=0,
                    help='drop players below this many minutes')
    a = ap.parse_args()

    session = requests.Session()
    P, TM, MX, failed = [], [], [], []
    for season in a.season:
        for lg in a.leagues:
            print(f'  {LEAGUES[lg]} {season_label(season)} ...', end=' ', flush=True)
            try:
                j = fetch(lg, season, session)
                p = players_frame(j['players'], lg, season)
                if a.min_minutes:
                    p = p[p['minutes'].fillna(0) >= a.min_minutes]
                P.append(p)
                tm = team_matches_frame(j.get('teams', {}), lg, season)
                if not tm.empty:
                    TM.append(tm)
                mx = matches_frame(j.get('dates', []), lg, season)
                if not mx.empty:
                    MX.append(mx)
                print(f'{len(p)} players, {len(tm)} team-matches, {len(mx)} fixtures')
            except Exception as e:                    # noqa: BLE001
                failed.append(f'{lg} {season}: {e}')
                print('FAILED')
            time.sleep(1.2)                           # be polite to Understat

    if not P:
        print('\nNothing pulled. Understat may be down or blocking; try again later.',
              file=sys.stderr)
        return 1

    players = pd.concat(P, ignore_index=True)
    team_matches = pd.concat(TM, ignore_index=True) if TM else pd.DataFrame()
    fixtures = pd.concat(MX, ignore_index=True) if MX else pd.DataFrame()
    teams = team_season_frame(team_matches)

    with pd.ExcelWriter(a.out, engine='xlsxwriter') as w:
        write_sheet(w, 'All players', players)
        for lg in a.leagues:
            write_sheet(w, LEAGUES[lg], players[players['league'] == LEAGUES[lg]])
        write_sheet(w, 'Teams season', teams)
        write_sheet(w, 'Team matches', team_matches)
        write_sheet(w, 'Fixtures', fixtures)

        notes = pd.DataFrame({'field': [
            'source', 'pulled (UTC)', 'seasons', 'leagues',
            'players', 'teams', 'team-match rows', 'fixtures',
            'min minutes filter', 'failures', '', 'NOTE'],
            'value': [
            'https://understat.com/getLeagueData',
            now.strftime('%Y-%m-%d %H:%M'),
            ', '.join(season_label(s) for s in a.season),
            ', '.join(LEAGUES[l] for l in a.leagues),
            len(players),
            int(teams['team'].nunique()) if not teams.empty else 0,
            len(team_matches), len(fixtures), a.min_minutes,
            '; '.join(failed) if failed else 'none', '',
            'Understat is an expected-goals site built from shot events. Player '
            'stats are attacking only - no tackles, interceptions, clearances or '
            'saves, and goalkeeper rows are zeros. Defensive data exists at TEAM '
            'level: see xGA, npxGA, deep_allowed and PPDA on the Teams sheets.']})
        write_sheet(w, 'About', notes)

    print(f'\nwrote {a.out}')
    print(f'  {len(players)} players | {len(team_matches)} team-matches | '
          f'{len(fixtures)} fixtures')
    if failed:
        print('failures:', *failed, sep='\n  ')
    return 0


if __name__ == '__main__':
    sys.exit(main())
