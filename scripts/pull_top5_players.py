"""Pull every player from the top five European leagues off Understat, into one Excel file.

Usage
-----
    python scripts/pull_top5_players.py                 # current season, all five leagues
    python scripts/pull_top5_players.py --season 2024   # 2024/25
    python scripts/pull_top5_players.py --season 2023 2024 2025
    python scripts/pull_top5_players.py --leagues EPL La_liga
    python scripts/pull_top5_players.py --out data/players.xlsx

Understat keeps its league data in a JavaScript variable on the league page rather
than behind an API, so this reads the page and pulls `playersData` out of the
script tag. That string is hex-escaped JSON, hence the unicode_escape decode.

Seasons are named by their STARTING year: 2025 means 2025/26.
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

HDR = {'User-Agent': 'Mozilla/5.0 (compatible; pl-supercomputer/1.0)'}
BASE = 'https://understat.com/league'

# Numeric columns as Understat names them. Everything arrives as a string.
NUMERIC = ['games', 'time', 'goals', 'xG', 'assists', 'xA', 'shots',
           'key_passes', 'yellow_cards', 'red_cards', 'npg', 'npxG',
           'xGChain', 'xGBuildup']


def scrape(league, season, session, retries=3):
    """Return playersData for one league-season as a list of dicts."""
    url = f'{BASE}/{league}/{season}'
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HDR, timeout=40)
            r.raise_for_status()
            m = re.search(r"var\s+playersData\s*=\s*JSON\.parse\('(.*?)'\)",
                          r.text, re.S)
            if not m:
                raise ValueError('playersData not found - page layout changed, '
                                 'or this league-season does not exist')
            raw = m.group(1).encode('utf8').decode('unicode_escape')
            return json.loads(raw)
        except Exception as e:                       # noqa: BLE001
            last = e
            if attempt < retries - 1:
                wait = 3 * (attempt + 1)
                print(f'    retry {attempt + 1}/{retries - 1} in {wait}s ({e})',
                      flush=True)
                time.sleep(wait)
    raise RuntimeError(f'{league} {season}: {last}')


def tidy(rows, league, season):
    """One league-season of raw records into a typed, sorted frame."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={'player_name': 'player', 'team_title': 'team',
                            'time': 'minutes'})
    for c in [c for c in NUMERIC if c in df.columns] + ['minutes']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df.insert(0, 'league', LEAGUES.get(league, league))
    df.insert(1, 'season', f'{season}/{str(season + 1)[-2:]}')

    # Per-90s are what anyone actually compares players on. Guard the divide:
    # a player with 0 minutes would otherwise produce inf and poison the sort.
    # NaN rather than pd.NA: pandas' NAType has no __round__, so .round()
    # below raises on any player with zero minutes.
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

    order = ['league', 'season', 'player', 'team', 'position', 'games',
             'minutes', 'goals', 'assists', 'xG', 'npxG', 'xA', 'npg',
             'shots', 'key_passes', 'xGChain', 'xGBuildup',
             'yellow_cards', 'red_cards',
             'goals_90', 'assists_90', 'xG_90', 'npxG_90', 'xA_90',
             'npxG_xA_90', 'shots_90', 'key_passes_90',
             'xGChain_90', 'xGBuildup_90', 'goals_minus_xG', 'id']
    df = df[[c for c in order if c in df.columns]]
    return df.sort_values(['minutes', 'npxG'], ascending=False, na_position='last')


def autofit(writer, sheet_name, df):
    """Column widths and a frozen, filtered header. Excel does neither by default."""
    ws = writer.sheets[sheet_name]
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))
    for i, col in enumerate(df.columns):
        longest = df[col].astype(str).str.len().max() if len(df) else 0
        width = min(max(len(str(col)) + 2, int(longest or 0) + 2), 38)
        ws.set_column(i, i, width)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, nargs='+',
                    default=[datetime.now(timezone.utc).year
                             - (1 if datetime.now(timezone.utc).month < 7 else 0)],
                    help='starting year(s), e.g. 2025 for 2025/26')
    ap.add_argument('--leagues', nargs='+', default=list(LEAGUES),
                    choices=list(LEAGUES))
    ap.add_argument('--out', default='top5_players.xlsx')
    ap.add_argument('--min-minutes', type=int, default=0,
                    help='drop players below this many minutes')
    a = ap.parse_args()

    session = requests.Session()
    frames, failed = [], []
    for season in a.season:
        for lg in a.leagues:
            print(f'  {LEAGUES[lg]} {season}/{str(season + 1)[-2:]} ...',
                  end=' ', flush=True)
            try:
                df = tidy(scrape(lg, season, session), lg, season)
                if a.min_minutes:
                    df = df[df['minutes'].fillna(0) >= a.min_minutes]
                frames.append(df)
                print(f'{len(df)} players')
            except Exception as e:                    # noqa: BLE001
                failed.append(f'{lg} {season}: {e}')
                print('FAILED')
            time.sleep(1.2)                           # be polite to Understat

    if not frames:
        print('\nNothing pulled. Understat may be down or blocking; try again later.',
              file=sys.stderr)
        return 1

    allp = pd.concat(frames, ignore_index=True)

    with pd.ExcelWriter(a.out, engine='xlsxwriter') as w:
        book = w.book
        pct = book.add_format({'font_name': 'Arial'})

        allp.to_excel(w, sheet_name='All players', index=False)
        autofit(w, 'All players', allp)

        for lg in a.leagues:
            name = LEAGUES[lg]
            sub = allp[allp['league'] == name]
            if sub.empty:
                continue
            sheet = name[:31]
            sub.to_excel(w, sheet_name=sheet, index=False)
            autofit(w, sheet, sub)

        # Team-level totals, which is the view most people want next.
        team = (allp.groupby(['league', 'season', 'team'], as_index=False)
                .agg(players=('player', 'nunique'), minutes=('minutes', 'sum'),
                     goals=('goals', 'sum'), assists=('assists', 'sum'),
                     xG=('xG', 'sum'), npxG=('npxG', 'sum'), xA=('xA', 'sum'),
                     shots=('shots', 'sum'))
                .round(2)
                .sort_values(['league', 'npxG'], ascending=[True, False]))
        team.to_excel(w, sheet_name='Teams', index=False)
        autofit(w, 'Teams', team)

        meta = pd.DataFrame({
            'field': ['source', 'pulled (UTC)', 'seasons', 'leagues',
                      'players', 'teams', 'min minutes filter', 'failures'],
            'value': ['https://understat.com',
                      datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),
                      ', '.join(str(s) for s in a.season),
                      ', '.join(LEAGUES[l] for l in a.leagues),
                      len(allp), team['team'].nunique(),
                      a.min_minutes, '; '.join(failed) if failed else 'none']})
        meta.to_excel(w, sheet_name='About', index=False)
        autofit(w, 'About', meta)

    print(f'\nwrote {a.out}: {len(allp)} players, {team["team"].nunique()} teams, '
          f'{len(a.leagues)} leagues')
    if failed:
        print('failures:', *failed, sep='\n  ')
    return 0


if __name__ == '__main__':
    sys.exit(main())
