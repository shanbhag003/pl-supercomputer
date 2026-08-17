"""Manager spells for every Premier League club, 1992-present.

Source: Wikipedia's List of Premier League managers (519 rows, with exact
appointment and departure dates).
"""
import io, re, sys
import numpy as np
import pandas as pd
import requests

OUT = '/home/claude/pl/data/processed/managers.csv'
URL = 'https://en.wikipedia.org/wiki/List_of_Premier_League_managers'

WIKI2US = {
    'Tottenham Hotspur': 'Tottenham', 'Brighton & Hove Albion': 'Brighton',
    'AFC Bournemouth': 'Bournemouth', 'West Ham United': 'West Ham',
    'Leeds United': 'Leeds', 'Leicester City': 'Leicester',
    'Hull City': 'Hull', 'Ipswich Town': 'Ipswich', 'Coventry City': 'Coventry',
    'Norwich City': 'Norwich', 'Stoke City': 'Stoke', 'Swansea City': 'Swansea',
    'Cardiff City': 'Cardiff', 'Luton Town': 'Luton',
    'Birmingham City': 'Birmingham', 'Blackburn Rovers': 'Blackburn',
    'Bolton Wanderers': 'Bolton', 'Wigan Athletic': 'Wigan',
    'Queens Park Rangers': 'QPR', 'Sheffield United': 'Sheffield United',
    'Sheffield Wednesday': 'Sheffield Weds', 'West Bromwich Albion': 'West Brom',
    'Charlton Athletic': 'Charlton', 'Bradford City': 'Bradford',
    'Derby County': 'Derby', 'Wimbledon': 'Wimbledon',
    'Nottingham Forest': 'Nottingham Forest', 'Newcastle United': 'Newcastle United',
    'Manchester City': 'Manchester City', 'Manchester United': 'Manchester United',
    'Wolverhampton Wanderers': 'Wolverhampton Wanderers',
    'Huddersfield Town': 'Huddersfield', 'Blackpool': 'Blackpool',
    'Burnley': 'Burnley', 'Watford': 'Watford', 'Fulham': 'Fulham',
    'Sunderland': 'Sunderland', 'Middlesbrough': 'Middlesbrough',
    'Southampton': 'Southampton', 'Portsmouth': 'Portsmouth',
    'Crystal Palace': 'Crystal Palace', 'Everton': 'Everton',
    'Chelsea': 'Chelsea', 'Arsenal': 'Arsenal', 'Liverpool': 'Liverpool',
    'Aston Villa': 'Aston Villa', 'Brentford': 'Brentford',
}


def clean_date(x, end=False):
    if pd.isna(x):
        return pd.NaT
    s = re.sub(r'\[.*?\]', '', str(x)).strip()
    if s.lower() in ('present', '', 'nan'):
        return pd.Timestamp('2026-08-17') if end else pd.NaT
    s = re.sub(r'^\D*?(\d)', r'\1', s)
    for f in (None, '%d %B %Y', '%B %Y', '%Y'):
        try:
            return pd.to_datetime(s, format=f, dayfirst=True) if f else \
                pd.to_datetime(s, dayfirst=True)
        except Exception:
            continue
    return pd.NaT


def scrape():
    r = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
    tabs = pd.read_html(io.StringIO(r.text))
    t = max((x for x in tabs if 'Club' in [str(c) for c in x.columns]),
            key=len).copy()
    t.columns = [str(c) for c in t.columns]
    t = t.rename(columns={'Name': 'manager', 'Club': 'club',
                          'From': 'start', 'Until': 'end'})
    t = t[['manager', 'club', 'start', 'end']].dropna(subset=['club'])
    t['club_raw'] = t.club.map(lambda x: re.sub(r'\[.*?\]', '', str(x)).strip())
    t['club'] = t.club_raw.map(lambda x: WIKI2US.get(x, x))
    t['manager'] = t.manager.map(
        lambda x: re.sub(r'[†§*‡]', '', re.sub(r'\[.*?\]', '', str(x))).strip())
    t['start'] = t.start.map(lambda x: clean_date(x))
    t['end'] = t.apply(lambda r_: clean_date(r_['end'], end=True), axis=1)
    t = t.dropna(subset=['start']).sort_values(['club', 'start'])
    t['end'] = t.end.fillna(pd.Timestamp('2026-08-17'))
    # a spell cannot outlast the start of the next spell at the same club
    nxt = t.groupby('club').start.shift(-1)
    t['end'] = np.where(nxt.notna() & (t.end > nxt), nxt, t.end)
    t['end'] = pd.to_datetime(t['end'])
    t['days'] = (t.end - t.start).dt.days
    t = t[t.days >= 0]
    return t.reset_index(drop=True)


if __name__ == '__main__':
    m = scrape()
    m.to_csv(OUT, index=False)
    print(f'{len(m)} spells, {m.manager.nunique()} managers, {m.club.nunique()} clubs')
    print(f'date range {m.start.min():%Y-%m-%d} to {m.start.max():%Y-%m-%d}')
    print('\nlongest tenures:')
    print(m.nlargest(6, 'days')[['manager', 'club', 'start', 'end', 'days']]
          .to_string(index=False))
    cur = m[m.end >= pd.Timestamp('2026-06-01')]
    print(f'\nspells still active mid-2026: {len(cur)}')
    print(cur[['manager', 'club', 'start']].tail(8).to_string(index=False))
