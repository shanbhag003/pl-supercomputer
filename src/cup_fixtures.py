"""Domestic cup fixtures for Premier League clubs, 2014-2026.

Wikipedia renders each cup tie as its own small table: the date sits in the
first cell, the two clubs either side of the score. We walk every table on the
page and keep the ones that look like a fixture.
"""
import io, re, sys, time
import pandas as pd
import requests

HDR = {'User-Agent': 'Mozilla/5.0'}
OUT = '/home/claude/pl/data/processed/cup_fixtures.csv'

# Wikipedia club names -> the names used in matches.parquet
ALIAS = {
    'Manchester City': 'Manchester City', 'Manchester United': 'Manchester United',
    'Tottenham Hotspur': 'Tottenham', 'Newcastle United': 'Newcastle United',
    'Brighton & Hove Albion': 'Brighton', 'Brighton and Hove Albion': 'Brighton',
    'Wolverhampton Wanderers': 'Wolverhampton Wanderers',
    'Nottingham Forest': 'Nottingham Forest', 'Leeds United': 'Leeds',
    'Leicester City': 'Leicester', 'Norwich City': 'Norwich',
    'Hull City': 'Hull', 'Ipswich Town': 'Ipswich', 'Coventry City': 'Coventry',
    'AFC Bournemouth': 'Bournemouth', 'Bournemouth': 'Bournemouth',
    'West Ham United': 'West Ham', 'West Bromwich Albion': 'West Bromwich Albion',
    'Sheffield United': 'Sheffield United', 'Stoke City': 'Stoke',
    'Swansea City': 'Swansea', 'Cardiff City': 'Cardiff',
    'Huddersfield Town': 'Huddersfield', 'Luton Town': 'Luton',
    'Queens Park Rangers': 'Queens Park Rangers',
}
DATE_RE = re.compile(
    r'^\s*(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})', re.M)


def clean_team(x):
    x = re.sub(r'\(.*?\)', '', str(x))
    x = re.sub(r'\[.*?\]', '', x).strip()
    x = re.sub(r'\s+', ' ', x)
    return ALIAS.get(x, x)


def fixtures_from_page(url):
    r = requests.get(url, headers=HDR, timeout=60)
    if r.status_code != 200:
        return []
    try:
        tabs = pd.read_html(io.StringIO(r.text))
    except Exception:
        return []
    out = []
    for t in tabs:
        if t.shape[1] < 4 or t.shape[0] < 1:
            continue
        try:
            c0 = str(t.iloc[0, 0])
            m = DATE_RE.match(c0)
            if not m:
                continue
            d = pd.to_datetime(m.group(1), errors='coerce', dayfirst=True)
            if pd.isna(d):
                continue
            h, a = clean_team(t.iloc[0, 1]), clean_team(t.iloc[0, 3])
            if not h or not a or h == 'nan' or a == 'nan':
                continue
            out.append((d, h, a))
        except Exception:
            continue
    return out


def season_urls(y):
    a, b = y, str(y + 1)[2:]
    return [(f'https://en.wikipedia.org/wiki/{a}%E2%80%93{b}_FA_Cup', 'FA Cup'),
            (f'https://en.wikipedia.org/wiki/{a}%E2%80%93{b}_EFL_Cup', 'EFL Cup'),
            (f'https://en.wikipedia.org/wiki/{a}%E2%80%93{b}_Football_League_Cup',
             'EFL Cup')]


if __name__ == '__main__':
    rows = []
    for y in range(2014, 2027):
        got = 0
        for url, comp in season_urls(y):
            fx = fixtures_from_page(url)
            if not fx:
                continue
            for d, h, a in fx:
                rows.append(dict(season=y, comp=comp, date=d, home=h, away=a))
            got += len(fx)
            time.sleep(0.5)
        print(f'{y}: {got} ties', flush=True)
    d = pd.DataFrame(rows).drop_duplicates(['date', 'home', 'away'])
    d.to_csv(OUT, index=False)
    print(f'\ntotal {len(d)} cup ties saved')
