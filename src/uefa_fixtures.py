"""UEFA fixtures involving English clubs, 2014-2026.

The rendered pages use bracket layouts that don't parse, but the underlying
wikitext stores every match as a `Football box` template with a machine-readable
date and both team names. We read that instead.
"""
import re, sys, time
import pandas as pd
import requests

HDR = {'User-Agent': 'Mozilla/5.0'}
OUT = '/home/claude/pl/data/processed/uefa_fixtures.csv'

ENG = {
    'Arsenal': 'Arsenal', 'Chelsea': 'Chelsea', 'Liverpool': 'Liverpool',
    'Manchester City': 'Manchester City', 'Manchester United': 'Manchester United',
    'Tottenham Hotspur': 'Tottenham', 'Tottenham': 'Tottenham',
    'Newcastle United': 'Newcastle United', 'Aston Villa': 'Aston Villa',
    'West Ham United': 'West Ham', 'Leicester City': 'Leicester',
    'Everton': 'Everton', 'Brighton & Hove Albion': 'Brighton',
    'Brighton and Hove Albion': 'Brighton', 'Brighton': 'Brighton',
    'Wolverhampton Wanderers': 'Wolverhampton Wanderers',
    'Nottingham Forest': 'Nottingham Forest', 'Southampton': 'Southampton',
    'Crystal Palace': 'Crystal Palace', 'Fulham': 'Fulham',
    'Bournemouth': 'Bournemouth', 'AFC Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford',
}
BOX = re.compile(r'Football box(.*?)(?=\{\{Football box|\n==|\Z)', re.S)
DATE1 = re.compile(r'\|\s*date\s*=\s*\{\{Start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})')
DATE2 = re.compile(r'\|\s*date\s*=\s*(\d{1,2}\s+\w+\s+\d{4})')
TEAM = re.compile(r'\|\s*team([12])\s*=\s*(.*)')


def team_name(raw):
    raw = re.sub(r'\{\{[^}]*\}\}', '', raw)
    m = re.search(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', raw)
    name = (m.group(2) or m.group(1)) if m else raw
    name = re.sub(r'\s+F\.?C\.?$', '', name.strip())
    return name.strip()


def parse_page(url):
    try:
        r = requests.get(url, headers=HDR, timeout=60)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    out = []
    for chunk in BOX.findall(r.text):
        d = None
        m = DATE1.search(chunk)
        if m:
            d = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        else:
            m2 = DATE2.search(chunk)
            if m2:
                d = pd.to_datetime(m2.group(1), errors='coerce', dayfirst=True)
        if d is None or pd.isna(d):
            continue
        teams = {}
        for num, raw in TEAM.findall(chunk):
            teams[num] = team_name(raw)
        if '1' not in teams or '2' not in teams:
            continue
        out.append((d, teams['1'], teams['2']))
    return out


def urls(y):
    a, b = y, str(y + 1)[2:]
    s = f'{a}%E2%80%93{b}'
    base = 'https://en.wikipedia.org/w/index.php?action=raw&title='
    pages = [
        (f'{s}_UEFA_Champions_League_knockout_phase', 'UCL'),
        (f'{s}_UEFA_Champions_League_group_stage', 'UCL'),
        (f'{s}_UEFA_Champions_League_league_phase', 'UCL'),
        (f'{s}_UEFA_Europa_League_knockout_phase', 'UEL'),
        (f'{s}_UEFA_Europa_League_group_stage', 'UEL'),
        (f'{s}_UEFA_Europa_League_league_phase', 'UEL'),
        (f'{s}_UEFA_Europa_Conference_League_knockout_phase', 'UECL'),
        (f'{s}_UEFA_Europa_Conference_League_group_stage', 'UECL'),
        (f'{s}_UEFA_Conference_League_knockout_phase', 'UECL'),
        (f'{s}_UEFA_Conference_League_league_phase', 'UECL'),
    ]
    return [(base + p, c) for p, c in pages]


KNOCKOUT = 'knockout'

if __name__ == '__main__':
    rows = []
    for y in range(2014, 2026):
        n = 0
        for url, comp in urls(y):
            for d, t1, t2 in parse_page(url):
                e1, e2 = ENG.get(t1), ENG.get(t2)
                if not e1 and not e2:
                    continue
                rows.append(dict(season=y, comp=comp,
                                 stage='KO' if KNOCKOUT in url else 'group',
                                 date=d, home=e1 or t1, away=e2 or t2,
                                 eng_home=bool(e1), eng_away=bool(e2)))
                n += 1
            time.sleep(0.3)
        print(f'{y}: {n} matches with an English club', flush=True)
    d = pd.DataFrame(rows).drop_duplicates(['date', 'home', 'away'])
    d.to_csv(OUT, index=False)
    print(f'\ntotal {len(d)} UEFA matches involving English clubs')
    print(d.groupby(['comp', 'stage']).size().to_string())
