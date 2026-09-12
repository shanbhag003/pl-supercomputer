"""Probe: will FBref serve a GitHub Actions runner, and can we parse what it sends?

Answers three questions and nothing else. Do not build on FBref until all three
come back green.

  1. Does FBref answer at all from this IP?  Sports Reference has a history of
     blocking datacentre ranges, and Actions runners are Azure IPs.
  2. Are the tables parseable?  FBref hides some inside HTML comments as an
     anti-scraping measure, which makes pandas.read_html return nothing.
  3. How hard is the rate limiting?  The maintained soccerdata library waits
     7 seconds between FBref requests, versus 1 second for its other sources.
     That is a maintainer who has been burned, so this measures it directly.

Exit code is 0 only if every page fetched and parsed.
"""
import re
import sys
import time
from io import StringIO

import pandas as pd
import requests

# A real browser UA. Sports Reference blocks obvious bot agents outright, and
# a 403 from that would tell us nothing about whether the IP itself is fine.
HDR = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/124.0.0.0 Safari/537.36'),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.9',
}

BASE = 'https://fbref.com/en/comps'
DELAY = 7          # what soccerdata uses. Do not lower it.

# One page per category we would actually need, all Premier League 2023/24 so
# the season is complete and the numbers are stable.
PROBES = [
    ('defensive actions', f'{BASE}/9/2023-2024/defense/2023-2024-Premier-League-Stats',
     'stats_defense', ['Tkl', 'Int', 'Blocks', 'Clr']),
    ('goalkeeping',       f'{BASE}/9/2023-2024/keepers/2023-2024-Premier-League-Stats',
     'stats_keeper', ['Saves', 'CS', 'GA']),
    ('advanced GK',       f'{BASE}/9/2023-2024/keepersadv/2023-2024-Premier-League-Stats',
     'stats_keeper_adv', ['PSxG', 'PSxG+/-']),
    ('miscellaneous',     f'{BASE}/9/2023-2024/misc/2023-2024-Premier-League-Stats',
     'stats_misc', ['Won', 'Lost', 'Recov']),
    ('possession',        f'{BASE}/9/2023-2024/possession/2023-2024-Premier-League-Stats',
     'stats_possession', ['Touches', 'Carries']),
]


def uncomment(html):
    """FBref buries several tables inside HTML comments. Strip the markers so
    pandas can see them; harmless when a table is not commented."""
    return html.replace('<!--', '').replace('-->', '')


def parse(html, table_id):
    """Pull one table out by id, and report whether it was inside a comment.

    A regex over the raw document finds commented tables anyway, since regex
    does not care about comment markers. So the useful diagnostic is not "did
    stripping help" but "was it commented at all" - that tells us whether a
    library using an HTML parser (which WOULD be blocked by the comments) needs
    the uncomment step. Counting unclosed markers before the match answers it.
    """
    m = re.search(rf'<table[^>]*id="{table_id}".*?</table>', html, re.S)
    if not m:
        m = re.search(rf'<table[^>]*id="{table_id}".*?</table>',
                      uncomment(html), re.S)
        if not m:
            return None, None
    before = html[:m.start()]
    inside_comment = before.count('<!--') > before.count('-->')
    try:
        df = pd.read_html(StringIO(m.group(0)))[0]
        return df, ('in comment' if inside_comment else 'plain')
    except Exception:                                 # noqa: BLE001
        return None, None


def main():
    session = requests.Session()
    results, ok = [], True

    print('FBref probe')
    print(f'{DELAY}s between requests, {len(PROBES)} pages\n')

    for i, (label, url, table_id, want_cols) in enumerate(PROBES):
        if i:
            time.sleep(DELAY)
        line = {'page': label, 'status': None, 'rows': 0, 'cols': 0,
                'commented': '', 'sample_cols': '', 'note': ''}
        t0 = time.time()
        try:
            r = session.get(url, headers=HDR, timeout=45)
            line['status'] = r.status_code
            print(f'  {label:20} HTTP {r.status_code}  {len(r.content)/1024:6.0f} KB  '
                  f'{time.time()-t0:.1f}s', end='')

            if r.status_code == 403:
                line['note'] = 'BLOCKED - datacentre IP or bot detection'
                ok = False
                print('   <- blocked')
                results.append(line)
                continue
            if r.status_code == 429:
                line['note'] = 'RATE LIMITED - back off further'
                ok = False
                print('   <- rate limited')
                results.append(line)
                continue
            r.raise_for_status()

            df, how = parse(r.text, table_id)
            if df is None:
                line['note'] = f'table #{table_id} not found or unparseable'
                ok = False
                print('   <- no table')
                results.append(line)
                continue

            # FBref uses two header rows; flatten so column checks work.
            cols = ([' '.join(str(x) for x in c if 'Unnamed' not in str(x)).strip()
                     for c in df.columns]
                    if isinstance(df.columns, pd.MultiIndex)
                    else [str(c) for c in df.columns])
            found = [c for c in want_cols
                     if any(c.lower() in x.lower() for x in cols)]
            line.update(rows=len(df), cols=len(cols), commented=how,
                        sample_cols=', '.join(cols[:8]),
                        note=f'found {len(found)}/{len(want_cols)} expected: '
                             f'{", ".join(found) or "none"}')
            if len(found) < len(want_cols):
                ok = False
            print(f'   {len(df):4} rows, {len(cols)} cols ({how})')
        except Exception as e:                        # noqa: BLE001
            line['note'] = f'{type(e).__name__}: {e}'
            ok = False
            print(f'   <- FAILED: {e}')
        results.append(line)

    print('\n' + '=' * 70)
    res = pd.DataFrame(results)
    print(res[['page', 'status', 'rows', 'cols', 'commented']].to_string(index=False))
    print()
    for r in results:
        print(f'  {r["page"]:20} {r["note"]}')

    if ok:
        print('\nVERDICT: FBref works from this runner. Safe to build.')
        good = [r for r in results if r['rows']]
        if good:
            print(f'\nSample columns from "{good[0]["page"]}":')
            print(f'  {good[0]["sample_cols"]}')
    else:
        print('\nVERDICT: FBref is NOT usable from this runner as configured.')
        print('  403 on everything  -> datacentre IP blocked. No fix from inside')
        print('                        Actions; needs a residential IP or a')
        print('                        different source.')
        print('  429                -> raise DELAY and retry.')
        print('  200 but no table   -> page layout changed; the table ids in')
        print('                        PROBES need updating.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
