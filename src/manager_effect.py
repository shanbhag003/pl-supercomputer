"""What actually happens to a club when the manager changes?

Rather than trying to give every manager a coefficient — impossible for anyone
who never left, Guardiola included — this measures the transition itself.

For each change: the club's opponent-adjusted goal difference per match over the
38 matches before, against the 38 after. Uses results back to 1993 for sample
size, since goals are enough here and xG is not required.
"""
import os as _os
# Repo root, resolved from this file. Never hardcode absolute paths:
# they differ between a laptop, a container and a GitHub runner.
ROOT = _os.environ.get(
    "PL_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, re, sys
import numpy as np
import pandas as pd
from scipy import stats

RAW = f'{ROOT}/data/raw'
WINDOW = 38
FD2US = {'Man City': 'Manchester City', 'Man United': 'Manchester United',
         'Newcastle': 'Newcastle United', 'Wolves': 'Wolverhampton Wanderers',
         "Nott'm Forest": 'Nottingham Forest', 'Tottenham': 'Tottenham',
         'Sheffield Weds': 'Sheffield Weds'}


def all_results():
    rows = []
    for f in sorted(glob.glob(f'{RAW}/E0_*.csv')):
        try:
            d = pd.read_csv(f, encoding='latin-1')
        except Exception:
            continue
        if 'Div' in d.columns:
            d = d[d['Div'] == 'E0']
        d = d.dropna(subset=['HomeTeam', 'FTHG'])
        if not len(d):
            continue
        d['date'] = pd.to_datetime(d['Date'], dayfirst=True, format='mixed',
                                   errors='coerce')
        d = d.dropna(subset=['date'])
        rows.append(d[['date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']])
    d = pd.concat(rows, ignore_index=True)
    d['home'] = d.HomeTeam.map(lambda x: FD2US.get(x, x))
    d['away'] = d.AwayTeam.map(lambda x: FD2US.get(x, x))
    return d.sort_values('date').reset_index(drop=True)


def team_matches(d):
    """One row per team per match, with opponent-adjusted goal difference."""
    a = pd.DataFrame(dict(date=d.date, club=d.home, opp=d.away,
                          gd=d.FTHG - d.FTAG, home=1))
    b = pd.DataFrame(dict(date=d.date, club=d.away, opp=d.home,
                          gd=d.FTAG - d.FTHG, home=0))
    t = pd.concat([a, b], ignore_index=True).sort_values('date')
    t['season'] = t.date.dt.year - (t.date.dt.month < 7)
    # opponent strength = that opponent's GD per match in the same season
    opp = t.groupby(['season', 'club']).gd.mean().rename('opp_str')
    t = t.merge(opp, left_on=['season', 'opp'], right_index=True, how='left')
    t['adj_gd'] = t.gd - (-t.opp_str) - 0.30 * (t.home - 0.5) * 2 / 2
    return t


def transitions(mgr, tm):
    out = []
    for club, g in mgr.sort_values('start').groupby('club'):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            prev, new = g.loc[i - 1], g.loc[i]
            d = new.start
            c = tm[tm.club == club].sort_values('date')
            before = c[c.date < d].tail(WINDOW)
            after = c[c.date >= d].head(WINDOW)
            if len(before) < 20 or len(after) < 20:
                continue
            out.append(dict(
                club=club, date=d, out_mgr=prev.manager, in_mgr=new.manager,
                out_days=int(prev.days), n_before=len(before), n_after=len(after),
                before=before.adj_gd.mean(), after=after.adj_gd.mean(),
                effect=after.adj_gd.mean() - before.adj_gd.mean(),
                midseason=int(d.month not in (5, 6, 7, 8))))
    return pd.DataFrame(out)


if __name__ == '__main__':
    mgr = pd.read_csv(f'{ROOT}/data/processed/managers.csv',
                      parse_dates=['start', 'end'])
    d = all_results()
    print(f'results: {len(d)} matches, {d.date.min():%Y} to {d.date.max():%Y}')
    tm = team_matches(d)
    tr = transitions(mgr, tm)
    tr.to_csv(f'{ROOT}/data/processed/manager_transitions.csv', index=False)
    print(f'usable transitions: {len(tr)}\n')

    def report(label, sub):
        if len(sub) < 3:
            print(f'{label:<34} n={len(sub):<4} too few'); return
        m, se = sub.effect.mean(), sub.effect.std() / np.sqrt(len(sub))
        t = stats.ttest_1samp(sub.effect, 0)
        print(f'{label:<34} n={len(sub):<4} effect {m:+.3f} '
              f'+/- {1.96*se:.3f}  p={t.pvalue:.3f}')

    report('ALL manager changes', tr)
    report('  mid-season (sacking)', tr[tr.midseason == 1])
    report('  summer change', tr[tr.midseason == 0])
    print()
    for lo, hi, lab in [(0, 365, 'departing tenure < 1yr'),
                        (365, 1095, '  1-3 yr'),
                        (1095, 1825, '  3-5 yr'),
                        (1825, 99999, '  5+ yr (long tenure)')]:
        report(lab, tr[(tr.out_days >= lo) & (tr.out_days < hi)])
    print()
    long_summer = tr[(tr.out_days >= 1825) & (tr.midseason == 0)]
    report('5+ yr AND summer departure', long_summer)
    if len(long_summer):
        print()
        print(long_summer.nlargest(12, 'out_days')[
            ['club', 'date', 'out_mgr', 'in_mgr', 'out_days', 'effect']]
            .round(3).to_string(index=False))
