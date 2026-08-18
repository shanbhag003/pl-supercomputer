"""Decide whether a gameweek has finished and settled.

GitHub Actions cron can't fire on an event, so we poll cheaply every 2 hours and
exit in ~2 seconds unless a gameweek is genuinely complete.

A gameweek R is ready when ALL of:
  - every fixture in R has been played (per the results feed), and
  - 5 hours have passed since the last match in R finished
    (finish = scheduled kickoff + 2h)
  - we have not already published for R

Postponement fallback: if a match is still unplayed 5 days after its scheduled
kickoff, it is treated as postponed and R can settle without it.
"""
import os, json, sys
import datetime as dt
import pandas as pd

SETTLE_HOURS = 5
MATCH_HOURS = 2
POSTPONE_DAYS = 5


def kickoffs(fx):
    return pd.to_datetime(fx['Date'], dayfirst=True, utc=True)


def decide(fx, played_pairs, now=None, last_published_gw=-1):
    """Returns (should_run, gameweek, reason)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    fx = fx.copy()
    fx['ko'] = kickoffs(fx)
    fx['played'] = [(h, a) in played_pairs for h, a in zip(fx.home, fx.away)]

    ready = []
    for r, g in fx.groupby('Round Number'):
        abandoned = (~g.played) & (now > g.ko + pd.Timedelta(days=POSTPONE_DAYS))
        if not (g.played | abandoned).all():
            continue
        counted = g[g.played]
        if not len(counted):
            continue
        settle = counted.ko.max() + pd.Timedelta(hours=MATCH_HOURS + SETTLE_HOURS)
        if now >= settle:
            ready.append((int(r), settle, int(abandoned.sum())))

    if not ready:
        nxt = fx[~fx.played].ko.min() if (~fx.played).any() else None
        return False, last_published_gw, (
            f'no completed gameweek is settled yet; next kickoff '
            f'{nxt:%Y-%m-%d %H:%M UTC}' if nxt is not None else 'season complete')

    gw, settle, postponed = max(ready)
    if gw <= last_published_gw:
        return False, gw, f'GW{gw} already published'
    note = f' ({postponed} postponed match(es) excluded)' if postponed else ''
    return True, gw, (f'GW{gw} settled at {settle:%Y-%m-%d %H:%M UTC}, '
                      f'{(now - settle).total_seconds() / 3600:.1f}h ago{note}')


def last_published(path):
    if not os.path.exists(path):
        return -1
    try:
        return int(json.load(open(path)).get('gameweek', -1))
    except Exception:
        return -1
