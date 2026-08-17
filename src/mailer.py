"""Email the weekly graphic plus a plain-text explanation of what changed.

Credentials come from environment variables (GitHub Secrets), never the repo:
  GMAIL_USER          e.g. you@gmail.com
  GMAIL_APP_PASSWORD  16-char Google App Password (NOT your Gmail password)
  MAIL_TO             recipient
"""
import os, smtplib, ssl
from email.message import EmailMessage


def build_body(pred, status, gw_label, results_note=''):
    gw = status.get('gameweek', 0)
    moves = status.get('biggest_moves') or []
    val = status.get('validation') or {}
    top = pred.head(3)

    L = []
    L.append(f'PREMIER LEAGUE SUPERCOMPUTER — {gw_label}')
    L.append('=' * 52)
    L.append('')
    L.append(f"{status.get('n_sims', 20000):,} Monte Carlo simulations of every "
             f"remaining fixture.")
    L.append(f"Data through: {status.get('matches_played', 0)} matches played.")
    if results_note:
        L.append(results_note)
    L.append('')

    L.append('TITLE RACE')
    L.append('-' * 52)
    for _, r in top.iterrows():
        L.append(f'  {r["pos"]}. {r.team:<20} {r.xPts:5.1f} xPts   '
                 f'{100*r.title:5.1f}% title')
    L.append('')

    rel = pred.nlargest(3, 'releg')
    L.append('RELEGATION')
    L.append('-' * 52)
    for _, r in rel.iterrows():
        L.append(f'  {r.team:<20} {100*r.releg:5.1f}%')
    L.append('')

    L.append('WHY THE MODEL CHANGED')
    L.append('-' * 52)
    if not moves:
        L.append('  First published run — no previous forecast to compare against.')
    else:
        any_move = False
        for m in moves:
            if abs(m.get('d_xPts', 0)) < 0.05:
                continue
            any_move = True
            d = m['d_xPts']
            bits = [f'{d:+.1f} xPts']
            if abs(m.get('d_title', 0)) >= 0.1:
                bits.append(f"title {m['d_title']:+.1f}pp")
            if abs(m.get('d_releg', 0)) >= 0.1:
                bits.append(f"relegation {m['d_releg']:+.1f}pp")
            L.append(f"  {m['team']:<20} {', '.join(bits)}")
        if not any_move:
            L.append('  No team moved by more than 0.1 expected points.')
        L.append('')
        L.append('  Movement comes from three things: results (a win shifts points'),
        L.append('  directly), underlying xG performance (a team can win and still')
        L.append('  have its rating fall), and shrinking uncertainty as the season')
        L.append('  goes on.')
    L.append('')

    L.append('MODEL ACCURACY SO FAR')
    L.append('-' * 52)
    if not val:
        L.append('  No scored predictions yet — accuracy tracking starts once the')
        L.append('  model has forecast a gameweek that has since been played.')
    else:
        L.append(f"  Matches scored : {val.get('n', 0)}")
        L.append(f"  RPS            : {val.get('rps', float('nan')):.4f}  (lower is better)")
        L.append(f"  Log loss       : {val.get('logloss', float('nan')):.4f}")
        L.append(f"  Outright hit   : {100*val.get('hit', 0):.0f}% of results called correctly")
        if 'bookie_rps' in val:
            b = val['bookie_rps']
            gap = 100 * (val['rps'] / b - 1)
            L.append(f"  Bookmaker RPS  : {b:.4f}  ->  we are {gap:+.1f}% vs the market")
    L.append('')

    L.append('-' * 52)
    L.append('Dixon-Coles xG model, 80 bootstrap refits for parameter uncertainty.')
    L.append('Squad layer: player values from a regularised plus-minus model over')
    L.append('12 seasons, applied to current rosters and availability (FPL).')
    L.append('Data: Understat (xG) + football-data.co.uk (results, odds) + FPL.')
    L.append('')
    L.append('Not modelled: manager changes (tested, made forecasts worse, so it')
    L.append('is switched off), European fixture congestion, per-club home advantage.')
    return '\n'.join(L)


def send(subject, body, image_path=None, attachments=()):
    user = os.environ.get('GMAIL_USER')
    pw = os.environ.get('GMAIL_APP_PASSWORD')
    to = os.environ.get('MAIL_TO', user)
    if not (user and pw and to):
        print('[mail] credentials not set, skipping email')
        return False

    m = EmailMessage()
    m['Subject'] = subject
    m['From'] = user
    m['To'] = to
    m.set_content(body)

    for p in ([image_path] if image_path else []) + list(attachments):
        if p and os.path.exists(p):
            sub = 'png' if p.endswith('.png') else 'csv'
            maj = 'image' if sub == 'png' else 'text'
            m.add_attachment(open(p, 'rb').read(), maintype=maj, subtype=sub,
                             filename=os.path.basename(p))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465,
                          context=ssl.create_default_context()) as s:
        s.login(user, pw)
        s.send_message(m)
    print(f'[mail] sent to {to}')
    return True
