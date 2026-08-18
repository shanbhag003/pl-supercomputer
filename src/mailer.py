"""Compose and send the weekly email.

Two versions of the same content: an HTML one styled like the graphic, and a
plain-text fallback for clients that block HTML. Everything is written for
someone who has never read a football model in their life.
"""
import os, smtplib, ssl, textwrap
from email.message import EmailMessage

PURPLE, PINK, GREEN, CYAN = '#37003C', '#FF2882', '#00B85C', '#0090A8'
GREY, LIGHT = '#6B5A72', '#F4F0F6'


def _pct(v):
    return '—' if v * 100 < 0.05 else f'{100 * v:.1f}%'


def _movement_reason():
    return ("Numbers move for three reasons: results, how well teams actually "
            "played underneath those results, and any change in squad or manager.")


# ------------------------------------------------------------------ plain text
def build_body(pred, status, gw_label, results_note=''):
    top = pred.head(6)
    bottom = pred.nlargest(3, 'releg')
    moves = status.get('biggest_moves') or []
    val = status.get('validation') or {}
    L = []
    A = L.append

    A(f'PREMIER LEAGUE PREDICTIONS — {gw_label}')
    A('=' * 58)
    A('')
    lead = pred.iloc[0]
    A(f'{lead.team} are favourites for the title, with a '
      f'{_pct(lead.title)} chance.')
    A('')
    A('We simulated the rest of the season 20,000 times. These are the')
    A('percentages of those runs in which each thing happened.')
    A('')

    A('TOP OF THE TABLE')
    A('-' * 58)
    A(f'{"":3}{"TEAM":<20}{"POINTS":>8}{"TITLE":>9}{"TOP 4":>9}')
    for _, r in top.iterrows():
        A(f'{int(r["pos"]):<3}{r.team:<20}{r.xPts:>8.0f}'
          f'{_pct(r.title):>9}{_pct(r.top4):>9}')
    A('')

    A('MOST LIKELY TO GO DOWN')
    A('-' * 58)
    for _, r in bottom.iterrows():
        A(f'   {r.team:<20}{_pct(r.releg):>9}')
    A('')

    A('WHAT CHANGED SINCE LAST WEEK')
    A('-' * 58)
    if not moves:
        A('  This is the first published forecast, so there is nothing to')
        A('  compare it against yet.')
    else:
        shown = False
        for m in moves:
            d = m.get('d_xPts', 0)
            if abs(d) < 0.05:
                continue
            shown = True
            direction = 'up' if d > 0 else 'down'
            bits = [f'{abs(d):.1f} expected points {direction}']
            if abs(m.get('d_title', 0)) >= 0.1:
                bits.append(f'title chance {m["d_title"]:+.1f} points')
            if abs(m.get('d_releg', 0)) >= 0.1:
                bits.append(f'relegation risk {m["d_releg"]:+.1f} points')
            A(f'  {m["team"]}: ' + ', '.join(bits))
        if not shown:
            A('  Nothing moved by more than a rounding error this week.')
        A('')
        for ln in textwrap.wrap(_movement_reason(), 56):
            A('  ' + ln)
    A('')

    A('HOW ACCURATE HAS THIS BEEN?')
    A('-' * 58)
    if not val:
        A('  No score yet. We grade only predictions made BEFORE a match was')
        A('  played, so the first scores appear once a forecast gameweek has')
        A('  actually happened.')
    else:
        A(f'  Matches predicted so far: {val.get("n", 0)}')
        A(f'  Results called correctly: {100 * val.get("hit", 0):.0f}%')
        A(f'  Sharpness score: {val.get("rps", float("nan")):.3f} '
          f'(lower is better)')
        if 'bookie_rps' in val:
            b = val['bookie_rps']
            gap = 100 * (val['rps'] / b - 1)
            side = 'behind' if gap > 0 else 'ahead of'
            A(f'  Bookmakers on the same matches: {b:.3f}')
            A(f'  So we are {abs(gap):.1f}% {side} the betting market.')
        A('')
        A('  Nothing here is graded with hindsight.')
    A('')

    A('HOW IT WORKS')
    A('-' * 58)
    A('  1. Every club gets an attack and a defence score, based on twelve')
    A('     seasons of results and the quality of chances created.')
    A('  2. Those scores are adjusted for who has arrived, who has left,')
    A('     who is injured or suspended, and who is now in charge.')
    A('  3. Every remaining match is then played out 20,000 times, and we')
    A('     count how often each club finished where.')
    A('')
    A('Deliberately left out: fixture congestion (tested across twelve')
    A('seasons - the effect was not consistent enough to trust), and manager')
    A('changes where the incoming manager has no Premier League record, since')
    A('there is nothing to measure them against.')
    return '\n'.join(L)


# ------------------------------------------------------------------------ HTML
def _row(r, i):
    bg = '#FFFFFF' if i % 2 == 0 else LIGHT
    return (
        f'<tr style="background:{bg};">'
        f'<td style="padding:9px 6px;color:{GREY};font-size:13px;width:28px;">'
        f'{int(r["pos"])}</td>'
        f'<td style="padding:9px 6px;font-weight:600;font-size:14px;">{r.team}</td>'
        f'<td style="padding:9px 6px;text-align:right;font-size:14px;">'
        f'{r.xPts:.0f}</td>'
        f'<td style="padding:9px 6px;text-align:right;font-size:14px;'
        f'color:{GREEN};font-weight:600;">{_pct(r.title)}</td>'
        f'<td style="padding:9px 6px;text-align:right;font-size:14px;'
        f'color:{CYAN};">{_pct(r.top4)}</td></tr>')


def build_html(pred, status, gw_label):
    moves = status.get('biggest_moves') or []
    val = status.get('validation') or {}
    lead = pred.iloc[0]
    H = []
    A = H.append

    A(f'<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,'
      f'Arial,sans-serif;max-width:600px;margin:0 auto;color:#1B1020;">')
    A(f'<div style="background:{PURPLE};padding:24px 22px;">'
      f'<div style="color:#FFF;font-size:24px;font-weight:700;'
      f'letter-spacing:-.3px;">Premier League Predictions</div>'
      f'<div style="color:{PINK};font-size:15px;font-weight:600;'
      f'margin-top:4px;">{gw_label}</div></div>')

    A(f'<div style="padding:20px 22px;font-size:15px;line-height:1.55;">'
      f'<b>{lead.team}</b> are favourites for the title, with a '
      f'<b>{_pct(lead.title)}</b> chance.<br><br>'
      f'<span style="color:{GREY};">We played out the rest of the season '
      f'20,000 times. The percentages below are how often each thing '
      f'happened.</span></div>')

    A(f'<div style="padding:0 22px;">'
      f'<table style="width:100%;border-collapse:collapse;">'
      f'<tr style="border-bottom:2px solid {PINK};">'
      f'<th></th><th style="text-align:left;padding:6px;font-size:11px;'
      f'color:{GREY};letter-spacing:.5px;">TEAM</th>'
      f'<th style="text-align:right;padding:6px;font-size:11px;color:{GREY};">'
      f'POINTS</th>'
      f'<th style="text-align:right;padding:6px;font-size:11px;color:{GREY};">'
      f'TITLE</th>'
      f'<th style="text-align:right;padding:6px;font-size:11px;color:{GREY};">'
      f'TOP 4</th></tr>')
    for i, (_, r) in enumerate(pred.head(6).iterrows()):
        A(_row(r, i))
    A('</table></div>')

    A(f'<div style="padding:18px 22px 4px;font-size:11px;color:{GREY};'
      f'letter-spacing:.5px;">MOST LIKELY TO GO DOWN</div>'
      f'<div style="padding:0 22px;"><table style="width:100%;'
      f'border-collapse:collapse;">')
    for i, (_, r) in enumerate(pred.nlargest(3, 'releg').iterrows()):
        bg = '#FFFFFF' if i % 2 == 0 else LIGHT
        A(f'<tr style="background:{bg};"><td style="padding:9px 6px;'
          f'font-weight:600;font-size:14px;">{r.team}</td>'
          f'<td style="padding:9px 6px;text-align:right;color:{PINK};'
          f'font-weight:600;font-size:14px;">{_pct(r.releg)}</td></tr>')
    A('</table></div>')

    A(f'<div style="padding:22px 22px 6px;font-size:16px;font-weight:700;">'
      f'What changed since last week</div><div style="padding:0 22px;'
      f'font-size:14px;line-height:1.6;color:#2C1F33;">')
    if not moves:
        A('This is the first published forecast, so there is nothing to '
          'compare it against yet.')
    else:
        shown = False
        A('<ul style="margin:4px 0;padding-left:20px;">')
        for m in moves:
            d = m.get('d_xPts', 0)
            if abs(d) < 0.05:
                continue
            shown = True
            direction = 'up' if d > 0 else 'down'
            bits = [f'{abs(d):.1f} expected points {direction}']
            if abs(m.get('d_title', 0)) >= 0.1:
                bits.append(f'title chance {m["d_title"]:+.1f} points')
            if abs(m.get('d_releg', 0)) >= 0.1:
                bits.append(f'relegation risk {m["d_releg"]:+.1f} points')
            A(f'<li><b>{m["team"]}</b> — {", ".join(bits)}</li>')
        A('</ul>')
        if not shown:
            A('Nothing moved by more than a rounding error this week.')
        A(f'<div style="color:{GREY};font-size:13px;margin-top:6px;">'
          f'{_movement_reason()}</div>')
    A('</div>')

    A(f'<div style="padding:22px 22px 6px;font-size:16px;font-weight:700;">'
      f'How accurate has this been?</div><div style="padding:0 22px;'
      f'font-size:14px;line-height:1.6;">')
    if not val:
        A(f'<span style="color:{GREY};">No score yet. We grade only '
          f'predictions made <i>before</i> a match was played, so the first '
          f'scores appear once a forecast gameweek has actually happened.'
          f'</span>')
    else:
        A(f'<table style="width:100%;border-collapse:collapse;font-size:14px;">')
        rows = [('Matches predicted', str(val.get('n', 0))),
                ('Results called correctly', f'{100 * val.get("hit", 0):.0f}%'),
                ('Sharpness score (lower is better)', f'{val.get("rps", 0):.3f}')]
        if 'bookie_rps' in val:
            gap = 100 * (val['rps'] / val['bookie_rps'] - 1)
            side = 'behind' if gap > 0 else 'ahead of'
            rows.append(('Bookmakers, same matches', f'{val["bookie_rps"]:.3f}'))
            rows.append(('Versus the market',
                         f'{abs(gap):.1f}% {side}'))
        for k, v in rows:
            A(f'<tr><td style="padding:5px 0;color:{GREY};">{k}</td>'
              f'<td style="padding:5px 0;text-align:right;font-weight:600;">'
              f'{v}</td></tr>')
        A('</table>')
        A(f'<div style="color:{GREY};font-size:13px;margin-top:8px;">'
          f'Nothing here is graded with hindsight.</div>')
    A('</div>')

    A(f'<div style="padding:22px 22px 6px;font-size:16px;font-weight:700;">'
      f'How it works</div>'
      f'<div style="padding:0 22px;font-size:14px;line-height:1.7;'
      f'color:#2C1F33;"><ol style="margin:4px 0;padding-left:20px;">'
      f'<li>Every club gets an attack and a defence score, from twelve '
      f'seasons of results and the quality of chances created.</li>'
      f'<li>Those scores are adjusted for arrivals, departures, injuries, '
      f'suspensions and who is now in charge.</li>'
      f'<li>Every remaining match is played out 20,000 times, and we count '
      f'how often each club finished where.</li></ol></div>')

    A(f'<div style="padding:18px 22px 26px;font-size:12px;color:{GREY};'
      f'line-height:1.6;border-top:1px solid #E3DAE8;margin-top:18px;">'
      f'Deliberately left out: fixture congestion (tested across twelve '
      f'seasons — the effect was not consistent enough to trust), and manager '
      f'changes where the incoming manager has no Premier League record, since '
      f'there is nothing to measure them against.<br>'
      f'Data: Understat, football-data.co.uk and the official Fantasy '
      f'Premier League API.</div></div>')
    return ''.join(H)


def send(subject, body, image_path=None, attachments=(), html=None):
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
    if html:
        m.add_alternative(html, subtype='html')

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
