"""Render the forecast as a Premier League themed PNG for social posting.

No club crests or logos are used (third-party IP). Club identity is shown with a
colour bar only.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle

FONTDIR = os.environ.get('FONTDIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts'))
for f in os.listdir(FONTDIR):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(FONTDIR, f))

ANTON = fm.FontProperties(family='Anton')
BC_B = fm.FontProperties(family='Barlow Condensed', weight='bold')
BC_M = fm.FontProperties(family='Barlow Condensed', weight='medium')

# Premier League brand palette
PURPLE = '#37003C'
PURPLE_D = '#22002A'
PINK = '#FF2882'
CYAN = '#04F5FF'
GREEN = '#00FF85'
WHITE = '#FFFFFF'

CLUB = {
    'Arsenal': '#EF0107', 'Manchester City': '#6CABDD', 'Liverpool': '#C8102E',
    'Manchester United': '#DA291C', 'Bournemouth': '#DA291C',
    'Aston Villa': '#95BFE5', 'Newcastle United': '#241F20',
    'Brighton': '#0057B8', 'Chelsea': '#034694', 'Leeds': '#FFCD00',
    'Brentford': '#E30613', 'Crystal Palace': '#1B458F',
    'Nottingham Forest': '#DD0000', 'Tottenham': '#132257',
    'Everton': '#003399', 'Fulham': '#CC0000', 'Sunderland': '#EB172B',
    'Hull': '#F5A12D', 'Coventry': '#78D0F3', 'Ipswich': '#0044A9',
    'Wolverhampton Wanderers': '#FDB913', 'West Ham': '#7A263A',
    'Burnley': '#6C1D45', 'Leicester': '#003090', 'Southampton': '#D71920',
}
SHORT = {
    'Manchester City': 'MAN CITY', 'Manchester United': 'MAN UNITED',
    'Newcastle United': 'NEWCASTLE', 'Nottingham Forest': "NOTT'M FOREST",
    'Tottenham': 'TOTTENHAM', 'Crystal Palace': 'CRYSTAL PALACE',
    'Wolverhampton Wanderers': 'WOLVES', 'Brighton': 'BRIGHTON',
}


def render(pred, out_path, gw_label, n_sims=20000,
           subtitle=None):
    subtitle = subtitle or BRAND
    d = pred.reset_index(drop=True)
    T = len(d)

    H_TOP, ROW, H_BOT = 2.30, 0.455, 0.72
    fig_h = H_TOP + ROW * T + H_BOT
    fig = plt.figure(figsize=(11.0, fig_h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # background gradient
    backdrop(ax)

    def Y(px):
        return 1 - px / fig_h

    # ---------- header ----------
    ax.add_patch(Rectangle((0, Y(H_TOP - 0.16)), 1, 0.16 / fig_h,
                           color=PINK, zorder=2))
    ax.text(0.045, Y(0.72), '2026-27 PREMIER LEAGUE',
            fontproperties=ANTON, fontsize=41, color=WHITE, va='center_baseline', zorder=3)
    ax.text(0.045, Y(1.22), subtitle,
            fontproperties=ANTON, fontsize=25, color=PINK, va='center_baseline', zorder=3)
    ax.text(0.045, Y(1.66), gw_label.upper(),
            fontproperties=BC_M, fontsize=15.5, color=CYAN, va='center',
            zorder=3, alpha=.95)

    # ---------- column headers ----------
    COLS = [('xPTS', .592), ('TITLE', .676), ('TOP 4', .759), ('TOP 6', .841),
            ('RELEG', .928)]
    yh = Y(H_TOP - 0.30)
    for name, x in COLS:
        ax.text(x, yh, name, fontproperties=BC_B, fontsize=13.5, color=WHITE,
                ha='center', va='center', alpha=.72, zorder=3)
    ax.text(.425, yh, 'LIKELY RANGE', fontproperties=BC_B, fontsize=13.5,
            color=WHITE, ha='center', va='center', alpha=.72, zorder=3)

    lo_all, hi_all = d.lo.min(), d.hi.max()

    def pct_col(v, bad=False):
        if bad:                                   # relegation: high = alarming
            if v >= 40: return PINK
            if v >= 15: return '#FF8FBE'
            if v >= 3: return WHITE
            return '#9C7FA6'
        if v >= 25: return GREEN
        if v >= 8: return CYAN
        if v >= 1: return WHITE
        return '#9C7FA6'

    for i, r in d.iterrows():
        yc = Y(H_TOP + ROW * i + ROW / 2)
        h = (ROW - 0.075) / fig_h

        zone = GREEN if i < 4 else (CYAN if i < 6 else (PINK if i >= T - 3 else None))
        ax.add_patch(FancyBboxPatch((0.037, yc - h / 2), 0.928, h,
                                    boxstyle='round,pad=0,rounding_size=0.009',
                                    fc='#FFFFFF', ec='none', alpha=.075, zorder=2))
        if zone:
            ax.add_patch(Rectangle((0.037, yc - h / 2), 0.0055, h,
                                   color=zone, zorder=3))

        ax.text(0.068, yc, str(i + 1), fontproperties=ANTON, fontsize=17,
                color=WHITE, ha='center', va='center', alpha=.55, zorder=3)
        ax.add_patch(Rectangle((0.089, yc - h / 2), 0.0085, h,
                               color=CLUB.get(r.team, '#888888'), zorder=3))
        ax.text(0.112, yc, SHORT.get(r.team, r.team.upper()),
                fontproperties=BC_B, fontsize=19.5, color=WHITE,
                va='center_baseline', zorder=3)

        # range bar
        x0, x1 = 0.340, 0.492
        def sx(v):
            return x0 + (v - lo_all) / (hi_all - lo_all) * (x1 - x0)
        ax.plot([sx(r.lo), sx(r.hi)], [yc, yc], lw=3.4, solid_capstyle='round',
                color=WHITE, alpha=.30, zorder=3)
        ax.plot([sx(r.xPts)], [yc], marker='o', ms=6.0,
                color=CLUB.get(r.team, WHITE), mec=WHITE, mew=1.3, zorder=4)
        ax.text(x0 - 0.013, yc, f'{r.lo:.0f}', fontproperties=BC_M, fontsize=12.5,
                color=WHITE, alpha=.5, ha='right', va='center_baseline', zorder=3)
        ax.text(x1 + 0.013, yc, f'{r.hi:.0f}', fontproperties=BC_M, fontsize=12.5,
                color=WHITE, alpha=.5, ha='left', va='center_baseline', zorder=3)

        ax.text(.592, yc, f'{r.xPts:.1f}', fontproperties=BC_B, fontsize=18.5,
                color=WHITE, ha='center', va='center_baseline', zorder=3)
        for val, x, bad in [(r.title, .676, False), (r.top4, .759, False),
                            (r.top6, .841, False), (r.releg, .928, True)]:
            v = 100 * val
            s = '—' if v < 0.05 else (f'{v:.1f}%' if v < 99.9 else '>99%')
            ax.text(x, yc, s, fontproperties=BC_B, fontsize=17,
                    color=pct_col(v, bad), ha='center', va='center_baseline', zorder=3)

    # ---------- footer ----------
    yf = Y(fig_h - H_BOT + 0.26)
    for lbl, col, x in [('TOP 4', GREEN, .045), ('TOP 6', CYAN, .135),
                        ('RELEGATION', PINK, .225)]:
        ax.add_patch(Rectangle((x, yf - 0.006), 0.0045, 0.013,
                               color=col, zorder=3))
        ax.text(x + 0.011, yf, lbl, fontproperties=BC_M, fontsize=11.5,
                color=WHITE, alpha=.7, va='center_baseline', zorder=3)
    ax.text(.965, yf, f'{n_sims:,} MONTE CARLO SIMULATIONS',
            fontproperties=BC_B, fontsize=12, color=PINK, ha='right',
            va='center_baseline', zorder=3)
    ax.text(.045, Y(fig_h - 0.22),
            'Dixon-Coles xG model  ·  data: Understat + football-data.co.uk  ·  '
            'range = middle 80% of simulated seasons',
            fontproperties=BC_M, fontsize=10.5, color=WHITE, alpha=.42,
            va='center_baseline', zorder=3)

    fig.savefig(out_path, facecolor=PURPLE_D)
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    import sys
    pred = pd.read_csv(sys.argv[1] if len(sys.argv) > 1
                       else '/home/claude/pl/data/processed/pred_2627.csv')
    gw = sys.argv[2] if len(sys.argv) > 2 else 'PRE-SEASON  ·  17 AUG 2026'
    print(render(pred, '/home/claude/pl/table.png', gw))


BRAND = 'MY PREDICTIONS'      # change this line to rename the graphic


def backdrop(ax, glow_x=.30, glow_y=.02):
    """Shared background: gradient, corner glow, stripes, vignette, chevrons."""
    g = np.linspace(0, 1, 600).reshape(-1, 1)
    ax.imshow(g, extent=[0, 1, 0, 1], aspect='auto', zorder=0,
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                  'pl', ['#1A0020', PURPLE, '#4E0055']))

    ny, nx = 520, 420
    Xg, Yg = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    d = np.sqrt(((Xg - glow_x) / .80) ** 2 + ((Yg - glow_y) / .46) ** 2)
    rgba = np.zeros((ny, nx, 4))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = .48, .05, .55
    rgba[..., 3] = np.clip(1 - d, 0, 1) ** 2 * .62
    ax.imshow(rgba, extent=[0, 1, 0, 1], aspect='auto', zorder=0,
              interpolation='bilinear')

    for x in np.arange(-.02, 1.02, .1042):          # faint vertical bands
        ax.add_patch(Rectangle((x, 0), .052, 1, color=WHITE, alpha=.013,
                               lw=0, zorder=0))

    for i in range(7):                               # corner chevrons
        x = .70 + i * .052
        ax.plot([x, x + .105], [1.005, .90], lw=2.2, solid_capstyle='round',
                color=PINK if i % 2 else CYAN, alpha=.10, zorder=0)

    v = np.linspace(0, 1, 300).reshape(-1, 1)
    vg = np.zeros((300, 2, 4)); vg[..., 3] = (v ** 3) * .55
    ax.imshow(vg, extent=[0, 1, 0, 1], aspect='auto', zorder=0,
              interpolation='bilinear')


def render_mobile(pred, out_path, gw_label, n_sims=20000, brand=None):
    """Portrait 4:5 graphic for phone / LinkedIn feed. No range column."""
    d = pred.reset_index(drop=True)
    T = len(d)
    W, H = 7.2, 9.0                     # 1080 x 1350 at dpi 150
    brand = brand or BRAND
    H_TOP, H_BOT = 2.26, 0.74
    ROW = (H - H_TOP - H_BOT) / T

    fig = plt.figure(figsize=(W, H), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    backdrop(ax)

    def Y(px):
        return 1 - px / H

    ax.text(.06, Y(0.58), '2026-27', fontproperties=ANTON, fontsize=43,
            color=WHITE, va='center', zorder=3)
    ax.text(.06, Y(1.02), 'PREMIER LEAGUE', fontproperties=ANTON, fontsize=26,
            color=WHITE, va='center_baseline', zorder=3)
    ax.text(.06, Y(1.42), brand, fontproperties=ANTON, fontsize=26,
            color=PINK, va='center_baseline', zorder=3)
    ax.text(.945, Y(0.58), gw_label.upper(), fontproperties=BC_B, fontsize=15.5,
            color=CYAN, ha='right', va='center_baseline', zorder=3)

    C = {'x': .620, 't': .762, 'r': .905}
    ax.add_patch(Rectangle((0, Y(H_TOP - .19)), 1, .075 / H, color=PINK, zorder=2))
    yh = Y(H_TOP - .42)
    for lbl, x in [('xPTS', C['x']), ('TITLE', C['t']), ('RELEG', C['r'])]:
        ax.text(x, yh, lbl, fontproperties=BC_B, fontsize=14, color=WHITE,
                ha='center', va='center', alpha=.7, zorder=3)

    for i, r in d.iterrows():
        yc = Y(H_TOP + ROW * i + ROW / 2)
        h = (ROW - .055) / H
        zone = GREEN if i < 4 else (CYAN if i < 6 else (PINK if i >= T - 3 else None))
        ax.add_patch(FancyBboxPatch((.045, yc - h / 2), .915, h,
                                    boxstyle='round,pad=0,rounding_size=.012',
                                    fc='#FFFFFF', ec='none', alpha=.075, zorder=2))
        if zone:
            ax.add_patch(Rectangle((.045, yc - h / 2), .008, h, color=zone, zorder=3))
        ax.text(.088, yc, str(i + 1), fontproperties=ANTON, fontsize=17,
                color=WHITE, alpha=.5, ha='center', va='center_baseline', zorder=3)
        ax.add_patch(Rectangle((.116, yc - h / 2), .011, h,
                               color=CLUB.get(r.team, '#888'), zorder=3))
        ax.text(.148, yc, SHORT.get(r.team, r.team.upper()), fontproperties=BC_B,
                fontsize=20, color=WHITE, va='center_baseline', zorder=3)
        ax.text(C['x'], yc, f'{r.xPts:.1f}', fontproperties=BC_B, fontsize=20,
                color=WHITE, ha='center', va='center_baseline', zorder=3)
        for val, x, bad in [(r.title, C['t'], False), (r.releg, C['r'], True)]:
            v = 100 * val
            s = '—' if v < .05 else (f'{v:.1f}%' if v < 99.9 else '>99%')
            col = ((PINK if v >= 40 else '#FF8FBE' if v >= 15 else WHITE if v >= 3
                    else '#9C7FA6') if bad else
                   (GREEN if v >= 25 else CYAN if v >= 8 else WHITE if v >= 1
                    else '#9C7FA6'))
            ax.text(x, yc, s, fontproperties=BC_B, fontsize=19, color=col,
                    ha='center', va='center_baseline', zorder=3)

    yf = Y(H - H_BOT + .30)
    for lbl, col, x in [('TOP 4', GREEN, .06), ('TOP 6', CYAN, .215),
                        ('RELEGATION', PINK, .37)]:
        ax.add_patch(Rectangle((x, yf - .0055), .006, .011, color=col, zorder=3))
        ax.text(x + .014, yf, lbl, fontproperties=BC_M, fontsize=12.5,
                color=WHITE, alpha=.72, va='center_baseline', zorder=3)
    ax.text(.945, yf, f'{n_sims:,} SIMULATIONS', fontproperties=BC_B,
            fontsize=13, color=PINK, ha='right', va='center_baseline', zorder=3)
    ax.text(.06, Y(H - .28), 'Dixon-Coles xG model  ·  Understat + football-data.co.uk',
            fontproperties=BC_M, fontsize=11, color=WHITE, alpha=.42,
            va='center_baseline', zorder=3)
    fig.savefig(out_path, facecolor=PURPLE_D)
    plt.close(fig)
    return out_path
