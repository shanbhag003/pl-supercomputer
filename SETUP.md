# Setup

Everything here can be done from a browser. No local machine, no terminal, no
git client. The project is designed that way on purpose: it runs on GitHub
Actions, edits happen in GitHub's web editor, and the only "deploy" step is a
commit.

If you do have a local environment, there's a short section for that at the end.

---

## 1. Fork the repository

Click **Fork** at the top right. Everything below happens in your fork.

Nothing in the repository is secret — no API keys are required by any data
source. The only credentials are for the optional email.

---

## 2. Turn on GitHub Pages

1. **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **main**, folder: **/docs**
4. **Save**

After a minute your site is live at
`https://<your-username>.github.io/<repo-name>/`.

The site is a single hand-written `docs/index.html` that fetches
`docs/data.json`. There is no build step, so nothing needs compiling and there
is no node_modules to install.

---

## 3. Turn on Actions

Forks start with workflows disabled.

1. Open the **Actions** tab
2. Click **I understand my workflows, go ahead and enable them**
3. In the left sidebar, select **PL Supercomputer**

You should now see a **Run workflow** button.

> **A GitHub quirk worth knowing:** scheduled workflows are automatically
> disabled after 60 days without any repository activity. This project commits
> its own outputs on most runs, so in practice it keeps itself alive — but if you
> ever come back to a silent repo, that's the first thing to check.

---

## 4. Email (optional)

Skip this and everything still works; the run just logs that it couldn't send.

Add three repository secrets under **Settings** → **Secrets and variables** →
**Actions** → **New repository secret**:

| Secret | Value |
|---|---|
| `GMAIL_USER` | the Gmail address to send from |
| `GMAIL_APP_PASSWORD` | a Google **App Password**, not your account password |
| `MAIL_TO` | where to send it; comma-separate for several |

To create an App Password you need 2-Step Verification enabled on the Google
account, then visit Google Account → Security → App passwords. A normal account
password will be rejected by Gmail's SMTP.

---

## 5. Analytics (optional)

The site ships with Google Analytics 4 and Microsoft Clarity wired in but
switched off — the ID placeholders are near the top of `docs/index.html`:

```js
window.PL_GA4_ID     = 'G-XXXXXXXXXX';   // GA4 Measurement ID
window.PL_CLARITY_ID = 'xxxxxxxxxx';     // Clarity Project ID
```

Leave either as an empty string to disable that provider. Nothing fires on
`localhost` or over `file://`, so local testing never pollutes the reports.

**GA4:** create a property, add a **Web** data stream for your Pages URL, and
copy the Measurement ID. Then, and this is the step people skip: GA4 collects
custom parameters but will not display them in any report until you declare
them. **Admin** → **Data display** → **Custom definitions** → create one
dimension per parameter, scope **Event**:

`tab`, `section_name`, `gameweek`, `gw_type`, `direction`, `accuracy_band`,
`n_settled`, `results_right`, `season_phase`, `run_type`, `percent_scrolled`,
`link_domain`

There is no backfill — data accrues only from the moment each dimension exists.
Two other settings worth changing while you're there: **Data retention** defaults
to 2 months, so raise it to 14; and define your own IP as internal traffic if you
don't want your own testing counted.

**Clarity:** create a project, copy the Project ID. Set masking to **Relaxed**
under Settings → Privacy — the site shows nothing but public football data, and
masked recordings are far less useful. Then link Clarity to GA4 under
Settings → Setup, which lets you jump from an aggregate finding straight to the
sessions behind it.

**What gets tracked.** Events: `forecast_loaded`, `forecast_load_failed`,
`tab_switch`, `gameweek_select`, `section_open`, `scroll_depth`,
`timezone_toggle`, `outbound_click`. Session tags: gameweek, season phase, run
type, forecast leader, whether fixtures are available, and a bucketed accuracy
band so you can filter recordings by how well the model was doing that week.

`forecast_load_failed` is worth an alert. It fires when the site cannot read
`data.json`, which is your earliest warning that a run broke — the reader notices
before you do.

---

## 6. Run it

**Actions** → **PL Supercomputer** → **Run workflow**. Two toggles:

| Toggle | Use it when |
|---|---|
| *(neither ticked)* | A gameweek has finished and you want to publish. Exits in seconds if the gate is shut. |
| **Rebuild the site with current squad data, no email** | You changed code or want fresh team news. Skips the gate, rebuilds everything, sends no email. **This is the one you usually want.** |
| **Publish even if no gameweek has settled** | Rarely. It bypasses the settle check and picks the gameweek differently. |

A full run takes 60–90 seconds. A gate check that decides there's nothing to do
takes about two.

### When it runs on its own

Three schedules, all in `.github/workflows/weekly.yml`:

- **Every two hours** (`17 */2 * * *`) — a cheap poll. Publishes only if a
  gameweek has settled, otherwise exits.
- **Tuesdays and Fridays at 17:00 UTC** — a refresh. Re-reads squads, injuries
  and suspensions, rebuilds the site, sends no email. This exists because team
  news lands Thursday and Friday while a publish happens Tuesday.

A gameweek publishes once every fixture in it has been played and **three hours**
have passed since the last one finished. That window is set by `SETTLE_HOURS` in
`src/gate.py`. Three is deliberate: the binding constraint is usually the results
feed rather than the timer, and shortening it further buys under an hour because
the poll only runs every two hours.

GitHub's scheduled runs are delayed under load, sometimes by 20–30 minutes, and
are occasionally skipped entirely. That is normal and the design absorbs it.

---

## 7. Editing without a local machine

Three ways, easiest first.

**github.dev** — press `.` on any repository page, or change `.com` to `.dev` in
the URL. A full VS Code opens in the browser with the whole repo. Edit, then
commit from the Source Control panel in the left rail.

**The web editor** — click any file, then the pencil icon. Fine for a one-line
change; `Ctrl+F` works for finding an anchor.

**Upload files** — navigate *into* the target folder first, then **Add file** →
**Upload files** and drag the replacement in. Same filename means GitHub replaces
it. Watch the folder: `index.html` belongs in `docs/`, Python in `src/`,
markdown in the root.

If a change breaks the page, the safety net is built in: **Commits** → find the
previous one → **Revert**. No git needed.

---

## 8. Making it your own

| To change | Edit |
|---|---|
| The settle window | `SETTLE_HOURS` in `src/gate.py` |
| How often it polls | the cron lines in `.github/workflows/weekly.yml` |
| Number of simulations | `N` in `src/update.py` (default 20,000) |
| Bootstrap ensemble size | `B` in `src/update.py` (default 80) |
| Draw threshold | `CLOSE_CALL` in `src/update.py` (default 0.04) |
| How much squad data counts | `SQUAD_W` in `src/update.py` (default 0.5) |
| How much manager changes count | `MGR_W` in `src/update.py` (default 0.25) |
| Colours, layout, copy | `docs/index.html` |

Every one of those defaults was chosen by backtest, not by taste. If you change
one, the honest thing is to re-run the relevant `src/backtest*.py` and see
whether it actually helped. Several plausible ideas made the forecast *worse* —
see the rejected-layers section of the README.

---

## 9. A different league or season

The model isn't hard-coded to the Premier League, but switching isn't a one-line
change either. You would need to:

1. Point `SEASON` and `FD_CODE` in `src/update.py` at the new competition
2. Supply a fixture list at `data/raw/fixtures_<code>.csv`
3. Rebuild `data/processed/matches.parquet` via `src/ingest.py`
4. Re-derive player values with `src/rapm.py`
5. Update the club colour and short-name maps in `docs/index.html`

The Dixon-Coles and simulation code itself is competition-agnostic. Everything
around it assumes 20 clubs and 38 rounds, so a league with a different shape
needs those constants revisited.

---

## Running locally

```bash
pip install -r requirements.txt

# full run: refit, simulate, render, email
FORCE_RUN=1 python src/update.py

# rebuild the site only, no email
REFRESH_ONLY=1 python src/update.py
```

About 40 seconds on one CPU core. Dependency versions are pinned, so an upstream
release can't break a scheduled run.

---

## Troubleshooting

**The site shows an error box instead of a table.** `data.json` couldn't be
fetched or parsed. Check the last Actions run.

**A tab is greyed out.** *Actual* is disabled before the season starts.
*Fixtures* is disabled until a run has written a `matches` array into
`data.json`. Both are correct, not broken — trigger a refresh run.

**The site hasn't changed after a commit.** GitHub Pages caches `index.html` for
around ten minutes. Hard-refresh with `Ctrl+Shift+R`.

**A run says "nothing to publish".** The gate is shut, which is the normal state
most of the time. If you want output now, use the refresh toggle.

**Predicted scores never exceed 2-2.** Expected. See the FAQ in
[HOW-IT-WORKS.md](HOW-IT-WORKS.md) — it's a property of predicting the most
likely score, not a bug.

**Analytics show nothing in GA4 but Clarity has data.** GA4's standard reports
lag 24–48 hours; Realtime and DebugView are immediate. Also check whether an
internal-traffic filter is excluding your own IP — Clarity has no equivalent
filter, which is exactly why the two disagree.
