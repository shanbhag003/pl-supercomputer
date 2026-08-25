# How Everything Works — a file-by-file walkthrough

This explains every file in the project in plain English, with examples. No
statistics background assumed.

Read the first two sections and you'll understand the whole thing. The rest is
reference.

---

## Part 1 — The idea in one page

You want to know each club's chance of winning the league.

You cannot calculate that directly. So instead:

1. Work out how good every club is at scoring and at preventing goals
2. Use those numbers to work out the likely score of each remaining match
3. Play the whole season out, letting each match land differently
4. Do that 20,000 times
5. Count how often each club came first

If a club wins the league in 6,560 of 20,000 simulated seasons, that's a
**32.8% title chance**. That's all a "supercomputer" prediction is — counting.

Everything else in this project exists to make step 1 as accurate as possible:
using chance quality rather than goals, accounting for who was sold, who is
injured, who is now the manager.

---

## Part 2 — What happens when the model runs

One command runs everything: `python src/update.py`. It does this:

```
 1. Download the latest results and expected goals        (update.py)
 2. Work out whether a gameweek has finished              (gate.py)
 3. Build the current league table from matches played    (update.py)
 4. Rate every club from twelve seasons of history        (ratings.py)
 5. Adjust for transfers, injuries and suspensions        (squad_live.py)
 6. Adjust for managerial changes                         (manager_model.py)
 7. Make 80 slightly different versions of those ratings  (simulate.py)
 8. Simulate every remaining match 20,000 times           (simulate.py)
 9. Score last week's predictions against reality         (update.py)
10. Draw the graphic                                      (render.py)
11. Send the email                                        (mailer.py)
```

Takes about 40 seconds.

**There are two kinds of run.** A **publish** happens three hours after a
gameweek finishes: it does everything above and emails you. A **refresh** runs
Friday and Tuesday at 17:00 UTC: it redoes steps 5 to 10 with the latest squad
news, updates the website, and sends nothing.

The refresh exists because of a timing problem. Team news comes out on Thursday
and Friday, when managers hold press conferences. But a gameweek publishes on
Tuesday. Without a refresh, the injury data would be a median of **6.7 days old**
by the time the next round kicked off — and after an international break, nearly
three weeks old. The refresh cuts that to under a day.

---

## Part 3 — The files that run every week

These seven files do the actual work. Everything else is research or setup.

### `update.py` — the conductor

The only file the scheduler runs. It calls everything else in order.

**What it does, step by step:**

**Downloads fresh data.** Fetches results from football-data.co.uk and expected
goals from Understat.

> **A real bug this caught.** When a season hasn't started, football-data.co.uk
> doesn't return "file not found". It returns a *webpage* suggesting similar
> filenames. Early on, the model saved that webpage as if it were results data
> and crashed. It now checks that a download actually looks like a spreadsheet
> before saving it, and checks the columns are right before using it.

**Builds the current table.** Adds up points, goals for and goals against from
every match played so far. In August this is all zeros. By April, Arsenal might
be on 70 points — and the model starts from 70, not from scratch.

**Sets the uncertainty level.** Early in a season we know little about how clubs
have changed over the summer, so predictions get wide error bars. As matches
accumulate, uncertainty shrinks:

```
uncertainty = max(0.04, 0.16 × (1 − games_played / 12))
```

By roughly gameweek 12 it hits the floor and this season's actual results
dominate.

**Applies the adjustment layers**, simulates, scores itself, renders, emails.

**Fails safely.** If Understat is down, or the FPL API changes, or Wikipedia is
unreachable, that layer switches off, logs the reason and the model still runs.
It never fails silently.

---

### `gate.py` — deciding when to publish

The problem: GitHub can only run things on a fixed schedule, but gameweeks don't
end on a fixed schedule.

**Example.** Gameweek 1 ends Monday 24 August at 19:00. A "run every Tuesday at
6am" schedule would fire eleven hours later — before the twelve-hour settling
window. And five gameweeks this season end on a *Wednesday*, which a weekly
schedule handles badly.

**The solution.** The job wakes every two hours and asks three questions:

1. Has every match in this round been played?
2. Has it been three hours since the last one finished?
3. Have we already published this gameweek?

If the answer isn't yes, no, no — it exits in about two seconds and costs
nothing.

**Postponements.** If a match is still unplayed five days after its scheduled
kickoff, it's treated as postponed so the round can settle without it. The email
mentions when this happens.

---

### `ratings.py` — how good is each club?

Gives every club two numbers: an **attack rating** and a **defence rating**.

**Why not just use goals?** Because goals are noisy. A club can play well and
lose 1-0 to a deflection. Over 38 matches, a club can finish six points from
where their performances deserved.

So the model mostly uses **expected goals** — a measure of chance quality. A
tap-in from two yards counts as a much better chance than a hopeful shot from
30 yards. The blend is 70% expected goals, 30% actual goals.

**Recent matches count more.** A match from three years ago tells you less about
a club today than one from April. Each match is weighted by age, with a half-life
of 154 days — so a match six months old counts about half as much as last week's.

That 154 days wasn't chosen by feel. It was picked by testing values across six
seasons and keeping whichever predicted best.

**Predicting a single match.** Once every club has ratings:

```
Arsenal at home to Everton
  Arsenal expected goals  = 2.08
  Everton expected goals  = 0.71
  → Arsenal win 68%, draw 21%, Everton win 10%
```

There's one small correction, called Dixon-Coles, because real football produces
slightly more 0-0s and 1-1s than pure maths predicts.

---

### `simulate.py` — playing the season out

**Two kinds of randomness, and you need both.**

**Match randomness.** For each fixture, sample a scoreline from its probability
grid. Don't take the most likely score — sample it. Arsenal vs Everton might come
out 2-0, then 1-1, then 3-1 across different simulations.

**Rating uncertainty.** This is the part most people skip. If you use one fixed
set of ratings for all 20,000 simulations, you're pretending you know exactly how
good every club is. You don't.

So the model fits **80 slightly different versions** of the ratings, each from a
resampled slice of history, and each simulation uses one of them.

> **Why it matters.** Skip this and the title probabilities come out far too
> confident — something like "Arsenal 62%" when the honest number is 33%.

**Then:** add up points, apply real Premier League tiebreaks (points, then goal
difference, then goals scored), and record where every club finished. Repeat
20,000 times.

---

### `squad_live.py` — who is actually available?

Ratings are built from *past* football. But Rodri has left, and a summer squad
is not last season's squad.

**Step 1 — who's at each club now?** Read the official Fantasy Premier League
API. It lists every player, their current club, and their status: available,
injured, doubtful, suspended, or gone.

**Step 2 — match them to their value.** Player values come from `rapm.py`. The
join is by name, which is fiddlier than it sounds.

> **A real bug.** Ødegaard, Fábio Vieira and others failed to match, because the
> standard way of stripping accents doesn't handle `ø` — it deleted the letter
> instead of converting it, turning "Ødegaard" into "degaard". A player who
> fails to match silently gets a value of zero, quietly weakening their club.
> Fixed with an explicit character map. The model now logs any unmatched player
> who has Premier League minutes.

**Step 3 — how many minutes will each player get?** Every club has a fixed
budget: 38 matches × 11 players × 90 minutes. Returning players keep roughly
last season's share. Whatever departures freed up is given to new signings, in
proportion to their FPL price.

So a club that sells a key player and buys nobody loses that value. One that
reinvests gets some of it back.

**Step 4 — who covers for an absent player.**

Every club has a fixed budget of 38 matches x 11 players x 90 minutes. When
somebody is unavailable, those minutes have to go to somebody else — and getting
that wrong quietly destroys the whole layer.

> **A real bug, found in review.** The first version handed the freed minutes to
> everyone in proportion to how much they already played. So a missing striker
> was covered by the goalkeeper, who ended up allocated **90.4 minutes a match** —
> more than a match contains. And because the club rating is a minutes-weighted
> *average*, replacing a star with other stars barely changed it. Injuring a key
> player moved the club rating by 0.002, about a tenth of what it should have.
>
> Two fixes. Nobody is allocated more than 90 minutes a match. And minutes go to
> available players **in the same position**, weighted by how much room they have
> left — the backup striker, not the centre-back.
>
> The effect more than doubled, and something new appeared: **squad depth**.
> Injuring Arsenal's forward barely moves them, because two capable deputies
> exist. Injuring Manchester City's moves them nearly three times as much,
> because one player absorbs it. The old version blurred both into the same
> answer.

**Step 5 — injuries and suspensions, scaled to time.**

> **A real bug.** Originally an injured or suspended player was removed for the
> *whole remaining season*. A one-match ban wiped a player out of 38 fixtures.
>
> Now: injuries are assumed to cost about six matches, suspensions two, doubts
> one. Divided by the fixtures remaining. A two-match ban with 30 to play costs
> about 7% of that player, not 100%. The same injury also matters more in April
> than in September, which is correct.

**Step 5 — compare.** The signal isn't "how good is this squad" — the club rating
already knows that. It's **how much the squad changed**. For 2026/27, Manchester
City come out at −0.125, the largest fall of any established club.

---

### `render.py` — the graphics

Draws the portrait table for the phone and a wider landscape version.

Worth knowing: text is centred by **measuring** it. Fonts reserve space for
letters like "y" and "g" that hang below the line. Words in capitals never use
that space, so centring by the usual method leaves everything sitting slightly
low. The code draws each label, measures its real pixel box, then nudges it.

No club crests or league logos — those are trademarked. Club identity is a colour
bar.

---

### `mailer.py` — the email

Builds two versions of the same email: a styled HTML one and a plain-text
fallback.

Deliberately written for someone with no background. "Sharpness score, lower is
better" rather than "RPS". "4.2 expected points up" rather than "d_xPts +4.2".

The accuracy section only grades predictions made **before** a match was played.
The model saves next gameweek's forecasts each run so the following run can mark
them.

---

## Part 4 — Files that built the model

These ran once to produce the data files. They don't run weekly, but they're kept
so every number can be checked or rebuilt.

### `rapm.py` — what is each player worth?

The hardest question in the project. Answered with a method borrowed from
basketball: **regularised adjusted plus-minus**.

**The naive approach and why it fails.** Compare how a club does with a player
versus without him.

```
Rodri at Manchester City, 2019–2025
  With Rodri (166 matches):     0.82 expected goals conceded per match
  Without Rodri (100 matches):  1.08
```

City concede 24% less with him. Real, and invisible in any attacking statistic —
his non-penalty expected goals for 2025/26 was 1.19, essentially nothing.

Now the same test on Haaland: **he makes City worse**. Obviously false. His
"without" matches are mostly from before he joined, when City were at their peak.
The comparison measures *eras*, not Haaland.

**The fix.** Solve for every player at once. One row per club per match, with the
response being expected goals scored, and every player's share of minutes as an
input. Teammates and opponents are all in the same equation, so each player's
number is already adjusted for who he played with and against. A heavy penalty
shrinks thin-sample players toward average, so five appearances buys a number
near zero rather than nonsense.

**Honest about the limit.** Football has far less lineup rotation than
basketball. The same eleven start together week after week, which makes
separating them genuinely hard. Goalkeepers who play every minute for a strong
club absorb that club's quality. This is documented in the README's limitations.

### `eu_rapm.py` — the same, for four more leagues

Fits the same model in La Liga, Serie A, the Bundesliga and Ligue 1, then works
out the conversion rate using players who moved to England.

**The result is a finding in itself.** Correlations of 0.10 to 0.26 — foreign
form explains under 7% of Premier League output. So the conversion shrinks
foreign values almost to league average, which is the honest answer.

### `managers.py` and `manager_model.py`

`managers.py` scrapes 519 manager spells from Wikipedia with exact dates.
Sanity check: it returns Ferguson's tenure as 9,704 days, which matches published
figures.

`manager_model.py` fits a value for each manager, alongside player values, with
club fixed effects — so a manager's number comes only from what changed *within*
a club when the manager changed.

**Why Maresca gets a number and some managers don't.** Manager values need a
comparison. Maresca managed Chelsea for 549 days, so we can see how his teams
performed relative to their squad. Marco Rose and Xabi Alonso have never worked
in England, so they get **no adjustment at all** — which honestly means "we don't
know."

### `market.py` — reading the bookmakers

Turns bookmaker odds into a view of club strength, by working backwards from the
prices on opening-day fixtures.

Odds of 2.50 imply a 40% chance. Add up the three prices for a match and they
come to more than 100% — that difference is the bookmaker's margin, which this
strips out before use.

This layer was **tested and rejected**. It's kept because the rejection is part
of the record.

### `forecast_2627.py` and `render_arch.py` — one-off tools

`forecast_2627.py` produced the very first pre-season forecast, before the
weekly job existed. Superseded by `update.py`, kept for reference.

`render_arch.py` draws the architecture diagram — the "how it works in seven
steps" graphic. Run it by hand whenever the model changes.

### `cup_fixtures.py` — when did clubs play outside the league?

Scrapes every FA Cup and EFL Cup tie from Wikipedia, 2014 to 2026, to answer one
question: does playing midweek hurt a club at the weekend?

The scraping is fiddly. Wikipedia renders each cup tie as its own little table —
144 of them on a single FA Cup page — with the date in the first cell and the two
clubs either side of the score. The code walks every table on the page and keeps
the ones that look like a fixture.

Result: **1,488 cup ties involving Premier League clubs**, a median of five per
club per season.

**Why domestic cups and not just Europe?** Because only six or seven clubs play
in Europe, and they're the good ones. Any comparison would really be measuring
"are you a big club". Every club plays the domestic cups, so congestion cases
show up right across the table, including at the bottom.

**The finding.** For every league match, work out days since that club's last
match of any kind. Then compare each club to its own season average:

| | ≤3 days rest | ≥7 days rest | Difference |
|---|---|---|---|
| xG created | +0.022 | −0.004 | +0.025 |
| xG conceded | −0.009 | +0.007 | −0.015 |
| Points won | +0.061 | +0.003 | +0.058 |

Nothing. And every sign points the *wrong* way — congested clubs created a bit
more, conceded a bit less, won a few more points.

**But this test was flawed**, and the flaw is worth understanding. The League Cup
is exactly where a big club rests players. So it measures *rotation*, not
congestion. Which led to the next file.

### `uefa_fixtures.py` — the test that actually mattered

The objection: clubs rotate for domestic cups, but nobody rotates for a Champions
League last-16 second leg. That's the best XI, maximum intensity, often away
travel, three days before a league game. If congestion exists anywhere, it's
there.

**Getting the data was harder.** The rendered Wikipedia pages use bracket layouts
that don't parse. But the underlying wiki source stores every match as a
`Football box` template with a machine-readable date and both team names. Reading
that instead gives **735 UEFA matches involving English clubs**, of which 328 are
knockout ties.

**And the objection was right:**

| Situation | xG conceded vs own average | p |
|---|---|---|
| After a UEFA **knockout** tie | **+0.094** (worse) | 0.067 |
| After an **away** knockout tie | **+0.129** (worse) | 0.067 |
| After a group / league-phase tie | −0.087 (better) | 0.024 |

Knockout ties leave a defensive mark. Group matches don't — clubs come out
slightly *better*, which is the rotation signature showing up exactly where you'd
expect it.

### `backtest_congestion.py` — and why it still didn't ship

A promising residual is not a usable model. The adjustment was added to the
ratings and tested walk-forward across seven seasons — refit weekly, never seeing
a result before predicting it.

| Period | Effect of the adjustment |
|---|---|
| 2019–2022 | +0.0038 — **worse** |
| 2023–2025 | −0.0050 — **better** |
| All seven seasons | +0.00002 — **nothing** |

The two halves of the data disagree. Four seasons prefer no adjustment, three
prefer the largest one. That's a coin flip.

The effect was real in the sample it was measured on and doesn't generalise.
Shipping it would have meant fitting three seasons of noise to 150 matches.

**So congestion isn't modelled** — not because nobody looked, but because it was
measured, found, tested, and failed. Both datasets stay in the repository. With
more seasons of knockout data it's worth re-running.

### `ingest.py`, `pull_players.py`, `promoted.py`

`ingest.py` builds `matches.parquet` — 4,560 matches with expected goals.

`pull_players.py` downloads per-match minutes for every player. It runs eight
downloads at once: 4,398 players in 141 seconds instead of over an hour.

`promoted.py` produced one of the project's better findings:

> Across 33 promoted clubs since 2015, the correlation between Championship
> points and first-season Premier League rating is **+0.043** — essentially zero.
> Norwich went up with 97 points and were the worst promoted side in the sample.
> Wolves went up with 99 and were the best.
>
> So all promoted clubs get the same starting estimate, with a deliberately wide
> error bar. Coventry winning promotion by 11 points earns them nothing.

---

## Part 5 — Files that tested the model

Nothing ships without passing a backtest: refit the model as it would have been
at the time, predict, compare to what really happened, across seven seasons.

| File | What it tested | Verdict |
|---|---|---|
| `backtest.py` | Core match predictions vs bookmakers | Within 1.1% of the market |
| `tune.py` | Which settings predict best | Set the 154-day half-life and the 70/30 blend |
| `preseason_bt.py` | Whole-season forecasts | Also measured how much clubs change over a summer |
| `backtest_squad.py` | Does squad accounting help? | **Yes — shipped** |
| `backtest_manager.py` | Do manager effects help? | **Only for within-league moves — shipped that version** |
| `backtest_market.py` | Does blending bookmaker odds help? | **No — rejected** |
| `cup_fixtures.py` | Does domestic cup congestion matter? | **No effect — rotation absorbs it** |
| `backtest_congestion.py` | Does European knockout congestion matter? | **Suggestive, but fails out-of-sample — not built** |

### The most instructive failure

The first manager model penalised Bournemouth for hiring Iraola and Liverpool for
hiring Slot — who then won the league.

The cause: neither had ever managed in England, so both were entered as "average
manager". The model wasn't saying *these managers are worse*. It was saying
*their predecessors were above average and we know nothing about the
replacements*. Two different statements, and the code couldn't tell them apart.

Restricting the layer to changes where **both** managers have a Premier League
record flips the result from harmful to helpful. That version ships, at quarter
strength because the evidence is thin.

---

## Part 5b — What a code review turned up

The whole codebase was read line by line. The interesting finds:

**A bad response could have wiped the season.** The Understat cache was
overwritten whenever the request returned any success code. Understat returns an
empty-but-valid payload when it has nothing — which happens pre-season, and could
happen mid-season during a fault. That would have replaced twenty gameweeks of
results with an empty file. The gate would have blocked the email, but a Friday
refresh skips the gate, so the website would have published a table with every
club on zero points.

Fixed by refusing any payload containing fewer results than the cache already
holds. Results only accumulate within a season, so a shrinking count is always a
fault, never an update.

**Postponed matches were escaping the accuracy scoring.** Predictions were
matched to results on date, club and opponent. A postponed game is replayed on a
different date, so it never matched and quietly vanished from the record —
losing precisely the matches most likely to have surprised the model. Now
matched on the fixture rather than the date.

**Everything else.** Dependency versions pinned, so an upstream release cannot
break a Tuesday morning. `data.json` written to a temporary file and moved into
place, so a crash cannot leave the site serving half a file. All twenty-three
exception handlers now log the exception type, so a deliberate skip looks
different from a real failure. Unrecognised player statuses treated as a doubt
rather than assumed fit.

**One thing checked and found sound.** League positions are decided by packing
points, goal difference and goals into a single number for sorting. That looked
fragile enough to be worth testing: 100,000 ranked rows, zero cases where the
order broke the real tiebreak rules. It stays, now with a comment explaining why
the multipliers cannot be compressed.

## Part 6 — The data files

| File | What it holds |
|---|---|
| `matches.parquet` | 4,560 matches, 2014–2026, with expected goals |
| `rapm_live.pkl` | A value for every player with Premier League history |
| `eu_values.pkl` | Values for 4,430 players from four other leagues |
| `eu_conversion.pkl` | The conversion rate between each league and England |
| `manager_fit.pkl` | A value for each manager |
| `managers.csv` | 519 manager spells with dates |
| `last_season_players.json` | Minutes played by every player in 2025/26 |
| `eu_name_index.json` | Name lookup for foreign players |
| `fixtures_2627.csv` | All 380 fixtures with kickoff times |
| `cup_fixtures.csv` | 2,833 domestic cup ties, 2014–2026 |
| `uefa_fixtures.csv` | 735 UEFA matches involving English clubs |

Two of these are precomputed summaries. Originally the model read 2.8 MB of raw
data files; those were condensed to 415 KB, which is why the upload is small.

---

## Part 7 — `weekly.yml`, the scheduler

The instruction file GitHub reads. In plain English:

```
Wake up every two hours, and also whenever I press the button.
Get a fresh computer. Install Python and the libraries.
Run src/update.py, with my email password kept secret.
If anything changed, save it back to the repository.
```

Three details worth knowing:

**It polls, it doesn't schedule.** Every two hours it wakes, `gate.py` checks
whether a gameweek has settled, and it exits in seconds if not. That's how an
event-based trigger is faked on a fixed-schedule system.

**Secrets stay secret.** The Gmail password is stored in GitHub's secrets, not in
any file. It's passed to the script as an environment variable and never appears
in a log.

**It commits its own output.** Every gameweek's forecast is saved back to the
repository, so there's a permanent, timestamped record of what was predicted
before each round — which is what makes the self-scoring trustworthy.

---

## Part 8 — Three things people ask

**"Why does the expected points total differ from the current table?"**
It doesn't, once the season starts. Points already won are carried forward. Give
Arsenal 70 points with five matches left and the model returns a final total
averaging 79.8, never below 70.

**"Why does Brentford have a better title chance than Leeds with fewer expected
points?"**
Because winning the league is a tail event, not an average one. Brentford's range
of outcomes is wider — 37 to 69 points, against Leeds' 39 to 67. They're more
likely to finish very high *and* more likely to go down. From fourth place
downward, Leeds overtake them. The crossover happens right at the top.

**"Doesn't a midweek game hurt them at the weekend?"**
It depends which midweek game, and the answer is more interesting than a flat no.

Domestic cups: no effect at all. Clubs rotate for those, so there's nothing to
detect.

European knockout ties: clubs do concede about 0.09 more xG than their own
average afterwards, rising to 0.13 after an away tie. Nobody rests players for a
Champions League last 16.

But when that adjustment was actually backtested, it helped in 2023–25 and hurt
in 2019–22, netting out to nothing across seven seasons. 150 affected matches
isn't enough to tell a real effect from a lucky one, so it isn't in the model.

**"Why isn't Guardiola leaving a bigger deal?"**
Because it can't be measured cleanly. Guardiola managed City for nearly the whole
period the data covers, so his contribution is hard to separate from City's own
quality. Rather than invent a number, the model uses Maresca's own measured
record instead, which is a smaller and more defensible adjustment. The README
lists this as a limitation.
