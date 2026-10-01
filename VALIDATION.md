# Validation

Every number in this file came from a walk-forward test: the model is refitted
using only data available at the time, and never sees a result before predicting
it. Where a test used a proxy rather than the model's own output, it says so.

The point of keeping this file is not to show the model is good. It is to make it
possible to check, and to record the things that failed — including one that
failed *after* it shipped.

---

## The rule

Nothing goes into the live model on the strength of an argument. It has to beat
the version without it, walk-forward, on data that was not used to tune it. Four
layers passed. Four did not. One passed the reasoning and then failed the test,
and is documented below rather than quietly removed.

---

## 1. Match level — held-out seasons

Four seasons never used for tuning. Metric is ranked probability score, lower is
better.

| Season | Model RPS | Bookmaker RPS |
|---|---|---|
| 2019/20 | 0.19865 | 0.19871 |
| 2020/21 | 0.21363 | 0.21315 |
| 2021/22 | 0.19459 | 0.18897 |
| 2025/26 | 0.20804 | 0.20528 |
| **All (1,520 matches)** | **0.20373** | **0.20153** |

Within **1.1%** of the pre-match market (football-data's pre-closing average
odds, collected in the days before kickoff), on free data, with no team news and
no in-play information. The market is the benchmark worth measuring against because
it aggregates far more information than this model can see.

Reproduce: `python src/backtest.py`

---

## 2. Season level — seven pre-season forecasts

Fitted the day before each season began, then simulated to a final table.

| Model | MAE (points) | Rank correlation |
|---|---|---|
| Naive baseline (last season's table) | 10.95 | 0.690 |
| Club ratings only | 9.389 | 0.758 |
| **+ squad layer** | **9.264** | **0.770** |
| **+ manager layer (shipped)** | **9.216** | **0.772** |

The eventual champion appeared in the model's top three in all seven backtested
seasons.

Reproduce: `python src/preseason_bt.py`, then `src/backtest_squad.py` and
`src/backtest_manager.py` for the layer comparisons.

---

## 3. Uncertainty calibration

Two sources of uncertainty, both required.

Bootstrap resampling alone gives 80% intervals that cover only **55–70%** of
outcomes — the model is overconfident, because resampling captures uncertainty
about the ratings but not genuine squad change over a summer.

Season-to-season drift was measured directly across **170 team-seasons**:
**0.171 SD** for attack and **0.180 SD** for defence. Adding it brings 80%
interval coverage to **77%**.

The value in the model is the value the data produced. It was not tuned to make
the coverage look better.

---

## 4. Tested and rejected

Four ideas failed the same gate the shipped layers passed. The code stays in the
repository so the results can be checked rather than taken on trust.

| Rejected | MAE | Rank corr | Verdict |
|---|---|---|---|
| Market odds blend | 9.322 | 0.753 | Marginal MAE gain, worse ranking and calibration |
| Manager effects, all changes | 9.311 | 0.760 | Worse on every metric |
| Manager values from foreign leagues | not built | — | Player conversion was already weak at n≈200; a dozen manager moves would be noise |
| Fixture congestion | no change | — | Effect reverses sign between eras; see §6 |

### The instructive one

Version one of the manager layer penalised Bournemouth for hiring Iraola and
Liverpool for hiring Slot — who then won the league. The layer could not
distinguish *"this manager is worse"* from *"we have never seen this manager
before"*: both had no Premier League record and were entered as league average.

Restricting the layer to moves where **both** managers have a Premier League
record flips the result, and that version ships at weight 0.25 — half the tested
optimum, deliberately.

---

## 5. Shipped, then measured, then reverted

The entry that matters most, because it went live before it was tested.

**The problem it was meant to solve is real.** Picking the largest of home / draw
/ away sounds obviously correct, and has two visible flaws. In a Dixon-Coles grid
the draw is almost never the largest of the three — it peaks near 28% — so the
model could essentially never predict a draw, while about 24% of matches are
drawn. And a genuinely even fixture at 36% / 28% / 36% got "called" for the home
side on a margin far below the model's own noise.

**The fix shipped without a backtest.** When the top two outcomes fell within 4
percentage points, publish a draw. The justification was that it produced draws
for about 20% of fixtures against a real rate of 24% — a calibration argument,
not an accuracy test.

**Then it was tested.** 6,090 matches, 2010–2026, using de-vigged Bet365 closing
probabilities as a stand-in for well-calibrated match probabilities:

| Rule | Hit rate | Draws called | Draws called right |
|---|---|---|---|
| **Plain argmax** | **54.60%** | 0 | 0 |
| Close-call < 2pp | 54.25% | 212 | 67 |
| Close-call < 3pp | 54.07% | 322 | 97 |
| Close-call < 4pp (shipped) | 53.89% | 423 | 129 |
| Close-call < 6pp | 53.61% | 675 | 206 |
| Close-call < 10pp | 52.96% | 1,125 | 346 |

Every threshold is worse, monotonically so as the threshold widens. The 4pp rule
cost **0.71 percentage points**.

This should have been predicted from first principles: argmax maximises expected
hit rate by construction, so any rule that overrides it must on average do worse.
"A coin-flip shouldn't be called for the home side" is a statement about
presentation. It was mistaken for one about accuracy.

**Then a second, worse problem surfaced.** In **422 of the 423** matches the rule
changed, the draw it published was the outcome the model rated **least likely of
the three**. Because the draw peaks near 28%, whenever home and away are level
the draw is usually *third*, not second. The rule was not splitting the
difference between two close options — it was advertising the model's least
favoured result as its prediction.

The live site made this visible before the backtest did: Tottenham v Newcastle
showed 38.4% / 25.7% / 35.9% and published *draw*.

**Reverted.** The published outcome is plain argmax again. Closeness is now a
display fact rather than a prediction: a fixture whose top two outcomes are within
4pp is labelled **"too close to call"**, and neither club is dimmed, but the
prediction and the score against it remain the model's actual favourite. That
keeps the honesty the rule was reaching for at zero cost in accuracy.

**What was never affected.** Ranked probability score and log loss score the
published probabilities and ignore which outcome is nominated, so the model's
forecasting quality was identical throughout. Only the hit rate moved. Gameweek 1
scores 6 of 10 under either rule.

**Caveat on the test.** It used bookmaker probabilities rather than the model's
own, because refitting 6,090 matches walk-forward is expensive. The model sits
within 1.1% of the market on RPS, so the conclusion should transfer, but this
validates the *decision rule* rather than the model.

---

## 6. Fixture congestion — real effect, unusable

Two datasets were built: **1,488** domestic cup ties and **735** UEFA matches
involving English clubs, 2014–2026. Every league match was given a "days since
this club's last match of any kind" figure, and each club compared against its
own season average so squad quality cannot contaminate the result.

**Domestic cups show nothing.** Clubs on three days' rest or fewer created
slightly *more* (+0.025 xG), conceded slightly *less* (−0.015) and won slightly
*more* (+0.058 points). None significant. The likely reason is rotation — the
League Cup is exactly where a big club rests players.

**European matches split by stage, which is the interesting part:**

| Situation | xG conceded vs own average | p |
|---|---|---|
| After a UEFA **knockout** tie | **+0.094** (worse) | 0.067 |
| After an **away** knockout tie | **+0.129** (worse) | 0.067 |
| After a group / league-phase tie | −0.087 (better) | 0.024 |

That is a rotation signature. Group matches are rested and cost nothing; knockout
ties are played with the best XI and appear to leave a defensive mark.

**But it does not survive out-of-sample testing.** Added to the ratings model and
backtested walk-forward over seven seasons and 150 affected matches:

| Period | Effect |
|---|---|
| 2019–2022 | +0.0038 RPS — worse |
| 2023–2025 | −0.0050 RPS — better |
| **All seven seasons** | **+0.00002 — nothing** |

Four seasons prefer no adjustment, three prefer the largest one. Shipping it
would have meant fitting three seasons of noise. With 150 affected matches this
is underpowered; worth re-running in a few more seasons.

Reproduce: `python src/backtest_congestion.py`

---

## 7. Findings that shaped the model

**Championship points do not predict Premier League performance.** Across 33
promoted clubs since 2015, the correlation between Championship points and
first-season Premier League rating is **+0.043**. Norwich went up with 97 points
and were the worst promoted side in the sample; Wolves went up with 99 and were
the best. A regression on Championship points does no better than a flat average,
so every promoted club gets the same prior with a deliberately wide error bar.

**Foreign form travels poorly.** Fitting the same plus-minus model in four other
leagues, then comparing players who moved to England:

| League | Players who moved | Correlation | Variance explained |
|---|---|---|---|
| La Liga | 197 | 0.103 | 1.1% |
| Serie A | 180 | 0.205 | 4.2% |
| Bundesliga | 154 | 0.256 | 6.5% |
| Ligue 1 | 213 | 0.180 | 3.2% |

A player's output abroad explains under 7% of what they do in the Premier League.
The conversion correctly shrinks foreign values almost to league average.

**A higher title chance does not require more expected points.** Title
probability lives in the far right tail, not the mean. A club with a wider
distribution can have fewer expected points and a better chance of winning the
league.

---

## 8. Scoreline predictions — where the ceiling is

Not a validation of the model so much as a measurement of the task, because the
number it produces looks like failure until you know the bound.

Empirical scoreline distribution, **4,560 matches** from 2014 onward in
`data/processed/matches.parquet`:

| Scoreline | Frequency |
|---|---|
| 1-1 | 10.79% |
| 1-0 | 8.93% |
| 2-1 | 8.22% |
| 2-0 | 7.76% |
| 0-1 | 7.28% |
| 1-2 | 7.00% |

Those six cover **50.0% of all matches**. By contrast 4-0 occurs **1.91%** of the
time, 4-1 **1.80%**, and either side reaching four or more happens in **12.61%**
of matches.

Two consequences:

**Predicted scores will almost never leave the 0–2 range**, and that is correct
rather than a fault. Goals arrive at roughly 1.5 per side per match; the most
likely count for almost any team in almost any game is 0, 1 or 2. A model
regularly predicting 4-0 would be wrong far more often.

**Exact-score accuracy is bounded near one in nine.** The single most likely
scoreline in a match carries only a low-teens percentage even when the fixture is
badly one-sided. So the modal scoreline is reported alongside expected goals,
which carry the information the mode discards — a 2-0 pick can conceal a
substantial chance of a rout.

---

## 9. Mid-season forecasts

Section 2 scores one forecast per season, made the day before it starts. The live
model makes thirty-eight, and every one after the first has to decide how far to
trust this season's results over last season's ratings. That was never tested.

Each season 2018/19–2025/26 was replayed at Gameweeks 5, 10, 19 and 28 as the
live pipeline would have seen it: same priors, same ridge, same bootstrap, actual
points as the starting table, remaining fixtures simulated, scored against the
final table. The squad and manager layers are not included — their player-match
data is not in the repository — but neither changes during a season, so this
tests the in-season core they sit on. Final tables are rebuilt from results, so
points deductions are ignored, identically for every model.

**Updating is clearly worth it.** Mean absolute error in final points, 8 seasons:

| Checkpoint | Live model | Frozen pre-season ratings | Points per game | Pre-match odds* |
|---|---|---|---|---|
| GW5 | **7.82** | 8.39 | 15.91 | 5.69 |
| GW10 | **6.84** | 7.69 | 10.39 | 5.25 |
| GW19 | **4.58** | 5.41 | 6.14 | 3.87 |
| GW28 | **3.57** | 4.10 | 4.25 | 3.23 |

\*Not achievable: market odds for each remaining match, collected in the days
before it is played, with information the checkpoint could not have. A ceiling, not a rival.
The gap is widest early in the season, which is where missing information — team
news, squad quality — costs most.

**The in-season settings were tuned: nothing ships.** 84 variants: time decay
`xi` ∈ {0.003, 0.0045, 0.007, 0.01}, an extra weight of 1, 2 or 3 on
current-season matches, and seven drift schedules. Every variant is compared to
live on the same season × checkpoint cells, with common random numbers.

| | MAE | vs live | Relegation Brier | 80% coverage |
|---|---|---|---|---|
| Live (`xi` 0.0045, weight 1, drift → 0.04 over 12 games) | 5.701 | — | 0.937 | 72.5% |
| Best of 84 (`xi` 0.003, current season ×2) | 5.648 | −0.053 | 0.968 | 72.3% |

The best variant is better by 0.05 points per club — under 1% — and worse on
relegation. It is the winner of 84 tries on the same data, so the split-half test
decides it:

| Chosen on | Tested on | Chosen variant MAE | Live MAE |
|---|---|---|---|
| 2018–2021 | 2022–2025 | 6.250 | 6.283 |
| 2022–2025 | 2018–2021 | 5.303 | **5.119** |

Choosing on one half and testing on the other gives a trivial gain one way and a
0.19-point loss the other. That is noise. The live settings stay. Tripling the
current-season weight was worse than doubling it: the data does not support
reacting harder to early results.

Drift schedules move expected points by at most ±0.03 — they change the width of
the distribution, not its centre.

**The finding that matters: mid-season intervals are too narrow.**

| Checkpoint | GW5 | GW10 | GW19 | GW28 | All |
|---|---|---|---|---|---|
| 80% interval coverage, live | 71.2% | 68.8% | 76.9% | 73.1% | 72.5% |

640 team forecasts, and the published 10th–90th percentile ranges contain the
final total about 72% of the time. The pre-season calibration of Section 3 does
not carry into the season. Holding drift at 0.16 all season brings coverage to
80–84%, but worsens title and relegation Brier scores — so the uncertainty is not
too small everywhere, it is in the wrong clubs. A handful change a lot during a
season and most do not. Uniform noise cannot express that; a per-club dynamic
rating can. **The published range is recalibrated in Section 12.**

The optional `boost` argument added to `fit_ratings` for this test defaults to
1.0 and is not used by the live model.

Reproduce: `python src/backtest_midseason.py run` (resumable, ~40 minutes on 7
cores), then `python src/backtest_midseason.py report`.

---

## 10. Market odds in match predictions — benchmark, not ingredient

The market beats the model at match level (Section 1). The obvious move is to
blend the odds into the published fixture probabilities. This is a different
question from the rejected market blend in Section 4, which was about
pre-season *tables*.

Held-out seasons 2019/20–2025/26, 2,660 matches. Odds are football-data's market
average **pre-closing** odds — collected Friday afternoon for weekend games,
Tuesday afternoon for midweek — because that is what the live pipeline can fetch
before kickoff. Each blend is fitted only on seasons before the one it is scored
on.

| Method | RPS | Log loss | Hit rate |
|---|---|---|---|
| Model alone | 0.20069 | 0.9768 | 53.8% |
| **Market alone** | **0.19806** | **0.9677** | **54.5%** |
| Linear blend | 0.19813 | 0.9680 | 54.4% |
| Log-linear blend | 0.19842 | 0.9694 | 54.3% |

| Paired, per match | RPS difference | 95% CI | Seasons better |
|---|---|---|---|
| Market vs model | −0.00263 | [−0.00434, −0.00087] | 5 / 7 |
| Linear blend vs model | −0.00256 | [−0.00398, −0.00118] | 6 / 7 |
| Linear blend vs market | +0.00007 | [−0.00035, +0.00045] | 4 / 7 |

The blend beats the model and does not beat the market. The weight it learns on
the market is **0.66–0.93**, rising in recent seasons: the model adds almost no
information the odds do not already contain. Publishing the blend would improve
the scorecard by about 1.3% by publishing, in effect, the bookmakers' numbers.

**Decision: the odds are a benchmark, not an ingredient.** The published
probabilities stay the model's own. Pre-closing odds are fetched on every run,
stored beside each prediction and frozen with it at kickoff, and the public
scorecard grades both on the same matches.

Caveat: the backtest model has no squad, manager or availability layer, which
the live model does — and those target exactly what the market knows. The live
gap may be smaller. The live scorecard is where that will show.

Reproduce: `python src/backtest_odds.py` (~1 minute).

---

## 11. Drift noise in match predictions — checked, harmless

The published fixture probabilities are not what Section 1 validated. Section 1
scores a single fitted model. The live pipeline averages the scoreline grid over
40 bootstrap models, each with the season-simulation drift added to every
rating — N(0, 0.16) at the start of a season, shrinking to 0.04 by Gameweek 12.
Averaging over noisy ratings flattens the probabilities, most of all early on.
Drift is right for the table simulation, where one draw of ratings persists for
38 matches. Whether it costs anything for a single match had to be measured.

Walk-forward, 2018/19–2025/26, 3,040 matches, weekly refits, identical bootstrap
draws for every variant. The single-fit column reproduces Section 1's held-out
figures exactly, so the harness matches the validated backtest.

| Variant | RPS | Log loss | Hit rate | Mean favourite prob. |
|---|---|---|---|---|
| Single fit (Section 1) | 0.19903 | 0.9676 | 54.6% | 51.6% |
| Bootstrap, no noise | 0.19916 | 0.9681 | 54.7% | 51.5% |
| **Bootstrap + live drift** | **0.19917** | **0.9682** | **54.7%** | **51.5%** |
| Bootstrap + ½ drift | 0.19915 | 0.9680 | 54.7% | 51.5% |
| Bootstrap + 2× drift | 0.19935 | 0.9690 | 54.5% | 51.5% |

| Paired against live, per match | RPS difference | 95% CI | Seasons better |
|---|---|---|---|
| Single fit | −0.00014 | [−0.00042, +0.00015] | 5 / 8 |
| Bootstrap, no noise | −0.00001 | [−0.00017, +0.00016] | 3 / 8 |
| ½ drift | −0.00002 | [−0.00011, +0.00006] | 3 / 8 |
| 2× drift | +0.00018 | [+0.00000, +0.00036] | 3 / 8 |

The flattening is real but small: it moves the favourite's probability by 0.1pp
on average, and the RPS cost is 0.07% with a confidence interval spanning zero.
Only doubling the drift is measurably worse. Early in the season, where the
drift is largest, live is no worse than the single fit (GW1–6: 0.19085 against
0.19090).

**No change.** Removing the noise from match predictions would mean a second
code path for a gain the data cannot distinguish from zero.

Reproduce: `python src/backtest_flatten.py run` (~20 minutes on 8 cores), then
`python src/backtest_flatten.py report`.

---

## 12. Mid-season points ranges — recalibrated

Section 9 found that the published 80% points range held the final total only
72.5% of the time mid-season. Same replay, same 640 team forecasts (8 seasons ×
GW 5/10/19/28 × 20 clubs).

**Where the misses fall.** 13.6% of clubs finished below their range and 13.9%
above it, against 10% each if calibrated. So the ranges are too narrow, not
biased. The misses concentrate at the extremes:

| Clubs, ranked by forecast | 80% coverage | Below range | Above range | Mean error (pts) |
|---|---|---|---|---|
| Top 5 | 73.8% | 9.4% | **16.9%** | +0.72 |
| Middle 10 | 74.1% | 12.8% | 13.1% | +0.18 |
| Bottom 5 | 68.1% | **19.4%** | 12.5% | −0.39 |

The model pulls the best and worst clubs slightly toward the middle, and their
seasons end further out than it allows.

**Tried and rejected: in-season random walk.** In the simulation a club's
strength is frozen from the checkpoint to May. `fixture_grids_walk` lets every
rating move weekly through the run-in, σ from 0.01 to 0.03 (the summer drift of
0.17/year would be ~0.024/week as a random walk). Checked first: σ = 0
reproduces Section 9 exactly.

| σ per week | CRPS | 80% coverage | Title Brier | Relegation Brier |
|---|---|---|---|---|
| 0 (live) | 4.0435 | 72.5% | 0.3326 | 0.9365 |
| 0.02 | 4.0415 | 73.4% | 0.3336 | 0.9367 |
| 0.03 | 4.0406 | 75.0% | 0.3378 | 0.9405 |

CRPS — a proper score for the whole points distribution — moves by −0.003 with
a 95% interval of ±0.02, and the two halves of the sample disagree. Coverage
barely moves. Rejected; the function stays in `simulate.py` for reproducibility
and is not used live.

**Shipped: a calibrated display range.** No change to the model improves the
forecast, so the fix is to publish the range that actually holds 80%. Across all
640 forecasts that is the **7th–93rd percentile** of the simulated totals:

| Percentiles published | 80% coverage | GW5 | GW10 | GW19 | GW28 |
|---|---|---|---|---|---|
| 10th–90th (before) | 72.5% | 71.2% | 68.8% | 76.9% | 73.1% |
| **7th–93rd (now)** | **79.7%** | 79.4% | 76.2% | 83.1% | 80.0% |

Split-half: choosing the width on 2018–21 and testing on 2022–25 gives 74.4%
(from 70.9%); the other way round gives 83.4% (from 74.1%). Noisy, but centred
on 80% where the old range was not.

This changes only the two numbers of the displayed range (`RANGE_PCT` in
`update.py`). Expected points, positions and every title, top-4, top-6 and
relegation probability are untouched. The underlying pull toward the middle is
still there; a dynamic rating model is the way to remove it, and this backtest
is how it will be judged.

Reproduce: `python src/backtest_ranges.py run` (~5 minutes), then
`python src/backtest_ranges.py report`.

---

## 13. Game-state adjusted xG — shipped

A team two goals up sits deep and creates less than it could; the team chasing
the game gets chances it would not get at 0-0. Raw xG from those spells
misstates both sides' strength ("score effects").

**Data.** Every shot Understat has recorded, 2014/15 onward — 4,610 matches, one
request each, cached in `data/understat/shots/` (2 MB). Understat stores each
shot's minute and result but not the score, so the score is replayed in shot
order. Own goals are listed under the team whose player scored them; with that
rule **every one of the 4,610 final scores rebuilds exactly**.

**The adjustment.** Each non-penalty shot is scaled by e^(β × goals ahead),
capped at ±2: shots taken while leading count up, shots taken while chasing
count down. A single parameter rather than measured factors, because measuring
them is biased in both directions. Against a team's season-long level rate,
leading looks *productive* (0.77–0.89) — a team that goes two up is usually
having a good day. Against its level rate in the same match the effect is
exaggerated (1.2–1.4 up, 0.55–0.7 down) — the side that scored first was
probably on top at level. The truth lies between, so the backtest chooses β.

**Match level.** Walk-forward exactly as Section 1, 2,660 matches. `orig` is the
npxG already in the model and reproduces Section 1 to the fifth decimal.

| β | RPS | Log loss | vs β = 0 | 95% CI | Seasons better |
|---|---|---|---|---|---|
| orig | 0.20069 | 0.9768 | +0.00008 | [−0.00004, +0.00020] | 1 / 7 |
| 0 (shots, unadjusted) | 0.20061 | 0.9766 | — | — | — |
| 0.05 | 0.20037 | 0.9757 | −0.00024 | [−0.00042, −0.00005] | 6 / 7 |
| **0.1 (shipped)** | **0.20021** | **0.9752** | **−0.00040** | **[−0.00078, −0.00002]** | **6 / 7** |
| 0.15 | 0.20013 | 0.9749 | −0.00048 | [−0.00107, +0.00012] | 5 / 7 |
| 0.2 | 0.20015 | 0.9750 | −0.00046 | [−0.00123, +0.00032] | 5 / 7 |
| 0.3 | 0.20049 | 0.9765 | −0.00012 | [−0.00128, +0.00113] | 4 / 7 |

The gain is the same in both halves at β = 0.1 (2019–21 −0.00043, 2022–25
−0.00039), and choosing β on either half and testing on the other still wins.
0.1 is the smallest setting whose gain is clearly real — the same hedge as the
squad and manager layers. Against the model as it was (`orig`) it is −0.00048,
about 18% of the gap to the market in Section 10.

**Season level.** The Section 9 replay (live settings, 32 cells) on adjusted xG:

| | MAE | CRPS | Title Brier | Top-4 Brier | Relegation Brier |
|---|---|---|---|---|---|
| Before | 5.701 | 4.044 | 0.3326 | 1.1972 | 0.9365 |
| **β = 0.1** | **5.643** | **4.001** | **0.3207** | **1.1910** | **0.9174** |

Better on every metric; MAE better in 7 of 8 seasons, CRPS in 6 of 8 (the other
two within 0.006). On the 2026/27 forecast after Gameweek 5 it spreads the table
— strong clubs up, weak clubs down — which is the pull toward the middle found
in Section 12 partly undone.

**Live.** `update.py` tops up the current season's shots on every run (ten or so
new matches a gameweek) and fits the ratings on adjusted xG
(`GAMESTATE_BETA = 0.1`). A match whose shots are not yet available keeps plain
npxG until the next run. The squad layer's scale and the "rode their luck" notes
still use plain npxG, as they were built and tuned on it. The 7th–93rd range of
Section 12 was re-measured on adjusted xG and still holds: 80.6% coverage
(79.4 / 78.8 / 85.6 / 79.4% at GW 5 / 10 / 19 / 28), so it is unchanged.

Reproduce: `python scripts/pull_understat_shots.py 2014 2025`, then
`python src/backtest_gamestate.py run` and `report`.

---

## 14. Set-piece xG weighting — tested, no change

Set pieces (corners, free kicks, other set plays) are 23.6% of non-penalty xG.
A club's set-piece output looks much less persistent than its open play: across
240 club-seasons, first-half xG predicts second-half xG with

| | Open play | Set pieces |
|---|---|---|
| xG for | r = 0.82 | r = 0.37 |
| xG against | r = 0.66 | r = 0.32 |

which suggested set pieces should count for less in the ratings. Tested with
the Section 13 harness — game-state β at its live 0.1, set-piece xG weighted by
w — on the same 2,660 matches:

| w | RPS | vs w = 1 | 95% CI | Seasons better |
|---|---|---|---|---|
| 0.4 | 0.20070 | +0.00049 | [+0.00006, +0.00093] | 1 / 7 |
| 0.7 | 0.20039 | +0.00018 | [−0.00003, +0.00039] | 2 / 7 |
| **1.0 (live)** | **0.20021** | — | — | — |
| 1.15 | 0.20016 | −0.00005 | [−0.00015, +0.00005] | 4 / 7 |
| 1.3 | 0.20014 | −0.00007 | [−0.00027, +0.00013] | 4 / 7 |

Down-weighting hurts, in both halves of the sample. The low split-half
correlation reflects how little clubs differ in set-piece output (between-club
SD 0.10 per game, against 0.37 for open play), not an absence of signal — and
the ratings already average hundreds of matches, which handles the noise.
Up-weighting gains too little to tell from zero. **No change.** For the same
reason a separate set-piece rating per club is not worth building: there is
little between clubs to separate. The `w_sp` argument in `gamestate.py`
defaults to 1 and is not used live.

Reproduce: `python src/backtest_gamestate.py setpiece`, then `setpiece-report`.

---

## 15. Dynamic (Kalman) ratings — tested and rejected

The static model refits every run with exponentially decaying match weights,
which treats every club as changing at the same rate. The alternative tried
here (`dynamic.py`): each club's attack and defence is a state that drifts
week to week with its own uncertainty, updated after every match day by an
iterated extended Kalman filter on the same quasi-Poisson target (game-state
adjusted xG blended with goals). Uncertainty grows by σ_w² a week and σ_s² over
the summer, with optional summer shrinkage κ; promoted clubs restart at the
promoted prior; league scoring level and home advantage drift too, so the
empty-stadium season could pull home advantage down and let it recover.

One pass through history gives walk-forward predictions on the same 7-day
snapshot schedule as Section 1's refits. 72 settings (σ_w 0.005–0.03, σ_s
0.1–0.25, κ 1 or 0.85, dispersion 0.5–1), chosen on 2016/17–2018/19, scored on
the same 2,660 held-out matches as Section 13, against the live static model:

| | RPS 2019–25 | vs live | 95% CI | Seasons better |
|---|---|---|---|---|
| Live static model | 0.20021 | — | — | — |
| Dynamic, chosen on tuning seasons | 0.20119 | +0.00098 | [+0.00007, +0.00192] | 1 / 7 |
| Dynamic, best of 72 *with hindsight* | 0.20032 | +0.00011 | — | — |

Worse at every point of the season, August and September included, where it
should have helped most. Even choosing the setting by its held-out score — which
flatters it — it does not reach the static model, so no tuning of this design
wins. The static fit, which re-estimates every club jointly on four seasons with
recency weighting and mild shrinkage, already is a well-tuned version of the
same idea; the filter commits to each update as it goes and pays for its
approximations. The motivation from Section 12, the pull toward the middle, is
largely addressed by Section 13 and the calibrated range.

**Rejected.** `dynamic.py` stays in the repository so the result can be checked.

Reproduce: `python src/backtest_dynamic.py run` (~5 minutes), then `report`.

---

## 16. Player values with an informed prior — shipped at squad weight 0.25

The squad layer turns squad changes into rating shifts using each player's
plus-minus (RAPM) value. Plain RAPM shrinks every player toward league average,
a heavy pull in football, where substitutions are few and the same players
share the pitch. The informed version (`players.py`, after basketball's
box-score-prior RAPM) shrinks each player toward what his own per-90 numbers
suggest — xG, xA, xGChain, xGBuildup and position — mapped onto RAPM values
from training data only. The strongest signal on both ends is xGBuildup:
players involved in moves that lead to shots make their side create more and
concede less.

**Data.** Every player in every match since 2014/15 from Understat's match
rosters — minutes, side, position, per-match xG/xA/xGChain/xGBuildup (4,610
matches, `data/understat/rosters/`). The side a player played for is recorded,
so no club inference is needed.

**Component test — predicting a side from its lineup.** For each season
2019/20–2025/26, values fitted on earlier seasons predict each club's xG for
and against from who actually played; ridge strength chosen on 2017–18.
Players with no earlier Premier League minutes count as zero in every model.

| Club-season xG error per match | Held out 2019–25 |
|---|---|
| No player information | 0.293 |
| Plain RAPM (best ridge) | 0.191 |
| **Informed prior** | **0.186** |

Better in 5 of 7 seasons; over 140 club-seasons −0.0051, 95% CI [−0.0119,
+0.0009]. Modest, but less fragile to the ridge setting: with heavier
shrinkage (ridge 1–4) informed wins by 7–10%; only at the lightest tested
(0.25) is plain better.

**Season test — the squad layer, which is what ships.** Section 13's replay
(game-state xG, live drift, common random numbers), pre-season plus GW
5/10/19/28, 2019/20–2025/26: 35 forecasts. As in the original squad backtest,
minute shares are those actually played — the optimistic version. Old values
reproduced: plain RAPM at ridge 1.0 correlates 0.978 with the live file.

| 35 forecasts | Points MAE | CRPS | Title Brier | Top-4 Brier | Relegation Brier |
|---|---|---|---|---|---|
| No squad layer | 6.473 | 4.567 | 0.3399 | 1.400 | 1.139 |
| Before: plain values, weight 0.5 | 6.384 | 4.485 | 0.3580 | 1.369 | 1.115 |
| **Informed values, weight 0.25** | **6.310** | **4.437** | **0.3577** | **1.364** | **1.104** |
| Informed values, weight 0.5 | 6.202 | 4.350 | 0.3857 | 1.345 | 1.076 |

At weight 0.5 informed values beat the old layer on points in **all 7
seasons** (−0.18 points per club) — but title Brier gets worse, and almost all
of it comes from 2022/23 and 2023/24. Both summers Manchester City replaced
established players with signings who had no Premier League record (Haaland;
Gvardiol and Doku), and the squad layer counts such a player as league average,
so it marked the eventual champions down. Sharper values mark them down harder.
That is a flaw in valuing new signings, not in the informed prior — and it was
already in the live layer, whose title Brier is on average worse than having no
squad layer at all (0.358 against 0.340, within noise).

**Shipped: informed values at weight 0.25.** Better or equal to the old layer
on every metric — points better in 29 of 35 forecasts (−0.074), title unchanged.
`build_player_values.py` writes `rapm_live.pkl`; `SQUAD_W = 0.25`. On the
2026/27 forecast after Gameweek 5 it moves Leeds above Manchester United and
Bournemouth above Chelsea (each within half a point); City's title chance 43.0%
→ 45.9%.

**Next.** Value signings with no Premier League record from something other
than zero — transfer fee, or minutes and role at the previous club. That is
the lever for title odds, and with it the full informed weight may become safe.

Reproduce: `python scripts/pull_understat_rosters.py 2014 2026`, then
`python src/backtest_players.py run` / `report` and
`python src/backtest_squad_players.py run` / `report` (~10 minutes).

---

## 17. Valuing new signings — tested, not shipped

Section 16 left one flaw: a signing with no Premier League record counts as
league average. Two ways to do better were tested.

**Who signed him.** Newcomers who play are not average. Since 2016/17,
minutes-weighted, they rate +0.49 against +0.27 for all players — so counting
them at zero also marks down every club that buys. And the buyer matters:
newcomers at the strongest quarter of clubs (by the previous season's xG
difference) average 1.27, at the weakest 0.53, at promoted clubs 0.14. City's
2022/23 and 2023/24 arrivals rated 2.2–2.9 (Haaland, Gvardiol, Álvarez, Doku).

**How much he lifted his previous team.** The same informed plus-minus fitted
in La Liga, Serie A, the Bundesliga and Ligue 1 — every match since 2014/15,
lineups and shots, fitted only on the four seasons before each transfer window
— so a player is judged by his on-pitch impact and output at his last club.
Across 465 movers it predicts Premier League value weakly:

| From | Movers | Correlation with PL value |
|---|---|---|
| Serie A | 98 | 0.34 |
| Ligue 1 | 131 | 0.27 |
| Bundesliga | 112 | 0.18 |
| La Liga | 124 | 0.17 |
| **All** | **465** | **0.24** |

No better than the old European values (Section 7). Haaland shows why: Dortmund
were only a little better with him on the pitch than without (+0.45), against
+2.90 at City — a strong side either way hides an individual's effect. Leagues
outside these four (Portugal, the Championship) are not covered at all.

**Season test.** Both, learned from earlier windows only, in the Section 16
replay (35 forecasts):

| | Points MAE | Title Brier | Top-4 Brier | Relegation Brier |
|---|---|---|---|---|
| Live (informed values, weight 0.25) | 6.310 | 0.3577 | 1.364 | 1.104 |
| + who signed him, 0.25 | 6.320 | 0.3564 | 1.362 | 1.107 |
| + who signed him + previous club, 0.25 | 6.317 | 0.3566 | 1.360 | 1.104 |
| + both, 0.5 | 6.205 | 0.3808 | 1.332 | 1.078 |
| + both, 1.0 | 6.054 | 0.4397 | 1.304 | 1.038 |

At the live weight it changes nothing measurable (title −0.0011, points
+0.008, both intervals spanning zero). It does fix the bias it targeted — City's
2022/23 squad downgrade shrinks from −0.46 to −0.20 — but raising the weight
still trades better positions for worse title odds. What remains is not about
signings: City lost established players (Gündoğan, Mahrez; Jesus and
Zinchenko to Arsenal) and won anyway. With seven champions in the sample, a
strong system absorbing departures cannot be told apart from two unlucky
seasons.

**Not shipped.** The newcomer models stay in `players.py` and the test in
`backtest_squad_players.py`, to re-run as seasons accumulate. The foreign
values cache (`foreign_values.parquet`) is committed; the 24 MB of foreign match
data is not, and `pull_understat_rosters.py` re-downloads it in ~18 minutes.

Reproduce: `python src/backtest_squad_players.py run` and `report` (~15
minutes) read the committed cache. To rebuild the cache itself, download the
foreign leagues first (`python scripts/pull_understat_rosters.py 2014 2025
<league>` for Bundesliga, La_liga, Serie_A, Ligue_1), then delete
`foreign_values.parquet`.

---

## 18. FPL expected points — stage 1, backtest

Expected Fantasy Premier League points for every player and fixture
(`fpl.py`, `fpl_model.py`), built from what the model already has. Each part
uses only data from before the gameweek's first kickoff:

| Part | From |
|---|---|
| Minutes | chance of 60+, of a substitute appearance, expected minutes — last 8 fixtures, recency-weighted, unused-sub rows included |
| Goals, assists | the player's share of his side's xG / xA while on the pitch (Understat, penalties included, so takers count) × the model's predicted team goals × time on the pitch |
| Clean sheets, goals conceded | the model's predicted goals for the opponent |
| Saves | the keeper's save rate × the opponent's predicted goals |
| Bonus | goals, assists, clean sheets and saves → bonus, by position |

Scale factors and the bonus model are learned from earlier seasons only
(2020/21 is calibration only). Scoring rules are FPL's 2020/21–2024/25.

**Data.** Real FPL points per player per fixture from the vaastav
Fantasy-Premier-League archive, 2020/21–2025/26 (163,404 player-fixtures).
FPL players are linked to Understat players by name within club and season —
exact, contained ("David Raya" in "David Raya Martin"), FPL's short web name
("Rodri"), surname — covering **99.5–100% of minutes** every season.

**The benchmark that could not be used.** The archive includes FPL's own
expected points (xP), and on its face they are far better than the model. They
were recorded with hindsight:

| Regular starters who then… | FPL archived xP | Model |
|---|---|---|
| scored 10+ points | 7.08 | 4.91 |
| scored 2–3 points | 3.10 | 3.62 |
| were unexpectedly benched (0 minutes) | 1.21 | 3.46 |

No pre-match forecast separates future hauls from blanks that sharply. The
archived xP is not used; the live pipeline will record FPL's genuine
pre-deadline figures instead. (Its double-gameweek values are also the
gameweek total repeated on each fixture row — summing them would double it.)

**Fair benchmarks** — information any FPL player has before the deadline:
*form*, mean points over the last five gameweeks (FPL's own expected points are
essentially this scaled by fixture), and *points per match played* × chance of
playing. Every listed player-gameweek, benched players included:

| Correlation with actual points | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| **Model** | **0.540** | **0.557** | **0.548** | **0.512** | **0.550** |
| Form | 0.476 | 0.491 | 0.488 | 0.495 | 0.492 |
| Points per match | 0.505 | 0.523 | 0.519 | 0.484 | 0.518 |

**Picking players** — mean actual points of each method's top 10, per
gameweek, 189 gameweeks:

| | Top-10 vs form | Top-10 vs points per match | Captain vs form |
|---|---|---|---|
| Model | **+0.72** [+0.51, +0.92] | **+0.60** [+0.38, +0.81] | +0.69 [−0.20, +1.60] |

Better every season. The captain edge is positive but not yet certain.

**Not in the backtest, in the live version:** injury and suspension flags
(FPL publishes them, the archive does not keep them), so live predictions
should do better than this. **Not modelled yet:** 2025/26's
defensive-contribution points, which is why 2025/26 predictions run a little
low (1.08 against 1.17 per player-gameweek).

Reproduce: download `merged_gw_<season>.csv` and `players_raw_<season>.csv`
for 2020-21 … 2025-26 from github.com/vaastav/Fantasy-Premier-League into
`data/fpl_history/` (gitignored, ~45 MB), then `python src/backtest_fpl.py
build` (~1 minute) and `report`. Team predictions per match are in
`fpl_team_lambdas.parquet`.

---

## 19. Live in-season record

Updated automatically. Only predictions saved **before** kickoff are scored, and
a prediction is frozen the moment its match starts.

**After Gameweek 1 (10 matches):**

| Metric | Value |
|---|---|
| Ranked probability score | 0.2215 |
| Log loss | 0.9458 |
| Outcomes called correctly | 6 of 10 |
| Matches drawn | 1 |
| Draws called | 2 |

Ten matches tells you essentially nothing. Both a good and a bad start are well
within what randomness produces at this sample size, and no conclusion should be
drawn until several gameweeks have accumulated.

Current figures always live in `outputs/status.json` and on the site.

---

## 20. Not validated

Stated plainly, because a validation document that only lists successes is
marketing.

- **The "too close to call" label** is a display choice, not a validated
  improvement. The 4pp threshold that triggers it is inherited from the reverted
  rule and has never been tuned; it changes nothing about what is predicted or
  scored, so it cannot cost accuracy, but nor has it been shown to help readers.
- **The two-gameweek prediction window and freeze-at-kickoff rule** are
  correctness properties, not accuracy improvements. Neither was backtested
  because neither claims to make forecasts better.
- **The predicted scoreline** is shown but not scored. Exact-score accuracy is
  bounded near one in nine (Section 8) and has no baseline in this repository,
  so it was removed from the scorecard rather than published without context.
- **`MGR_W = 0.25`** was chosen at half its tested optimum as a hedge against
  overfitting. That is a judgement call, not a result. (`SQUAD_W`, once the same,
  is now 0.25 on the evidence in Section 16.)
- **Signings with no Premier League record** count as league average in the
  squad layer, or get a heavily shrunk European value. Section 17 shows better
  estimates exist (who signed him) but change nothing measurable at the live
  squad weight, so the simpler treatment stays.
- **The manager layer rests on a handful of qualifying moves per season.** It
  improves the backtest, but the sample is small enough that the improvement
  could be luck.
- **Everything about 2026/27 specifically.** The promoted clubs have no Premier
  League record, and the model says so with a wide error bar rather than a
  confident number.

---

## Reproducing any of it

```bash
pip install -r requirements.txt

python src/backtest.py              # match level, walk-forward
python src/preseason_bt.py          # season level + drift measurement
python src/backtest_squad.py        # does the squad layer help?
python src/backtest_manager.py      # does the manager layer help?
python src/backtest_market.py       # does blending market odds help?
python src/backtest_congestion.py   # does fixture congestion help?
python src/backtest_midseason.py run  # forecasts at GW 5/10/19/28; then: report
python src/backtest_odds.py         # blend market odds into match predictions?
python src/backtest_flatten.py run  # does drift noise hurt match predictions? then: report
python src/backtest_ranges.py run   # mid-season points ranges; then: report
python scripts/pull_understat_shots.py 2014 2025   # shot data, once
python src/backtest_gamestate.py run  # game-state adjusted xG; then: report
python src/backtest_gamestate.py setpiece  # set-piece weighting; then: setpiece-report
python src/backtest_dynamic.py run  # Kalman-filter ratings vs static; then: report
python scripts/pull_understat_rosters.py 2014 2026   # player-match data, once
python src/backtest_players.py run  # informed vs plain player values; then: report
python src/backtest_squad_players.py run  # squad layer, season level; then: report
python src/backtest_fpl.py build    # FPL expected points (needs data/fpl_history/); then: report
# Section 17 runs from the committed foreign_values.parquet. Rebuilding that
# cache needs the foreign leagues first (~18 min, not committed):
#   python scripts/pull_understat_rosters.py 2014 2025 Bundesliga   (and La_liga,
#   Serie_A, Ligue_1)
python src/tune.py                  # resumable hyperparameter grid search
```

Several are resumable and write partial results after each season, so they can be
run in chunks. The congestion and market backtests take the longest.

If a number in this file disagrees with what a script produces, the script is
right and this file is out of date. Please open an issue.
