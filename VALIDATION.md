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
rating can. **Open, and the next thing this backtest will judge.**

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

## 11. Live in-season record

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

## 12. Not validated

Stated plainly, because a validation document that only lists successes is
marketing.

- **The "too close to call" label** is a display choice, not a validated
  improvement. The 4pp threshold that triggers it is inherited from the reverted
  rule and has never been tuned; it changes nothing about what is predicted or
  scored, so it cannot cost accuracy, but nor has it been shown to help readers.
- **The two-gameweek prediction window and freeze-at-kickoff rule** are
  correctness properties, not accuracy improvements. Neither was backtested
  because neither claims to make forecasts better.
- **The exact-scoreline metric** has no baseline in this repository. It should be
  compared against a bookmaker correct-score market before any claim is made
  about it.
- **`SQUAD_W = 0.5` and `MGR_W = 0.25`** were chosen at half their tested optima
  as a hedge against overfitting. That is a judgement call, not a result.
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
python src/tune.py                  # resumable hyperparameter grid search
```

Several are resumable and write partial results after each season, so they can be
run in chunks. The congestion and market backtests take the longest.

If a number in this file disagrees with what a script produces, the script is
right and this file is out of date. Please open an issue.
