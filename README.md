[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Vikkry88/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Vikkry88/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers — forecast entry

Forecasts how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (2026-11-19 UTC), hours 2–24, as five
quantiles per hour. Competition: `gta6-twitch-ccu-forecast-01`
($250 pot, free entry).

## Method

Hour 1 (00:00–01:00 UTC) only catches the earliest rolling-midnight
unlocks, so the category dips through the small hours and climbs hard
as Europe and the US unlock, peaking in the UTC evening. How far it
climbs depends strongly on how big hour 1 already was: across the 37
usable comparables, corr(log hour-1, log peak ratio) ≈ −0.94 — a game
whose hour 1 is already huge barely grows, while one whose hour 1 is
tiny can grow a hundredfold. So the forecast is conditioned on hour
1's print (`p1`), as the average (in log space) of two members that
estimate the same conditional quantiles two robust ways:

A. **Quantile regression, per hour and level** — each of the five
   quantiles of log(viewers_h) is fitted directly by a quadratic in
   log(p1) (centred at log p1 = 10), minimising that level's own
   pinball loss by iteratively reweighted least squares. Fitting
   quantiles directly, instead of a mean plus residual quantiles,
   fits the heavy right tail of launch growth without letting the few
   hundred-fold growers drag the fit. Blended 70/30 with the
   k-nearest-neighbour ratio curve below (k=12).
B. **Least-absolute-deviation regression, per hour** — the
   conditional median of log(viewers_h) is fitted by a quadratic in
   log(p1) that minimises absolute error, so the few hundred-fold
   growers cannot pull it; the spread at each hour is the empirical
   distribution of that hour's residuals around the LAD fit, read off
   at the requested quantile levels. Blended 70/30 with the
   k-nearest-neighbour ratio curve below (k=8).

The **k-nearest-neighbour ratio curve** — the comparables nearest in
log hour-1; per hour the 5th/25th/50th/75th/95th percentiles of
their log ratios ln(viewers_h / viewers_1), each quantile curve
smoothed the same way. Each member's quantiles are widened 1.1× in
log space around its median, and the two members are averaged in log
space: leave-one-out over the comparables gives v3 a loss of 0.369
(0.251 on launches with a large hour-1 print, 0.259
practice-weighted) against v2's 0.375 (0.260, 0.266). The median
curve of each member is smoothed with a 3-point moving average.

Per-hour fallbacks: the other component → all-comparables median
ratio → house persistence (log spread 0.15·√(h−1)). A case that fails
entirely still gets a valid persistence forecast, so the program
never crashes on a case and always writes a valid output file.

Nothing is fetched from the network; all calibration comes from the
comparables shipped inside each case file (the GTA VI category's own
pre-release history has no hour-1 print on its nominal day start and
is skipped). The case's pre-release history level was tested as an
extra predictor and adds nothing (corr with peak ≈ 0.02), so it is
not used. Standard library only, deterministic, no randomness, no
wall-clock use.

## Calibration summary (leave-one-out over the 37 comparables)

Pinball loss in the competition's units (2 × standard pinball):

| method | all 37 | p1 5k–50k | p1 50k–200k | p1 ≥ 200k |
|---|---|---|---|---|
| persistence (house baseline) | 0.930 | 0.601 | 0.262 | 0.226 |
| this entry (v3) | 0.369 | 0.366 | 0.235 | 0.133 |

Weighted to the practice cases' hour-1 sizes (23k, 117k, 270k,
119k), the estimate is **≈ 0.26 vs ≈ 0.34 for persistence** on the
same games. The entry is strongest where the final will live: for
large hour-1 prints (≥ 200k) it scores 0.133 vs 0.225 for
persistence. LOO 90% coverage 0.897. On the 12 practice-like folds
(single-player, tier-comparable, p1 10k–400k) v3 scores 0.312 vs
v2's 0.325. v2's measured practice loss was 0.222 (rank #3,643 of
5,992); v3's practice grade comes from the server after submission.

## Practice results

The kit's `grade-local` cannot grade this competition locally — the
scorer and reference data are server-only (`makefun grade-local`
reports: "its scorer and reference data are not published in the
kit, so it is graded on the server only"). The house baseline's
published practice loss is **0.387**. The leave-one-out estimate
above (≈ 0.26 practice-weighted, 0.312 practice-like) is the honest
expectation, not a measured practice grade; the real practice grade comes from the
server.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` per
case. In the grader it runs as `python3 -P -s /bot/main.py`
(Python 3.13, stdlib only, ≤ 8 s per case; this build takes ~0.4 s
per case). Output: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75,
q95]}, ...]}` for hours 2–24, quantiles rounded to 2 decimals,
half up. Local validation on all 4 practice cases: 23 rows, hours
2–24 once each, 5 non-decreasing quantiles > 0 and ≤ 1e9, ≤ 65,536
bytes, two runs byte-identical; edge cases (no comparables, hour-1
print 0, no hour-1 row, malformed JSON) all still produce a valid
output file.
