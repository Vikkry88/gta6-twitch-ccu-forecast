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
1's print (`p1`), in two complementary ways, blended 50/50 in log
space:

1. **Quadratic log-log regression, per hour** — log(viewers_h) on
   log(p1) and log(p1)² over the case's comparables (centred at
   log p1 = 10 for conditioning). The spread at each hour is the
   empirical distribution of that hour's regression residuals, read
   off at the requested quantile levels. The median curve is smoothed
   with a 3-point moving average.
2. **k-nearest-neighbour ratio curve** — the 8 comparables nearest in
   log hour-1; per hour the 5th/25th/50th/75th/95th percentiles of
   their log ratios ln(viewers_h / viewers_1), each quantile curve
   smoothed the same way.

The blended quantiles are widened 1.1× in log space around the
median: leave-one-out over the comparables showed the un-widened 90%
band covering only 0.866 of hours, and 1.1× brings that to 0.894 at
no cost to the loss.

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
| this entry (v2) | 0.376 | 0.382 | 0.235 | 0.141 |

Weighted to the practice cases' hour-1 sizes (23k, 117k, 270k,
119k), the estimate is **≈ 0.25 vs ≈ 0.34 for persistence** on the
same games. The entry is strongest where the final will live: for
large hour-1 prints (≥ 200k) it scores 0.141 vs 0.226 for
persistence. LOO 90% coverage 0.894, typical median miss 76%.

## Practice results

The kit's `grade-local` cannot grade this competition locally — the
scorer and reference data are server-only (`makefun grade-local`
reports: "its scorer and reference data are not published in the
kit, so it is graded on the server only"). The house baseline's
published practice loss is **0.387**. The leave-one-out estimate
above (≈ 0.25 practice-like) is the honest expectation, not a
measured practice grade; the real practice grade comes from the
server.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` per
case. In the grader it runs as `python3 -P -s /bot/main.py`
(Python 3.13, stdlib only, ≤ 8 s per case; this build takes ~0.07 s
per case). Output: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75,
q95]}, ...]}` for hours 2–24, quantiles rounded to 2 decimals,
half up. Local validation on all 4 practice cases: 23 rows, hours
2–24 once each, 5 non-decreasing quantiles > 0 and ≤ 1e9, ≤ 65,536
bytes, two runs byte-identical; edge cases (no comparables, hour-1
print 0, no hour-1 row, malformed JSON) all still produce a valid
output file.
