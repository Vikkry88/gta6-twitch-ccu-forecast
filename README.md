# GTA VI launch-day Twitch viewers — forecast entry

Forecasts how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (2026-11-19 UTC), hours 2–24, as five
quantiles per hour. Competition: `gta6-twitch-ccu-forecast-01`.

## Method (draft)

Hour 1 (00:00–01:00 UTC) only catches the earliest rolling-midnight
unlocks, so the category dips through the small hours and climbs hard
as Europe and the US unlock, peaking in the UTC evening. How far it
climbs depends strongly on how big hour 1 already was
(corr(log hour-1, log peak ratio) ≈ −0.94 across the comparables), so
the forecast is conditioned on hour 1's print in two ways, blended
50/50 in log space:

1. **Quadratic log-log regression, per hour** — log(viewers_h) on
   log(p1) and log(p1)² over the case's comparables, spread from the
   empirical residual distribution at each hour; median curve smoothed
   (3-point moving average).
2. **k-nearest-neighbour ratio curve** — the 8 comparables nearest in
   log hour-1; per hour the 5th/25th/50th/75th/95th percentiles of
   their log ratios ln(viewers_h / viewers_1), smoothed the same way.

Per-hour fallbacks: other component → all-comparables median ratio →
house persistence (spread 0.15·√(h−1)). Nothing is fetched from the
network; all calibration comes from the comparables shipped inside
each case file. Standard library only, deterministic, no randomness.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` per
case. In the grader it runs as `python3 -P -s /bot/main.py`.

## Practice results (draft — to be filled after grading)

- House persistence baseline: loss ≈ 0.387.
- This entry: TBD.
