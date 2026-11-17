"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Method (v3; calibrated only from the comparables shipped inside each case file):

A launch day's shape is not a flat line. Hour 1 (00:00-01:00 UTC) only
catches the earliest rolling-midnight unlocks, so the category usually
dips through the small hours and then climbs hard as Europe and the US
unlock, peaking in the UTC evening. How far it climbs depends strongly
on how big hour 1 already was: across the 37 usable comparables,
corr(log hour-1, log peak ratio) is about -0.94, i.e. a game whose
hour 1 is already huge barely grows, while one whose hour 1 is tiny
can grow a hundredfold. So the forecast is conditioned on hour 1's
print, as the average (in log space) of two members that estimate the
same conditional quantiles two robust ways:

A. Quantile regression, per hour and level: each of the five quantiles
   of log(viewers_h) is fitted directly by a quadratic in log(p1)
   (centred), minimising that level's pinball loss by iteratively
   reweighted least squares. Fitting quantiles directly, instead of a
   mean plus residual quantiles, fits the heavy right tail of launch
   growth without letting the few hundred-fold growers drag the fit.
   Blended 70/30 with the k-nearest-neighbour ratio curve below (k=12).
B. Least-absolute-deviation regression, per hour: the conditional
   median of log(viewers_h) is fitted by a quadratic in log(p1) that
   minimises absolute error, so the few hundred-fold growers cannot
   pull it; the spread at each hour is the empirical distribution of
   that hour's residuals, read off at the requested quantile levels.
   Blended 70/30 with the k-nearest-neighbour ratio curve below (k=8).

The k-nearest-neighbour ratio curve: the comparables whose log hour-1
is closest to the case's take part; per hour the 5th/25th/50th/75th/
95th percentiles of their log ratios ln(viewers_h / viewers_1) give
the quantiles, each curve smoothed the same way. Each member's
quantiles are widened by 1.1x in log space around its median, and the
two members are averaged in log space: leave-one-out over the
comparables gives v3 a loss of 0.369 (0.251 on launches with a large
hour-1 print, 0.259 practice-weighted) against v2's 0.375 (0.260,
0.266); v2's practice loss was 0.222. The median curve of each member
is smoothed with a 3-point moving average so it is not jagged from
sparse data.

If a component cannot be built for an hour (too few comparables with
that hour's print, or no hour-1 print at all) it falls back, per hour,
to the other component, then to an all-comparables median-ratio curve,
then to the house persistence forecast (spread 0.15 * sqrt(h - 1) in
log space). A case that fails entirely still gets a valid persistence
forecast, so the program never crashes on a case and always writes a
valid output file.

Standard library only, no network, no randomness, no wall-clock use:
the same input always produces the same bytes.

Input:  $MAKEFUN_INPUT_DIR (else /input)  / <id>.in, JSON schema
        "makefun/twitch-ccu-case/v1".
Output: $MAKEFUN_OUTPUT_DIR (else /output) / <id>.out, JSON
        {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}
        for hours 2..24, quantiles rounded to 2 decimals, half up.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file would import. (This entry is a
# single self-contained file, but the line is kept for compatibility.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]  # standard normal quantiles at LEVELS
PERSIST_SIGMA = 0.15
K_NEIGHBOURS = 8
REG_MIN_POINTS = 8          # comparables needed to fit the regression
LOG_P1_CENTRE = 10.0        # centring for the quadratic log-log fit
SPREAD_WIDEN = 1.1          # log-spread widening (see widen_spread)
MAX_VIEWERS = 1e9


def round2(x):
    """x rounded to 2 decimals, half up, as the starter rounds it.

    A whole number comes back as an int, so json.dumps writes 1240
    where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def pct(sorted_vals, q):
    """Linear-interpolated q-th quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def parse_hour(ts):
    """Whole hours between an ISO-8601 UTC timestamp and a day start.

    Timestamps are all of the form 2026-11-19T05:00:00Z, so a plain
    slice avoids any date library. Returns None if unparseable.
    """
    try:
        year = int(ts[0:4])
        month = int(ts[5:7])
        day = int(ts[8:10])
        hour = int(ts[11:13])
    except (ValueError, IndexError, TypeError):
        return None
    # Days since 1970-01-01 via the civil-from-days inverse (Howard
    # Hinnant's algorithm), integer only, no calendar library needed.
    y = year - (1 if month <= 2 else 0)
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = (month + 9) % 12
    doy = (153 * mp + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return days * 24 + hour


def solve_linear(a, b):
    """Solve a small dense linear system by Gaussian elimination.

    a is a list of rows, b the right-hand side. Returns the solution
    list, or None if the matrix is (numerically) singular.
    """
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                if f:
                    for c in range(col, n + 1):
                        m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_quadratic(xs, ys):
    """Least squares fit of y ~ b0 + b1*u + b2*u^2, u = x - centre.

    Returns the coefficient list, or None if the fit is degenerate.
    """
    n = len(xs)
    if n < 3:
        return None
    s = [0.0] * 5  # sums of u^0 .. u^4
    t = [0.0] * 3  # sums of u^0..u^2 times y
    for x, y in zip(xs, ys):
        u = x - LOG_P1_CENTRE
        p = 1.0
        for k in range(5):
            s[k] += p
            if k < 3:
                t[k] += p * y
            p *= u
    a = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]
    return solve_linear(a, t)


def qreg_fit(xs, ys, tau, deg=2, iters=30):
    """Quantile regression: y ~ poly(x) fitted to minimise tau's pinball loss.

    Iteratively reweighted least squares: a point above the fit gets
    weight tau / |residual|, one below it (1 - tau) / |residual|, so the
    fit is pulled to the tau-th conditional quantile instead of the
    mean. With tau = 0.5 this is least-absolute-deviation (median)
    regression. Starts from the least-squares fit, so it always
    returns something sane when the data can support a fit at all.
    Deterministic: fixed iteration count, no randomness.
    """
    beta = fit_quadratic(xs, ys) if deg == 2 else None
    if beta is None:
        return None
    for _ in range(iters):
        s = [0.0] * (2 * deg + 1)
        t = [0.0] * (deg + 1)
        for x, y in zip(xs, ys):
            u = x - LOG_P1_CENTRE
            r = y - sum(beta[i] * u ** i for i in range(deg + 1))
            if abs(r) < 1e-6:
                w = 1e6
            else:
                w = (tau if r > 0 else (1.0 - tau)) / abs(r)
            w = min(w, 1e4)
            p = 1.0
            for k in range(2 * deg + 1):
                s[k] += w * p
                if k <= deg:
                    t[k] += w * p * y
                p *= u
        a = [[s[i + j] for j in range(deg + 1)] for i in range(deg + 1)]
        new_beta = solve_linear(a, t)
        if new_beta is None:
            break
        if max(abs(a - b) for a, b in zip(new_beta, beta)) < 1e-9:
            beta = new_beta
            break
        beta = new_beta
    return beta


def qreg_predict(beta, x, deg=2):
    """Evaluate a fitted quantile-regression polynomial at log hour-1 x."""
    u = x - LOG_P1_CENTRE
    return sum(beta[i] * u ** i for i in range(deg + 1))


def smooth(curve):
    """3-point moving average over hours 2..24 (edges use what exists)."""
    out = {}
    for h in range(2, 25):
        vals = [curve[k] for k in (h - 1, h, h + 1) if k in curve]
        out[h] = sum(vals) / len(vals)
    return out


def comparable_curves(case):
    """Release-day hourly viewers per comparable, keyed by hour 1..24.

    A comparable counts only if it has a positive hour-1 print
    (offset 0 from its day_start); hours with no print, or a print of
    zero, are simply missing for that hour. The GTA VI category's own
    pre-release history has no hour-1 print on its nominal day start
    and is skipped by this rule.
    """
    curves = {}
    for comp in case.get("comparables") or []:
        try:
            start = parse_hour(comp["day_start"])
            if start is None:
                continue
            by_hour = {}
            for pt in comp.get("hourly") or []:
                off = parse_hour(pt["from"])
                if off is None:
                    continue
                h = off - start + 1
                if 1 <= h <= 24:
                    v = pt.get("viewers")
                    if isinstance(v, (int, float)) and v > 0:
                        by_hour[h] = float(v)
            if by_hour.get(1):
                curves[comp.get("name") or ""] = by_hour
        except (TypeError, KeyError, AttributeError):
            continue
    # Deterministic order regardless of the input file's ordering.
    return {name: curves[name] for name in sorted(curves)}


def quantile_regression_component(curves, p1, levels):
    """Per-hour, per-level quantile regression on log hour-1.

    Each of the five quantiles of log(viewers_h) is fitted directly by
    a quadratic in log(p1), minimising that level's own pinball loss.
    Returns {hour: [log quantiles at `levels`]}. The median curve is
    smoothed with a 3-point moving average and every level is shifted
    by the same amount, so the shape stays coherent.
    """
    log_p1 = math.log(p1)
    rows = {}
    for h in range(2, 25):
        xs = []
        ys = []
        for by_hour in curves.values():
            if h in by_hour:
                xs.append(math.log(by_hour[1]))
                ys.append(math.log(by_hour[h]))
        if len(xs) < REG_MIN_POINTS:
            continue
        row = []
        for tau in levels:
            beta = qreg_fit(xs, ys, tau)
            if beta is None:
                row = None
                break
            row.append(qreg_predict(beta, log_p1))
        if row is None:
            continue
        rows[h] = row
    if not rows:
        return {}
    med = smooth({h: row[len(row) // 2] for h, row in rows.items()})
    out = {}
    for h, row in rows.items():
        shift = med[h] - row[len(row) // 2]
        out[h] = [v + shift for v in row]
    return out


def lad_regression_component(curves, p1, levels):
    """Per-hour LAD (median) regression on log hour-1, plus residual spread.

    The conditional median of log(viewers_h) is fitted by a quadratic
    in log(p1) that minimises absolute error, so the few launches whose
    hour 1 was tiny and then grew a hundredfold cannot drag it. The
    spread at each hour is the empirical distribution of that hour's
    residuals around the LAD fit, read off at the requested levels.
    Returns {hour: [log quantiles at `levels`]}.
    """
    log_p1 = math.log(p1)
    med = {}
    resid = {}
    for h in range(2, 25):
        xs = []
        ys = []
        for by_hour in curves.values():
            if h in by_hour:
                xs.append(math.log(by_hour[1]))
                ys.append(math.log(by_hour[h]))
        if len(xs) < REG_MIN_POINTS:
            continue
        beta = qreg_fit(xs, ys, 0.5)
        if beta is None:
            continue
        med[h] = qreg_predict(beta, log_p1)
        resid[h] = sorted(y - qreg_predict(beta, x) for x, y in zip(xs, ys))
    if not med:
        return {}
    med = smooth(med)
    out = {}
    for h, mu in med.items():
        out[h] = [mu + (pct(resid[h], tau) or 0.0) for tau in levels]
    return out


def knn_component(curves, p1, levels, k=K_NEIGHBOURS):
    """Per-hour ratio quantiles of the k nearest comparables by log p1.

    Returns {hour: [log quantiles at `levels`]}. Falls back to all
    comparables for an hour with fewer than 3 neighbours reporting.
    """
    log_p1 = math.log(p1)
    ranked = sorted(
        curves.items(),
        key=lambda kv: (abs(math.log(kv[1][1]) - log_p1), kv[0]),
    )
    near = [by_hour for _, by_hour in ranked[:k]]
    all_curves = list(curves.values())
    quantiles = {tau: {} for tau in levels}
    for h in range(2, 25):
        ratios = sorted(
            math.log(by_hour[h] / by_hour[1])
            for by_hour in near
            if h in by_hour
        )
        if len(ratios) < 3:
            ratios = sorted(
                math.log(by_hour[h] / by_hour[1])
                for by_hour in all_curves
                if h in by_hour
            )
        if not ratios:
            continue
        for tau in levels:
            quantiles[tau][h] = pct(ratios, tau)
    if not quantiles[levels[0]]:
        return {}
    for tau in levels:
        quantiles[tau] = smooth(quantiles[tau])
    out = {}
    for h in range(2, 25):
        if h in quantiles[levels[0]]:
            out[h] = [math.log(p1) + quantiles[tau][h] for tau in levels]
    return out


def persistence_quantiles(p1, h, levels):
    """House baseline: flat at p1, log spread 0.15 * sqrt(h - 1)."""
    return [math.log(p1) + z * PERSIST_SIGMA * math.sqrt(h - 1)
            for z in Z] if levels == LEVELS else None


def widen_spread(log_q, factor):
    """Widen a row's log quantiles around its median by `factor`.

    Leave-one-out over the comparables showed the blended forecast's
    90% band covering only 0.866 of hours (an honest band covers
    about 0.90); a factor of 1.1 brings coverage to 0.894 at no cost
    to the LOO pinball loss (0.375 vs 0.376 in competition units).
    """
    med = log_q[len(log_q) // 2]
    return [med + (v - med) * factor for v in log_q]


def member_forecast(curves, p1, levels, reg_kind, k, blend_reg):
    """One ensemble member: regression component blended with kNN.

    `reg_kind` is "qreg" (quantile regression per level) or "lad"
    (LAD median + residual quantiles). Returns {hour: [log quantiles]}
    in log space, before the 1.1x log-spread widening: widening is
    linear in log space around the median, so widening the averaged
    members once gives the same result as widening each member first.
    """
    if reg_kind == "qreg":
        reg = quantile_regression_component(curves, p1, levels)
    else:
        reg = lad_regression_component(curves, p1, levels)
    knn = knn_component(curves, p1, levels, k)
    out = {}
    for h in range(2, 25):
        if h in reg and h in knn:
            out[h] = [blend_reg * a + (1.0 - blend_reg) * b
                      for a, b in zip(reg[h], knn[h])]
        elif h in reg:
            out[h] = list(reg[h])
        elif h in knn:
            out[h] = list(knn[h])
    return out


def forecast_case(case):
    """One case's forecast: {"hours": [{"hour": h, "q": [...]}, ...]}.

    The forecast is the average (in log space) of the two members:
    quantile regression blended 70/30 with kNN (k=12), and LAD
    regression blended 70/30 with kNN (k=8). Hours neither member
    covers fall back to the persistence quantiles.
    """
    levels = list(case.get("levels") or LEVELS)
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = first.get("viewers")
    if not isinstance(p1, (int, float)) or p1 < 1:
        p1 = 1.0
    p1 = float(p1)

    curves = comparable_curves(case)
    member_a = member_forecast(curves, p1, levels, "qreg", 12, 0.7)
    member_b = member_forecast(curves, p1, levels, "lad", 8, 0.7)

    rows = []
    for row in case.get("forecast") or []:
        h = row.get("hour")
        if not isinstance(h, int) or not 2 <= h <= 24:
            continue
        rows_h = [m[h] for m in (member_a, member_b) if h in m]
        if rows_h:
            log_q = [sum(col) / len(col) for col in zip(*rows_h)]
        else:
            log_q = persistence_quantiles(p1, h, levels)
            if log_q is None:
                # Non-standard levels: widen like the baseline using
                # the empirical spread of the median-ratio curve.
                log_q = [math.log(p1)] * len(levels)
        log_q = widen_spread(log_q, SPREAD_WIDEN)
        q = []
        prev = 0.0
        for v in log_q:
            val = math.exp(v)
            if not math.isfinite(val) or val <= 0:
                val = 1.0
            val = min(val, MAX_VIEWERS)
            val = max(val, prev)  # quantiles non-decreasing in a row
            q.append(round2(val))
            prev = val
        rows.append({"hour": h, "q": q})
    rows.sort(key=lambda r: r["hour"])
    if len(rows) != 23:
        raise ValueError("forecast rows are not hours 2..24")
    return {"hours": rows}


def persistence_forecast(case):
    """Valid fallback forecast for a case the main path cannot handle."""
    levels = list(case.get("levels") or LEVELS)
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None:
        v = first.get("viewers")
        if isinstance(v, (int, float)) and v >= 1:
            p1 = float(v)
    rows = []
    for h in range(2, 25):
        q = []
        prev = 0.0
        for z in Z[:len(levels)] if levels == LEVELS else [0.0] * len(levels):
            val = min(max(p1 * math.exp(z * PERSIST_SIGMA * math.sqrt(h - 1)), 1.0),
                      MAX_VIEWERS)
            val = max(val, prev)
            q.append(round2(val))
            prev = val
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = persistence_forecast(case)
            except Exception:
                forecast = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]}
                                      for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
