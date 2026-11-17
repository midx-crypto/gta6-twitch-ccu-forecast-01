"""Forecast GTA VI launch-day Twitch viewers, hour by hour (make.fun bounty
gta6-twitch-ccu-forecast-01).

For each case the program reads hour 1's print p1 and the past launches in
`comparables`, then forecasts hours 2..24 as five log-normal quantiles.

Model (fit fresh at runtime from the case's own comparables, nothing tuned
to any practice answer):

  For each hour h, fit a locally weighted regression of log viewers on
  log hour-1 viewers across the comparables:

      log v_h = a_h + b_h * log p1 + eps,   eps ~ N(0, sigma_h^2)

  with Gaussian kernel weights centred on the target's log p1 (bandwidth
  1.5 log units, widened until the effective sample is at least 6).  The
  local fit matters because hour 1 means different things for different
  launches: a game with a simultaneous global unlock already peaks in
  hour 1 (ratio ~1), while a rolling-midnight console launch is barely
  awake in hour 1 and grows several-fold by the evening peak.  The slope
  b_h captures that: near 1 for hour 2, decaying toward ~0 by the peak
  hours, where the level is almost independent of hour 1.  The median is
  exp(a_h + b_h * log p1); the spread sigma_h is the weighted residual
  standard deviation (floored at 0.30, capped at 2.0), and the five
  quantiles are the median times exp(z_tau * sigma_h).

  If an hour has too few comparable points, or there are no usable
  comparables at all, it falls back to persistence (median = p1) with a
  wide spread, so the output is always a valid forecast.

Only the standard library is used.  No network, no randomness, no state:
two runs on the same input write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; put it back
# first, as the starter does.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# Normal quantiles for other levels, used only if a case ever asks for them.
Z_TABLE = {0.05: -1.6448536269514722, 0.10: -1.2815515655446004,
           0.25: -0.6744897501960817, 0.50: 0.0,
           0.75: 0.6744897501960817, 0.90: 1.2815515655446004,
           0.95: 1.6448536269514722}

# Gaussian kernel bandwidth in log-viewer units for the local fit.
BANDWIDTH = 1.5
# Widen the bandwidth until the weighted fit rests on this many points.
MIN_EFFECTIVE_N = 6.0
# A fit needs at least this many comparables with the hour present.
MIN_POINTS = 6
# Spread floor / cap for the log-normal quantiles.
SIGMA_FLOOR = 0.30
SIGMA_CAP = 2.0
# Spread used when an hour cannot be fit from comparables.
FALLBACK_SIGMA = 0.80


def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_viewers(comp):
    """Map release-day UTC hour (1..24) to viewers for one comparable.

    A point counts only when it lies inside the comparable's release day
    and its viewer count is a positive number.  Hours SullyGnome did not
    sample simply stay absent.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            when = parse_ts(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (when - day_start).total_seconds() / 3600.0
        if 0 <= offset < 24:
            out[int(offset) + 1] = float(viewers)
    return out


def weighted_ols(xs, ys, ws):
    """Weighted least squares of ys on xs; returns (intercept, slope, sd).

    sd is the weighted residual standard deviation, the spread the
    quantiles are built from.
    """
    total_w = sum(ws)
    if total_w <= 0:
        return 0.0, 0.0, FALLBACK_SIGMA
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total_w
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    intercept = mean_y - slope * mean_x
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / max(total_w - 2.0, 1.0)
    return intercept, slope, math.sqrt(var) if var > 0 else 0.0


def round2(x):
    """x rounded to 2 decimals, half up, as the starter rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def fit_hour(points, log_p1):
    """Local fit for one hour; returns (log median, sigma).

    points is a list of (log hour-1 viewers, log hour-h viewers) from the
    comparables that have both hours.  The kernel is centred on the
    target's log hour-1 print and widens until the effective sample is
    big enough to trust, so a target far outside the comparables' range
    still gets a stable, near-global fit instead of a noisy local one.
    """
    if len(points) < MIN_POINTS:
        return log_p1, FALLBACK_SIGMA
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bandwidth = BANDWIDTH
    while True:
        weights = [math.exp(-0.5 * ((x - log_p1) / bandwidth) ** 2)
                   for x in xs]
        total_w = sum(weights)
        sum_w2 = sum(w * w for w in weights)
        effective_n = (total_w * total_w / sum_w2) if sum_w2 > 0 else 0.0
        if effective_n >= MIN_EFFECTIVE_N or bandwidth > 6.0:
            break
        bandwidth *= 1.5
    intercept, slope, sd = weighted_ols(xs, ys, weights)
    sigma = min(max(sd, SIGMA_FLOOR), SIGMA_CAP)
    return intercept + slope * log_p1, sigma


def hour_one_print(case):
    """Hour 1's viewer print, from `known`, else the last hourly point."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers"):
            return max(1.0, float(row["viewers"]))
    for row in reversed(case.get("history_hourly") or []):
        if row.get("viewers"):
            return max(1.0, float(row["viewers"]))
    return 1.0


def forecast_case(case, name):
    """Forecast hours 2..24 of one case as five quantiles each."""
    levels = case.get("levels") or LEVELS
    z_scores = [Z_TABLE.get(level, 0.0) if level in Z_TABLE else 0.0
                for level in levels]
    if levels == LEVELS:
        z_scores = list(Z)

    p1 = hour_one_print(case)
    log_p1 = math.log(p1)

    # One (log hour-1, {hour: log viewers}) pair per usable comparable.
    comparables = []
    for comp in case.get("comparables") or []:
        hours = day_viewers(comp)
        if 1 in hours:
            comparables.append((math.log(hours[1]),
                                {h: math.log(v) for h, v in hours.items()}))

    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours_out = []
    for row in sorted(rows, key=lambda r: r["hour"]):
        hour = row["hour"]
        if hour < 2 or hour > 24:
            continue
        points = [(log_first, by_hour[hour]) for log_first, by_hour
                  in comparables if hour in by_hour]
        log_median, sigma = fit_hour(points, log_p1)
        quantiles = []
        previous = 0.0
        for z in z_scores:
            value = math.exp(log_median + z * sigma)
            value = min(max(value, 1.0), 1e9)
            value = round2(value)
            # Rounding must never break the non-decreasing order.
            if value < previous:
                value = previous
            previous = value
            quantiles.append(value)
        hours_out.append({"hour": hour, "q": quantiles})
    return {"hours": hours_out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
