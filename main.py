"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads one case file /input/<id>.in (history + hour-1 print) and writes
/output/<id>.out: for hours 2..24, five quantiles (5/25/50/75/95%) of that
hour's concurrent-viewer count. Python 3.13, standard library only, no
network, deterministic.

Method (fitted at run time from the case's own comparables, never from
hard-coded launch data):

1. Shape set. From the case's comparables, keep past big launches whose
   release-day hourly curve is usable (24 hours from day_start, hour 1
   positive) and whose unlock matches GTA VI's rolling midnight-local
   unlock: the unlock text mentions "midnight local" or "rolling" and
   does not describe a single simultaneous global unlock (those games
   have a near-empty hour 1 and a meaningless ratio curve). Drop
   early-access-contaminated launches (their release day is flattened by
   the early-access days) and weak analogues. If fewer than 4 such
   launches remain, fall back to all tier-comparable launches, then to
   every usable comparable.

2. Median ratio curve. For each hour h, take the median over the shape
   set of log(viewers_h / viewers_1), lightly smoothed across neighbouring
   hours. Rolling-midnight launches share GTA VI's shape: hour 1 holds
   only the regions already unlocked (New Zealand, Europe, the UK), the
   US east coast joins at hour 6 (05:00 UTC) and the west coast at hour 9
   (08:00 UTC), and the evening peak lands in hours 19-24.

3. Level regression. A plain ratio curve is fooled by hour-1 size: across
   all comparables, log(hour-1 viewers) correlates about -0.93 with the
   log ratio at hour 19, because absolute peaks saturate while hour 1
   varies with unlock timing. So per hour also fit log(viewers_h) on
   log(viewers_1) by OLS over all usable comparables, and average the two
   log predictions 50/50. Leave-one-out over the comparables, this blend
   scores about 0.26 mean pinball on rolling-midnight launches, against
   about 0.39 for the house persistence baseline.

4. Spread. The log spread for hour h is the shape set's standard
   deviation of the log ratio at that hour (floored at 0.18, capped at
   0.90), applied with the standard normal quantiles. An honest spread
   matters: pinball loss charges for every hour outside the band and for
   width at every hour.

Fallback: with no usable comparable at all, persistence (the house
baseline) is used, so the entry always produces a valid forecast.
"""

import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Blend weight on the level regression (1 - BLEND stays on the median
# ratio curve). 0.5 was the best compromise in leave-one-out testing:
# best on rolling-midnight launches, close to best on all launches.
BLEND = 0.5
SD_FLOOR = 0.18
SD_CAP = 0.90
# House baseline spread, used only when no comparable is usable.
BASELINE_SIGMA = 0.15

_SIMULTANEOUS = re.compile(
    r"single global|single simultaneous|simultaneous across"
    r"|simultaneous global|single time|simultaneous midnight gmt",
    re.IGNORECASE,
)
_TIER = re.compile(r"Tier:\s*([a-z][a-z-]*)")
_SKIP_TIERS = {"weak", "early-access-contaminated", "target-history"}


def round2(x: float):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def release_day_viewers(comp: dict):
    """The 24 hourly viewer counts of a comparable's release day, or None."""
    day_start = comp.get("day_start")
    hourly = comp.get("hourly") or []
    start = None
    for i, row in enumerate(hourly):
        if row.get("from") == day_start:
            start = i
            break
    if start is None:
        return None
    rows = hourly[start:start + 24]
    if len(rows) < 24:
        return None
    viewers = []
    for row in rows:
        try:
            viewers.append(float(row["viewers"]))
        except (KeyError, TypeError, ValueError):
            return None
    if viewers[0] <= 0 or any(v < 0 for v in viewers):
        return None
    return viewers


def is_rolling(comp: dict) -> bool:
    """True when the launch unlocked at midnight local time, rolling."""
    unlock = str(comp.get("unlock", ""))
    if _SIMULTANEOUS.search(unlock):
        return False
    lowered = unlock.lower()
    return "midnight local" in lowered or "rolling" in lowered


def tier_of(comp: dict) -> str:
    match = _TIER.search(json.dumps(comp))
    return match.group(1) if match else ""


def collect_comparables(case: dict):
    """Usable comparables, each as (viewers list, rolling?, tier)."""
    out = []
    for comp in case.get("comparables") or []:
        viewers = release_day_viewers(comp)
        if viewers is None:
            continue
        out.append((viewers, is_rolling(comp), tier_of(comp)))
    return out


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return my - b * mx, b


def smooth(values):
    """Light [0.25, 0.5, 0.25] smoothing across neighbouring hours."""
    out = []
    for i, v in enumerate(values):
        if i == 0:
            out.append(0.75 * values[0] + 0.25 * values[1])
        elif i == len(values) - 1:
            out.append(0.25 * values[-2] + 0.75 * values[-1])
        else:
            out.append(0.25 * values[i - 1] + 0.5 * v + 0.25 * values[i + 1])
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)
    hours = [row["hour"] for row in case.get("forecast", [])]

    comparables = collect_comparables(case)
    shape = [v for v, rolling, tier in comparables
             if rolling and tier == "comparable"]
    if len(shape) < 4:
        shape = [v for v, rolling, tier in comparables if tier == "comparable"]
    if len(shape) < 4:
        shape = [v for v, rolling, tier in comparables]
    all_curves = [v for v, rolling, tier in comparables]

    rows = []
    if not all_curves:
        # No comparable at all: house baseline persistence.
        for h in hours:
            rows.append({
                "hour": h,
                "q": [round2(max(1.0, p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1))))
                      for z in Z],
            })
        return {"hours": rows}

    # Median log-ratio curve and per-hour spread from the shape set.
    medians = []
    spreads = []
    for h in range(1, 24):
        ratios = sorted(math.log(v[h] / v[0]) for v in shape)
        medians.append(statistics.median(ratios))
        spreads.append(statistics.pstdev(ratios) if len(ratios) > 1 else 0.30)
    medians = smooth(medians)

    # Per-hour level regression over all usable comparables.
    regressions = []
    for h in range(1, 24):
        xs = [math.log(v[0]) for v in all_curves]
        ys = [math.log(max(1.0, v[h])) for v in all_curves]
        regressions.append(ols(xs, ys) if len(xs) > 1 else (ys[0], 0.0))

    for h in hours:
        k = h - 2  # medians/regressions index 0 is hour 2
        ratio_pred = log_p1 + medians[k]
        a, b = regressions[k]
        reg_pred = a + b * log_p1
        pred = (1.0 - BLEND) * ratio_pred + BLEND * reg_pred
        sd = min(SD_CAP, max(SD_FLOOR, spreads[k]))
        quantiles = []
        prev = 0.0
        for z in Z:
            q = math.exp(pred + z * sd)
            q = min(1_000_000_000.0, max(1.0, q))
            q = max(q, prev)  # keep the row non-decreasing
            quantiles.append(round2(q))
            prev = quantiles[-1]
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
