"""Hour-by-hour forecast of GTA VI's launch-day Twitch viewers.

Model (all learned at run time from the case's own comparables, no
hard-coded launch data):

  * Every comparable is a past big launch with hourly Twitch viewers
    for its release day.  For each forecast hour h we regress
    log(viewers_h) on log(hour-1 viewers) across the comparables.
    Big launches grow far less through the day than small ones do,
    so the level of hour 1 carries real information about the shape
    of the rest of the day; the regression is how the program learns
    that from the data it is given.
  * The per-hour slopes are shrunk halfway toward their mean, which
    steadies the fit, and the intercept is refit through the cloud's
    centre at the shrunken slope.
  * Past a certain size a launch stops changing shape: the biggest
    launches on record stay roughly flat through the day and peak
    near hour 1 or in the evening at about the hour-1 level.  So for
    very large hour-1 prints the regression median is blended toward
    the median log-ratio profile of the largest comparables, reaching
    that profile outright at the top of the range.  This keeps the
    forecast sane when the case is bigger than anything in the data.
  * The spread of each hour's quantiles comes from the residuals of
    that hour's regression, measured robustly (median absolute
    deviation), and narrowed a little: the raw residual spread is
    fattened by a few launches whose unlock pattern was nothing like
    a midnight console launch, and the pinball loss prefers the
    honest-but-tighter band.

Everything is deterministic pure arithmetic over the input file, so
two runs on the same case write identical bytes.  Standard library
only, no network.
"""

import json
import math
import os
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# How the model is tuned.  These are structural choices, not data:
# the regression floor drops the three comparables whose hour-1 print
# was a few hundred viewers (their categories were effectively empty
# at midnight, so their ratios are not launch shapes at all); the
# spread factor and the slope shrinkage were picked on the four
# practice cases and then left alone.
REGRESSION_FLOOR = 1000.0
SHRINK = 0.5
SPREAD_FACTOR = 0.65
MEGA_FLOOR = 300000.0
BLEND_START = 300000.0
BLEND_FULL = 1200000.0


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to about 1e-9, which is far tighter than any forecast
    needs; it lets the program honour whatever levels a case asks
    for instead of hard-coding five z-values.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("level out of (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def mad_sigma(residuals):
    """Robust spread of residuals: median absolute deviation, scaled
    to match a normal standard deviation."""
    centre = median_of(residuals)
    return 1.4826 * median_of([abs(r - centre) for r in residuals])


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_hours(comparable):
    """Map hour number (1..24) to viewers for the comparable's own
    release day, from its hourly series.  Hours with no point, or a
    non-positive count, are left out."""
    start = parse_time(comparable["day_start"])
    end = start + timedelta(days=1)
    found = {}
    for row in comparable.get("hourly") or []:
        try:
            moment = parse_time(row["from"])
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= moment < end and viewers > 0:
            hour = int((moment - start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24:
                found[hour] = viewers
    return found


def collect_profiles(case):
    """One release-day profile per usable comparable: (hour-1 viewers,
    {hour: viewers}).  A comparable only counts when its release day
    is fully observed -- that is what leaves out the GTA VI category
    itself, whose series is pre-launch history rather than a launch
    day."""
    profiles = []
    for comparable in case.get("comparables") or []:
        hours = release_day_hours(comparable)
        if len(hours) == 24 and hours.get(1, 0) > 0:
            profiles.append((hours[1], hours))
    return profiles


def fit_hour(points):
    """Ordinary least squares of log viewers on log hour-1 viewers.
    Returns (intercept, slope, residuals)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return mean_y, 0.0, [y - mean_y for y in ys]
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return intercept, slope, residuals


def round_half_up_2dp(value):
    """Round to 2 decimals, halves up, as a JSON number (whole numbers
    come back as ints)."""
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    as_float = float(rounded)
    return int(as_float) if as_float.is_integer() else as_float


def forecast_case(case, name):
    levels = list(case["levels"])
    z_scores = [norm_ppf(level) for level in levels]

    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None or float(first.get("viewers") or 0) <= 0:
        p1 = 1.0
    else:
        p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    profiles = collect_profiles(case)

    # The median log-ratio profile of the largest launches: the shape
    # a forecast saturates toward when hour 1 is enormous.
    mega_profiles = [hours for h1, hours in profiles if h1 >= MEGA_FLOOR]
    mega_ratio = {}
    for hour in range(2, 25):
        ratios = [math.log(hours[hour] / hours[1]) for hours in mega_profiles
                  if hour in hours and 1 in hours]
        mega_ratio[hour] = median_of(ratios) if ratios else 0.0

    if p1 <= BLEND_START:
        blend = 0.0
    elif p1 >= BLEND_FULL:
        blend = 1.0
    else:
        blend = (log_p1 - math.log(BLEND_START)) / \
            (math.log(BLEND_FULL) - math.log(BLEND_START))

    # Fit every hour first, so the slopes can be shrunk toward their
    # own mean before any forecast is produced.  Each entry is
    # (intercept, slope, points, fallback_centre): the fallback centre
    # is only set when an hour has too few points to regress, and
    # then it already is the log-median for this case, so the row
    # loop must use it as-is instead of adding log(p1) again.
    fits = {}
    slopes = []
    for hour in range(2, 25):
        points = [(math.log(h1), math.log(hours[hour]))
                  for h1, hours in profiles
                  if h1 >= REGRESSION_FLOOR and hour in hours]
        if len(points) >= 4:
            intercept, slope, residuals = fit_hour(points)
            fits[hour] = (intercept, slope, points, None)
            slopes.append(slope)
        else:
            # Too little data for a regression: fall back to the
            # pooled median ratio for this hour, with a wide spread.
            ratios = [math.log(hours[hour] / hours[1])
                      for h1, hours in profiles if hour in hours]
            centre = log_p1 + (median_of(ratios) if ratios else 0.0)
            fits[hour] = (0.0, 0.0, [], centre)
    mean_slope = sum(slopes) / len(slopes) if slopes else 0.0

    rows = []
    for spec in case["forecast"]:
        hour = spec["hour"]
        intercept, slope, points, fallback = fits[hour]
        if fallback is not None:
            centre = fallback
            sigma = 0.5
        else:
            shrunk = slope + SHRINK * (mean_slope - slope)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            centre_x = sum(xs) / len(xs)
            centre_y = sum(ys) / len(ys)
            intercept = centre_y - shrunk * centre_x
            fitted = [y - (intercept + shrunk * x) for x, y in zip(xs, ys)]
            sigma = mad_sigma(fitted) * SPREAD_FACTOR
            centre = intercept + shrunk * log_p1
        mega_centre = log_p1 + mega_ratio[hour]
        centre = (1.0 - blend) * centre + blend * mega_centre

        quantiles = []
        previous = 0.0
        for z in z_scores:
            value = math.exp(centre + z * sigma)
            value = min(max(value, 1.0), 1e9)
            value = max(value, previous)
            quantiles.append(round_half_up_2dp(value))
            previous = quantiles[-1]
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
