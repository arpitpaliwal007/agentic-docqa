"""Dependency-free, question-level 95% intervals and exact paired McNemar tests."""
import math
import random


def quantile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def mean_ci(values, *, binary=False, seed=42, repetitions=2000):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    n = len(values)
    if not n:
        return {"mean": None, "low": None, "high": None, "n": 0, "method": "unavailable"}
    average = sum(values) / n
    if binary:
        if any(v not in (0, 1) for v in values):
            raise ValueError("Wilson intervals require binary observations")
        z = 1.959963984540054
        divisor = 1 + z*z/n
        center = (average + z*z/(2*n)) / divisor
        radius = z * math.sqrt(average*(1-average)/n + z*z/(4*n*n)) / divisor
        low, high, method = max(0., center-radius), min(1., center+radius), "Wilson 95%"
    else:
        if repetitions < 100:
            raise ValueError("Use at least 100 bootstrap repetitions")
        rng = random.Random(seed)
        estimates = [sum(rng.choices(values, k=n))/n for _ in range(repetitions)]
        low, high, method = quantile(estimates, .025), quantile(estimates, .975), "question bootstrap percentile 95%"
    return {"mean": average, "low": low, "high": high, "n": n, "method": method}


def paired_test(left, right, *, seed=42, repetitions=2000):
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if len(left) != len(right):
        raise ValueError("Paired arrays must have equal lengths")
    if any(a not in (0, 1) or b not in (0, 1) for a, b in pairs):
        raise ValueError("McNemar requires paired binary outcomes")
    losses = sum(a == 1 and b == 0 for a, b in pairs)
    gains = sum(a == 0 and b == 1 for a, b in pairs)
    discordant = gains + losses
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses)+1)) / 2**discordant) if discordant else 1.
    interval = mean_ci([b-a for a, b in pairs], seed=seed, repetitions=repetitions)
    return {"n": len(pairs), "left_only": losses, "right_only": gains, "p_value": p if pairs else None,
            "significant_0_05": p < .05 if pairs else None, "difference_right_minus_left": interval,
            "method": "exact two-sided McNemar; exploratory, unadjusted"}
