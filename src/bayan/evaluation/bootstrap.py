"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("values must not be empty")

    rng = np.random.default_rng(seed)

    point = float(np.mean(values))
    boot = np.empty(n_boot, dtype=float)

    n = len(values)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(values[idx])

    lower = float(np.quantile(boot, alpha / 2))
    upper = float(np.quantile(boot, 1 - alpha / 2))

    return point, lower, upper


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b):
        raise ValueError("paired inputs must have equal length")

    diff = b - a
    delta = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)

    n = len(diff)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(diff[idx])

    lower = float(np.quantile(boot, alpha / 2))
    upper = float(np.quantile(boot, 1 - alpha / 2))

    return delta, lower, upper
