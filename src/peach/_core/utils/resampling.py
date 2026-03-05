"""Shared permutation testing and bootstrap CI infrastructure."""

import numpy as np
from typing import Any, Callable


def permutation_test(
    fit_fn: Callable,
    stat_fn: Callable,
    data: Any,
    *,
    shuffle_fn: Callable,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Generic permutation test.

    Parameters
    ----------
    fit_fn : callable
        fit_fn(data) -> model
    stat_fn : callable
        stat_fn(model) -> scalar test statistic
    data : Any
        Input data (array, tuple, etc.)
    shuffle_fn : callable
        shuffle_fn(data, rng) -> permuted data
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: observed_stat, null_distribution, p_value
    """
    rng = np.random.default_rng(seed)
    observed_model = fit_fn(data)
    observed_stat = stat_fn(observed_model)

    null_distribution = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_data = shuffle_fn(data, rng)
        perm_model = fit_fn(perm_data)
        null_distribution[i] = stat_fn(perm_model)

    # Two-sided p-value: fraction of null >= observed
    p_value = (np.sum(null_distribution >= observed_stat) + 1) / (n_permutations + 1)

    return {
        "observed_stat": observed_stat,
        "null_distribution": null_distribution,
        "p_value": p_value,
    }


def bootstrap_ci(
    fit_fn: Callable,
    stat_fn: Callable,
    data: Any,
    *,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Generic bootstrap confidence interval.

    Parameters
    ----------
    fit_fn : callable
        fit_fn(data) -> model
    stat_fn : callable
        stat_fn(model) -> scalar statistic
    data : Any
        Input data (must support integer indexing for resampling).
    n_bootstrap : int
        Number of bootstrap samples.
    ci_level : float
        Confidence level (0, 1).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: point_estimate, ci_lower, ci_upper, bootstrap_distribution
    """
    rng = np.random.default_rng(seed)
    data_arr = np.asarray(data)
    n = len(data_arr)

    point_model = fit_fn(data_arr)
    point_estimate = stat_fn(point_model)

    bootstrap_distribution = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_sample = data_arr[idx]
        boot_model = fit_fn(boot_sample)
        bootstrap_distribution[i] = stat_fn(boot_model)

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_distribution, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_distribution, 100 * (1 - alpha / 2))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_distribution": bootstrap_distribution,
    }
