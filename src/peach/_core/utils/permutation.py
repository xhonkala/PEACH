"""Shared permutation testing utilities for PEACH statistical controls."""

import numpy as np
from statsmodels.stats.multitest import multipletests


def permutation_pvalue(observed, null_distribution, alternative="two-sided"):
    """Compute empirical p-value from a null distribution.

    Uses the Phipson-Smyth +1/+1 correction (2010) to avoid zero p-values
    and correct for the discrete nature of permutation p-values.

    Parameters
    ----------
    observed : float or np.ndarray
        Observed test statistic(s). Shape [n_features] for vectorized.
    null_distribution : np.ndarray
        Null samples. Shape [n_permutations] for scalar, or [n_permutations, n_features].
    alternative : str
        "two-sided", "greater", or "less".

    Returns
    -------
    p_value : float or np.ndarray
        Empirical p-value(s).
    """
    null = np.asarray(null_distribution)
    obs = np.asarray(observed)
    n_perm = null.shape[0]

    if alternative == "greater":
        count = (null >= obs).sum(axis=0)
    elif alternative == "less":
        count = (null <= obs).sum(axis=0)
    else:  # two-sided
        count = (np.abs(null) >= np.abs(obs)).sum(axis=0)

    # +1/+1 correction (Phipson & Smyth 2010)
    return (count + 1) / (n_perm + 1)


def fdr_correct(pvalues, alpha=0.05, method="fdr_bh"):
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : np.ndarray
        Raw p-values.
    alpha : float
        FDR threshold.
    method : str
        Correction method (default: Benjamini-Hochberg).

    Returns
    -------
    rejected : np.ndarray[bool]
        Which hypotheses are rejected.
    pvalues_corrected : np.ndarray
        Corrected p-values.
    """
    pvals = np.asarray(pvalues).ravel()
    # Handle NaN/inf
    valid = np.isfinite(pvals)
    corrected = np.ones_like(pvals)
    rejected = np.zeros_like(pvals, dtype=bool)
    if valid.sum() > 0:
        rej, corr, _, _ = multipletests(pvals[valid], alpha=alpha, method=method)
        rejected[valid] = rej
        corrected[valid] = corr
    return rejected, corrected


def bootstrap_ci(data, statistic_fn, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape [n_samples, ...].
    statistic_fn : callable
        Function that takes data array and returns scalar or 1D array.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (default 0.95).
    seed : int
        Random seed.

    Returns
    -------
    point_estimate : float or np.ndarray
        Statistic on original data.
    ci_low : float or np.ndarray
        Lower CI bound.
    ci_high : float or np.ndarray
        Upper CI bound.
    """
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    point_estimate = statistic_fn(data)

    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats.append(statistic_fn(data[idx]))
    boot_stats = np.array(boot_stats)

    alpha = 1 - ci
    ci_low = np.percentile(boot_stats, 100 * alpha / 2, axis=0)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2), axis=0)
    return point_estimate, ci_low, ci_high
