"""Isometric Log-Ratio (ILR) transform for simplex compositions.

Maps K-dimensional simplex (weights summing to 1) to unconstrained R^{K-1}
using the Helmert sub-composition basis. Used by GMM decomposition and
driver regression modules.

Zero handling: weights are smoothed with epsilon=1e-3 before log transform
to avoid -inf values at simplex vertices. This is documented behavior --
cells at exact vertices are slightly pulled inward.

Reference: Egozcue et al. (2003), "Isometric Logratio Transformations for
Compositional Data Analysis", Mathematical Geology 35(3).
"""

import numpy as np

ILR_EPSILON = 1e-3


def _helmert_basis(K):
    """Construct Helmert sub-composition basis matrix.

    The Helmert basis is one of several valid orthonormal bases for ILR.
    Choice doesn't matter for full-covariance GMM since it is rotation-
    invariant.

    Parameters
    ----------
    K : int
        Number of components (simplex dimension).

    Returns
    -------
    np.ndarray [K, K-1]
        Orthonormal contrast matrix for ILR transform.
        Each column j has:
          rows 0..j:   1 / sqrt((j+1)(j+2))
          row j+1:     -(j+1) / sqrt((j+1)(j+2))
          rows j+2..:  0
    """
    V = np.zeros((K, K - 1))
    for j in range(K - 1):
        scale = np.sqrt((j + 1) * (j + 2))
        V[:j + 1, j] = 1.0 / scale
        V[j + 1, j] = -(j + 1) / scale
    return V


def ilr_transform(W, epsilon=ILR_EPSILON):
    """Transform simplex compositions to ILR coordinates.

    Parameters
    ----------
    W : np.ndarray [n, K]
        Simplex compositions (rows sum to 1). K >= 2.
    epsilon : float
        Smoothing constant for zero weights. Added before log, then
        renormalized. Default 1e-3 avoids extreme outliers near vertices
        (log(1e-10) ~ -23 would distort GMM fitting).

    Returns
    -------
    np.ndarray [n, K-1]
        ILR coordinates in unconstrained R^{K-1}.
    """
    W = np.asarray(W, dtype=np.float64)
    K = W.shape[1]

    # Zero smoothing: add epsilon, renormalize
    W_smooth = W + epsilon
    W_smooth = W_smooth / W_smooth.sum(axis=1, keepdims=True)

    # CLR transform: log then center
    log_W = np.log(W_smooth)
    clr = log_W - log_W.mean(axis=1, keepdims=True)

    # Project onto Helmert basis to get ILR coordinates
    V = _helmert_basis(K)  # [K, K-1]
    ilr_coords = clr @ V  # [n, K-1]

    return ilr_coords


def inverse_ilr(ilr_coords):
    """Transform ILR coordinates back to simplex compositions.

    Parameters
    ----------
    ilr_coords : np.ndarray [n, K-1]
        ILR coordinates in R^{K-1}.

    Returns
    -------
    np.ndarray [n, K]
        Simplex compositions (rows sum to 1, all positive).
    """
    ilr_coords = np.asarray(ilr_coords, dtype=np.float64)
    K = ilr_coords.shape[1] + 1

    V = _helmert_basis(K)  # [K, K-1]

    # ILR -> CLR: project back via V^T
    clr = ilr_coords @ V.T  # [n, K]

    # CLR -> composition: exp and normalize
    W = np.exp(clr)
    W = W / W.sum(axis=1, keepdims=True)

    return W
