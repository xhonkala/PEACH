"""
scATAC-seq Preprocessing: TF-IDF Normalization and LSI
======================================================

Core functions for processing scATAC-seq peak count matrices into
dimensionality-reduced embeddings suitable for archetypal analysis.

Pipeline: raw peaks → TF-IDF normalization → Truncated SVD (LSI) → embeddings

Main Functions
--------------
tfidf_normalize : TF-IDF normalize a sparse peak count matrix
compute_lsi : Truncated SVD on TF-IDF matrix, producing LSI embeddings

Notes
-----
- First LSI component typically captures sequencing depth, not biology — drop it by default.
- scATAC-seq matrices are ~98% sparse; all operations preserve sparsity where possible.
- Uses only existing dependencies (sklearn, scipy, numpy).
"""

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def tfidf_normalize(X, log_tf=True):
    """TF-IDF normalize a sparse peak count matrix.

    Computes term frequency-inverse document frequency, the standard
    normalization for scATAC-seq data before dimensionality reduction.

    Parameters
    ----------
    X : scipy.sparse matrix or numpy.ndarray
        Peak count matrix [n_cells, n_peaks]. Can be binary or integer counts.
    log_tf : bool, default: True
        If True, use log(1 + TF) instead of raw TF. Reduces influence of
        high-count peaks and is the standard in scATAC-seq pipelines.

    Returns
    -------
    scipy.sparse.csr_matrix
        TF-IDF normalized matrix [n_cells, n_peaks], sparse CSR format.
    """
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    # Term frequency: normalize each cell by its total counts
    row_sums = np.asarray(X.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # avoid division by zero for empty cells

    # Sparse row-wise division
    row_inv = sp.diags(1.0 / row_sums)
    tf = row_inv @ X

    if log_tf:
        # log(1 + TF) — standard scATAC-seq variant
        tf = tf.log1p() if hasattr(tf, 'log1p') else sp.csr_matrix(np.log1p(tf.toarray()))

    # Inverse document frequency: upweight peaks found in fewer cells
    n_cells = X.shape[0]
    col_sums = np.asarray(X.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1  # avoid division by zero for empty peaks
    idf = np.log1p(n_cells / col_sums)

    # TF-IDF = TF * IDF (broadcast IDF across rows)
    tfidf = tf @ sp.diags(idf)

    return sp.csr_matrix(tfidf)


def compute_lsi(X_tfidf, n_components=50, drop_first=True, random_state=42):
    """Compute LSI (Latent Semantic Indexing) via Truncated SVD.

    Parameters
    ----------
    X_tfidf : scipy.sparse matrix or numpy.ndarray
        TF-IDF normalized matrix [n_cells, n_peaks].
    n_components : int, default: 50
        Number of LSI components to compute. If drop_first=True, computes
        n_components+1 and drops the first, returning exactly n_components.
    drop_first : bool, default: True
        Drop first SVD component, which typically captures sequencing depth
        rather than biological signal in scATAC-seq data.
    random_state : int, default: 42
        Random seed for reproducibility.

    Returns
    -------
    embeddings : numpy.ndarray
        LSI embeddings [n_cells, n_components].
    variance_ratio : numpy.ndarray
        Explained variance ratio for each returned component.
    components : numpy.ndarray
        Feature loadings [n_components, n_peaks].
    """
    n_compute = n_components + 1 if drop_first else n_components

    # Cap at matrix dimensions
    max_components = min(X_tfidf.shape) - 1
    if n_compute > max_components:
        n_compute = max_components
        print(f"  Capped LSI components at {n_compute} (matrix rank limit)")

    svd = TruncatedSVD(n_components=n_compute, random_state=random_state)
    embeddings = svd.fit_transform(X_tfidf)
    variance_ratio = svd.explained_variance_ratio_
    components = svd.components_

    if drop_first:
        embeddings = embeddings[:, 1:]
        variance_ratio = variance_ratio[1:]
        components = components[1:]

    return embeddings, variance_ratio, components
