"""Dirichlet Mixture Model for simplex-valued data.

EM algorithm with fixed-point Dirichlet MLE (Minka 2000) for fitting
mixtures of Dirichlet distributions directly on the weight simplex,
without ILR transformation.

Reference: Minka (2000), "Estimating a Dirichlet distribution",
Technical Report, MIT.
"""

import numpy as np
from scipy.special import digamma, gammaln, polygamma
from sklearn.cluster import KMeans


class DirichletMixture:
    """Mixture of Dirichlet distributions fitted via EM.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.
    n_init : int
        Number of random restarts (best log-likelihood wins).
    random_state : int or None
        Random seed.
    alpha_max : float
        Upper bound on individual alpha parameters to prevent overflow.
    """

    def __init__(
        self,
        n_components=2,
        max_iter=200,
        tol=1e-6,
        n_init=1,
        random_state=None,
        alpha_max=1e4,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.alpha_max = alpha_max

        # Fitted parameters
        self.alphas_ = None  # [n_components, K]
        self.weights_ = None  # [n_components] mixing weights
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = -np.inf

    def fit(self, W):
        """Fit Dirichlet mixture to simplex data.

        Parameters
        ----------
        W : np.ndarray [n_samples, K]
            Rows sum to 1, all positive.

        Returns
        -------
        self
        """
        W = np.asarray(W, dtype=np.float64)
        # Clamp small values to avoid log(0)
        W = np.clip(W, 1e-300, None)
        W = W / W.sum(axis=1, keepdims=True)

        # Warn about vertex-heavy data
        vertex_fraction = np.mean(np.any(W < 1e-6, axis=1))
        if vertex_fraction > 0.1:
            import warnings
            warnings.warn(
                f"{vertex_fraction:.0%} of cells are near simplex vertices "
                f"(at least one weight < 1e-6). Dirichlet mixture fitting may "
                f"be unstable for boundary-heavy data. Consider using "
                f"model_type='gaussian' (GMM on ILR coordinates) instead.",
                UserWarning,
            )

        rng = np.random.default_rng(self.random_state)
        best_ll = -np.inf
        best_alphas = None
        best_weights = None
        best_converged = False
        best_n_iter = 0

        for init_idx in range(self.n_init):
            seed = int(rng.integers(0, 2**31))
            alphas, mix_weights, ll, converged, n_iter = self._fit_single(W, seed)
            if ll > best_ll:
                best_ll = ll
                best_alphas = alphas
                best_weights = mix_weights
                best_converged = converged
                best_n_iter = n_iter

        self.alphas_ = best_alphas
        self.weights_ = best_weights
        self.log_likelihood_ = best_ll
        self.converged_ = best_converged
        self.n_iter_ = best_n_iter
        return self

    def _fit_single(self, W, seed):
        """Single EM run."""
        n, K = W.shape
        rng = np.random.default_rng(seed)

        # Initialize via k-means on log(W + eps)
        log_W = np.log(np.clip(W, 1e-300, None))
        km = KMeans(n_clusters=self.n_components, n_init=1, random_state=seed)
        km.fit(log_W)
        labels = km.labels_

        # Moment-matching initialization for each component
        alphas = np.ones((self.n_components, K))
        mix_weights = np.ones(self.n_components) / self.n_components

        for c in range(self.n_components):
            mask = labels == c
            if mask.sum() < 2:
                # Fallback: uniform Dirichlet
                alphas[c] = np.ones(K)
                continue
            W_c = W[mask]
            mean_c = W_c.mean(axis=0)
            mean_c = np.clip(mean_c, 1e-10, None)
            mean_c = mean_c / mean_c.sum()
            var_c = W_c.var(axis=0)
            # Moment matching: s = mean*(1-mean)/var - 1, average over dimensions
            s_estimates = mean_c * (1 - mean_c) / np.clip(var_c, 1e-10, None) - 1
            s = np.clip(np.median(s_estimates), 0.1, self.alpha_max)
            alphas[c] = mean_c * s
            mix_weights[c] = mask.sum() / n

        mix_weights = np.clip(mix_weights, 1e-10, None)
        mix_weights /= mix_weights.sum()

        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            # E-step: compute responsibilities and log-likelihood together
            log_joint = self._log_joint(W, alphas, mix_weights)  # [n, C]
            log_joint_max = log_joint.max(axis=1, keepdims=True)
            log_sum_exp = log_joint_max.squeeze() + np.log(
                np.exp(log_joint - log_joint_max).sum(axis=1)
            )
            ll = log_sum_exp.sum()
            log_resp = log_joint - log_sum_exp[:, np.newaxis]
            resp = np.exp(log_resp)

            if abs(ll - prev_ll) < self.tol and iteration > 0:
                converged = True
                break
            prev_ll = ll

            # M-step
            N_c = resp.sum(axis=0)  # [n_components]
            mix_weights = N_c / n
            mix_weights = np.clip(mix_weights, 1e-10, None)
            mix_weights /= mix_weights.sum()

            # Update alphas via fixed-point iteration (Minka 2000)
            for c in range(self.n_components):
                if N_c[c] < 1e-10:
                    continue
                # Weighted sufficient statistics
                w_c = resp[:, c]
                w_c_sum = w_c.sum()
                log_W_bar = (w_c[:, None] * np.log(np.clip(W, 1e-300, None))).sum(axis=0) / w_c_sum

                alphas[c] = self._minka_update(alphas[c], log_W_bar)

        return alphas, mix_weights, prev_ll, converged, iteration + 1

    def _minka_update(self, alpha, log_x_bar, n_iter=50):
        """Fixed-point Dirichlet MLE update (Minka 2000).

        Parameters
        ----------
        alpha : np.ndarray [K]
            Current alpha parameters.
        log_x_bar : np.ndarray [K]
            Weighted mean of log(x) across samples.
        n_iter : int
            Number of fixed-point iterations.

        Returns
        -------
        np.ndarray [K]
            Updated alpha parameters.
        """
        alpha = np.clip(alpha, 1e-10, self.alpha_max)

        for _ in range(n_iter):
            alpha_sum = alpha.sum()
            # Minka's fixed-point update:
            # alpha_new_k = alpha_k * (digamma(alpha_sum) + log_x_bar_k) / digamma(alpha_k)
            # but we use the Newton-like update for better convergence:
            # alpha_new_k = alpha_k * (log_x_bar_k - digamma(alpha_k) + digamma(alpha_sum))
            # applied as: psi(alpha_new_k) = psi(alpha_sum) + log_x_bar_k
            # Solve via inverse digamma approximation

            # Use the simpler fixed-point iteration
            psi_sum = digamma(alpha_sum)
            numerator = psi_sum + log_x_bar
            # alpha_new s.t. digamma(alpha_new) = numerator
            alpha_new = self._inv_digamma(numerator)
            alpha_new = np.clip(alpha_new, 1e-10, self.alpha_max)

            if np.max(np.abs(alpha_new - alpha) / np.clip(alpha, 1e-10, None)) < 1e-8:
                break
            alpha = alpha_new

        return alpha

    @staticmethod
    def _inv_digamma(y, n_iter=5):
        """Inverse digamma function via Newton's method.

        Finds x such that digamma(x) = y.
        """
        # Initial estimate (Minka's approximation)
        x = np.where(
            y >= -2.22,
            np.exp(y) + 0.5,
            -1.0 / (y - digamma(1))
        )
        x = np.clip(x, 1e-10, None)

        for _ in range(n_iter):
            x = x - (digamma(x) - y) / polygamma(1, x)
            x = np.clip(x, 1e-10, None)
        return x

    def _log_dirichlet_pdf(self, W, alpha):
        """Log Dirichlet PDF for each sample.

        Parameters
        ----------
        W : np.ndarray [n, K]
        alpha : np.ndarray [K]

        Returns
        -------
        np.ndarray [n]
        """
        # log B(alpha) = sum(gammaln(alpha)) - gammaln(sum(alpha))
        log_B = gammaln(alpha).sum() - gammaln(alpha.sum())
        # log p(w|alpha) = -log_B + sum((alpha_k - 1) * log(w_k))
        log_p = -log_B + ((alpha - 1) * np.log(np.clip(W, 1e-300, None))).sum(axis=1)
        return log_p

    def _log_joint(self, W, alphas, mix_weights):
        """Log joint probabilities: log(pi_c * p(w|alpha_c)).

        Returns
        -------
        np.ndarray [n, n_components]
            Unnormalized log probabilities.
        """
        n = W.shape[0]
        log_joint = np.zeros((n, self.n_components))
        for c in range(self.n_components):
            log_joint[:, c] = (
                np.log(np.clip(mix_weights[c], 1e-300, None))
                + self._log_dirichlet_pdf(W, alphas[c])
            )
        return log_joint

    def _log_responsibilities(self, W, alphas, mix_weights):
        """Compute log responsibilities (E-step).

        Returns
        -------
        np.ndarray [n, n_components]
            Log posterior probabilities (normalized).
        """
        log_joint = self._log_joint(W, alphas, mix_weights)

        # Log-sum-exp normalization
        log_resp_max = log_joint.max(axis=1, keepdims=True)
        log_resp_norm = log_joint - log_resp_max - np.log(
            np.exp(log_joint - log_resp_max).sum(axis=1, keepdims=True)
        )
        return log_resp_norm

    def _log_likelihood(self, W, alphas, mix_weights):
        """Compute total log-likelihood."""
        log_joint = self._log_joint(W, alphas, mix_weights)

        # log-sum-exp per sample
        max_log = log_joint.max(axis=1)
        ll = max_log + np.log(np.exp(log_joint - max_log[:, None]).sum(axis=1))
        return ll.sum()

    def predict(self, W):
        """Predict component labels.

        Parameters
        ----------
        W : np.ndarray [n, K]

        Returns
        -------
        np.ndarray [n] int
        """
        W = np.asarray(W, dtype=np.float64)
        W = np.clip(W, 1e-300, None)
        W = W / W.sum(axis=1, keepdims=True)
        log_resp = self._log_responsibilities(W, self.alphas_, self.weights_)
        return np.argmax(log_resp, axis=1)

    def predict_proba(self, W):
        """Predict posterior probabilities.

        Parameters
        ----------
        W : np.ndarray [n, K]

        Returns
        -------
        np.ndarray [n, n_components]
        """
        W = np.asarray(W, dtype=np.float64)
        W = np.clip(W, 1e-300, None)
        W = W / W.sum(axis=1, keepdims=True)
        log_resp = self._log_responsibilities(W, self.alphas_, self.weights_)
        return np.exp(log_resp)

    def bic(self, W):
        """Bayesian Information Criterion.

        BIC = -2 * log_likelihood + n_params * log(n_samples)

        Parameters
        ----------
        W : np.ndarray [n, K]

        Returns
        -------
        float
        """
        W = np.asarray(W, dtype=np.float64)
        W = np.clip(W, 1e-300, None)
        W = W / W.sum(axis=1, keepdims=True)

        n = W.shape[0]
        K = W.shape[1]
        # Parameters: n_components * K alphas + (n_components - 1) mixing weights
        n_params = self.n_components * K + (self.n_components - 1)
        ll = self._log_likelihood(W, self.alphas_, self.weights_)
        return -2 * ll + n_params * np.log(n)

    @property
    def means_(self):
        """Component means on the simplex: alpha_k / sum(alpha_k).

        Returns
        -------
        np.ndarray [n_components, K]
        """
        alpha_sums = self.alphas_.sum(axis=1, keepdims=True)
        return self.alphas_ / alpha_sums
