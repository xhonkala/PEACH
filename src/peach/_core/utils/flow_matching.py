"""Flow matching: velocity network, training, transport, and Jacobian analysis.

Learns continuous transport maps between cell populations in PCA/latent space
using Facebook Research's ``flow_matching`` library for conditional flow matching
with linear interpolation (conditional optimal transport) paths.

Main Classes
------------
VelocityNetwork : MLP mapping (x, t) -> v
FlowModel : Training via CondOTProbPath, ODESolver transport, velocity evaluation, Jacobian

Utility Functions
-----------------
compute_mmd : Maximum Mean Discrepancy for evaluating transport quality

Dependencies
------------
- torch (already required by PEACH)
- flow_matching (Facebook Research) — required for flow analysis
- scipy (already required by PEACH, used in MMD computation)

References
----------
Lipman et al. (2023). "Flow Matching for Generative Modeling." ICLR 2023.
https://github.com/facebookresearch/flow_matching
"""

import warnings

import numpy as np
import torch
import torch.nn as nn


def _check_flow_matching():
    """Import flow_matching with helpful error on failure."""
    try:
        import flow_matching
        return flow_matching
    except ImportError:
        raise ImportError(
            "flow_matching is required for flow analysis. "
            "Install it with: pip install flow-matching"
        )


class VelocityNetwork(nn.Module):
    """MLP velocity field: (x, t) -> v.

    Concatenates position x and scalar time t, passes through an MLP.
    Output has the same dimension as x.

    Parameters
    ----------
    dim : int
        Dimensionality of the input/output space.
    hidden_dims : tuple of int
        Sizes of hidden layers.
    """

    def __init__(self, dim, hidden_dims=(128, 128, 128)):
        super().__init__()
        layers = []
        in_dim = dim + 1  # position + time
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Positions, shape ``[batch, dim]``.
        t : torch.Tensor
            Times, shape ``[batch, 1]`` or ``[batch]``.

        Returns
        -------
        torch.Tensor
            Velocity vectors, shape ``[batch, dim]``.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


class _VelocityWrapper(nn.Module):
    """Wraps VelocityNetwork for fb ODESolver compatibility.

    ODESolver expects a callable(x, t) where t has the same batch dim as x.
    Our VelocityNetwork already handles this, so this is a thin adapter.
    """

    def __init__(self, velocity_net):
        super().__init__()
        self.velocity_net = velocity_net

    def forward(self, x, t, **extras):
        # ODESolver passes t as shape [batch] or scalar; ensure [batch, 1]
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.velocity_net(x, t)


class FlowModel:
    """Wraps velocity network with fb flow_matching training and transport.

    Uses ``CondOTProbPath`` for conditional optimal transport path sampling
    during training, and ``ODESolver`` for integration during transport.

    Parameters
    ----------
    dim : int
        Dimensionality of the space (e.g., number of PCs).
    hidden_dims : tuple of int
        MLP hidden layer dimensions.
    lr : float
        Learning rate for Adam optimizer.
    solver_method : str
        ODE solver method for transport. 'dopri5' (default, adaptive),
        'euler', 'midpoint', or 'heun3'.
    device : str
        ``'cpu'`` or ``'cuda'``.
    random_state : int or None
        If provided, seeds ``torch.manual_seed`` before network weight
        initialization for full reproducibility.
    """

    def __init__(self, dim, hidden_dims=(128, 128, 128), lr=1e-3,
                 solver_method="dopri5", device="cpu", random_state=None):
        _check_flow_matching()

        # Seed torch before weight initialization for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)

        self.dim = dim
        self.device = device
        self.solver_method = solver_method
        self.velocity_net = VelocityNetwork(dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.velocity_net.parameters(), lr=lr)
        self._losses = []
        self._trained = False

    def train(self, source, target, n_epochs=1000, batch_size=256, use_ot=False, ot_reg=0.1, random_state=None):
        """Train velocity field to transport source to target.

        Uses fb ``CondOTProbPath`` for conditional optimal transport path
        sampling: ``x_t = (1-t)*x_0 + t*x_1``, learns ``v(x_t, t)`` to
        predict the conditional velocity ``dx_t = x_1 - x_0``.

        Parameters
        ----------
        source : np.ndarray
            Source points, shape ``[n_source, dim]``.
        target : np.ndarray
            Target points, shape ``[n_target, dim]``.
        n_epochs : int
            Number of training epochs.
        batch_size : int
            Number of pairs sampled per epoch.
        use_ot : bool
            If True, use minibatch Sinkhorn optimal transport coupling
            to pair source and target samples within each epoch instead
            of random pairing. Requires the ``POT`` package.
        ot_reg : float
            Entropic regularization for Sinkhorn OT coupling. Only used
            when ``use_ot=True``. Smaller values give sparser (truer OT)
            coupling; larger values give smoother coupling. Default: 0.1.
        random_state : int or None
            Seed for reproducibility. Seeds both torch and numpy RNGs
            used during training.

        Returns
        -------
        list of float
            Training losses per epoch.
        """
        from flow_matching.path import CondOTProbPath

        prob_path = CondOTProbPath()

        # Seed both RNGs for full reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
        ot_rng = np.random.default_rng(random_state)

        if source.ndim != 2 or source.shape[1] != self.dim:
            raise ValueError(f"source has shape {source.shape}, expected (n, {self.dim})")
        if target.ndim != 2 or target.shape[1] != self.dim:
            raise ValueError(f"target has shape {target.shape}, expected (n, {self.dim})")

        source_t = torch.tensor(source, dtype=torch.float32, device=self.device)
        target_t = torch.tensor(target, dtype=torch.float32, device=self.device)

        ratio = max(len(source_t), len(target_t)) / min(len(source_t), len(target_t))
        if ratio > 5:
            warnings.warn(
                f"Source/target size imbalance: {len(source_t)} vs {len(target_t)} "
                f"(ratio {ratio:.1f}x). The smaller population will be heavily "
                f"resampled during training, which may degrade flow quality. "
                f"Consider subsampling the larger population.",
                UserWarning,
            )

        if use_ot:
            try:
                import ot as pot
            except ImportError:
                raise ImportError(
                    "POT (Python Optimal Transport) is required for OT-CFM. "
                    "Install with: pip install POT"
                )

        self.velocity_net.train()
        self._losses = []

        for epoch in range(n_epochs):
            # Sample with replacement — replacement handles any population size
            n = batch_size
            idx_s = torch.randint(0, len(source_t), (n,), device=self.device)
            idx_t = torch.randint(0, len(target_t), (n,), device=self.device)
            x0 = source_t[idx_s]
            x1 = target_t[idx_t]

            # OT-CFM: reshuffle pairings via Sinkhorn coupling
            if use_ot:
                cost = torch.cdist(x0, x1).detach().cpu().numpy()
                coupling = pot.sinkhorn(
                    np.ones(len(x0)) / len(x0),
                    np.ones(len(x1)) / len(x1),
                    cost,
                    reg=ot_reg,
                )
                coupling_flat = coupling.ravel()
                coupling_flat /= coupling_flat.sum()
                pair_idx = ot_rng.choice(
                    len(x0) * len(x1), size=len(x0), p=coupling_flat
                )
                idx_i = pair_idx // len(x1)
                idx_j = pair_idx % len(x1)
                x0 = x0[idx_i]
                x1 = x1[idx_j]

            # Random time
            t = torch.rand(n, device=self.device)

            # fb library: sample conditional path
            path_sample = prob_path.sample(x_0=x0, x_1=x1, t=t)
            # path_sample.x_t: interpolated point
            # path_sample.dx_t: target velocity (x_1 - x_0 for CondOT)

            # Predicted velocity
            v_pred = self.velocity_net(path_sample.x_t, t)

            # MSE loss against conditional velocity
            loss = torch.mean((v_pred - path_sample.dx_t) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._losses.append(loss.item())

        self.velocity_net.eval()
        self._trained = True
        return self._losses

    def transport(self, x0, n_steps=50, return_trajectory=False):
        """Transport points from source to target via ODE integration.

        Uses fb ``ODESolver`` to integrate the learned velocity field
        from t=0 to t=1.

        Parameters
        ----------
        x0 : np.ndarray
            Starting points, shape ``[n_points, dim]``.
        n_steps : int
            Number of integration steps (used as step_size=1/n_steps for
            fixed-step solvers). Ignored for adaptive solvers (``dopri5``
            etc.); use ``atol``/``rtol`` to control those instead.
        return_trajectory : bool
            If True, return the full trajectory with shape
            ``[n_steps + 1, n_points, dim]``.

        Returns
        -------
        np.ndarray
            Transported points ``[n_points, dim]``, or full trajectory
            ``[n_steps + 1, n_points, dim]`` if ``return_trajectory=True``.
        """
        if not self._trained:
            raise RuntimeError("Call train() before transport().")

        _known_adaptive = {"dopri5", "dopri8", "bosh3", "adaptive_heun"}
        _known_fixed = {"euler", "midpoint", "heun3", "rk4"}
        if self.solver_method not in _known_adaptive and self.solver_method not in _known_fixed:
            warnings.warn(
                f"Unrecognized solver_method '{self.solver_method}'. "
                f"Expected one of {sorted(_known_adaptive | _known_fixed)}. "
                "Treating as fixed-step.",
                UserWarning,
            )

        from flow_matching.solver import ODESolver

        self.velocity_net.eval()
        wrapper = _VelocityWrapper(self.velocity_net).to(self.device)
        solver = ODESolver(velocity_model=wrapper)

        x_init = torch.tensor(x0, dtype=torch.float32, device=self.device)

        # Adaptive solvers (dopri5 etc.) use atol/rtol, not step_size.
        # ODESolver.sample() requires step_size but accepts None for adaptive methods.
        _adaptive = self.solver_method in _known_adaptive
        _step_size = None if _adaptive else 1.0 / n_steps

        if return_trajectory:
            time_grid = torch.linspace(0, 1, n_steps + 1, device=self.device)
            with torch.no_grad():
                result = solver.sample(
                    x_init=x_init,
                    step_size=_step_size,
                    method=self.solver_method,
                    time_grid=time_grid,
                    return_intermediates=True,
                )
            trajectory = np.stack(
                [r.detach().cpu().numpy() for r in result], axis=0
            )
            return trajectory
        else:
            with torch.no_grad():
                result = solver.sample(
                    x_init=x_init,
                    step_size=_step_size,
                    method=self.solver_method,
                )
            return result.detach().cpu().numpy()

    def velocity_at(self, x, t):
        """Evaluate learned velocity field at given points and time.

        Parameters
        ----------
        x : np.ndarray
            Points to evaluate at, shape ``[n_points, dim]``.
        t : float
            Time in [0, 1].

        Returns
        -------
        np.ndarray
            Velocity vectors, shape ``[n_points, dim]``.
        """
        self.velocity_net.eval()
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        t_t = torch.full((len(x_t), 1), t, device=self.device)
        with torch.no_grad():
            v = self.velocity_net(x_t, t_t)
        return v.detach().cpu().numpy()

    def _velocity_jacobian_tensor(self, phi, t_scalar):
        """Compute dv/dx at given positions and time, returning a Tensor.

        Internal helper used by both ``jacobian()`` and ``flow_map_jacobian()``.
        Requires autograd to be active (no torch.no_grad wrapper).

        Parameters
        ----------
        phi : torch.Tensor
            Cell positions, shape [n_cells, dim]. May be detached.
        t_scalar : torch.Tensor
            Scalar time value.

        Returns
        -------
        torch.Tensor
            Velocity Jacobians dv/dx, shape [n_cells, dim, dim].
        """
        from torch.func import jacrev, vmap, functional_call

        params = dict(self.velocity_net.named_parameters())
        buffers = dict(self.velocity_net.named_buffers())

        def vel_fn(params_dict, x_single):
            x_2d = x_single.unsqueeze(0)
            t_2d = t_scalar.reshape(1, 1)
            out = functional_call(self.velocity_net, (params_dict, buffers), (x_2d, t_2d))
            return out.squeeze(0)

        batched_jac = vmap(jacrev(vel_fn, argnums=1), in_dims=(None, 0))
        return batched_jac(params, phi)  # [n_cells, dim, dim]

    def jacobian(self, x, t):
        """Compute velocity Jacobian dv/dx at given positions and time.

        Evaluates the Jacobian of the velocity field at fixed (x, t).
        Used by ``flow_bifurcation`` for divergence/eigenvalue analysis.
        For flow map Jacobians ∂φ_t/∂x₀ (accumulated stretching over the
        full trajectory), use ``flow_map_jacobian`` instead.

        Parameters
        ----------
        x : np.ndarray
            Points to evaluate at, shape [n_points, dim].
        t : float
            Time in [0, 1].

        Returns
        -------
        np.ndarray
            Jacobian matrices dv/dx, shape [n_points, dim, dim].
            Entry [i, j, k] is dv_j/dx_k at point i.
        """
        if not self._trained:
            raise RuntimeError("Call train() before jacobian().")
        self.velocity_net.eval()
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        t_scalar = torch.tensor(t, dtype=torch.float32, device=self.device)
        jacs = self._velocity_jacobian_tensor(x_t, t_scalar)
        return jacs.detach().cpu().numpy()

    def flow_map_jacobian(self, x0, t_eval, n_steps=100):
        """Compute flow map Jacobian ∂φ_t/∂x₀ via coupled ODE integration.

        Integrates the augmented system:

            dφ/dt = v(φ(t), t)               [position ODE]
            dJ/dt = (∂v/∂x)|_{φ(t)} @ J      [variational ODE]

        starting from φ(0) = x0, J(0) = I. At each requested time t, J(t)
        gives the flow map Jacobian for each cell — how a neighborhood around
        x0 is stretched, compressed, or rotated by the full transport up to t.

        **Solver note**: Uses torchdiffeq midpoint (fixed-step RK2) internally.
        This differs from the dopri5 adaptive solver used by ``transport()``.
        Autograd nesting prevents dopri5 when ``jacrev`` is called inside the
        ODE RHS. At the default n_steps=100 the discretization error in φ at
        t=1 is small but non-zero relative to ``flow_result['transported']``
        (which was integrated with dopri5). Do not expect exact agreement
        between ``phi`` from this method and the transported array.

        Parameters
        ----------
        x0 : np.ndarray
            Source cell positions, shape [n_cells, dim].
        t_eval : list of float
            Time points in (0, 1] at which to return results. Must be
            increasing and non-empty. Do not include 0.0.
        n_steps : int
            Number of midpoint integration steps over [0, 1]. The step
            size is 1/n_steps. Default: 100.

        Returns
        -------
        dict with keys:

        - ``phi`` : np.ndarray [n_t, n_cells, dim] — transported positions
          at each requested time.
        - ``J`` : np.ndarray [n_t, n_cells, dim, dim] — flow map Jacobians
          ∂φ_t/∂x₀ at each requested time.
        - ``t_eval`` : list of float — requested time points (sorted).
        """
        try:
            import torchdiffeq
        except ImportError:
            raise ImportError(
                "torchdiffeq is required for flow_map_jacobian. "
                "Install with: pip install torchdiffeq"
            )

        self.velocity_net.eval()
        n_cells, d = x0.shape
        phi0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
        # J(0) = I for each cell
        J0 = torch.eye(d, device=self.device).unsqueeze(0).expand(n_cells, -1, -1).clone()

        def augmented_rhs(t, state):
            phi, J_mat = state  # phi: [n_cells, d], J_mat: [n_cells, d, d]
            t_scalar = t.squeeze() if t.dim() > 0 else t

            # Velocity for all cells
            t_batch = t_scalar.reshape(1).expand(n_cells).unsqueeze(-1)  # [n_cells, 1]
            v = self.velocity_net(phi, t_batch)  # [n_cells, d]

            # dv/dx at current transported positions.
            # Detach phi so the Jacobian computation doesn't propagate
            # gradients back through the torchdiffeq integration history.
            dv_dx = self._velocity_jacobian_tensor(phi.detach(), t_scalar.detach())

            # Variational equation: J_dot = dv_dx @ J
            J_dot = torch.einsum('...ij,...jk->...ik', dv_dx, J_mat)
            return v, J_dot

        t_sorted = sorted(set(t_eval))
        if not t_sorted or any(t <= 0.0 or t > 1.0 for t in t_sorted):
            raise ValueError(
                "t_eval must be non-empty and all values must be in (0, 1]. "
                f"Got: {t_eval}"
            )
        # torchdiffeq evaluates at all times in t_span; prepend 0.0
        t_span = torch.tensor([0.0] + t_sorted, dtype=torch.float32, device=self.device)

        phi_traj, J_traj = torchdiffeq.odeint(
            augmented_rhs,
            (phi0, J0),
            t_span,
            method='midpoint',
            options={'step_size': 1.0 / n_steps},
        )
        # phi_traj: [len(t_span), n_cells, d]
        # J_traj:   [len(t_span), n_cells, d, d]
        # Index [0] is t=0 (initial condition); drop it.
        return {
            'phi': phi_traj[1:].detach().cpu().numpy(),
            'J': J_traj[1:].detach().cpu().numpy(),
            't_eval': t_sorted,
        }


def compute_mmd(X, Y, bandwidth=None, max_samples=5000, random_state=None):
    """Compute Maximum Mean Discrepancy between two point sets.

    Uses an RBF (Gaussian) kernel. Useful for evaluating how well
    transport maps source distribution onto target distribution.

    For large inputs (>max_samples), subsamples to avoid O(n^2) memory.

    Parameters
    ----------
    X : np.ndarray
        First point set, shape ``[n, dim]``.
    Y : np.ndarray
        Second point set, shape ``[m, dim]``.
    bandwidth : float or None
        RBF kernel bandwidth. If None, uses the median pairwise distance
        of a subsample (median heuristic).
    max_samples : int
        Maximum number of points per set. Subsamples if exceeded.
    random_state : int or None
        Seed for the subsampling RNG. If None, results may vary across
        calls when subsampling occurs.

    Returns
    -------
    float
        MMD^2 value. Near zero means distributions are similar.
    """
    from scipy.spatial.distance import cdist, pdist

    # Guard against degenerate inputs
    if len(X) < 2 or len(Y) < 2:
        return float("nan")

    # Subsample large inputs to avoid O(n^2) kernel matrix OOM
    rng = np.random.default_rng(random_state)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    if bandwidth is None:
        XY = np.vstack([X, Y])
        bandwidth = np.median(pdist(XY[:min(500, len(XY))]))
        # Scale by 1/sqrt(n_dims): heuristic correction for bandwidth concentration in high
        # dimensions (not canonical; Gretton et al. recommend alternatives for d > 20).
        bandwidth /= max(np.sqrt(XY.shape[1]), 1.0)
        bandwidth = max(bandwidth, 1e-6)

    def rbf_kernel(A, B, bw):
        dists = cdist(A, B, 'sqeuclidean')
        return np.exp(-dists / (2 * bw ** 2))

    K_XX = rbf_kernel(X, X, bandwidth)
    K_YY = rbf_kernel(Y, Y, bandwidth)
    K_XY = rbf_kernel(X, Y, bandwidth)

    n = len(X)
    m = len(Y)
    # Unbiased estimator: exclude diagonal (self-similarity) terms
    mmd2 = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) \
         + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) \
         - 2 * K_XY.sum() / (n * m)
    return float(mmd2)
