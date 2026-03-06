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
        ODE solver method for transport. 'euler' (default), 'midpoint',
        'heun3', or 'dopri5' (adaptive).
    device : str
        ``'cpu'`` or ``'cuda'``.
    """

    def __init__(self, dim, hidden_dims=(128, 128, 128), lr=1e-3,
                 solver_method="euler", device="cpu"):
        _check_flow_matching()

        self.dim = dim
        self.device = device
        self.solver_method = solver_method
        self.velocity_net = VelocityNetwork(dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.velocity_net.parameters(), lr=lr)
        self._losses = []

    def train(self, source, target, n_epochs=1000, batch_size=256):
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

        Returns
        -------
        list of float
            Training losses per epoch.
        """
        from flow_matching.path import CondOTProbPath

        prob_path = CondOTProbPath()

        source_t = torch.tensor(source, dtype=torch.float32, device=self.device)
        target_t = torch.tensor(target, dtype=torch.float32, device=self.device)

        self.velocity_net.train()
        self._losses = []

        for epoch in range(n_epochs):
            # Random pairing (sample with replacement to match sizes)
            n = min(len(source_t), len(target_t), batch_size)
            idx_s = torch.randint(0, len(source_t), (n,), device=self.device)
            idx_t = torch.randint(0, len(target_t), (n,), device=self.device)
            x0 = source_t[idx_s]
            x1 = target_t[idx_t]

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
            Number of integration steps (used as step_size=1/n_steps).
        return_trajectory : bool
            If True, return the full trajectory with shape
            ``[n_steps + 1, n_points, dim]``.

        Returns
        -------
        np.ndarray
            Transported points ``[n_points, dim]``, or full trajectory
            ``[n_steps + 1, n_points, dim]`` if ``return_trajectory=True``.
        """
        from flow_matching.solver import ODESolver

        self.velocity_net.eval()
        wrapper = _VelocityWrapper(self.velocity_net).to(self.device)
        solver = ODESolver(velocity_model=wrapper)

        x_init = torch.tensor(x0, dtype=torch.float32, device=self.device)
        step_size = 1.0 / n_steps

        if return_trajectory:
            # Create time grid for intermediates
            time_grid = torch.linspace(0, 1, n_steps + 1, device=self.device)
            with torch.no_grad():
                result = solver.sample(
                    x_init=x_init,
                    step_size=step_size,
                    method=self.solver_method,
                    time_grid=time_grid,
                    return_intermediates=True,
                )
            # result is a list of tensors at each time step
            trajectory = np.stack(
                [r.detach().cpu().numpy() for r in result], axis=0
            )
            return trajectory
        else:
            with torch.no_grad():
                result = solver.sample(
                    x_init=x_init,
                    step_size=step_size,
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

    def jacobian(self, x, t):
        """Compute Jacobian of velocity field dv/dx at each point.

        Uses ``torch.autograd`` to compute per-output-dimension gradients.
        This is O(n_points * dim) backward passes, so it will be slow for
        large datasets -- subsample first.

        Parameters
        ----------
        x : np.ndarray
            Points to evaluate at, shape ``[n_points, dim]``.
        t : float
            Time in [0, 1].

        Returns
        -------
        np.ndarray
            Jacobian matrices, shape ``[n_points, dim, dim]``.
            Entry ``[i, j, k]`` is ``dv_j/dx_k`` at point i.
        """
        self.velocity_net.eval()
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        t_scalar = t

        jacobians = []
        for i in range(len(x_t)):
            xi = x_t[i:i+1].clone().detach().requires_grad_(True)
            ti = torch.full((1, 1), t_scalar, device=self.device)
            v = self.velocity_net(xi, ti)  # [1, dim]
            jac = torch.zeros(self.dim, self.dim, device=self.device)
            for d in range(self.dim):
                if xi.grad is not None:
                    xi.grad.zero_()
                v[0, d].backward(retain_graph=True)
                jac[d] = xi.grad[0]
            jacobians.append(jac.detach().cpu().numpy())

        return np.stack(jacobians, axis=0)


def compute_mmd(X, Y, bandwidth=None, max_samples=5000):
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

    Returns
    -------
    float
        MMD^2 value. Near zero means distributions are similar.
    """
    from scipy.spatial.distance import cdist, pdist

    # Guard against degenerate inputs
    if len(X) < 2 or len(Y) < 2:
        return 0.0

    # Subsample large inputs to avoid O(n^2) kernel matrix OOM
    rng = np.random.default_rng(42)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    if bandwidth is None:
        XY = np.vstack([X, Y])
        bandwidth = np.median(pdist(XY[:min(500, len(XY))]))
        bandwidth = max(bandwidth, 1e-6)

    def rbf_kernel(A, B, bw):
        dists = cdist(A, B, 'sqeuclidean')
        return np.exp(-dists / (2 * bw ** 2))

    K_XX = rbf_kernel(X, X, bandwidth)
    K_YY = rbf_kernel(Y, Y, bandwidth)
    K_XY = rbf_kernel(X, Y, bandwidth)

    n = len(X)
    m = len(Y)
    mmd2 = K_XX.sum() / (n * n) + K_YY.sum() / (m * m) - 2 * K_XY.sum() / (n * m)
    return float(mmd2)
