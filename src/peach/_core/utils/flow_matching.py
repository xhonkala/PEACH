"""Flow matching: velocity network, training, transport, and Jacobian analysis.

Learns continuous transport maps between cell populations in PCA/latent space.
Uses conditional flow matching with linear interpolation paths:
    x_t = (1 - t) * x_0 + t * x_1

This implements the standard conditional flow matching objective directly,
without requiring external libraries. The ``_check_flow_matching()`` helper
is provided for future integration with Facebook's ``flow_matching`` library,
which offers more sophisticated probability paths (e.g., optimal transport
conditional paths), but is not currently called.

Main Classes
------------
VelocityNetwork : MLP mapping (x, t) -> v
FlowModel : Training, Euler transport, velocity evaluation, Jacobian

Utility Functions
-----------------
compute_mmd : Maximum Mean Discrepancy for evaluating transport quality

Dependencies
------------
- torch (already required by PEACH)
- scipy (already required by PEACH, used in MMD computation)
- flow_matching (optional, Facebook Research -- reserved for future use)
"""

import numpy as np
import torch
import torch.nn as nn


def _check_flow_matching():
    """Import flow_matching with helpful error on failure.

    Reserved for future use. The current implementation uses standard
    conditional flow matching with linear interpolation and does not
    require this library.
    """
    try:
        import flow_matching
        return flow_matching
    except ImportError:
        raise ImportError(
            "flow_matching is required for advanced flow analysis. "
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


class FlowModel:
    """Wraps velocity network with training and transport functionality.

    Implements conditional flow matching with linear interpolation paths.
    Given source distribution x_0 and target distribution x_1, learns a
    velocity field v(x, t) such that integrating from t=0 to t=1 transports
    source samples to target samples.

    Parameters
    ----------
    dim : int
        Dimensionality of the space (e.g., number of PCs).
    hidden_dims : tuple of int
        MLP hidden layer dimensions.
    lr : float
        Learning rate for Adam optimizer.
    device : str
        ``'cpu'`` or ``'cuda'``.
    """

    def __init__(self, dim, hidden_dims=(128, 128, 128), lr=1e-3, device="cpu"):
        self.dim = dim
        self.device = device
        self.velocity_net = VelocityNetwork(dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.velocity_net.parameters(), lr=lr)
        self._losses = []

    def train(self, source, target, n_epochs=1000, batch_size=256):
        """Train velocity field to transport source to target.

        Uses conditional flow matching: sample random time t, interpolate
        ``x_t = (1 - t) * x_0 + t * x_1``, learn ``v(x_t, t)`` to predict
        ``x_1 - x_0`` (the constant velocity along the linear path).

        Parameters
        ----------
        source : np.ndarray
            Source points, shape ``[n_source, dim]``.
        target : np.ndarray
            Target points, shape ``[n_target, dim]``.
        n_epochs : int
            Number of training epochs.
        batch_size : int
            Not used directly for batching (full random pairing each epoch),
            but controls the number of pairs sampled per epoch.

        Returns
        -------
        list of float
            Training losses per epoch.
        """
        source_t = torch.tensor(source, dtype=torch.float32, device=self.device)
        target_t = torch.tensor(target, dtype=torch.float32, device=self.device)

        self.velocity_net.train()
        self._losses = []

        for epoch in range(n_epochs):
            # Random pairing (sample with replacement to match sizes)
            n = min(len(source_t), len(target_t))
            idx_s = torch.randint(0, len(source_t), (n,), device=self.device)
            idx_t = torch.randint(0, len(target_t), (n,), device=self.device)
            x0 = source_t[idx_s]
            x1 = target_t[idx_t]

            # Random time
            t = torch.rand(n, 1, device=self.device)

            # Interpolation: x_t = (1-t)*x0 + t*x1
            x_t = (1 - t) * x0 + t * x1

            # Target velocity: x1 - x0 (constant along linear path)
            v_target = x1 - x0

            # Predicted velocity
            v_pred = self.velocity_net(x_t, t)

            # MSE loss
            loss = torch.mean((v_pred - v_target) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._losses.append(loss.item())

        self.velocity_net.eval()
        return self._losses

    def transport(self, x0, n_steps=50, return_trajectory=False):
        """Transport points from source to target via Euler integration.

        Integrates the learned velocity field from t=0 to t=1.

        Parameters
        ----------
        x0 : np.ndarray
            Starting points, shape ``[n_points, dim]``.
        n_steps : int
            Number of Euler integration steps.
        return_trajectory : bool
            If True, return the full trajectory with shape
            ``[n_steps + 1, n_points, dim]``.

        Returns
        -------
        np.ndarray
            Transported points ``[n_points, dim]``, or full trajectory
            ``[n_steps + 1, n_points, dim]`` if ``return_trajectory=True``.
        """
        self.velocity_net.eval()
        x = torch.tensor(x0, dtype=torch.float32, device=self.device)
        dt = 1.0 / n_steps

        trajectory = [x.detach().cpu().numpy()] if return_trajectory else None

        with torch.no_grad():
            for step in range(n_steps):
                t = torch.full((len(x), 1), step * dt, device=self.device)
                v = self.velocity_net(x, t)
                x = x + v * dt
                if return_trajectory:
                    trajectory.append(x.detach().cpu().numpy())

        if return_trajectory:
            return np.stack(trajectory, axis=0)
        return x.detach().cpu().numpy()

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


def compute_mmd(X, Y, bandwidth=None):
    """Compute Maximum Mean Discrepancy between two point sets.

    Uses an RBF (Gaussian) kernel. Useful for evaluating how well
    transport maps source distribution onto target distribution.

    Parameters
    ----------
    X : np.ndarray
        First point set, shape ``[n, dim]``.
    Y : np.ndarray
        Second point set, shape ``[m, dim]``.
    bandwidth : float or None
        RBF kernel bandwidth. If None, uses the median pairwise distance
        of a subsample (median heuristic).

    Returns
    -------
    float
        MMD^2 value. Near zero means distributions are similar.
    """
    from scipy.spatial.distance import cdist, pdist

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
