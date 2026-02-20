"""
VAE Base Class
==============

Foundational Variational Autoencoder implementation without archetypal constraints.

This module provides the base VAE architecture that Deep_AA extends.
It implements standard VAE components: encoder, decoder, reparameterization
trick, and ELBO loss.

Main Classes
------------
VAE_Base : Standard VAE with encoder-decoder architecture

Type Definitions
----------------
See ``peach._core.types`` for Pydantic models:

- ``VAEConfig`` : Configuration parameters
- ``VAEForwardOutput`` : Documents forward() return list
- ``VAELossOutput`` : loss_function() return dict

Examples
--------
>>> from peach._core.models.VAE_Base import VAE_Base
>>> model = VAE_Base(input_dim=30, latent_dim=10, hidden_dims=[128, 64, 32])
>>> # Forward pass
>>> recons, input, mu, log_var = model(data)
>>> # Compute loss
>>> loss_dict = model.loss_function(recons, input, mu, log_var, kld_weight=1.0)
>>> loss_dict["loss"].backward()
"""

import torch
from torch import nn
from torch.nn import functional as F


class VAE_Base(nn.Module):
    """Standard Variational Autoencoder base class.

    Implements encoder-decoder architecture with reparameterization trick
    for differentiable sampling. This class is extended by Deep_AA to add
    archetypal constraints.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    latent_dim : int
        Latent space dimension.
    hidden_dims : list[int] | None, default: None
        Hidden layer sizes for encoder/decoder.
        If None, defaults to [128, 64, 32].
    n_archetypes : int | None, default: None
        Number of archetypes (not used in base class, passed to subclasses).
    archetypal_weights : float, default: 1.0
        Archetypal loss weight (not used in base class).
    latent_size : int | None, default: None
        Alias for latent_dim (deprecated).

    Attributes
    ----------
    input_dim : int
        Stored input dimension.
    latent_dim : int
        Stored latent dimension.
    encoder : nn.Sequential
        Encoder network (input_dim → hidden_dims → bottleneck).
    decoder : nn.Sequential
        Decoder network (bottleneck → hidden_dims reversed → output).
    fc_mu : nn.Linear
        Linear layer for latent mean.
    fc_var : nn.Linear
        Linear layer for latent log variance.

    Examples
    --------
    >>> model = VAE_Base(input_dim=30, latent_dim=10)
    >>> # Training loop
    >>> for batch in dataloader:
    ...     outputs = model(batch)
    ...     loss_dict = model.loss_function(*outputs, kld_weight=1.0)
    ...     loss_dict["loss"].backward()
    ...     optimizer.step()

    See Also
    --------
    Deep_AA : Archetypal analysis extension
    peach._core.types.VAEConfig : Configuration model
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = None,
        n_archetypes: int = None,
        archetypal_weights: float = 1.0,
        latent_size: int = None,
        **kwargs,
    ) -> None:
        """Initialize VAE architecture.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        latent_dim : int
            Latent space dimension.
        hidden_dims : list[int] | None
            Hidden layer sizes. Default: [128, 64, 32].
        n_archetypes : int | None
            Number of archetypes (for subclass use).
        archetypal_weights : float
            Archetypal loss weight (for subclass use).
        latent_size : int | None
            Deprecated alias for latent_dim.
        **kwargs
            Additional arguments (ignored).
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_archetypes = n_archetypes
        self.archetypal_weight = archetypal_weights
        self.latent_size = latent_size

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]  # sequential

        # build encoder
        modules = []
        in_features = input_dim

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),  # not clear these are the best layers to use for base class
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # build decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims = hidden_dims[::-1]  # mirror hinge

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.BatchNorm1d(hidden_dims[i + 1]), nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)  # *modules unpacks

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], input_dim)  # ,
            # nn.Sigmoid() # modify based on data distribution
        )

    def encode(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Encode input to latent space parameters.

        Parameters
        ----------
        input : torch.Tensor
            Input data [batch_size, input_dim].

        Returns
        -------
        list[torch.Tensor]
            Two-element list [mu, log_var]:

            - ``mu`` : Latent mean [batch_size, latent_dim]
            - ``log_var`` : Latent log variance [batch_size, latent_dim]

        Examples
        --------
        >>> mu, log_var = model.encode(data)
        >>> print(f"Latent shape: {mu.shape}")
        """
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
        # no return of the latent space?

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent variables to input space.

        Parameters
        ----------
        z : torch.Tensor
            Latent variables [batch_size, latent_dim].

        Returns
        -------
        torch.Tensor
            Reconstructed data [batch_size, input_dim].
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        return self.final_layer(result)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick for differentiable sampling.

        Samples z ~ N(mu, exp(logvar)) using the reparameterization trick:
        z = mu + exp(0.5 * logvar) * epsilon, where epsilon ~ N(0, 1).

        Parameters
        ----------
        mu : torch.Tensor
            Latent mean [batch_size, latent_dim].
        logvar : torch.Tensor
            Latent log variance [batch_size, latent_dim].

        Returns
        -------
        torch.Tensor
            Sampled latent variables [batch_size, latent_dim].

        Notes
        -----
        The reparameterization trick allows gradients to flow through
        the sampling operation by expressing the random sample as a
        deterministic function of the parameters plus noise.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Complete VAE forward pass.

        Encodes input to latent parameters, samples using reparameterization,
        and decodes back to input space.

        Parameters
        ----------
        input : torch.Tensor
            Input data [batch_size, input_dim].

        Returns
        -------
        list[torch.Tensor]
            Four-element list [recons, input, mu, log_var]:

            - ``recons`` : Reconstructed data [batch_size, input_dim]
            - ``input`` : Original input (passed through for loss)
            - ``mu`` : Latent mean [batch_size, latent_dim]
            - ``log_var`` : Latent log variance [batch_size, latent_dim]

        Examples
        --------
        >>> recons, input, mu, log_var = model(data)
        >>> loss_dict = model.loss_function(recons, input, mu, log_var)

        See Also
        --------
        peach._core.types.VAEForwardOutput : Documents return structure
        """
        mu, log_var = self.encode(input)  # actual latent space encoding step?
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)  # reconstruction on sampled latent vector z
        return [recons, input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Compute VAE ELBO loss.

        Computes reconstruction loss (MSE) plus KL divergence from
        standard normal prior.

        Parameters
        ----------
        *args : torch.Tensor
            Positional arguments in order: [recons, input, mu, log_var].
            Typically the output of forward().
        **kwargs
            kld_weight : float, default: 1.0
                Weight for KL divergence term (beta in beta-VAE).

        Returns
        -------
        dict[str, torch.Tensor]
            Loss components:

            - ``loss`` : Total loss (requires grad)
            - ``reconstruction_loss`` : MSE loss (detached)
            - ``KLD`` : KL divergence (detached)

        Examples
        --------
        >>> outputs = model(data)
        >>> loss_dict = model.loss_function(*outputs, kld_weight=1.0)
        >>> loss_dict["loss"].backward()
        >>> # Monitor components
        >>> print(f"Recon: {loss_dict['reconstruction_loss'].item():.4f}")
        >>> print(f"KLD: {loss_dict['KLD'].item():.4f}")

        See Also
        --------
        peach._core.types.VAELossOutput : Return type structure
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # reconstruction loss
        recons_loss = F.mse_loss(recons, input)  # sweet, PyTorch has an mse built-in

        # KLD
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1))

        # total loss
        loss = recons_loss + kwargs.get("kld_weight", 1.0) * kld_loss

        return {"loss": loss, "reconstruction_loss": recons_loss.detach(), "KLD": kld_loss.detach()}

    def get_metrics(
        self, recons: torch.Tensor, input: torch.Tensor, mu: torch.Tensor = None, log_var: torch.Tensor = None
    ) -> dict[str, float]:
        """Calculate monitoring metrics.

        Parameters
        ----------
        recons : torch.Tensor
            Reconstructed data.
        input : torch.Tensor
            Original input data.
        mu : torch.Tensor | None
            Latent mean. If None, computed via encode().
        log_var : torch.Tensor | None
            Latent log variance. If None, computed via encode().

        Returns
        -------
        dict[str, float]
            Metrics from calculate_vae_metrics():

            - ``rmse`` : Root mean squared error
            - ``kld`` : KL divergence
            - ``elbo`` : Evidence lower bound

        See Also
        --------
        peach._core.utils.metrics.calculate_vae_metrics : Metric computation
        peach._core.types.VAEMetrics : Return type structure
        """
        from ..utils.metrics import calculate_vae_metrics

        # If mu and log_var not provided, encode the input to get them
        if mu is None or log_var is None:
            mu, log_var = self.encode(input)

        return calculate_vae_metrics(recons, input, mu, log_var)
