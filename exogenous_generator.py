import math
import torch
import torch.nn as nn
import pyro.distributions as dist
from pyro.distributions.transforms import Spline


class ExogenousGenerator(nn.Module):
    def __init__(self, num_bins=8, bound=3.0):
        super().__init__()

        if not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer greater than or equal to 2.")

        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("bound must be positive.")

        self.num_bins = num_bins
        self.bound = float(bound)

        """Learnable spline transform"""
        self.transform = Spline(
            input_dim=1,
            count_bins=num_bins,
            bound=self.bound,
            order="quadratic",
        )

        """Set up the non-trainable base distribution"""
        """Move with the model when its device or dtype changes"""
        self.register_buffer("base_loc", torch.zeros(1))
        self.register_buffer("base_scale", torch.ones(1))

    def _distribution(self):
        """Create the base distribution on the current device with the current dtype"""
        base_dist = dist.Normal(self.base_loc, self.base_scale)

        return dist.TransformedDistribution(
            base_dist,
            [self.transform],
        )

    def _as_column(self, value):
        """Use shape (N,) for the external interface"""
        if value.ndim != 1:
            raise ValueError("Input shape must be (N,).")

        if value.device != self.base_loc.device:
            raise ValueError("Input and generator must be on the same device.")

        if value.dtype != self.base_loc.dtype:
            raise ValueError("Input and generator must have the same dtype.")

        return value.unsqueeze(-1)  # (N,) → (N, 1)

    def forward(self, z):
        z = self._as_column(z)
        u = self.transform(z)
        return u.squeeze(-1)

    def inverse(self, u):
        u = self._as_column(u)
        z = self.transform.inv(u)
        return z.squeeze(-1)

    def sample(self, n):
        """Generate samples without gradient tracking"""
        samples = self._distribution().sample((n,))
        return samples.squeeze(-1)

    def rsample(self, n):
        """Generate samples differentiable with respect to the generator parameters"""
        samples = self._distribution().rsample((n,))
        return samples.squeeze(-1)

    def log_prob(self, u):
        u = self._as_column(u)
        log_density = self._distribution().log_prob(u)
        return log_density.squeeze(-1)

    def clear_cache(self):
        self.transform.clear_cache()