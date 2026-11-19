"""Squashed-Gaussian distribution shared by the graph-based policies."""
from __future__ import annotations

import math

import torch
from torch.distributions import Normal


class TanhNormal:
    """tanh(N(mean, std)); beyond +/-threshold, log_prob uses the Normal tail mass."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, threshold: float = 0.999) -> None:
        self.mean = mean
        self.std = std
        self.normal = Normal(mean, std)

        self.threshold = threshold
        self.inverse_threshold = math.atanh(threshold)

        # average(pdf) = p / epsilon -> log(average(pdf)) = log(p) - log(epsilon)
        self.log_epsilon = math.log(1.0 - threshold)

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.normal.rsample())

    def _log_cdf(self, x: float) -> torch.Tensor:
        """log P(u <= x) of the base Normal, stable in the tail."""
        z = (x - self.mean) / self.std
        return torch.special.log_ndtr(z)

    def _log_survival(self, x: float) -> torch.Tensor:
        """log P(u > x) of the base Normal, stable in the tail."""
        z = (x - self.mean) / self.std
        return torch.special.log_ndtr(-z)

    def log_prob(self, action: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Log-density of a squashed action."""
        # Clip to avoid atanh(+/-1) -> inf.
        a_clipped = torch.clamp(action, -self.threshold, self.threshold)

        u = torch.atanh(a_clipped)
        log_p_u = self.normal.log_prob(u)
        # Inverse log det Jacobian: -log(1 - y^2)
        log_det_inv = torch.log(torch.clamp(1.0 - a_clipped.pow(2), min=eps))
        inner_log_prob = log_p_u - log_det_inv

        # At the clip, the tail mass spread over the last epsilon.
        log_prob_left = self._log_cdf(-self.inverse_threshold) - self.log_epsilon
        log_prob_right = self._log_survival(self.inverse_threshold) - self.log_epsilon

        log_prob_components = torch.where(
            a_clipped <= -self.threshold,
            log_prob_left,
            torch.where(a_clipped >= self.threshold, log_prob_right, inner_log_prob),
        )

        return log_prob_components.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """One-sample estimate H(N) + log(1 - tanh(u)^2), u ~ N; there is no closed form."""
        base_entropy = self.normal.entropy()
        u = self.normal.rsample()
        a = torch.tanh(u)
        log_det_jacobian = torch.log(torch.clamp(1.0 - a.pow(2), min=1e-8))
        return (base_entropy + log_det_jacobian).sum(dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mean)
