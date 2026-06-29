"""Generator for full-scale white noise.

This module is part of the libdamp package.
"""

import torch

from .generator import Generator


class WhiteNoise(Generator):
    """Generator for full-scale white noise in time domain."""

    def generate(self, shape, device=None) -> torch.Tensor:
        """Generate full-scale white noise.

        Parameters
        ----------
        shape : tuple of int
            Shape of the generated noise tensor, e.g. (B, channels, num_samples).
        device : torch.device or None
            Device to generate the noise tensor on (default: None, uses the default device).

        Returns
        -------
        torch.Tensor
            Generated white noise signal of the given shape, with values in [-1, 1).
        """
        return 2 * torch.rand(shape).to(device) - 1

    def update(self):
        """No-op; `WhiteNoise` is stateless and has no parameters."""

    def clear(self):
        """No-op; `WhiteNoise` is stateless and has no internal state to reset."""
