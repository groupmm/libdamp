"""Processor that multiplies audio with a gain envelope.

This module is part of the libdamp package.
"""

from typing import Literal

import torch

from ..helpers.tensors import ensure_tensor, interpolate_samples
from .processor import Processor


class Envelope(Processor):
    """Multiply audio with a prescribed gain envelope"""

    def __init__(
        self,
        interp_mode: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
    ):
        """Multiply audio with a prescribed gain envelope.

        Parameters
        ----------
        interp_mode : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the instantaneous gain at each sample from the given gain for each frame.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        """
        super().__init__()

        self.interp_mode = interp_mode

        self.clear()  # reset state

    def process(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio with the gain envelope.

        Parameters
        ----------
        x : torch.Tensor or array-like
            Input audio signal(s). Shape can be (batch, length) or (batch, channels, length).

        Returns
        -------
        torch.Tensor
            Processed audio signal with envelope applied, same shape as input.
        """
        x = ensure_tensor(x)
        has_channel = x.ndim == 3
        if not has_channel:
            # add a channel dimension in the middle for broadcasting against `self.g`
            x = x[:, None, :]

        assert self.g is not None, "update() must be called at least once before process()"

        N = int(x.shape[-1] // self.g.shape[-1])
        assert x.shape[-1] / self.g.shape[-1] == N, "Input signal has an unexpected shape"

        env = interpolate_samples(self.g, N, mode=self.interp_mode, prev_val=self.prev_g)
        self.prev_g = env[:, :, -1]

        y = env * x

        if not has_channel and y.shape[-2] == 1:
            # only the dummy channel dimension we added above is removed. If `g` itself defines
            # more than one channel, the output legitimately gains that channel dimension.
            y = y[:, 0, :]

        return y

    def update(self, g, initial_g=None):
        """Update the gain envelope.

        Parameters
        ----------
        g : torch.Tensor or array-like
            Gain envelope, shape (batch, frames) or (batch, channels, frames).
        initial_g : torch.Tensor or array-like, optional
            Initial gain value for state continuity from previous processing, shape
            (batch, frames) or (batch, channels, frames) (default: None).
        """
        self.g = ensure_tensor(g, min_dims=2)
        if self.g.ndim == 2:
            # add a channel dimension in the middle
            self.g = self.g[:, None, :]

        if initial_g is not None:
            self.prev_g = ensure_tensor(initial_g, min_dims=2)

    def clear(self):
        self.g = None
        self.prev_g = None
