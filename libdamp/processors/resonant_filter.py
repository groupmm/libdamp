"""Parallel resonant filter processor.

This module is part of the libdamp package.
"""

import torch

from ..helpers.filters import design_resonant_filter, iir_freq_sampling
from .processor import Processor


class ResonantFilter(Processor):
    """Parallel processing with biquad filters that represent individual resonances (or "formants")"""

    def __init__(self, frame_len: int, fs: float) -> None:
        """Initialize resonant filter processor.

        Parameters
        ----------
        frame_len : int
            Frame length in samples for the IIR frequency sampling.
        fs : float
            Sampling rate in Hz.
        """
        super().__init__()

        self.fs = fs
        self.frame_len = frame_len

        self.clear()  # reset state

    def process(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio with the parallel resonant filters.

        Parameters
        ----------
        x : torch.Tensor
            Input audio signal(s). Shape can be (batch, length) or (batch, channels, length).

        Returns
        -------
        torch.Tensor
            Processed audio signal, same shape as input.
        """
        assert self.num_filters > 0, "update() must be called at least once before process()."

        has_channel = x.ndim == 3
        if has_channel:
            B, C, L = x.shape
            x = x.reshape(B * C, L)

        y = torch.zeros_like(x)
        for i in range(self.num_filters):
            b_i = self.b[:, i, :, :]
            a_i = self.a[:, i, :, :]
            if has_channel:
                b_i = torch.repeat_interleave(b_i, C, dim=0)
                a_i = torch.repeat_interleave(a_i, C, dim=0)
            y += iir_freq_sampling(b_i, a_i, x, N=self.frame_len)

        if has_channel:
            y = y.reshape(B, C, -1)

        return y

    def update(self, f, r):
        """Update the resonant filter parameters.

        Parameters
        ----------
        f : torch.Tensor or array-like
            Center frequencies in Hz, shape (batch, num_filters, frames).
        r : torch.Tensor or array-like
            Resonance radii between 0 and 1, shape (batch, num_filters, frames).
        """
        # f/r shape: (batch, num_filters, frames)
        self.b, self.a = design_resonant_filter(f, r, self.fs)
        # b/a shape: (batch, num_filters, frames, num_coeffs = 3)
        self.num_filters = self.b.shape[1]

    def clear(self):
        self.b = None
        self.a = None
        self.num_filters = 0
