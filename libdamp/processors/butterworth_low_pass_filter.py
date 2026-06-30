"""Butterworth low-pass filter processor.

This module is part of the libdamp package.
"""

import torch

from ..helpers.filters import design_butter_filter, iir_freq_sampling
from .processor import Processor


class ButterworthLowPassFilter(Processor):
    """Butterworth low-pass filter processor.

    Processes signal with cascaded Butterworth low-pass filters using IIR frequency sampling design.
    """

    def __init__(self, frame_len: int, fs: float, order: int = 2, cascades: int = 1) -> None:
        """Initialize Butterworth low-pass filter processor.

        Parameters
        ----------
        N : int
            Frame length in samples for the IIR frequency sampling.
        fs : float
            Sampling rate in Hz.
        order : int
            Filter order (default: 2).
        cascades : int
            Number of filter cascades applied in series (default: 1).
        """
        super().__init__()

        self.order = order
        self.cascades = cascades
        self.frame_len = frame_len
        self.fs = fs

        self.clear()  # reset state

    def process(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio with the cascaded Butterworth low-pass filters.

        Parameters
        ----------
        x : torch.Tensor
            Input audio signal(s). Shape can be (batch, length) or (batch, channels, length).

        Returns
        -------
        torch.Tensor
            Processed audio signal, same shape as input.
        """
        assert self.b is not None, "update() must be called at least once before process()."

        has_channel = x.ndim == 3
        if has_channel:
            B, C, L = x.shape
            x = x.reshape(B * C, L)
            b = torch.repeat_interleave(self.b, C, dim=0)
            a = torch.repeat_interleave(self.a, C, dim=0)
        else:
            b, a = self.b, self.a

        y = x
        for _ in range(self.cascades):
            y = iir_freq_sampling(b, a, y, N=self.frame_len)

        if has_channel:
            y = y.reshape(B, C, -1)

        return y

    def update(self, fc: torch.Tensor) -> None:
        """Update the filter cutoff frequency.

        Parameters
        ----------
        fc : torch.Tensor or array-like
            Cutoff frequency in Hz, shape (batch, frames).
        """
        # fc shape: (batch, frames)
        self.b, self.a = design_butter_filter(fc, self.fs, order=self.order, dtype=torch.float64)
        # b/a shape: (batch, frames, num_coeffs = order + 1)

    def clear(self):
        self.b = None
        self.a = None
