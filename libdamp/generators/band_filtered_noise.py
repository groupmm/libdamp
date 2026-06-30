"""Generator for band-filtered white noise.

This module is part of the libdamp package.
"""

import torch

from ..helpers.filters import design_butter_bandpass, iir_freq_sampling
from ..helpers.tensors import interpolate_samples
from .generator import Generator


class BandFilteredNoise(Generator):
    """Generator for band-filtered white noise"""

    def __init__(self, frame_len: int, num_bands: int, order: int, fs: float) -> None:
        """Generator for band-filtered white noise.

        This generator can produce multiple bands simultaneously with variable
        amplitude, bandwidth, and center frequency.

        Parameters
        ----------
        frame_len : int
            Length of each frame in samples.
        num_bands : int
            Number of bands that are added before being returned.
        order : int
            IIR bandpass filter order.
        fs : float
            Sampling rate in Hz.
        """
        super().__init__()

        self.frame_len = frame_len
        self.num_bands = num_bands
        self.order = order
        self.fs = fs

        self.clear()  # reset generator state

    def update(self, fc: torch.Tensor, bw: torch.Tensor, ba: torch.Tensor):
        """Set amplitudes, bandwidths, and center frequencies.

        Parameters
        ----------
        fc : torch.Tensor
            Center frequencies of the noise bands in Hz, shape (B, frames, N).
        bw : torch.Tensor
            Band width of the noise bands in Hz, shape (B, frames, N).
        ba : torch.Tensor
            Band amplitudes of the noise bands, shape (B, frames, N).

        Returns
        -------
        torch.Tensor
            Generated filtered noise signal consisting of summed-up noise bands, shape (B, frames * L).
        """
        self.B, self.F, N = fc.shape
        self.dtype = fc.dtype
        self.device = fc.device

        assert self.num_bands == N, "Wrong number of bands given in input to BandFilteredNoise."

        self.b, self.a = design_butter_bandpass(fc, bw, self.fs, order=self.order)
        self.ba = ba

    def generate(self, sum_up: bool = True) -> torch.Tensor:
        """Generate band-filtered white noise with specified amplitudes, bandwidths, and center frequencies.

        Parameters
        ----------
        sum_up : bool
            whether or not the individual sinusoids are summed up before returning the signal tensor in `generate()`
            (default: True)

        Returns
        -------
        torch.Tensor
            Generated filtered noise signal consisting of summed-up noise bands, shape (B, frames * L).
        """
        assert self.b is not None, "Not initialized."

        ba_inst = interpolate_samples(self.ba.transpose(-1, -2), self.frame_len, mode="const").transpose(-1, -2)

        # same device and dtype as fc
        x = 2 * torch.rand((self.B * self.num_bands, self.F * self.frame_len)).to(device=self.device, dtype=self.dtype) - 1

        # iir_freq_sampling supports only one batch dimension so we concatenate the bands and batch dim
        b = self.b.transpose(2, 1).view(self.B * self.num_bands, self.F, -1)
        a = self.a.transpose(2, 1).view(self.B * self.num_bands, self.F, -1)
        y = iir_freq_sampling(b, a, x, N=self.frame_len)
        y = y.view(self.B, self.num_bands, self.F * self.frame_len)

        y = ba_inst.transpose(-2, -1) * y  # apply gain

        if sum_up:
            y = torch.sum(y, dim=1)

        return y

    def clear(self):
        self.b = None
        self.a = None
