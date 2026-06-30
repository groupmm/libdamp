"""Generator for simple filtered white noise.

This module is part of the libdamp package.
"""

import torch

from ..helpers.tensors import ensure_tensor, interpolate_linear
from ..helpers.transforms import get_window
from .generator import Generator


class SimpleFilteredNoise(Generator):
    """Generator for simple filtered white noise with linear filters."""

    def __init__(self, frame_len: int, filt_len: int, fs: float, freq_bands) -> None:
        """Generator for simple filtered white noise with linear filters.

        Uses filtering in frequency domain and subsequent overlap-add (OLA) in time domain.

        TODO: include checks and convenience functionalities

        Parameters
        ----------
        frame_len : int
            Length of each frame in samples.
        filt_len : int
            FIR filter length in samples, which also defines the filtered block length.
        fs : float
            Sampling rate in Hz.
        freq_bands : torch.Tensor or array-like
            Center frequencies of the filter bands in Hz, shape (num_freq_bands,).
        """
        super().__init__()

        self.frame_len = frame_len
        self.filt_len = filt_len

        freq_bands = ensure_tensor(freq_bands)

        self.register_buffer("filter_win", get_window("hann", self.filt_len))
        self.register_buffer("freq_bands", freq_bands)
        self.register_buffer("f_fft", torch.fft.rfftfreq(self.filt_len, 1 / fs))

    def generate(self, mags: torch.Tensor) -> torch.Tensor:
        """Generate filtered white noise with specified magnitude response.

        Parameters
        ----------
        mags : torch.Tensor
            Magnitudes for each frequency band, per frame, shape (B, frames, num_freq_bands).

        Returns
        -------
        torch.Tensor
            Generated filtered noise signal, shape (B, frames * L).
        """
        H = interpolate_linear(self.freq_bands, mags, self.f_fft, extrapolate="const")

        B, F, K = H.shape

        h = torch.fft.irfft(H)

        h = h.roll(K - 1, -1)
        h *= self.filter_win[None, None, :]

        h_pad = torch.nn.functional.pad(h, (0, self.frame_len))
        H_pad = torch.fft.rfft(h_pad)

        n = torch.rand((B, F, self.frame_len)).to(h) * 2 - 1
        n_pad = torch.nn.functional.pad(n, (0, self.filt_len))
        N_pad = torch.fft.rfft(n_pad)

        Y = H_pad * N_pad
        y = torch.fft.irfft(Y)

        ola = torch.eye(y.shape[-1], requires_grad=False).unsqueeze(1).to(h)
        y = torch.nn.functional.conv_transpose1d(y.transpose(1, 2), ola, stride=self.frame_len, padding=0).squeeze(1)

        return y[..., : self.frame_len * F]

    def update(self):
        """No-op. `SimpleFilteredNoise` is stateless and all parameters are passed directly to `generate()`."""

    def clear(self):
        """No-op. `SimpleFilteredNoise` is stateless and has no internal state to reset."""
