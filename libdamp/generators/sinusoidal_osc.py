"""Generator for independent sinusoids.

This module is part of the libdamp package.
"""

from typing import Literal

import torch

from ..helpers.tensors import ensure_tensor, interpolate_samples
from .generator import Generator


class SinusoidalOsc(Generator):
    """Generator for independent sinusoids with given amplitude and frequency."""

    def __init__(
        self,
        frame_len: int,
        fs: float,
        interp_f: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
        interp_a: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
    ):
        """Generator for independent sinusoids with given amplitude and frequency.

        Parameters
        ----------
        frame_len : int
            Number of samples per frame.
        fs : float
            Sampling rate in Hz.
        interp_f : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the instantaneous frequency at each sample from the given frequency for each frame.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        interp_a : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the instantaneous amplitude at each sample from the given amplitude for each frame.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        """
        super().__init__()

        self.frame_len = frame_len
        self.fs = fs
        self.interp_f = interp_f
        self.interp_a = interp_a

        self.clear()  # reset generator state

    def generate(self, sum_up: bool = True) -> torch.Tensor:
        """Generate audio from the current parameters.

        Parameters
        ----------
        sum_up : bool
            whether or not the individual sinusoids are summed up before returning the signal tensor in `generate()`
            (default: True)

        Returns
        -------
        torch.Tensor
            Audio signal of shape (B, samples) if sum_up=True or
            (B, sinusoids, samples) if sum_up=False.
        """
        assert self.initialized, "update() must be called at least once before generate()"

        f_inst = interpolate_samples(self.f, self.frame_len, mode=self.interp_f, prev_val=self.prev_f)
        a_inst = interpolate_samples(self.a, self.frame_len, mode=self.interp_a, prev_val=self.prev_a)

        # remove all components above Nyquist frequency from the synthesis
        a_inst[(f_inst >= self.fs / 2)] = 0

        phi = torch.cumsum(2 * torch.pi * f_inst / self.fs, dim=-1)  # dim: (B, sinusoids, samples)

        if self.prev_phi is not None:  # add offset from previous generation if available
            phi += self.prev_phi[:, :, None]

        # save state for next call
        self.prev_phi = phi[:, :, -1] % (2 * torch.pi)
        self.prev_f = f_inst[:, :, -1]
        self.prev_a = a_inst[:, :, -1]

        x = a_inst * torch.sin(phi)

        if sum_up:
            return torch.sum(x, dim=-2)

        return x

    def update(self, f: torch.Tensor, a: torch.Tensor) -> None:
        """Update the sinusoid parameters.

        Parameters
        ----------
        f : torch.Tensor or array-like
            Frequencies for each sinusoid, shape (B, sinusoids, frames) in Hz.
        a : torch.Tensor or array-like
            Amplitudes for each sinusoid, shape (B, sinusoids, frames).
        """
        self.f = ensure_tensor(f)
        self.a = ensure_tensor(a)

        assert len(self.f.shape) == 3, "SinusoidalOsc expects all parameters to have shape (B, sinusoids, frames)"
        assert len(self.a.shape) == 3, "SinusoidalOsc expects all parameters to have shape (B, sinusoids, frames)"

        assert self.f.shape[0] == self.a.shape[0], "All parameters must be given with the same batch size"
        assert self.f.shape[1] == self.a.shape[1], "All parameters must be given for the same number of sinusoids"
        assert self.f.shape[2] == self.a.shape[2], "All parameters must be given for the same number of frames"

        if self.prev_f is not None:
            if self.f.shape[0] != self.prev_f.shape[0] or self.f.shape[1] != self.prev_f.shape[1]:
                # if the batch size or number of sinusoids changed, we start from scratch
                self.clear()

        self.initialized = True

    def clear(self):
        self.prev_phi = None
        self.prev_f = None
        self.prev_a = None
        self.initialized = False
