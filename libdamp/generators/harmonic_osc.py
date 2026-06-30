"""Generator for harmonic sinusoids.

This module is part of the libdamp package.
"""

from typing import Literal

import torch

from ..helpers.tensors import ensure_tensor
from .generator import Generator
from .sinusoidal_osc import SinusoidalOsc


class HarmonicOsc(Generator):
    """Generator for harmonic sinusoids with given amplitude and fundamental frequency."""

    def __init__(
        self,
        frame_len: int,
        fs: float,
        sum_up: bool = True,
        interp_f: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
        interp_a: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
    ):
        """Generator for harmonic sinusoids with given amplitude and fundamental frequency.

        Uses a `SinusoidalOsc` under the hood.

        Parameters
        ----------
        frame_len : int
            Number of samples per frame.
        fs : float
            Sampling rate in Hz.
        sum_up : bool
            whether or not the individual sinusoids are summed up before returning the signal tensor in `generate()`
            (default: True)
        interp_f : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the instantaneous frequency at each sample from the given frequency for each frame.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        interp_a : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the instantaneous amplitude at each sample from the given amplitude for each frame.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        """
        super().__init__()

        self.osc = SinusoidalOsc(frame_len, fs, interp_f, interp_a)

    def generate(self) -> torch.Tensor:
        """Generate audio from the current harmonic parameters.

        Returns
        -------
        torch.Tensor
            Audio signal of shape (B, samples) if sum_up=True or
            (B, harmonics, samples) if sum_up=False.
        """
        return self.osc.generate()

    def update(
        self,
        f0: torch.Tensor,
        a: torch.Tensor,
        inharmonicity: torch.Tensor | None = None,
    ) -> None:
        """Update the harmonic oscillator parameters.

        Parameters
        ----------
        f0 : torch.Tensor or array-like
            Fundamental frequencies, shape (B, frames) in Hz.
        a : torch.Tensor or array-like
            Harmonic amplitudes, shape (B, harmonics, frames).
        inharmonicity : torch.Tensor or array-like, optional
            Inharmonicity factors (frequency multipliers for each harmonic), shape (B, harmonics, frames).
            Default: None (uses perfect harmonics).
        """
        f0 = ensure_tensor(f0)
        a = ensure_tensor(a)

        assert len(f0.shape) == 2, "HarmonicOsc expects f0 to have shape (B, frames)"
        assert len(a.shape) == 3, "HarmonicOsc expects amplitudes to have shape (B, harmonics, frames)"

        assert f0.shape[0] == a.shape[0], "All parameters must be given with the same batch size"
        assert f0.shape[1] == a.shape[2], "All parameters must be given for the same number of frames"

        H = a.shape[1]  # number of harmonics
        f = torch.einsum("bf,h->bhf", f0, torch.arange(1, H + 1).to(f0))

        if inharmonicity is not None:
            inharmonicity = ensure_tensor(inharmonicity)
            assert inharmonicity.shape == a.shape, "HarmonicOsc expects inharmonicity factors to the same shape as `a` (B, harmonics, frames)"
            f = f * inharmonicity

        self.osc.update(f, a)

    def clear(self):
        self.osc.clear()
