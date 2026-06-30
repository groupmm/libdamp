"""Generator for a band-limited impulse train.

This module is part of the libdamp package.
"""

from typing import Literal

import torch

from ..helpers.freq import timbre2harmonics
from ..helpers.tensors import ensure_tensor
from .generator import Generator
from .sinusoidal_osc import SinusoidalOsc


class ImpulseTrain(Generator):
    """Generator for a band-limited impulse train with a given fundamental frequency."""

    def __init__(
        self,
        frame_len: int,
        num_harm: int,
        fs: float,
        sum_up: bool = True,
        interp_f: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
    ):
        """Generator for a band-limited impulse train with a given fundamental frequency.

        Uses a `SinusoidalOsc` under the hood.

        Parameters
        ----------
        frame_len : int
            Number of samples per frame.
        num_harm : int
            Maximum number of harmonics.
        fs : float
            Sampling rate in Hz.
        sum_up : bool
            whether or not the individual harmonics are summed up before returning the signal tensor in `generate()`
            (default: True)
        interp_f : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the instantaneous frequency at each sample from the given frequency for each frame.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        """
        super().__init__()

        self.sum_up = sum_up
        self.osc = SinusoidalOsc(frame_len, fs, interp_f, interp_a="const")
        self.timbre = timbre2harmonics("flat", num_harm)

    def generate(self):
        x = self.osc.generate(sum_up=self.sum_up)
        # this normalization is necessary because the band-limited impulse train has different amplitudes
        # depending on F0, but it can lead to discontinuities when calling generate() multiple times...
        # using the peak absolute value (rather than the signed max) keeps the output within [-1, 1]
        # even when the waveform is not symmetric around zero.
        x_max, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
        return x / (x_max + 1e-8)

    def update(self, f0, inharmonicity=None):
        """Update the impulse train fundamental frequency and optionally inharmonicity.

        Parameters
        ----------
        f0 : torch.Tensor or array-like
            Fundamental frequencies, shape (B, frames) in Hz.
        inharmonicity : torch.Tensor or array-like, optional
            Inharmonicity factors (frequency multipliers for each harmonic), shape (B, harmonics, frames).
            Default: None (uses perfect harmonics).
        """
        f0 = ensure_tensor(f0)

        assert len(f0.shape) == 2, "ImpulseTrain expects f0 to have shape (B, frames)"

        B, F = f0.shape
        a = torch.repeat_interleave(torch.repeat_interleave(self.timbre[None, :, [1]], B, dim=0), F, dim=2)

        H = a.shape[1]  # number of harmonics
        f = torch.einsum("bf,h->bhf", f0, torch.arange(1, H + 1).to(f0))

        if inharmonicity is not None:
            inharmonicity = ensure_tensor(inharmonicity)
            assert inharmonicity.shape == a.shape, "ImpulseTrain expects inharmonicity factors to the same shape as `a` (B, harmonics, frames)"
            f = f * inharmonicity

        self.osc.update(f, a)

    def clear(self):
        self.osc.clear()
