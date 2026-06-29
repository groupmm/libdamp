"""Collection of helper functions for calculating spectral representations.

This module is part of the libdamp package.
"""

from typing import Literal

import torch

from libdamp.helpers.tensors import ensure_tensor


def hz2midi(f: torch.Tensor | float, a4_ref: float = 440.0, zero_val: int | float = -1) -> torch.Tensor:
    """Convert frequency to MIDI pitch with respect to a reference.

    Frequency 0 Hz is handled separately and mapped to the given value `zero_val`.

    Parameters
    ----------
    f : torch.Tensor | float
        Frequency in Hz.
    a4_ref : float
        Reference frequency for MIDI pitch 69 (A4) in Hz (default: 440).
    zero_val : int or float
        Special value to assign to 0 Hz input (default: -1).

    Returns
    -------
    torch.Tensor
        MIDI pitch with the same shape as input `f`.
    """
    f = ensure_tensor(f)
    p = torch.ones_like(f) * zero_val
    mask = f > 0
    p[mask] = 12 * torch.log2(f[mask] / a4_ref) + 69
    return p


def midi2hz(p: torch.Tensor | float, a4_ref: float = 440.0, zero_val: int | float = -1) -> torch.Tensor:
    """Convert MIDI pitch to frequency.

    Frequency 0 Hz is handled separately based on the given value `zero_val`.

    Parameters
    ----------
    p : torch.Tensor | float
        MIDI pitch.
    a4_ref : float
        Reference frequency for MIDI pitch 69 (A4) in Hz (default: 440).
    zero_val : int or float
        Special value that indicates 0 Hz output (default: -1).

    Returns
    -------
    torch.Tensor
        Frequency in Hz with the same shape as input `p`.
    """
    p = ensure_tensor(p)
    f = a4_ref * 2 ** ((p - 69) / 12)
    mask = p == zero_val
    f[mask] = 0
    return f


def timbre2harmonics(
    timbre: Literal[
        "square",
        "triangle",
        "sawtooth",
        "flat",
        "clarinet-like",
        "random_harmonic",
        "random_inharmonic",
        "random_inharmonic_sawtooth",
    ],
    H: int,
    harmonic_sigma: float = 0.02,
    amplitude_sigma: float = 0.5,
) -> torch.Tensor:
    """Generate an overtone distribution from a timbre descriptor.

    Parameters
    ----------
    timbre : str
        Name of the desired timbre.

        - "square": odd harmonics with amplitudes ~ 1/n (even harmonics are zero).
        - "triangle": alternating-sign series with amplitudes ~ 1/n^2.
        - "sawtooth": alternating-sign series with amplitudes ~ 1/n. Fundamental has amplitude 1.
        - "flat": all harmonics with unit amplitude.
        - "clarinet-like": odd harmonics only, amplitudes ~ 1/n.
        - "random_harmonic": harmonic factors 1..H, amplitudes ~ N(1, amplitude_sigma), clipped at 0.
        - "random_inharmonic": harmonic factors for n>=2 are jittered by N(n, harmonic_sigma), and
          amplitudes ~ N(1, amplitude_sigma), clipped at 0. The first harmonic factor remains 1.
        - "random_inharmonic_sawtooth": harmonic factors for n>=2 are jittered by N(n, harmonic_sigma), and
          amplitudes follow the sawtooth series (no amplitude randomness). The first harmonic amplitude (of fundamental) remains 1.
    H : int
        number of harmonics to return
    harmonic_sigma : float
        Standard deviation of the harmonic randomness
        (used only for "random_inharmonic" and "random_inharmonic_sawtooth").
    amplitude_sigma : float
        Standard deviation of the amplitude randomness
        (used only for "random_harmonic" and "random_inharmonic").

    Returns
    -------
    A : torch.Tensor
        (H, 2) tensor, where the first column contains the harmonic frequency factors (including fundamental)
        and the second column contains the harmonic amplitudes.
    """
    if timbre == "square":
        A = torch.zeros((H, 2))
        A[:, 0] = torch.arange(1, H + 1)
        A[2::2, 1] = 1.0 / (torch.arange(2, H, 2) + 1)
        A[0, 1] = 1
    elif timbre == "triangle":
        A = torch.zeros((H, 2))
        A[:, 0] = torch.arange(1, H + 1)
        odd_harmonics = torch.arange(1, H + 1, 2)
        A[0::2, 1] = 8 / (torch.pi**2) * torch.pow(-1, (odd_harmonics - 1) // 2) * torch.pow(odd_harmonics, -2.0)
    elif timbre == "sawtooth":
        A = torch.ones((H, 2))
        A[:, 0] = torch.arange(1, H + 1)
        n = torch.arange(1, H)
        A[1:, 1] = 2 / torch.pi * torch.pow(-1, n) / n
    elif timbre == "flat":
        A = torch.ones((H, 2))
        A[:, 0] = torch.arange(1, H + 1)
    elif timbre == "clarinet-like":
        A = torch.ones((H, 2))
        A[:, 0] = torch.arange(1, H + 1)
        A[:, 1] = 1 / torch.arange(1, H + 1)
        A[1::2, 1] = 0
    elif timbre == "random_harmonic":
        A = torch.ones((H, 2))
        A[:, 0] = torch.arange(1, H + 1)
        A[:, 1] = torch.clip(torch.normal(torch.ones(H), amplitude_sigma), 0)
    elif timbre == "random_inharmonic":
        A = torch.ones((H, 2))
        A[1:, 0] = torch.normal(torch.arange(2, H + 1).to(torch.float64), harmonic_sigma)
        A[:, 1] = torch.clip(torch.normal(torch.ones(H), amplitude_sigma), 0)
    elif timbre == "random_inharmonic_sawtooth":
        A = torch.ones((H, 2))
        A[1:, 0] = torch.normal(torch.arange(2, H + 1).to(torch.float64), harmonic_sigma)
        n = torch.arange(1, H)
        A[1:, 1] = 2 / torch.pi * torch.pow(-1, n) / n
    else:
        raise ValueError(f"Unknown timbre '{timbre}'.")

    return A
