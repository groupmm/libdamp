"""Collection of helper functions and classes for libdamp.

This module is part of the libdamp package.
"""

from .convolution import Convolution
from .filters import (
    OCTAVE_BANDS,
    THIRD_OCTAVE_BANDS,
    combined_freqz,
    design_butter_bandpass,
    design_butter_filter,
    design_fir_filter,
    design_resonant_filter,
    freqz,
    iir_freq_sampling,
)
from .freq import hz2midi, midi2hz, timbre2harmonics
from .incremental_mod import incremental_mod
from .modules import ConvStack, FreqToBins, LogitsToFreq, SelectItem
from .scaling import exp_sigmoid
from .tensors import (
    apply_along_dim,
    cubic_hermite_splines,
    ensure_tensor,
    interpolate_linear,
    interpolate_pchip,
    interpolate_samples,
    poly,
    smooth,
    tensor_linspace,
)
from .transforms import get_window, hilbert, istft, stft

__all__ = [
    "Convolution",
    "THIRD_OCTAVE_BANDS",
    "OCTAVE_BANDS",
    "freqz",
    "combined_freqz",
    "design_resonant_filter",
    "design_butter_bandpass",
    "design_butter_filter",
    "design_fir_filter",
    "iir_freq_sampling",
    "hz2midi",
    "midi2hz",
    "timbre2harmonics",
    "incremental_mod",
    "SelectItem",
    "ConvStack",
    "FreqToBins",
    "LogitsToFreq",
    "exp_sigmoid",
    "tensor_linspace",
    "apply_along_dim",
    "ensure_tensor",
    "interpolate_samples",
    "smooth",
    "cubic_hermite_splines",
    "interpolate_pchip",
    "interpolate_linear",
    "poly",
    "get_window",
    "stft",
    "istft",
    "hilbert",
]
