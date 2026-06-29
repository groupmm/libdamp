"""Time-varying FIR filter processor defined by magnitude response.

This module is part of the libdamp package.
"""

from typing import Literal

import torch

from ..helpers.filters import OCTAVE_BANDS, design_fir_filter
from ..helpers.tensors import ensure_tensor
from .tv_fir_filter import TVFIRFilter


class TVMagnitudesFilter(TVFIRFilter):
    """Time-varying FIR filter processor based on magnitude response.

    Applies blockwise processing of audio with time-varying FIR filters defined
    by magnitude specifications using `design_fir_filter`. This is a specialization of
    [`TVFIRFilter`][libdamp.processors.tv_fir_filter.TVFIRFilter] that derives the filter
    taps from a magnitude response instead of taking them directly; `process()` is
    inherited unchanged.
    """

    def __init__(
        self,
        N: int,
        M: int,
        fs: float,
        center_freqs: torch.Tensor = OCTAVE_BANDS,
        with_crossfade: bool = False,
        crossfade_len: int | None = None,
        filter_phase: Literal["linear", "minimum"] = "minimum",
    ) -> None:
        """Initialize magnitude-based time-varying FIR filter processor.

        Parameters
        ----------
        N : int
            Frame length in samples.
        M : int
            Filter length in samples.
        fs : float
            Sampling rate in Hz.
        center_freqs : torch.Tensor
            Center frequencies in Hz at which the magnitudes are given in `update()`
            for the filter design. Shape: 1D or 2D (for different center frequencies
            per batch item). Default: `libdamp.OCTAVE_BANDS`.
        with_crossfade : bool
            Whether to apply crossfading between filter transitions in time domain
            for smooth transitions (default: False).
        crossfade_len : int or None
            Length of crossfade in samples, defaults to N if `with_crossfade` is True (default: None).
        filter_phase : Literal["linear", "minimum"]
            Phase characteristics of the designed filter. Options: "linear" for linear
            phase (symmetric), "minimum" for minimum phase (default: "minimum").

        Notes
        -----
        Currently only supports (B, 1, L) and (B, L) input tensors.
        """
        super().__init__(N, M, with_crossfade, crossfade_len)

        self.fs = fs
        self.filter_phase = filter_phase

        # using register_buffer to make sure the variable ends up on the right device with Lightning
        self.register_buffer("center_freqs", center_freqs)

    def update(self, magnitudes):
        """Update the filter magnitude response.

        Parameters
        ----------
        magnitudes : torch.Tensor or array-like
            Desired magnitude response at center frequencies, shape (batch, num_filters, num_freqs)
            where num_freqs matches the number of center frequencies.
        """
        magnitudes = ensure_tensor(magnitudes, min_dims=2)
        assert magnitudes.shape[-1] == self.center_freqs.shape[-1], "Number of frequency bands must match."

        h = design_fir_filter(self.center_freqs, magnitudes, self.fs, self.M, phase=self.filter_phase)
        super().update(h)
