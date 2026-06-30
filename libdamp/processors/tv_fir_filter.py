"""Time-varying FIR filter processor.

This module is part of the libdamp package.
"""

import math

import torch

from ..helpers.tensors import ensure_tensor
from .processor import Processor


class TVFIRFilter(Processor):
    """Time-varying FIR filter processor.

    Applies blockwise processing of audio with time-varying FIR filters.
    Supports optional crossfading for smooth transitions between filters.
    """

    def __init__(self, frame_len: int, filt_len: int, with_crossfade: bool = False, crossfade_len: int | None = None) -> None:
        """Initialize time-varying FIR filter processor.

        Parameters
        ----------
        frame_len : int
            Frame length in samples.
        filt_len : int
            Filter length in samples.
        with_crossfade : bool
            Whether to apply crossfading between filter transitions in time domain
            for smooth transitions (default: False).
        crossfade_len : int or None
            Length of crossfade in samples, defaults to N if `with_crossfade` is True (default: None).

        Notes
        -----
        Currently only supports (B, 1, L) and (B, L) input tensors.
        """
        super().__init__()

        self.frame_len = frame_len
        self.filt_len = filt_len

        self.cf = with_crossfade

        if self.cf:
            self.cf_len = crossfade_len if crossfade_len is not None else self.frame_len
            self.register_buffer("fade_in", torch.linspace(0, 1, self.cf_len))
            self.register_buffer("fade_out", torch.linspace(1, 0, self.cf_len))

        self.clear()  # reset state

    def process(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio with time-varying FIR filters.

        Parameters
        ----------
        x : torch.Tensor
            Input audio signal(s). Shape must be (batch, length) or (batch, channels, length).
            The length must be divisible by the frame length `N` given at initialization.

        Returns
        -------
        torch.Tensor
            Processed audio signal with time-varying FIR filtering applied.
        """
        assert self.H is not None, "update() must be called at least once before process()."

        B = x.shape[0]
        L = x.shape[-1]

        num_frames = L // self.frame_len
        frames_per_filter = int(math.ceil(num_frames / self.H.shape[-2]))

        assert self.frame_len * num_frames == L, "Input length must be divisible by the initialized frame length."

        has_channel = len(x.shape) == 3
        if has_channel:
            assert x.shape[-2] == 1, "TVFIRFilter only supports a single channel."
            # squeeze channel dimension
            x = torch.squeeze(x, dim=-2)

        x_frm = x.view(B, num_frames, self.frame_len)
        x_pad = torch.nn.functional.pad(x_frm, (0, self.filt_len))
        X = torch.fft.rfft(x_pad)

        H_int = torch.repeat_interleave(self.H, frames_per_filter, dim=-2)
        Y = X * H_int[:, :num_frames, :]
        y = torch.fft.irfft(Y)

        # process each frame twice and do a time-domain crossfade to allow for a smooth transition between filters
        if self.cf:
            Y2 = X[:, 1:, :] * H_int[:, : num_frames - 1, :]
            y2 = torch.fft.irfft(Y2)
            y[:, 1:, : self.cf_len] = y[:, 1:, : self.cf_len] * self.fade_in + y2[:, :, : self.cf_len] * self.fade_out

        ola = torch.eye(y.shape[-1], requires_grad=False).unsqueeze(1).to(x)
        y = torch.nn.functional.conv_transpose1d(y.transpose(1, 2), ola, stride=self.frame_len, padding=0).squeeze(1)

        if has_channel:
            # unsqueeze channel dimension
            y = y[:, None, :]

        return y[..., :L]

    def update(self, h) -> None:
        """Update the FIR filter coefficients.

        Parameters
        ----------
        h : torch.Tensor or array-like
            Filter coefficients, shape (batch, num_filters, filter_length). The filters are
            repeated across frames if fewer filters than signal frames are provided.
        """
        h = ensure_tensor(h, min_dims=2)

        h_pad = torch.nn.functional.pad(h, (0, self.frame_len))
        self.H = torch.fft.rfft(h_pad)

    def clear(self):
        self.H = None
