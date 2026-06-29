"""Fast and flexible implementation of partitioned convolution.

This module is part of the libdamp package.
"""

import math

import torch


class Convolution(torch.nn.Module):
    """Fast and Flexible FIR Filter implementation

    An implementation of the Generalized Uniformly Partitioned Overlap Save (GUPOLS) algorithm as described in [1]_.
    It allows for a time-varying filter. Upon filter exchange, the next input signal frame is processed both with
    the new and the previous filter, and a time-domain cross-fade between the two filter output signals is performed
    to avoid discontinuities.

    Note that variable names follow the algorithmic description in [1]_.

    .. [1] F. Wefers, Partitioned Convolution Algorithms for Real-Time Auralization.
           Aachener Beiträge zur technischen Akustik, Band 20. Logos Verlag, Berlin, 2015.
    """

    def __init__(self, B, N, L=None, K=None, C=1):
        """
        Convolution can be used in four different modes: mono, multi-channel filter, multi-channel input, or batch mode.
        - Mono mode:
            A single-channel input signal is convolved with a single filter.
            The filter `h` should have shape (N) or (1, N), the input signal `x` should have shape (B) or (1, B),
            the output signal `y` has shape (1, B).
        - Multi-channel filter mode:
            The same single-channel input signal is convolved with multiple filters.
            The filter `h` should have shape (F, N), where F is the number of filters,
            the input signal `x` should have shape (B) or (1, B),
            the output signal `y` has shape (F, B).
        - Multi-channel input mode:
            Multiple channels of the input signal are convolved with a single filters.
            The filter `h` should have shape (N) or (1, N), the input signal `x` should have shape (C, B),
            the output signal `y` has shape (C, B).
        - Batch mode:
            Multiple channels of the input signal are convolved with a different filter each.
            The filter `h` should have shape (C, N), the input signal `x` should have shape (C, B),
            the output signal `y` has shape (C, B).

        Parameters
        ----------
        B : unsigned int
            number of samples per input signal frame
        N : unsigned int
            total filter length in samples
        L : unsigned int
            (optional) partition length of the filter, defaults to L = B
        K : unsigned int
            (optional) transform length for the frequency domain convolution, defaults to K = 2*B.
            Must be at least K = B + L + R - 1, where R is the maximum remainder delay (see Ch. 5.3.1 in [1]).
        C : unsigned int
            (optional) number of input signal channels.
            If C is larger than 1, the filter can only have 1 or exactly C channels (see modes description above).
        """
        super().__init__()

        if L is None:
            L = B
        if K is None:
            K = 2 * B

        self.B = B
        self.N = N
        self.L = L
        self.K = K
        self.C = C

        self.P = int(math.ceil(self.N / self.L))
        self.rem = torch.arange(self.P) * (self.L - self.B)

        assert L >= B, "libdamp.Convolution: For this implementation, L must be >= B."
        assert B + L + torch.max(self.rem) - 1 <= K, (
            f"libdamp.Convolution: Transform size K must be larger or equal to {(B + L + torch.max(self.rem) - 1)}."
        )

        # using register_buffer to make sure the variable ends up on the right device with Lightning
        self.register_buffer("buf", torch.zeros((self.C, self.K)))
        self.register_buffer("fdl", torch.zeros((self.C, self.P, int(math.ceil((self.K + 1) / 2))), dtype=torch.cfloat))
        self.register_buffer("fade_in", torch.linspace(0, 1, self.B))
        self.register_buffer("fade_out", torch.linspace(1, 0, self.B))

        self.filt = None
        self.filt2 = None

    def set_filter(self, h):
        """Set the (possibly time-varying) filter

        Parameters
        ----------
        h : torch.Tensor
            Time-domain filter impulse response(s) with shape (1, N) (mono or multi-channel input mode),
            shape (F, N) (multi-channel filter mode), or shape (C, N) (batch mode).
            See `__init__()` documentation for more info about modes.
        """

        h = torch.atleast_2d(h)
        assert h.shape[-1] == self.N, "Filter must have prescribed length N."
        if self.C > 1:
            assert h.shape[0] == 1 or h.shape[0] == self.C, (
                "If input signal has more than one channel (C > 1), the filter must either have 1 or C channels."
            )

        # pad to multiple of L and partition
        h = torch.nn.functional.pad(h, (0, (self.P * self.L) - self.N)).reshape(-1, self.P, self.L)

        # pad to length K
        h = torch.nn.functional.pad(h, (0, self.K - self.L))

        # incorporate possible remainder delays
        for i in range(h.shape[1]):
            h[:, i] = torch.roll(h[:, i], self.rem[i].item(), dims=-1)

        # transform to frequency domain
        if self.filt is None:
            # only for the first call
            self.filt = torch.fft.rfft(h)
        else:
            self.filt2 = torch.fft.rfft(h)

    def forward(self, x):
        """Implements forward pass of `torch.nn.Module`"""
        return self.process(x)

    def process(self, x):
        """Process a frame of input signal samples

        Parameters
        ----------
        x : torch.Tensor
            time-domain input signal with shape (1, B) (mono or multi-channel filter mode) or shape (C, B)
            (multi-channel input or batch mode). See `__init__()` documentation for more info about modes.
        """

        assert self.filt is not None, "A filter must be set before input signals can be processed."

        x = torch.atleast_2d(x)
        assert x.ndim == 2 and x.shape[0] == self.C and x.shape[1] == self.B, (
            "Input signal must contain the correct number of channels and samples (C, B)."
        )

        self.buf = torch.roll(self.buf, -self.B, dims=-1)
        self.buf[:, -self.B :] = x

        self.fdl = torch.roll(self.fdl, 1, dims=-2)
        self.fdl[:, 0, :] = torch.fft.rfft(self.buf)

        y = torch.fft.irfft(torch.sum(self.fdl * self.filt, axis=-2), self.K)

        if self.filt2 is not None:
            # we have a filter exchange: crossfade between both outputs and exchange main and sidechain filter
            y2 = torch.fft.irfft(torch.sum(self.fdl * self.filt2, axis=-2), self.K)
            y[:, -self.B :] = self.fade_out * y[:, -self.B :] + self.fade_in * y2[:, -self.B :]
            # fade_phase = 0.5 * torch.pi * torch.arange(self.B) / self.B
            # y[:,-self.B:] = torch.cos(fade_phase) * y[:,-self.B:] + torch.sin(fade_phase) * y2[:,-self.B:]

            self.filt = self.filt2
            self.filt2 = None

        return y[:, -self.B :]
