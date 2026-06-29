"""Collection of helper functions and classes related to FIR filters.

This module is part of the libdamp package.
"""

import math
from typing import Literal

import torch

from libdamp.helpers.tensors import ensure_tensor, interpolate_linear, poly

__all__ = [
    "THIRD_OCTAVE_BANDS",
    "OCTAVE_BANDS",
    "freqz",
    "combined_freqz",
    "design_resonant_filter",
    "design_butter_bandpass",
    "design_butter_filter",
    "design_fir_filter",
    "iir_freq_sampling",
]


# Center frequencies for the most common third-octave bands, splitting the common range of HiFi audio into 31 bands
# fmt: off
THIRD_OCTAVE_BANDS = torch.Tensor(
    [
        19.69, 24.80, 31.25, 39.37, 49.61, 62.50, 78.75, 99.21,
        125.00, 157.49, 198.43, 250.00, 314.98, 396.85, 500.00, 629.96,
        793.70, 1000.00, 1259.92, 1587.40, 2000.00, 2519.84, 3174.80, 4000.00,
        5039.68, 6349.60, 8000.00, 10079.37, 12699.21, 16000.00, 20158.74,
    ]
)
# fmt: on

# Center frequencies for the most common octave bands, splitting the common range of HiFi audio into 10 octaves
OCTAVE_BANDS = torch.Tensor([31.25, 62.50, 125.00, 250.00, 500.00, 1000.00, 2000.00, 4000.00, 8000.00, 16000.00])


def freqz(b: torch.Tensor, a: torch.Tensor, N: int = 1024, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """Sample frequency response of a digital filter.

    Computes the frequency response H(z) of an IIR filter given numerator (b) and
    denominator (a) coefficients. The frequency response is computed by evaluating
    the transfer function on the unit circle in the z-plane.

    Parameters
    ----------
    b : torch.Tensor
        Numerator coefficients of shape (batch, order) or (order,).
    a : torch.Tensor
        Denominator coefficients of shape (batch, order) or (order,). First coefficient must be 1.0.
    N : int
        Determines frequency resolution: the frequency response is evaluated at (N+1)//2 + 1
        uniformly spaced frequency points from 0 to Nyquist frequency (default: 1024)
    dtype : torch.dtype
        Compute frequency response in this dtype - complex64 for 32-bit or complex128 for 64-bit precision

    Returns
    -------
    torch.Tensor
        Complex-valued frequency response of shape (batch, (N+1)//2 + 1) or ((N+1)//2 + 1,)
        containing only the positive frequency half.
    """
    order = max(b.shape[-1], a.shape[-1])

    # pad if one dim is shorter
    if b.shape[-1] < order:
        b = torch.nn.functional.pad(b, (0, order - b.shape[-1]), mode="constant", value=0.0)
    if a.shape[-1] < order:
        a = torch.nn.functional.pad(a, (0, order - a.shape[-1]), mode="constant", value=0.0)

    assert torch.all(a[..., 0] == 1), "a_0 coefficient must be one for all filters"

    b = b.type(dtype)
    a = a.type(dtype)

    K = (N + 1) // 2 + 1
    freqs = torch.linspace(0, torch.pi, K + 1)[:-1]

    omega = torch.outer(torch.arange(1, order), freqs).type(dtype)
    z = torch.exp(-1j * omega).to(b.device)

    H = (b[..., [0]] + b[..., 1:] @ z) / (1 + a[..., 1:] @ z)

    return H


def combined_freqz(b: torch.Tensor, a: torch.Tensor, N: int = 1024, parallel: bool = False) -> torch.Tensor:
    """Sample frequency response of a series of digital filters.

    Computes the combined frequency response of multiple filters arranged either in
    series (cascaded) or parallel configuration.

    Parameters
    ----------
    b : torch.Tensor
        Numerator coefficients of shape (..., num_filters, order).
    a : torch.Tensor
        Denominator coefficients of shape (..., num_filters, order).
    N : int
        Determines frequency resolution: the frequency response is evaluated at (N+1)//2 + 1
        uniformly spaced frequency points
    parallel : bool
        If True, computes parallel configuration (sum of responses).
        If False, computes series configuration (product of responses) (default: False)

    Returns
    -------
    torch.Tensor
        Combined complex-valued frequency response of shape (..., (N+1)//2 + 1) containing only the positive frequency half.
    """
    H = None
    for i in range(b.shape[-2]):
        if H is None:
            H = freqz(b[..., i, :], a[..., i, :], N)
        else:
            if parallel:
                H += freqz(b[..., i, :], a[..., i, :], N)
            else:
                H *= freqz(b[..., i, :], a[..., i, :], N)
    return H


def design_resonant_filter(f: torch.Tensor, r: torch.Tensor, fs: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Design second-order IIR resonance (peak) filters.

    Creates filter coefficients for resonant (peaking) filters that amplify
    energy around specific frequencies.

    Parameters
    ----------
    f : torch.Tensor
        Center frequency or frequencies in Hz, shape (batch,) or scalar.
    r : torch.Tensor
        Notch radius between 0 and 1 controlling filter sharpness. Values closer to 1
        produce sharper, narrower resonances; values closer to 0 produce wider resonances.
        Shape must match f or be broadcastable to f.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    b : torch.Tensor
        Numerator coefficients of shape (batch, 3) or (3,).
    a : torch.Tensor
        Denominator coefficients of shape (batch, 3) or (3,).
    """
    f = ensure_tensor(f, dtype=torch.float32)
    r = ensure_tensor(r, min_dims=f.ndim, dtype=torch.float32)

    v = 2 * torch.cos(2 * torch.pi * f / fs)
    c = (1 - r * r) / 2

    b = torch.zeros(v.shape + (3,)).to(f)
    b[..., 0] = c
    b[..., 2] = -c

    a = torch.ones(v.shape + (3,)).to(f)
    a[..., 1] = -r * v
    a[..., 2] = r * r

    return b, a


def design_butter_bandpass(
    fc: torch.Tensor,
    bw: torch.Tensor,
    fs: float,
    order: int = 2,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Design digital Butterworth bandpass filter coefficients.

    Creates an IIR bandpass filter using the bilinear transform method with an
    analog lowpass-to-bandpass prototype transformation. The resulting filter has
    order 2*order (each lowpass pole maps to two bandpass poles).

    The design follows the standard procedure:
    1. Design an analog Butterworth lowpass prototype with unit cutoff frequency
    2. Warp center frequency and bandwidth for digital filter design
    3. Apply LP to BP frequency transformation in the s-plane
    4. Transform from s-plane to z-plane using Tustin's bilinear method

    Parameters
    ----------
    fc : torch.Tensor
        Center frequency in Hz, shape (batch,) or scalar.
    bw : torch.Tensor
        Bandwidth in Hz (distance between -3 dB points), shape (batch,) or scalar.
    fs : float
        Sampling rate in Hz.
    order : int
        Prototype lowpass filter order. The resulting bandpass filter has order 2*order.
    dtype : torch.dtype
        Computation dtype. Use float64 for higher filter orders to avoid numerical issues.

    Returns
    -------
    b : torch.Tensor
        Numerator (feedforward) coefficients, shape (batch, 2*order+1) or (2*order+1,).
    a : torch.Tensor
        Denominator (feedback) coefficients, shape (batch, 2*order+1) or (2*order+1,).
    """
    fc = ensure_tensor(fc, dtype=dtype)
    bw = ensure_tensor(bw, dtype=dtype)

    # Warp center frequency and bandwidth to analog domain
    fc_w = 4 * torch.tan(torch.pi * fc / fs)  # warped center frequency
    bw_w = 4 * torch.tan(torch.pi * bw / fs)  # warped bandwidth

    # Analog Butterworth lowpass prototype poles (unit cutoff)
    m = torch.arange(-order + 1, order, 2, device=fc.device)
    proto_poles = -torch.exp(1j * torch.pi * m / (2 * order))  # shape (order,)

    # LP→BP transformation: each prototype pole s_lp maps to two poles via
    #   s^2 - s_lp * bw_w * s + fc_w^2 = 0
    # giving s = (s_lp*bw_w ± sqrt((s_lp*bw_w)^2 - 4*fc_w^2)) / 2
    bw_poles = bw_w[..., None] * proto_poles[None, :]  # shape (batch, order)
    discriminant = bw_poles**2 - 4 * (fc_w[..., None] ** 2)  # shape (batch, order)
    sqrt_disc = torch.sqrt(discriminant.to(torch.complex128 if dtype == torch.float64 else torch.complex64))
    bp_poles = torch.cat([(bw_poles + sqrt_disc) / 2, (bw_poles - sqrt_disc) / 2], dim=-1)  # shape (batch, 2*order)

    # Bandpass gain: bw_w^order (matches scipy's normalisation)
    gain = bw_w**order

    # Bilinear transform: s-plane → z-plane
    p = (4 + bp_poles) / (4 - bp_poles)

    # Zeros: order zeros at z=+1 (DC) and order zeros at z=-1 (Nyquist)
    z = torch.cat([torch.ones_like(p[..., :order]), -torch.ones_like(p[..., :order])], dim=-1)  # shape (batch, 2*order)

    k = gain / torch.real(torch.prod(4 - bp_poles, dim=-1))

    b = k[..., None] * torch.real(poly(z))
    a = torch.real(poly(p))

    # normalize to unit amplitude at fc
    n = torch.arange(b.shape[-1], device=b.device, dtype=b.dtype)
    z_inv = torch.exp(-1j * 2 * torch.pi * fc[..., None] / fs * n)  # z^{-n}
    B = torch.sum(b.to(z_inv.dtype) * z_inv, dim=-1)
    A = torch.sum(a.to(z_inv.dtype) * z_inv, dim=-1)
    H_fc = B / A
    b = b / torch.abs(H_fc)[..., None]

    return b, a


def design_butter_filter(fc: torch.Tensor, fs: float, order: int = 2, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Design digital Butterworth lowpass filter coefficients.

    Creates an IIR lowpass filter with maximally flat magnitude response in the passband
    using the bilinear transform method. Butterworth filters have the property
    that they have zero ripple in both the passband and stopband.

    The design method follows scipy.signal.iirfilter:
    1. Design an analog Butterworth prototype with unit cutoff frequency
    2. Warp frequencies for digital filter design
    3. Transform from s-plane to z-plane using Tustin's bilinear method

    Parameters
    ----------
    fc : torch.Tensor
        Cutoff frequency in Hz, shape (batch,) or scalar.
    fs : float
        Sampling rate in Hz.
    order : int
        Filter order. Higher orders produce steeper rolloff.
    dtype : torch.dtype
        Computation dtype. Use float64 for higher filter orders to avoid numerical issues.

    Returns
    -------
    b : torch.Tensor
        Numerator (feedforward) coefficients of shape (batch, order+1) or (order+1,).
    a : torch.Tensor
        Denominator (feedback) coefficients of shape (batch, order+1) or (order+1,).
    """
    fc = ensure_tensor(fc, dtype=dtype)
    fcn = fc / fs * 2  # normalized cutoff
    fcw = 4 * torch.tan(torch.pi * fcn / 2)  # warp frequencies for digital filter design

    # calculate poles of an analog Butterworth filter with cutoff 1
    m = torch.arange(-order + 1, order, 2).to(fc.device)
    proto_poles = -torch.exp(1j * torch.pi * m / (2 * order))

    # shift prototype poles to cutoff frequency
    lp_poles = fcw[..., None] * proto_poles[None, :]
    gain = fcw**order

    # transform from s- to z-plane using Tustin's method
    p = (4 + lp_poles) / (4 - lp_poles)
    z = -1 * torch.ones_like(p)
    k = gain / torch.real(torch.prod(4 - lp_poles, dim=-1))

    b = k[..., None] * torch.real(poly(z))
    a = torch.real(poly(p))

    return b, a


def design_fir_filter(
    f: torch.Tensor,
    m: torch.Tensor,
    fs: float,
    N: int,
    phase: Literal["linear", "minimum"] = "linear",
    return_fd: bool = False,
    min_mag: float = 1e-10,
) -> torch.Tensor:
    """Design an FIR filter based on desired magnitude response.

    Creates a finite impulse response (FIR) filter with prescribed magnitude response
    at specified frequencies using frequency sampling with optional window smoothing.
    Supports both linear-phase and minimum-phase designs.

    Parameters
    ----------
    f : torch.Tensor
        Frequencies at which desired magnitudes are specified, shape (M,) or (batch, M) in Hz.
    m : torch.Tensor
        Desired magnitudes at those frequencies, shape (M,) or (batch, M).
    fs : float
        Sampling rate in Hz.
    N : int
        Resulting filter length in samples. Shorter filters may not accurately represent
        low-frequency characteristics due to frequency resolution limitations.
    phase : Literal["linear", "minimum"]
        How the phase response of the filter should be calculated (default: "linear")
        Options:
        - "linear": return a linear-phase filter with a symmetric impulse response
        - "minimum": return a minimum-phase filter, see also [1]
    return_fd : bool
        If True, returns filter in frequency domain; if False, returns in time domain (default: False).
    min_mag : float
        Minimum magnitude floor to avoid log(0) issues with minimum-phase design (default: 1e-10).

    Returns
    -------
    res : torch.Tensor
        Filter coefficients either in time domain (shape (N,)) or frequency domain (shape ((N+1)//2+1,)),
        depending on `return_fd`.

    References
    ----------
    [1]: A. V. Oppenheim and R. W. Schafer, 'Discrete-time signal processing', pp. 781-787
    """
    f = ensure_tensor(f)
    m = ensure_tensor(m)

    if phase == "minimum":
        # use a longer FFT size to minimize the error
        # (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.minimum_phase.html)
        N_fft = 2 ** int(math.ceil(math.log2(2 * (N - 1) / 0.01)))
    else:
        N_fft = N

    f_fft = torch.fft.rfftfreq(N_fft, 1 / fs).to(f)
    X = interpolate_linear(f, m, f_fft, extrapolate="const")
    X = torch.clip(X, min=min_mag)

    if phase == "linear":
        # it's easy to shift in time domain to get a nice causal filter
        # (calculating linear phase in FD is probably more efficient)
        h = torch.fft.irfft(X.to(torch.cfloat))
        h = torch.roll(h, N // 2, -1)
    elif phase == "minimum":
        # get minimum phase filter via the windowed cepstrum

        # log_mag = torch.log(X)
        # log_mag = torch.cat([log_mag, log_mag.flip(-1)[..., 1:-1]], dim=-1)
        # min_phase = -hilbert(log_mag).imag
        # frequency_response = torch.exp(log_mag + 1j * min_phase)
        # h = torch.fft.ifft(frequency_response, dim=-1).real

        cepstrum_window = torch.zeros(N_fft).to(X)
        cepstrum_window[0] = 1
        cepstrum_window[1 : N_fft // 2] = 2
        X_l = torch.log(X).to(torch.cfloat)
        C = torch.fft.irfft(X_l, N_fft)
        X_mp = torch.exp(torch.fft.rfft(C * cepstrum_window))
        h = torch.real(torch.fft.irfft(X_mp))
        h = h[..., :N]
    else:
        raise ValueError(f"Phase calculation method '{phase}' unknown.")

    return torch.fft.rfft(h) if return_fd else h


def iir_freq_sampling(b: torch.Tensor, a: torch.Tensor, x: torch.Tensor, N: int = 1024) -> torch.Tensor:
    """Apply IIR filter using frequency sampling method with frame-based processing.

    This method transforms signal and filter to frequency domain in frames, multiplies them and transforms back.

    Parameters
    ----------
    b : torch.Tensor
        Numerator (feedforward) coefficients of shape (batch, order) or (batch, frames, order).
    a : torch.Tensor
        Denominator (feedback) coefficients of shape (batch, order) or (batch, frames, order).
        The first coefficient must be 1.0.
    x : torch.Tensor
        Input signal of shape (batch, length).
    N : int
        Length of the sampled impulse response in time domain (sampling (N+1)//2+1 points in frequency domain).

    Returns
    -------
    y : torch.Tensor
        Filtered signal of same shape as input `x`.
    """
    x = ensure_tensor(x)
    b = ensure_tensor(b)
    a = ensure_tensor(a)

    # add frame dimension if necessary
    if len(b.shape) == 2:
        b = b.unsqueeze(1)
    if len(a.shape) == 2:
        a = a.unsqueeze(1)

    # calculate with larger dtype to avoid numerical issues with more extreme filters
    H = freqz(b, a, N, dtype=torch.complex128).type(torch.complex64)

    # transform to time domain for padding
    h = torch.fft.irfft(H)
    h_pad = torch.nn.functional.pad(h, (0, N))
    H_pad = torch.fft.rfft(h_pad)

    # reshape (i.e., split into frames of length N) and pad input signal
    B, L = x.shape
    L_p = math.ceil(L / N) * N
    x_pad = torch.nn.functional.pad(x, (0, L_p - L))
    x_frames = x_pad.view(B, -1, N)
    x_frames_pad = torch.nn.functional.pad(x_frames, (0, N))
    X = torch.fft.rfft(x_frames_pad)

    # check that the number of frames is divisible by the number of filters, and repeat filters as necessary
    repeats = X.shape[-2] // H_pad.shape[-2]
    assert X.shape[-2] / H_pad.shape[-2] == repeats, (
        "Number of frames must be divisible by the number of filters. Possibly, N can be changed to achieve this."
    )
    H_pad = torch.repeat_interleave(H_pad, repeats, dim=-2)

    # perform frequency domain convolution
    Y = X * H_pad
    y = torch.fft.irfft(Y)

    ola = torch.eye(y.shape[-1], requires_grad=False).unsqueeze(1).to(x)
    y = torch.nn.functional.conv_transpose1d(y.transpose(1, 2), ola, stride=N, padding=0).squeeze(1)

    return y[..., :L]
