"""Collection of helper functions and classes related to signal transforms.

This module is part of the libdamp package.
"""

import scipy.signal.windows
import torch

__all__ = ["get_window", "stft", "istft", "hilbert"]


def get_window(win_type: str, win_length: int) -> torch.Tensor:
    """Return a window function.

    Parameters
    ----------
    win_type : str
        window type, can either be a function name of a torch window function [1] or a window name recognized by
        `scipy.signal.windows.get_window()` [2].
    win_length : int
        window length in samples

    Returns
    -------
    win : torch.Tensor
        a 1D tensor containing the sampled window function

    References
    ----------
    [1]: https://pytorch.org/docs/stable/torch.html#spectral-ops
    [2]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html
    """

    try:
        win = getattr(torch, win_type)(win_length)
    except AttributeError:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))

    return win


def stft(
    x: torch.Tensor, N: int, H: int, win_type: str = "hann", with_phase: bool = True, center: bool = True
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Calculate the short-time Fourier transform (STFT) of a signal.

    Parameters
    ----------
    x : torch.Tensor
        Input audio signal(s) where the STFT is calculated along the last axis.
    N : int
        Window and transform size in samples.
    H : int
        Hop size in samples.
    win_type : str
        Window function to use for the STFT. See `get_window()` for options
        (default: "hann").
    with_phase : bool
        Whether to return magnitude and phase separately (True) or magnitude only (False) (default: True).
    center : bool
        Whether to center-pad the signal before STFT (default: True).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] or torch.Tensor
        If with_phase=True: (magnitude, phase) tuple, each shape (..., freq_bins, time_frames).
        If with_phase=False: magnitude-only tensor of shape (..., freq_bins, time_frames).
    """
    w = get_window(win_type, N).to(x.dtype)
    X = torch.stft(x, N, H, N, w, return_complex=True, center=center)
    Y = torch.abs(X).to(x.dtype)

    if with_phase:
        Y_ph = torch.angle(X).to(x.dtype)
        return Y, Y_ph

    return Y


def istft(X: torch.Tensor, N: int, H: int, win_type: str = "hann") -> torch.Tensor:
    """Calculate the inverse STFT of a signal.

    Parameters
    ----------
    X : torch.Tensor
        complex input STFT(s), where the inverse transform is calculated on the last two axes
    N : int
        window and transform size in samples
    H : int
        hop size in samples
    win_type : str
        window function to be used for the STFT (see `get_window()` for options, default: "hann")

    Returns
    -------
    x : torch.Tensor
        time-domain signal reconstructed from the input STFT
    """
    w = get_window(win_type, N).to(X.real.dtype)
    return torch.istft(X, n_fft=N, hop_length=H, win_length=N, window=w)


def hilbert(x: torch.Tensor) -> torch.Tensor:
    """Hilbert transform along the last dimension of input signal `x`.

    Parameters
    ----------
    x : torch.Tensor
        Input signal of shape (..., num_samples).

    Returns
    -------
    torch.Tensor, shape (..., num_samples)
        Complex-valued analytic signal with the same shape as input.
    """
    N = x.shape[-1]
    x = torch.as_tensor(x, dtype=torch.complex64)
    X = torch.fft.fft(x, n=N, dim=-1)

    # window the spectrum and throw away negative frequencies
    if N % 2 == 0:
        n1 = int(N / 2)
        n2 = n1 + 1
    else:
        n1 = int((N + 1) / 2)
        n2 = n1

    positive = torch.arange(1, n1)
    negative = torch.arange(n2, N)

    X[..., positive] *= 2
    X[..., negative] *= 0

    return torch.fft.ifft(X, n=N, dim=-1)
