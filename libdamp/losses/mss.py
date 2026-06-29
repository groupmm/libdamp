"""Multi-Scale Spectral Loss.

This module is part of the libdamp package.
"""

from collections.abc import Sequence

import gin
import torch

from libdamp.helpers.transforms import get_window

__all__ = ["MSSLoss"]


@gin.configurable
class MSSLoss(torch.nn.Module):
    """Multi-scale spectral loss."""

    def __init__(
        self,
        w_mag: float | Sequence[float] = 1.0,
        w_log: float | Sequence[float] = 0.0,
        w_mlog: float | Sequence[float] = 0.0,
        mgamma: float = 1.0,
        p: int = 1,
        fft_sizes: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64),
        win: str = "hann",
    ) -> None:
        """Multi-scale spectral loss.

        Parameters
        ----------
        w_mag : float | Sequence[float]
            Weight for linear magnitude comparison.
            If a sequence is provided, it should have the same length as `fft_sizes`,
            and each weight will be applied to the corresponding scale.
        w_log : float | Sequence[float]
            Weight for log-magnitude comparison.
            If a sequence is provided, it should have the same length as `fft_sizes`,
            and each weight will be applied to the corresponding scale.
        w_mlog : float | Sequence[float]
            Weight for modified ("Muller") log-compressed magnitude comparison `log(1 + gamma * x)`.
            If a sequence is provided, it should have the same length as `fft_sizes`,
            and each weight will be applied to the corresponding scale.
        mgamma : float
            Compression parameter for modified log-compression.
        p : int
            Exponent for difference between spectrograms.
        fft_sizes : Iterable
            List of window sizes for the different spectrogram scales.
        win : str
            Window type (see `libdamp.helpers.transforms.get_window`).
        """
        super().__init__()

        self.w_mag = [w_mag] * len(fft_sizes) if isinstance(w_mag, (int, float)) else w_mag
        self.w_log = [w_log] * len(fft_sizes) if isinstance(w_log, (int, float)) else w_log
        self.w_mlog = [w_mlog] * len(fft_sizes) if isinstance(w_mlog, (int, float)) else w_mlog
        self.mgamma = mgamma
        self.p = p
        self.fft_sizes = fft_sizes
        self.win = win

    def update(self, w_mag=None, w_log=None, w_mlog=None, mgamma=None, p=None, fft_sizes=None, win=None):
        """Update loss parameters. Useful for dynamic loss balancing / scheduling."""
        if w_mag is not None:
            self.w_mag = [w_mag] * len(self.fft_sizes) if isinstance(w_mag, (int, float)) else w_mag
        if w_log is not None:
            self.w_log = [w_log] * len(self.fft_sizes) if isinstance(w_log, (int, float)) else w_log
        if w_mlog is not None:
            self.w_mlog = [w_mlog] * len(self.fft_sizes) if isinstance(w_mlog, (int, float)) else w_mlog
        if mgamma is not None:
            self.mgamma = mgamma
        if p is not None:
            self.p = p
        if fft_sizes is not None:
            self.fft_sizes = fft_sizes
        if win is not None:
            self.win = win

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Forward pass with two input signals to compare.

        Parameters
        ----------
        y : torch.Tensor
            Reference audio signal, shape (batch, samples).
        y_hat : torch.Tensor
            Estimated audio signal, shape (batch, samples).

        Returns
        -------
        torch.Tensor
            Loss value per batch item, shape (batch,).
        """
        loss = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)

        for i, N in enumerate(self.fft_sizes):
            w = get_window(self.win, N).type_as(y)

            Y = torch.abs(torch.stft(y, n_fft=N, hop_length=N // 2, window=w, center=False, return_complex=True))
            Y_hat = torch.abs(torch.stft(y_hat, n_fft=N, hop_length=N // 2, window=w, center=False, return_complex=True))

            if self.w_mag[i] > 0:
                loss += self.w_mag[i] * torch.mean(torch.abs(Y - Y_hat) ** self.p, axis=(-2, -1))

            if self.w_log[i] > 0:
                Y_log = torch.log(torch.clip(Y, min=1e-7))
                Y_hat_log = torch.log(torch.clip(Y_hat, min=1e-7))
                loss += self.w_log[i] * torch.mean(torch.abs(Y_log - Y_hat_log) ** self.p, axis=(-2, -1))

            if self.w_mlog[i] > 0:
                Y_log = torch.log(1 + self.mgamma * Y)
                Y_hat_log = torch.log(1 + self.mgamma * Y_hat)
                loss += self.w_mlog[i] * torch.mean(torch.abs(Y_log - Y_hat_log) ** self.p, axis=(-2, -1))

        return loss
