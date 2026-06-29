"""RMS signal energy loss function.

This module is part of the libdamp package.
"""

import gin
import torch

from libdamp.helpers.tensors import ensure_tensor
from libdamp.helpers.transforms import get_window


@gin.configurable
class RMSLoss(torch.nn.Module):
    """RMS signal energy loss function.

    Compares the energy of two input signals within a moving window.
    """

    def __init__(self, N: int = 512, H: int = 256, window_type: str = "hann") -> None:
        """Initialize RMS signal energy loss function.

        Parameters
        ----------
        N : int
            Moving window size in samples (default: 512).
        H : int
            Windowing hop size in samples (default: 256).
        window_type : str
            Window type used for segmenting the signal (default: "hann").
        """
        super().__init__()

        self.H = H
        self.register_buffer("win", get_window(window_type, N)[None, None].to(torch.float32))
        self.win /= N

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Compare energy of two input signals using moving window RMS.

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
        y = ensure_tensor(y, min_dims=2)
        y_hat = ensure_tensor(y_hat, min_dims=2)
        assert y.ndim == 2, "No channel dimension supported."

        rms_y = torch.sqrt(torch.nn.functional.conv1d(y[:, None, :] ** 2, self.win, stride=self.H, padding="valid"))[:, 0, :]
        rms_y_hat = torch.sqrt(torch.nn.functional.conv1d(y_hat[:, None, :] ** 2, self.win, stride=self.H, padding="valid"))[:, 0, :]

        return torch.mean(torch.abs(10 * torch.log10((rms_y_hat + 1e-6) / (rms_y + 1e-6))), dim=-1)
