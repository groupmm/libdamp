"""Collection of torch helper modules.

This module is part of the libdamp package.
"""

from typing import Literal

import torch

from libdamp.helpers.freq import hz2midi
from libdamp.helpers.tensors import ensure_tensor, tensor_linspace


class SelectItem(torch.nn.Module):
    """Select an output in a torch.nn.Sequential pipeline."""

    def __init__(self, item_index: int) -> None:
        """Select an output in a torch.nn.Sequential pipeline.

        Parameters
        ----------
        item_index : int
            The index of the item that should be forwarded to the next module in the pipeline.
        """

        super().__init__()
        self._name = "selectitem"
        self.item_index = item_index

    def forward(self, inputs):
        """Return only the selected item from a tuple/list of inputs.

        Parameters
        ----------
        inputs : tuple or list
            A sequence of items from which the indexed item will be selected.

        Returns
        -------
        object
            The item at index specified during initialization.
        """
        return inputs[self.item_index]


class ConvStack(torch.nn.Module):
    """Stack of convolutional layers.
    (adapted from https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/transcriber.py)
    """

    def __init__(self, N_inp: int, N_out: int) -> None:
        super().__init__()

        self.cnn = torch.nn.Sequential(
            # layer 0
            torch.nn.Conv2d(1, N_out // 16, (3, 3), padding=1),
            torch.nn.BatchNorm2d(N_out // 16),
            torch.nn.ReLU(),
            # layer 1
            torch.nn.Conv2d(N_out // 16, N_out // 16, (3, 3), padding=1),
            torch.nn.BatchNorm2d(N_out // 16),
            torch.nn.ReLU(),
            # layer 2
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.25),
            torch.nn.Conv2d(N_out // 16, N_out // 8, (3, 3), padding=1),
            torch.nn.BatchNorm2d(N_out // 8),
            torch.nn.ReLU(),
            # layer 3
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.Dropout(0.25),
        )
        self.fc = torch.nn.Sequential(torch.nn.Linear((N_out // 8) * (N_inp // 4), N_out), torch.nn.Dropout(0.5))

    def forward(self, mel):
        """Implementation of forward step.

        Parameters
        ----------
        mel : torch.Tensor
            Mel spectrogram input with shape (B, 1, F, N_inp)."""
        x = torch.transpose(mel, -2, -1)  # swap time and frequency dimension
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class FreqToBins(torch.nn.Module):
    """Convert frequencies to a binned distribution."""

    def __init__(
        self,
        num_bins: int = 360,
        f_min: float = 32.7,
        f_max: float = 1975.5,
        target_smoothing: float = 20.0,
        out_of_bounds: Literal["snap", "smooth"] = "snap",
    ):
        """Convert frequencies to binned Gaussian distributions.

        Each input frequency is converted to a binned Gaussian distribution.
        The bins have a fixed width in cents, and the Gaussian is centered around the target frequency.
        The width of the Gaussian can be controlled with the `target_smoothing` parameter.

        Parameters
        ----------
        num_bins : int
            Number of bins in one distribution, bins have fixed width in cents.
        f_min : float or torch.Tensor
            Center frequency of the lowest bin in Hz.
        f_max : float or torch.Tensor
            Center frequency of the highest bin in Hz.
        target_smoothing : float or torch.Tensor
            Width of the Gaussian around the target frequency.
        out_of_bounds : str
            How to deal with values that are below `f_min` or above `f_max`.
            "snap" (default): set the lowest or highest bin to 1
            "smooth": still calculate the smoothed target normally, with the effect that the "visible"
                      probability distribution behaves unexpectedly for out-of-bounds values
        """
        super().__init__()
        self.N = num_bins
        self.out_of_bounds = out_of_bounds
        assert out_of_bounds in ["snap", "smooth"], f"Unknown out of bounds behavior '{out_of_bounds}'."

        f_min = ensure_tensor(f_min)
        f_max = ensure_tensor(f_max)
        target_smoothing = ensure_tensor(target_smoothing / 100)

        max_oct = torch.log2(f_max / f_min)
        midi_steps = 12 * tensor_linspace(torch.zeros_like(max_oct), max_oct, self.N)

        # using register_buffer to make sure the variable ends up on the right device with Lightning
        self.register_buffer("f_min", f_min)
        self.register_buffer("f_max", f_max)
        self.register_buffer("target_smoothing", target_smoothing)
        self.register_buffer("midi_steps", midi_steps)

    def forward(self, f):
        """Forward pass to convert single frequencies to binned distributions.

        Parameters
        ----------
        f : torch.Tensor
            Frequency values in Hz, shape (B, ..., K).

        Returns
        -------
        torch.Tensor
            Normalized probability distribution over bins, shape (B, ..., K, num_bins).
        """
        p = hz2midi(f, a4_ref=self.f_min, zero_val=0) - 69
        y = torch.exp(-1 * (p[..., None] - self.midi_steps[..., None, :]) ** 2 / (2 * self.target_smoothing**2))

        if self.out_of_bounds == "snap":
            mask_lower = f < self.f_min
            y[mask_lower, :] = torch.zeros(self.N).to(y)
            y[mask_lower, [0]] = 1
            mask_higher = f > self.f_max
            y[mask_higher, :] = torch.zeros(self.N).to(y)
            y[mask_higher, [-1]] = 1

        return y / (y.sum(dim=-1, keepdim=True) + torch.finfo(torch.float32).eps)


class LogitsToFreq(torch.nn.Module):
    """Convert bin-frequency network outputs to frequencies."""

    def __init__(self, bins_per_freq: int = 360, f_min: float = 32.7, f_max: float = 1975.5):
        """Convert bin-frequency network outputs to frequencies.

        This uses the CREPE [1] method of weighted average (Eq. 2 in [1])

        Parameters
        ----------
        bins_per_freq : int
            Number of bins in one distribution, bins have fixed width in cents.
        f_min : float or torch.Tensor
            Center frequency of the lowest bin in Hz.
        f_max : float or torch.Tensor
            Center frequency of the highest bin in Hz.

        References
        ----------
        [1]: J. W. Kim, J. Salamon, P. Li, and J. P. Bello,
             "CREPE: A Convolutional Representation for Pitch Estimation,”
             Feb. 2018, Available: http://arxiv.org/abs/1802.06182

        """
        super().__init__()
        self.N = bins_per_freq

        # f_min/f_max are kept at least 1D so that `bin_freqs` below has a leading axis of size 1,
        # which `forward()` relies on to broadcast against an arbitrary number of frequency groups.
        f_min = ensure_tensor(f_min, min_dims=1)
        f_max = ensure_tensor(f_max, min_dims=1)

        max_oct = torch.log2(f_max / f_min)
        oct_steps = tensor_linspace(torch.zeros_like(max_oct), max_oct, self.N)
        # using register_buffer to make sure the variable ends up on the right device with Lightning
        self.register_buffer("bin_freqs", f_min[..., None] * torch.pow(2, oct_steps))

    def forward(self, x):
        """Forward pass to convert binned frequency outputs to frequencies using weighted average.

        Parameters
        ----------
        x : torch.Tensor
            Logits tensor whose last dimension is a multiple of `bins_per_freq`.

        Returns
        -------
        torch.Tensor
            Estimated frequencies in Hz with shape x.shape[:-1] + (x.shape[-1] // bins_per_freq,).
        """
        shape = list(x.shape)
        L = shape.pop()
        assert L % self.N == 0, "Last axis not divisible by number of bins."
        shape.append(L // self.N)
        shape.append(self.N)

        x = torch.nn.functional.softmax(x.view(shape), dim=-1)
        return torch.sum(torch.einsum("...mn,mn->...mn", x, self.bin_freqs), dim=-1)
