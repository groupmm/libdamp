"""Wave/Pulse Table Oscillator that weights interpolated waveforms from a table.

This module is part of the libdamp package.
"""

import math
from typing import Literal

import torch

from ..helpers.incremental_mod import incremental_mod
from ..helpers.tensors import ensure_tensor, interpolate_samples
from .generator import Generator


class WeightedTableOsc(Generator):
    """Wave/Pulse Table Oscillator for reading out waveforms from a table."""

    def __init__(
        self,
        N: int,
        table,
        fs: float,
        mode: Literal["pulse", "wave"] = "pulse",
        normalize: Literal["power", "peak"] | None = None,
        learnable_table: bool = False,
        interp_f0: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
        interp_w: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
        table_freq: float | None = None,
        table_mask: torch.Tensor | None = None,
    ):
        """Wave/Pulse Table Oscillator for reading out waveforms from a table. As opposed to `TableOsc`, this generator
        first interpolates all waveforms in the table and then weights them according to an externally given factor.

        Parameters
        ----------
        N : int
            Number of samples per frame.
        table : tensor or tensor-like
            2D (or 3D if batched) lookup table with dimensions (B, M, L), where B is the (optional) batch size,
            M is the number of different entries, and L is the length of each entry in samples
        fs : float
            Sampling rate in Hz.
        mode : Literal["pulse", "wave"]
            either "pulse" or "wave" to select which table read mode to use (default: "pulse")
        normalize : Literal["power", "peak"] | None
            Normalization method for the table, either "power" for equal power in each entry or "peak" (default: None)
        learnable_table : bool
            Whether or not to store the table as a trainable `torch.nn.Parameter()`
        interp_f0 : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the instantaneous frequency at each sample from the given F0 for each frame.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        interp_w : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the `weighting` parameter at each sample.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        table_freq : float or None
            Assigned frequency for all table entries if "wave" mode
        table_mask : tensor or None
            Optional mask that will be applied to the table before any processing happens (e.g. to learn pulses
            of different length in the same table). Must have a compatible shape with the table
        """
        super().__init__()

        self.fs = fs
        self.interp_f0 = interp_f0
        self.interp_w = interp_w

        self.N = N

        if table.ndim == 2:
            # add batch dimension
            table = table[None, :]

        _, self.M, self.L = table.shape

        pw_table = ensure_tensor(table, dtype=torch.float32)

        # a frequency for the entries is needed for wavetable mode
        if table_freq is not None:
            self.pw_freq = table_freq
        else:
            self.pw_freq = self.fs / self.L  # assuming one period in the wavetable

        # optional normalization
        if normalize == "power":
            pw_table = pw_table / pw_table.norm(dim=-1, keepdim=True) * math.sqrt(pw_table.shape[-1])
        elif normalize == "peak":
            pw_table = pw_table / torch.max(pw_table, dim=-1, keepdim=True).values
        elif normalize is None:
            pass
        else:
            raise ValueError(f"Unknown normalization method '{normalize}'.")

        if mode == "wave":
            self.read_pulse = False
        elif mode == "pulse":
            self.read_pulse = True
        else:
            raise ValueError(f"Unknown TableOsc mode '{mode}'.")

        # register buffers (for correct handling of devices, etc.)
        if learnable_table:
            self.register_parameter("pw_table", torch.nn.Parameter(pw_table))
        else:
            self.pw_table = pw_table  # not registering the table avoids saving and loading it in checkpoints

        if table_mask is not None:
            if table_mask.ndim == 2:
                # add batch dimension
                table_mask = table_mask[None, :]

            self.register_buffer("table_mask", table_mask)
        else:
            self.table_mask = None

        self.clear()

    def generate(self):
        B, _ = self.f0.shape

        # calculate instantaneous F0 for each sample
        f0_inst = interpolate_samples(self.f0.unsqueeze(1), self.N, mode=self.interp_f0, prev_val=self.prev_f0)
        self.prev_f0 = f0_inst[..., -1]
        f0_inst = f0_inst.squeeze(1)  # shape now (B, F*N)

        # interpolate parameter for table lookup
        w_inst = interpolate_samples(self.w.transpose(-2, -1), self.N, mode=self.interp_w, prev_val=self.prev_w)
        self.prev_w = w_inst[..., -1]
        w_inst = w_inst.squeeze(1)  # shape now (B, M, F*N)

        # calculate read pointer for each sample
        # the _incremental_mod is NOT the same as normal modulo!
        if self.read_pulse:
            # read table entries with normal speed and cut off or pad pulse, depending on the instantaneous F0
            read_sample = incremental_mod(self.fs / f0_inst, torch.ones_like(f0_inst))
        else:
            # read table entries with adjusted speed according to instantaneous F0 (squish or stretch table entries in time)
            increment = f0_inst / self.pw_freq
            read_sample = incremental_mod(torch.ones_like(increment) * self.L, increment)

        # create a table sampling grid with shape (B, F*N, 1, 2)
        # read_sample values are in range [0, L], need to scale to [-1, 1] for grid_sample function
        ph_inst_scaled = read_sample / self.L * 2 - 1
        unused_height = torch.zeros_like(ph_inst_scaled)
        sampling_grid = torch.stack([ph_inst_scaled, unused_height], dim=2).unsqueeze(2)

        table = self.pw_table
        if table.shape[0] == 1:
            # expand to batch size
            table = self.pw_table.expand(B, -1, -1)

        if self.table_mask is not None:
            table = table * self.table_mask

        y_all = torch.nn.functional.grid_sample(
            table.unsqueeze(2),  # (B, M, 1, L)
            sampling_grid,  # (B, N*F, 1, 2)
            mode="bilinear",
            padding_mode="zeros",  # this allows for reading outside of the table in the pulse table case
            align_corners=True,  # "(-1 and 1) are considered as the center points of the input’s corner pixels"
        ).squeeze(-1)

        y = torch.sum(w_inst * y_all, dim=1)

        return y

    def update(self, f0, weighting):
        """Update the fundamental frequency and per-entry table weights.

        Parameters
        ----------
        f0 : torch.Tensor or array-like
            Fundamental frequencies, shape (B, frames) in Hz.
        weighting : torch.Tensor or array-like
            Weight of each table entry, shape (B, frames, num_entries).
        """
        self.f0 = ensure_tensor(f0, dtype=torch.float32)
        self.w = ensure_tensor(weighting, dtype=torch.float32)

        assert self.f0.ndim == 2, "WeightedTableOsc expects F0 to have shape (B, F)."
        assert self.w.ndim == 3, "WeightedTableOsc expects table weights to have shape (B, F, M)."
        assert self.f0.shape[0:2] == self.w.shape[0:2]

        if self.prev_f0 is not None:
            if self.f0.shape[0] != self.prev_f0.shape[0]:
                # if the batch size changed, we start from scratch
                self.clear()

    def clear(self):
        self.prev_f0 = None
        self.prev_w = None
