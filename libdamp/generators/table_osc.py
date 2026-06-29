"""Wave/Pulse Table Oscillator that reads waveforms from a table.

This module is part of the libdamp package.
"""

import math
from typing import Literal

import torch

from ..helpers.incremental_mod import incremental_mod
from ..helpers.tensors import ensure_tensor, interpolate_samples
from .generator import Generator


class TableOsc(Generator):
    """Wave/Pulse Table Oscillator for reading out waveforms from a table."""

    def __init__(
        self,
        N: int,
        table,
        table_param,
        fs: float,
        mode: Literal["pulse", "wave"] = "pulse",
        normalize: Literal["power", "peak"] | None = None,
        learnable_table: bool = False,
        interp_f0: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
        interp_ts: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
        interp_entry: bool = True,
        table_freq=None,
        table_mask: torch.Tensor | None = None,
    ):
        """Wave/Pulse Table Oscillator for reading out waveforms from a table.

        Parameters
        ----------
        N : int
            Number of samples per frame.
        table : tensor or tensor-like
            Lookup table with shape (M, L) or (B, M, L), where B is the (optional) batch size,
            M is the number of different entries (same as `table_param`), and L is the length of each entry in samples
        table_param : tensor or tensor-like
            Sorting parameter for each entry in the table, shape (M).
            The corresponding entry is selected by the `table_select` parameter given in `update()`
        fs : float
            Sampling rate in Hz.
        mode : Literal["pulse", "wave"]
            either "pulse" or "wave" to select which table read mode to use (default: "pulse")
        normalize : Literal["power", "peak"] | None
            Normalization method for the table, either "power" for equal power in each entry or "peak" (default: None)
        learnable_table : bool
            Store the table as a trainable `torch.nn.Parameter()`
        interp_f0 : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for: F0 per frame -> instantaneous frequency per sample.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        interp_ts : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for: `table_select` parameter per frame -> `table_select` parameter per sample.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        interp_entry : bool
            Whether or not to interpolate between entries (using bilinear interpolation) when the `table_select`
            parameter lies in between two entries (default: True)
        table_freq : tensor, tensor-like or None
            Only used in "wave" mode.
            The fundamental frequency that each waveform entry in the table is designed for or represents.
        table_mask : tensor or None
            Optional mask that will be applied to the table before any processing happens (e.g. to learn pulses
            of different length in the same table). Must have a compatible shape with the table.
        """
        super().__init__()

        self.fs = fs
        self.interp_f0 = interp_f0
        self.interp_ts = interp_ts
        self.interp_entry = interp_entry

        self.N = N  # number of samples per frame

        table_param = ensure_tensor(table_param, dtype=torch.float32)

        if table.ndim == 2:
            # add batch dimension
            table = table[None, :]

        _, self.M, self.L = table.shape  # (number of entries, length of each entry in samples)

        # sort entries by pitch (ascending)
        sorting = torch.argsort(table_param)
        pw_table = ensure_tensor(table, dtype=torch.float32).index_select(index=sorting, dim=1)
        pw_param = table_param[sorting]
        if table_freq is not None:
            self.fixed_table_freq = False
            pw_freq = ensure_tensor(table_freq, dtype=torch.float32)[sorting]
        else:
            self.fixed_table_freq = True
            pw_freq = torch.Tensor([self.fs / self.L])  # assuming one period in the wavetable

        # optional normalization
        if normalize == "power":
            pw_table = pw_table / pw_table.norm(dim=-1, keepdim=True) * math.sqrt(pw_table.shape[1])
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

        self.register_buffer("pw_param", pw_param)
        self.register_buffer("pw_freq", pw_freq)

        if table_mask is not None:
            if table_mask.ndim == 2:
                # add batch broadcast dimension
                table_mask = table_mask[None, :]

            self.register_buffer("table_mask", table_mask)
        else:
            self.table_mask = None

        self.clear()

    def generate(self):
        B, F = self.f0.shape  # batch size, number of frames

        # calculate instantaneous F0 for each sample
        f0_inst = interpolate_samples(self.f0.unsqueeze(1), self.N, mode=self.interp_f0, prev_val=self.prev_f0)
        self.prev_f0 = f0_inst[..., -1]
        f0_inst = f0_inst.squeeze(1)

        # interpolate parameter for table lookup
        if self.interp_ts == "const":
            # if F0 is constant per frame, we only need a frame-wise selection of the table entries
            ts_inst = self.ts
        else:
            ts_inst = interpolate_samples(self.ts.unsqueeze(1), self.N, mode=self.interp_ts, prev_val=self.prev_ts)
            self.prev_ts = ts_inst[..., -1]
            ts_inst = ts_inst.squeeze(1)

        upper_bound = torch.searchsorted(self.pw_param, ts_inst, side="right")
        # for pitches out of bounds, just use the boundary entry
        upper_bound[(upper_bound >= self.M)] = self.M - 1

        # also find lower bound and weightings if we interpolate between the entries
        if self.interp_entry:
            lower_bound = upper_bound - 1
            # for pitches out of bounds, just use the boundary entry
            lower_bound[(lower_bound < 0)] = 0

            # calculate a weight between upper and lower bound
            upper_param = self.pw_param[upper_bound]
            lower_param = self.pw_param[lower_bound]
            delta_param = upper_param - lower_param

            delta_param[(delta_param == 0)] = 1

            lower_weight = (upper_param - ts_inst) / delta_param
            upper_weight = 1.0 - lower_weight

        # calculate read pointer for each sample
        # the _incremental_mod is NOT the same as normal modulo!
        if self.read_pulse:
            read_sample = incremental_mod(self.fs / f0_inst, torch.ones_like(f0_inst))
        else:
            if self.fixed_table_freq:
                increment = f0_inst / self.pw_freq
            else:
                if self.interp_entry:
                    read_freq = lower_weight * self.pw_freq[lower_bound] + upper_weight * self.pw_freq[upper_bound]
                else:
                    read_freq = self.pw_freq[upper_bound]
                increment = f0_inst / read_freq
            read_sample = incremental_mod(torch.ones_like(increment) * self.L, increment)

        # create a table sampling grid with shape (B, N*F, 1, 2)
        # with time and phase axis in range [-1, 1]
        ph_inst_scaled = read_sample / self.L * 2 - 1
        t_inst_scaled = torch.arange(F * self.N, device=ph_inst_scaled.device, dtype=ph_inst_scaled.dtype)[None, :].broadcast_to(B, -1)
        t_inst_scaled = t_inst_scaled / (F * self.N) * 2 - 1
        sampling_grid = torch.stack([ph_inst_scaled, t_inst_scaled], dim=2).unsqueeze(2)

        table = self.pw_table

        if self.table_mask is not None:
            table = table * self.table_mask

        if table.shape[0] == 1:
            # expand to batch size
            table = self.pw_table.expand(B, -1, -1)

        idx = upper_bound.unsqueeze(-1).expand(-1, -1, table.shape[2])
        selected_entries = table.gather(1, idx)

        y_upper = torch.nn.functional.grid_sample(
            selected_entries.unsqueeze(1),  # add channel dimension, resulting in (B, 1, N*F, L)
            sampling_grid,  # (B, N*F, 1, 2)
            mode="bilinear",
            padding_mode="zeros",  # this allows for reading outside of the table in the pulse table case
            align_corners=True,  # "(-1 and 1) are considered as the center points of the input’s corner pixels"
        ).squeeze([1, 3])

        if self.interp_entry:
            # interpolate waveform bilinearly between different table entries and phase progression
            # if we don't want to interpolate, we need to align the waves perfectly
            idx = lower_bound.unsqueeze(-1).expand(-1, -1, table.shape[2])
            selected_entries = table.gather(1, idx)

            y_lower = torch.nn.functional.grid_sample(
                selected_entries.unsqueeze(1),  # (B, 1, N*F, L)
                sampling_grid,  # (B, N*F, 1, 2)
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze([1, 3])

            if self.interp_ts == "const":
                lower_weight = interpolate_samples(lower_weight, self.N, mode="const")
                upper_weight = interpolate_samples(upper_weight, self.N, mode="const")

            y = lower_weight * y_lower + upper_weight * y_upper
        else:
            y = y_upper

        return y

    def update(self, f0, table_select):
        """Update the fundamental frequency and table selection parameter.

        Parameters
        ----------
        f0 : torch.Tensor or array-like
            Fundamental frequencies, shape (B, frames) in Hz.
        table_select : torch.Tensor or array-like
            Table selection parameter (matched against `table_param`), shape (B, frames).
        """
        self.f0 = ensure_tensor(f0, dtype=torch.float32)
        self.ts = ensure_tensor(table_select, dtype=torch.float32)

        assert len(self.f0.shape) == 2, "TableOsc expects F0 to have shape (B, frames)."
        assert self.ts.shape == self.f0.shape, "TableOsc expects F0 and table selection parameter to have the same shape."

        if self.prev_f0 is not None:
            if self.f0.shape[0] != self.prev_f0.shape[0]:
                # if the batch size changed, we start from scratch
                self.clear()

    def clear(self):
        self.prev_f0 = None
        self.prev_ts = None
