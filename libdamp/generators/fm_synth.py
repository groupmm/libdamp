"""Frequency Modulation (FM) synthesis generator.

This module is part of the libdamp package.
"""

from collections import deque
from typing import Literal

import torch

from ..helpers.tensors import ensure_tensor, interpolate_samples
from .generator import Generator


class FMSynth(Generator):
    """Frequency Modulation (FM) Synthesis"""

    def __init__(
        self,
        N: int,
        fs: float,
        P: int,
        connections: list[tuple[int, int]],
        interp_f0: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
        interp_r: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
        interp_m: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
    ):
        """Frequency Modulation (FM) Synthesis

        Parameters
        ----------
        N : int
            number of samples per frame
        fs : float
            sampling rate in Hz
        P : int
            number of operators
        connections : List[Tuple[int, int]]
            list of connections, given as integer tuples (src, dst) (i.e., edges of a directed graph)
        interp_f0 : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the instantaneous fundamental frequency at each sample from the given frequency for each frame.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        interp_r : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the frame-wise ratio for each operator.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        interp_m : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation method for the frame-wise gain for each operator.
            For options, see `helpers.tensors.interpolate_samples` (default: "const")
        """
        super().__init__()
        self.N = N
        self.fs = fs
        self.interp_f0 = interp_f0
        self.interp_r = interp_r
        self.interp_m = interp_m

        assert len(connections) >= P - 1, f"At least {P - 1} connections required when using {P} operators."

        # in the following, we use Kahn's algorithm to determine in which order the operators are processed
        # the goal is to find a processing order and store which operator outputs feed into each operator's
        # phase calculation
        self.processing_order = []
        self.receives = {}

        # 1. create empty in/out degree counters
        in_deg = {}
        out_deg = {}
        for p in range(P):
            in_deg[p] = 0
            out_deg[p] = 0

        # 2. store the sends and receives
        sends = {}
        for src, dst in connections:
            if src not in sends:
                sends[src] = []
            if dst not in self.receives:
                self.receives[dst] = []

            self.receives[dst].append(src)
            sends[src].append(dst)
            in_deg[dst] += 1
            out_deg[src] += 1

        for p in range(1, P):
            assert out_deg[p] >= 1, f"Operator {p} is not connected."

        # 3. add root nodes to a FIFO queue
        queue = deque()
        for p in range(P):
            if in_deg[p] == 0:
                queue.append(p)

        # 4. use Kahn’s algorithm to topologically sort the nodes
        while queue:
            p = queue.popleft()
            self.processing_order.append(p)
            if p not in sends:
                continue
            for q in sends[p]:
                in_deg[q] -= 1
                if in_deg[q] == 0:
                    queue.append(q)

        self.clear()  # reset generator state

    def generate(self):
        assert self.initialized, "update() must be called at least once before generate()"

        f0_inst = interpolate_samples(self.f0.unsqueeze(1), self.N, mode=self.interp_f0, prev_val=self.prev_f0)[:, 0, :]
        r_inst = interpolate_samples(self.r, self.N, mode=self.interp_r, prev_val=self.prev_r)
        m_inst = interpolate_samples(self.m, self.N, mode=self.interp_m, prev_val=self.prev_m)

        # save state for next call
        self.prev_f0 = f0_inst[:, -1]
        self.prev_r = r_inst[:, :, -1]
        self.prev_m = m_inst[:, :, -1]

        # not checking for Nyquist aliasing here – in FM this may be intentional?

        # generate signal
        x = {}
        for p in self.processing_order:
            # own frequency-accumulated phase: this is the only part that carries over between
            # calls, since the modulation added below is an instantaneous per-sample contribution
            # (not itself accumulated), matching what an uninterrupted single-call generation does.
            phi = torch.cumsum(2 * torch.pi * f0_inst * r_inst[:, p, :] / self.fs, dim=-1)
            if p in self.prev_phi:  # add offset from previous generation if available
                phi += self.prev_phi[p][:, None]
            # save phase state for next call (wrapped to keep it bounded), before adding modulation
            self.prev_phi[p] = phi[:, -1] % (2 * torch.pi)

            if p in self.receives:
                for q in self.receives[p]:
                    phi += x[q]  # the processing order ensures that x[q] already exists
            x[p] = m_inst[:, p, :] * torch.sin(phi)

        return x[0]

    def update(self, f0, r, m):
        assert f0.ndim == 2, "f0 should have dimensions (batch, num_frames)."
        assert m.ndim == 3, "m should have dimensions (batch, num_frames, num_operators)."
        assert r.ndim == 3, "r should have dimensions (batch, num_frames, num_operators)."

        if self.initialized and self.f0.shape[0] != f0.shape[0]:
            # if the batch size changed, we start from scratch
            self.clear()

        self.f0 = ensure_tensor(f0)  # (batch, num_frames)
        self.r = ensure_tensor(r).transpose(-2, -1)  # (batch, num_operators, num_frames)
        self.m = ensure_tensor(m).transpose(-2, -1)  # (batch, num_operators, num_frames)

        self.initialized = True

    def clear(self):
        self.prev_f0 = None
        self.prev_r = None
        self.prev_m = None
        self.prev_phi = {}

        self.initialized = False
