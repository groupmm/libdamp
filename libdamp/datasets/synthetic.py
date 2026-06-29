"""Collection of synthetic datasets.

This module is part of the libdamp package.
"""

import math
from typing import Literal

import gin
import torch

from libdamp.helpers.freq import timbre2harmonics
from libdamp.helpers.tensors import ensure_tensor, interpolate_samples, smooth

__all__ = ["SyntheticSinusoidsDataset"]


def _create_f0_trajectories(
    N: int,
    M: int,
    f_base: float | torch.Tensor,
    method: Literal["random_walk", "log_sweep", "lin_sweep", "vibrato", "scale"] = "random_walk",
    rand_range: float = 256,
    rand_smoothing: int = 16,
    sweep_max_hz: float = 8000.0,
    vib_rate: float | torch.Tensor = 6,
    vib_phase: float | torch.Tensor = 0.2,
    vib_depth: float | torch.Tensor = 20,
    scale_direction: Literal["random", "up", "down"] = "random",
    scale_step_ct: float = 100,
    scale_step_length: int = 32,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate F0 trajectories based on the specified method.

    Parameters
    ----------
    N : int
        Number of trajectories.
    M : int
        Number of frames.
    f_base : float or torch.Tensor
        Initial frequency of the trajectory in Hz (can be scalar or shape (M,)).
    method : Literal["random_walk", "log_sweep", "lin_sweep", "vibrato", "scale"]
        Method to create the F0 trajectory.
    rand_range : float
        Range of one random walk step in cents.
    rand_smoothing : int
        Smoothing window size for the random walk trajectory.
    sweep_max_hz : float
        Highest reachable sweep frequency in Hz.
    vib_rate : float
        Vibrato rate in Hz.
    vib_phase : float
        Normalized initial phase of the vibrato in the interval [0, 1].
    vib_depth : float
        Depth of the vibrato in cents.
    scale_direction : Literal["random", "up", "down"]
        Direction of the scale steps when using the "scale" method.
    scale_step_ct : float
        Size of a scale step in cents.
    scale_step_length : int
        Number of frames for each scale step.
    rng : torch.Generator or None
        Random number generator for reproducibility (default: None).
    """

    f_base = ensure_tensor(f_base, min_dims=1, throw_larger=True)

    if method == "random_walk":
        f0_ct = smooth(torch.cumsum(torch.rand((N, M), generator=rng) * rand_range - rand_range // 2, axis=-1), rand_smoothing)
    elif method == "log_sweep":
        end_ct = torch.rand((N), generator=rng) * torch.log2(sweep_max_hz / f_base) * 1200
        # manual version of torch.linspace, since the function does not support tensors as inputs
        f0_ct = torch.arange(M) / M * end_ct[..., None]
    elif method == "lin_sweep":
        # manual version of torch.linspace, since the function does not support tensors as inputs
        dist_ct = torch.log2(torch.rand((N), generator=rng) * sweep_max_hz / f_base) * 1200
        f0_ct = torch.arange(M) / M * dist_ct
    elif method == "vibrato":
        vib_rate = ensure_tensor(vib_rate, min_dims=1, throw_larger=True)
        vib_phase = ensure_tensor(vib_phase, min_dims=1, throw_larger=True)
        vib_depth = ensure_tensor(vib_depth, min_dims=1, throw_larger=True)
        f0_ct = torch.sin(2 * torch.pi * (torch.arange(M) / M * vib_rate[..., None] + vib_phase[..., None]))
        f0_ct *= vib_depth[..., None]
    elif method == "scale":
        L = math.ceil(M / scale_step_length)
        if scale_direction == "random":
            f0_ct = torch.cumsum(torch.randint(2, (N, L), generator=rng) * 2 - 1, dim=-1) * scale_step_ct
            f0_ct = torch.repeat_interleave(f0_ct, scale_step_length, dim=-1)
        elif scale_direction == "up":
            f0_ct = torch.repeat_interleave(torch.arange(L)[None, :] * scale_step_ct, scale_step_length, dim=-1)
        elif scale_direction == "down":
            f0_ct = torch.repeat_interleave(torch.arange(0, -L, -1)[None, :] * scale_step_ct, scale_step_length, dim=-1)
        else:
            raise ValueError(f"Unknown scale direction '{scale_direction}'.")

        f0_ct = f0_ct[:, :M]
    else:
        raise ValueError(f"Unknown F0 trajectory method '{method}'.")

    return f_base[:, None] * torch.pow(2, f0_ct / 1200.0)


def _expw(N, w, inv=False):
    if N == 0:
        return torch.Tensor([])
    x = torch.linspace(0, 1, N) if not inv else torch.linspace(1, 0, N)
    y = torch.exp(w * x) - 1
    y /= torch.max(y) + 1e-9

    return y


def _adsr(L, attack=0.05, decay=0.05, sustain_amplitude=0.5, release=0.1, w_a=1, w_d=1, w_r=1):
    L_a = int(L * attack)
    L_d = int(L * decay)
    L_r = int(L * release)
    L_s = L - L_a - L_d - L_r

    a = _expw(L_a, w_a)
    d = _expw(L_d, w_d, True) * (1 - sustain_amplitude) + sustain_amplitude
    s = torch.ones(L_s) * sustain_amplitude
    r = _expw(L_r, w_r, True) * sustain_amplitude

    return torch.concatenate([a, d, s, r])


def _create_envelope(
    L: int,
    shape: Literal["const", "plucked", "fade", "tremolo", "random"] = "const",
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate an amplitude envelope of the given shape.

    Parameters
    ----------
    L : int
        Number of frames.
    shape : Literal["const", "plucked", "fade", "tremolo", "random"]
        Shape of the envelope.
    rng : torch.Generator or None
        Random number generator for reproducibility, used for the "random" shape (default: None).
    """
    if shape == "const":
        return torch.ones(L)

    if shape == "plucked":
        return _adsr(L, attack=0.003, decay=0.3, sustain_amplitude=0, release=0, w_a=2, w_d=5)

    if shape == "fade":
        return _adsr(L, attack=0.05, decay=0.1, sustain_amplitude=0.5, release=0.7, w_a=1, w_r=3)

    if shape == "tremolo":
        return torch.sin(2 * torch.pi * 5 * torch.arange(L) / L) + 1

    if shape == "random":
        if L > 8:
            return smooth(torch.rand(L, generator=rng), L // 8)

        return torch.rand(L, generator=rng)

    raise ValueError(f"Unknown envelope shape '{shape}'.")


@gin.register
class SyntheticSinusoidsDataset(torch.utils.data.Dataset):
    """Generate sinusoid signals for synthetic training data."""

    def __init__(
        self,
        N: int,
        L: int,
        M: int,
        H: int,
        F: int,
        fs: float,
        amp_min: float = -30,
        amp_max: float = 0,
        noise_min: float = -96,
        noise_max: float = -10,
        f0_min: float = 50,
        f0_max: float = 2000,
        f0_distr: Literal["log", "lin"] = "log",
        randomize_phase: bool = False,
        interp_mode: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "end_linear",
        random_seed: int | None = None,
    ):
        """Generate sinusoid signals for synthetic training data.

        Parameters
        ----------
        N : int
            Number of examples in the dataset.
        L : int
            Length of each example in samples.
        M : int
            Number of independent tones in each example.
        H : int
            Number of harmonics for each tone.
        F : int
            Number of frames in which the fundamental frequency (F0) of each example changes.
        fs : float
            Sampling rate in Hz.
        amp_min : float
            Minimum maximum amplitude for each tone in dB.
        amp_max : float
            Maximum maximum amplitude for each tone in dB.
        noise_min : float
            Minimum amplitude for the additive noise component in dB.
        noise_max : float
            Maximum amplitude for the additive noise component in dB.
        f0_min : float
            Minimum F0 in Hz.
        f0_max : float
            Maximum F0 in Hz.
        f0_distr : Literal["log", "lin"]
            Distribution from which F0s are drawn; options: "log", "lin".
        randomize_phase : bool
            Whether or not the initial phase of each example should be randomized.
        interp_mode : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
            Interpolation mode for the frame-wise values of F0 and overall amplitude.
        random_seed : None or int
            Optional random seed to create a reproducible dataset.
        """

        rng = torch.Generator()
        if random_seed is not None:
            rng.manual_seed(random_seed)
        else:
            rng.manual_seed(rng.seed())

        self.N = N
        self.L = L
        self.fs = fs
        self.interp_mode = interp_mode

        self.ampl_noise = torch.pow(10, (torch.rand((self.N), generator=rng) * (noise_max - noise_min) + noise_min) / 20)
        self.ampl_glob = torch.pow(10, (torch.rand((self.N), generator=rng) * (amp_max - amp_min) + amp_min) / 20)

        if randomize_phase:
            self.phase_init = torch.rand((self.N, M, H), generator=rng) * 2 * torch.pi
        else:
            self.phase_init = torch.zeros((self.N, M, H))

        # these will be filled individually
        self.f0_traj = torch.zeros((self.N, M, F))
        self.timbre = torch.zeros((self.N, M, H, 2))
        self.ampl_traj = torch.ones((self.N, M, F))

        avail_timbres = ["sawtooth", "triangle", "flat", "square", "clarinet-like"]
        avail_f0_trajs = ["random_walk", "log_sweep", "lin_sweep", "vibrato", "scale"]
        avail_envelopes = ["random", "tremolo", "plucked", "fade", "const"]

        for i in range(self.N):
            for j in range(M):
                # create timbre
                if H > 1:
                    self.timbre[i, j] = timbre2harmonics(avail_timbres[torch.randint(len(avail_timbres), (1,), generator=rng)], H)
                else:
                    self.timbre[i, j] = torch.ones((1, 2))

                # create F0 trajectory
                if f0_distr == "lin":
                    f_base = torch.rand(1, generator=rng) * (f0_max - f0_min) + f0_min
                elif f0_distr == "log":
                    f_base = f0_min * torch.pow(2, torch.rand(1, generator=rng) * math.log2(f0_max / f0_min))

                self.f0_traj[i, j] = _create_f0_trajectories(
                    1,
                    F,
                    f_base,
                    method=avail_f0_trajs[torch.randint(len(avail_f0_trajs), (1,), generator=rng)],
                    rand_range=torch.rand(1, generator=rng) * 256 + 32,
                    rand_smoothing=torch.randint(100, (1,), generator=rng) + 16,
                    sweep_max_hz=self.fs / 4,
                    vib_rate=torch.rand(1, generator=rng) * 30 + 3,
                    vib_phase=torch.rand(1, generator=rng),
                    vib_depth=torch.rand(1, generator=rng) * 45 + 5,
                    scale_direction="random",
                    scale_step_ct=torch.rand(1, generator=rng) * 200 + 50,
                    scale_step_length=torch.randint(50, (1,), generator=rng) + 10,
                    rng=rng,
                )[0]

                # create envelope
                if F > 1:
                    env = avail_envelopes[torch.randint(len(avail_envelopes), (1,), generator=rng)]
                    self.ampl_traj[i, j] = _create_envelope(F, env, rng)
                    if env in ["random", "tremolo"]:
                        minimum = torch.rand(1, generator=rng) * 0.2 + 0.001
                        scale = torch.rand(1, generator=rng) * (1 - minimum)
                        self.ampl_traj[i, j] = self.ampl_traj[i, j] * scale + minimum
                else:
                    self.ampl_traj[i, j] = 1

    def _generate_signal(self, f0, harmonics, a_traj, a_noise, a_glob, iph):
        f = torch.einsum("mf,mh->mhf", f0, harmonics[..., 0])
        a_harm = torch.repeat_interleave(harmonics[..., [1]], f.shape[-1], dim=-1)
        a_harm[f > self.fs / 2] = 0  # silence harmonics that are aliasing

        B = self.L // f.shape[-1]
        assert self.L / f.shape[-1] == B, "Number of samples must be divisible by number of frames."

        f_inst = interpolate_samples(f, B, mode=self.interp_mode)  # shape: (M, H, L)
        # harmonic amplitudes are frame-wise constant to make sure freqs above Nyquist are removed
        a_h_inst = interpolate_samples(a_harm, B, mode="const")  # shape: (M, H, L)
        # overall amplitude can be interpolated nicely
        a_inst = interpolate_samples(a_traj[:, None, :], B, mode=self.interp_mode)  # shape: (M, 1, L)

        phi = torch.cumsum(2 * torch.pi * f_inst / self.fs, dim=-1) + iph[:, :, None]
        x = a_inst * a_h_inst * torch.sin(phi)
        n = a_noise * 2 * (torch.rand(self.L) - 0.5)

        y = x.sum(dim=[0, 1]) + n  # shape: (L)
        y_scale = 1 / torch.max(torch.abs(y)) * a_glob
        y *= y_scale

        # calculate true resulting amplitude of each harmonic
        a_scaled = torch.abs(a_harm * a_traj[:, None, :] * y_scale)

        return y[None, ...], a_scaled, f

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x, a_scaled, f_harm = self._generate_signal(
            self.f0_traj[idx],
            self.timbre[idx],
            self.ampl_traj[idx],
            self.ampl_noise[idx],
            self.ampl_glob[idx],
            self.phase_init[idx],
        )
        return x, f_harm, a_scaled, self.phase_init[idx]
