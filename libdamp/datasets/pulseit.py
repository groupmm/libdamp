"""Dataset wrapper for previously extracted F0/FC/Gain tracks for the PULSE-IT method."""

import glob
import os
import re

import gin
import numpy as np
import soundfile as sf
import torch

__all__ = ["PulseItDataset"]


@gin.register
class PulseItDataset(torch.utils.data.Dataset):
    """Dataset for F0/FC/Gain tracks for the PULSE-IT method."""

    def __init__(self, path: str, selection: str = r".*", M: int = 1000, H: int = 256, L: int = 256, random_seed: int | None = None) -> None:
        """Dataset for F0/FC/Gain tracks for the PULSE-IT method.

        Parameters
        ----------
        path : str
            Path to the dataset directory containing wav files and extracted features.
        selection : str
            Regular expression to select which files from the dataset directory should be considered.
        M : int
            Number of snippets per file to pre-select (default: 1000).
        H : int
            Hop size and Frame size in samples for the annotations (default: 256).
        L : int
            Number of frames per snippet for the annotations (default: 256).
        random_seed : int or None
            Optional random seed for reproducible snippet selection (default: None).
        """

        rng = torch.Generator()
        if random_seed is not None:
            rng.manual_seed(random_seed)
        else:
            rng.manual_seed(rng.seed())

        self.fs = None
        self.M = M
        self.H = H
        self.L = L

        wav_files = glob.glob(os.path.join(path, "*.wav"))

        # apply selection regex
        # (not using glob directly, because it only supports shell-style wildcards instead of complete regex)
        pattern = re.compile(selection)
        wav_files = [f for f in wav_files if pattern.match(os.path.basename(f))]

        self.N = len(wav_files)

        self.signals = []
        self.f0s = []
        self.fcs = []
        self.gains = []
        self.snippet_starts = torch.zeros((self.N, self.M), dtype=int)
        for i, wav_file in enumerate(wav_files):
            folder = os.path.dirname(wav_file)
            basename, _ = os.path.splitext(os.path.basename(wav_file))
            x, fs = sf.read(wav_file)
            if self.fs is None:
                self.fs = fs
            else:
                assert self.fs == fs, "Sampling rate mismatch."

            f0 = np.load(os.path.join(folder, basename + "_f0.npy"))
            fc = np.load(os.path.join(folder, basename + "_fc.npy"))
            gain = np.load(os.path.join(folder, basename + "_gain.npy"))

            self.signals.append(torch.Tensor(x))
            self.f0s.append(torch.Tensor(f0))
            self.fcs.append(torch.Tensor(fc))
            self.gains.append(torch.Tensor(gain))

            # pre-define snippets to use
            L_samples = self.L * self.H
            found = 0
            while found < M:
                start = torch.randint(x.shape[0] - L_samples, (1,), generator=rng)
                if np.mean(fc[start : start + L_samples] > 100) > 0.25 and np.mean(gain[start : start + L_samples] > 0.01) > 0.25:
                    self.snippet_starts[i, found] = start
                    found += 1

    def __len__(self):
        return self.N * self.M

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            (audio_signal, f0_subsample, fc_subsample, gain_subsample), where
            `audio_signal` has shape (L * H,) and the subsampled tracks have shape (L,).
            Subsampled f0, fc, and gain are returned according to the hop size H defined in the dataset initialization.
        """
        n = idx // self.M
        m = idx - n * self.M

        L_samples = self.L * self.H
        start = self.snippet_starts[n, m]

        x = self.signals[n][start : start + L_samples]
        f0 = self.f0s[n][start : start + L_samples]
        fc = self.fcs[n][start : start + L_samples]
        g = self.gains[n][start : start + L_samples]

        # subsample f0, fc, and g to then learn a frame-wise representation
        f0_s = f0[:: self.H]
        fc_s = fc[:: self.H]
        g_s = g[:: self.H]

        assert len(f0_s) == self.L

        return x, f0_s, fc_s, g_s
