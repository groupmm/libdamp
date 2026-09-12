"""Dataset for audio plus previously extracted F0 track."""

import glob
import os
import re

import gin
import numpy as np
import soundfile as sf
import torch


@gin.register
class SignalF0Dataset(torch.utils.data.Dataset):
    """Dataset for audio plus previously extracted F0 track."""

    def __init__(
        self, path: str, selection: str = r".*", num_excerpts: int = 1000, frame_len: int = 256, num_frames: int = 256, random_seed: int | None = None
    ) -> None:
        """Dataset for audio plus previously extracted F0 track.

        Parameters
        ----------
        path : str
            Path to the dataset directory containing wav files and extracted features.
        selection : str
            Regular expression to select which files from the dataset directory should be considered.
        num_excerpts : int
            Number of excerpts per file to pre-select (default: 1000).
        frame_len : int
            Frame length (and hop size) in samples for the annotations (default: 256).
        num_frames : int
            Number of frames per excerpt for the annotations (default: 256).
        random_seed : int or None
            Optional random seed for reproducible snippet selection (default: None).
        """

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        rng = torch.Generator()
        if random_seed is not None:
            rng.manual_seed(random_seed)
        else:
            rng.manual_seed(rng.seed())

        self.fs = None
        self.num_excerpts = num_excerpts
        self.frame_len = frame_len
        self.num_frames = num_frames

        wav_files = glob.glob(os.path.join(path, "*.wav"))

        # apply selection regex
        # (not using glob directly, because it only supports shell-style wildcards instead of complete regex)
        pattern = re.compile(selection)
        wav_files = [f for f in wav_files if pattern.match(os.path.basename(f))]

        if len(wav_files) == 0:
            raise FileNotFoundError(f"No wav files matching selection '{selection}' found in {path}")

        self.num_files = len(wav_files)

        self.signals = []
        self.f0s = []
        self.snippet_starts = torch.zeros((self.num_files, self.num_excerpts), dtype=int)
        for i, wav_file in enumerate(wav_files):
            folder = os.path.dirname(wav_file)
            basename, _ = os.path.splitext(os.path.basename(wav_file))
            x, fs = sf.read(wav_file)
            if self.fs is None:
                self.fs = fs
            else:
                assert self.fs == fs, "Sampling rate mismatch."

            f0 = np.load(os.path.join(folder, basename + "_f0.npy"))

            self.signals.append(torch.Tensor(x))
            self.f0s.append(torch.Tensor(f0))

            gain = np.sqrt(np.convolve(x**2, np.ones(self.frame_len)/self.frame_len, mode="same"))

            # pre-define snippets to use
            L_samples = self.num_frames * self.frame_len
            found = 0
            while found < self.num_excerpts:
                start = torch.randint(x.shape[0] - L_samples, (1,), generator=rng)
                if np.mean(gain[start : start + L_samples] > 0.01) > 0.25:
                    self.snippet_starts[i, found] = start
                    found += 1

    def __len__(self):
        return self.num_files * self.num_excerpts

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            (audio_signal, f0_subsample), where
            `audio_signal` has shape (L * H,) and the subsampled F0 track has shape (L,).
            Subsampled f0 is returned according to the hop size H defined in the dataset initialization.
        """
        n = idx // self.num_excerpts
        m = idx - n * self.num_excerpts

        L_samples = self.num_frames * self.frame_len
        start = self.snippet_starts[n, m]

        x = self.signals[n][start : start + L_samples]
        f0 = self.f0s[n][start : start + L_samples]

        # subsample f0 to then learn a frame-wise representation
        f0_s = f0[:: self.frame_len]

        assert len(f0_s) == self.num_frames

        return x, f0_s
