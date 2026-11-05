"""Dataset wrapper for the Larynx Microphone Singer-Songwriter Dataset (LM-SSD).

This module is part of the libdamp package.
"""

import glob
import os
import re

import gin
import soundfile as sf
import torch
import torchaudio as taudio


@gin.register
class LMSSDDataset(torch.utils.data.Dataset):
    """Dataset wrapper for the Larynx Microphone Singer-Songwriter Dataset (LM-SSD).

    LM-SSD (https://doi.org/10.5281/zenodo.20287765) provides paired larynx microphone (LM) recordings
    with clean close-up microphone (CM) recordings of the same takes.
    Audio files follow the naming convention::

        SSD[uid]_[song]_[signal]_[crosstalk]_[singer]_[take].wav

    e.g. `SSD001_AB_LM_C0_1M_T1.wav`, where `[signal]` is one of `LM`, `CM`, `GL`,
    `GR`, `GP`, `MixA`, `MixB`, `[crosstalk]` is `C0` (no guitar bleed on the CM
    signal) or `C1`, `[singer]` identifies the singer (e.g. `1M`), and `[take]` is
    `T1`-`T6` (`T1`-`T3` were recorded with microphone LM-A, `T4`-`T6` with LM-B).

    This dataset only uses the `LM`/`CM` pair of each take. It selects takes by applying a
    regular expression to the LM file's basename, which makes it possible to choose specific
    songs, singers, crosstalk conditions, or takes for a given train/val/test split, e.g.
    `selection=r".*_1M_T[1-3]"` for singer 1M, takes recorded with LM-A only.
    """

    def __init__(
        self,
        path: str,
        selection: str = r".*",
        target_fs: float = 16000.0,
        excerpt_len: int = 64000,
        num_excerpts: int = 1000,
        min_voice_activity: float = 0.05,
        silence_threshold_db: float = -40.0,
        random_seed: int | None = None,
    ) -> None:
        """Dataset wrapper for the Larynx Microphone Singer-Songwriter Dataset (LM-SSD).

        Parameters
        ----------
        path : str
            Path to the directory containing the LM-SSD audio files (flat directory of `.wav`
            files, i.e. the dataset's "audio" folder).
        selection : str
            Regular expression (matched with `re.search`) against the basename of each LM
            file to select which takes are part of this dataset, e.g. to pick specific songs,
            singers, crosstalk conditions, or takes.
        target_fs : float
            Target sampling rate in Hz (loaded audio is resampled if needed).
        excerpt_len : int
            Length of the randomly selected excerpts in samples, at `target_fs`.
        num_excerpts : int
            Number of excerpts to pre-select per take (default: 1000).
        min_voice_activity : float
            Minimum fraction of 20 ms frames within an excerpt that must exceed
            `silence_threshold_db` on the LM signal for the excerpt to be accepted during
            random pre-selection (default: 0.05).
        silence_threshold_db : float
            Level, in dB relative to the loudest 20 ms frame of the LM signal, below which
            a frame is considered silent (default: -40.0).
        random_seed : int or None
            Optional random seed for reproducible excerpt selection (default: None).
        """
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        rng = torch.Generator()
        if random_seed is not None:
            rng.manual_seed(random_seed)
        else:
            rng.manual_seed(rng.seed())

        self.excerpt_len = excerpt_len
        self.num_excerpts = num_excerpts
        self.target_fs = target_fs

        lm_files = sorted(glob.glob(os.path.join(path, "*_LM_*.wav")))

        # apply selection regex
        # (not using glob directly, because it only supports shell-style wildcards instead of complete regex)
        pattern = re.compile(selection)
        lm_files = [f for f in lm_files if pattern.search(os.path.basename(f))]

        if len(lm_files) == 0:
            raise FileNotFoundError(f"No LM-SSD takes matching selection '{selection}' found in {path}")

        self.num_takes = len(lm_files)

        self.lm_signals = []
        self.cm_signals = []
        self.snippet_starts = torch.zeros((self.num_takes, self.num_excerpts), dtype=int)
        for i, lm_file in enumerate(lm_files):
            cm_file = lm_file.replace("_LM_", "_CM_")
            if not os.path.isfile(cm_file):
                raise FileNotFoundError(f"No matching CM file found for '{lm_file}' (expected '{cm_file}')")

            x_lm = self._load_resampled(lm_file)
            x_cm = self._load_resampled(cm_file)

            L = min(x_lm.shape[-1], x_cm.shape[-1])
            assert self.excerpt_len <= L, f"Take '{os.path.basename(lm_file)}' is shorter than `excerpt_len`."

            self.lm_signals.append(x_lm[:L])
            self.cm_signals.append(x_cm[:L])

            # avoid pre-selecting excerpts that are completely silent on the LM signal (some
            # silence within an excerpt is fine, as long as there is some voice activity)
            vad_frame_len = max(1, int(self.target_fs * 0.02))  # 20 ms frames
            active = self._voice_activity_mask(x_lm[:L], vad_frame_len, silence_threshold_db)
            frames_per_excerpt = max(1, self.excerpt_len // vad_frame_len)
            min_active_frames = max(1, int(min_voice_activity * frames_per_excerpt))

            found = 0
            while found < self.num_excerpts:
                start = int(torch.randint(L - self.excerpt_len + 1, (1,), generator=rng))
                start_frame = start // vad_frame_len
                if active[start_frame : start_frame + frames_per_excerpt].sum() >= min_active_frames:
                    self.snippet_starts[i, found] = start
                    found += 1

    def _load_resampled(self, file: str) -> torch.Tensor:
        """Load a mono audio file and resample it to `self.target_fs` if necessary."""
        x, fs = sf.read(file, always_2d=True)
        x = torch.Tensor(x).mean(dim=-1)  # ensure mono
        if fs != self.target_fs:
            x = taudio.transforms.Resample(orig_freq=fs, new_freq=self.target_fs)(x)
        return x

    @staticmethod
    def _voice_activity_mask(x: torch.Tensor, frame_len: int, threshold_db: float) -> torch.Tensor:
        """Return a boolean mask of non-overlapping frames whose RMS exceeds `threshold_db` relative to the loudest frame."""
        num_frames = x.shape[-1] // frame_len
        envelope = x[: num_frames * frame_len].view(num_frames, frame_len).pow(2).mean(dim=-1).sqrt()
        threshold = envelope.max() * 10 ** (threshold_db / 20)
        return envelope > threshold

    def __len__(self):
        return self.num_takes * self.num_excerpts

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            `(lm_excerpt, cm_excerpt)`, time-aligned excerpts of the larynx microphone and
            close-up microphone signals, each of shape `(excerpt_len,)`.
        """
        n = idx // self.num_excerpts
        m = idx - n * self.num_excerpts

        start = self.snippet_starts[n, m]
        end = start + self.excerpt_len

        return self.lm_signals[n][start:end], self.cm_signals[n][start:end]
