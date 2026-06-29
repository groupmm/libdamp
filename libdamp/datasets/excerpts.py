"""Dataset wrapper for prepared excerpts."""

import os

import gin
import numpy as np
import torch

from libdamp.helpers.tensors import interpolate_pchip
from libdamp.helpers.transforms import get_window


@gin.register
class ExcerptsDataset(torch.utils.data.Dataset):
    """Generic dataset wrapper for prepared excerpts of data stored as `.npy` files.

    The dataset expects `path` to be a directory containing one `.npy` file per
    quantity of interest, named `[key]_[ds_split].npy`, where `[key]` is the name
    of the quantity (e.g. `signal`, `f0`, or any other custom key) and `[ds_split]`
    is the dataset split (e.g. `train`, `val`, `test`). Each file must contain an
    array whose first axis indexes the excerpts, so that all requested files have
    the same length (i.e. the same number of excerpts) for a given split. For
    example, for `return_keys=["signal", "f0"]` and `ds_split="train"`, `path` must
    contain the files::

        path/
            signal_train.npy   # shape (excerpts, ...)
            f0_train.npy       # shape (excerpts, ...)

    Every requested key is loaded verbatim from its corresponding file. This means
    an arbitrary number of precomputed quantities can be made available to the
    dataset, as long as they follow this naming convention and are stored as one
    array per split with excerpts along the first axis.
    """

    def __init__(
        self,
        path: str,
        ds_split: str = "train",
        return_keys: list = ["signal"],
    ):
        """Generic dataset wrapper for prepared excerpts of data stored as `.npy` files.

        Parameters
        ----------
        path : str
            Directory containing the prepared `.npy` files. It must contain one
            `[key]_[ds_split].npy` file for every key in `return_keys`.
        ds_split : str
            Dataset split to load, e.g. "train", "val", or "test" (default: "train").
        return_keys : list of str
            Specifies which items to return per sample. Each entry `[key]` is
            loaded verbatim from `path/[key]_[ds_split].npy`.
        """
        self.path = path
        self.ds_split = ds_split
        self.return_keys = list(return_keys)

        self.data = {key: torch.Tensor(np.load(self._npy_path(key))) for key in self.return_keys}

    def _npy_path(self, key: str) -> str:
        return os.path.join(self.path, key + "_" + self.ds_split + ".npy")

    def __len__(self):
        return len(next(iter(self.data.values())))

    def _get_sample(self, idx: int) -> dict:
        """Returns a dict with one entry per loaded key, indexed at `idx`."""
        return {key: data[idx] for key, data in self.data.items()}

    def __getitem__(self, idx: int) -> list:
        y = self._get_sample(idx)
        return [y[key] for key in self.return_keys]


@gin.register
class SignalF0ExcerptsDataset(ExcerptsDataset):
    """Dataset wrapper for prepared excerpts of signal/F0 data stored as `.npy` files.

    This is a specialization of :class:`~libdamp.datasets.excerpts.ExcerptsDataset`
    for the common case of audio signals paired with F0 trajectories. In addition
    to the files required for any plain `return_keys` (see `ExcerptsDataset`), it
    always requires `path` to contain a `signal_[ds_split].npy` file, and, if "f0",
    "f", or "a" is among `return_keys`, also a `f0_[ds_split].npy` file::

        path/
            signal_train.npy   # shape (excerpts, 1, samples)
            f0_train.npy       # shape (excerpts, frames)
            ...

    The two special return keys "f" and "a" are not loaded from disk. Instead,
    they are derived on the fly from the F0 trajectory (loaded from
    `f0_[ds_split].npy`) and the corresponding audio signal (loaded from
    `signal_[ds_split].npy`): the frequencies of the first `M` harmonics of F0
    ("f") and their amplitudes ("a") are determined via spectral peak picking,
    i.e. by computing the STFT of the signal and interpolating its magnitude at
    the harmonic frequencies.
    """

    def __init__(
        self,
        path: str,
        num_harm: int = 100,
        num_frames: int = 128,
        fs: float = 16000.0,
        ds_split: str = "train",
        return_keys: list = ["f", "a"],
    ):
        """Dataset wrapper for prepared excerpts of signal/F0 data stored as `.npy` files.

        Parameters
        ----------
        path : str
            Directory containing the prepared `.npy` files. It must contain
            `signal_<ds_split>.npy` (always required) and `f0_<ds_split>.npy`
            (required if "f0", "f", or "a" is among `return_keys`), as well as
            one `<key>_<ds_split>.npy` file for every other key in `return_keys`.
        num_harm : int
            Number of harmonics of F0 to consider when computing "f" and "a"
            (default: 100).
        num_frames : int
            Number of time frames per excerpt used for STFT-based computation of
            "f" and "a" (default: 128). The signal length must be divisible by `F`.
        fs : float
            Sampling rate of the audio signals in Hz, used for STFT-based
            computation of "f" and "a" (default: 16000).
        ds_split : str
            Dataset split to load, e.g. "train", "val", or "test" (default: "train").
        return_keys : list of str
            Specifies which items to return per sample, in addition to "signal"
            (which is always returned first). Each entry is either:
            - "f": harmonic frequencies of F0, shape (harmonics, frames), computed
              via peak picking (see class docstring).
            - "a": amplitudes of the harmonic frequencies in "f", shape
              (harmonics, frames), computed via peak picking (see class docstring).
            - any other string `<key>`: loaded verbatim from
              `path/<key>_<ds_split>.npy`, e.g. "f0" for the fundamental frequency
              trajectory, shape (frames,).
        """
        self.num_harm = num_harm
        self.num_frames = num_frames
        self.fs = fs

        output_keys = ["signal"] + list(return_keys)

        # "f" and "a" are derived from F0 + signal rather than loaded directly,
        # but still require the F0 trajectory to be loaded.
        load_keys = [key for key in output_keys if key not in ("f", "a")]
        if ("f" in output_keys or "a" in output_keys) and "f0" not in load_keys:
            load_keys.append("f0")

        super().__init__(path, ds_split=ds_split, return_keys=load_keys)

        self.output_keys = output_keys

    def __getitem__(self, idx: int) -> list:
        y = self._get_sample(idx)

        if "f" in self.output_keys or "a" in self.output_keys:
            x = y["signal"]

            # calculate f_ref and a_ref from F0 + spectrogram
            f = y["f0"][:, None] * torch.arange(1, self.num_harm + 1)

            N = 4096
            L = x.shape[-1]
            frame_len = L // self.num_frames
            assert frame_len == L / self.num_frames, "Signal length must be divisible by F."

            w = get_window("hann", N).type_as(x)
            X = torch.abs(torch.stft(x, n_fft=N, hop_length=frame_len, window=w, center=True, return_complex=True, normalized=False))
            X /= N  # normalize for both forward and backward transform
            X *= 3.14  # account for Hann window spread of the energy when peak picking
            X = torch.transpose(X[..., : self.num_frames], -1, -2).squeeze()

            f_fft = torch.fft.rfftfreq(N, 1 / self.fs)

            a = torch.zeros((self.num_frames, self.num_harm))
            for fr in range(self.num_frames):
                if torch.count_nonzero(f[fr]) == 0:
                    continue
                a[fr] = interpolate_pchip(f_fft, X[fr], f[fr])

            mask = f > self.fs / 2
            a[mask] = 0
            # f[mask] = 0

            y["f"] = f.transpose(-2, -1)
            y["a"] = torch.clip(a.transpose(-2, -1), 0, None)

        return [y[key] for key in self.output_keys]
