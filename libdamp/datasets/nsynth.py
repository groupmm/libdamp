"""Dataset wrapper for NSynth (https://magenta.tensorflow.org/datasets/nsynth)"""

import glob
import json
import os

import gin
import torch
import torchaudio as taudio

__all__ = ["NSynthDataset"]


@gin.register
class NSynthDataset(torch.utils.data.Dataset):
    """Dataset wrapper for NSynth (https://magenta.tensorflow.org/datasets/nsynth)"""

    def __init__(self, path: str, target_fs: float, return_keys: list = []) -> None:
        """Dataset wrapper for NSynth (https://magenta.tensorflow.org/datasets/nsynth)

        Parameters
        ----------
        path : str
            Full path to the dataset location (json/wav version).
        target_fs : float
            Target sampling rate in Hz (loaded audio will be resampled if this does not match).
        return_keys : list of str
            Specifies which items to return per sample, in addition to "audio" (which is always
            returned first). Each entry is the name of a key in the NSynth example metadata, e.g.
            "pitch", "velocity", or "instrument_family_str".
        """
        self.path = path
        self.files = glob.glob(os.path.join(self.path, "audio/*.wav"))

        assert len(self.files) > 0, "Audio files for NSynth not found"

        with open(os.path.join(self.path, "examples.json"), encoding="utf-8") as examples:
            self.metadata = json.load(examples)

        self.target_fs = target_fs
        self.resample = taudio.transforms.Resample(orig_freq=16000.0, new_freq=self.target_fs)

        self.return_keys = ["audio"] + return_keys

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> list:
        """Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        list
            List of values according to `return_keys`.
        """
        x, fs = taudio.load(self.files[idx])
        if fs != self.target_fs:
            x = self.resample(x)
        y = self.metadata[os.path.splitext(os.path.basename(self.files[idx]))[0]]
        return [x if key == "audio" else y[key] for key in self.return_keys]
