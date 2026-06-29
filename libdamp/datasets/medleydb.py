"""Dataset wrapper for MDB-[stem|melody]-synth (http://synthdatasets.weebly.com/)"""

import glob
import os

import gin
import numpy as np
import torch
import torchaudio as taudio

__all__ = ["MDBSynthDataset"]


@gin.register
class MDBSynthDataset(torch.utils.data.Dataset):
    """Dataset wrapper for MDB-[stem|melody]-synth (http://synthdatasets.weebly.com/)"""

    def __init__(self, audio_path: str, annot_path: str, target_fs: float, return_keys: list = []) -> None:
        """Dataset wrapper for MDB-[stem|melody]-synth (http://synthdatasets.weebly.com/)

        Parameters
        ----------
        audio_path : str
            Full path to the audio file folder.
        annot_path : str
            Full path to the annotation file folder.
        target_fs : float
            Target sampling rate in Hz (loaded audio will be resampled if this does not match).
        return_keys : list of str
            Specifies which items to return per sample, in addition to "audio" (which is always
            returned first). Each entry is one of "time" (annotation timestamps, shape (frames,))
            or "f0" (fundamental frequency trajectory, shape (frames,)).
        """
        self.audio_path = audio_path
        self.annot_path = annot_path
        self.files = glob.glob(os.path.join(self.audio_path, "*.wav"))

        assert len(self.files) > 0, "Audio files for MDB-*-synth not found"

        self.target_fs = target_fs
        self.resample = taudio.transforms.Resample(orig_freq=44100.0, new_freq=self.target_fs)

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
        file_id = os.path.splitext(os.path.basename(self.files[idx]))[0]
        x, fs = taudio.load(self.files[idx])
        if fs != self.target_fs:
            x = self.resample(x)

        f0 = np.loadtxt(os.path.join(self.annot_path, file_id + ".csv"), delimiter=",", skiprows=0)
        y = {
            "time": f0[:, 0],
            "f0": f0[:, 1],
        }
        return [x if key == "audio" else y[key] for key in self.return_keys]
