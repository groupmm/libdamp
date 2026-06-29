"""Collection of data augmentations that can be applied directly to an audio signal."""

import math

import gin
import torch

from libdamp.augment.augment import Augmentation
from libdamp.helpers.tensors import ensure_tensor

__all__ = ["NoiseAugmentation", "DropoutArtifactAugmentation", "NoiseArtifactAugmentation"]


@gin.register
class NoiseAugmentation(Augmentation):
    """Add white noise to each audio signal in a batch."""

    def __init__(self, prob: float, noise_level_db: float = 0.0):
        """Add white noise to each audio signal in a batch.

        Parameters
        ----------
        prob : float
            Probability of applying the augmentation to a batch (1: always, 0: never).
        noise_level_db : float
            Level of the noise relative to the RMS level of the input signal
            (0: noise level equals signal level, -10: noise level is 10 dB below signal level).
        """
        super().__init__(prob)

        self.g = 10 ** (noise_level_db / 20)

    def apply(self, inp: torch.Tensor) -> torch.Tensor:
        """Add white noise to each audio signal in the batch.

        Parameters
        ----------
        inp : torch.Tensor
            Input audio tensor of shape (batch, num_samples).

        Returns
        -------
        torch.Tensor
            Audio tensor with added white noise, same shape as input.
        """
        inp = ensure_tensor(inp)
        a_inp = torch.sqrt(torch.mean(inp**2, dim=-1, keepdim=True))
        # uniform noise on [-1, 1] has RMS 1/sqrt(3), so scale by sqrt(3) to make the
        # actual noise RMS match a_inp * self.g (i.e. match `noise_level_db` exactly)
        n = (torch.rand_like(inp) * 2 - 1) * math.sqrt(3) * a_inp * self.g
        return inp + n


@gin.register
class DropoutArtifactAugmentation(Augmentation):
    """Add dropout artifacts (silent segments) to each audio signal in a batch."""

    def __init__(self, prob: float, min_len: int = 1, max_len: int = 20):
        """Add dropout artifacts (silent segments) to each audio signal in a batch.

        Parameters
        ----------
        prob : float
            Probability of applying the augmentation to a batch (1: always, 0: never).
        min_len : int
            Minimum length of dropout segments in samples.
        max_len : int
            Maximum length of dropout segments in samples.
        """
        super().__init__(prob)

        self.min_len = min_len
        self.max_len = max_len

    def apply(self, inp: torch.Tensor) -> torch.Tensor:
        """Add dropout artifacts (silent segments) to audio signals.

        Parameters
        ----------
        inp : torch.Tensor
            Input audio tensor of shape (batch, num_samples).

        Returns
        -------
        torch.Tensor
            Audio tensor with dropout artifacts added, same shape as input.

        Notes
        -----
        Dropout segment lengths are sampled uniformly in samples from
        [min_len, max_len], and start indices are sampled so that the segment
        ends before the last sample (index L - 1).
        """
        x = ensure_tensor(inp).clone()
        B, L = x.shape

        start = torch.randint(L - self.max_len - 1, (B,), device=x.device)
        end = start + torch.randint(self.min_len, self.max_len + 1, (B,), device=x.device)

        for b in range(B):
            x[b, start[b] : end[b]] = 0

        return x


@gin.register
class NoiseArtifactAugmentation(Augmentation):
    """Add noise artifacts to random segments of each audio signal in a batch."""

    def __init__(self, prob: float, min_len: int = 10, max_len: int = 1000, min_ampl_db: float = -60, max_ampl_db: float = 0):
        """Add noise artifacts to random segments of each audio signal in a batch.

        Parameters
        ----------
        prob : float
            Probability of applying the augmentation to a batch (1: always, 0: never).
        min_len : int
            Minimum length of added noise artifact segments in samples.
        max_len : int
            Maximum length of added noise artifact segments in samples.
        min_ampl_db : float
            Minimum amplitude in dBFS of the added noise.
        max_ampl_db : float
            Maximum amplitude in dBFS of the added noise (0 dB -> noise with maximum amplitude of 1).
        """
        super().__init__(prob)

        self.min_len = min_len
        self.max_len = max_len
        self.min_ampl_db = min_ampl_db
        self.max_ampl_db = max_ampl_db

    def apply(self, inp: torch.Tensor) -> torch.Tensor:
        """Add noise artifacts to random segments of audio signals.

        Parameters
        ----------
        inp : torch.Tensor
            Input audio tensor of shape (batch, num_samples).

        Returns
        -------
        torch.Tensor
            Audio tensor with noise artifacts added to random segments, same shape as input.

        Notes
        -----
        Noise artifact segment lengths are sampled uniformly in samples from
        [min_len, max_len], and start indices are sampled so that the segment
        ends before the last sample (index L - 1).
        """
        x = ensure_tensor(inp).clone()
        B, L = x.shape

        start = torch.randint(L - self.max_len - 1, (B,), device=x.device)
        end = start + torch.randint(self.min_len, self.max_len + 1, (B,), device=x.device)

        for b in range(B):
            a_db = torch.rand((1,)) * (self.max_ampl_db - self.min_ampl_db) + self.min_ampl_db
            x[b, start[b] : end[b]] += (torch.rand(end[b] - start[b]) * 2 - 1) * 10 ** (a_db / 20)

        return x
