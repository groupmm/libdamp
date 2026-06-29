"""Collection of data augmentations for libdamp.

This module is part of the libdamp package.
"""

from .audio import DropoutArtifactAugmentation, NoiseArtifactAugmentation, NoiseAugmentation
from .augment import Augmentation

__all__ = [
    "Augmentation",
    "NoiseAugmentation",
    "DropoutArtifactAugmentation",
    "NoiseArtifactAugmentation",
]
