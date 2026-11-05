"""Collection of datasets for libdamp.

This module is part of the libdamp package.
"""

from .combined import CombinedDataset, ZipDataset
from .excerpts import ExcerptsDataset, SignalF0ExcerptsDataset
from .lm_ssd import LMSSDDataset
from .medleydb import MDBSynthDataset
from .nsynth import NSynthDataset
from .pulseit import PulseItDataset
from .synthetic import SyntheticSinusoidsDataset

__all__ = [
    "CombinedDataset",
    "ZipDataset",
    "ExcerptsDataset",
    "SignalF0ExcerptsDataset",
    "LMSSDDataset",
    "MDBSynthDataset",
    "NSynthDataset",
    "PulseItDataset",
    "SyntheticSinusoidsDataset",
]
