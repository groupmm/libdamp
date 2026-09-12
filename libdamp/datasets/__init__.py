"""Collection of datasets for libdamp.

This module is part of the libdamp package.
"""

from .combined import CombinedDataset, ZipDataset
from .excerpts import ExcerptsDataset, SignalF0ExcerptsDataset
from .medleydb import MDBSynthDataset
from .nsynth import NSynthDataset
from .pulseit import PulseItDataset
from .signal_f0 import SignalF0Dataset
from .synthetic import SyntheticSinusoidsDataset

__all__ = [
    "CombinedDataset",
    "ZipDataset",
    "ExcerptsDataset",
    "SignalF0ExcerptsDataset",
    "MDBSynthDataset",
    "NSynthDataset",
    "PulseItDataset",
    "SignalF0Dataset",
    "SyntheticSinusoidsDataset",
]
