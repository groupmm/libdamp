"""Collection of audio signal generators.

This module is part of the libdamp package.
"""

from .band_filtered_noise import BandFilteredNoise
from .fm_synth import FMSynth
from .generator import Generator
from .harmonic_osc import HarmonicOsc
from .impulse_train import ImpulseTrain
from .simple_filtered_noise import SimpleFilteredNoise
from .sinusoidal_osc import SinusoidalOsc
from .table_osc import TableOsc
from .weighted_table_osc import WeightedTableOsc
from .white_noise import WhiteNoise

__all__ = [
    "Generator",
    "SinusoidalOsc",
    "HarmonicOsc",
    "BandFilteredNoise",
    "SimpleFilteredNoise",
    "WhiteNoise",
    "ImpulseTrain",
    "TableOsc",
    "WeightedTableOsc",
    "FMSynth",
]
