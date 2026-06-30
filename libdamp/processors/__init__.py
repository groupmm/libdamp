"""Collection of processors for audio signals.

This module is part of the libdamp package.
"""

from .butterworth_low_pass_filter import ButterworthLowPassFilter
from .gain_envelope import GainEnvelope
from .processor import Processor
from .resonant_filter import ResonantFilter
from .tv_fir_filter import TVFIRFilter
from .tv_magnitudes_filter import TVMagnitudesFilter

__all__ = [
    "Processor",
    "GainEnvelope",
    "ButterworthLowPassFilter",
    "ResonantFilter",
    "TVFIRFilter",
    "TVMagnitudesFilter",
]
