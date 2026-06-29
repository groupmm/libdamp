"""Collection of processors for audio signals.

This module is part of the libdamp package.
"""

from .envelope import Envelope
from .formant_filters import FormantFilters
from .low_pass_filter import LowPassFilter
from .processor import Processor
from .tv_fir_filter import TVFIRFilter
from .tv_magnitudes_filter import TVMagnitudesFilter

__all__ = [
    "Processor",
    "Envelope",
    "LowPassFilter",
    "FormantFilters",
    "TVFIRFilter",
    "TVMagnitudesFilter",
]
