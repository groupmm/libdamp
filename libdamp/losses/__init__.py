"""Collection of loss functions for libdamp.

This module is part of the libdamp package.
"""

from .mss import MSSLoss
from .rms import RMSLoss

__all__ = ["RMSLoss", "MSSLoss"]
