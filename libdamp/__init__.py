"""libdamp

Framework for experiments with Differentiable Digital Signal Processing (DDSP) in PyTorch
"""

from .version import version as __version__

# import modules as sub-namespaces (e.g. `libdamp.generators.SinusoidalOsc`)
from . import generators
from . import processors

# the flat top-level API (e.g. `libdamp.RMSLoss`) is defined in one place, decoupled
# from where the implementation actually lives. See libdamp/api.py for details
from .api import *
