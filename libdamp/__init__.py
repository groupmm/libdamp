"""libdamp

Framework for experiments with Differentiable Digital Signal Processing (DDSP) in PyTorch
"""

from .version import version as __version__

# import modules as sub-namespaces (e.g. `libdamp.generators.SinusoidalOsc`)
from . import generators
from . import processors

# import each function/class into global namespace
from .augment.augment import Augmentation
from .augment.audio import *

from .experiment import Experiment

from .datasets.combined import *
from .datasets.excerpts import *
from .datasets.medleydb import *
from .datasets.nsynth import *
from .datasets.pulseit import *
from .datasets.synthetic import *

from .helpers.convolution import *
from .helpers.filters import *
from .helpers.freq import *
from .helpers.incremental_mod import *
from .helpers.modules import *
from .helpers.scaling import *
from .helpers.tensors import *
from .helpers.transforms import *

from .losses.rms import *
from .losses.mss import *

from .models.lstm import *
from .models.resnet import *
