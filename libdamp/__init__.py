"""libdamp

Framework for experiments with Differentiable Digital Signal Processing (DDSP) in PyTorch
"""

from .version import version as __version__

# import modules as sub-namespaces (e.g. `libdamp.generators.SinusoidalOsc`)
from . import generators
from . import processors

# In addition to individual sub-packages (`helpers`, `losses`, etc.), we expose a flat top-level API similar to PyTorch,
# to make access to commonly used functionalities more convenient.

from .augment import Augmentation, DropoutArtifactAugmentation, NoiseArtifactAugmentation, NoiseAugmentation
from .datasets import (
    CombinedDataset,
    ExcerptsDataset,
    MDBSynthDataset,
    NSynthDataset,
    PulseItDataset,
    SignalF0ExcerptsDataset,
    SyntheticSinusoidsDataset,
    ZipDataset,
)
from .experiment import Experiment
from .helpers import (
    OCTAVE_BANDS,
    THIRD_OCTAVE_BANDS,
    Convolution,
    ConvStack,
    FreqToBins,
    LogitsToFreq,
    SelectItem,
    apply_along_dim,
    combined_freqz,
    cubic_hermite_splines,
    design_butter_bandpass,
    design_butter_filter,
    design_fir_filter,
    design_resonant_filter,
    ensure_tensor,
    exp_sigmoid,
    freqz,
    get_window,
    hilbert,
    hz2midi,
    iir_freq_sampling,
    incremental_mod,
    interpolate_linear,
    interpolate_pchip,
    interpolate_samples,
    istft,
    midi2hz,
    poly,
    smooth,
    stft,
    tensor_linspace,
    timbre2harmonics,
)
from .losses import MSSLoss, RMSLoss
from .models import BasicBlock, BiLSTM, ResNet, resnet18

__all__ = [
    "Augmentation",
    "NoiseAugmentation",
    "DropoutArtifactAugmentation",
    "NoiseArtifactAugmentation",
    "Experiment",
    "CombinedDataset",
    "ZipDataset",
    "ExcerptsDataset",
    "SignalF0ExcerptsDataset",
    "MDBSynthDataset",
    "NSynthDataset",
    "PulseItDataset",
    "SyntheticSinusoidsDataset",
    "Convolution",
    "THIRD_OCTAVE_BANDS",
    "OCTAVE_BANDS",
    "freqz",
    "combined_freqz",
    "design_resonant_filter",
    "design_butter_bandpass",
    "design_butter_filter",
    "design_fir_filter",
    "iir_freq_sampling",
    "hz2midi",
    "midi2hz",
    "timbre2harmonics",
    "incremental_mod",
    "SelectItem",
    "ConvStack",
    "FreqToBins",
    "LogitsToFreq",
    "exp_sigmoid",
    "tensor_linspace",
    "apply_along_dim",
    "ensure_tensor",
    "interpolate_samples",
    "smooth",
    "cubic_hermite_splines",
    "interpolate_pchip",
    "interpolate_linear",
    "poly",
    "get_window",
    "stft",
    "istft",
    "hilbert",
    "RMSLoss",
    "MSSLoss",
    "BiLSTM",
    "BasicBlock",
    "ResNet",
    "resnet18",
]
