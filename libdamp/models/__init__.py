"""Collection of generic neural network models for libdamp.

This module is part of the libdamp package.
"""

from .lstm import BiLSTM
from .resnet import BasicBlock, ResNet, resnet18

__all__ = ["BiLSTM", "BasicBlock", "ResNet", "resnet18"]
