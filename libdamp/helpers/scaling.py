"""Collection of scaling functions to keep values in a prescribed range.

This module is part of the libdamp package.
"""

import gin
import torch


@gin.configurable
def exp_sigmoid(x, x_max: float = 2.0, x_min: float = 1e-8, exp: float = 1.0) -> torch.Tensor:
    """Scale values between a minimum and maximum value with an exponentiated sigmoid

    Parameters
    ----------
    x : torch.Tensor or array-like
        Input values to scale
    x_max : float
        Maximum output value (default: 2.0)
    x_min : float
        Minimum output value (default: 1e-8)
    exp : float
        Exponent for the sigmoid (default: 1.0)

    Returns
    -------
    torch.Tensor
        Scaled values in range (x_min, x_min + x_max).
    """
    return x_max * torch.sigmoid(x) ** exp + x_min
