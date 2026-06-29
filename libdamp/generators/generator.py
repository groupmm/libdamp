"""Base class for audio signal generators.

This module is part of the libdamp package.
"""

from abc import ABC, abstractmethod

import torch


class Generator(ABC, torch.nn.Module):
    """Base class for stateful audio generators in libdamp.

    Child classes must implement three core methods:
    - generate(): generate an audio signal from the current parameters
    - update(): set parameters of the generator
    - clear(): reset internal states to initial state
    """

    @abstractmethod
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Generate audio from the current parameters.

        Returns
        -------
        torch.Tensor
            Generated audio signal.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update generator parameters.

        Change the parameters that influence how the signal is generated in `generate()`.
        Actual arguments depend on the implemented functionality and should be documented in derived classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """Reset generator internal state.

        If the generator has state (i.e., a call to `generate()` depends on previous calls),
        this method should reset any state variables so that the generator behaves as if
        it was just initialized.
        """
        raise NotImplementedError()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Generate audio (implements torch.nn.Module forward pass).

        Calls `generate()` with the provided arguments.

        Returns
        -------
        torch.Tensor
            Generated audio signal.
        """
        return self.generate(*args, **kwargs)
