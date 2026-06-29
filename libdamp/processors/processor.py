"""Abstract base class for audio processors.

This module is part of the libdamp package.
"""

from abc import ABC, abstractmethod

import torch


class Processor(ABC, torch.nn.Module):
    """Abstract base class for stateful audio processors in libdamp.

    Child classes must implement three core methods:
    - process(): process a given input signal
    - update(): set parameters of the processor
    - clear(): reset internal states to initial state
    """

    @abstractmethod
    def process(self, *args, **kwargs) -> torch.Tensor:
        """Process an audio signal.

        Receives an audio signal `x` as input and should return a processed audio signal `y`, where the appropriate
        processing depends on the set parameters in `update()`.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update processor parameters.

        Change the parameters that influence how the signal is processed in `process()`.
        Actual arguments depend on the implemented functionality and should be documented in derived classes.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """Reset processor internal state.

        If the processor has state (i.e., a call to `process()` depends on previous calls),
        this method should reset any state variables so that the processor behaves as if
        `process()` was never called on that instance before.
        """
        raise NotImplementedError()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Process audio (implements torch.nn.Module forward pass).

        Calls `process()` with the provided arguments.

        Returns
        -------
        torch.Tensor
            Processed audio signal.
        """
        return self.process(*args, **kwargs)
