"""Base class for an augmentation that is applied with a certain probability."""

import torch


class Augmentation(torch.nn.Module):
    """Base class for an augmentation that is applied with a certain probability."""

    def __init__(self, prob: float, rng: torch.Generator | None = None):
        """
        Parameters
        ----------
        prob : float
            Probability of applying the augmentation to a batch (1: always, 0: never).
        rng : torch.Generator or None
            Optional `torch.Generator` instance to fix a random seed for reproducible experiments. If `None`,
            a new generator is created (default: `None`).
        """
        super().__init__()

        self.rng = rng if rng is not None else torch.Generator()
        self.prob = prob
        assert self.prob >= 0 and self.prob <= 1, "Augmentation probability must be between 0 and 1."

    def apply(self, inp: torch.Tensor) -> torch.Tensor:
        """Apply the actual augmentation.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor to augment.

        Returns
        -------
        torch.Tensor
            Augmented tensor.
        """
        raise NotImplementedError()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation with the given probability, otherwise do nothing."""
        if self._rand() < self.prob:
            return self.apply(inp)

        # noop

        return inp

    def _rand(self) -> torch.Tensor:
        """Generate a random number. Helper for checking whether the augmentation should be applied or not.

        Returns
        -------
        torch.Tensor
            Random value in [0, 1).
        """
        return torch.rand((1,), generator=self.rng)
