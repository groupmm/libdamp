"""Unit tests for `libdamp.augment.augment.Augmentation`."""

import pytest
import torch

from libdamp.augment.augment import Augmentation


class _ConstantAugmentation(Augmentation):
    """Minimal concrete `Augmentation` that adds a constant offset, for testing the base class."""

    def apply(self, inp: torch.Tensor) -> torch.Tensor:
        return inp + 1.0


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
def test_prob_in_valid_range_is_accepted(prob):
    _ConstantAugmentation(prob=prob)


@pytest.mark.parametrize("prob", [-0.1, 1.1])
def test_prob_outside_valid_range_raises(prob):
    with pytest.raises(AssertionError):
        _ConstantAugmentation(prob=prob)


def test_forward_always_applies_when_prob_is_one():
    aug = _ConstantAugmentation(prob=1.0)
    inp = torch.zeros(4)
    assert torch.equal(aug(inp), inp + 1.0)


def test_forward_never_applies_when_prob_is_zero():
    aug = _ConstantAugmentation(prob=0.0)
    inp = torch.zeros(4)
    assert torch.equal(aug(inp), inp)


def test_forward_does_not_modify_input_in_place_when_noop():
    aug = _ConstantAugmentation(prob=0.0)
    inp = torch.randn(4)
    out = aug(inp)
    assert out is inp


def test_rand_returns_value_in_unit_interval():
    aug = _ConstantAugmentation(prob=1.0)
    for _ in range(100):
        r = aug._rand()
        assert r.shape == (1,)
        assert 0.0 <= r.item() < 1.0


def test_default_rng_is_not_shared_between_instances():
    """Regression test: each instance must get its own `torch.Generator` if none is given.

    Using a mutable default argument would make all instances share the same generator and
    therefore couple their random streams.
    """
    aug_a = _ConstantAugmentation(prob=1.0)
    aug_b = _ConstantAugmentation(prob=1.0)
    assert aug_a.rng is not aug_b.rng


def test_explicit_rng_is_used_and_makes_forward_reproducible():
    rng_a = torch.Generator().manual_seed(0)
    rng_b = torch.Generator().manual_seed(0)
    aug_a = _ConstantAugmentation(prob=0.5, rng=rng_a)
    aug_b = _ConstantAugmentation(prob=0.5, rng=rng_b)

    inp = torch.zeros(4)
    results_a = [aug_a(inp).clone() for _ in range(10)]
    results_b = [aug_b(inp).clone() for _ in range(10)]
    for a, b in zip(results_a, results_b, strict=True):
        assert torch.equal(a, b)
