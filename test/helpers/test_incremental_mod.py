"""Unit tests for `libdamp.helpers.incremental_mod`."""

import torch

from libdamp.helpers.incremental_mod import incremental_mod, incremental_mod_python


class TestIncrementalModPython:
    def test_matches_docstring_example(self):
        mod = torch.Tensor([5, 3, 6])
        inc = torch.Tensor([2, 2, 2])
        out = incremental_mod_python(mod, inc)
        assert torch.equal(out, torch.Tensor([0, 2, 1]))

    def test_default_increment_is_ones(self):
        mod = torch.Tensor([3, 3, 3, 3])
        out = incremental_mod_python(mod, None)
        assert torch.equal(out, torch.Tensor([0, 1, 2, 0]))

    def test_result_never_reaches_modulus(self):
        # the running value must always stay in [0, mod), even when it lands exactly on the boundary.
        mod = torch.Tensor([5, 5])
        inc = torch.Tensor([5, 5])
        out = incremental_mod_python(mod, inc)
        assert torch.equal(out, torch.Tensor([0, 0]))

    def test_batched_input(self):
        mod = torch.Tensor([[5, 3, 6], [4, 4, 4]])
        inc = torch.Tensor([[2, 2, 2], [3, 3, 3]])
        out = incremental_mod_python(mod, inc)
        assert torch.equal(out, torch.Tensor([[0, 2, 1], [0, 3, 2]]))


class TestIncrementalMod:
    def test_cpu_dispatch_matches_python_implementation(self):
        mod = torch.Tensor([5, 3, 6])
        inc = torch.Tensor([2, 2, 2])
        assert torch.equal(incremental_mod(mod, inc), incremental_mod_python(mod, inc))
