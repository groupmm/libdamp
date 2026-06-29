"""Unit tests for `libdamp.generators.weighted_table_osc`."""

import pytest
import torch

from libdamp.generators.weighted_table_osc import WeightedTableOsc


def _sine_table(L=64, M=2):
    t = torch.arange(L)
    return torch.stack([torch.sin(2 * torch.pi * t / L) for _ in range(M)])


class TestWeightedTableOsc:
    def test_output_shape(self):
        table = _sine_table(L=64, M=3)
        osc = WeightedTableOsc(N=32, table=table, fs=16000.0, mode="wave")
        osc.update(f0=torch.full((1, 2), 100.0), weighting=torch.ones(1, 2, 3))
        out = osc.generate()
        assert out.shape == (1, 32 * 2)

    def test_zero_weights_produce_silence(self):
        table = _sine_table(L=64, M=2)
        osc = WeightedTableOsc(N=32, table=table, fs=16000.0, mode="wave")
        osc.update(f0=torch.full((1, 2), 100.0), weighting=torch.zeros(1, 2, 2))
        out = osc.generate()
        assert torch.allclose(out, torch.zeros_like(out))

    def test_only_selected_entry_weight_matches_table_osc_single_entry(self):
        table = _sine_table(L=64, M=2)
        osc = WeightedTableOsc(N=32, table=table, fs=16000.0, mode="wave", table_freq=16000.0 / 64)
        weighting = torch.zeros(1, 2, 2)
        weighting[:, :, 0] = 1.0  # only entry 0 active
        osc.update(f0=torch.full((1, 2), 100.0), weighting=weighting)
        out = osc.generate()
        assert out.shape == (1, 64)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_unknown_mode_raises(self):
        table = _sine_table()
        with pytest.raises(ValueError, match="Unknown TableOsc mode"):
            WeightedTableOsc(N=32, table=table, fs=16000.0, mode="bogus")

    def test_mismatched_f0_and_weighting_shapes_raise(self):
        table = _sine_table()
        osc = WeightedTableOsc(N=32, table=table, fs=16000.0)
        with pytest.raises(AssertionError):
            osc.update(f0=torch.zeros(1, 3), weighting=torch.zeros(1, 4, 2))
