"""Unit tests for `libdamp.helpers.modules`."""

import torch

from libdamp.helpers.modules import FreqToBins, LogitsToFreq, SelectItem


class TestSelectItem:
    def test_selects_requested_index(self):
        mod = SelectItem(1)
        inputs = ("a", "b", "c")
        assert mod(inputs) == "b"

    def test_works_as_part_of_a_sequential_pipeline(self):
        class _ReturnsTuple(torch.nn.Module):
            def forward(self, x):
                return x, x * 2

        seq = torch.nn.Sequential(_ReturnsTuple(), SelectItem(1))
        x = torch.tensor(3.0)
        assert seq(x) == 6.0


class TestFreqToBins:
    def test_output_is_a_normalized_distribution(self):
        mod = FreqToBins(num_bins=24, f_min=100.0, f_max=400.0)
        f = torch.tensor([150.0, 200.0])
        y = mod(f)
        assert y.shape == (2, 24)
        assert torch.allclose(y.sum(dim=-1), torch.ones(2), atol=1e-4)

    def test_peak_bin_is_near_target_frequency(self):
        mod = FreqToBins(num_bins=120, f_min=100.0, f_max=200.0, target_smoothing=5.0)
        f = torch.tensor([150.0])
        y = mod(f)
        # bin index for 150 Hz should be roughly in the middle of [0, 120)
        peak_bin = torch.argmax(y[0]).item()
        assert 50 <= peak_bin <= 70

    def test_snap_behavior_for_out_of_bounds_frequency(self):
        mod = FreqToBins(num_bins=10, f_min=100.0, f_max=200.0, out_of_bounds="snap")
        f = torch.tensor([50.0, 500.0])
        y = mod(f)
        assert torch.allclose(y[0], torch.eye(10)[0])
        assert torch.allclose(y[1], torch.eye(10)[-1])


class TestLogitsToFreq:
    def test_output_shape_collapses_bin_dimension(self):
        # output shape is x.shape[:-1] + (x.shape[-1] // bins_per_freq,)
        mod = LogitsToFreq(bins_per_freq=12, f_min=100.0, f_max=200.0)
        x = torch.randn(2, 3, 12)
        out = mod(x)
        assert out.shape == (2, 3, 1)

    def test_one_hot_logits_select_corresponding_bin_frequency(self):
        mod = LogitsToFreq(bins_per_freq=12, f_min=100.0, f_max=200.0)
        x = torch.full((1, 12), -1e9)
        x[0, 5] = 1e9
        out = mod(x)
        assert torch.allclose(out[0], mod.bin_freqs[0, 5], atol=1e-2)

    def test_output_within_frequency_range(self):
        mod = LogitsToFreq(bins_per_freq=12, f_min=100.0, f_max=200.0)
        x = torch.randn(5, 12)
        out = mod(x)
        assert torch.all(out >= 100.0 - 1e-4)
        assert torch.all(out <= 200.0 + 1e-4)
