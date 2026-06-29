"""Unit tests for `libdamp.processors.tv_fir_filter`."""

import pytest
import torch

from libdamp.processors.tv_fir_filter import TVFIRFilter


class TestTVFIRFilter:
    def test_requires_update_before_process(self):
        tv = TVFIRFilter(N=64, M=32)
        with pytest.raises(AssertionError):
            tv.process(torch.randn(1, 256))

    def test_does_not_add_channel_dim_when_input_has_none(self):
        tv = TVFIRFilter(N=64, M=32)
        tv.update(h=torch.randn(1, 1, 32))
        out = tv.process(torch.randn(1, 256))
        assert out.shape == (1, 256)

    def test_does_not_remove_channel_dim_when_input_has_one(self):
        tv = TVFIRFilter(N=64, M=32)
        tv.update(h=torch.randn(1, 1, 32))
        out = tv.process(torch.randn(1, 1, 256))
        assert out.shape == (1, 1, 256)

    def test_multi_channel_input_raises_a_clear_error(self):
        tv = TVFIRFilter(N=64, M=32)
        tv.update(h=torch.randn(1, 1, 32))
        with pytest.raises(AssertionError, match="single channel"):
            tv.process(torch.randn(1, 2, 256))

    def test_identity_filter_passes_signal_through(self):
        N, M = 64, 32
        tv = TVFIRFilter(N=N, M=M)
        h = torch.zeros(1, 1, M)
        h[..., 0] = 1.0  # identity impulse response
        tv.update(h=h)

        x = torch.randn(1, N * 4)
        y = tv.process(x)
        assert torch.allclose(y, x, atol=1e-4)

    def test_input_length_not_divisible_by_n_raises(self):
        tv = TVFIRFilter(N=64, M=32)
        tv.update(h=torch.randn(1, 1, 32))
        with pytest.raises(AssertionError, match="divisible"):
            tv.process(torch.randn(1, 100))

    def test_filters_are_repeated_across_extra_frames(self):
        N, M = 64, 32
        tv = TVFIRFilter(N=N, M=M)
        h = torch.zeros(1, 1, M)
        h[..., 0] = 1.0
        tv.update(h=h)  # single filter, should repeat across all frames

        x = torch.randn(1, N * 3)
        y = tv.process(x)
        assert y.shape == x.shape
