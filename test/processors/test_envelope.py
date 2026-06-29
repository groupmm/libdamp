"""Unit tests for `libdamp.processors.envelope`."""

import pytest
import torch

from libdamp.processors.envelope import Envelope


class TestEnvelope:
    def test_requires_update_before_process(self):
        env = Envelope()
        with pytest.raises(AssertionError):
            env.process(torch.randn(1, 256))

    def test_does_not_add_channel_dim_when_input_has_none(self):
        # regression test: process() used to unconditionally add a channel dimension and never
        # remove it again, so a (B, L) input would come back as (B, 1, L).
        env = Envelope()
        env.update(g=torch.ones(1, 4))
        out = env.process(torch.randn(1, 256))
        assert out.shape == (1, 256)

    def test_does_not_remove_channel_dim_when_input_has_one(self):
        env = Envelope()
        env.update(g=torch.ones(1, 4))
        out = env.process(torch.randn(1, 3, 256))
        assert out.shape == (1, 3, 256)

    def test_multi_channel_gain_legitimately_expands_channel_less_input(self):
        env = Envelope()
        env.update(g=torch.ones(1, 2, 4))
        out = env.process(torch.randn(1, 256))
        assert out.shape == (1, 2, 256)

    def test_constant_gain_scales_signal(self):
        env = Envelope()
        env.update(g=torch.full((1, 4), 0.5))
        x = torch.ones(1, 256)
        out = env.process(x)
        assert torch.allclose(out, torch.full_like(x, 0.5))

    def test_zero_gain_silences_signal(self):
        env = Envelope()
        env.update(g=torch.zeros(1, 4))
        out = env.process(torch.randn(1, 256))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_unexpected_shape_raises(self):
        env = Envelope()
        env.update(g=torch.ones(1, 4))
        with pytest.raises(AssertionError, match="unexpected shape"):
            env.process(torch.randn(1, 257))  # 257 not divisible by 4 frames

    def test_clear_resets_state(self):
        env = Envelope()
        env.update(g=torch.ones(1, 4))
        env.process(torch.randn(1, 256))
        env.clear()
        assert env.g is None
        with pytest.raises(AssertionError):
            env.process(torch.randn(1, 256))
