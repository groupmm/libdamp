"""Unit tests for `libdamp.generators.sinusoidal_osc`."""

import pytest
import torch

from libdamp.generators.sinusoidal_osc import SinusoidalOsc


class TestSinusoidalOsc:
    def test_requires_update_before_generate(self):
        osc = SinusoidalOsc(N=64, fs=16000.0)
        with pytest.raises(AssertionError):
            osc.generate()

    def test_output_shape_with_sum_up(self):
        osc = SinusoidalOsc(N=64, fs=16000.0)
        osc.update(f=torch.full((2, 3, 4), 440.0), a=torch.ones(2, 3, 4))
        out = osc.generate(sum_up=True)
        assert out.shape == (2, 64 * 4)

    def test_output_shape_without_sum_up(self):
        osc = SinusoidalOsc(N=64, fs=16000.0)
        osc.update(f=torch.full((2, 3, 4), 440.0), a=torch.ones(2, 3, 4))
        out = osc.generate(sum_up=False)
        assert out.shape == (2, 3, 64 * 4)

    def test_matches_analytic_sine_for_constant_frequency(self):
        fs = 16000.0
        f0 = 440.0
        N = 256
        osc = SinusoidalOsc(N=N, fs=fs)
        osc.update(f=torch.full((1, 1, 4), f0), a=torch.ones(1, 1, 4))
        out = osc.generate()
        t = torch.arange(1, out.shape[-1] + 1)
        expected = torch.sin(2 * torch.pi * f0 * t / fs)
        assert torch.allclose(out[0], expected, atol=1e-3)

    def test_components_above_nyquist_are_silenced(self):
        fs = 16000.0
        osc = SinusoidalOsc(N=64, fs=fs)
        osc.update(f=torch.tensor([[[fs / 2 + 100, 100.0]]]), a=torch.ones(1, 1, 2))
        out = osc.generate(sum_up=False)
        # first frame's frequency is above Nyquist -> should be fully silenced
        assert torch.all(out[0, 0, :64] == 0)
        assert not torch.all(out[0, 0, 64:] == 0)

    def test_phase_is_continuous_across_successive_generate_calls(self):
        fs = 16000.0
        f0 = 300.0
        N = 64

        osc = SinusoidalOsc(N=N, fs=fs)
        osc.update(f=torch.full((1, 1, 2), f0), a=torch.ones(1, 1, 2))
        x1 = osc.generate()
        osc.update(f=torch.full((1, 1, 2), f0), a=torch.ones(1, 1, 2))
        x2 = osc.generate()
        x_split = torch.cat([x1, x2], dim=-1)

        osc_long = SinusoidalOsc(N=N, fs=fs)
        osc_long.update(f=torch.full((1, 1, 4), f0), a=torch.ones(1, 1, 4))
        x_long = osc_long.generate()

        assert torch.allclose(x_split, x_long, atol=1e-4)

    def test_clear_resets_state(self):
        osc = SinusoidalOsc(N=64, fs=16000.0)
        osc.update(f=torch.full((1, 1, 2), 440.0), a=torch.ones(1, 1, 2))
        osc.generate()
        osc.clear()
        assert osc.initialized is False
        with pytest.raises(AssertionError):
            osc.generate()

    def test_batch_size_change_resets_state_automatically(self):
        osc = SinusoidalOsc(N=64, fs=16000.0)
        osc.update(f=torch.full((1, 1, 2), 440.0), a=torch.ones(1, 1, 2))
        osc.generate()
        # batch size changes from 1 to 2: should not raise due to mismatched prev_* state
        osc.update(f=torch.full((2, 1, 2), 440.0), a=torch.ones(2, 1, 2))
        out = osc.generate()
        assert out.shape == (2, 128)

    def test_update_rejects_wrong_dims(self):
        osc = SinusoidalOsc(N=64, fs=16000.0)
        with pytest.raises(AssertionError):
            osc.update(f=torch.full((1, 2), 440.0), a=torch.ones(1, 1, 2))
