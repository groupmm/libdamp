"""Unit tests for `libdamp.generators.simple_filtered_noise`."""

import torch

from libdamp.generators.simple_filtered_noise import SimpleFilteredNoise


class TestSimpleFilteredNoise:
    def test_output_shape(self):
        gen = SimpleFilteredNoise(frame_len=128, filt_len=64, fs=16000.0, freq_bands=[100.0, 1000.0, 4000.0])
        mags = torch.ones(2, 3, 3)  # (B, frames, num_freq_bands)
        out = gen.generate(mags)
        assert out.shape == (2, 128 * 3)

    def test_update_and_clear_are_safe_no_ops(self):
        gen = SimpleFilteredNoise(frame_len=64, filt_len=32, fs=16000.0, freq_bands=[100.0, 1000.0])
        gen.update()
        gen.clear()
        out = gen.generate(torch.ones(1, 1, 2))
        assert out.shape == (1, 64)

    def test_zero_magnitude_produces_silence(self):
        torch.manual_seed(0)
        gen = SimpleFilteredNoise(frame_len=128, filt_len=64, fs=16000.0, freq_bands=[100.0, 1000.0])
        mags = torch.zeros(1, 1, 2)
        out = gen.generate(mags)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)
