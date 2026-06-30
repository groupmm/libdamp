"""Unit tests for `libdamp.processors.tv_magnitudes_filter`."""

import pytest
import torch

from libdamp.processors.tv_magnitudes_filter import TVMagnitudesFilter


class TestTVMagnitudesFilter:
    def test_requires_update_before_process(self):
        tvm = TVMagnitudesFilter(frame_len=64, filt_len=128, fs=16000.0)
        with pytest.raises(AssertionError):
            tvm.process(torch.randn(1, 256))

    def test_output_shape_matches_input(self):
        tvm = TVMagnitudesFilter(frame_len=64, filt_len=128, fs=16000.0)
        num_bands = tvm.center_freqs.shape[-1]
        tvm.update(magnitudes=torch.ones(1, 1, num_bands))
        out = tvm.process(torch.randn(1, 256))
        assert out.shape == (1, 256)

    def test_does_not_remove_channel_dim_when_input_has_one(self):
        tvm = TVMagnitudesFilter(frame_len=64, filt_len=128, fs=16000.0)
        num_bands = tvm.center_freqs.shape[-1]
        tvm.update(magnitudes=torch.ones(1, 1, num_bands))
        out = tvm.process(torch.randn(1, 1, 256))
        assert out.shape == (1, 1, 256)

    def test_mismatched_band_count_raises(self):
        tvm = TVMagnitudesFilter(frame_len=64, filt_len=128, fs=16000.0)
        num_bands = tvm.center_freqs.shape[-1]
        with pytest.raises(AssertionError, match="frequency bands must match"):
            tvm.update(magnitudes=torch.ones(1, 1, num_bands + 1))

    def test_attenuating_a_band_reduces_energy_near_that_frequency(self):
        fs = 16000.0
        N, M = 256, 128
        tvm = TVMagnitudesFilter(frame_len=N, filt_len=M, fs=fs)
        num_bands = tvm.center_freqs.shape[-1]

        target_band = num_bands // 2
        target_freq = tvm.center_freqs[target_band].item()

        mags_flat = torch.ones(1, 1, num_bands)
        mags_attenuated = torch.ones(1, 1, num_bands)
        mags_attenuated[..., target_band] = 0.01

        t = torch.arange(N * 4) / fs
        x = torch.sin(2 * torch.pi * target_freq * t)[None]

        tvm_flat = TVMagnitudesFilter(frame_len=N, filt_len=M, fs=fs)
        tvm_flat.update(magnitudes=mags_flat)
        y_flat = tvm_flat.process(x)

        tvm_atten = TVMagnitudesFilter(frame_len=N, filt_len=M, fs=fs)
        tvm_atten.update(magnitudes=mags_attenuated)
        y_atten = tvm_atten.process(x)

        assert y_atten.abs().mean() < y_flat.abs().mean() * 0.5
