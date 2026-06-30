"""Unit tests for `libdamp.processors.butterworth_low_pass_filter`."""

import pytest
import torch

from libdamp.processors.butterworth_low_pass_filter import ButterworthLowPassFilter


class TestButterworthLowPassFilter:
    def test_requires_update_before_process(self):
        lpf = ButterworthLowPassFilter(frame_len=64, fs=16000.0)
        with pytest.raises(AssertionError):
            lpf.process(torch.randn(1, 256))

    def test_does_not_add_channel_dim_when_input_has_none(self):
        lpf = ButterworthLowPassFilter(frame_len=64, fs=16000.0)
        lpf.update(fc=torch.full((1, 4), 2000.0))
        out = lpf.process(torch.randn(1, 256))
        assert out.shape == (1, 256)

    def test_does_not_remove_channel_dim_when_input_has_one(self):
        lpf = ButterworthLowPassFilter(frame_len=64, fs=16000.0)
        lpf.update(fc=torch.full((1, 4), 2000.0))
        out = lpf.process(torch.randn(1, 1, 256))
        assert out.shape == (1, 1, 256)

    def test_multi_channel_output_matches_per_channel_2d_processing(self):
        torch.manual_seed(0)
        x = torch.randn(1, 256)

        lpf_2d = ButterworthLowPassFilter(frame_len=64, fs=16000.0)
        lpf_2d.update(fc=torch.full((1, 4), 2000.0))
        y_2d = lpf_2d.process(x)

        lpf_3d = ButterworthLowPassFilter(frame_len=64, fs=16000.0)
        lpf_3d.update(fc=torch.full((1, 4), 2000.0))
        x_3d = torch.stack([x[0], x[0], x[0]])[None]  # (1, 3, 256), each channel identical
        y_3d = lpf_3d.process(x_3d)

        assert y_3d.shape == (1, 3, 256)
        for c in range(3):
            assert torch.allclose(y_3d[0, c], y_2d[0])

    def test_attenuates_high_frequency_content(self):
        fs = 16000.0
        t = torch.arange(4096) / fs
        x = torch.sin(2 * torch.pi * 6000 * t)[None]  # well above the cutoff

        lpf = ButterworthLowPassFilter(frame_len=64, fs=fs, order=4)
        lpf.update(fc=torch.full((1, 64), 500.0))
        y = lpf.process(x)

        assert y[..., 512:].abs().max() < x[..., 512:].abs().max() * 0.5

    def test_cascades_apply_filter_multiple_times(self):
        fs = 16000.0
        t = torch.arange(4096) / fs
        x = torch.sin(2 * torch.pi * 3000 * t)[None]

        lpf_1 = ButterworthLowPassFilter(frame_len=64, fs=fs, order=2, cascades=1)
        lpf_1.update(fc=torch.full((1, 64), 1000.0))
        y_1 = lpf_1.process(x)

        lpf_3 = ButterworthLowPassFilter(frame_len=64, fs=fs, order=2, cascades=3)
        lpf_3.update(fc=torch.full((1, 64), 1000.0))
        y_3 = lpf_3.process(x)

        # more cascades of the same low-pass filter should attenuate the (above-cutoff) tone more
        assert y_3[..., 512:].abs().max() < y_1[..., 512:].abs().max()
