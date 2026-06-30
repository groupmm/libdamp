"""Unit tests for `libdamp.generators.band_filtered_noise`."""

import pytest
import torch

from libdamp.generators.band_filtered_noise import BandFilteredNoise


class TestBandFilteredNoise:
    def test_requires_update_before_generate(self):
        gen = BandFilteredNoise(frame_len=64, num_bands=2, order=2, fs=16000.0)
        with pytest.raises(AssertionError, match="Not initialized"):
            gen.generate()

    def test_output_shape_with_sum_up(self):
        gen = BandFilteredNoise(frame_len=64, num_bands=2, order=2, fs=16000.0)
        gen.update(
            fc=torch.full((1, 4, 2), 1000.0),
            bw=torch.full((1, 4, 2), 200.0),
            ba=torch.ones(1, 4, 2),
        )
        out = gen.generate(sum_up=True)
        assert out.shape == (1, 4 * 64)

    def test_output_shape_without_sum_up(self):
        gen = BandFilteredNoise(frame_len=64, num_bands=2, order=2, fs=16000.0)
        gen.update(
            fc=torch.full((1, 4, 2), 1000.0),
            bw=torch.full((1, 4, 2), 200.0),
            ba=torch.ones(1, 4, 2),
        )
        out = gen.generate(sum_up=False)
        assert out.shape == (1, 2, 4 * 64)

    def test_wrong_number_of_bands_raises(self):
        gen = BandFilteredNoise(frame_len=64, num_bands=2, order=2, fs=16000.0)
        with pytest.raises(AssertionError, match="Wrong number of bands"):
            gen.update(
                fc=torch.full((1, 4, 3), 1000.0),
                bw=torch.full((1, 4, 3), 200.0),
                ba=torch.ones(1, 4, 3),
            )

    def test_clear_allows_reinitialization(self):
        gen = BandFilteredNoise(frame_len=64, num_bands=2, order=2, fs=16000.0)
        gen.update(
            fc=torch.full((1, 4, 2), 1000.0),
            bw=torch.full((1, 4, 2), 200.0),
            ba=torch.ones(1, 4, 2),
        )
        gen.generate()
        gen.clear()
        with pytest.raises(AssertionError, match="Not initialized"):
            gen.generate()
