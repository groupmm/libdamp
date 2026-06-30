"""Unit tests for `libdamp.processors.resonant_filter`."""

import pytest
import torch

from libdamp.processors.resonant_filter import ResonantFilter


class TestResonantFilter:
    def test_requires_update_before_process(self):
        rf = ResonantFilter(N=64, fs=16000.0)
        with pytest.raises(AssertionError):
            rf.process(torch.randn(1, 256))

    def test_does_not_add_channel_dim_when_input_has_none(self):
        rf = ResonantFilter(N=64, fs=16000.0)
        rf.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        out = rf.process(torch.randn(1, 256))
        assert out.shape == (1, 256)

    def test_does_not_remove_channel_dim_when_input_has_one(self):
        rf = ResonantFilter(N=64, fs=16000.0)
        rf.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        out = rf.process(torch.randn(1, 1, 256))
        assert out.shape == (1, 1, 256)

    def test_multi_channel_output_matches_per_channel_2d_processing(self):
        torch.manual_seed(0)
        x = torch.randn(1, 256)

        rf_2d = ResonantFilter(N=64, fs=16000.0)
        rf_2d.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        y_2d = rf_2d.process(x)

        rf_3d = ResonantFilter(N=64, fs=16000.0)
        rf_3d.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        x_3d = torch.stack([x[0], x[0]])[None]
        y_3d = rf_3d.process(x_3d)

        assert y_3d.shape == (1, 2, 256)
        for c in range(2):
            assert torch.allclose(y_3d[0, c], y_2d[0])

    def test_more_resonances_changes_the_output(self):
        torch.manual_seed(0)
        x = torch.randn(1, 256)

        rf_1 = ResonantFilter(N=64, fs=16000.0)
        rf_1.update(f=torch.full((1, 1, 4), 1000.0), r=torch.full((1, 1, 4), 0.9))
        y_1 = rf_1.process(x)

        rf_2 = ResonantFilter(N=64, fs=16000.0)
        rf_2.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        y_2 = rf_2.process(x)

        assert not torch.allclose(y_1, y_2)

    def test_clear_resets_state(self):
        rf = ResonantFilter(N=64, fs=16000.0)
        rf.update(f=torch.full((1, 1, 4), 1000.0), r=torch.full((1, 1, 4), 0.9))
        rf.process(torch.randn(1, 256))
        rf.clear()
        assert rf.num_filters == 0
        with pytest.raises(AssertionError):
            rf.process(torch.randn(1, 256))
