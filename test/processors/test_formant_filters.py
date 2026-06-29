"""Unit tests for `libdamp.processors.formant_filters`."""

import pytest
import torch

from libdamp.processors.formant_filters import FormantFilters


class TestFormantFilters:
    def test_requires_update_before_process(self):
        ff = FormantFilters(N=64, fs=16000.0)
        with pytest.raises(AssertionError):
            ff.process(torch.randn(1, 256))

    def test_does_not_add_channel_dim_when_input_has_none(self):
        ff = FormantFilters(N=64, fs=16000.0)
        ff.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        out = ff.process(torch.randn(1, 256))
        assert out.shape == (1, 256)

    def test_does_not_remove_channel_dim_when_input_has_one(self):
        ff = FormantFilters(N=64, fs=16000.0)
        ff.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        out = ff.process(torch.randn(1, 1, 256))
        assert out.shape == (1, 1, 256)

    def test_multi_channel_output_matches_per_channel_2d_processing(self):
        torch.manual_seed(0)
        x = torch.randn(1, 256)

        ff_2d = FormantFilters(N=64, fs=16000.0)
        ff_2d.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        y_2d = ff_2d.process(x)

        ff_3d = FormantFilters(N=64, fs=16000.0)
        ff_3d.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        x_3d = torch.stack([x[0], x[0]])[None]
        y_3d = ff_3d.process(x_3d)

        assert y_3d.shape == (1, 2, 256)
        for c in range(2):
            assert torch.allclose(y_3d[0, c], y_2d[0])

    def test_more_formants_changes_the_output(self):
        torch.manual_seed(0)
        x = torch.randn(1, 256)

        ff_1 = FormantFilters(N=64, fs=16000.0)
        ff_1.update(f=torch.full((1, 1, 4), 1000.0), r=torch.full((1, 1, 4), 0.9))
        y_1 = ff_1.process(x)

        ff_2 = FormantFilters(N=64, fs=16000.0)
        ff_2.update(f=torch.full((1, 2, 4), 1000.0), r=torch.full((1, 2, 4), 0.9))
        y_2 = ff_2.process(x)

        assert not torch.allclose(y_1, y_2)

    def test_clear_resets_state(self):
        ff = FormantFilters(N=64, fs=16000.0)
        ff.update(f=torch.full((1, 1, 4), 1000.0), r=torch.full((1, 1, 4), 0.9))
        ff.process(torch.randn(1, 256))
        ff.clear()
        assert ff.num_filters == 0
        with pytest.raises(AssertionError):
            ff.process(torch.randn(1, 256))
