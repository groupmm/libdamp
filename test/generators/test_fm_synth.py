"""Unit tests for `libdamp.generators.fm_synth`."""

import pytest
import torch

from libdamp.generators.fm_synth import FMSynth


class TestFMSynth:
    def test_requires_update_before_generate(self):
        fm = FMSynth(frame_len=64, fs=16000.0, num_ops=1, connections=[])
        with pytest.raises(AssertionError):
            fm.generate()

    def test_single_unconnected_operator_is_a_pure_sine(self):
        fs = 16000.0
        N = 256
        f0 = 220.0
        fm = FMSynth(frame_len=N, fs=fs, num_ops=1, connections=[])
        fm.update(f0=torch.full((1, 2), f0), r=torch.ones(1, 2, 1), m=torch.ones(1, 2, 1))
        out = fm.generate()

        t = torch.arange(1, out.shape[-1] + 1)
        expected = torch.sin(2 * torch.pi * f0 * t / fs)
        assert torch.allclose(out[0], expected, atol=1e-3)

    def test_disconnected_non_root_operator_raises(self):
        with pytest.raises(AssertionError, match="is not connected"):
            FMSynth(frame_len=64, fs=16000.0, num_ops=3, connections=[(0, 1), (1, 0)])

    def test_too_few_connections_raises(self):
        with pytest.raises(AssertionError, match="connections required"):
            FMSynth(frame_len=64, fs=16000.0, num_ops=3, connections=[(1, 0)])

    def test_modulation_changes_output_compared_to_no_modulation(self):
        fs = 16000.0
        N = 256
        f0 = 220.0

        fm_mod = FMSynth(frame_len=N, fs=fs, num_ops=2, connections=[(1, 0)])
        fm_mod.update(f0=torch.full((1, 1), f0), r=torch.ones(1, 1, 2), m=torch.tensor([[[1.0, 5.0]]]))
        x_mod = fm_mod.generate()

        fm_nomod = FMSynth(frame_len=N, fs=fs, num_ops=2, connections=[(1, 0)])
        fm_nomod.update(f0=torch.full((1, 1), f0), r=torch.ones(1, 1, 2), m=torch.tensor([[[1.0, 0.0]]]))
        x_nomod = fm_nomod.generate()

        assert not torch.allclose(x_mod, x_nomod)

    def test_phase_is_continuous_across_successive_generate_calls(self):
        fs = 16000.0
        N = 64
        f0 = 220.0

        fm = FMSynth(frame_len=N, fs=fs, num_ops=2, connections=[(1, 0)])
        fm.update(f0=torch.full((1, 2), f0), r=torch.ones(1, 2, 2), m=torch.ones(1, 2, 2))
        x1 = fm.generate()
        fm.update(f0=torch.full((1, 2), f0), r=torch.ones(1, 2, 2), m=torch.ones(1, 2, 2))
        x2 = fm.generate()
        x_split = torch.cat([x1, x2], dim=-1)

        fm_long = FMSynth(frame_len=N, fs=fs, num_ops=2, connections=[(1, 0)])
        fm_long.update(f0=torch.full((1, 4), f0), r=torch.ones(1, 4, 2), m=torch.ones(1, 4, 2))
        x_long = fm_long.generate()

        assert torch.allclose(x_split, x_long, atol=1e-4)

    def test_clear_resets_state(self):
        fm = FMSynth(frame_len=64, fs=16000.0, num_ops=1, connections=[])
        fm.update(f0=torch.full((1, 1), 220.0), r=torch.ones(1, 1, 1), m=torch.ones(1, 1, 1))
        fm.generate()
        fm.clear()
        assert fm.initialized is False
        assert fm.prev_phi == {}
        with pytest.raises(AssertionError):
            fm.generate()
