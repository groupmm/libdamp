"""Unit tests for `libdamp.generators.table_osc`."""

import pytest
import torch

from libdamp.generators.table_osc import TableOsc


def _sine_table(L=64, M=2):
    t = torch.arange(L)
    return torch.stack([torch.sin(2 * torch.pi * t / L) for _ in range(M)])


class TestTableOsc:
    def test_output_shape(self):
        table = _sine_table(L=64, M=2)
        osc = TableOsc(frame_len=32, table=table, table_param=torch.tensor([0.0, 1.0]), fs=16000.0, mode="wave")
        osc.update(f0=torch.full((1, 3), 100.0), table_select=torch.zeros(1, 3))
        out = osc.generate()
        assert out.shape == (1, 32 * 3)

    def test_pulse_mode_period_matches_fs_over_f0(self):
        # in "pulse" mode the table entry is read once per period of f0, regardless of table_freq
        L = 256
        table = _sine_table(L=L, M=1)
        fs = 16000.0
        f0 = fs / L  # exactly one period of the table per period of f0
        osc = TableOsc(frame_len=L * 4, table=table, table_param=torch.tensor([0.0]), fs=fs, mode="pulse", interp_entry=False)
        osc.update(f0=torch.full((1, 1), f0), table_select=torch.zeros(1, 1))
        out = osc.generate()
        # the waveform should repeat with period L since one period of f0 reads exactly one table length
        assert torch.allclose(out[0, :L], out[0, L : 2 * L], atol=1e-2)

    def test_unknown_mode_raises(self):
        table = _sine_table()
        with pytest.raises(ValueError, match="Unknown TableOsc mode"):
            TableOsc(frame_len=32, table=table, table_param=torch.tensor([0.0, 1.0]), fs=16000.0, mode="bogus")

    def test_unknown_normalization_raises(self):
        table = _sine_table()
        with pytest.raises(ValueError, match="Unknown normalization method"):
            TableOsc(frame_len=32, table=table, table_param=torch.tensor([0.0, 1.0]), fs=16000.0, normalize="bogus")

    def test_peak_normalization_scales_table_to_unit_peak(self):
        table = torch.stack([torch.full((64,), 0.5), torch.full((64,), 2.0)])
        osc = TableOsc(frame_len=32, table=table, table_param=torch.tensor([0.0, 1.0]), fs=16000.0, normalize="peak")
        assert torch.allclose(osc.pw_table.abs().amax(dim=-1), torch.ones(1, 2), atol=1e-5)

    def test_mismatched_f0_and_table_select_shapes_raise(self):
        table = _sine_table()
        osc = TableOsc(frame_len=32, table=table, table_param=torch.tensor([0.0, 1.0]), fs=16000.0)
        with pytest.raises(AssertionError):
            osc.update(f0=torch.zeros(1, 3), table_select=torch.zeros(1, 4))
