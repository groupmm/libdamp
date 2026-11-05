"""Unit tests for `libdamp.datasets.lm_ssd`."""

import numpy as np
import pytest
import soundfile as sf

from libdamp.datasets.lm_ssd import LMSSDDataset


def _write_take(tmp_path, song, singer, take, crosstalk="C0", fs=16000, duration_s=1.0, uid="001"):
    n = int(fs * duration_s)
    lm = 0.1 * np.random.randn(n).astype(np.float32)
    cm = 0.1 * np.random.randn(n).astype(np.float32)
    sf.write(tmp_path / f"SSD{uid}_{song}_LM_{crosstalk}_{singer}_{take}.wav", lm, fs)
    sf.write(tmp_path / f"SSD{uid}_{song}_CM_{crosstalk}_{singer}_{take}.wav", cm, fs)
    # a non-LM/CM signal that should always be ignored by the dataset
    sf.write(tmp_path / f"SSD{uid}_{song}_GL_{crosstalk}_{singer}_{take}.wav", cm, fs)
    return lm, cm


class TestLMSSDDataset:
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LMSSDDataset(str(tmp_path / "does_not_exist"), excerpt_len=100, num_excerpts=1)

    def test_selection_regex_filters_takes(self, tmp_path):
        _write_take(tmp_path, song="AB", singer="1M", take="T1")
        _write_take(tmp_path, song="AB", singer="2M", take="T1")

        ds = LMSSDDataset(str(tmp_path), selection=r"_1M_", excerpt_len=100, num_excerpts=2)
        assert ds.num_takes == 1
        assert len(ds) == 2

    def test_no_match_raises(self, tmp_path):
        _write_take(tmp_path, song="AB", singer="1M", take="T1")
        with pytest.raises(FileNotFoundError):
            LMSSDDataset(str(tmp_path), selection=r"_9X_", excerpt_len=100, num_excerpts=1)

    def test_missing_cm_counterpart_raises(self, tmp_path):
        n = 16000
        sf.write(tmp_path / "SSD001_AB_LM_C0_1M_T1.wav", 0.1 * np.random.randn(n).astype(np.float32), 16000)
        with pytest.raises(FileNotFoundError):
            LMSSDDataset(str(tmp_path), excerpt_len=100, num_excerpts=1)

    def test_getitem_returns_aligned_excerpt_pair(self, tmp_path):
        lm, cm = _write_take(tmp_path, song="AB", singer="1M", take="T1", duration_s=1.0)

        ds = LMSSDDataset(str(tmp_path), excerpt_len=100, num_excerpts=3, random_seed=0)
        x_lm, x_cm = ds[0]

        assert x_lm.shape == (100,)
        assert x_cm.shape == (100,)

        # excerpts must come from a matching position in both signals
        start = ds.snippet_starts[0, 0].item()
        assert np.allclose(x_lm.numpy(), lm[start : start + 100], atol=1e-4)
        assert np.allclose(x_cm.numpy(), cm[start : start + 100], atol=1e-4)

    def test_resamples_to_target_fs(self, tmp_path):
        _write_take(tmp_path, song="AB", singer="1M", take="T1", fs=48000, duration_s=1.0)

        ds = LMSSDDataset(str(tmp_path), target_fs=16000.0, excerpt_len=100, num_excerpts=1)
        assert ds.lm_signals[0].shape[-1] == pytest.approx(16000, abs=2)

    def test_avoids_completely_silent_excerpts(self, tmp_path):
        fs = 16000
        n = fs * 4  # 4 s take
        lm = np.zeros(n, dtype=np.float32)
        active_start, active_end = 20000, 36000  # the only non-silent region of the LM signal
        lm[active_start:active_end] = 0.5 * np.random.randn(active_end - active_start).astype(np.float32)
        cm = 0.1 * np.random.randn(n).astype(np.float32)  # CM activity is irrelevant to the LM-based VAD check
        sf.write(tmp_path / "SSD001_AB_LM_C0_1M_T1.wav", lm, fs)
        sf.write(tmp_path / "SSD001_AB_CM_C0_1M_T1.wav", cm, fs)

        excerpt_len = 16000  # 1 s
        ds = LMSSDDataset(str(tmp_path), target_fs=fs, excerpt_len=excerpt_len, num_excerpts=50, random_seed=0)

        for start in ds.snippet_starts[0].tolist():
            assert start < active_end and start + excerpt_len > active_start, (
                f"excerpt at {start} does not overlap the only non-silent region [{active_start}, {active_end})"
            )

    def test_mostly_silent_excerpt_with_some_activity_is_accepted(self, tmp_path):
        fs = 16000
        n = fs * 4
        lm = np.zeros(n, dtype=np.float32)
        lm[: int(fs * 0.3)] = 0.5 * np.random.randn(int(fs * 0.3)).astype(np.float32)  # 0.3 s of activity
        cm = 0.1 * np.random.randn(n).astype(np.float32)
        sf.write(tmp_path / "SSD001_AB_LM_C0_1M_T1.wav", lm, fs)
        sf.write(tmp_path / "SSD001_AB_CM_C0_1M_T1.wav", cm, fs)

        # should not hang or raise despite the take being >90% silent
        ds = LMSSDDataset(str(tmp_path), target_fs=fs, excerpt_len=16000, num_excerpts=10, random_seed=0)
        assert len(ds) == 10
