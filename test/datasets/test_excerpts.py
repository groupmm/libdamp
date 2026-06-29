"""Unit tests for `libdamp.datasets.excerpts`."""

import numpy as np
import pytest
import torch

from libdamp.datasets.excerpts import ExcerptsDataset, SignalF0ExcerptsDataset


def _write_npy(tmp_path, key, ds_split, array):
    np.save(tmp_path / f"{key}_{ds_split}.npy", array)


class TestExcerptsDataset:
    def test_loads_requested_keys_from_matching_files(self, tmp_path):
        signal = np.random.randn(4, 1, 16).astype(np.float32)
        f0 = np.random.rand(4, 8).astype(np.float32)
        _write_npy(tmp_path, "signal", "train", signal)
        _write_npy(tmp_path, "f0", "train", f0)

        ds = ExcerptsDataset(str(tmp_path), ds_split="train", return_keys=["signal", "f0"])

        assert len(ds) == 4
        x, y = ds[0]
        assert torch.allclose(x, torch.Tensor(signal[0]))
        assert torch.allclose(y, torch.Tensor(f0[0]))

    def test_return_order_follows_return_keys(self, tmp_path):
        a = np.zeros((2, 3), dtype=np.float32)
        b = np.ones((2, 3), dtype=np.float32)
        _write_npy(tmp_path, "a", "train", a)
        _write_npy(tmp_path, "b", "train", b)

        ds = ExcerptsDataset(str(tmp_path), ds_split="train", return_keys=["b", "a"])
        out = ds[0]
        assert torch.equal(out[0], torch.Tensor(b[0]))
        assert torch.equal(out[1], torch.Tensor(a[0]))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ExcerptsDataset(str(tmp_path), ds_split="train", return_keys=["signal"])

    def test_respects_dataset_split(self, tmp_path):
        train = np.zeros((2, 4), dtype=np.float32)
        val = np.ones((3, 4), dtype=np.float32)
        _write_npy(tmp_path, "signal", "train", train)
        _write_npy(tmp_path, "signal", "val", val)

        ds_train = ExcerptsDataset(str(tmp_path), ds_split="train", return_keys=["signal"])
        ds_val = ExcerptsDataset(str(tmp_path), ds_split="val", return_keys=["signal"])
        assert len(ds_train) == 2
        assert len(ds_val) == 3


class TestSignalF0ExcerptsDataset:
    # The STFT inside SignalF0ExcerptsDataset uses a fixed n_fft of 4096 with center-padding,
    # so the signal must be longer than n_fft // 2 for the padding to be valid.
    F = 8
    SAMPLES = 8192  # divisible by F, long enough for n_fft=4096 with center padding

    def _write_signal_f0(self, tmp_path, n=2, samples=SAMPLES, f0_hz=200.0):
        signal = (0.1 * np.random.randn(n, 1, samples)).astype(np.float32)
        f0 = np.full((n, self.F), f0_hz, dtype=np.float32)
        _write_npy(tmp_path, "signal", "train", signal)
        _write_npy(tmp_path, "f0", "train", f0)
        return signal, f0

    def test_default_return_keys_are_signal_f_a(self, tmp_path):
        self._write_signal_f0(tmp_path)
        ds = SignalF0ExcerptsDataset(str(tmp_path), M=3, F=self.F, fs=16000.0, ds_split="train")
        signal, f, a = ds[0]
        assert signal.shape == (1, self.SAMPLES)
        # "f"/"a" have shape (harmonics, frames) = (M, F)
        assert f.shape == (3, self.F)
        assert a.shape == (3, self.F)

    def test_f_contains_integer_multiples_of_f0(self, tmp_path):
        _, f0 = self._write_signal_f0(tmp_path, f0_hz=200.0)
        ds = SignalF0ExcerptsDataset(str(tmp_path), M=3, F=self.F, fs=16000.0, ds_split="train")
        _, f, _ = ds[0]
        expected = (torch.Tensor(f0[0])[:, None] * torch.arange(1, 4)).transpose(-2, -1)
        assert torch.allclose(f, expected)

    def test_amplitudes_above_nyquist_are_zeroed_but_frequencies_are_kept(self, tmp_path):
        # f0 = 6000 Hz with fs = 16000 Hz: the 2nd harmonic (12000 Hz) exceeds Nyquist (8000 Hz).
        # Only the corresponding amplitude is silenced; the (now meaningless) frequency value
        # itself is left untouched (see commented-out `f[mask] = 0` in the implementation).
        self._write_signal_f0(tmp_path, f0_hz=6000.0)
        ds = SignalF0ExcerptsDataset(str(tmp_path), M=2, F=self.F, fs=16000.0, ds_split="train")
        _, f, a = ds[0]
        assert torch.all(f[1] == 12000.0)
        assert torch.all(a[1] == 0)

    def test_amplitudes_are_non_negative(self, tmp_path):
        self._write_signal_f0(tmp_path, f0_hz=200.0)
        ds = SignalF0ExcerptsDataset(str(tmp_path), M=3, F=self.F, fs=16000.0, ds_split="train")
        _, _, a = ds[0]
        assert torch.all(a >= 0)

    def test_extra_return_key_is_loaded_and_appended(self, tmp_path):
        signal, f0 = self._write_signal_f0(tmp_path)
        extra = np.arange(len(signal), dtype=np.float32)
        _write_npy(tmp_path, "label", "train", extra)

        ds = SignalF0ExcerptsDataset(str(tmp_path), M=2, F=self.F, fs=16000.0, ds_split="train", return_keys=["f0", "label"])
        signal_out, f0_out, label_out = ds[0]
        assert torch.allclose(f0_out, torch.Tensor(f0[0]))
        assert label_out == extra[0]

    def test_signal_length_not_divisible_by_F_raises(self, tmp_path):
        signal = np.zeros((1, 1, 15), dtype=np.float32)  # not divisible by F=4
        f0 = np.zeros((1, self.F), dtype=np.float32)
        _write_npy(tmp_path, "signal", "train", signal)
        _write_npy(tmp_path, "f0", "train", f0)

        ds = SignalF0ExcerptsDataset(str(tmp_path), M=2, F=self.F, fs=16000.0, ds_split="train")
        with pytest.raises(AssertionError):
            ds[0]
