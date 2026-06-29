"""Unit tests for `libdamp.helpers.transforms`."""

import torch

from libdamp.helpers.transforms import get_window, hilbert, istft, stft


class TestGetWindow:
    def test_returns_requested_length(self):
        w = get_window("hann", 64)
        assert w.shape == (64,)

    def test_torch_native_window_function(self):
        w = get_window("hann_window", 64)
        assert torch.allclose(w, torch.hann_window(64))

    def test_scipy_fallback_window(self):
        w = get_window("hann", 64)
        assert torch.allclose(w, torch.hann_window(64, periodic=True).to(w.dtype), atol=1e-6)

    def test_rectangular_window_is_all_ones(self):
        w = get_window("boxcar", 32)
        assert torch.allclose(w, torch.ones(32, dtype=w.dtype))


class TestStftIstft:
    def test_istft_reconstructs_original_signal(self):
        torch.manual_seed(0)
        N, H = 256, 64
        x = torch.randn(1, 4096)
        mag, phase = stft(x, N, H)
        X = mag * torch.exp(1j * phase.to(torch.complex64))
        x_hat = istft(X, N, H)

        # ignore edge effects from windowing/centering
        L = min(x.shape[-1], x_hat.shape[-1])
        assert torch.allclose(x[..., H : L - H], x_hat[..., H : L - H], atol=1e-4)

    def test_magnitude_only_output_when_with_phase_false(self):
        x = torch.randn(1, 1024)
        out = stft(x, 256, 64, with_phase=False)
        assert torch.is_tensor(out)
        assert torch.all(out >= 0)

    def test_stft_output_dtype_matches_input_regardless_of_window_source(self):
        x = torch.randn(1, 1024, dtype=torch.float32)
        for win_type in ["hann", "hann_window"]:
            mag, phase = stft(x, 256, 64, win_type=win_type)
            assert mag.dtype == torch.float32
            assert phase.dtype == torch.float32

    def test_istft_output_dtype_matches_input_regardless_of_window_source(self):
        x = torch.randn(1, 1024, dtype=torch.float32)
        X = torch.stft(x, 256, 64, 256, get_window("hann_window", 256), return_complex=True, center=True)
        for win_type in ["hann", "hann_window"]:
            x_hat = istft(X, 256, 64, win_type=win_type)
            assert x_hat.dtype == torch.float32


class TestHilbert:
    def test_real_part_equals_original_signal(self):
        torch.manual_seed(0)
        x = torch.randn(8, 256)
        analytic = hilbert(x)
        assert torch.allclose(analytic.real, x, atol=1e-4)

    def test_analytic_signal_of_cosine_is_complex_exponential(self):
        N = 256
        t = torch.arange(N)
        f = 10
        x = torch.cos(2 * torch.pi * f * t / N)
        analytic = hilbert(x)
        expected = torch.exp(1j * 2 * torch.pi * f * t / N)
        # skip edge regions affected by the FFT-based implicit periodicity
        assert torch.allclose(analytic[N // 4 : -N // 4], expected[N // 4 : -N // 4], atol=1e-3)
