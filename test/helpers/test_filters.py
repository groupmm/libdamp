"""Unit tests for `libdamp.helpers.filters`."""

import scipy.signal
import torch

from libdamp.helpers.filters import combined_freqz, design_butter_filter, design_resonant_filter, freqz


class TestFreqz:
    def test_matches_scipy_freqz_for_a_known_filter(self):
        b, a = scipy.signal.butter(2, 0.2)
        H = freqz(torch.Tensor(b), torch.Tensor(a), N=512)
        _, H_ref = scipy.signal.freqz(b, a, worN=H.shape[-1])
        assert torch.allclose(H, torch.tensor(H_ref, dtype=H.dtype), atol=1e-4)

    def test_dc_gain_of_normalized_lowpass_is_unity(self):
        b, a = scipy.signal.butter(2, 0.2)
        H = freqz(torch.Tensor(b), torch.Tensor(a), N=2048)
        assert torch.allclose(H[0], torch.tensor(1.0 + 0j), atol=1e-4)

    def test_requires_a0_equal_to_one(self):
        b = torch.tensor([1.0])
        a = torch.tensor([2.0])
        try:
            freqz(b, a)
        except AssertionError:
            pass
        else:
            assert False, "Expected an AssertionError when a[0] != 1."


class TestCombinedFreqz:
    def test_series_combination_multiplies_responses(self):
        b = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        a = torch.tensor([[[1.0, -0.5], [1.0, -0.5]]])
        H_combined = combined_freqz(b, a, N=256, parallel=False)
        H_single = freqz(b[..., 0, :], a[..., 0, :], N=256)
        assert torch.allclose(H_combined, H_single**2, atol=1e-4)

    def test_parallel_combination_sums_responses(self):
        b = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        a = torch.tensor([[[1.0, -0.5], [1.0, -0.5]]])
        H_combined = combined_freqz(b, a, N=256, parallel=True)
        H_single = freqz(b[..., 0, :], a[..., 0, :], N=256)
        assert torch.allclose(H_combined, 2 * H_single, atol=1e-4)


class TestDesignButterFilter:
    def test_minus_3db_at_cutoff_frequency(self):
        fs = 16000.0
        fc = torch.tensor(2000.0)
        b, a = design_butter_filter(fc, fs, order=2)
        H = freqz(b, a, N=4096)[0]
        fc_bin = int(fc / (fs / 2) * (H.shape[-1] - 1))
        assert torch.isclose(torch.abs(H[fc_bin]), torch.tensor(0.7071), atol=0.01)

    def test_dc_gain_is_unity(self):
        fs = 16000.0
        fc = torch.tensor(2000.0)
        b, a = design_butter_filter(fc, fs, order=2)
        H = freqz(b, a, N=4096)[0]
        assert torch.isclose(torch.abs(H[0]), torch.tensor(1.0), atol=1e-3)

    def test_higher_order_filter_has_steeper_rolloff(self):
        fs = 16000.0
        fc = torch.tensor(2000.0)
        b2, a2 = design_butter_filter(fc, fs, order=2)
        b4, a4 = design_butter_filter(fc, fs, order=4)
        N = 4096
        H2 = freqz(b2, a2, N=N)[0]
        H4 = freqz(b4, a4, N=N)[0]
        # well above cutoff, the higher-order filter should attenuate more strongly
        stop_bin = int(2 * fc / (fs / 2) * (H2.shape[-1] - 1))
        assert torch.abs(H4[stop_bin]) < torch.abs(H2[stop_bin])


class TestDesignResonantFilter:
    def test_peak_is_located_at_target_frequency(self):
        fs = 16000.0
        f = torch.tensor(1000.0)
        r = torch.tensor(0.99)
        b, a = design_resonant_filter(f, r, fs)
        H = freqz(b, a, N=4096)
        peak_bin = torch.argmax(torch.abs(H))
        peak_freq = peak_bin / (H.shape[-1] - 1) * (fs / 2)
        assert torch.isclose(peak_freq, f, atol=fs / 4096 * 2)

    def test_sharper_resonance_for_radius_closer_to_one(self):
        fs = 16000.0
        f = torch.tensor(1000.0)
        b_sharp, a_sharp = design_resonant_filter(f, torch.tensor(0.999), fs)
        b_wide, a_wide = design_resonant_filter(f, torch.tensor(0.9), fs)
        N = 4096
        H_sharp = torch.abs(freqz(b_sharp, a_sharp, N=N))
        H_wide = torch.abs(freqz(b_wide, a_wide, N=N))
        # a sharper resonance has a smaller -3dB bandwidth, i.e. its response away
        # from the peak (but still near it) drops off faster
        offset_bin = 10
        peak_bin = torch.argmax(H_sharp)
        assert H_sharp[peak_bin + offset_bin] < H_wide[peak_bin + offset_bin]
