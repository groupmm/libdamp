"""Unit tests for `libdamp.losses.mss`."""

import torch

from libdamp.losses.mss import MSSLoss


class TestMSSLoss:
    def test_zero_for_identical_signals(self):
        torch.manual_seed(0)
        loss_fn = MSSLoss(fft_sizes=(64, 32))
        y = torch.randn(2, 4096)
        assert torch.allclose(loss_fn(y, y), torch.zeros(2), atol=1e-5)

    def test_output_shape_matches_batch_size(self):
        loss_fn = MSSLoss(fft_sizes=(64, 32))
        y = torch.randn(3, 4096)
        y_hat = torch.randn(3, 4096)
        out = loss_fn(y, y_hat)
        assert out.shape == (3,)

    def test_integer_weight_does_not_crash(self):
        # w_mag=1 (an int, not 1.0) used to crash: the float-only isinstance check left it as a
        # bare scalar instead of broadcasting it into a per-scale list.
        loss_fn = MSSLoss(w_mag=1, fft_sizes=(64, 32))
        y = torch.randn(2, 4096)
        y_hat = torch.randn(2, 4096)
        out = loss_fn(y, y_hat)
        assert torch.is_tensor(out)
        assert out.shape == (2,)

    def test_all_zero_weights_returns_zero_tensor_of_correct_shape(self):
        # previously returned a bare Python float 0.0 instead of a (batch,) tensor.
        loss_fn = MSSLoss(w_mag=0.0, w_log=0.0, w_mlog=0.0, fft_sizes=(64, 32))
        y = torch.randn(2, 4096)
        y_hat = torch.randn(2, 4096)
        out = loss_fn(y, y_hat)
        assert torch.is_tensor(out)
        assert out.shape == (2,)
        assert torch.all(out == 0)

    def test_per_scale_weight_list_matches_equivalent_scalar(self):
        torch.manual_seed(0)
        y = torch.randn(2, 4096)
        y_hat = torch.randn(2, 4096)

        loss_scalar = MSSLoss(w_mag=0.5, fft_sizes=(64, 32))(y, y_hat)
        loss_list = MSSLoss(w_mag=[0.5, 0.5], fft_sizes=(64, 32))(y, y_hat)
        assert torch.allclose(loss_scalar, loss_list)

    def test_disabling_a_scale_changes_the_loss(self):
        torch.manual_seed(0)
        y = torch.randn(2, 4096)
        y_hat = torch.randn(2, 4096)

        loss_both = MSSLoss(w_mag=[1.0, 1.0], fft_sizes=(64, 32))(y, y_hat)
        loss_one = MSSLoss(w_mag=[1.0, 0.0], fft_sizes=(64, 32))(y, y_hat)
        assert not torch.allclose(loss_both, loss_one)

    def test_log_magnitude_weight_is_sensitive_to_quiet_differences(self):
        # a small additive difference at very low magnitude barely changes linear-magnitude loss,
        # but changes log-magnitude loss much more (since log spreads out small values).
        torch.manual_seed(0)
        y = torch.randn(1, 4096) * 1e-3
        y_hat = y + 1e-4 * torch.randn(1, 4096)

        loss_mag = MSSLoss(w_mag=1.0, w_log=0.0, fft_sizes=(64,))(y, y_hat)
        loss_log = MSSLoss(w_mag=0.0, w_log=1.0, fft_sizes=(64,))(y, y_hat)
        assert loss_log > loss_mag

    def test_update_changes_subsequent_forward_calls(self):
        torch.manual_seed(0)
        loss_fn = MSSLoss(w_mag=1.0, fft_sizes=(64, 32))
        y = torch.randn(2, 4096)
        y_hat = torch.randn(2, 4096)

        out_before = loss_fn(y, y_hat)
        loss_fn.update(w_mag=0.0)
        out_after = loss_fn(y, y_hat)
        assert torch.all(out_after == 0)
        assert not torch.allclose(out_before, out_after)

    def test_update_with_integer_weight_does_not_crash(self):
        loss_fn = MSSLoss(w_mag=1.0, fft_sizes=(64, 32))
        loss_fn.update(w_mag=1)  # int, not float
        y = torch.randn(2, 4096)
        y_hat = torch.randn(2, 4096)
        out = loss_fn(y, y_hat)
        assert torch.is_tensor(out)

    def test_p_exponent_increases_sensitivity_to_large_differences(self):
        torch.manual_seed(0)
        y = torch.randn(1, 4096)
        y_hat = y + torch.randn(1, 4096)  # comparatively large difference

        loss_p1 = MSSLoss(w_mag=1.0, p=1, fft_sizes=(64,))(y, y_hat)
        loss_p2 = MSSLoss(w_mag=1.0, p=2, fft_sizes=(64,))(y, y_hat)
        assert loss_p1 > 0
        assert loss_p2 > 0
