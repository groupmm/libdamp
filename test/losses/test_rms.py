"""Unit tests for `libdamp.losses.rms`."""

import torch

from libdamp.losses.rms import RMSLoss


class TestRMSLoss:
    def test_zero_for_identical_signals(self):
        torch.manual_seed(0)
        loss_fn = RMSLoss(N=256, H=128)
        y = torch.randn(2, 4096)
        assert torch.allclose(loss_fn(y, y), torch.zeros(2), atol=1e-5)

    def test_symmetric_in_its_two_inputs(self):
        torch.manual_seed(0)
        loss_fn = RMSLoss(N=256, H=128)
        y = torch.randn(2, 4096)
        y_hat = torch.randn(2, 4096)
        assert torch.allclose(loss_fn(y, y_hat), loss_fn(y_hat, y), atol=1e-5)

    def test_output_shape_matches_batch_size(self):
        loss_fn = RMSLoss(N=256, H=128)
        y = torch.randn(3, 4096)
        y_hat = torch.randn(3, 4096)
        out = loss_fn(y, y_hat)
        assert out.shape == (3,)

    def test_amplitude_scaling_matches_log_formula(self):
        torch.manual_seed(0)
        loss_fn = RMSLoss(N=256, H=128)
        y = torch.randn(1, 4096).abs() + 0.1  # avoid sign flips from the gain
        gain = 2.0
        y_hat = y * gain
        out = loss_fn(y, y_hat)
        # rms scales linearly with amplitude gain, so the per-frame ratio is exactly `gain`
        assert torch.allclose(out, torch.tensor([10 * torch.log10(torch.tensor(gain))]), atol=1e-3)

    def test_accepts_1d_input_via_ensure_tensor(self):
        loss_fn = RMSLoss(N=256, H=128)
        y = torch.randn(4096)
        y_hat = torch.randn(4096)
        out_1d = loss_fn(y, y_hat)
        out_2d = loss_fn(y[None, :], y_hat[None, :])
        assert torch.allclose(out_1d, out_2d)
