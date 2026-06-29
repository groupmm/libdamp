"""Unit tests for `libdamp.helpers.scaling`."""

import torch

from libdamp.helpers.scaling import exp_sigmoid


class TestExpSigmoid:
    def test_output_is_within_bounds(self):
        x = torch.linspace(-20, 20, 1000)
        y = exp_sigmoid(x)
        assert torch.all(y > 0)
        # x_min=1e-8 is below float32 precision at this magnitude, so y saturates to exactly x_max
        assert torch.all(y <= 2.0 + 1e-6)

    def test_large_negative_input_approaches_x_min(self):
        y = exp_sigmoid(torch.tensor(-1e6), x_min=1e-8)
        assert torch.isclose(y, torch.tensor(1e-8), atol=1e-9)

    def test_large_positive_input_approaches_x_max_plus_x_min(self):
        y = exp_sigmoid(torch.tensor(1e6), x_max=2.0, x_min=1e-8)
        assert torch.isclose(y, torch.tensor(2.0 + 1e-8), atol=1e-6)

    def test_zero_input_is_midpoint_of_sigmoid(self):
        y = exp_sigmoid(torch.tensor(0.0), x_max=2.0, x_min=0.0, exp=1.0)
        assert torch.isclose(y, torch.tensor(1.0))

    def test_exp_increases_steepness_above_midpoint(self):
        x = torch.tensor(2.0)
        y_exp1 = exp_sigmoid(x, x_max=1.0, x_min=0.0, exp=1.0)
        y_exp4 = exp_sigmoid(x, x_max=1.0, x_min=0.0, exp=4.0)
        # sigmoid(2) < 1, so raising it to a higher power makes it smaller
        assert y_exp4 < y_exp1
