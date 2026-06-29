"""Unit tests for `libdamp.helpers.tensors`."""

import torch

from libdamp.helpers.tensors import (
    apply_along_dim,
    ensure_tensor,
    interpolate_linear,
    interpolate_pchip,
    interpolate_samples,
    poly,
    smooth,
    tensor_linspace,
)


class TestTensorLinspace:
    def test_matches_torch_linspace_for_scalars(self):
        out = tensor_linspace(torch.tensor(0.0), torch.tensor(10.0), steps=5)
        assert torch.allclose(out, torch.linspace(0, 10, 5))

    def test_endpoints_match_start_and_end(self):
        start = torch.tensor([1.0, 2.0])
        end = torch.tensor([5.0, 10.0])
        out = tensor_linspace(start, end, steps=4)
        assert torch.allclose(out[:, 0], start)
        assert torch.allclose(out[:, -1], end)

    def test_output_shape(self):
        start = torch.zeros(3, 2)
        end = torch.ones(3, 2)
        out = tensor_linspace(start, end, steps=7)
        assert out.shape == (3, 2, 7)


class TestApplyAlongDim:
    def test_applies_function_to_each_row(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = apply_along_dim(x, torch.sum, dim=0)
        assert torch.allclose(out, torch.tensor([6.0, 15.0]))


class TestEnsureTensor:
    def test_converts_list_to_tensor(self):
        out = ensure_tensor([1, 2, 3])
        assert torch.is_tensor(out)
        assert torch.equal(out, torch.tensor([1, 2, 3]))

    def test_passes_through_existing_tensor(self):
        x = torch.tensor([1.0, 2.0])
        assert ensure_tensor(x) is x

    def test_casts_dtype(self):
        out = ensure_tensor([1, 2, 3], dtype=torch.float64)
        assert out.dtype == torch.float64

    def test_expands_scalar_to_min_dims_by_default(self):
        out = ensure_tensor(1.0, min_dims=2)
        assert out.shape == (1, 1)

    def test_throw_smaller_raises_if_too_few_dims(self):
        try:
            ensure_tensor(torch.tensor(1.0), min_dims=2, throw_smaller=True)
        except AssertionError:
            pass
        else:
            assert False, "Expected an AssertionError."

    def test_throw_larger_raises_if_too_many_dims(self):
        try:
            ensure_tensor(torch.zeros(2, 2, 2), min_dims=2, throw_larger=True)
        except AssertionError:
            pass
        else:
            assert False, "Expected an AssertionError."

    def test_does_not_raise_when_larger_and_throw_larger_false(self):
        out = ensure_tensor(torch.zeros(2, 2, 2), min_dims=2, throw_larger=False)
        assert out.shape == (2, 2, 2)


class TestInterpolateSamples:
    def test_const_mode_repeats_each_frame(self):
        x = torch.tensor([[[1.0, 2.0, 3.0]]])
        out = interpolate_samples(x, 4, mode="const")
        assert out.shape == (1, 1, 12)
        assert torch.equal(out[0, 0], torch.tensor([1.0] * 4 + [2.0] * 4 + [3.0] * 4))

    def test_output_length_for_all_modes(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        for mode in ["const", "center_linear", "end_linear", "half_linear", "const_smooth"]:
            out = interpolate_samples(x, 8, mode=mode)
            assert out.shape == (1, 1, 32), f"mode={mode}"

    def test_unknown_mode_raises(self):
        x = torch.tensor([[[1.0, 2.0]]])
        try:
            interpolate_samples(x, 4, mode="bogus")
        except ValueError:
            pass
        else:
            assert False, "Expected a ValueError."


class TestSmooth:
    def test_constant_input_is_unchanged(self):
        x = torch.ones(20) * 3.0
        out = smooth(x, win_length=5)
        assert torch.allclose(out, x, atol=1e-5)

    def test_preserves_shape(self):
        x = torch.randn(10)
        out = smooth(x, win_length=3)
        assert out.shape == x.shape

    def test_reduces_variance_of_noisy_signal(self):
        torch.manual_seed(0)
        x = torch.randn(1000)
        out = smooth(x, win_length=9)
        assert out.var() < x.var()


class TestInterpolatePchip:
    def test_interpolates_through_support_points(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        out = interpolate_pchip(x, y, x)
        assert torch.allclose(out, y, atol=1e-5)

    def test_const_extrapolation_holds_edge_value(self):
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 0.0])
        xs = torch.tensor([-5.0, 10.0])
        out = interpolate_pchip(x, y, xs, extrapolate="const")
        assert torch.allclose(out, torch.tensor([0.0, 0.0]), atol=1e-5)

    def test_monotonic_data_stays_monotonic_between_support_points(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([0.0, 1.0, 2.0, 3.0])
        xs = torch.linspace(0, 3, 50)
        out = interpolate_pchip(x, y, xs)
        assert torch.all(out[1:] >= out[:-1] - 1e-5)


class TestInterpolateLinear:
    def test_interpolates_through_support_points(self):
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 2.0, 4.0])
        out = interpolate_linear(x, y, x)
        assert torch.allclose(out, y)

    def test_midpoint_is_linear_average(self):
        x = torch.tensor([0.0, 2.0])
        y = torch.tensor([0.0, 10.0])
        out = interpolate_linear(x, y, torch.tensor([1.0]))
        assert torch.allclose(out, torch.tensor([5.0]))

    def test_const_extrapolation_holds_edge_value(self):
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 4.0])
        xs = torch.tensor([-5.0, 10.0])
        out = interpolate_linear(x, y, xs, extrapolate="const")
        assert torch.allclose(out, torch.tensor([0.0, 4.0]))

    def test_linear_extrapolation_continues_edge_slope(self):
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 3.0])
        xs = torch.tensor([3.0])  # slope at the last segment is 2
        out = interpolate_linear(x, y, xs, extrapolate="linear")
        assert torch.allclose(out, torch.tensor([5.0]))


class TestPoly:
    def test_known_roots_one_and_two(self):
        # (x - 1)(x - 2) = x^2 - 3x + 2
        out = poly(torch.tensor([[1.0, 2.0]]))
        assert torch.allclose(out, torch.tensor([[1.0, -3.0, 2.0]]))

    def test_single_root(self):
        out = poly(torch.tensor([[5.0]]))
        assert torch.allclose(out, torch.tensor([[1.0, -5.0]]))
