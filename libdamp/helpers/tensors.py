"""Collection of helper functions that operate on tensors.

This module is part of the libdamp package.
"""

from typing import Literal

import torch
from torchaudio.functional import convolve

from libdamp.helpers.transforms import get_window


def tensor_linspace(start: torch.Tensor, end: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Vectorized version of torch.linspace.

    Parameters
    ----------
    start : torch.Tensor
        Starting values of any shape.
    end : torch.Tensor
        Ending values of the same shape as `start`.
    steps : int
        Number of steps to interpolate between start and end.

    Returns
    -------
    torch.Tensor
        Interpolated values of shape start.size() + (steps,), where the first element
        equals `start`, the last equals `end`, and intermediate elements linearly interpolate.

    Notes
    -----
    From https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def apply_along_dim(x: torch.Tensor, func, dim: int = 0) -> torch.Tensor:
    """Apply a function along a specified dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input data tensor.
    func : callable
        Function to be applied on each subtensor.
    dim : int
        Dimension along which to apply the function (default: 0).

    Returns
    -------
    torch.Tensor
        Result tensor with `func` applied along the specified dimension.

    Notes
    -----
    Use with caution as this is typically slow compared to vectorized operations.
    """
    return torch.stack([func(row) for row in torch.unbind(x, dim=dim)], dim=dim)


def ensure_tensor(
    x,
    dtype: torch.dtype | None = None,
    min_dims: int | None = None,
    throw_smaller: bool = False,
    throw_larger: bool = False,
) -> torch.Tensor:
    """Converts lists or numpy arrays into a torch.Tensor if required

    Parameters
    ----------
    x : Any
        Data that will be converted to a torch.Tensor if it not already is one.
    dtype : torch.dtype or None
        Optional target dtype for the tensor. If `None` is given, the type is inferred from `x` (default: `None`).
    min_dims : int or None
        Optional target number of dimensions for the tensor. If `None` is given, the tensor is not expanded. The
        behavior if `x` has too few or too many dimensions depends on `throw_smaller` and `throw_larger` (see below).
    throw_smaller : bool
        If `False`, new dimensions are added to the beginning of the tensor if it has too few dimensions. Otherwise an
        error is thrown if the number of dimensions is too small (default: `False`).
    throw_larger : bool
        If `False`, nothing happens if the number of dimensions of `x` is larger than `min_dims`. Otherwise an
        error is thrown if the number of dimensions is too large (default: `False`).

    Returns
    -------
    torch.Tensor
        The input converted to a tensor with the specified properties.
    """

    if not torch.is_tensor(x):
        y = torch.tensor(x)
    else:
        y = x

    if dtype is not None:
        y = y.to(dtype)

    if min_dims is not None:
        if len(y.shape) < min_dims and not throw_smaller:
            while len(y.shape) < min_dims:
                y = y[None, ...]

        if throw_smaller:
            assert len(y.shape) >= min_dims, "Tensor does not have the required number of dimensions."
        if throw_larger:
            assert len(y.shape) <= min_dims, "Tensor does not have the required number of dimensions."

    return y


def interpolate_samples(
    x: torch.Tensor,
    N: int,
    mode: Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"] = "const",
    prev_val: torch.Tensor | None = None,
):
    """Interpolation of the last axis of a tensor `x`.

    This function provides several ways to expand a frame-wise representation of some parameters to a sample-wise one.

    Parameters
    ----------
    x : torch.Tensor
        input tensor with shape (batch, channels, num_frames), where the last axis gives one value per frame.
    N : int
        Target number of samples per frame.
    mode : Literal["const", "center_linear", "end_linear", "half_linear", "const_smooth"]
        interpolation mode (default: "const")

        Available options:

        - "const": the parameter value stays constant for each whole frame
        - "center_linear": linear interpolation over the full length-N frame, where the given value in `x` is reached
          in the middle of the frame. Note that this mode may behave similar to other DDSP libraries, but is not
          compatible with consecutive frame-wise processing, since the values in the second half of the last current
          frame depend on the target value for the following frame.
        - "end_linear": linear interpolation over the full length-N frame, where the given value in `x` is reached at
          the end of the frame. This mode is compatible with consecutive frame-wise processing if a `prev_val` is
          provided.
        - "half_linear": linear interpolation over the first half of a length-N frame, where the given value in `x` is
          reached in the middle of the frame and stays constant until the end. This mode is compatible with consecutive
          frame-wise processing if a `prev_val` is provided.
        - "const_smooth": like "const", but the resulting tensor is smoothed by convolving with a Hann window of size
          `N // 4`. This mode is not compatible with consecutive frame-wise processing, since the values in the second
          half of the last current frame depend on the target value for the following frame.
    prev_val : torch.Tensor
        optional starting value for the interpolation method (see above, default: None)

    Returns
    -------
    y : torch.Tensor
        output tensor with shape (batch, channels, N*num_frames), where the last axis contains interpolated values.

    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    if mode == "const":
        y = torch.repeat_interleave(x, N, dim=-1)
    elif mode == "center_linear":
        F = x.shape[-1]
        y = torch.nn.functional.interpolate(x, size=F * N, mode="linear", align_corners=False)
    elif mode == "end_linear":
        v = x[..., 0] if prev_val is None else prev_val
        p = torch.cat([v[..., None], x], dim=-1)
        F = p.shape[-1]
        start = N // 2
        end = -1 * (N - start)
        y = torch.nn.functional.interpolate(p, size=F * N, mode="linear", align_corners=False)[..., start:end]
    elif mode == "half_linear":
        p = torch.repeat_interleave(x, 2, dim=-1)
        v = p[..., 0] if prev_val is None else prev_val
        p = torch.cat([v[..., None], p], dim=-1)
        F = p.shape[-1]
        start = N // 4
        end = -1 * (N // 2 - start)
        y = torch.nn.functional.interpolate(p, size=F * N // 2, mode="linear", align_corners=False)[..., start:end]
    elif mode == "const_smooth":
        y = smooth(torch.repeat_interleave(x, N, dim=-1), win_length=N // 4, win_type="hann")
    else:
        raise ValueError(f"Interpolation mode '{mode}' unknown.")

    return y


def smooth(x: torch.Tensor, win_length: int = 3, win_type: str = "rectangular", pad_mode: str = "replicate") -> torch.Tensor:
    """Smooth a tensor along its last axis using windowed convolution.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to smooth along the last axis.
    win_length : int
        Window length for the smoothing (default: 3).
    win_type : str
        Window type for the smoothing. See `libdamp.helpers.transforms.get_window`
        for valid options (default: "rectangular").
    pad_mode : str
        Padding mode for convolution borders (default: "replicate").

    Returns
    -------
    torch.Tensor
        Smoothed tensor with the same shape as input.
    """
    win_length += 1
    x = ensure_tensor(x)
    orig_shape = x.shape
    y = ensure_tensor(x, min_dims=2)

    w = ensure_tensor(get_window(win_type, win_length), min_dims=len(y.shape), dtype=y.dtype)

    z = convolve(torch.nn.functional.pad(y, (win_length - 1, win_length - 1), mode=pad_mode), w / w.sum(), mode="valid")

    start = win_length // 2
    end = -win_length // 2 + 1
    return z[..., start:end].reshape(orig_shape)


def cubic_hermite_splines(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the first four cubic Hermite splines at the given points x

    Parameters
    ----------
    x : torch.Tensor
        Points at which the splines should be evaluated along the last axis, shape: (..., N).

    Returns
    -------
    y : torch.Tensor
        Evaluated values for the first four cubic Hermite splines, shape: (..., 4, N).
    """
    xx = torch.transpose(x[..., None] ** torch.arange(4).to(x), -2, -1)
    A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=x.dtype).to(x)
    return A @ xx


def interpolate_pchip(x: torch.Tensor, y: torch.Tensor, xs: torch.Tensor, extrapolate: Literal["const", "linear"] = "const") -> torch.Tensor:
    """Interpolation with a Piecewise Cubic Hermite Interpolating Polynomial

    This function interpolates points based on a Piecewise Cubic Hermite Interpolating Polynomial (PCHIP). It ensures a
    smooth function by making sure that the slope (i.e., first derivative) at each support point is continuous.
    For efficiency, the inputs `x` and `xs` are expected to be sorted.

    Parameters
    ----------
    x : torch.Tensor
        x values of the support points, shape: (..., N) – it is possible to give a 1D `x` and a 2D `y`, e.g., if the
        x values are constant across batches.
    y : torch.Tensor
        y values of the support points, shape same as `x`, or with an extra batch dimension.
    xs : torch.Tensor
        x values at which the PCHIP shall be evaluated, shape: (..., M) – a single `xs` can be given, e.g., if the
        interpolation points are constant across batches.
    extrapolate : Literal["const", "linear"]
        method used for values in `xs` that are outside of the support points

        Options:

        - "const": use the y value for the first and last support point for all `xs` outside of the support point range.
          The slope at the edge points is set to 0 to ensure a smooth transition.
        - "linear": use the slope of the edge points to calculate a linearly interpolated y value for all `xs` outside
          of the support point range.

    Returns
    -------
    ys : torch.Tensor
        Interpolated values at `xs`, shape: (..., M).

    See also
    --------
    This implementation is roughly based on the 1D-version in
    https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch
    """

    # if x has only one dimension while y has more, we expand it accordingly
    if len(x.shape) == 1 and len(x.shape) != len(y.shape):
        for i in range(len(y.shape) - 2, -1, -1):
            x = torch.repeat_interleave(x[None, ...], y.shape[i], dim=0)

    # if xs has only one dimension while x and y have more, we expand it accordingly
    if len(xs.shape) == 1 and len(xs.shape) != len(y.shape):
        for i in range(len(y.shape) - 2, -1, -1):
            xs = torch.repeat_interleave(xs[None, ...], y.shape[i], dim=0)

    assert x.shape == y.shape, "Shapes of x and y must be the same."
    assert x.shape[:-1] == xs.shape[:-1], "Batch dimensions of x and xs must be the same."

    # calculate slopes at the support points
    step = x[..., 1:] - x[..., :-1]
    a = (y[..., 1:] - y[..., :-1]) / step

    # refine slopes to avoid any "overshooting" of the interpolating polynomials
    stepm = step[..., :-1] + step[..., 1:]

    # custom slope for first support point
    if extrapolate == "const":
        # for a smooth transition to const extrapolation
        a0 = a[..., [0]] * 0  # preserve shape
    else:
        w1 = (step[..., [0]] + stepm[..., [0]]) / stepm[..., [0]]
        w2 = -1 * step[..., [0]] / stepm[..., [0]]
        a0 = w1 * a[..., [0]] + w2 * a[..., [1]]

    # custom slope for last support point
    if extrapolate == "const":
        # for a smooth transition to const extrapolation
        am1 = a[..., [-1]] * 0  # preserve shape
    else:
        w1 = -1 * step[..., [-1]] / stepm[..., [-1]]
        w2 = (step[..., [-1]] + stepm[..., [-1]]) / stepm[..., [-1]]
        am1 = w1 * a[..., [-2]] + w2 * a[..., [-1]]

    # slopes for support points with two neighbors
    w1 = (1 + step[..., :-1] / stepm) / 3
    w2 = (1 + step[..., 1:] / stepm) / 3
    a_min = torch.minimum(torch.abs(a[..., :-1]), torch.abs(a[..., 1:]))
    a_max = torch.maximum(torch.abs(a[..., :-1]), torch.abs(a[..., 1:]))
    a1_to_am2 = a_min / (w1 * (a[..., :-1] / a_max) + w2 * (a[..., 1:] / a_max))

    changing_sign = (a[..., :-1] * a[..., 1:]) < 0  # neighboring slopes have different signs
    zero_mask = (changing_sign) | (a[..., :-1] == 0) | (a[..., 1:] == 0)
    a1_to_am2[zero_mask] = 0

    a = torch.cat([a0, a1_to_am2, am1], dim=-1)

    # find nearest lower support point for each interpolation target
    # `.contiguous()` avoids a torch warning: slicing off the first element along the last
    # dimension breaks contiguity for x.ndim > 1, since the row stride no longer matches the
    # now-shorter row length, so torch.searchsorted would otherwise silently make this same copy.
    idxs = torch.searchsorted(x[..., 1:].contiguous(), xs)

    # cannot interpolate outside of range (we will take care of extrapolation later)
    # this currently calculates some values twice, is there a more efficient way?
    idxs[(idxs >= x.shape[-1] - 1)] = 0
    x_idx = torch.gather(x, -1, idxs)

    # evaluate first 4 hermite splines on the normalized intervals between each support point
    dx = torch.gather(x, -1, idxs + 1) - x_idx
    h = cubic_hermite_splines((xs - x_idx) / dx)

    # calculate y values that can be interpolated
    ys = (
        h[..., 0, :] * torch.gather(y, -1, idxs)
        + h[..., 1, :] * torch.gather(a, -1, idxs) * dx
        + h[..., 2, :] * torch.gather(y, -1, idxs + 1)
        + h[..., 3, :] * torch.gather(a, -1, idxs + 1) * dx
    )

    # extrapolate values that are outside of the given support points
    L = xs.shape[-1]

    outer_dims = []
    D = len(ys.shape) - 1
    for i in range(D):
        shape = [1] * D
        shape[i] = -1
        outer_dims.append(torch.arange(ys.shape[i]).reshape(shape))

    L_el = (xs < x[..., [0]]).sum(dim=-1)
    mask_el = torch.zeros_like(ys)
    mask_el[tuple(outer_dims + [L_el])] = 1
    mask_el = 1 - mask_el.cumsum(dim=-1)
    mask_el = mask_el.bool()

    L_eu = (xs <= x[..., [-1]]).sum(dim=-1)
    mask_eu = torch.zeros_like(ys)
    if torch.all(L_eu < mask_eu.shape[-1]):  # this only works if some xs values are larger than the largest x
        mask_eu[tuple(outer_dims + [L_eu])] = 1
        mask_eu = mask_eu.cumsum(dim=-1)
    mask_eu = mask_eu.bool()

    def rep(x, by):  # helper function to repeat each value in a tensor by a variable amount along each dimension
        return torch.repeat_interleave(x.flatten(), by.flatten())

    if extrapolate == "linear":
        ys[mask_el] = rep(a[..., 0], L_el) * (xs[mask_el] - rep(x[..., 0], L_el)) + rep(y[..., 0], L_el)
        ys[mask_eu] = rep(a[..., -1], L - L_eu) * (xs[mask_eu] - rep(x[..., -1], L - L_eu)) + rep(y[..., -1], L - L_eu)
    elif extrapolate == "const":
        ys[mask_el] = rep(y[..., 0], L_el)
        ys[mask_eu] = rep(y[..., -1], L - L_eu)
    else:
        raise ValueError(f"Extrapolation method '{extrapolate}' unknown.")

    return ys


def interpolate_linear(x: torch.Tensor, y: torch.Tensor, xs: torch.Tensor, extrapolate: Literal["const", "linear"] = "const") -> torch.Tensor:
    """Linear interpolation with arbitrary support and sampling points.

    For efficiency, the inputs `x` and `xs` are expected to be sorted.

    Parameters
    ----------
    x : torch.Tensor
        x values of the support points, shape: (..., N) – it is possible to give a 1D `x` and a 2D `y`, e.g., if the
        x values are constant across batches.
    y : torch.Tensor
        y values of the support points, shape same as `x`, or with an extra batch dimension.
    xs : torch.Tensor
        x values at which the linear interpolation shall be evaluated, shape: (..., M) – a single `xs` can be given,
        e.g., if the interpolation points are constant across batches.
    extrapolate : Literal["const", "linear"]
        method used for values in `xs` that are outside of the support points

        Options:

        - "const": use the y value for the first and last support point for all `xs` outside of the support point range.
          The slope at the edge points is set to 0 to ensure a smooth transition.
        - "linear": use the slope of the edge points to calculate a linearly interpolated y value for all `xs` outside
          of the support point range.

    Returns
    -------
    ys : torch.Tensor
        Interpolated values at `xs`, shape: (..., M).

    """

    # if x has only one dimension while y has more, we expand it accordingly
    if len(x.shape) == 1 and len(x.shape) != len(y.shape):
        for i in range(len(y.shape) - 2, -1, -1):
            x = torch.repeat_interleave(x[None, ...], y.shape[i], dim=0)

    # if xs has only one dimension while x and y have more, we expand it accordingly
    if len(xs.shape) == 1 and len(xs.shape) != len(y.shape):
        for i in range(len(y.shape) - 2, -1, -1):
            xs = torch.repeat_interleave(xs[None, ...], y.shape[i], dim=0)

    assert x.shape == y.shape, "Shapes of x and y must be the same."
    assert x.shape[:-1] == xs.shape[:-1], "Batch dimensions of x and xs must be the same."

    # find nearest lower support point for each interpolation target
    # `.contiguous()` avoids a torch warning: slicing off the first element along the last
    # dimension breaks contiguity for x.ndim > 1, since the row stride no longer matches the
    # now-shorter row length, so torch.searchsorted would otherwise silently make this same copy.
    idxs = torch.searchsorted(x[..., 1:].contiguous(), xs)

    # cannot interpolate outside of range (we will take care of extrapolation later)
    # this currently calculates some values twice, is there a more efficient way?
    idxs[(idxs >= x.shape[-1] - 1)] = 0
    x_idx = torch.gather(x, -1, idxs)

    # find relative position of new point on the interval
    dx = torch.gather(x, -1, idxs + 1) - x_idx
    h = (xs - x_idx) / dx

    # calculate y values that can be interpolated
    ys = (1 - h) * torch.gather(y, -1, idxs) + h * torch.gather(y, -1, idxs + 1)

    # extrapolate values that are outside of the given support points
    L = xs.shape[-1]

    outer_dims = []
    D = len(ys.shape) - 1
    for i in range(D):
        shape = [1] * D
        shape[i] = -1
        outer_dims.append(torch.arange(ys.shape[i]).reshape(shape))

    L_el = (xs < x[..., [0]]).sum(dim=-1)
    mask_el = torch.zeros_like(ys)
    mask_el[tuple(outer_dims + [L_el])] = 1
    mask_el = 1 - mask_el.cumsum(dim=-1)
    mask_el = mask_el.bool()

    L_eu = (xs <= x[..., [-1]]).sum(dim=-1)
    mask_eu = torch.zeros_like(ys)
    if torch.all(L_eu < mask_eu.shape[-1]):  # this only works if some xs values are larger than the largest x
        mask_eu[tuple(outer_dims + [L_eu])] = 1
        mask_eu = mask_eu.cumsum(dim=-1)
    mask_eu = mask_eu.bool()

    def rep(x, by):  # helper function to repeat each value in a tensor by a variable amount along each dimension
        return torch.repeat_interleave(x.flatten(), by.flatten())

    if extrapolate == "linear":
        a0 = (y[..., 1] - y[..., 0]) / (x[..., 1] - x[..., 0])  # slope at first point
        am1 = (y[..., -1] - y[..., -2]) / (x[..., -1] - x[..., -2])  # slope at last point
        ys[mask_el] = rep(a0, L_el) * (xs[mask_el] - rep(x[..., 0], L_el)) + rep(y[..., 0], L_el)
        ys[mask_eu] = rep(am1, L - L_eu) * (xs[mask_eu] - rep(x[..., -1], L - L_eu)) + rep(y[..., -1], L - L_eu)
    elif extrapolate == "const":
        ys[mask_el] = rep(y[..., 0], L_el)
        ys[mask_eu] = rep(y[..., -1], L - L_eu)
    else:
        raise ValueError(f"Extrapolation method '{extrapolate}' unknown.")

    return ys


def poly(roots):
    """
    Equivalent of `numpy.poly` in PyTorch

    Parameters
    ----------
    roots : torch.Tensor
        roots along the last dimension
    """

    roots = ensure_tensor(roots, min_dims=2)

    shape = list(roots.shape)
    N = shape[-1]  # number of roots
    shape[-1] = 1

    coeffs = torch.zeros(shape).to(roots)
    coeffs[..., 0] = 1.0

    for i in range(N):
        r = roots[..., i]
        coeffs = torch.nn.functional.pad(coeffs, (0, 1)) - r.unsqueeze(-1) * torch.nn.functional.pad(coeffs, (1, 0))

    return coeffs
