"""Helper function to calculate an incremental modulo

This can be a huge bottleneck if calculated over a long sequence,
so that we leverage Triton to accelerate it on CUDA devices.

Disclaimer: The Triton-accelerated version of `incremental_mod` was generated with Claude Sonnet 4.5
"""

import torch

__all__ = ["incremental_mod"]


def incremental_mod(mod: torch.Tensor, increment: torch.Tensor | None = None) -> torch.Tensor:
    """Compute an "incremental modulo" along the last dimension.

    Unlike a standard elementwise modulo, this operation maintains a running
    counter that is updated step by step. At each position k, the counter is
    incremented, compared against mod[..., k], and wrapped if needed. This
    makes the result stateful, in contrast to `torch.cumsum(increment) % mod`.

    Example
    -------
    >>> mod = torch.Tensor([5, 3, 6])
    >>> inc = torch.Tensor([2, 2, 2])
    >>> incremental_mod(mod, inc)
    Tensor([0., 2., 1.])

    Parameters
    ----------
    mod : torch.Tensor
        Shape (..., N) Tensor of positive moduli
    increment : torch.Tensor, optional
        Increments of the same shape as `mod`. Defaults to ones.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as `mod` containing the incremental modulo
    """
    if increment is None:
        increment = torch.ones_like(mod)

    if mod.is_cuda:
        if _is_triton_available():
            return incremental_mod_triton(mod, increment)
        else:
            # Missing Trition fallback: JIT-compiled direct implementation
            return incremental_mod_jit(mod, increment)
    else:
        # CPU fallback: direct implementation
        return incremental_mod_python(mod, increment)


def _is_triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        return True
    except Exception:
        return False


if _is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _incremental_mod_kernel(mod_ptr, inc_ptr, out_ptr, B: tl.constexpr, N: tl.constexpr):
        b = tl.program_id(0)
        if b >= B:
            return

        base = b * N

        rp = tl.zeros((), dtype=tl.float32)

        for k in range(0, N):
            idx = base + k
            tl.store(out_ptr + idx, rp)
            inc = tl.load(inc_ptr + idx)
            m = tl.load(mod_ptr + idx)
            rp = rp + inc
            rp = tl.where(rp >= m, rp - m, rp)

    def incremental_mod_triton(mod, increment=None):
        """Triton GPU implementation (at least 100x faster than `incremental_mod_python`)"""
        assert mod.is_cuda, "Triton version requires CUDA tensors."
        if increment is None:
            increment = torch.ones_like(mod)

        need_cast_back = mod.dtype != torch.float32
        mod_fp32 = mod.to(torch.float32).contiguous()
        inc_fp32 = increment.to(torch.float32).contiguous()

        N = mod_fp32.shape[-1]
        B = int(mod_fp32.numel() // N)
        mod_2d = mod_fp32.view(B, N).contiguous()
        inc_2d = inc_fp32.view(B, N).contiguous()
        out_2d = torch.empty_like(mod_2d)

        grid = (B,)
        _incremental_mod_kernel[grid](mod_2d, inc_2d, out_2d, B=B, N=N)

        out = out_2d.view_as(mod_fp32)
        if need_cast_back:
            out = out.to(mod.dtype)
        return out

else:

    def _incremental_mod_jit_impl(mod, increment):
        """Compiled direct implementation (still slow, 10x faster than `incremental_mod_python`)"""
        res = torch.zeros(mod.shape, device=mod.device, dtype=mod.dtype)
        rp = torch.zeros(mod.shape[:-1], device=mod.device, dtype=mod.dtype)
        if increment is None:
            increment = torch.ones_like(mod)
        for k in range(mod.shape[-1]):
            res[..., k] = rp
            rp += increment[..., k]
            mask = rp >= mod[..., k]
            rp = torch.where(mask, rp - mod[..., k], rp)

        return res

    # torch.compile() eagerly imports the inductor backend, which is comparatively heavy. Since this function is
    # only used as a CUDA-without-Triton fallback, compile lazily on first actual use instead of  import time,
    # to avoid unnecessary overhead.
    _incremental_mod_jit_compiled = None

    def incremental_mod_jit(mod, increment):
        global _incremental_mod_jit_compiled
        if _incremental_mod_jit_compiled is None:
            _incremental_mod_jit_compiled = torch.compile(_incremental_mod_jit_impl)
        return _incremental_mod_jit_compiled(mod, increment)


def incremental_mod_python(mod: torch.Tensor, increment: torch.Tensor | None) -> torch.Tensor:
    """Direct implementation (very slow)"""
    res = torch.zeros(mod.shape, device=mod.device, dtype=mod.dtype)
    rp = torch.zeros(mod.shape[:-1], device=mod.device, dtype=mod.dtype)
    if increment is None:
        increment = torch.ones_like(mod)
    for k in range(mod.shape[-1]):
        res[..., k] = rp
        rp += increment[..., k]
        mask = rp >= mod[..., k]
        rp = torch.where(mask, rp - mod[..., k], rp)

    return res
