"""Check whether triton is available."""

import sys

import torch

from libdamp.helpers.incremental_mod import incremental_mod, incremental_mod_python


def main() -> int:
    if not torch.cuda.is_available():
        print("No CUDA device available, nothing to check here.")
        return 0

    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception as e:
        print("`import triton` failed.")
        print()
        print(f"Import error: {e!r}")
        return 1

    # check if incremental_mod is producing correct results
    torch.manual_seed(0)
    mod = torch.randint(2, 10, (4, 128), dtype=torch.float32)
    increment = torch.ones_like(mod)

    fast_out = incremental_mod(mod.cuda(), increment.cuda()).cpu()
    reference_out = incremental_mod_python(mod, increment)

    if not torch.allclose(fast_out, reference_out):
        print("Triton is available, but `incremental_mod` produces incorrect results with it.")
        print("This points to a broken Triton installation rather than a missing dependency.")
        return 1

    print("Triton is available and libdamp is using it correctly on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
