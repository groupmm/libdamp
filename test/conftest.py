"""Shared pytest fixtures and import setup.

`libdamp/__init__.py` eagerly imports the entire package (datasets, models, losses, ...) on first
import, which pulls in heavy/optional dependencies unrelated to the module under test. To keep the
test suite for individual submodules independent of those, we register a lightweight stand-in for
the top-level `libdamp` package (pointing at the real package directory) *before* any
`libdamp.<submodule>` is imported, so Python uses it as the parent package instead of executing the
real `__init__.py`.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_LIBDAMP_DIR = Path(__file__).resolve().parent.parent / "libdamp"

if "libdamp" not in sys.modules:
    _spec = importlib.machinery.ModuleSpec("libdamp", loader=None, is_package=True)
    _spec.submodule_search_locations = [str(_LIBDAMP_DIR)]
    _stub = importlib.util.module_from_spec(_spec)
    sys.modules["libdamp"] = _stub
