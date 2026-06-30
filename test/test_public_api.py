"""Regression test for the public API surface (`__all__` lists across the package).

The whole point is to catch cases where a name was moved between modules but a stale `__all__`
entry was left behind (which previously only surfaced as a confusing `AttributeError` at
docs-build or import time, instead of a clear test failure).
"""

import importlib
import pkgutil

import libdamp


def _modules_with_all():
    """Yield every (importable) module in the libdamp package tree, real submodules included."""
    yield libdamp
    for module_info in pkgutil.walk_packages(libdamp.__path__, prefix="libdamp."):
        try:
            yield importlib.import_module(module_info.name)
        except ImportError as exc:
            raise AssertionError(f"Failed to import '{module_info.name}': {exc}") from exc


_MODULES = list(_modules_with_all())
_MODULES_WITH_ALL = [m for m in _MODULES if hasattr(m, "__all__")]


def test_found_modules_with_all():
    # `__all__` is only defined at the "public surface" layer (the top-level `libdamp` package
    # itself, plus each subpackage's own `__init__.py`), not in individual implementation
    # modules - so this checks the exact expected set rather than just a loose lower bound, both
    # as a sanity check that this test is exercising something, and as a guard against `__all__`
    # creeping back into an implementation module by accident.
    names = {m.__name__ for m in _MODULES_WITH_ALL}
    assert names == {
        "libdamp",
        "libdamp.augment",
        "libdamp.datasets",
        "libdamp.generators",
        "libdamp.helpers",
        "libdamp.losses",
        "libdamp.models",
        "libdamp.processors",
    }


def test_every_module_was_collected():
    # every subpackage should be importable (this fails if e.g. a circular import
    # or a missing file breaks package discovery, rather than `_modules_with_all` silently
    # yielding an incomplete list)
    names = {m.__name__ for m in _MODULES}
    for expected in [
        "libdamp",
        "libdamp.augment",
        "libdamp.datasets",
        "libdamp.generators",
        "libdamp.helpers",
        "libdamp.losses",
        "libdamp.models",
        "libdamp.processors",
    ]:
        assert expected in names, f"Expected module '{expected}' was not discovered."


def test_all_entries_resolve():
    """Every name listed in a module's `__all__` must actually exist on that module."""
    failures = []
    for module in _MODULES_WITH_ALL:
        for name in module.__all__:
            if not hasattr(module, name):
                failures.append(f"{module.__name__}.__all__ lists '{name}', but it does not exist on the module.")

    assert not failures, "\n".join(failures)


def test_all_entries_are_unique_per_module():
    """Catches accidental duplicate entries in a single `__all__` list."""
    for module in _MODULES_WITH_ALL:
        names = list(module.__all__)
        duplicates = {name for name in names if names.count(name) > 1}
        assert not duplicates, f"{module.__name__}.__all__ has duplicate entries: {duplicates}"
