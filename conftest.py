"""Root conftest — enables `from ..barrier.conftest` imports in test files.

Test files in python/tests/barrier/ use both:
  - `from .conftest import ...`  (single-dot, same package)
  - `from ..barrier.conftest import ...`  (double-dot, parent package)

The double-dot form requires python/tests/ to be a Python package (so
`..` can resolve to its parent).  Since python/tests/ has no __init__.py
and the directory is read-only, we register it as a namespace package in
sys.modules during pytest startup.
"""
import importlib
import importlib.util
import os
import sys
import types


def pytest_configure(config):
    """Register python/tests as a namespace package in sys.modules."""
    project_root = os.path.dirname(__file__)
    tests_dir = os.path.join(project_root, "python", "tests")
    barrier_dir = os.path.join(tests_dir, "barrier")

    # Add project root to sys.path so `from scripts.xxx import ...` works
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 1. Register 'tests' as a namespace package
    if "tests" not in sys.modules:
        pkg = types.ModuleType("tests")
        pkg.__path__ = [tests_dir]
        pkg.__package__ = "tests"
        sys.modules["tests"] = pkg

    # 2. When pytest discovers barrier test modules, they'll be imported as
    #    `barrier.test_foo`.  We also need them accessible as `tests.barrier.test_foo`
    #    so that relative imports (`..barrier.conftest`) resolve correctly.
    #    We'll fix this in pytest_collectstart by re-parenting collected modules.


def pytest_collect_file(parent, file_path):
    """Re-parent barrier test modules under tests.barrier namespace."""
    pass  # Handled via itemcollected hook


def pytest_itemcollected(item):
    """Fix __package__ for barrier test modules that use `from ..barrier`."""
    mod = item.module
    if mod is None:
        return

    # Only fix modules whose __package__ is 'barrier' (should be 'tests.barrier')
    if getattr(mod, "__package__", None) == "barrier":
        mod.__package__ = "tests.barrier"

        # Also register under tests.barrier namespace in sys.modules
        old_name = mod.__name__  # e.g., 'barrier.test_training_diagnostics'
        new_name = "tests." + old_name  # e.g., 'tests.barrier.test_training_diagnostics'
        sys.modules[new_name] = mod

        # Ensure tests.barrier exists
        if "tests.barrier" not in sys.modules:
            barrier_init = os.path.join(
                os.path.dirname(__file__), "python", "tests", "barrier", "__init__.py"
            )
            if os.path.exists(barrier_init):
                spec = importlib.util.spec_from_file_location(
                    "tests.barrier", barrier_init,
                    submodule_search_locations=[
                        os.path.join(os.path.dirname(__file__), "python", "tests", "barrier")
                    ],
                )
                if spec:
                    barrier_mod = importlib.util.module_from_spec(spec)
                    barrier_mod.__package__ = "tests.barrier"
                    sys.modules["tests.barrier"] = barrier_mod
                    spec.loader.exec_module(barrier_mod)

        # Register barrier.conftest under tests.barrier.conftest
        if "tests.barrier.conftest" not in sys.modules and "barrier.conftest" in sys.modules:
            sys.modules["tests.barrier.conftest"] = sys.modules["barrier.conftest"]
