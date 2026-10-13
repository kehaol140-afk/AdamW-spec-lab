"""Run old CPU tests with their output ROOT redirected to a disposable child directory."""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from .core import ROOT


def main():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        shadow = Path(directory)
        (shadow / "config").mkdir()
        (shadow / "work").mkdir()
        shutil.copy2(ROOT / "config" / "local_v1.json", shadow / "config" / "local_v1.json")
        shutil.copy2(ROOT / "bindings.local.json", shadow / "bindings.local.json")
        name = "isolated_parent_numerics"
        spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / "test_numerics.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.ROOT = shadow
        numerics = unittest.defaultTestLoader.loadTestsFromModule(module)
        resource_name = "isolated_parent_resources"
        resource_spec = importlib.util.spec_from_file_location(resource_name, ROOT / "tests" / "test_resources.py")
        resources_module = importlib.util.module_from_spec(resource_spec)
        sys.modules[resource_name] = resources_module
        resource_spec.loader.exec_module(resources_module)
        resources = unittest.defaultTestLoader.loadTestsFromModule(resources_module)
        suite = unittest.TestSuite([numerics, resources])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
