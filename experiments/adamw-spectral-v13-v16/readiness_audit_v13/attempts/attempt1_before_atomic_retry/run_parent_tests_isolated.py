"""Run old CPU tests with every possible output redirected to V1.3 temporary files."""
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from .common import HERE, ROOT


def main():
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        shadow = Path(directory)
        (shadow / "config").mkdir()
        (shadow / "work").mkdir()
        shutil.copy2(ROOT / "config" / "local_v1.json", shadow / "config" / "local_v1.json")
        shutil.copy2(ROOT / "bindings.local.json", shadow / "bindings.local.json")
        suite = unittest.TestSuite()
        for name, file in (("isolated_parent_numerics", "test_numerics.py"),
                           ("isolated_parent_resources", "test_resources.py")):
            spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            module.ROOT = shadow
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
