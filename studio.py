"""Run the project directly, without pip-installing it or changing PowerShell policy."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    from uznorm_studio.cli import main
    raise SystemExit(main(["--home", str(ROOT), *sys.argv[1:]]))
