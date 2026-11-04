"""Adversarial checks for pinned Calc loader's private module lifecycle."""

from __future__ import annotations

import builtins
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from trainingday import calc_curriculum as curriculum


class CalcPrivateModuleTransactionTests(unittest.TestCase):
    def _case(self, folder: str):
        package = Path(folder) / "calc"
        package.mkdir()
        init = package / "__init__.py"
        source = package / "symbolic.py"
        init.write_text("__version__ = '0.1.0'\n", encoding="utf-8")
        source.write_text("Const = object\nVar = object\n", encoding="utf-8")
        return init, source

    def _verification(self, init: Path, source: Path):
        return (
            patch.object(curriculum, "_preflight_calc_files", return_value=(init, source)),
            patch.object(curriculum, "CALC_INIT_SHA256", hashlib.sha256(init.read_bytes()).hexdigest()),
            patch.object(curriculum, "CALC_SYMBOLIC_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()),
        )

    def test_hash_mismatch_after_exec_restores_private_module_identity(self):
        with TemporaryDirectory() as folder:
            init, source = self._case(folder)
            name = "trainingday._verified_calc_symbolic"
            previous = sys.modules.get(name)
            sentinel = ModuleType(name)
            sys.modules[name] = sentinel
            def swapped_after_exec(code, globals_):
                builtins.exec(code, globals_)
                source.write_text("Const = object\nVar = object\n# changed\n", encoding="utf-8")
            preflight, init_hash, source_hash = self._verification(init, source)
            try:
                with preflight, init_hash, source_hash, patch.object(
                    curriculum, "exec", side_effect=swapped_after_exec, create=True
                ):
                    with self.assertRaisesRegex(curriculum.CalcCurriculumError, "changed during execution"):
                        curriculum._load_verified_symbolic()
                self.assertIs(sys.modules.get(name), sentinel)
            finally:
                sys.modules.pop(name, None)
                if previous is not None:
                    sys.modules[name] = previous

    def test_interrupt_during_exec_restores_private_module_identity(self):
        with TemporaryDirectory() as folder:
            init, source = self._case(folder)
            name = "trainingday._verified_calc_symbolic"
            previous = sys.modules.get(name)
            sentinel = ModuleType(name)
            sys.modules[name] = sentinel
            preflight, init_hash, source_hash = self._verification(init, source)
            try:
                with preflight, init_hash, source_hash, patch.object(
                    curriculum, "exec", side_effect=KeyboardInterrupt, create=True
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        curriculum._load_verified_symbolic()
                self.assertIs(sys.modules.get(name), sentinel)
            finally:
                sys.modules.pop(name, None)
                if previous is not None:
                    sys.modules[name] = previous


if __name__ == "__main__":
    unittest.main()
