"""Offline/optional-real-Calc checks for the new R201 curriculum candidate."""

from __future__ import annotations

import hashlib
from importlib import _bootstrap_external
import importlib.util
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from trainingday import calc_curriculum as curriculum


class CalcCurriculumSourceTests(unittest.TestCase):
    def test_verified_source_cannot_execute_forged_timestamp_bytecode(self):
        # A .pyc may be timestamp-valid while containing different executable
        # code from the SHA-pinned Python source. Test both import targets.
        for target in ("__init__.py", "symbolic.py"):
            with self.subTest(target=target), TemporaryDirectory() as temp:
                package = Path(temp) / "calc"
                package.mkdir()
                init = package / "__init__.py"
                symbolic = package / "symbolic.py"
                init_source = "__version__ = '0.1.0'\n"
                sym_source = "Const = object\nVar = object\n"
                init.write_text(init_source)
                symbolic.write_text(sym_source)
                marker = Path(temp) / "PYC_EXECUTED"
                forged = package / target
                forged_code = (
                    f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
                    + (init_source if target == "__init__.py" else sym_source)
                )
                metadata = forged.stat()
                pyc = _bootstrap_external._code_to_timestamp_pyc(
                    compile(forged_code, str(forged), "exec"),
                    int(metadata.st_mtime), metadata.st_size,
                )
                pyc_path = Path(importlib.util.cache_from_source(str(forged)))
                pyc_path.parent.mkdir(parents=True, exist_ok=True)
                pyc_path.write_bytes(pyc)
                saved = {name: sys.modules.pop(name) for name in
                         ("calc.symbolic", "calc") if name in sys.modules}
                try:
                    with patch.object(sys, "path", [temp] + sys.path), \
                            patch.object(curriculum, "CALC_INIT_SHA256",
                                         hashlib.sha256(init_source.encode()).hexdigest()), \
                            patch.object(curriculum, "CALC_SYMBOLIC_SHA256",
                                         hashlib.sha256(sym_source.encode()).hexdigest()):
                        loaded = curriculum._load_verified_symbolic()
                        self.assertIsNotNone(loaded.Const)
                finally:
                    for name in ("calc.symbolic", "calc"):
                        sys.modules.pop(name, None)
                    sys.modules.update(saved)
                self.assertFalse(marker.exists(), "source-hash-valid but forged bytecode executed")

    def test_untrusted_calc_source_never_executes_before_hash_check(self):
        # A stale/malicious installed package must be rejected *before* its
        # __init__ or symbolic.py top-level code has any chance to execute.
        with TemporaryDirectory() as temp:
            package = Path(temp) / "calc"
            package.mkdir()
            marker = Path(temp) / "EXECUTED"
            payload = f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
            (package / "__init__.py").write_text(payload)
            (package / "symbolic.py").write_text(payload)
            saved = {name: sys.modules.pop(name) for name in
                     ("calc.symbolic", "calc") if name in sys.modules}
            try:
                with patch.object(sys, "path", [temp] + sys.path):
                    with self.assertRaises(curriculum.CalcCurriculumError):
                        curriculum._load_verified_symbolic()
            finally:
                for name in ("calc.symbolic", "calc"):
                    sys.modules.pop(name, None)
                sys.modules.update(saved)
            self.assertFalse(marker.exists(), "unverified Calc code was executed")

    def test_untrusted_calc_initializer_never_executes(self):
        with TemporaryDirectory() as temp:
            package = Path(temp) / "calc"
            package.mkdir()
            marker = Path(temp) / "PACKAGE_EXECUTED"
            initializer = f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
            (package / "__init__.py").write_text(initializer)
            source = "Const = object\nVar = object\n"
            (package / "symbolic.py").write_text(source)
            saved = {name: sys.modules.pop(name) for name in
                     ("calc.symbolic", "calc") if name in sys.modules}
            try:
                with patch.object(sys, "path", [temp] + sys.path), \
                        patch.object(curriculum, "CALC_SYMBOLIC_SHA256",
                                     hashlib.sha256(source.encode()).hexdigest()):
                    with self.assertRaises(curriculum.CalcCurriculumError):
                        curriculum._load_verified_symbolic()
            finally:
                for name in ("calc.symbolic", "calc"):
                    sys.modules.pop(name, None)
                sys.modules.update(saved)
            self.assertFalse(marker.exists(), "unverified Calc initializer was executed")

    def test_freeze_and_no_hidden_dev_or_grant(self):
        self.assertEqual(curriculum.TRAIN_COUNT, 1024)
        self.assertEqual(curriculum.CALC_REVISION,
                         "17e12d0f659b42732fb98e5730732aba9ff1b488")
        self.assertEqual(curriculum.CALC_SYMBOLIC_SHA256,
                         "aba705f0d4370afa35538280f3db00ecbe36da1b3f3ca083ee29ebc292113b0b")
        self.assertEqual(curriculum._parameters(0), (-6, -23, 24, -32, 3))
        self.assertEqual(curriculum._parameters(1023)[-1], 1)

    def test_invalid_train_ids_rejected(self):
        for value in (-1, 1024, True, 0.0, "0", None):
            with self.subTest(value=value):
                with self.assertRaises(curriculum.CalcCurriculumError):
                    curriculum._parameters(value)

    def test_missing_calc_fails_closed(self):
        with patch.object(curriculum, "_preflight_calc_files",
                          side_effect=curriculum.CalcCurriculumError(
                              "Calc symbolic source unavailable")):
            with self.assertRaisesRegex(curriculum.CalcCurriculumError,
                                        "Calc symbolic source unavailable"):
                curriculum.train_rows()

    def test_untrusted_module_filename_fails_closed(self):
        with TemporaryDirectory() as temp:
            init = Path(temp) / "__init__.py"
            wrong = Path(temp) / "impostor.py"
            init.write_text("__version__ = '0.1.0'\n")
            source = "Const = object\nVar = object\n"
            wrong.write_text(source)
            with patch.object(curriculum, "_preflight_calc_files",
                              return_value=(init, wrong)), \
                    patch.object(curriculum, "CALC_SYMBOLIC_SHA256",
                                 hashlib.sha256(source.encode()).hexdigest()):
                with self.assertRaisesRegex(curriculum.CalcCurriculumError,
                                            "filename"):
                    curriculum._load_verified_symbolic()

    def test_spoofed_module_file_and_symbols_rejected(self):
        # A cached forged module cannot impersonate the checked source: the
        # verifier must execute checked bytes into a fresh private module.
        with TemporaryDirectory() as temp:
            init = Path(temp) / "__init__.py"
            source_file = Path(temp) / "symbolic.py"
            init_source = "__version__ = '0.1.0'\n"
            source = "Const = 'trusted'\nVar = 'trusted'\n"
            init.write_text(init_source)
            source_file.write_text(source)
            fake = ModuleType("trainingday._verified_calc_symbolic")
            fake.Const = "forged"
            fake.Var = "forged"
            with patch.dict(sys.modules, {fake.__name__: fake}), \
                    patch.object(curriculum, "_preflight_calc_files",
                                 return_value=(init, source_file)), \
                    patch.object(curriculum, "CALC_INIT_SHA256",
                                 hashlib.sha256(init_source.encode()).hexdigest()), \
                    patch.object(curriculum, "CALC_SYMBOLIC_SHA256",
                                 hashlib.sha256(source.encode()).hexdigest()):
                verified = curriculum._load_verified_symbolic()
                self.assertIsNot(verified, fake)
                self.assertEqual(verified.Const, "trusted")

    @unittest.skipUnless(importlib.util.find_spec("calc") is not None,
                         "Calc source not installed in this test environment")
    def test_real_calc_oracle_and_independent_answer_checks(self):
        rows = curriculum.train_rows()
        self.assertEqual(len(rows), 1024)
        self.assertEqual(len({row["prompt"] for row in rows}), 1024)
        self.assertEqual(len({row["id"] for row in rows}), 1024)
        pattern = re.compile(
            r"f\(x\)=\((-?\d+)\)x\^3\+\((-?\d+)\)x\^2"
            r"\+\((-?\d+)\)x\+\((-?\d+)\); f'\((-?\d+)\)=\?"
        )
        for index, row in enumerate(rows):
            with self.subTest(index=index):
                self.assertEqual(set(row), {"id", "prompt", "answer", "split"})
                self.assertEqual(row["id"], f"r201-calc-train-{index:04d}")
                self.assertEqual(row["split"], "train")
                self.assertLessEqual(len("Q: " + row["prompt"] + "\nA: " + row["answer"]), 64)
                match = pattern.fullmatch(row["prompt"])
                self.assertIsNotNone(match)
                a, b, c, _d, x0 = map(int, match.groups())
                self.assertEqual(row["answer"], str(3*a*x0*x0 + 2*b*x0 + c))
        manifest = curriculum.source_commitment()
        self.assertEqual(manifest["train_sha256"],
                         "fd7d0fb78f2ecf8a20c17dd80e0073572512a71f1f6f9e51623a8a02a314050b")
        self.assertEqual(manifest["train_count"], 1024)
        self.assertFalse(manifest["r185_corpus_replacement"])
        self.assertFalse(manifest["private_dev_included"])
        self.assertIsNone(manifest["private_dev_commitment"])
        for key in ("rights_holder_authenticated", "data_rights_granted",
                    "protected_evaluator_custody_verified", "gpu_lease_granted",
                    "authorizes_training"):
            self.assertIs(manifest[key], False)
        self.assertIsInstance(json.dumps(manifest, sort_keys=True), str)


if __name__ == "__main__":
    unittest.main()
