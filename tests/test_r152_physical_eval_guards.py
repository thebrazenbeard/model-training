"""R152: direct CUDA loaders and private reads cannot bypass host authority."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest
from types import SimpleNamespace

from trainingday.physical_launch_authority import PhysicalLaunchAuthorityUnavailable

ROOT = Path(__file__).resolve().parents[1]
GUARD = "require_physical_launch_authority"


def _call_line(function: str, filename: str, called_name: str) -> int:
    tree = ast.parse((ROOT / "trainingday" / filename).read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == function]
    if len(functions) != 1:
        raise AssertionError(f"function {function} is missing or duplicated in {filename}")
    lines = [node.lineno for node in ast.walk(functions[0])
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == called_name]
    if not lines:
        raise AssertionError(f"{function} does not call {called_name}")
    return min(lines)


class PhysicalEvalGuards(unittest.TestCase):
    def test_r15_heldout_entry_denies_before_any_args_or_import(self):
        from trainingday.eval_heldout import evaluate
        with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
            evaluate(SimpleNamespace())

    def test_r18_direct_private_access_and_model_loading_deny(self):
        from trainingday import r18_eval
        for fn, arguments in (
            (r18_eval._cuda_preflight, ()),
            (r18_eval._bank, (None, None, None, None, None, None)),
            (r18_eval._paired, ([], None, "")),
            (r18_eval._load_model, (None, None, None)),
            (lambda: r18_eval._json(None, private=True), ()),
        ):
            with self.subTest(function=fn.__name__):
                with self.assertRaises(PhysicalLaunchAuthorityUnavailable):
                    fn(*arguments)

    def test_r16_structural_source_preflight_can_remain_unprivileged(self):
        filename = "eval_structural_r16.py"
        self.assertGreater(_call_line("evaluate", filename, GUARD), 0)
        source = (ROOT/"trainingday"/filename).read_text(encoding="utf-8")
        self.assertLess(source.index("if args.preflight_only:"),
                        source.index('operation="R16 structural evaluation"'))
        self.assertLess(source.index('operation="R16 structural evaluation"'),
                        source.index("    import torch"))

    def test_reload_guard_precedes_torch(self):
        source = (ROOT/"trainingday"/"reload_verify.py").read_text(encoding="utf-8")
        self.assertLess(source.index('operation="R15/R16 independent adapter reload"'),
                        source.index("    import torch"))

    def test_r18_eval_direct_reader_and_preunseal_guard_order(self):
        filename = "r18_eval.py"
        source = (ROOT/"trainingday"/filename).read_text(encoding="utf-8")
        self.assertIn('operation="R18 private evaluation"', source)
        self.assertIn('operation="R18 private bank read"', source)
        self.assertIn('operation="R18 private peer receipt read"', source)
        self.assertIn('operation="R18 private JSON read"', source)
        self.assertLess(source.index('operation="R18 private evaluation"'),
                        source.index("_preunseal_inventory(",
                                     source.index("def evaluate(args)")))
        for fn in ("_bank", "_paired", "_load_model", "_cuda_preflight"):
            self.assertTrue(_call_line(fn, filename, GUARD) > 0)


if __name__ == "__main__":
    unittest.main()
