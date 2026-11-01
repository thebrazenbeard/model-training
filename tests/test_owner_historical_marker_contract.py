"""Guard the explicit boundary between portable tests and owner-held evidence."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "test_v10r3r10_vera_voice.py": {
        "test_generated_spec_targets_qwen_successor_no_external_spend",
    },
    "test_v10r3r11_vera_voice.py": {
        "test_generated_spec_binds_real_completed_parent",
        "test_disk_spec_exactly_binds_outputs",
    },
    "test_v10r3r11_pagefile_retry1.py": {
        "test_old_attempt_and_r10_parent_not_retired",
    },
    "test_r12_fp32_precision.py": {
        "test_r12_source_is_new_and_no_heldout_prompt_overlap",
    },
    "test_r12_source_portability.py": {
        "test_existing_r12_train_corpus_still_exact",
    },
    "test_r13_corpus_contract.py": {
        "test_new_128_targeted_rows_balance_identity_and_judgment",
        "test_training_window_alternates_and_never_changes_outer_rows",
        "test_exact_heldout_prompt_excluded",
        "test_frozen_source_rebuilds_bit_identical_spec_and_corpus",
    },
}


class OwnerHistoricalMarkerContractTests(unittest.TestCase):
    def test_only_exact_owner_artifact_tests_are_marked(self):
        """A marker is a classification, not a skip or an evidence substitute."""
        for filename, expected in EXPECTED.items():
            with self.subTest(filename=filename):
                tree = ast.parse((ROOT / "tests" / filename).read_text(encoding="utf8"))
                marked = {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    if any(
                        isinstance(decorator, ast.Attribute)
                        and isinstance(decorator.value, ast.Attribute)
                        and isinstance(decorator.value.value, ast.Name)
                        and decorator.value.value.id == "pytest"
                        and decorator.value.attr == "mark"
                        and decorator.attr == "owner_historical"
                        for decorator in node.decorator_list
                    )
                }
                self.assertEqual(marked, expected)

    def test_marker_is_registered_without_skipping_full_suite(self):
        config = (ROOT / "pytest.ini").read_text(encoding="utf8")
        self.assertIn("owner_historical:", config)
        self.assertNotIn("addopts", config)


if __name__ == "__main__":
    unittest.main()
