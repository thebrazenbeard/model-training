"""Source-contract tests: candidate dialogue is not admitted training or heldout."""

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from trainingday import dialogue_curriculum as dialogue


class DialogueCandidateTests(unittest.TestCase):
    def test_balanced_distinct_fictional_train_dev(self):
        all_rows = dialogue.candidate_rows()
        train = dialogue.candidate_rows("train")
        dev = dialogue.candidate_rows("dev")
        self.assertEqual((len(all_rows), len(train), len(dev)), (80, 64, 16))
        self.assertEqual(train + dev, [r for r in all_rows if r["split"] == "train"]
                         + [r for r in all_rows if r["split"] == "dev"])
        self.assertTrue({r["id"] for r in train}.isdisjoint({r["id"] for r in dev}))
        self.assertTrue({r["scenario_id"] for r in train}.isdisjoint(
            {r["scenario_id"] for r in dev}))
        for subset, count in ((train, 8), (dev, 2)):
            self.assertEqual(Counter(r["family"] for r in subset),
                             Counter({family: count for family in dialogue.FAMILIES}))
        self.assertEqual(len({r["messages"][-2]["content"] for r in all_rows}), 80)
        self.assertEqual(sum(len(r["messages"]) == 4 for r in all_rows), 19)
        self.assertTrue(any(len(r["messages"]) == 4 and r["split"] == "dev"
                            for r in all_rows))

    def test_source_holds_are_immutable(self):
        for row in dialogue.candidate_rows():
            self.assertEqual(row["rights_status"], "NOT_ADMITTED")
            self.assertIs(row["training_authorized"], False)
            self.assertIs(row["heldout"], False)
            self.assertIn([m["role"] for m in row["messages"]],
                          [["user", "assistant"],
                           ["user", "assistant", "user", "assistant"]])
            self.assertEqual(row["source_kind"], "new_synthetic_fictional_candidate")
        for split in ("heldout", "eval", "test", "TRAIN", "protected", ""):
            with self.assertRaises(ValueError, msg=split):
                dialogue.candidate_rows(split)

    def test_manifest_binds_exact_canonical_bytes(self):
        manifest = dialogue.candidate_manifest()
        self.assertEqual(manifest["protected_heldout_count_generated"], 0)
        self.assertIs(manifest["independent_heldout_available"], False)
        self.assertIs(manifest["tokenizer_budget_verified"], False)
        self.assertIs(manifest["training_authorized"], False)
        for split in ("train", "dev"):
            payload = (json.dumps(dialogue.candidate_rows(split), sort_keys=True,
                                  ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            self.assertEqual(manifest[f"{split}_sha256"],
                             hashlib.sha256(payload).hexdigest())
        self.assertEqual(dialogue.candidate_manifest(), manifest)
        self.assertEqual(manifest["candidate_sha256"],
                         "4fbe54d97504628aa88ac7be37d994ed5f39be8d2a73db99f4cf54684989f31e")

    def test_candidate_mutation_fails_closed(self):
        tests = (
            ("training_authorized", True), ("heldout", True),
            ("rights_status", "ADMITTED"), ("split", "heldout"),
            ("source_kind", "private_conversation"),
        )
        for field, invalid in tests:
            rows = dialogue.candidate_rows()
            rows[0][field] = invalid
            with self.assertRaises(ValueError, msg=field):
                dialogue.validate(rows)

    def test_attestation_schema_requires_actual_signed_scope(self):
        path = (Path(__file__).resolve().parents[1] / "docs"
                / "VERA_DIALOGUE_ADMISSION_ATTESTATION_SCHEMA_V1.json")
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"],
                         "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        for required in ("candidate_sha256", "train_sha256", "dev_sha256",
                         "generator_commit", "model_id", "base_revision",
                         "rights_holder", "data_steward", "lawful_basis",
                         "human_review", "retention_and_removal", "processor_scope",
                         "excluded_material", "signing_contract",
                         "signing_payload_sha256", "signatures"):
            self.assertIn(required, schema["required"])
        self.assertEqual(schema["properties"]["model_id"]["const"],
                         "Qwen/Qwen3.5-4B")


if __name__ == "__main__":
    unittest.main()
