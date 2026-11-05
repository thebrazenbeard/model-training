"""Tiny deterministic source-only sparse weight candidate regression tests."""

import copy
import hashlib
import math
import struct
import unittest

from trainingday.no_gradient_weight_edit import (
    FALSE_CLAIMS,
    HOLD,
    LANE,
    SCHEMA,
    WeightEditCandidateError,
    reverse_candidate,
    source_only_receipt,
    validate_candidate,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


BASE = struct.pack("<fff", 1.0, 2.0, 3.0)
AFTER = struct.pack("<fff", 1.0, 4.0, 3.0)


def candidate(arm="treatment"):
    active = arm in ("treatment", "random")
    return {
        "schema": SCHEMA,
        "lane": LANE,
        "admission": HOLD,
        "base": {
            "kind": "synthetic_fixture",
            "repository": "synthetic/test-only",
            "revision": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "config_sha256": "d" * 64,
        },
        "tensor": {
            "name": "synthetic.layer.15.mlp.out",
            "shape": [3],
            "dtype": "float32_le",
            "before_sha256": sha(BASE),
            "after_sha256": sha(AFTER if active else BASE),
        },
        "edits": ([{"flat_index": 1, "before_hex": BASE[4:8].hex(), "after_hex": AFTER[4:8].hex()}] if active else []),
        "control": {"arm": arm, "seed": 81 if arm == "random" else None},
        "feedback": {"practice_split_id": "practice-synthetic", "heldout_split_id": "heldout-synthetic", "heldout_data_in_edit": False},
        "claims": dict(FALSE_CLAIMS),
    }


MUTANTS = [
    lambda m: m.update(admission="ADMITTED"),
    lambda m: m.update(lane="OFFICIAL_QWEN_LORA"),
    lambda m: m["claims"].update(qwen_weight_edit_verified=True),
    lambda m: m["base"].update(kind="official_qwen", repository="Qwen/Qwen3.5-4B"),
    lambda m: m["base"].update(checkpoint_sha256="not-a-hash"),
    lambda m: m["tensor"].update(dtype="bfloat16"),
    lambda m: m["tensor"].update(shape=[4]),
    lambda m: m["tensor"].update(name="synthetic.layer.20.mlp.out"),
    lambda m: m["tensor"].update(before_sha256="0" * 64),
    lambda m: m["tensor"].update(after_sha256="0" * 64),
    lambda m: m["feedback"].update(heldout_data_in_edit=True),
    lambda m: m["feedback"].update(heldout_split_id="practice-synthetic"),
    lambda m: m["control"].update(seed=5),
    lambda m: m["edits"][0].update(flat_index=3),
    lambda m: m["edits"][0].update(before_hex="00000000"),
    lambda m: m["edits"][0].update(after_hex="0000807f"),  # +infinity
    lambda m: m["edits"][0].update(after_hex=m["edits"][0]["before_hex"]),
    lambda m: m["edits"].append(copy.deepcopy(m["edits"][0])),  # duplicate index
    lambda m: m["edits"].extend(copy.deepcopy(m["edits"][0]) for _ in range(64)),
    lambda m: m["tensor"].update(shape=[True, 3]),
    lambda m: m.update(secret_heldout_gold="leaked"),
    lambda m: m["claims"].update(persona_verified=0),
]


class WeightEditCandidateTests(unittest.TestCase):
    def test_exact_tensor_forward_reverse_and_hard_hold(self):
        for arm in ("treatment", "random", "sham", "zero"):
            with self.subTest(arm=arm):
                m = candidate(arm)
                patched = validate_candidate(m, BASE)
                self.assertEqual(patched, AFTER if arm in ("treatment", "random") else BASE)
                self.assertEqual(reverse_candidate(m, patched), BASE)
                receipt = source_only_receipt(m, BASE)
                self.assertIs(receipt["synthetic_tensor_byte_roundtrip_verified"], True)
                self.assertEqual(receipt["admission"], HOLD)
                self.assertTrue(all(receipt[key] is False for key in FALSE_CLAIMS))
                self.assertEqual(BASE, struct.pack("<fff", 1.0, 2.0, 3.0))

    def test_fail_closed_mutants(self):
        for index, mutate in enumerate(MUTANTS):
            with self.subTest(mutant=index):
                m = candidate()
                mutate(m)
                with self.assertRaises(WeightEditCandidateError):
                    validate_candidate(m, BASE)

    def test_wrong_base_and_corrupt_reverse_rejected(self):
        m = candidate()
        for bad_base in (struct.pack("<fff", 9, 2, 3), struct.pack("<ff", 1, 2), struct.pack("<fff", math.nan, 2, 3)):
            with self.subTest(bad_base=bad_base.hex()):
                with self.assertRaises(WeightEditCandidateError):
                    validate_candidate(m, bad_base)
        with self.assertRaises(WeightEditCandidateError):
            reverse_candidate(m, BASE)

    def test_control_negative_cases_rejected(self):
        random = candidate("random")
        random["control"]["seed"] = None
        with self.assertRaises(WeightEditCandidateError):
            validate_candidate(random, BASE)
        zero = candidate("zero")
        zero["edits"] = candidate()["edits"]
        with self.assertRaises(WeightEditCandidateError):
            validate_candidate(zero, BASE)

    def test_no_dependence_on_python_assertions(self):
        # python -O removes assert, but critical guards are explicit exceptions.
        m = candidate()
        m["admission"] = "ADMITTED"
        with self.assertRaises(WeightEditCandidateError):
            validate_candidate(m, BASE)


if __name__ == "__main__":
    unittest.main()
