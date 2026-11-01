import copy
import hashlib
import json
import unittest

from trainingday.private_dev_commitment_v1 import (
    CustodyInputError,
    build_source_only_private_commitment,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def index_entry(case_id, prompt):
    normalized = " ".join(prompt.split()).casefold()
    return {"id": case_id, "prompt_sha256": hashlib.sha256(normalized.encode()).hexdigest()}


class PrivateDevCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.train_index = [index_entry(f"train-{n:04d}", f"Start with {n}. add 1.")
                            for n in range(1024)]
        ids = [item["id"] for item in self.train_index]
        selected_v2 = hashlib.sha256(
            b"VERA_TRAININGDAY_SELECTED_ID_DIGEST_V2\x00"
            + json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        self.subject = {
            "corpus_id": "R18_ARITHMETIC",
            "manifest_sha256": "a" * 64,
            "train_sha256": "b" * 64,
            "selected_id_v2_sha256": selected_v2,
            "generator_source_sha256": "d" * 64,
        }
        self.dev_rows = [
            {"id": "dev-1", "prompt": "Start with 7. add 2.", "answer": "9", "split": "dev"},
            {"id": "dev-2", "prompt": "Start with 8. multiply 2.", "answer": "16", "split": "dev"},
        ]
        self.excluded = [
            {"id": f"r18-reasoning-private-{n}", "prompt_sha256": hashlib.sha256(f"r18-{n}".encode()).hexdigest(), "bank": "R18_EXTERNAL_OOD"}
            for n in range(64)
        ] + [
            {"id": f"r16-private-{n}", "prompt_sha256": hashlib.sha256(f"r16-{n}".encode()).hexdigest(), "bank": "R16_PROTECTED"}
            for n in range(64)
        ]

    def make(self, **overrides):
        args = {"subject": self.subject, "train_index": self.train_index,
                "dev_jsonl": b"\n".join(canonical(row) for row in self.dev_rows) + b"\n",
                "protected_exclusions_json": canonical(self.excluded)}
        args.update(overrides)
        return build_source_only_private_commitment(**args)

    def test_only_aggregate_hashes_and_non_grant_status(self):
        result = self.make()
        self.assertEqual(result["status"], "HASHES_ONLY_UNAUTHENTICATED_HOLD")
        self.assertEqual(result["dev_count"], 2)
        self.assertEqual(result["protected_exclusion_count"], 128)
        self.assertEqual(result["dev_sha256"], hashlib.sha256(
            b"\n".join(canonical(row) for row in self.dev_rows) + b"\n").hexdigest())
        self.assertFalse(result["independent_custody_authenticated"])
        self.assertFalse(result["authorizes_training"])
        output = json.dumps(result)
        for marker in ("Start with", "dev-1", "train-0001", "r16-private", '"answer"'):
            self.assertNotIn(marker, output)

    def test_ordered_byte_exact_commitment(self):
        original = self.make()
        modified = copy.deepcopy(self.dev_rows)
        modified.reverse()
        other = self.make(dev_jsonl=b"\n".join(canonical(row) for row in modified) + b"\n")
        self.assertNotEqual(original["dev_sha256"], other["dev_sha256"])

    def test_reject_train_dev_identity_or_prompt_overlap(self):
        copied = copy.deepcopy(self.dev_rows)
        copied[0]["id"] = "train-0001"
        with self.assertRaises(CustodyInputError):
            self.make(dev_jsonl=b"\n".join(canonical(row) for row in copied) + b"\n")
        copied[0]["id"] = "dev-1"
        copied[0]["prompt"] = "START with   1. ADD 1."
        with self.assertRaises(CustodyInputError):
            self.make(dev_jsonl=b"\n".join(canonical(row) for row in copied) + b"\n")

    def test_reject_dev_protected_overlap(self):
        excluded = copy.deepcopy(self.excluded)
        excluded[0]["id"] = "dev-1"
        with self.assertRaises(CustodyInputError):
            self.make(protected_exclusions_json=canonical(excluded))

    def test_reject_malformed_or_implicit_provenance(self):
        for raw in (b"not json\n", b"{}\n", b"{}", b"\xef\xbb\xbf{}\n"):
            with self.subTest(raw=raw), self.assertRaises(CustodyInputError):
                self.make(dev_jsonl=raw)
        extra = copy.deepcopy(self.dev_rows)
        extra[0]["training_use"] = "ADMITTED"
        with self.assertRaises(CustodyInputError):
            self.make(dev_jsonl=b"\n".join(canonical(row) for row in extra) + b"\n")
        subject = dict(self.subject)
        subject["corpus_id"] = "LABYRINTH_COGNITIVE"
        with self.assertRaises(CustodyInputError):
            self.make(subject=subject)

    def test_reject_non_builtin_source_values(self):
        class Trick(str):
            pass
        subject = dict(self.subject)
        subject["manifest_sha256"] = Trick("a" * 64)
        with self.assertRaises(CustodyInputError):
            self.make(subject=subject)
        with self.assertRaises(CustodyInputError):
            self.make(dev_jsonl=bytearray(b"{}\n"))


if __name__ == "__main__":
    unittest.main()
