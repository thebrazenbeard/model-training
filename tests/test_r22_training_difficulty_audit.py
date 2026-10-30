"""Public synthetic fixtures for R22 training-only difficulty measurements."""

import hashlib
import unittest

from trainingday.r18_benchmark import selected_id_digest
from trainingday.difficulty_audit import AuditError, audit_admitted_training


def row(start, ops):
    value = start
    for name, operand in ops:
        if name == "add":
            value += operand
        elif name == "subtract":
            value -= operand
        else:
            value *= operand
    prompt = (f"Start with {start}. "
              + " ".join(f"{name} {operand}." for name, operand in ops)
              + " What is the final integer?")
    answer = str(value)
    content = f"Question: {prompt}\nAnswer: {answer}"
    return {"id": "reasoning-" + hashlib.sha256(content.encode()).hexdigest(),
            "family": "verified_integer_chains", "prompt": prompt, "answer": answer}


class R22DifficultyAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row(2, [("add", 3), ("multiply", 2)]),
            row(5, [("subtract", 2), ("add", 7)]),
            row(1, [("add", 1), ("multiply", 2)]),
            row(3, [("multiply", 2), ("subtract", 1)]),
        ]
        self.lengths = {r["id"]: n for r, n in zip(self.rows, (1, 2, 1, 3))}

    def audit(self, rows=None, lengths=None, **kw):
        items = self.rows if rows is None else rows
        return audit_admitted_training(
            items, selected_id_sha256=selected_id_digest(self.rows),
            source_manifest_sha256="a" * 64,
            tokenizer_binding_sha256="b" * 64,
            token_lengths=self.lengths if lengths is None else lengths,
            expected_count=len(items), **kw)

    def test_aggregates_only_and_hash_binding(self):
        result = self.audit()
        self.assertEqual(result["sample_size"], 4)
        self.assertEqual(result["operation_patterns"],
                         {"add>multiply": 2, "subtract>add": 1,
                          "multiply>subtract": 1})
        self.assertEqual(result["operation_count_distribution"], {"2": 4})
        self.assertEqual(result["target_token_length_distribution"],
                         {"1": 2, "2": 1, "3": 1})
        self.assertEqual(result["structural_duplicate_count"], 1)
        self.assertFalse(result["neural_improvement_proven"])
        self.assertFalse(result["tokenizer_physically_verified"])
        self.assertFalse(result["qualified_for_neural_training"])
        self.assertEqual(len(result["aggregate_sha256"]), 64)
        self.assertNotIn("prompt", str(result))
        self.assertNotIn("answer", result)

    def test_duplicate_prompt_and_eval_flag_rejected(self):
        with self.assertRaises(AuditError):
            self.audit(rows=self.rows + [self.rows[0]])
        contaminated = [dict(r) for r in self.rows]
        contaminated[0]["evaluation_holdout"] = True
        with self.assertRaises(AuditError):
            self.audit(rows=contaminated)

    def test_tampered_or_noncanonical_oracle_rejected(self):
        for mutation in (
                lambda r: r.update(answer="99"),
                lambda r: r.update(id="reasoning-" + "0" * 64),
                lambda r: r.update(prompt=r["prompt"].replace("add", "divide")),
        ):
            mutated = [dict(r) for r in self.rows]
            mutation(mutated[0])
            with self.subTest(mutation=mutation):
                with self.assertRaises(AuditError):
                    self.audit(rows=mutated)

    def test_token_length_coverage_and_type_rejected(self):
        missing = dict(self.lengths)
        missing.pop(self.rows[0]["id"])
        with self.assertRaises(AuditError):
            self.audit(lengths=missing)
        bad = dict(self.lengths)
        bad[self.rows[0]["id"]] = True
        with self.assertRaises(AuditError):
            self.audit(lengths=bad)

    def test_ordered_selection_and_manifest_hash_bound(self):
        with self.assertRaises(AuditError):
            audit_admitted_training(
                self.rows, selected_id_sha256=selected_id_digest(self.rows[::-1]),
                source_manifest_sha256="a" * 64,
                tokenizer_binding_sha256="b" * 64,
                token_lengths=self.lengths, expected_count=4)
        with self.assertRaises(AuditError):
            self.audit(source_admission_status="UNKNOWN")


if __name__ == "__main__":
    unittest.main()
