"""Adversarial source tests for a HOLD-only private calculus DEV commitment.

The constant test nonce below is PUBLIC and is NOT an evaluator secret.
"""

import hashlib
import json
from pathlib import Path
import re
import unittest

from trainingday import calc_eval_commitment as candidate
from trainingday import calc_curriculum
from trainingday import synthetic_arithmetic


PUBLIC_TEST_NONCE = bytes(range(1, 33))  # Fixture only. Never enroll it.
SECOND_TEST_NONCE = bytes(range(33, 65))
_PROMPT = re.compile(
    r"Evaluate P'\(x\) at x=(-?\d+) for the polynomial with ascending coefficients (\[[^\]]+\])\."
)


class CalcEvalCommitmentTests(unittest.TestCase):
    @staticmethod
    def _public_r201_prompts():
        prompts = []
        for index in range(calc_curriculum.TRAIN_COUNT):
            a, b, c, d, x = calc_curriculum._parameters(index)
            prompts.append(f"f(x)=({a})x^3+({b})x^2+({c})x+({d}); f'({x})=?")
        return prompts

    def _check_math(self, row):
        match = _PROMPT.fullmatch(row["prompt"])
        self.assertIsNotNone(match)
        x = int(match.group(1))
        coeffs = json.loads(match.group(2))
        self.assertIn(len(coeffs), (5, 6, 7))
        self.assertNotEqual(coeffs[-1], 0)
        self.assertGreaterEqual(abs(x), 11)
        self.assertLessEqual(abs(x), 53)
        # Oracle computed independently from the candidate's production sum:
        # finite difference of the exact formal polynomial at a symbolic point
        # would otherwise incur numerical rounding. Compute a derivative
        # coefficient vector and evaluate it with Horner's method instead.
        derived = [i * c for i, c in enumerate(coeffs)][1:]
        expected = 0
        for coefficient in reversed(derived):
            expected = expected * x + coefficient
        self.assertEqual(row["answer"], str(expected))

    def test_private_example_math_distribution_uniqueness(self):
        rows = candidate._private_dev_rows(PUBLIC_TEST_NONCE)
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({row["id"] for row in rows}), 64)
        self.assertEqual(len({row["prompt"] for row in rows}), 64)
        for row in rows:
            self.assertEqual(row["split"], "dev")
            self._check_math(row)

    def test_replay_stable_and_distinct_nonce(self):
        first = candidate.commitment(PUBLIC_TEST_NONCE)
        second = candidate.commitment(PUBLIC_TEST_NONCE)
        other = candidate.commitment(SECOND_TEST_NONCE)
        self.assertEqual(first, second)
        self.assertNotEqual(first["private_dev_sha256"], other["private_dev_sha256"])
        self.assertRegex(first["private_dev_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(first["private_dev_sha256"],
                         "19fbede7005fde5e98e6e1899a70cde8db5c32518312783258ee0c8a597d20d8")
        self.assertEqual(first["dev_count"], 64)

    def test_public_commitment_leaks_no_rows_nonce_or_answers(self):
        rows = candidate._private_dev_rows(PUBLIC_TEST_NONCE)
        result = candidate.commitment(PUBLIC_TEST_NONCE)
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn(PUBLIC_TEST_NONCE.hex(), encoded)
        self.assertNotIn("prompt", encoded)
        self.assertNotIn("answer", encoded)
        self.assertNotIn(rows[0]["id"], encoded)
        self.assertNotIn(rows[0]["prompt"], encoded)
        self.assertNotIn(rows[0]["answer"], encoded)
        self.assertNotIn("nonce", result)
        self.assertNotIn("rows", result)
        for name in ("source_rights_authenticated", "protected_evaluator_custody_verified",
                     "gpu_lease_granted", "authorizes_training", "authorizes_eval_unseal"):
            self.assertIs(result[name], False)

    def test_invalid_nonce_rejected(self):
        for nonce in (None, "a" * 32, bytearray(range(32)), bytes(32), b"short"):
            with self.subTest(nonce_type=type(nonce).__name__):
                with self.assertRaises(candidate.CalcEvalCommitmentError):
                    candidate.commitment(nonce)

    def test_surface_overlap_duplicate_and_malformed_public_train_rejected(self):
        row = candidate._private_dev_rows(PUBLIC_TEST_NONCE)[0]
        with self.assertRaises(candidate.CalcEvalCommitmentError):
            candidate.commitment(PUBLIC_TEST_NONCE, public_train_prompts=[row["prompt"]])
        with self.assertRaises(candidate.CalcEvalCommitmentError):
            candidate.commitment(PUBLIC_TEST_NONCE, public_train_prompts=["same", "same"])
        for malformed in ("string-not-sequence", ["", "valid"], [b"unsafe"], [True],
                          ["a" * 2_049], ["safe"] * 16_385):
            with self.subTest(malformed_type=type(malformed).__name__):
                with self.assertRaises(candidate.CalcEvalCommitmentError):
                    candidate.commitment(PUBLIC_TEST_NONCE, public_train_prompts=malformed)

    def test_existing_r199_train_has_no_prompt_overlap(self):
        public = [row["prompt"] for row in synthetic_arithmetic.train_rows()]
        result = candidate.commitment(PUBLIC_TEST_NONCE, public_train_prompts=public)
        self.assertIs(result["public_train_surface_overlap_checked"], True)
        self.assertIs(result["semantic_disjointness_independently_verified"], False)

    def test_format_alias_is_not_misreported_as_semantic_disjointness(self):
        prompt = candidate._private_dev_rows(PUBLIC_TEST_NONCE)[0]["prompt"]
        semantically_same = prompt.replace("ascending coefficients ",
                                           "ascending coefficients  ")
        result = candidate.commitment(PUBLIC_TEST_NONCE,
                                      public_train_prompts=[semantically_same])
        # Only an exact-surface-string comparison ran; a shared problem can
        # be rephrased. The issuer must not infer semantic independence.
        self.assertIs(result["public_train_surface_overlap_checked"], True)
        self.assertIs(result["semantic_disjointness_independently_verified"], False)

    def test_exact_calc_algebra_parity_if_installed(self):
        try:
            from calc.algebra import Polynomial
            import calc.algebra as module
        except ImportError:
            self.skipTest("optional exact Calc source checkout not installed")
        self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                         candidate.CALC_ALGEBRA_SHA256)
        for row in candidate._private_dev_rows(PUBLIC_TEST_NONCE):
            match = _PROMPT.fullmatch(row["prompt"])
            self.assertIsNotNone(match)
            x, coeffs = int(match.group(1)), json.loads(match.group(2))
            self.assertEqual(Polynomial(coeffs).differentiate()(x), int(row["answer"]))

    def test_matched_private_dev_is_bounded_correct_unique_and_separate(self):
        rows = candidate._private_matched_dev_rows(PUBLIC_TEST_NONCE)
        shifted = candidate._private_dev_rows(PUBLIC_TEST_NONCE)
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({row["id"] for row in rows}), 64)
        self.assertEqual(len({row["prompt"] for row in rows}), 64)
        self.assertTrue({row["prompt"] for row in rows}.isdisjoint(
            {row["prompt"] for row in shifted}))
        self.assertTrue({row["prompt"] for row in rows}.isdisjoint(
            set(self._public_r201_prompts())))
        pattern = re.compile(
            r"f\(x\)=\((-?\d+)\)x\^3\+\((-?\d+)\)x\^2\+\((-?\d+)\)x\+\((-?\d+)\); f'\((-?\d+)\)=\?"
        )
        for row in rows:
            self.assertEqual(row["split"], "dev")
            match = pattern.fullmatch(row["prompt"])
            self.assertIsNotNone(match)
            a, b, c, d, x = map(int, match.groups())
            self.assertTrue(1 <= abs(a) <= 17)
            self.assertTrue(-30 <= b <= 30)
            self.assertTrue(-50 <= c <= 50)
            self.assertTrue(-50 <= d <= 50)
            self.assertTrue(-5 <= x <= 5)
            self.assertEqual(int(row["answer"]), 3*a*x*x + 2*b*x + c)

    def test_matched_commitment_is_digest_only_deterministic_and_hard_hold(self):
        public = self._public_r201_prompts()
        first = candidate.matched_commitment(PUBLIC_TEST_NONCE,
                                             public_train_prompts=public)
        again = candidate.matched_commitment(PUBLIC_TEST_NONCE,
                                             public_train_prompts=public)
        other = candidate.matched_commitment(SECOND_TEST_NONCE,
                                             public_train_prompts=public)
        self.assertEqual(first, again)
        self.assertNotEqual(first["private_dev_sha256"], other["private_dev_sha256"])
        self.assertNotEqual(first["private_dev_sha256"],
                            candidate.commitment(PUBLIC_TEST_NONCE)["private_dev_sha256"])
        self.assertRegex(first["private_dev_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(first["dev_count"], 64)
        self.assertEqual(first["distribution"], "R201_CUBIC_SMALL_X_MATCHED")
        self.assertIs(first["public_train_surface_overlap_checked"], True)
        encoded = json.dumps(first, sort_keys=True)
        self.assertNotIn(PUBLIC_TEST_NONCE.hex(), encoded)
        self.assertNotIn("prompt", encoded)
        self.assertNotIn("answer", encoded)
        private = candidate._private_matched_dev_rows(PUBLIC_TEST_NONCE)[0]
        self.assertNotIn(private["id"], encoded)
        self.assertNotIn(private["prompt"], encoded)
        self.assertNotIn(private["answer"], encoded)
        for name in ("source_rights_authenticated", "protected_evaluator_custody_verified",
                     "gpu_lease_granted", "authorizes_training", "authorizes_eval_unseal",
                     "semantic_disjointness_independently_verified",
                     "private_nonce_entropy_independently_verified"):
            self.assertIs(first[name], False, name)

    def test_matched_commitment_rejects_partial_spoofed_and_overlapping_train(self):
        public = self._public_r201_prompts()
        leaked = candidate._private_matched_dev_rows(PUBLIC_TEST_NONCE)[0]["prompt"]
        poisoned = public.copy()
        poisoned[0] = leaked
        with self.assertRaises(candidate.CalcEvalCommitmentError):
            candidate.matched_commitment(PUBLIC_TEST_NONCE,
                                         public_train_prompts=poisoned)
        for malformed in (None, [], public[:100], [public[0]]*1024):
            with self.subTest(malformed_type=type(malformed).__name__):
                with self.assertRaises(candidate.CalcEvalCommitmentError):
                    candidate.matched_commitment(PUBLIC_TEST_NONCE,
                                                 public_train_prompts=malformed)
        with self.assertRaises(candidate.CalcEvalCommitmentError):
            candidate.matched_commitment(bytes(32), public_train_prompts=public)


if __name__ == "__main__":
    unittest.main()
