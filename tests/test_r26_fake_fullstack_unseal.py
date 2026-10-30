"""R26 public-only evaluator rehearsal. Never import/open the real private bank.

The eight-arm receipts, model, adapter, scores and private-bank callback are
synthetic fixtures. Test-only patches of the unconditional attester HOLD are
necessary for exercising later guards; they confer no runtime authority.
"""

from contextlib import ExitStack, contextmanager
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import tests.test_r18_eval as public_fixtures

from trainingday.admission import AdmissionError, _hash_file
from trainingday.eval_structural_r16 import CORPUS_SHA256
from trainingday.r18_eval import canonical_sha, evaluate


class R26SyntheticFullStackRehearsal(unittest.TestCase):
    """Run the actual R18 evaluator ordering with no real source/secret bytes."""

    def setUp(self):
        fixture = public_fixtures.R18EvaluatorTests(
            "test_complete_unscored_eight_arm_freeze_is_hash_bound")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.plan, self.inventory, self.inventory_path = fixture.make_freeze_inventory()
        self.args = fixture.change(
            preflight_only=False,
            freeze_inventory=self.inventory_path,
            expected_freeze_inventory_sha256=_hash_file(self.inventory_path),
        )

    @contextmanager
    def evaluator(self):
        """Patch only source prerequisites; leave REAL eight-arm checker active."""
        f = self.fixture
        with ExitStack() as stack:
            stack.enter_context(patch(
                "trainingday.r18_eval._required_base",
                return_value=(f.base, f.root),
            ))
            stack.enter_context(patch(
                "trainingday.r18_eval._hash_file",
                side_effect=lambda p: CORPUS_SHA256 if p == f.manifest else _hash_file(p),
            ))
            stack.enter_context(patch(
                "trainingday.r18_eval._source_manifest",
                return_value=({}, "f" * 64),
            ))
            stack.enter_context(patch(
                "trainingday.r18_eval._plan",
                return_value=(self.plan, canonical_sha(self.plan)),
            ))
            stack.enter_context(patch(
                "trainingday.r18_eval._adapter",
                return_value=(None, None, None),
            ))
            bank = stack.enter_context(patch(
                "trainingday.r18_eval._bank",
                side_effect=AssertionError("test-only bank must remain sealed"),
            ))
            yield bank

    def repin_inventory(self):
        """Simulate an adversary who is allowed to recompute public digests."""
        self.inventory_path.write_text(json.dumps(self.inventory), encoding="utf-8")
        self.args.expected_freeze_inventory_sha256 = _hash_file(self.inventory_path)

    @staticmethod
    def fake_blinded_scores():
        """Not a real scorer: no prompts, answers, tokens, or sealed labels."""
        return [{"blind_case_sha256": f"{i:064x}",
                 "exact": i % 3 == 0,
                 "canonical_format_exact": i % 6 == 0,
                 "prompt_tokens": 8 + i % 3,
                 "multiply_count": i % 4,
                 "peak_abs_intermediate": 100 + i,
                 "answer_digits": 2,
                 "nll_sum": 2.0,
                 "target_tokens": 2}
                for i in range(64)]

    def test_genuine_complete_fake_inventory_still_hits_independent_attester_hold(self):
        with self.evaluator() as bank, \
             patch("trainingday.r18_eval._cuda_preflight") as cuda, \
             patch("trainingday.r18_eval._load_model") as load:
            with self.assertRaisesRegex(AdmissionError, "independent pre-unseal"):
                evaluate(self.args)
            bank.assert_not_called()
            cuda.assert_not_called()
            load.assert_not_called()
        self.assertFalse(self.args.output.exists())
        self.assertFalse(self.args.private_output.exists())

    def test_missing_duplicate_and_wrong_plan_arms_reject_before_attester_or_bank(self):
        for name in ("missing_arm", "duplicate_arm", "plan_mismatch"):
            with self.subTest(name=name):
                saved = json.loads(json.dumps(self.inventory))
                if name == "missing_arm":
                    self.inventory["arms"].pop()
                elif name == "duplicate_arm":
                    self.inventory["arms"][1] = self.inventory["arms"][0].copy()
                else:
                    self.inventory["plan_sha256"] = "0" * 64
                self.repin_inventory()
                with self.evaluator() as bank, \
                     patch("trainingday.r18_eval._require_independent_preunseal_attestation") as attester, \
                     patch("trainingday.r18_eval._load_model") as load:
                    with self.assertRaisesRegex(AdmissionError, "pre-unseal"):
                        evaluate(self.args)
                    bank.assert_not_called()
                    attester.assert_not_called()
                    load.assert_not_called()
                self.inventory = saved
                self.repin_inventory()

    def test_forged_receipt_repin_and_saved_adapter_tamper_reject_before_bank(self):
        receipt_path = Path(self.inventory["arms"][0]["receipt_path"])
        original_receipt = receipt_path.read_bytes()
        original_receipt_sha = self.inventory["arms"][0]["receipt_sha256"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["optimizer_window_seconds"] = 1e6
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.inventory["arms"][0]["receipt_sha256"] = _hash_file(receipt_path)
        self.repin_inventory()
        with self.evaluator() as bank, \
             patch("trainingday.r18_eval._require_independent_preunseal_attestation") as attester:
            with self.assertRaisesRegex(AdmissionError, "independent contract"):
                evaluate(self.args)
            bank.assert_not_called()
            attester.assert_not_called()

        # Reset the first independent negative condition. Otherwise the
        # later adapter-corruption assertion could pass for the wrong reason.
        receipt_path.write_bytes(original_receipt)
        self.inventory["arms"][0]["receipt_sha256"] = original_receipt_sha
        self.repin_inventory()
        with self.evaluator() as bank:
            with self.assertRaisesRegex(AdmissionError, "independent pre-unseal"):
                evaluate(self.args)
            bank.assert_not_called()

        receipt_path = Path(next(x["receipt_path"] for x in self.inventory["arms"]
                                 if x["arm"] == "mezo" and x["control"] == "correct"))
        saved = receipt_path.parent / "adapter" / "adapter_model.safetensors"
        saved.write_bytes(b"synthetic attacker edits fake adapter bytes")
        with self.evaluator() as bank, \
             patch("trainingday.r18_eval._require_independent_preunseal_attestation") as attester:
            with self.assertRaisesRegex(AdmissionError, "adapter bytes"):
                evaluate(self.args)
            bank.assert_not_called()
            attester.assert_not_called()

    def test_fake_gpu_preflight_failure_denies_preload_and_bank(self):
        with self.evaluator() as bank, \
             patch("trainingday.r18_eval._require_independent_preunseal_attestation"), \
             patch("trainingday.r18_eval._cuda_preflight",
                   side_effect=RuntimeError("synthetic VRAM fit failure")), \
             patch("trainingday.r18_eval._load_model") as load:
            with self.assertRaisesRegex(RuntimeError, "synthetic VRAM fit failure"):
                evaluate(self.args)
            bank.assert_not_called()
            load.assert_not_called()

    def test_full_base_or_adapter_load_failure_never_touches_bank(self):
        for reason in ("synthetic base OOM", "synthetic partial adapter failure"):
            with self.subTest(reason=reason), \
                 self.evaluator() as bank, \
                 patch("trainingday.r18_eval._require_independent_preunseal_attestation"), \
                 patch("trainingday.r18_eval._cuda_preflight"), \
                 patch("trainingday.r18_eval._load_model",
                       side_effect=RuntimeError(reason)) as load:
                with self.assertRaisesRegex(RuntimeError, reason):
                    evaluate(self.args)
                load.assert_called_once()
                bank.assert_not_called()
                self.assertFalse(self.args.output.exists())
                self.assertFalse(self.args.private_output.exists())

    def test_fake_bank_open_then_scorer_exception_releases_same_session(self):
        events = []
        session = Mock()
        session.close.side_effect = lambda: events.append("close")

        def load(*_args):
            events.append("load")
            return session

        def fake_bank(*_args, on_unseal):
            events.append("fake-bank")
            on_unseal(123456789)
            return [{}] * 64

        def fail_score(_cases, received, _plan_sha):
            self.assertIs(received, session)
            events.append("score")
            raise RuntimeError("synthetic scorer failed")

        with self.evaluator(), \
             patch("trainingday.r18_eval._require_independent_preunseal_attestation"), \
             patch("trainingday.r18_eval._cuda_preflight"), \
             patch("trainingday.r18_eval._load_model", side_effect=load), \
             patch("trainingday.r18_eval._bank", side_effect=fake_bank), \
             patch("trainingday.r18_eval._run_model", side_effect=fail_score):
            with self.assertRaisesRegex(RuntimeError, "synthetic scorer failed"):
                evaluate(self.args)
        self.assertEqual(events, ["load", "fake-bank", "score", "close"])
        self.assertFalse(self.args.output.exists())
        self.assertFalse(self.args.private_output.exists())

    def test_fake_bank_without_open_timestamp_fails_before_scorer(self):
        session = Mock()
        with self.evaluator(), \
             patch("trainingday.r18_eval._require_independent_preunseal_attestation"), \
             patch("trainingday.r18_eval._cuda_preflight"), \
             patch("trainingday.r18_eval._load_model", return_value=session), \
             patch("trainingday.r18_eval._bank", return_value=[{}] * 64), \
             patch("trainingday.r18_eval._run_model") as scorer:
            with self.assertRaisesRegex(RuntimeError, "without an open timestamp"):
                evaluate(self.args)
            scorer.assert_not_called()
        session.close.assert_called_once_with()
        self.assertFalse(self.args.output.exists())
        self.assertFalse(self.args.private_output.exists())

    def test_complete_fake_lifecycle_outputs_aggregate_but_never_authorizes_promotion(self):
        events = []
        session = Mock()
        session.close.side_effect = lambda: events.append("close")

        def fake_bank(*_args, on_unseal):
            events.append("fake-bank")
            on_unseal(123456789)
            return [{}] * 64

        def fake_scores(_cases, received, _plan_sha):
            self.assertIs(received, session)
            events.append("fake-score")
            return self.fake_blinded_scores()

        with self.evaluator(), \
             patch("trainingday.r18_eval._require_independent_preunseal_attestation"), \
             patch("trainingday.r18_eval._cuda_preflight"), \
             patch("trainingday.r18_eval._load_model",
                   side_effect=lambda *_args: events.append("fake-load") or session), \
             patch("trainingday.r18_eval._bank", side_effect=fake_bank), \
             patch("trainingday.r18_eval._run_model", side_effect=fake_scores):
            report = evaluate(self.args)
        self.assertEqual(events, ["fake-load", "fake-bank", "fake-score", "close"])
        self.assertEqual(report["cases"], 64)
        self.assertEqual(report["greedy_exact_match"], 22)
        self.assertEqual(report["answer_eos_target_tokens"], 128)
        self.assertFalse(report["automatic_promotion"])
        self.assertFalse(report["fresh_adapter_fp32_reload_verified"])
        public = json.loads(self.args.output.read_text(encoding="utf-8"))
        private = json.loads(self.args.private_output.read_text(encoding="utf-8"))
        self.assertEqual(public["greedy_exact_match"], 22)
        self.assertEqual(len(private["outcomes"]), 64)
        self.assertNotIn("cases", private)
        self.assertNotIn("answers", private)
        self.assertNotIn("prompts", private)

    def test_fake_receipt_mutation_during_scoring_is_detected_before_publication(self):
        receipt_path = Path(self.inventory["arms"][0]["receipt_path"])
        session = Mock()

        def fake_bank(*_args, on_unseal):
            on_unseal(123456789)
            return [{}] * 64

        def malicious_fake_score(*_args):
            receipt_path.write_text('{"synthetic": "changed after freeze"}', encoding="utf-8")
            return self.fake_blinded_scores()

        with self.evaluator(), \
             patch("trainingday.r18_eval._require_independent_preunseal_attestation"), \
             patch("trainingday.r18_eval._cuda_preflight"), \
             patch("trainingday.r18_eval._load_model", return_value=session), \
             patch("trainingday.r18_eval._bank", side_effect=fake_bank), \
             patch("trainingday.r18_eval._run_model", side_effect=malicious_fake_score):
            with self.assertRaisesRegex(RuntimeError, "arm/adapter bytes changed"):
                evaluate(self.args)
        session.close.assert_called_once_with()
        self.assertFalse(self.args.output.exists())
        self.assertFalse(self.args.private_output.exists())


if __name__ == "__main__":
    unittest.main()
