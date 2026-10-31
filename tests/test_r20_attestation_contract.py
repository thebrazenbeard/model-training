"""R20 adversarial source-only tests for R18 pre-unseal trust boundaries.

No private bank, GPU, training process, or signing material is accessed.
"""

import unittest
from unittest.mock import Mock, patch

import test_r18_eval as r18_fixture

from trainingday.admission import AdmissionError, _hash_file
from trainingday.eval_structural_r16 import CORPUS_SHA256
from trainingday.r18_eval import _preunseal_inventory, evaluate


class IndependentAttestationContractTests(unittest.TestCase):
    def setUp(self):
        # Exercise the actual eight-arm hash/receipt verifier, not a mock of
        # the behavior under review. Fixture receipts are fabricated locally.
        self.fixture = r18_fixture.R18EvaluatorTests(
            "test_complete_unscored_eight_arm_freeze_is_hash_bound")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_self_consistent_forged_receipts_pass_source_custody_only(self):
        plan, inventory, path = self.fixture.make_freeze_inventory()
        args = self.fixture.change(
            freeze_inventory=path,
            expected_freeze_inventory_sha256=_hash_file(path),
        )
        digest, receipts, saved = _preunseal_inventory(
            args, plan, inventory["plan_sha256"])
        self.assertEqual(digest, _hash_file(path))
        self.assertEqual(len(receipts), 8)
        self.assertEqual(len(saved), 12)
        # These fields are attacker-selected in a self-consistent fixture.
        # The acceptance above is NOT a host or independent trainer attestation.

    def test_forged_eight_arm_custody_cannot_authorize_private_unseal(self):
        plan, inventory, path = self.fixture.make_freeze_inventory()
        args = self.fixture.change(
            preflight_only=False,
            freeze_inventory=path,
            expected_freeze_inventory_sha256=_hash_file(path),
        )
        with patch("trainingday.r18_eval._required_base", return_value=(
                self.fixture.base, self.fixture.root)), \
             patch("trainingday.r18_eval._hash_file", side_effect=lambda p:
                   CORPUS_SHA256 if p == self.fixture.manifest else _hash_file(p)), \
             patch("trainingday.r18_eval._source_manifest", return_value=(
                   self.fixture.source_manifest, self.fixture.source_sha)), \
             patch("trainingday.r18_eval._plan", return_value=(
                   plan, inventory["plan_sha256"])), \
             patch("trainingday.r18_eval._cuda_preflight") as cuda, \
             patch("trainingday.r18_eval._load_model", return_value=Mock()) as load, \
             patch("trainingday.r18_eval._bank") as bank:
            with self.assertRaisesRegex(AdmissionError, "independent.*attestation"):
                evaluate(args)
            cuda.assert_not_called()
            load.assert_not_called()
            bank.assert_not_called()


if __name__ == "__main__":
    unittest.main()
