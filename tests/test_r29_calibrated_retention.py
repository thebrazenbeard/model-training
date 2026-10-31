"""CPU-only PUBLIC TOY calibration/retention composition and negative controls."""

from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch

import trainingday.r29_calibrated_retention as r29

from trainingday.r25_multigeneration_retention import (
    GenerationEvidence, PublicBankManifest, case_order_sha256,
    public_score_vector_sha256,
)
from trainingday.admission import AdmissionError
from trainingday.r27_pilot_preregistration import (
    PilotCase, PilotProtocol, preregister_pilot,
)
from trainingday.r28_calibration import (
    CalibrationPlan, CalibrationSample, preregister_calibration,
)
from trainingday.r29_calibrated_retention import (
    CalibratedRetentionError, CalibratedRetentionPlan,
    preregister_calibrated_retention, score_synthetic_calibrated_retention,
)


def sha(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def fixture():
    ids = tuple(f"publiccase{i:02d}" for i in range(24))
    frozen_bits = (True,) * 6 + (False,) * 18
    samples = (
        (True,) * 18 + (False,) * 6,
        (True,) * 20 + (False,) * 4,
        (True,) * 22 + (False,) * 2,
    )
    pilot = PilotProtocol(
        "R29ToyPilot", sha("frozen-base"), sha("tokenizer"),
        sha("train-corpus"), (sha("known-train-prompt"),),
        tuple(PilotCase(case, ("ARITHMETIC", "PROGRAM_REASONING", "SOURCE_PRIORITY")[i % 3],
                        sha("prompt" + case), sha("rights" + case))
              for i, case in enumerate(ids)),
        ("opaqueOne", "opaqueTwo", "opaqueThree", "opaqueFour"),
        0.1, 0.49,
    )
    pilot_receipt = preregister_pilot(pilot)
    calibration_plan = CalibrationPlan("R29ToyRun", "R29PublicSplit",
                                       pilot_receipt.manifest_sha256, 10, 0.05, 0.05)
    calibration_receipt = preregister_calibration(pilot, pilot_receipt, calibration_plan)
    manifest = PublicBankManifest(sha("public-source"), pilot.frozen_base_sha256,
                                  sha("initial-adapter"), ids)
    role_map = dict(zip(pilot.opaque_arm_ids,
                        ("FROZEN", "ZERO_UPDATE", "CORRECT_LABEL", "CORRUPTED_LABEL")))
    previous = manifest.initial_adapter_sha256
    records = []
    sample_maps = {}
    for generation, bits, conf in zip("ABC", samples, (0.77, 0.82, 0.88)):
        adapter = sha("adapter" + generation)
        records.append(GenerationEvidence(
            generation, previous, adapter, manifest.base_sha256,
            case_order_sha256(ids), sha("scorer" + generation),
            public_score_vector_sha256(case_order_sha256(ids), bits),
            sha("trainer" + generation), bits,
            zero_control_present=True, corrupt_control_present=True,
        ))
        previous = adapter
        per_role = {}
        for role, role_bits, confidence in (
            ("FROZEN", frozen_bits, 0.25),
            ("ZERO_UPDATE", frozen_bits, 0.25),
            ("CORRECT_LABEL", bits, conf),
            ("CORRUPTED_LABEL", (False,) * 24, 0.1),
        ):
            per_role[role] = {
                case: CalibrationSample(case, pilot.cases[i].prompt_sha256,
                                        confidence, correct, True, False)
                for i, (case, correct) in enumerate(zip(ids, role_bits))
            }
        sample_maps[generation] = {arm: per_role[role] for arm, role in role_map.items()}
    plan = CalibratedRetentionPlan(manifest.source_sha256, case_order_sha256(ids),
                                   pilot_receipt.manifest_sha256,
                                   calibration_receipt.manifest_sha256,
                                   ())
    receipt = preregister_calibrated_retention(
        manifest, pilot, pilot_receipt, calibration_plan, calibration_receipt,
        plan, role_map)
    return (manifest, tuple(records), pilot, pilot_receipt, calibration_plan,
            calibration_receipt, plan, receipt, role_map, sample_maps)


def score(args):
    manifest, records, pilot, pilot_receipt, cp, cr, plan, receipt, roles, maps = args
    return score_synthetic_calibrated_retention(
        manifest, records, pilot, pilot_receipt, cp, cr, plan, receipt, roles, maps)


class CalibratedRetentionTests(unittest.TestCase):
    def test_stable_toy_sequence_remains_no_claim(self):
        result = score(fixture())
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["hold_reasons"], ["DECLARED_ONLY_NO_INDEPENDENT_EVIDENCE"])
        self.assertEqual([row["transition"] for row in result["transitions"]],
                         ["A->B", "B->C", "A->C"])
        self.assertTrue(all(row["case_regressions"] == 0 for row in result["transitions"]))
        for key in ("neural_improvement_proven", "release_authorized",
                    "independent_oracle_authenticated", "fresh_reload_verified",
                    "private_eval_unsealed"):
            self.assertIs(result[key], False)

    def test_forgetting_masked_by_better_average_calibration_is_exposed(self):
        args = list(fixture())
        manifest, records = args[0], list(args[1])
        # C trades previously competent item 0 for an extra late item.
        bits = (False,) + (True,) * 22 + (False,)
        records[2] = replace(records[2], case_passes=bits,
                             score_vector_sha256=public_score_vector_sha256(
                                 case_order_sha256(manifest.case_ids), bits))
        maps = args[9]
        arm = next(k for k, v in args[8].items() if v == "CORRECT_LABEL")
        for i, case in enumerate(manifest.case_ids):
            maps["C"][arm][case] = replace(maps["C"][arm][case], correct=bits[i])
        args[1] = tuple(records)
        report = score(tuple(args))
        self.assertEqual(report["transitions"][2]["case_regressions"], 1)
        self.assertIn("DECLARED_CATASTROPHIC_FORGETTING_RISK", report["hold_reasons"])
        self.assertLessEqual(report["generation_calibration"]["C"]["ece"],
                             report["generation_calibration"]["A"]["ece"] + 0.05)

    def test_reject_wrong_case_source_and_role_binding(self):
        args = list(fixture())
        args[6] = replace(args[6], source_sha256=sha("different-public-source"))
        with self.assertRaises(CalibratedRetentionError):
            score(tuple(args))
        args = list(fixture())
        args[0] = replace(args[0], case_ids=tuple(reversed(args[0].case_ids)))
        with self.assertRaises(CalibratedRetentionError):
            score(tuple(args))
        args = list(fixture())
        args[8] = {arm: ("ZERO_UPDATE" if role == "FROZEN" else "FROZEN"
                         if role == "ZERO_UPDATE" else role)
                   for arm, role in args[8].items()}
        with self.assertRaises(CalibratedRetentionError):
            score(tuple(args))

    def test_public_abstention_not_preregistered_and_format_regression(self):
        args = list(fixture())
        arm = next(k for k, v in args[8].items() if v == "CORRECT_LABEL")
        case = args[0].case_ids[-1]
        args[9]["C"][arm][case] = replace(args[9]["C"][arm][case], refused=True)
        report = score(tuple(args))
        self.assertIn("UNDECLARED_ABSTENTION", report["hold_reasons"])
        args = list(fixture())
        arm = next(k for k, v in args[8].items() if v == "CORRECT_LABEL")
        case = args[0].case_ids[-1]
        args[9]["C"][arm][case] = replace(args[9]["C"][arm][case], format_ok=False)
        report = score(tuple(args))
        self.assertIn("FORMAT_REGRESSION", report["hold_reasons"])

    def test_zero_or_corrupt_control_drift_is_detected(self):
        args = list(fixture())
        arm = next(k for k, v in args[8].items() if v == "ZERO_UPDATE")
        case = args[0].case_ids[0]
        args[9]["C"][arm][case] = replace(args[9]["C"][arm][case], confidence=0.3)
        report = score(tuple(args))
        self.assertIn("CONTROL_DRIFT", report["hold_reasons"])
        args = list(fixture())
        arm = next(k for k, v in args[8].items() if v == "CORRUPTED_LABEL")
        case = args[0].case_ids[0]
        args[9]["C"][arm][case] = replace(args[9]["C"][arm][case], confidence=0.3)
        report = score(tuple(args))
        self.assertIn("CONTROL_DRIFT", report["hold_reasons"])

    def test_invalid_predeclared_abstention_and_fake_receipts_rejected(self):
        args = list(fixture())
        args[6] = replace(args[6], allowed_abstain_case_ids=("absent-case",))
        with self.assertRaises(CalibratedRetentionError):
            score(tuple(args))
        args = list(fixture())
        args[7] = replace(args[7], manifest_sha256=sha("fake-plan"))
        with self.assertRaises(CalibratedRetentionError):
            score(tuple(args))
        args = list(fixture())
        args[6] = replace(args[6], allowed_abstain_case_ids=[args[0].case_ids[0]])
        with self.assertRaises(CalibratedRetentionError):
            score(tuple(args))

    def test_malformed_confidence_and_skewed_per_case_commitments_rejected(self):
        args = list(fixture())
        arm = next(k for k, v in args[8].items() if v == "CORRECT_LABEL")
        case = args[0].case_ids[0]
        args[9]["C"][arm][case] = replace(args[9]["C"][arm][case], confidence=float("nan"))
        with self.assertRaises(CalibratedRetentionError):
            score(tuple(args))

    def test_r30_assessor_mutation_cannot_change_calibration_source_mid_score(self):
        args = list(fixture())
        arm = next(k for k, v in args[8].items() if v == "CORRECT_LABEL")
        case = args[0].case_ids[0]
        original = r29.score_synthetic_calibration
        calls = []

        def sabotage(*values):
            receipt = original(*values)
            calls.append(1)
            if len(calls) == 3:
                public = values[-1]
                public[arm][case] = replace(public[arm][case], confidence=0.5)
            return receipt

        with patch.object(r29, "score_synthetic_calibration", side_effect=sabotage):
            result = score(tuple(args))
        self.assertEqual(len(calls), 3)
        # Post-score side effects against an assessor-owned *copy* cannot
        # rewrite either our source snapshot or the caller's original input.
        self.assertEqual(args[9]["C"][arm][case].confidence, 0.88)
        self.assertAlmostEqual(result["generation_calibration"]["C"]["ece"], 0.03666666667)
        self.assertEqual(result["hold_reasons"], ["DECLARED_ONLY_NO_INDEPENDENT_EVIDENCE"])

    def test_r30_prereg_mutation_cannot_change_effective_role_map(self):
        args = list(fixture())
        original_roles = dict(args[8])
        original = r29.preregister_calibrated_retention

        def sabotage(*values):
            receipt = original(*values)
            roles = values[-1]
            for arm, role in list(roles.items()):
                if role == "FROZEN":
                    roles[arm] = "ZERO_UPDATE"
                elif role == "ZERO_UPDATE":
                    roles[arm] = "FROZEN"
            return receipt

        with patch.object(r29, "preregister_calibrated_retention", side_effect=sabotage):
            result = score(tuple(args))
        self.assertEqual(args[8], original_roles)
        self.assertEqual(result["preregistered_manifest_sha256"], args[7].manifest_sha256)
        self.assertEqual(result["hold_reasons"], ["DECLARED_ONLY_NO_INDEPENDENT_EVIDENCE"])

    def test_r31_spoofed_public_case_id_cannot_launder_through_deepcopy(self):
        args = list(fixture())
        expected = args[0].case_ids[0]
        calls = []

        class Spoof:
            def __deepcopy__(self, memo):
                calls.append("executed")
                return expected

        args[0] = replace(args[0], case_ids=(Spoof(),) + args[0].case_ids[1:])
        with self.assertRaises(AdmissionError):
            case_order_sha256(args[0].case_ids)
        with self.assertRaisesRegex(CalibratedRetentionError, "SOURCE.*INVALID"):
            score(tuple(args))
        self.assertEqual(calls, [], "caller __deepcopy__ hook must never run")

    def test_r31_spoofed_nested_pilot_and_sample_scalars_rejected_precopy(self):
        for place in ("pilot", "sample", "receipt", "generation"):
            with self.subTest(place=place):
                args = list(fixture())
                calls = []

                class Spoof:
                    def __deepcopy__(self, memo):
                        calls.append("executed")
                        return {"pilot": args[2].cases[0].case_id,
                                "sample": 0.88,
                                "receipt": args[7].manifest_sha256,
                                "generation": args[1][0].name}[place]

                if place == "pilot":
                    rows = list(args[2].cases)
                    rows[0] = replace(rows[0], case_id=Spoof())
                    args[2] = replace(args[2], cases=tuple(rows))
                elif place == "sample":
                    arm = next(k for k, v in args[8].items() if v == "CORRECT_LABEL")
                    key = args[0].case_ids[0]
                    args[9]["C"][arm][key] = replace(args[9]["C"][arm][key],
                                                      confidence=Spoof())
                elif place == "receipt":
                    args[7] = replace(args[7], manifest_sha256=Spoof())
                else:
                    args[1] = (replace(args[1][0], name=Spoof()),) + args[1][1:]
                with self.assertRaisesRegex(CalibratedRetentionError, "SOURCE.*INVALID"):
                    score(tuple(args))
                self.assertEqual(calls, [])

    def test_r31_hidden_dataclass_fields_cannot_trigger_deepcopy_hooks(self):
        for place in ("manifest", "pilot_case", "sample", "generation"):
            with self.subTest(place=place):
                args = list(fixture())
                calls = []

                class Spy:
                    def __deepcopy__(self, memo):
                        calls.append("executed")
                        return "source-laundered"

                if place == "manifest":
                    target = args[0]
                elif place == "pilot_case":
                    target = args[2].cases[0]
                elif place == "sample":
                    arm = next(k for k, v in args[8].items() if v == "CORRECT_LABEL")
                    target = args[9]["C"][arm][args[0].case_ids[0]]
                else:
                    target = args[1][0]
                object.__setattr__(target, "_unexpected_hidden_extra", Spy())
                with self.assertRaisesRegex(CalibratedRetentionError, "SOURCE.*INVALID"):
                    score(tuple(args))
                self.assertEqual(calls, [], "hidden source hook must not run")
        args = list(fixture())
        args[1] = (replace(args[1][0], score_vector_sha256=sha("stale")),) + args[1][1:]
        with self.assertRaises(CalibratedRetentionError):
            score(tuple(args))


if __name__ == "__main__":
    unittest.main()
