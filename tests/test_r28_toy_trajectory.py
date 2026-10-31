"""R28: CPU PUBLIC TOY replay tests, never Qwen or private evidence."""

import hashlib
import json
import subprocess
import sys
import unittest

from trainingday.r28_toy_trajectory import (
    PublicTrajectoryError,
    advance,
    compare_trajectories,
    environment_fingerprint,
    load_checkpoint,
    new_trajectory,
    receipt_digest,
    save_checkpoint,
    verify_public_reference_trajectory,
)


class GoldenTrajectoryTests(unittest.TestCase):
    def test_fixed_seed_exact_replay_with_step_receipts(self):
        a, left = advance(new_trajectory(721), 40)
        b, right = advance(new_trajectory(721), 40)
        self.assertEqual(a, b)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 40)
        self.assertEqual([r["step"] for r in left], list(range(1, 41)))
        self.assertTrue(all(len(r["state_sha256"]) == 64 for r in left))
        self.assertEqual(compare_trajectories(left, right)["status"], "EXACT_MATCH")
        self.assertEqual(len(compare_trajectories(left, right)["per_step_differences"]), 40)
        self.assertEqual(verify_public_reference_trajectory(left)["status"], "EXACT_MATCH")
        self.assertFalse(compare_trajectories(left, right)["qwen4b_qualified"])

    def test_split_optimizer_and_rng_resume_matches_continuous(self):
        one_shot, all_steps = advance(new_trajectory(1985), 35)
        stopped, first = advance(new_trajectory(1985), 11)
        encoded = save_checkpoint(stopped)
        self.assertEqual(encoded, save_checkpoint(stopped))
        restarted = load_checkpoint(encoded)
        final, rest = advance(restarted, 24)
        self.assertEqual(final, one_shot)
        self.assertEqual(first + rest, all_steps)
        self.assertEqual(compare_trajectories(first + rest, all_steps)["status"], "EXACT_MATCH")

    def test_multiple_seed_resume_matrix_and_maximum_bounded_steps(self):
        for seed in (1, 2, 7, 41, 721, 1985, 2**31, 2**32 - 1):
            with self.subTest(seed=seed):
                whole, receipts = advance(new_trajectory(seed), 64)
                first, left = advance(new_trajectory(seed), 31)
                result, right = advance(load_checkpoint(save_checkpoint(first)), 33)
                self.assertEqual(result, whole)
                self.assertEqual(left + right, receipts)
        at_cap, _ = advance(new_trajectory(721), 256)
        with self.assertRaises(PublicTrajectoryError):
            advance(at_cap, 1)

    def test_clean_process_checkpoint_reload_with_same_future_updates(self):
        halfway, _ = advance(new_trajectory(71), 13)
        payload_hex = save_checkpoint(halfway).hex()
        script = (
            "import json,sys; "
            "from trainingday.r28_toy_trajectory import load_checkpoint,advance,save_checkpoint; "
            "s=load_checkpoint(bytes.fromhex(sys.argv[1])); "
            "t,r=advance(s,17); "
            "print(json.dumps({'checkpoint':save_checkpoint(t).hex(), 'steps':r}))"
        )
        p = subprocess.run(
            [sys.executable, "-c", script, payload_hex], capture_output=True,
            text=True, check=True, timeout=10,
        )
        parsed = json.loads(p.stdout)
        expected, receipts = advance(halfway, 17)
        self.assertEqual(parsed["checkpoint"], save_checkpoint(expected).hex())
        self.assertEqual(parsed["steps"], list(receipts))

    def test_cli_public_toy_and_clean_process_resume(self):
        first = subprocess.run(
            [sys.executable, "-m", "trainingday.r28_toy_trajectory",
             "--seed", "721", "--steps", "3"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        partial = json.loads(first.stdout)
        self.assertEqual(partial["scope"], "PUBLIC_CPU_TOY_ONLY")
        self.assertEqual(len(partial["steps"]), 3)
        self.assertFalse(partial["gpu_training_authorized"])
        second = subprocess.run(
            [sys.executable, "-m", "trainingday.r28_toy_trajectory",
             "--resume-checkpoint-hex", partial["checkpoint_hex"], "--steps", "5"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        following = json.loads(second.stdout)
        self.assertEqual(following["steps"][0]["step"], 4)
        self.assertEqual(
            verify_public_reference_trajectory(partial["steps"] + following["steps"])["status"],
            "EXACT_MATCH",
        )

    def test_tampered_checkpoint_digest_rejected(self):
        state, _ = advance(new_trajectory(5), 7)
        payload = save_checkpoint(state)
        mutated = bytearray(payload)
        pos = payload.index(b"0x") + 3
        mutated[pos] = ord("8") if mutated[pos] != ord("8") else ord("9")
        with self.assertRaises(PublicTrajectoryError):
            load_checkpoint(bytes(mutated))
        with self.assertRaises(PublicTrajectoryError):
            load_checkpoint(payload + b" ")

    def test_seed_change_diverges_and_is_not_reported_as_improvement(self):
        _, left = advance(new_trajectory(1), 20)
        _, right = advance(new_trajectory(2), 20)
        comparison = compare_trajectories(left, right)
        self.assertEqual(comparison["status"], "TRAJECTORY_DIVERGENCE")
        self.assertIsNotNone(comparison["first_divergence_step"])
        self.assertEqual(len(comparison["per_step_differences"]), 20)
        self.assertFalse(comparison["qwen4b_qualified"])

    def test_reordered_or_repeated_receipts_not_a_pass(self):
        _, records = advance(new_trajectory(19), 8)
        for bad in (records[:-1], records[1:] + records[:1], records[:3] + records[2:]):
            with self.subTest(bad=bad), self.assertRaises(PublicTrajectoryError):
                compare_trajectories(records, bad)

    def test_declared_tolerance_cannot_cover_wrong_sample_or_rng(self):
        _, records = advance(new_trajectory(82), 8)
        sample_bad = [dict(r) for r in records]
        sample_bad[3]["sample_index"] = (sample_bad[3]["sample_index"] + 1) % 6
        sample_bad[3]["state_sha256"] = receipt_digest(sample_bad[3])
        self.assertEqual(compare_trajectories(records, sample_bad, tolerance=0.000001)["status"],
                         "TRAJECTORY_DIVERGENCE")
        rng_bad = [dict(r) for r in records]
        rng_bad[5]["rng"] += 1
        rng_bad[5]["state_sha256"] = receipt_digest(rng_bad[5])
        self.assertEqual(compare_trajectories(records, rng_bad, tolerance=0.000001)["status"],
                         "TRAJECTORY_DIVERGENCE")

    def test_tolerance_is_bounded_and_makes_no_model_claim(self):
        _, records = advance(new_trajectory(35), 3)
        variant = [dict(r) for r in records]
        variant[-1]["weights"] = list(variant[-1]["weights"])
        variant[-1]["weights"][0] += 0.0000001
        variant[-1]["state_sha256"] = receipt_digest(variant[-1])
        result = compare_trajectories(records, variant, tolerance=0.000001)
        self.assertEqual(result["status"], "WITHIN_DECLARED_TOLERANCE")
        self.assertFalse(result["independent_receipts_authenticated"])
        self.assertFalse(result["qwen4b_qualified"])
        self.assertEqual(compare_trajectories(records, variant)["status"],
                         "TRAJECTORY_DIVERGENCE")
        self.assertEqual(verify_public_reference_trajectory(variant, tolerance=0.000001)["status"],
                         "WITHIN_DECLARED_TOLERANCE")

    def test_reference_verifier_falsifies_self_consistent_wrong_gradient(self):
        _, records = advance(new_trajectory(721), 4)
        bad = [dict(r) for r in records]
        bad[2]["weights"] = list(bad[2]["weights"])
        bad[2]["weights"][1] += 0.25
        bad[2]["state_sha256"] = receipt_digest(bad[2])
        self.assertEqual(compare_trajectories(bad, bad)["status"], "EXACT_MATCH")
        verdict = verify_public_reference_trajectory(bad)
        self.assertEqual(verdict["status"], "TRAJECTORY_DIVERGENCE")
        self.assertEqual(verdict["first_divergence_step"], 3)
        self.assertTrue(verdict["reference_recomputed_from_pinned_public_toy"])
        self.assertFalse(verdict["physical_cross_host_attested"])

    def test_manual_first_step_weights_loss_and_sample(self):
        _, records = advance(new_trajectory(721), 1)
        first = records[0]
        # Independent closed-form check: selected toy sample is x=(1,1), y=.2.
        # At zero weights, error=-.2; gradient per coordinate=-.4;
        # fixed learning rate .0125 yields +.005 weight update on both.
        self.assertEqual(first["sample_index"], 2)
        self.assertAlmostEqual(first["weights"][0], 0.005, places=15)
        self.assertAlmostEqual(first["weights"][1], 0.005, places=15)
        self.assertAlmostEqual(first["pre_loss"], 0.04, places=15)
        self.assertAlmostEqual(first["post_loss"], 0.0361, places=15)

    def test_changed_hash_cannot_pass_even_with_generous_tolerance(self):
        _, records = advance(new_trajectory(91), 4)
        bad = [dict(r) for r in records]
        bad[2]["state_sha256"] = "0" * 64
        with self.assertRaises(PublicTrajectoryError):
            compare_trajectories(records, bad, tolerance=0.0001)

    def test_invalid_numeric_step_and_checkpoint_shapes_fail_closed(self):
        for seed in (0, -1, True, 2**32, "42"):
            with self.subTest(seed=seed), self.assertRaises(PublicTrajectoryError):
                new_trajectory(seed)
        state = new_trajectory(10)
        for count in (-1, 513, True, 1.0, "2"):
            with self.subTest(count=count), self.assertRaises(PublicTrajectoryError):
                advance(state, count)
        for tolerance in (-1, float("nan"), float("inf"), "0.0", True):
            with self.subTest(tolerance=tolerance), self.assertRaises(PublicTrajectoryError):
                compare_trajectories((), (), tolerance=tolerance)
        for payload in (b"", b"{}", b"x" * 10000):
            with self.subTest(payload=payload[:8]), self.assertRaises(PublicTrajectoryError):
                load_checkpoint(payload)

    def test_corpus_digest_tamper_not_a_resume(self):
        state, _ = advance(new_trajectory(22), 4)
        envelope = json.loads(save_checkpoint(state))
        envelope["payload"]["public_dataset_sha256"] = "0" * 64
        replacement = json.dumps(envelope["payload"], sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8")
        envelope["payload_sha256"] = hashlib.sha256(replacement).hexdigest()
        attack = json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode("utf-8")
        with self.assertRaises(PublicTrajectoryError):
            load_checkpoint(attack)

    def test_fingerprint_is_bounded_nonidentifying_and_explicit(self):
        fp = environment_fingerprint()
        self.assertTrue(fp["python_implementation"])
        self.assertIn(fp["byteorder"], ("little", "big"))
        self.assertIn(fp["os_family"], ("Windows", "Linux", "Darwin", "Other"))
        self.assertNotIn("username", fp)
        self.assertNotIn("hostname", fp)
        self.assertNotIn("address", fp)
        self.assertEqual(len(fp["source_fingerprint_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
