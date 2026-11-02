"""Fail-closed source-only Minotaur adaptive consumer regression tests."""

import copy
import hashlib
import json
import random
import unittest

from trainingday.minotaur_adaptive_source_intake_v1 import (
    PINNED_LABYRINTH_HEAD,
    PINNED_LABYRINTH_TREE,
    UNVERIFIED_REJECTED,
    inspect_adaptive_minotaur_source,
)


_DOMAIN = b"labyrinth.hf-adaptive-source-manifest.v1\x00"


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("utf-8")


def _example():
    rooms = []
    feedback = []
    for index in range(2):
        room_id = f"{index + 1:032x}"
        case_id = f"train-taskmaster-{room_id}-{index + 1:012x}"
        prompt_hash = hashlib.sha256(f"training prompt {index}".encode()).hexdigest()
        rooms.append({
            "room_id": room_id,
            "case_id": case_id,
            "skill": "arithmetic",
            "tactic": "practice" if index == 0 else "transfer",
            "difficulty": 1,
            "planner_source": "initial_train_bootstrap" if index == 0 else
                              "policy_plus_adaptive_train",
            "prompt_sha256": prompt_hash,
        })
        feedback.append([
            room_id, case_id, prompt_hash, int(index == 0),
            f"{index + 8:064x}", f"{index + 9:064x}",
        ])
    shuffled = list(range(len(rooms)))
    random.Random("13:epoch:0:tier:1").shuffle(shuffled)
    payload = {
        "schema": "labyrinth.hf-adaptive-source-manifest.v1",
        "scope": {
            "schema": "labyrinth.training-scope.v1",
            "objective_id": "minotaur-arithmetic", "domain": "arithmetic",
            "skills": ["arithmetic"], "specificity": 2, "scale": 1,
            "max_cycles": 1, "seed": 7,
        },
        "cycle_index": 0,
        "pilot_budget": {
            "seed": 13, "train_count": 2, "eval_count": 2,
            "tiers": [1], "steps": 2,
        },
        "train_rooms": rooms,
        "ordered_optimizer_exposures": [
            [rooms[index]["case_id"], rooms[index]["prompt_sha256"]]
            for index in shuffled
        ],
        "initial_checkpoint_sha256": f"{8:064x}",
        "adaptive_feedback_rows": feedback,
    }
    return payload


def _serialized(payload):
    value = copy.deepcopy(payload)
    value["source_sha256"] = hashlib.sha256(_DOMAIN + _json(value)).hexdigest()
    raw = _json(value)
    return raw, hashlib.sha256(raw).hexdigest()


def _inspect(payload=None, **overrides):
    raw, expected = _serialized(_example() if payload is None else payload)
    args = {
        "manifest_bytes": raw,
        "expected_manifest_sha256": expected,
        "producer_git_head": PINNED_LABYRINTH_HEAD,
        "producer_git_tree": PINNED_LABYRINTH_TREE,
        "explicit_adaptive_opt_in": True,
        "staged_roster_present": False,
    }
    args.update(overrides)
    return inspect_adaptive_minotaur_source(**args)


class MinotaurAdaptiveIntakeTests(unittest.TestCase):
    def assert_all_grants_closed(self, result):
        self.assertEqual(result["status"], UNVERIFIED_REJECTED)
        for field in (
            "source_authenticated", "rights_holder_authorized",
            "data_steward_authorized", "private_evaluator_custody_authenticated",
            "q6_monotonic_witness_verified", "host_process_custody_verified",
            "gpu_lease_granted", "optimizer_steps_verified",
            "physical_training_authorized", "model_training_granted",
        ):
            self.assertIs(result[field], False, field)

    def test_consistent_source_shape_still_cannot_authorize_training(self):
        result = _inspect()
        self.assert_all_grants_closed(result)
        self.assertIs(result["structural_source_consistent"], True)
        self.assertEqual(result["train_room_count"], 2)
        self.assertEqual(result["optimizer_exposure_count"], 2)
        self.assertNotIn("adaptive_feedback_rows", result)
        self.assertNotIn("train_rooms", result)

    def test_missing_opt_in_and_staged_mix_fail_closed(self):
        for kwargs in (
            {"explicit_adaptive_opt_in": False},
            {"staged_roster_present": True},
            {"explicit_adaptive_opt_in": 1},
        ):
            with self.subTest(kwargs=kwargs):
                result = _inspect(**kwargs)
                self.assert_all_grants_closed(result)
                self.assertIs(result["structural_source_consistent"], False)

    def test_revision_assertions_and_source_hash_required(self):
        for kwargs in (
            {"producer_git_head": "0" * 40},
            {"producer_git_tree": "0" * 40},
            {"expected_manifest_sha256": "0" * 64},
        ):
            with self.subTest(kwargs=kwargs):
                result = _inspect(**kwargs)
                self.assert_all_grants_closed(result)
                self.assertIs(result["structural_source_consistent"], False)

    def test_malformed_source_and_secret_field_rejected(self):
        for mutation in (
            lambda p: p.update({"answer": "1"}),
            lambda p: p["train_rooms"][0].update({"gold": "1"}),
            lambda p: p["pilot_budget"].update({"steps": True}),
            lambda p: p["pilot_budget"].update({"train_count": 1024}),
            lambda p: p["scope"].update({"skills": ["cognitive"]}),
            lambda p: p.update({"cycle_index": 1}),
            lambda p: p["train_rooms"][1].update({"room_id": "0" * 32}),
            lambda p: p["adaptive_feedback_rows"][1].__setitem__(4, "0" * 64),
            lambda p: p["ordered_optimizer_exposures"][0].__setitem__(0, "wrong"),
        ):
            payload = _example()
            mutation(payload)
            with self.subTest(mutation=mutation):
                result = _inspect(payload)
                self.assert_all_grants_closed(result)
                self.assertIs(result["structural_source_consistent"], False)

    def test_duplicate_keys_and_bomb_inputs_fail_closed(self):
        raw = b'{"schema": "x", "schema": "y"}'
        result = _inspect(manifest_bytes=raw,
                          expected_manifest_sha256=hashlib.sha256(raw).hexdigest())
        self.assert_all_grants_closed(result)
        self.assertIs(result["structural_source_consistent"], False)
        result = _inspect(manifest_bytes=b"[]" * 131_073)
        self.assert_all_grants_closed(result)
        self.assertIs(result["structural_source_consistent"], False)
        deep = b"[" * 2_000 + b"0" + b"]" * 2_000
        result = _inspect(manifest_bytes=deep,
                          expected_manifest_sha256=hashlib.sha256(deep).hexdigest())
        self.assert_all_grants_closed(result)
        self.assertIs(result["structural_source_consistent"], False)

    def test_forbidden_private_case_overlap_rejected_without_echo(self):
        case_id = _example()["train_rooms"][0]["case_id"]
        result = _inspect(forbidden_case_ids=(case_id,))
        self.assert_all_grants_closed(result)
        self.assertIs(result["structural_source_consistent"], False)
        self.assertNotIn(case_id, str(result))
        for forbidden in ((case_id,) * 1_025, ("unbounded",), (case_id, case_id)):
            with self.subTest(forbidden=forbidden[:2]):
                result = _inspect(forbidden_case_ids=forbidden)
                self.assert_all_grants_closed(result)
                self.assertIs(result["structural_source_consistent"], False)

    def test_coforged_final_bit_is_undetectable_and_never_authorized(self):
        payload = _example()
        payload["adaptive_feedback_rows"][-1][3] ^= 1
        result = _inspect(payload)
        self.assertIs(result["structural_source_consistent"], True)
        self.assert_all_grants_closed(result)


if __name__ == "__main__":
    unittest.main()
