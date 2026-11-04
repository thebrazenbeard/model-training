"""Adversarial source-only checks for the rights-pending curriculum bridge."""

import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from trainingday import public_dev_bridge as bridge


def payloads():
    def row(domain, split):
        family = "00" if split == "train" else "24"
        name = f"oc-v1-{domain}-{family}-0"
        family_id = f"oc-v1-{domain}-{family}"
        return {"id": name, "case_id": name, "domain": domain,
                "family_id": family_id, "template_id": family_id,
                "source_id": f"{bridge.GENERATOR}/{domain}/{family}",
                "archetype_id": f"oc-v1-{domain}", "variant": 0,
                "split": split, "prompt": f"{domain} {split} question?",
                "answer": f"{domain} answer", "rights_status": "NOT_ADMITTED",
                "training_authorized": False,
                "provenance": {"kind": "source_authored", "generator": bridge.GENERATOR,
                               "external_corpus": False}}

    trains = [row("logic", "train"), row("math", "train")]
    devs = [row("logic", "dev"), row("math", "dev")]
    encode = lambda rows: "".join(json.dumps(r, sort_keys=True, ensure_ascii=False,
                                            separators=(",", ":")) + "\n" for r in rows).encode()
    train_bytes, dev_bytes = encode(trains), encode(devs)
    manifest = {"schema_version": "original-curriculum-candidate-v1",
                "generator": bridge.GENERATOR,
                "generator_source_sha256": bridge.SOURCE_SHA256,
                "archetype_overlap": True,
                "evaluation_scope": "same-archetype diagnostic only; NOT unseen-archetype transfer",
                "rights_status": "NOT_ADMITTED", "training_authorized": False,
                "files": {"train.jsonl": {"rows": 2, "sha256": hashlib.sha256(train_bytes).hexdigest()},
                          "dev.jsonl": {"rows": 2, "sha256": hashlib.sha256(dev_bytes).hexdigest()}}}
    return train_bytes, dev_bytes, json.dumps(manifest).encode()


def fake_profile(payload):
    return patch.multiple(bridge, TRAIN_SHA256=hashlib.sha256(payload[0]).hexdigest(),
                          DEV_SHA256=hashlib.sha256(payload[1]).hexdigest(),
                          TRAIN_COUNT=2, DEV_COUNT=2,
                          DOMAIN_COUNTS={"logic": (1, 1), "math": (1, 1)})


def rewrite_row(payload, split_index, edit):
    updated = list(payload)
    rows = [json.loads(line) for line in updated[split_index].splitlines()]
    edit(rows)
    updated[split_index] = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False,
                                              separators=(",", ":")) + "\n" for r in rows).encode()
    manifest = json.loads(updated[2])
    name = "train.jsonl" if split_index == 0 else "dev.jsonl"
    manifest["files"][name]["sha256"] = hashlib.sha256(updated[split_index]).hexdigest()
    updated[2] = json.dumps(manifest).encode()
    return updated


class SourceBridgeTest(unittest.TestCase):
    def test_builds_public_schema_but_never_predictions_or_training_grant(self):
        payload = payloads()
        with fake_profile(payload):
            s = bridge.build_w31_public_scaffold(*payload)
        self.assertEqual(len(s["fixture"]["cases"]), 2)
        self.assertEqual(s["fixture"]["cases"][0]["expected_semantic"], "logic answer")
        self.assertEqual(s["fixture"]["cases"][0]["source_group_id"],
                         f"{bridge.GENERATOR}/logic/24")
        self.assertEqual(s["fixture"]["split"], "transfer")
        self.assertEqual(s["actual_evaluation_scope"], "SAME_ARCHETYPE_DIAGNOSTIC_ONLY")
        self.assertEqual(s["train_manifest"]["rights_status"], "NOT_ADMITTED")
        self.assertFalse(s["train_manifest"]["training_authorized"])
        self.assertFalse(s["ready_for_training"])
        self.assertNotIn("predictions", s)
        self.assertEqual(s["prediction_status"], "MISSING_INDEPENDENT_MODEL_PREDICTIONS")

    def test_declines_mutated_bytes_and_fabricated_rights(self):
        src = payloads()
        with fake_profile(src):
            for i in (0, 1):
                bad = list(src)
                bad[i] += b" "
                with self.assertRaises(bridge.BridgeError):
                    bridge.build_w31_public_scaffold(*bad)
            bad_manifest = json.loads(src[2])
            for key, value in (("rights_status", "ADMITTED"), ("training_authorized", True),
                               ("archetype_overlap", False)):
                mutated = copy.deepcopy(bad_manifest)
                mutated[key] = value
                with self.assertRaises(bridge.BridgeError):
                    bridge.build_w31_public_scaffold(src[0], src[1], json.dumps(mutated).encode())

    def test_disallows_public_to_private_bank_custody_upgrade(self):
        src = payloads()
        profile = fake_profile(src)
        profile.start()
        self.addCleanup(profile.stop)
        s = bridge.build_w31_public_scaffold(*src)
        ids = [x["case_id"] for x in s["fixture"]["cases"]]
        arms = {a: [{"case_id": i, "semantic_answer": "externally generated"} for i in ids]
                for a in bridge.ARMS}
        b = bridge.attach_independent_predictions(s, arms)
        self.assertEqual(set(b["predictions"]), set(bridge.ARMS))
        self.assertEqual(b["provenance"]["evaluation_bank_id"], "UNKNOWN")
        self.assertNotIn("prediction_status", b)
        for mutate in (lambda x: x.pop("sham"),
                       lambda x: x["trained"].pop(),
                       lambda x: x["trained"].append(x["trained"][0]),
                       lambda x: x["trained"][0].update({"sealed": True})):
            bad = copy.deepcopy(arms)
            mutate(bad)
            with self.assertRaises(bridge.BridgeError):
                bridge.attach_independent_predictions(s, bad)
        with self.assertRaises(bridge.BridgeError):
            bridge.attach_independent_predictions(s, None)

        for mutate in (lambda x: x["fixture"].update({"visibility": "protected"}),
                       lambda x: x["fixture"].update({"sealed": True}),
                       lambda x: x["fixture"]["cases"][0].update({"expected_semantic": "forged gold"}),
                       lambda x: x["train_manifest"].update({"training_authorized": True}),
                       lambda x: x["source_byte_witnesses"].update({"train_sha256": "bad"})):
            bad = copy.deepcopy(s)
            mutate(bad)
            with self.assertRaises(bridge.BridgeError):
                bridge.attach_independent_predictions(bad, arms)

    def test_leakage_rejected_even_if_caller_rehashes_source_files(self):
        original = payloads()
        for split_idx, mutate in (
            (1, lambda rows: rows[0].update({"prompt": "  LOGIC   TRAIN question?  "})),
            (1, lambda rows: rows[0].update({"family_id": "oc-v1-logic-00",
                                                 "template_id": "oc-v1-logic-00"})),
            (1, lambda rows: rows[0].update({"source_id": f"{bridge.GENERATOR}/logic/00"})),
            (0, lambda rows: rows[1].update({"case_id": rows[0]["case_id"],
                                                 "id": rows[0]["id"]})),
        ):
            altered = rewrite_row(original, split_idx, mutate)
            with fake_profile(altered):
                with self.assertRaises(bridge.BridgeError):
                    bridge.build_w31_public_scaffold(*altered)


if __name__ == "__main__":
    unittest.main()
