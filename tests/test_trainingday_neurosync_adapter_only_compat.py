"""Offline Neuro-Sync adapter-only ↔ official TRAININGDAY source handoff."""

import hashlib
import json
import unittest

from trainingday.official_public_base import OFFICIAL_REVISION, SHARDS
from trainingday.neurosync_adapter_only_compat import (
    AdapterOnlySourceMismatch,
    check_adapter_only_source_handoff,
    expected_official_source_bundle,
)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def fixture():
    target_map = {
        "rank": 2, "lora_alpha": 4, "target_modules": ["q_proj"],
        "lora_pairs": [{"module": "base_model.model.layers.0.self_attn.q_proj",
                        "rank": 2, "input_features": 3, "output_features": 4,
                        "dtype": "BF16", "target": "q_proj"}],
    }
    artifacts = {name: {"sha256": hash_char * 64, "size_bytes": 1000}
                 for name, hash_char in (("donor_base", "1"), ("recipient_base", "1"),
                                         ("adapter_weights", "4"), ("adapter_config", "5"),
                                         ("train_manifest", "6"),
                                         ("sealed_holdout_manifest", "7"),
                                         ("shuffled_control_manifest", "8"))}
    intent = {
        "schema": "NEUROSYNC_PEFT_ADAPTER_ONLY_INTENT_V1",
        "method": "peft-adapter-only",
        "base": {"base_revision": OFFICIAL_REVISION,
                 "donor_base_weights_sha256": "1" * 64,
                 "recipient_base_weights_sha256": "1" * 64,
                 "donor_tokenizer_sha256": "2" * 64,
                 "recipient_tokenizer_sha256": "2" * 64,
                 "donor_config_sha256": "3" * 64,
                 "recipient_config_sha256": "3" * 64},
        "adapter": {"weights_sha256": "4" * 64,
                    "config_sha256": "5" * 64,
                    "target_map": target_map,
                    "target_map_sha256": canonical_sha(target_map)},
        "input_artifacts": artifacts,
        "training_provenance": {"source_revision_sha256": "a" * 64,
                                "optimizer_receipt_sha256": None,
                                "status": "UNVERIFIED"},
        "data_rights": {"record_sha256": "b" * 64,
                        "rights_holder_label": "synthetic-steward",
                        "usage_scope": "local-experiment-only"},
        "evaluation": {"donor_disconnected": True,
                       "cold_recipient_reload": True,
                       "baseline_arm": "untouched-recipient",
                       "negative_control_arm": "shuffled-donor",
                       "evaluator_custody": "independent-host-proposed"},
        "budget": {"max_external_cost_usd": 0, "max_gpu_minutes": 90},
    }
    envelope = {
        "schema": "NEUROSYNC_TRAININGDAY_PEFT_ADAPTER_ONLY_ENVELOPE_V1",
        "method": "peft-adapter-only",
        "evidence_class": "UNTRUSTED_PEFT_ADAPTER_ONLY_PROPOSAL",
        "intent_sha256_untrusted": canonical_sha(intent),
        "proposed_base_weights_sha256": "1" * 64,
        "proposed_artifact_digests": {role: item["sha256"] for role, item in artifacts.items()},
        "target_map_sha256_untrusted": canonical_sha(target_map),
        "optimizer_receipt_sha256_untrusted": None,
        "admission": {
            "host_artifact_verification": "UNVERIFIED",
            "dataset_training_rights": "UNVERIFIED",
            "independent_target_layer_compatibility": "UNVERIFIED",
            "optimizer_and_adapter_restore": "UNVERIFIED",
            "sealed_independent_evaluation": "UNVERIFIED",
            "cold_recipient_reload": "UNVERIFIED",
            "serving_pid_model_binding": "UNVERIFIED",
            "physical_launch_authority": "HOLD",
            "trainingday_ready": False,
        },
        "interpretation": "caller-declared source-only adapter candidate",
    }
    return intent, envelope, expected_official_source_bundle()


class NeuroSyncAdapterOnlySourceTests(unittest.TestCase):
    def test_official_two_shard_source_join_stays_hard_hold(self):
        intent, envelope, bundle = fixture()
        report = check_adapter_only_source_handoff(intent, envelope, bundle)
        self.assertEqual([x["path"] for x in bundle["ordered_shards"]],
                         [item[0] for item in SHARDS])
        self.assertTrue(report["source_bundle_aligned"])
        self.assertFalse(report["base_weight_bytes_verified"])
        self.assertFalse(report["trainingday_ready"])
        self.assertEqual(report["physical_launch_authority"], "HOLD")
        self.assertEqual(report["source_evidence_class"], "SOURCE_PIN_METADATA_ONLY")

    def test_swapped_missing_or_wrong_pinned_shard_rejected(self):
        for mutate in (
            lambda b: b["ordered_shards"].reverse(),
            lambda b: b["ordered_shards"].pop(),
            lambda b: b["ordered_shards"][0].update(sha256_lfs_oid="f" * 64),
        ):
            intent, envelope, bundle = fixture()
            mutate(bundle)
            with self.assertRaises(AdapterOnlySourceMismatch):
                check_adapter_only_source_handoff(intent, envelope, bundle)

    def test_wrong_revision_or_mismatched_intent_envelope_rejected(self):
        intent, envelope, bundle = fixture()
        intent["base"]["base_revision"] = "e" * 40
        with self.assertRaises(AdapterOnlySourceMismatch):
            check_adapter_only_source_handoff(intent, envelope, bundle)

    def test_aggressive_third_party_base_family_cannot_replace_official_source(self):
        intent, envelope, bundle = fixture()
        bundle["repository"] = "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive"
        bundle["revision"] = "d61dd146c8fd44c9a49cdb7f59f34e17b61902d8"
        bundle["ordered_shards"] = [
            {"path": "model.safetensors-00001-of-00002.safetensors",
             "size_bytes": 5329398720,
             "sha256_lfs_oid": "36dd0f7de62c546e7d163c2a4461208d731fad91ebea9bc31226b54c1c86511f"},
            {"path": "model.safetensors-00002-of-00002.safetensors",
             "size_bytes": 3990429440,
             "sha256_lfs_oid": "e2fb50b735a8e10210eece763097f906955ca9db6ba557215f549a46b622157f"},
        ]
        with self.assertRaises(AdapterOnlySourceMismatch):
            check_adapter_only_source_handoff(intent, envelope, bundle)
        intent, envelope, bundle = fixture()
        envelope["proposed_artifact_digests"]["adapter_weights"] = "f" * 64
        with self.assertRaises(AdapterOnlySourceMismatch):
            check_adapter_only_source_handoff(intent, envelope, bundle)

    def test_shard_or_source_manifest_cannot_masquerade_as_full_model_byte_sha(self):
        for digest in (SHARDS[0][2], expected_official_source_bundle()["manifest_sha256"]):
            intent, envelope, bundle = fixture()
            intent["base"]["donor_base_weights_sha256"] = digest
            intent["base"]["recipient_base_weights_sha256"] = digest
            intent["input_artifacts"]["donor_base"]["sha256"] = digest
            intent["input_artifacts"]["recipient_base"]["sha256"] = digest
            envelope["proposed_base_weights_sha256"] = digest
            envelope["proposed_artifact_digests"]["donor_base"] = digest
            envelope["proposed_artifact_digests"]["recipient_base"] = digest
            envelope["intent_sha256_untrusted"] = canonical_sha(intent)
            with self.assertRaises(AdapterOnlySourceMismatch):
                check_adapter_only_source_handoff(intent, envelope, bundle)

    def test_permission_smuggling_or_relabeling_is_rejected(self):
        for change in ("ready", "extra", "evidence"):
            intent, envelope, bundle = fixture()
            if change == "ready":
                envelope["admission"]["trainingday_ready"] = True
            elif change == "extra":
                envelope["admission"]["owner_authorized"] = True
            else:
                envelope["evidence_class"] = "HOST_VERIFIED"
            with self.assertRaises(AdapterOnlySourceMismatch):
                check_adapter_only_source_handoff(intent, envelope, bundle)

    def test_incomplete_or_extra_neurosync_intent_cannot_pass_source_join(self):
        for change in ("missing_rights", "missing_target_map", "extra_host_grant"):
            intent, envelope, bundle = fixture()
            if change == "missing_rights":
                del intent["data_rights"]
            elif change == "missing_target_map":
                del intent["adapter"]["target_map"]
            else:
                intent["host_authorized"] = True
            envelope["intent_sha256_untrusted"] = canonical_sha(intent)
            with self.subTest(change=change), self.assertRaises(AdapterOnlySourceMismatch):
                check_adapter_only_source_handoff(intent, envelope, bundle)

    def test_impossible_neurosync_value_semantics_cannot_pass_source_join(self):
        for change in ("unlicensed_rights", "wrong_optimizer_status", "invalid_target_rank",
                       "role_alias", "bad_dotted_module", "unhashable_provenance_status"):
            intent, envelope, bundle = fixture()
            if change == "unlicensed_rights":
                intent["data_rights"]["usage_scope"] = "unrestricted-commercial"
            elif change == "wrong_optimizer_status":
                intent["training_provenance"]["status"] = "RECEIPT_REFERENCED_UNVERIFIED"
            elif change == "invalid_target_rank":
                intent["adapter"]["target_map"]["rank"] = 0
                intent["adapter"]["target_map_sha256"] = canonical_sha(
                    intent["adapter"]["target_map"])
                envelope["target_map_sha256_untrusted"] = intent["adapter"]["target_map_sha256"]
            elif change == "bad_dotted_module":
                intent["adapter"]["target_map"]["lora_pairs"][0]["module"] = "base_model..q_proj"
                intent["adapter"]["target_map_sha256"] = canonical_sha(
                    intent["adapter"]["target_map"])
                envelope["target_map_sha256_untrusted"] = intent["adapter"]["target_map_sha256"]
            elif change == "unhashable_provenance_status":
                intent["training_provenance"]["status"] = []
            else:
                intent["base"]["donor_tokenizer_sha256"] = "1" * 64
                intent["base"]["recipient_tokenizer_sha256"] = "1" * 64
            envelope["intent_sha256_untrusted"] = canonical_sha(intent)
            with self.subTest(change=change), self.assertRaises(AdapterOnlySourceMismatch):
                check_adapter_only_source_handoff(intent, envelope, bundle)


if __name__ == "__main__":
    unittest.main()
