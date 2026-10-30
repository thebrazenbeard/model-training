"""R25 public/synthetic serving-custody negatives, no model or host effects."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from trainingday.r25_serving_envelope import (
    MODEL_ARCH, MODEL_ID, SCHEMA, SOURCE_FORMAT, TARGET_FORMAT,
    assess_serving_candidate, simulate_reversible_cutover,
    tensor_manifest_digest,
)


def h(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def fixtures():
    tensors = [
        {"name": "model.layers.0.attn.lora_A.weight", "shape": [4, 16], "dtype": "float32"},
        {"name": "model.layers.0.attn.lora_B.weight", "shape": [16, 4], "dtype": "float32"},
    ]
    pinned = {
        "base_model_id": MODEL_ID, "base_architecture": MODEL_ARCH,
        "base_revision": "a" * 40, "base_binding_sha256": h("base-bytes"),
        "tokenizer_binding_sha256": h("tokenizer"),
        "corpus_manifest_sha256": h("public-corpus"),
        "adapter_bytes_sha256": h("adapter-bytes"),
        "adapter_tensor_manifest_sha256": tensor_manifest_digest(tensors),
    }
    candidate = {
        **pinned, "tensor_manifest": tensors,
        "quantizer": {
            "source_format": SOURCE_FORMAT, "target_format": TARGET_FORMAT,
            "source_base_binding_sha256": pinned["base_binding_sha256"],
            "source_tokenizer_binding_sha256": pinned["tokenizer_binding_sha256"],
            "source_adapter_sha256": pinned["adapter_bytes_sha256"],
            "converter_binary_sha256": h("converter-binary"),
            "converter_args_sha256": h("converter-arguments"),
            "target_model_sha256": h("synthetic-GGUF-output"),
            "transformation_receipt_sha256": h("fake-transform-receipt"),
        },
        "fresh_process_reload": None,
    }
    return pinned, candidate


def forged_metadata_pin(tensors):
    """Attacker-controlled metadata digest; not proof of actual tensor bytes."""
    raw = json.dumps(tensors, sort_keys=True, ensure_ascii=True, allow_nan=False,
                     separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


class R25ServingCustodyTests(unittest.TestCase):
    def test_cross_module_lora_pair_rejected_even_when_metadata_rehashed(self):
        pinned, candidate = fixtures()
        candidate["tensor_manifest"][1]["name"] = "model.layers.1.attn.lora_B.weight"
        forged = forged_metadata_pin(candidate["tensor_manifest"])
        pinned["adapter_tensor_manifest_sha256"] = forged
        candidate["adapter_tensor_manifest_sha256"] = forged
        result = assess_serving_candidate(pinned, candidate)
        self.assertEqual(result["reason"], "MALFORMED_TENSOR_MANIFEST")
        self.assertFalse(result["source_bindings_match"])
        self.assertFalse(result["authorizes_deployment"])

    def test_incompatible_lora_ab_rank_rejected_after_metadata_repin(self):
        pinned, candidate = fixtures()
        candidate["tensor_manifest"][1]["shape"] = [16, 5]
        forged = forged_metadata_pin(candidate["tensor_manifest"])
        pinned["adapter_tensor_manifest_sha256"] = forged
        candidate["adapter_tensor_manifest_sha256"] = forged
        self.assertEqual(assess_serving_candidate(pinned, candidate)["reason"],
                         "MALFORMED_TENSOR_MANIFEST")

    def test_astronomical_fp32_lora_tensor_bytes_rejected(self):
        pinned, candidate = fixtures()
        for tensor in candidate["tensor_manifest"]:
            tensor["shape"] = [65536, 65536]
        forged = forged_metadata_pin(candidate["tensor_manifest"])
        pinned["adapter_tensor_manifest_sha256"] = forged
        candidate["adapter_tensor_manifest_sha256"] = forged
        result = assess_serving_candidate(pinned, candidate)
        self.assertEqual(result["reason"], "MALFORMED_TENSOR_MANIFEST")
        self.assertFalse(result["source_bindings_match"])

    def test_matching_lora_adapter_namespace_and_multiple_pairs_are_accepted(self):
        pinned, candidate = fixtures()
        first = copy.deepcopy(candidate["tensor_manifest"])
        for tensor in first:
            tensor["name"] = tensor["name"].replace(".weight", ".default.weight")
        second = copy.deepcopy(first)
        for tensor in second:
            tensor["name"] = tensor["name"].replace("layers.0", "layers.1")
        candidate["tensor_manifest"] = first + second
        new_sha = tensor_manifest_digest(candidate["tensor_manifest"])
        pinned["adapter_tensor_manifest_sha256"] = new_sha
        candidate["adapter_tensor_manifest_sha256"] = new_sha
        result = assess_serving_candidate(pinned, candidate)
        self.assertTrue(result["source_bindings_match"])
        self.assertFalse(result["authorizes_deployment"])

    def test_cross_adapter_namespace_cannot_be_paired_by_module_only(self):
        pinned, candidate = fixtures()
        candidate["tensor_manifest"][0]["name"] = (
            "model.layers.0.attn.lora_A.default.weight")
        candidate["tensor_manifest"][1]["name"] = (
            "model.layers.0.attn.lora_B.other.weight")
        forged = forged_metadata_pin(candidate["tensor_manifest"])
        pinned["adapter_tensor_manifest_sha256"] = forged
        candidate["adapter_tensor_manifest_sha256"] = forged
        self.assertEqual(assess_serving_candidate(pinned, candidate)["reason"],
                         "MALFORMED_TENSOR_MANIFEST")

    def test_cumulative_fp32_ceiling_rejects_many_individually_bounded_modules(self):
        tensors = []
        for module in ("model.layers.0.attn", "model.layers.1.attn",
                       "model.layers.2.attn"):
            tensors.extend([
                {"name": f"{module}.lora_A.weight", "shape": [64, 65536],
                 "dtype": "float32"},
                {"name": f"{module}.lora_B.weight", "shape": [65536, 64],
                 "dtype": "float32"},
            ])
        with self.assertRaisesRegex(ValueError, "synthetic admission ceiling"):
            tensor_manifest_digest(tensors)

    def test_structurally_matched_candidate_remains_hold(self):
        pinned, candidate = fixtures()
        result = assess_serving_candidate(pinned, candidate)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["reason"], "DECLARATIONS_CONSISTENT_NOT_INDEPENDENTLY_ATTESTED")
        self.assertTrue(result["source_bindings_match"])
        self.assertRegex(result["candidate_sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(result["fresh_process_reference_parity_verified"])
        self.assertFalse(result["quantizer_transform_physically_verified"])
        self.assertFalse(result["authorizes_deployment"])
        self.assertFalse(result["authorizes_model_promotion"])

    def test_pin_only_content_canonicalization_is_stable(self):
        pinned, candidate = fixtures()
        digest = assess_serving_candidate(pinned, candidate)["candidate_sha256"]
        reordered = dict(reversed(list(candidate.items())))
        reordered["quantizer"] = dict(reversed(list(candidate["quantizer"].items())))
        self.assertEqual(assess_serving_candidate(pinned, reordered)["candidate_sha256"], digest)

    def test_base_architecture_and_derivative_model_refused(self):
        pinned, candidate = fixtures()
        for field, bad in (("base_model_id", "rodrigomt/Qwen3.5-4B-Uncensored-Aggressive"),
                           ("base_architecture", "GptOssForCausalLM"),
                           ("base_revision", "not-pinned")):
            with self.subTest(field=field):
                changed = copy.deepcopy(pinned)
                changed[field] = bad
                report = assess_serving_candidate(changed, candidate)
                self.assertEqual(report["reason"], "UNQUALIFIED_BASE_OR_PIN")

    def test_candidate_substitution_refused_even_if_self_rehashed(self):
        pinned, candidate = fixtures()
        for field in ("base_model_id", "base_architecture", "base_revision",
                      "base_binding_sha256", "tokenizer_binding_sha256",
                      "corpus_manifest_sha256", "adapter_bytes_sha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(candidate)
                changed[field] = h("substitute")
                report = assess_serving_candidate(pinned, changed)
                self.assertEqual(report["reason"], "PINNED_SOURCE_MISMATCH")

    def test_tampered_tensor_metadata_or_shape_rejected(self):
        pinned, candidate = fixtures()
        for mutation in (lambda x: x[0].update(shape=[4, 15]),
                         lambda x: x[0].update(dtype="float16"),
                         lambda x: x[0].update(name="model.layers.0.attn.weight"),
                         lambda x: x[1].update(name=x[0]["name"]),
                         lambda x: x[1].update(shape=[True, 4])):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(candidate)
                mutation(changed["tensor_manifest"])
                self.assertIn(assess_serving_candidate(pinned, changed)["reason"],
                              {"MALFORMED_TENSOR_MANIFEST", "ADAPTER_TENSOR_MANIFEST_MISMATCH"})

    def test_tensor_manifest_repinning_without_pinned_adapter_match_refused(self):
        pinned, candidate = fixtures()
        candidate["tensor_manifest"][0]["shape"] = [4, 17]
        self.assertEqual(assess_serving_candidate(pinned, candidate)["reason"],
                         "ADAPTER_TENSOR_MANIFEST_MISMATCH")

    def test_quantizer_source_mismatch_refused(self):
        pinned, candidate = fixtures()
        for field in ("source_base_binding_sha256", "source_tokenizer_binding_sha256",
                      "source_adapter_sha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(candidate)
                changed["quantizer"][field] = h("other-model")
                self.assertEqual(assess_serving_candidate(pinned, changed)["reason"],
                                 "QUANTIZATION_SOURCE_MISMATCH")

    def test_wrong_format_or_missing_transform_digest_refused(self):
        pinned, candidate = fixtures()
        for field, bad in (("source_format", "FP16_UNPINNED"),
                           ("target_format", "GGUF_Q4_UNQUALIFIED")):
            with self.subTest(field=field):
                changed = copy.deepcopy(candidate)
                changed["quantizer"][field] = bad
                self.assertEqual(assess_serving_candidate(pinned, changed)["reason"],
                                 "UNQUALIFIED_QUANTIZATION_FORMAT")
        for field in ("converter_binary_sha256", "converter_args_sha256",
                      "target_model_sha256", "transformation_receipt_sha256"):
            changed = copy.deepcopy(candidate)
            changed["quantizer"][field] = "MISSING"
            self.assertEqual(assess_serving_candidate(pinned, changed)["reason"],
                             "MALFORMED_QUANTIZATION_DIGEST")

    def test_caller_declared_reload_match_is_not_fresh_process_proof(self):
        pinned, candidate = fixtures()
        candidate["fresh_process_reload"] = {
            "observed_base_binding_sha256": pinned["base_binding_sha256"],
            "observed_tokenizer_binding_sha256": pinned["tokenizer_binding_sha256"],
            "observed_adapter_bytes_sha256": pinned["adapter_bytes_sha256"],
            "observed_tensor_manifest_sha256": pinned["adapter_tensor_manifest_sha256"],
            "fresh_process_nonce": "fake-process-001",
        }
        result = assess_serving_candidate(pinned, candidate)
        self.assertTrue(result["claimed_reload_fields_match"])
        self.assertFalse(result["fresh_process_reference_parity_verified"])
        self.assertFalse(result["authorizes_deployment"])

    def test_forged_or_malformed_reload_claim_refused(self):
        pinned, candidate = fixtures()
        candidate["fresh_process_reload"] = {
            "observed_base_binding_sha256": pinned["base_binding_sha256"],
            "observed_tokenizer_binding_sha256": pinned["tokenizer_binding_sha256"],
            "observed_adapter_bytes_sha256": h("forged"),
            "observed_tensor_manifest_sha256": pinned["adapter_tensor_manifest_sha256"],
            "fresh_process_nonce": "fake-process-001",
        }
        self.assertEqual(assess_serving_candidate(pinned, candidate)["reason"],
                         "RELOAD_CLAIM_MISMATCH")
        candidate["fresh_process_reload"]["fresh_process_nonce"] = "../escape"
        self.assertEqual(assess_serving_candidate(pinned, candidate)["reason"],
                         "MALFORMED_RELOAD_CLAIM")

    def test_source_cannot_encode_arbitrary_paths_or_private_bank_data(self):
        pinned, candidate = fixtures()
        candidate["raw_bank_path"] = "C:/hidden/private.jsonl"
        result = assess_serving_candidate(pinned, candidate)
        self.assertEqual(result["reason"], "INVALID_ENVELOPE")
        self.assertFalse(result["authorizes_deployment"])
        self.assertNotIn("raw_bank_path", json.dumps(result))

    def test_untyped_and_overflow_envelopes_fail_closed(self):
        pinned, candidate = fixtures()
        for bad in (None, 1, [], {"anything": 1}):
            with self.subTest(bad=bad):
                self.assertEqual(assess_serving_candidate(bad, candidate)["reason"],
                                 "INVALID_ENVELOPE")
                self.assertEqual(assess_serving_candidate(pinned, bad)["reason"],
                                 "INVALID_ENVELOPE")
        changed = copy.deepcopy(candidate)
        changed["tensor_manifest"][0]["shape"] = [float("inf"), 16]
        self.assertEqual(assess_serving_candidate(pinned, changed)["reason"],
                         "MALFORMED_TENSOR_MANIFEST")

    def test_each_synthetic_cutover_failure_restores_previous_digest(self):
        prev, new = h("existing-active"), h("new-candidate")
        for fail_at in ("STAGE", "VERIFY", "SWAP", "POSTCHECK", "COMMIT"):
            with self.subTest(fail_at=fail_at):
                result = simulate_reversible_cutover(prev, new, fail_at=fail_at)
                self.assertEqual(result["simulated_active_sha256"], prev)
                self.assertTrue(result["rolled_back"])
                self.assertTrue(result["previous_sha256_retained"])
                self.assertTrue(result["production_active_untouched"])
                self.assertFalse(result["authorizes_deployment"])
                self.assertIn("RESTORE_PREVIOUS", result["trace"])

    def test_synthetic_success_never_changes_real_serving_model(self):
        prev, new = h("existing-active"), h("new-candidate")
        result = simulate_reversible_cutover(prev, new)
        self.assertEqual(result["simulated_active_sha256"], new)
        self.assertTrue(result["previous_sha256_retained"])
        self.assertTrue(result["production_active_untouched"])
        self.assertFalse(result["authorizes_deployment"])

    def test_invalid_cutover_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            simulate_reversible_cutover(h("same"), h("same"))
        with self.assertRaises(ValueError):
            simulate_reversible_cutover(h("one"), "BAD")
        with self.assertRaises(ValueError):
            simulate_reversible_cutover(h("one"), h("two"), fail_at="PRODUCTION_RESTART")


if __name__ == "__main__":
    unittest.main()
