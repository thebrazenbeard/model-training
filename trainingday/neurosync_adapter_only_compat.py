"""Source-only Neuro-Sync PEFT handoff for the *official* TRAININGDAY line.

This does not import/execute Neuro-Sync, read weights, enroll trust pins, grant
rights or qualify physical work. It checks a source metadata/proposal join.
"""

import hashlib
import json
import math
import re

from .official_public_base import OFFICIAL_ID, OFFICIAL_REVISION, SHARDS


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TARGET = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\Z")
_ARTIFACT_ROLES = (
    "donor_base", "recipient_base", "adapter_weights", "adapter_config",
    "train_manifest", "sealed_holdout_manifest", "shuffled_control_manifest",
)
_ADMISSION = {
    "host_artifact_verification": "UNVERIFIED",
    "dataset_training_rights": "UNVERIFIED",
    "independent_target_layer_compatibility": "UNVERIFIED",
    "optimizer_and_adapter_restore": "UNVERIFIED",
    "sealed_independent_evaluation": "UNVERIFIED",
    "cold_recipient_reload": "UNVERIFIED",
    "serving_pid_model_binding": "UNVERIFIED",
    "physical_launch_authority": "HOLD",
    "trainingday_ready": False,
}


class AdapterOnlySourceMismatch(ValueError):
    """A proposed adapter source does not match the pinned official lineage."""


def _require(value):
    if not value:
        raise AdapterOnlySourceMismatch("Neuro-Sync adapter source handoff rejected")


def _dict_keys(value, keys):
    _require(type(value) is dict and set(value) == set(keys))
    return value


def _digest(value):
    _require(type(value) is str and _SHA.fullmatch(value) is not None)
    return value


def _canonical_sha(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise AdapterOnlySourceMismatch("noncanonical untrusted proposal") from exc
    return hashlib.sha256(encoded).hexdigest()


def _geometry_sha(target):
    """Mirror the versioned Neuro-Sync V1 *declared* target map fingerprint.

    Real PEFT dimensions/receiver layers still require independent host probes.
    """
    target = _dict_keys(target, {"rank", "lora_alpha", "target_modules", "lora_pairs"})
    rank, alpha = target["rank"], target["lora_alpha"]
    _require(type(rank) is int and 0 < rank <= 16384 and
             type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0)
    modules = target["target_modules"]
    _require(type(modules) is list and modules and
             all(type(name) is str and _TARGET.fullmatch(name) for name in modules) and
             len(modules) == len(set(modules)))
    pairs = target["lora_pairs"]
    _require(type(pairs) is list and pairs)
    observed = set()
    seen = set()
    for pair in pairs:
        pair = _dict_keys(pair, {"module", "rank", "input_features", "output_features",
                                 "dtype", "target"})
        name = pair["module"]
        _require(type(name) is str and _TARGET.fullmatch(name) and name not in seen)
        seen.add(name)
        _require(type(pair["rank"]) is int and pair["rank"] == rank and
                 type(pair["input_features"]) is int and pair["input_features"] > 0 and
                 type(pair["output_features"]) is int and pair["output_features"] > 0 and
                 type(pair["dtype"]) is str and pair["dtype"] in
                 {"F16", "BF16", "F32", "F64"} and
                 type(pair["target"]) is str and pair["target"] in modules)
        matching = [module for module in modules
                    if name == module or name.endswith("." + module)]
        _require(len(matching) == 1 and matching[0] == pair["target"])
        observed.add(pair["target"])
    _require(observed == set(modules))
    return _canonical_sha({"rank": rank, "lora_alpha": alpha,
                          "target_modules": sorted(modules),
                          "lora_pairs": sorted(pairs, key=lambda item: item["module"])})


def _typed_equals(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _typed_equals(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _typed_equals(a, b) for a, b in zip(left, right))
    return left == right


def _check_intent_contract_shape(intent):
    """Require complete external V1 field topology; this is NOT PEFT validation.

    Neuro-Sync owns geometric/tensor semantics. Here unknown/missing fields
    cannot be promoted into a source-aligned handoff by rehashing the JSON.
    """
    _dict_keys(intent, {"schema", "method", "base", "adapter", "input_artifacts",
                        "training_provenance", "data_rights", "evaluation", "budget"})
    _dict_keys(intent["base"], {"base_revision", "donor_base_weights_sha256",
                                "recipient_base_weights_sha256", "donor_tokenizer_sha256",
                                "recipient_tokenizer_sha256", "donor_config_sha256",
                                "recipient_config_sha256"})
    _dict_keys(intent["adapter"], {"weights_sha256", "config_sha256",
                                   "target_map", "target_map_sha256"})
    _dict_keys(intent["training_provenance"], {"source_revision_sha256",
                                               "optimizer_receipt_sha256", "status"})
    _dict_keys(intent["data_rights"], {"record_sha256", "rights_holder_label",
                                      "usage_scope"})
    _dict_keys(intent["evaluation"], {"donor_disconnected", "cold_recipient_reload",
                                     "baseline_arm", "negative_control_arm",
                                     "evaluator_custody"})
    _dict_keys(intent["budget"], {"max_external_cost_usd", "max_gpu_minutes"})
    _require(_digest(intent["adapter"]["target_map_sha256"]) ==
             _geometry_sha(intent["adapter"]["target_map"]))
    _require(type(intent["budget"]["max_external_cost_usd"]) is int and
             intent["budget"]["max_external_cost_usd"] == 0 and
             type(intent["budget"]["max_gpu_minutes"]) is int and
             0 < intent["budget"]["max_gpu_minutes"] <= 1440)
    _require(type(intent["evaluation"]["donor_disconnected"]) is bool and
             intent["evaluation"]["donor_disconnected"] and
             type(intent["evaluation"]["cold_recipient_reload"]) is bool and
             intent["evaluation"]["cold_recipient_reload"] and
             intent["evaluation"]["baseline_arm"] == "untouched-recipient" and
             intent["evaluation"]["negative_control_arm"] == "shuffled-donor" and
             intent["evaluation"]["evaluator_custody"] == "independent-host-proposed")
    provenance = intent["training_provenance"]
    _digest(provenance["source_revision_sha256"])
    _require(type(provenance["status"]) is str)
    _require((provenance["status"] == "UNVERIFIED" and
              provenance["optimizer_receipt_sha256"] is None) or
             (provenance["status"] == "RECEIPT_REFERENCED_UNVERIFIED" and
              _digest(provenance["optimizer_receipt_sha256"])))
    rights = intent["data_rights"]
    _digest(rights["record_sha256"])
    label = rights["rights_holder_label"]
    _require(type(label) is str and 1 <= len(label) <= 128 and
             all(ch.isprintable() and not 0xD800 <= ord(ch) <= 0xDFFF for ch in label))
    _require(type(rights["usage_scope"]) is str and rights["usage_scope"] in
             {"local-experiment-only", "explicit-permission-proposed"})


def expected_official_source_bundle():
    """Pinned metadata digest, NOT a SHA-256 of concatenated model weights."""
    result = {
        "schema": "TRAININGDAY_OFFICIAL_SOURCE_SHARD_BUNDLE_V1",
        "repository": OFFICIAL_ID,
        "revision": OFFICIAL_REVISION,
        "ordered_shards": [{"path": path, "size_bytes": size,
                            "sha256_lfs_oid": digest}
                           for path, size, digest in SHARDS],
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    result["manifest_sha256"] = hashlib.sha256(
        b"TRAININGDAY_OFFICIAL_SOURCE_SHARD_BUNDLE_V1\x00" + payload).hexdigest()
    return result


def check_adapter_only_source_handoff(intent, envelope, bundle):
    """Check metadata/envelope consistency, *never* return launch readiness.

    `base_weights_sha256` is an opaque unverified bytes claim in Neuro-Sync.
    Official Qwen is TWO shards; a shard hash or this metadata bundle hash is
    never a substitute for a verified aggregate model-byte identity. Without
    host byte evidence the proposed full-base claim remains UNVERIFIED.
    """
    _require(_typed_equals(bundle, expected_official_source_bundle()))
    _check_intent_contract_shape(intent)
    _require(
             intent.get("schema") == "NEUROSYNC_PEFT_ADAPTER_ONLY_INTENT_V1" and
             intent.get("method") == "peft-adapter-only")
    base = intent.get("base")
    _require(type(base) is dict and base.get("base_revision") == OFFICIAL_REVISION)
    donor = _digest(base.get("donor_base_weights_sha256"))
    _require(donor == _digest(base.get("recipient_base_weights_sha256")))
    tokenizer = _digest(base.get("donor_tokenizer_sha256"))
    config_base = _digest(base.get("donor_config_sha256"))
    _require(tokenizer == _digest(base.get("recipient_tokenizer_sha256")) and
             config_base == _digest(base.get("recipient_config_sha256")))
    _require(donor not in {item[2] for item in SHARDS} |
             {bundle["manifest_sha256"]})

    adapter = intent.get("adapter")
    _require(type(adapter) is dict)
    weights = _digest(adapter.get("weights_sha256"))
    config = _digest(adapter.get("config_sha256"))
    target_map = _digest(adapter.get("target_map_sha256"))
    _require(len({donor, tokenizer, config_base, weights, config}) == 5)
    artifacts = _dict_keys(intent.get("input_artifacts"), _ARTIFACT_ROLES)
    artifact_digests = {}
    observed_size_by_sha = {}
    for name in _ARTIFACT_ROLES:
        item = _dict_keys(artifacts[name], {"sha256", "size_bytes"})
        _require(type(item["size_bytes"]) is int and item["size_bytes"] > 0)
        artifact_digests[name] = _digest(item.get("sha256"))
        digest, size = artifact_digests[name], item["size_bytes"]
        _require(digest not in observed_size_by_sha or observed_size_by_sha[digest] == size)
        observed_size_by_sha[digest] = size
    _require(artifact_digests["donor_base"] == donor and
             artifact_digests["recipient_base"] == donor and
             artifact_digests["adapter_weights"] == weights and
             artifact_digests["adapter_config"] == config)
    _require(len({artifact_digests[name] for name in (
        "train_manifest", "sealed_holdout_manifest", "shuffled_control_manifest")}) == 3)
    manifests = {artifact_digests[name] for name in (
        "train_manifest", "sealed_holdout_manifest", "shuffled_control_manifest")}
    _require(not manifests.intersection({donor, tokenizer, config_base, weights, config}))
    provenance = intent.get("training_provenance")
    _require(type(provenance) is dict)
    receipt = provenance.get("optimizer_receipt_sha256")
    _require(receipt is None or _digest(receipt) == receipt)

    envelope = _dict_keys(envelope, {
        "schema", "method", "evidence_class", "intent_sha256_untrusted",
        "proposed_base_weights_sha256", "proposed_artifact_digests",
        "target_map_sha256_untrusted", "optimizer_receipt_sha256_untrusted",
        "admission", "interpretation",
    })
    _require(envelope["schema"] ==
             "NEUROSYNC_TRAININGDAY_PEFT_ADAPTER_ONLY_ENVELOPE_V1" and
             envelope["method"] == "peft-adapter-only" and
             envelope["evidence_class"] == "UNTRUSTED_PEFT_ADAPTER_ONLY_PROPOSAL" and
             _digest(envelope["intent_sha256_untrusted"]) == _canonical_sha(intent) and
             envelope["proposed_base_weights_sha256"] == donor and
             _typed_equals(envelope["proposed_artifact_digests"], artifact_digests) and
             envelope["target_map_sha256_untrusted"] == target_map and
             envelope["optimizer_receipt_sha256_untrusted"] == receipt and
             type(envelope["interpretation"]) is str and
             _typed_equals(envelope["admission"], _ADMISSION))

    return {
        "schema": "TRAININGDAY_NEUROSYNC_PEFT_SOURCE_JOIN_V1",
        "source_bundle_aligned": True,
        "source_bundle_manifest_sha256": bundle["manifest_sha256"],
        "source_evidence_class": "SOURCE_PIN_METADATA_ONLY",
        "neurosync_intent_sha256_untrusted": envelope["intent_sha256_untrusted"],
        "base_weight_bytes_verified": False,
        "training_data_rights_verified": False,
        "adapter_loaded_and_evaluated": False,
        "trainingday_ready": False,
        "physical_launch_authority": "HOLD",
    }
