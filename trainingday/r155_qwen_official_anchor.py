"""R155 published Qwen3.5-4B SHA-256 anchors, deliberately not a host grant.

These reference digests were read from the *official, revision-specific* Qwen
Hugging Face file pages on 2026-11-01. A self-reported R140 dictionary can
claim to match them, so assessing such a dictionary is never authentication.
For real local byte measurement, call ``audit_offline_against_official``;
R140 then streams the files, verifies safetensors headers/index, and returns
an inventory. Even that check cannot prove an immutable host filesystem,
exclusive GPU lease, model loading, tokenizer roundtrip, or GGUF conversion.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

from trainingday.r140_qwen_attestation import audit_snapshot


SCHEMA = "TRAININGDAY_R155_QWEN_PUBLIC_SHA256_ANCHOR_V1"
R140_SCHEMA = "VERA_TRAININGDAY_R140_QWEN_OFFLINE_SNAPSHOT_ATTESTATION_V1"
OFFICIAL_MODEL_ID = "Qwen/Qwen3.5-4B"
OFFICIAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_ANCHORS = (
    ("model.safetensors-00001-of-00002.safetensors",
     "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"),
    ("model.safetensors-00002-of-00002.safetensors",
     "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"),
    ("tokenizer.json",
     "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
)
OFFICIAL_SHA256: Mapping[str, str] = dict(_ANCHORS)
_HEX64 = re.compile(r"[a-f0-9]{64}\Z")
_R140_FALSE_FIELDS = (
    "physical_tokenizer_verified", "runtime_loader_verified",
    "actual_model_tensors_materialized", "finite_forward_loss_verified",
    "external_pin_provenance_verified", "host_immutable_custody_verified",
    "exclusive_cuda_lease_verified", "authorizes_training",
)


class OfficialAnchorError(ValueError):
    """The source inventory cannot be reconciled with official pinned SHA-256."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialAnchorError(message)


def assess_r140_report(report: object) -> dict:
    """Compare R140 *claims* to official public hashes, never authorize use.

    This function intentionally accepts a pure dictionary so callers can
    review a detached R140 report. A forged dictionary with the correct
    hash strings will pass this comparison. Its result is therefore an
    untrusted source-claim consistency check, not a signature or host probe.
    """
    _require(type(report) is dict, "exact R140 inventory mapping required")
    _require(report.get("schema") == R140_SCHEMA,
             "unknown R140 inventory schema")
    _require(report.get("status") == "SOURCE_SNAPSHOT_INVENTORY_VALID_RUNTIME_HOLD"
             and report.get("evidence_class") == "LOCAL_SOURCE_AND_FILE_INVENTORY_ONLY",
             "only R140 unqualified source inventory accepted")
    _require(report.get("model_id") == OFFICIAL_MODEL_ID
             and report.get("revision") == OFFICIAL_REVISION,
             "official Qwen3.5 model/revision differs")
    _require(type(report.get("safetensors_shard_count")) is int
             and report["safetensors_shard_count"] == 2
             and type(report.get("safetensors_tensor_count")) is int
             and report["safetensors_tensor_count"] > 0,
             "two indexed nonempty safetensors shards required")
    for name in _R140_FALSE_FIELDS:
        _require(report.get(name) is False,
                 "source report asserted unauthenticated authority: " + name)

    observed = report.get("observed_file_sha256")
    _require(type(observed) is dict and all(type(key) is str
             and type(value) is str and _HEX64.fullmatch(value) is not None
             for key, value in observed.items()),
             "malformed observed source file digest inventory")
    for filename, official in _ANCHORS:
        _require(observed.get(filename) == official,
                 "digest does not match published upstream SHA256: " + filename)
    return {
        "schema": SCHEMA,
        "status": "UPSTREAM_DIGEST_CLAIM_MATCH_HOST_HOLD",
        "evidence_class": "PUBLIC_REFERENCE_VS_CALLER_SUPPLIED_R140_REPORT_ONLY",
        "model_id": OFFICIAL_MODEL_ID,
        "revision": OFFICIAL_REVISION,
        "public_reference_sha256": dict(_ANCHORS),
        "authenticates_host_snapshot": False,
        "authenticates_weight_loader": False,
        "authenticates_tokenizer_runtime": False,
        "authorizes_training": False,
        "authorizes_gguf_release": False,
        "authorizes_host_execution": False,
    }


def audit_offline_against_official(base_binding_path: Path, *,
                                   expected_binding_sha256: str,
                                   expected_config_sha256: str,
                                   expected_tokenizer_sha256: str,
                                   source_root: Path) -> dict:
    """Check actual local files with R140, then anchor three public digests.

    R140 is a bounded streaming read of all declared snapshot files, including
    the two large safetensors shards. This wrapper never downloads weights or
    promotes its result above the host-custody HARD HOLD.
    """
    report = audit_snapshot(
        base_binding_path,
        expected_binding_sha256=expected_binding_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        source_root=source_root,
    )
    return assess_r140_report(report)
