"""Read-only, fail-closed inventory of the TRAININGDAY integration chain.

This is an evidence-gap report, *not* an admission, certificate, GPU lease,
rights grant, evaluator, checkpoint validator, or physical launch controller.
It performs no downloads, training, conversion, checkpoint writes or serving.

The point of this command is to make missing integration gates visible in one
machine-readable place without laundering source files into runtime proof.
"""

from __future__ import annotations

import argparse
import json

from trainingday.official_public_base import (
    METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS,
    OFFICIAL_ID,
    OFFICIAL_REVISION,
    expected_source_record,
    validate_official_source_record,
)
from trainingday.physical_launch_authority import (
    PhysicalLaunchAuthorityUnavailable,
    require_physical_launch_authority,
)
from trainingday.pinned_base import (
    BASE_ID,
    BASE_REVISION,
    HEX64,
    TRUSTED_OFFICIAL_METADATA_PIN_ITEMS,
)


SCHEMA = "TRAININGDAY_INTEGRATION_PREFLIGHT_V1"


class UnexpectedAuthorityChange(RuntimeError):
    """The source hold no longer has its expected deny-only behavior."""


class SourceIdentityMismatch(ValueError):
    """Two supposed official-Qwen modules disagree about the exact base."""


def _step(name: str, status: str, evidence: str, *dependencies: str) -> dict:
    return {
        "name": name,
        "status": status,
        "evidence_class": evidence,
        "depends_on": list(dependencies),
    }


def _source_metadata_pin_shape_enrolled(pins: object) -> bool:
    """Mirror the production gate's exact name/HEX64 validation, offline.

    A syntactically enrolled source tuple remains unverified and is NEVER a
    trusted signature, on-disk byte check, or physical launch admission.
    """
    required = METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS
    if (type(pins) is not tuple or len(pins) != len(required)
            or not required or len(set(required)) != len(required)):
        return False
    seen: set[str] = set()
    for entry in pins:
        if type(entry) is not tuple or len(entry) != 2:
            return False
        name, digest = entry
        if (type(name) is not str or name not in required or name in seen
                or type(digest) is not str or HEX64.fullmatch(digest) is None):
            return False
        seen.add(name)
    return seen == set(required)


def report() -> dict:
    """Inspect current in-process *source* guarantees only.

    No user-supplied receipts or JSON files can promote this report to READY.
    Qualified host, data, artifact and runtime checks belong to independent
    authoritative observers outside this source-only utility.
    """
    validate_official_source_record(expected_source_record())
    if BASE_ID != OFFICIAL_ID or BASE_REVISION != OFFICIAL_REVISION:
        raise SourceIdentityMismatch("official source and byte verifier identity differ")
    metadata_complete = _source_metadata_pin_shape_enrolled(
        TRUSTED_OFFICIAL_METADATA_PIN_ITEMS)

    try:
        require_physical_launch_authority(operation="integration source preflight")
    except PhysicalLaunchAuthorityUnavailable:
        launcher_denies = True
    else:
        # A changed physical launcher must never silently become a preflight
        # approval: this utility is incapable of validating an actual grant.
        raise UnexpectedAuthorityChange(
            "physical launcher did not deny; independent requalification needed"
        )

    steps = [
        _step("official_source_record", "SOURCE_ONLY", "SOURCE_INSPECTION"),
        _step("authenticated_local_weight_bytes",
              "UNKNOWN" if metadata_complete else "HOLD",
              "NOT_A_RUNTIME_PROBE", "official_source_record"),
        _step("recovered_R16_adapter_provenance", "UNKNOWN",
              "NO_FRESH_CHECKPOINT_BYTE_OR_RECEIPT_PROBE"),
        _step("independent_artifact_restore", "UNKNOWN",
              "NO_DESTINATION_RESTORE_PROBE", "authenticated_local_weight_bytes",
              "recovered_R16_adapter_provenance"),
        _step("admitted_train_and_dev_rights", "UNKNOWN", "NOT_A_RIGHTS_GRANT"),
        _step("host_owned_process_gpu_and_replay_lease", "HOLD",
              "SOURCE_HARD_HOLD"),
        _step("real_base_model_inference", "UNKNOWN", "NO_LIVE_BASE_INFERENCE_PROBE",
              "authenticated_local_weight_bytes"),
        _step("reloaded_optimizer_checkpoint", "UNKNOWN", "NO_LIVE_TRAINING_PROBE",
              "authenticated_local_weight_bytes",
              "admitted_train_and_dev_rights",
              "host_owned_process_gpu_and_replay_lease"),
        _step("independent_protected_evaluation", "UNKNOWN", "NO_EVALUATOR_READBACK",
              "reloaded_optimizer_checkpoint"),
        _step("verified_adapter_gguf_conversion", "UNKNOWN",
              "NO_OFFICIAL_ADAPTER_CONVERSION_PROBE",
              "recovered_R16_adapter_provenance",
              "real_base_model_inference"),
        _step("trained_vera_conversation", "UNKNOWN", "NO_LIVE_SERVING_PROBE",
              "verified_adapter_gguf_conversion"),
    ]
    assert launcher_denies
    return {
        "schema": SCHEMA,
        "model_id": OFFICIAL_ID,
        "revision": OFFICIAL_REVISION,
        "source_metadata_pins_enrolled": metadata_complete,
        "ready": False,
        "physical_launch_attempted": False,
        "steps": steps,
        "warning": (
            "Source-only gap inventory. A stage status is NOT an authorization "
            "or an independent observation that it ran. Synthetic GGUF base "
            "tests do not qualify official adapter GGUF conversion."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)
    result = report()
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        for item in result["steps"]:
            print(f"{item['status']:11} {item['name']} [{item['evidence_class']}]")
        print("HOLD: source-only integration report; no physical authority issued")
    return 1  # Non-zero is intentional; no qualified release was demonstrated.


if __name__ == "__main__":
    raise SystemExit(main())
