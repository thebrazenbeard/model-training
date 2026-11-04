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


def report() -> dict:
    """Inspect current in-process *source* guarantees only.

    No user-supplied receipts or JSON files can promote this report to READY.
    Qualified host, data, artifact and runtime checks belong to independent
    authoritative observers outside this source-only utility.
    """
    validate_official_source_record(expected_source_record())
    if BASE_ID != OFFICIAL_ID or BASE_REVISION != OFFICIAL_REVISION:
        raise SourceIdentityMismatch("official source and byte verifier identity differ")
    required = set(METADATA_FILES_NEEDING_TRUSTED_BYTE_PINS)
    enrolled = dict(TRUSTED_OFFICIAL_METADATA_PIN_ITEMS)
    metadata_complete = (
        len(enrolled) == len(TRUSTED_OFFICIAL_METADATA_PIN_ITEMS)
        and set(enrolled) == required
        and len(required) > 0
    )

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
        _step("admitted_train_and_dev_rights", "UNKNOWN", "NOT_A_RIGHTS_GRANT",
              "authenticated_local_weight_bytes"),
        _step("host_owned_process_gpu_and_replay_lease", "HOLD",
              "SOURCE_HARD_HOLD", "admitted_train_and_dev_rights"),
        _step("reloaded_optimizer_checkpoint", "UNKNOWN", "NO_LIVE_TRAINING_PROBE",
              "host_owned_process_gpu_and_replay_lease"),
        _step("independent_protected_evaluation", "UNKNOWN", "NO_EVALUATOR_READBACK",
              "reloaded_optimizer_checkpoint"),
        _step("verified_adapter_gguf_conversion", "UNKNOWN",
              "NO_OFFICIAL_ADAPTER_CONVERSION_PROBE",
              "reloaded_optimizer_checkpoint"),
        _step("real_local_conversation", "UNKNOWN", "NO_LIVE_SERVING_PROBE",
              "independent_protected_evaluation",
              "verified_adapter_gguf_conversion"),
        _step("independent_artifact_recovery", "UNKNOWN",
              "NO_DESTINATION_RESTORE_PROBE", "real_local_conversation"),
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
