"""CPU-only R20 precomparison evidence check; never a GPU lease or launch.

This checks the *shape and internal consistency* of declared R18/R19 evidence.
External host/corpus/source custody must still be independently verified by the
operator; booleans, hashes and device names supplied here are not attestation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

from trainingday.r18_benchmark import canonical_sha256, validate_plan


SCHEMA = "VERA_TRAININGDAY_R20_PRECOMPARISON_V1"
OBJECTIVE = "SUPERVISED_TOKEN_WEIGHTED_MEAN_BOTH_ARMS"
CHECKPOINT_POLICY = "END_ONLY_NO_RESUME"
ABORT_POLICY = "FAIL_CLOSED_ON_OVERRUN_OR_SUSPEND"
MAX_OBSERVATION_AGE_SECONDS = 60
MAX_LEASE_REMAINING_SECONDS = 600
# Exit 0 must never be mistaken by a shell runner for GPU launch authority.
# Exit 2 means declarations are incomplete; exit 3 means declarations are
# internally consistent but all external owner/host authority is still absent.
EXIT_INCOMPLETE_DECLARATIONS = 2
EXIT_COMPLETE_DECLARATIONS_NOT_AUTHORIZED = 3


def _part(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _sha(value: object, width: int = 64) -> bool:
    return (type(value) is str and len(value) == width
            and all(ch in "0123456789abcdef" for ch in value))


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _recent(timestamp: object, now_unix: float) -> bool:
    return (_number(timestamp) and timestamp <= now_unix
            and now_unix - timestamp <= MAX_OBSERVATION_AGE_SECONDS)


def assess_precomparison(plan: Mapping, evidence: object, *, now_unix: float) -> dict:
    """Reject absent/mismatched proof fields; NEVER authorize optimizer execution.

    Every status is a *source-level* result. Even an internally complete bundle
    requires separately authenticated live host leases and an operator review.
    """
    plan_digest = validate_plan(plan)
    if not _number(now_unix) or now_unix <= 0:
        raise ValueError("now_unix must be a current positive timestamp")
    e = _part(evidence)
    source, data = _part(e.get("source")), _part(e.get("data"))
    initial, host = _part(e.get("initial")), _part(e.get("host"))
    experiment = _part(e.get("experiment"))
    blockers = []

    if (e.get("schema") != SCHEMA or e.get("plan_sha256") != plan_digest
            or not _sha(e.get("source_head"), 40)):
        blockers.append("PLAN_OR_SOURCE_UNBOUND")

    if (source.get("verified_source_head") != e.get("source_head")
            or source.get("objective_contract") != OBJECTIVE
            or source.get("objective_parity_tests_pass") is not True):
        blockers.append("OBJECTIVE_PARITY_UNVERIFIED")
    if (source.get("checkpoint_policy") != CHECKPOINT_POLICY
            or source.get("checkpoint_failure_tests_pass") is not True):
        blockers.append("CHECKPOINT_POLICY_UNVERIFIED")

    if (data.get("corpus_manifest_sha256") != plan["corpus_manifest_sha256"]
            or data.get("selected_id_sha256") != plan["selected_id_sha256"]
            or data.get("train_content_sha256") != plan["train_content_sha256"]
            or data.get("eval_access_before_arm_freeze") is not False):
        blockers.append("TRAIN_SELECTION_UNBOUND")

    if (not _sha(initial.get("mezo_adapter_sha256"))
            or initial.get("mezo_adapter_sha256") != initial.get("qlora_adapter_sha256")
            or not _sha(initial.get("mezo_trainable_digest"))
            or initial.get("mezo_trainable_digest") != initial.get("qlora_trainable_digest")
            or initial.get("mezo_base_binding_sha256") != plan["base_binding_sha256"]
            or initial.get("qlora_base_binding_sha256") != plan["base_binding_sha256"]):
        blockers.append("INITIAL_ADAPTER_MISMATCH")

    lease_valid = (host.get("device_id") == "Lappy"
                   and host.get("resource_id") == "CUDA0"
                   and type(host.get("lease_id")) is str
                   and bool(host.get("lease_id").strip())
                   and host.get("exclusive_owner_lease") is True
                   and _recent(host.get("lease_observed_at_unix"), now_unix)
                   and _number(host.get("lease_expires_at_unix"))
                   and now_unix < host.get("lease_expires_at_unix")
                   <= now_unix + MAX_LEASE_REMAINING_SECONDS)
    if not lease_valid:
        blockers.append("HOST_LEASE_UNVERIFIED")
    if (not _recent(host.get("awake_observed_at_unix"), now_unix)
            or host.get("lid_open") is not True
            or host.get("suspend_guard_active") is not True
            or host.get("standby_since_plan") is not False):
        blockers.append("HOST_POWER_UNSAFE")

    if (not _sha(experiment.get("fit_mezo_sha256"))
            or not _sha(experiment.get("fit_qlora_sha256"))
            or experiment.get("mezo_budget_seconds") != plan["training_budget_seconds"]
            or experiment.get("qlora_budget_seconds") != plan["training_budget_seconds"]
            or experiment.get("mezo_probe_budget_seconds") != plan["probe_budget_seconds"]
            or experiment.get("qlora_probe_budget_seconds") != plan["probe_budget_seconds"]
            or experiment.get("abort_policy") != ABORT_POLICY
            or experiment.get("arms") != ["mezo", "qlora"]
            or experiment.get("controls") != ["frozen", "zero", "correct", "corrupt"]):
        blockers.append("BUDGET_OR_CONTROLS_UNMATCHED")

    return {
        "schema": SCHEMA + ".report",
        "status": ("HOLD_MISSING_PRECOMPARISON_EVIDENCE" if blockers else
                   "EVIDENCE_SHAPES_COMPLETE_OWNER_VERIFICATION_REQUIRED"),
        "blockers": blockers,
        "plan_sha256": plan_digest,
        "declared_evidence_sha256": canonical_sha256(e),
        "decision_scope": "DECLARED_EVIDENCE_CONSISTENCY_ONLY",
        "gpu_launch_decision": "NOT_AUTHORIZED",
        "authorizes_gpu_start": False,
        "host_lease_independently_authenticated": False,
        "eval_unsealed": False,
        "neural_optimizer_steps_run": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Source-only R20 comparison gate, no GPU authority")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--now-unix", type=float, required=True)
    args = parser.parse_args(argv)
    report = assess_precomparison(
        json.loads(args.plan.read_text(encoding="utf-8")),
        json.loads(args.evidence.read_text(encoding="utf-8")),
        now_unix=args.now_unix,
    )
    print(json.dumps(report, sort_keys=True))
    return (EXIT_INCOMPLETE_DECLARATIONS if report["blockers"]
            else EXIT_COMPLETE_DECLARATIONS_NOT_AUTHORIZED)


if __name__ == "__main__":
    raise SystemExit(main())
