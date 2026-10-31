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
# JSON's recursive encoder/decoder has a finite interpreter stack.  Bound
# untrusted in-memory evidence depth before constructing its digest.
MAX_CANONICAL_JSON_DEPTH = 64
# Exit 0 must never be mistaken by a shell runner for GPU launch authority.
# Exit 2 means declarations are incomplete; exit 3 means declarations are
# internally consistent but all external owner/host authority is still absent.
EXIT_INCOMPLETE_DECLARATIONS = 2
EXIT_COMPLETE_DECLARATIONS_NOT_AUTHORIZED = 3


def _part(value: object) -> Mapping:
    return value if type(value) is dict else {}


def _plain_json_evidence(value: object) -> bool:
    """Only inspect values whose Python semantics match their JSON digest.

    Subclassed mappings can override ``get``; subclassed lists/strings can
    override comparisons.  Neither may be trusted to describe the bytes later
    canonicalized into a source-only evidence receipt.  A bounded iterative
    walk also rejects cycles, excessive nesting, and pathological in-memory
    evidence objects before the recursive canonical JSON encoder is reached.
    """
    pending = [(value, 0)]
    visited = set()
    inspected = 0
    while pending:
        current, depth = pending.pop()
        inspected += 1
        if inspected > 100_000 or depth > MAX_CANONICAL_JSON_DEPTH:
            return False
        kind = type(current)
        if current is None or kind in (str, bool):
            continue
        if kind is int:
            if current.bit_length() > 63:
                return False
            continue
        if kind is float:
            if not math.isfinite(current):
                return False
            continue
        if kind in (dict, list):
            identity = id(current)
            if identity in visited:
                return False
            visited.add(identity)
            if kind is dict:
                if any(type(key) is not str for key in current):
                    return False
                pending.extend((item, depth + 1) for item in current.values())
            else:
                pending.extend((item, depth + 1) for item in current)
            continue
        return False
    return True


def _sha(value: object, width: int = 64) -> bool:
    return (type(value) is str and len(value) == width
            and all(ch in "0123456789abcdef" for ch in value))


def _number(value: object) -> bool:
    if type(value) is int:
        return value.bit_length() <= 63
    return type(value) is float and math.isfinite(value)


def _byte_count(value: object) -> bool:
    # A declaration of physical bytes must be integral, nonnegative, and
    # bounded so Boolean/float/astronomical values cannot pass shape checks.
    return type(value) is int and 0 <= value < (1 << 63)


def _recent(timestamp: object, now_unix: float) -> bool:
    return (_number(timestamp) and timestamp <= now_unix
            and now_unix - timestamp <= MAX_OBSERVATION_AGE_SECONDS)


def _load_strict_json(path: Path) -> object:
    """Refuse ambiguous JSON before any parsed-value digest or comparison."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate evidence JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonstandard evidence JSON constant: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite evidence JSON number")
        return result

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=unique_keys,
                      parse_constant=reject_constant,
                      parse_float=finite_float)


def assess_precomparison(plan: Mapping, evidence: object, *, now_unix: float) -> dict:
    """Reject absent/mismatched proof fields; NEVER authorize optimizer execution.

    Every status is a *source-level* result. Even an internally complete bundle
    requires separately authenticated live host leases and an operator review.
    """
    if type(plan) is not dict or not _plain_json_evidence(plan):
        raise ValueError("plan must contain only plain canonical JSON values")
    plan_digest = validate_plan(plan)
    if not _number(now_unix) or now_unix <= 0:
        raise ValueError("now_unix must be a current positive timestamp")
    plain_evidence = (type(evidence) is dict
                      and _plain_json_evidence(evidence))
    e = evidence if plain_evidence else {}
    source, data = _part(e.get("source")), _part(e.get("data"))
    initial, host = _part(e.get("initial")), _part(e.get("host"))
    experiment = _part(e.get("experiment"))
    blockers = [] if plain_evidence else ["EVIDENCE_NOT_CANONICAL_JSON"]

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

    # These checks reject *declared* resource conflicts. They are not live
    # telemetry or proof that an OS/GPU lease actually exists. The report
    # remains explicitly nonauthorizing even with perfect declarations.
    thermal = _part(host.get("thermal"))
    if (not _recent(thermal.get("observed_at_unix"), now_unix)
            or thermal.get("thermal_warning") is not False
            or thermal.get("thermal_throttling_active") is not False):
        blockers.append("HOST_THERMAL_UNSAFE")

    memory = _part(host.get("gpu_memory"))
    free, mezo, qlora, reserve = (
        memory.get("free_bytes"), memory.get("mezo_fit_peak_bytes"),
        memory.get("qlora_fit_peak_bytes"), memory.get("reserve_bytes"))
    if (not _recent(memory.get("observed_at_unix"), now_unix)
            or not all(_byte_count(v) for v in (free, mezo, qlora, reserve))
            or mezo == 0 or qlora == 0 or reserve == 0
            or free < max(mezo, qlora) + reserve):
        blockers.append("GPU_MEMORY_FIT_UNVERIFIED")

    # One declared active lease ID, matching the local lease. A multi-owner
    # inventory is a collision, not extra confidence in exclusivity.
    active_leases = host.get("active_resource_lease_ids")
    if (type(active_leases) is not list
            or len(active_leases) != 1
            or type(active_leases[0]) is not str
            or not active_leases[0]
            or active_leases[0] != host.get("lease_id")):
        blockers.append("GPU_LEASE_COLLISION_UNVERIFIED")

    if (not _sha(experiment.get("fit_mezo_sha256"))
            or not _sha(experiment.get("fit_qlora_sha256"))
            or not all(_number(experiment.get(field)) for field in (
                "mezo_budget_seconds", "qlora_budget_seconds",
                "mezo_probe_budget_seconds", "qlora_probe_budget_seconds"))
            or experiment.get("mezo_budget_seconds") != plan["training_budget_seconds"]
            or experiment.get("qlora_budget_seconds") != plan["training_budget_seconds"]
            or experiment.get("mezo_probe_budget_seconds") != plan["probe_budget_seconds"]
            or experiment.get("qlora_probe_budget_seconds") != plan["probe_budget_seconds"]
            or experiment.get("abort_policy") != ABORT_POLICY
            or experiment.get("arms") != ["mezo", "qlora"]
            or experiment.get("controls") != ["frozen", "zero", "correct", "corrupt"]):
        blockers.append("BUDGET_OR_CONTROLS_UNMATCHED")

    declared_evidence_sha256 = None
    if plain_evidence:
        try:
            declared_evidence_sha256 = canonical_sha256(e)
        except (TypeError, ValueError, RecursionError):
            # Do not replace invalid evidence with a fabricated commitment.
            # Nonfinite numbers and unsupported Python values must yield a HOLD.
            blockers.append("EVIDENCE_NOT_CANONICAL_JSON")
    return {
        "schema": SCHEMA + ".report",
        "status": ("HOLD_MISSING_PRECOMPARISON_EVIDENCE" if blockers else
                   "EVIDENCE_SHAPES_COMPLETE_OWNER_VERIFICATION_REQUIRED"),
        "blockers": blockers,
        "plan_sha256": plan_digest,
        "declared_evidence_sha256": declared_evidence_sha256,
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
    plan = _load_strict_json(args.plan)
    try:
        evidence = _load_strict_json(args.evidence)
    except (OSError, ValueError, RecursionError):
        # A corrupt receipt is explicitly HOLD, never a last-key-wins claim.
        evidence = None
    report = assess_precomparison(plan, evidence, now_unix=args.now_unix)
    print(json.dumps(report, sort_keys=True))
    return (EXIT_INCOMPLETE_DECLARATIONS if report["blockers"]
            else EXIT_COMPLETE_DECLARATIONS_NOT_AUTHORIZED)


if __name__ == "__main__":
    raise SystemExit(main())
