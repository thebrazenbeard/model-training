"""Bind R26 declared rights lineage to R27 declared learning-cycle conditions.

Only public synthetic declarations are accepted. This is a source-consistency
consumer, never a permission to train, read private evaluation, or promote.
"""

from __future__ import annotations

from dataclasses import fields

from trainingday.r26_rights_taint import RightsTaintError, assess_revocation
from trainingday.r27_continuous_learning_policy import (
    ConsentClaim, CycleEvent, CycleEvidence, CycleRequest, FrozenBaseline,
    PolicyError, assess_cycle,
)


def _snapshot_declared_graph(nodes: object, events: object) -> tuple[list, list]:
    """Bounded plain-builtin copy; no caller alias may survive to later stages.

    This is not an OS-atomic snapshot of a real rights system. R26's own
    validator checks the complete graph and event schemas on the private copy.
    """
    if type(nodes) is not list or not 1 <= len(nodes) <= 128:
        raise RightsTaintError("node snapshot must be a bounded plain list")
    if type(events) is not list or len(events) > 64:
        raise RightsTaintError("event snapshot must be a bounded plain list")
    private_nodes = []
    for node in nodes:
        if type(node) is not dict or len(node) > 6:
            raise RightsTaintError("node snapshot requires bounded plain dict")
        parents = node.get("parents")
        if type(parents) is not list or len(parents) > 16:
            raise RightsTaintError("parent snapshot requires bounded plain list")
        private_nodes.append({**node, "parents": list(parents)})
    private_events = []
    for event in events:
        if type(event) is not dict or len(event) > 8:
            raise RightsTaintError("event snapshot requires bounded plain dict")
        private_events.append(dict(event))
    return private_nodes, private_events


def _snapshot_cycle(baseline: object, request: object, consent: object,
                    evidence: object, events: object) -> tuple:
    """Copy every public R27 frozen dataclass before any assessment callback.

    Exact class and builtin primitive values prevent attacker-defined equality
    or mutable field references from crossing validation/commitment boundaries.
    R27 subsequently validates all field-specific syntax and integer bounds.
    """
    def clone(value: object, expected: type) -> object:
        if type(value) is not expected:
            raise PolicyError("cycle snapshot dataclass type malformed")
        captured = {field.name: getattr(value, field.name)
                    for field in fields(expected)}
        if any(type(item) not in (str, int, bool) for item in captured.values()):
            raise PolicyError("cycle snapshot requires exact primitive fields")
        return expected(**captured)

    if type(events) is not tuple or not 4 <= len(events) <= 32:
        raise PolicyError("cycle snapshot event sequence malformed or unbounded")
    return (
        clone(baseline, FrozenBaseline),
        clone(request, CycleRequest),
        clone(consent, ConsentClaim),
        clone(evidence, CycleEvidence),
        tuple(clone(event, CycleEvent) for event in events),
    )


def assess_rights_bound_cycle(*, nodes: object, rights_events: object,
                              baseline: object, request: object, consent: object,
                              evidence: object, cycle_events: object,
                              now_unix: int) -> dict:
    """Fail closed if declared rights and cycle refer to different artifacts.

    A single public TRAIN_SHARD is supported, so an ambiguous multi-shard
    commitment cannot be interpreted as consent for a different corpus.
    Source hashes are caller declarations, not independently observed bytes.
    """
    nodes_snapshot, events_snapshot = _snapshot_declared_graph(nodes, rights_events)
    baseline_snapshot, request_snapshot, consent_snapshot, evidence_snapshot, cycle_snapshot = (
        _snapshot_cycle(baseline, request, consent, evidence, cycle_events))
    frozen_rights = assess_revocation(nodes_snapshot, [])
    current_rights = assess_revocation(nodes_snapshot, events_snapshot)
    cycle = assess_cycle(baseline=baseline_snapshot, request=request_snapshot,
                         consent=consent_snapshot, evidence=evidence_snapshot,
                         events=cycle_snapshot, now_unix=now_unix)

    # R26 validators have already bounded and validated plain-dict DAG nodes.
    graph = {node["id"]: node for node in nodes_snapshot}
    blockers = list(cycle["blockers"])

    def block(reason: str) -> None:
        if reason not in blockers:
            blockers.append(reason)

    # The v1 contract intentionally admits exactly one training source.
    # A disconnected second shard is still ambiguity, even when the caller
    # repins the graph and the selected adapter ancestry itself stays clean.
    if sum(node["kind"] == "TRAIN_SHARD" for node in nodes_snapshot) != 1:
        block("MULTI_SOURCE_LINEAGE_UNSUPPORTED")

    if baseline_snapshot.rights_sha256 != frozen_rights["assessment_sha256"]:
        block("FROZEN_RIGHTS_COMMITMENT_MISMATCH")

    def matched(kind: str, digest: str) -> str | None:
        candidates = [node["id"] for node in nodes_snapshot
                      if node["kind"] == kind and node["sha256"] == digest]
        return candidates[0] if len(candidates) == 1 else None

    source = matched("TRAIN_SHARD", request_snapshot.source_sha256)
    adapter = matched("ADAPTER", request_snapshot.proposed_adapter_sha256)
    checkpoint = matched("CHECKPOINT", request_snapshot.proposed_checkpoint_sha256)
    if source is None:
        block("TRAINING_SOURCE_UNBOUND")
    if adapter is None or checkpoint is None:
        block("CANDIDATE_ARTIFACT_UNBOUND")

    if source is not None and adapter is not None and checkpoint is not None:
        def ancestors(node_id: str) -> set[str]:
            found: set[str] = set()
            pending = [node_id]
            while pending:
                current = pending.pop()
                if current in found:
                    continue
                found.add(current)
                pending.extend(graph[current]["parents"])
            return found

        adapter_lineage = ancestors(adapter)
        checkpoint_lineage = ancestors(checkpoint)
        source_set = {node_id for node_id in checkpoint_lineage | adapter_lineage
                      if graph[node_id]["kind"] == "TRAIN_SHARD"}
        if source_set != {source} or source not in adapter_lineage:
            block("MULTI_SOURCE_LINEAGE_UNSUPPORTED")
        if adapter not in checkpoint_lineage:
            block("CANDIDATE_CHECKPOINT_NOT_DESCENDED_FROM_ADAPTER")
        for artifact in (source, adapter, checkpoint):
            if current_rights["status_by_artifact"][artifact] != "DECLARED_CLEAN_ONLY":
                block("RIGHTS_TAINTED_LINEAGE")

    return {
        "schema": "TRAININGDAY_R28_RIGHTS_BOUND_CYCLE_V1",
        "status": "HOLD", "blockers": blockers,
        "synthetic_source_conditions_satisfied": not blockers,
        "frozen_rights_assessment_sha256": frozen_rights["assessment_sha256"],
        "current_rights_assessment_sha256": current_rights["assessment_sha256"],
        "rights_status_by_artifact": current_rights["status_by_artifact"],
        "cycle_source_conditions_satisfied_before_binding":
            cycle["synthetic_source_conditions_satisfied"],
        # Current R26 graphs can reference the new shard and candidate but
        # cannot attest the historical base model and distinct past corpora.
        "previous_baseline_lineage_verified": False,
        "rights_independently_attested": False,
        "consent_independently_authenticated": False,
        "physical_tensor_change_verified": False,
        "independent_unseen_evaluation_verified": False,
        "gpu_start_authorized": False,
        "continuous_weight_update_authorized": False,
        "model_promotion_authorized": False,
        "protected_eval_unseal_authorized": False,
    }
