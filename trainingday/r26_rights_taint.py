"""R26 synthetic provenance-taint cascade. No model rights or removal authority.

This deliberately accepts public *declarations*, not an authenticated rights
ledger. A clean result is a graph-consistency observation and always HOLD.
No data, artifact, adapter, checkpoint, or evidence file is opened or changed.
"""

from __future__ import annotations

from hashlib import sha256
import json
import re


class RightsTaintError(ValueError):
    """Invalid, ambiguous, or unbounded declared lineage or revocation."""


_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z", re.ASCII)
_KINDS = frozenset({"TRAIN_SHARD", "ADAPTER", "CHECKPOINT", "EVAL_RECEIPT", "PROMOTION_CANDIDATE"})
_RIGHTS = frozenset({"DECLARED_ADMITTED", "UNKNOWN", "REVIEW_REQUIRED", "REVOKED"})
_ACTIONS = frozenset({"REVOKED", "LICENSE_CHANGED", "PRIVACY_TAINT"})
_STATUS_BY_RANK = (
    "DECLARED_CLEAN_ONLY", "RIGHTS_REVIEW_REQUIRED", "LICENSE_REVIEW_REQUIRED",
    "PRIVACY_TAINTED_LINEAGE", "REVOKED_LINEAGE",
)
_EVENT_RANK = {"LICENSE_CHANGED": 2, "PRIVACY_TAINT": 3, "REVOKED": 4}
_MAX_NODES, _MAX_EVENTS, _MAX_PARENTS, _MAX_GRAPH_BYTES = 128, 64, 16, 65536


def _require(condition: bool, why: str) -> None:
    if not condition:
        raise RightsTaintError(why)


def _digest(value: object) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _identifier(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _canonical(value: object) -> bytes:
    """Canonical byte representation of *already structurally bounded* input."""
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise RightsTaintError("noncanonical declaration") from exc
    _require(len(payload) <= _MAX_GRAPH_BYTES, "lineage declaration byte cap exceeded")
    return payload


def _validate_nodes(nodes: object) -> dict[str, dict]:
    # Reject unbounded lists before any canonical serialization or copy.
    _require(type(nodes) is list and 1 <= len(nodes) <= _MAX_NODES,
             "node list malformed or unbounded")
    graph: dict[str, dict] = {}
    digest_owner: set[str] = set()
    for raw in nodes:
        _require(type(raw) is dict and len(raw) <= 6, "node malformed or unbounded")
        _require(type(raw.get("kind")) is str and raw["kind"] in _KINDS
                 and _identifier(raw.get("id")),
                 "node kind/id malformed")
        _require(_digest(raw.get("sha256")), "node digest malformed")
        node_id = raw["id"]
        _require(node_id not in graph and raw["sha256"] not in digest_owner,
                 "duplicate node or artifact digest")
        digest_owner.add(raw["sha256"])
        parents = raw.get("parents")
        _require(type(parents) is list and len(parents) <= _MAX_PARENTS,
                 "parent list malformed or unbounded")
        _require(all(_identifier(p) for p in parents)
                 and len(parents) == len(set(parents)),
                 "parent id invalid or duplicated")
        if raw["kind"] == "TRAIN_SHARD":
            _require(set(raw) == {"id", "kind", "sha256", "parents", "license_sha256", "rights"},
                     "training shard fields invalid")
            _require(not parents and _digest(raw["license_sha256"])
                     and type(raw["rights"]) is str and raw["rights"] in _RIGHTS,
                     "training source rights/license or parent invalid")
        else:
            _require(set(raw) == {"id", "kind", "sha256", "parents"}
                     and len(parents) > 0, "derived artifact fields/parents invalid")
        graph[node_id] = dict(raw, parents=sorted(parents))

    for node in graph.values():
        _require(all(parent in graph for parent in node["parents"]),
                 "lineage references missing parent")
        parent_kinds = {graph[parent]["kind"] for parent in node["parents"]}
        kind = node["kind"]
        if kind == "ADAPTER":
            _require(parent_kinds <= {"TRAIN_SHARD", "CHECKPOINT", "ADAPTER"},
                     "adapter has invalid parent type")
        elif kind == "CHECKPOINT":
            _require(parent_kinds <= {"TRAIN_SHARD", "CHECKPOINT", "ADAPTER"},
                     "checkpoint has invalid parent type")
        elif kind == "EVAL_RECEIPT":
            _require(parent_kinds <= {"ADAPTER", "CHECKPOINT"},
                     "evaluation receipt parent type invalid")
        elif kind == "PROMOTION_CANDIDATE":
            _require(parent_kinds <= {"EVAL_RECEIPT", "ADAPTER", "CHECKPOINT"}
                     and "EVAL_RECEIPT" in parent_kinds
                     and ("ADAPTER" in parent_kinds or "CHECKPOINT" in parent_kinds),
                     "promotion candidate lacks eval and artifact parents")

    # An ancestor adapter is not interchangeable with its later checkpoint.
    # The candidate must name exactly the adapter/checkpoint IDs that its
    # evaluation receipt(s) *directly* scored, never a transitive ancestor.
    for node in graph.values():
        if node["kind"] == "PROMOTION_CANDIDATE":
            proposed_models = {p for p in node["parents"]
                               if graph[p]["kind"] in {"ADAPTER", "CHECKPOINT"}}
            evaluated_models: set[str] = set()
            for p in node["parents"]:
                if graph[p]["kind"] == "EVAL_RECEIPT":
                    evaluated_models.update(graph[p]["parents"])
            _require(proposed_models == evaluated_models,
                     "candidate exact model identity differs from evaluation receipt")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visited:
            return True
        _require(node_id not in visiting, "cyclic artifact lineage")
        visiting.add(node_id)
        has_train_source = graph[node_id]["kind"] == "TRAIN_SHARD"
        for p in graph[node_id]["parents"]:
            has_train_source = visit(p) or has_train_source
        visiting.remove(node_id)
        _require(has_train_source, "derived artifact lacks training-shard ancestor")
        visited.add(node_id)
        return has_train_source

    for node_id in graph:
        visit(node_id)
    return graph


def _validate_events(events: object, graph: dict[str, dict]) -> list[dict]:
    _require(type(events) is list and len(events) <= _MAX_EVENTS,
             "event list malformed or unbounded")
    seen: set[str] = set()
    result = []
    for raw in events:
        _require(type(raw) is dict and len(raw) <= 7 and _identifier(raw.get("id"))
                 and _identifier(raw.get("source_id")), "event id/source malformed")
        _require(raw["id"] not in seen, "duplicate event id")
        seen.add(raw["id"])
        action = raw.get("action")
        _require(type(action) is str and action in _ACTIONS,
                 "unknown revocation event action")
        allowed = {"id", "source_id", "source_sha256", "action", "evidence_sha256"}
        if action == "LICENSE_CHANGED":
            allowed |= {"prior_license_sha256", "replacement_license_sha256"}
        _require(set(raw) == allowed, "event fields malformed or authority smuggling")
        _require(_digest(raw["source_sha256"]) and _digest(raw["evidence_sha256"]),
                 "event hash malformed")
        source = graph.get(raw["source_id"])
        _require(source is not None and source["kind"] == "TRAIN_SHARD",
                 "event does not reference declared training shard")
        _require(raw["source_sha256"] == source["sha256"],
                 "event references different source artifact bytes")
        if action == "LICENSE_CHANGED":
            _require(_digest(raw["prior_license_sha256"])
                     and _digest(raw["replacement_license_sha256"]),
                     "license change digest malformed")
            _require(raw["prior_license_sha256"] == source["license_sha256"]
                     and raw["replacement_license_sha256"] != raw["prior_license_sha256"],
                     "license change not bound to declared prior license")
        result.append(dict(raw))
    return sorted(result, key=lambda x: x["id"])


def assess_revocation(nodes: object, events: object) -> dict:
    """Cascade *declared* rights concerns. Always HOLD, never mutate or authorize.

    Inputs are public/synthetic artifact metadata and events. SHA fields are
    syntax-checked references, *not* checks of actual bytes, signatures or rights.
    Callers must separately obtain authenticated revocation, independent data
    stewardship and any future model-unlearning evidence.
    """
    graph = _validate_nodes(nodes)
    checked_events = _validate_events(events, graph)
    canonical_nodes = [graph[k] for k in sorted(graph)]
    canonical_claim = {"nodes": canonical_nodes, "events": checked_events,
                       "policy": "R26_DECLARED_TAINT_V1"}
    commitment = sha256(_canonical(canonical_claim)).hexdigest()

    rank_by_source = {}
    cause_by_source: dict[str, set[str]] = {}
    for node_id, node in graph.items():
        if node["kind"] == "TRAIN_SHARD":
            rank_by_source[node_id] = {"DECLARED_ADMITTED": 0, "UNKNOWN": 1,
                                       "REVIEW_REQUIRED": 1, "REVOKED": 4}[node["rights"]]
            cause_by_source[node_id] = {node_id} if rank_by_source[node_id] else set()
    for e in checked_events:
        source = e["source_id"]
        rank_by_source[source] = max(rank_by_source[source], _EVENT_RANK[e["action"]])
        cause_by_source[source].add(source)

    ranks: dict[str, int] = {}
    causes: dict[str, set[str]] = {}

    def descend(node_id: str) -> int:
        if node_id in ranks:
            return ranks[node_id]
        node = graph[node_id]
        rank = rank_by_source.get(node_id, 0)
        node_causes = set(cause_by_source.get(node_id, ()))
        for p in node["parents"]:
            rank = max(rank, descend(p))
            node_causes.update(causes[p])
        ranks[node_id] = rank
        causes[node_id] = node_causes
        return rank

    for node_id in sorted(graph):
        descend(node_id)

    statuses = {key: _STATUS_BY_RANK[ranks[key]] for key in sorted(graph)}
    return {
        "schema": "TRAININGDAY_R26_RIGHTS_TAINT_V1",
        "assessment_sha256": commitment,
        "status_by_artifact": statuses,
        "source_causes_by_artifact": {key: sorted(causes[key]) for key in sorted(graph)},
        "impacted_artifacts": [key for key in sorted(graph) if ranks[key] > 0],
        "release_gate": "HOLD",
        "promotion_allowed": False,
        "protected_eval_unseal_allowed": False,
        "training_allowed": False,
        "rights_attested": False,
        "independent_custodian_verified": False,
        "unlearning_verified": False,
        "evidence_retention_action": "NO_DELETION_ATTEMPTED",
        "evaluation_claim_level": "UNTRUSTED_DECLARED_LINEAGE_ONLY",
    }
