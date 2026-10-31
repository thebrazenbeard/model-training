"""Public-only R33 historical/current provenance bridge; never effect authority.

R28 deliberately supported exactly one *current* training shard and could not
bind the frozen base/previous adapter/checkpoint to a different past corpus.
This v2 shadow models a separately declared historical graph plus the R28
current graph. All data, signatures, provenance and rights remain untrusted.
"""

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
import re

from trainingday.r27_continuous_learning_policy import (
    ConsentClaim, CycleEvent, CycleEvidence, CycleRequest, FrozenBaseline,
)
from trainingday.r28_rights_cycle_integration import assess_rights_bound_cycle


SCHEMA = "TRAININGDAY_R33_HISTORICAL_CURRENT_V2_PUBLIC_ONLY"
FROZEN_SCHEMA = "TRAININGDAY_R33_HISTORICAL_FROZEN_GRAPH_V1"
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HIST_KINDS = frozenset({
    "BASE_MODEL", "HISTORICAL_SHARD", "HISTORICAL_ADAPTER",
    "HISTORICAL_CHECKPOINT", "HISTORICAL_EVIDENCE",
})
_ROOTS = frozenset({"BASE_MODEL", "HISTORICAL_SHARD"})
_RIGHTS = frozenset({"DECLARED_ADMITTED", "UNKNOWN", "REVIEW_REQUIRED", "REVOKED"})
_ACTIONS = {"LICENSE_CHANGED": 2, "PRIVACY_TAINT": 3, "REVOKED": 4}
_STATUS = (
    "DECLARED_CLEAN_ONLY", "RIGHTS_REVIEW_REQUIRED", "LICENSE_REVIEW_REQUIRED",
    "PRIVACY_TAINTED", "REVOKED",
)


class HistoricalProvenanceError(ValueError):
    """Malformed, ambiguous or unbounded caller-supplied public declaration."""


def _require(ok: bool, detail: str) -> None:
    if not ok:
        raise HistoricalProvenanceError(detail)


def _sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _private_tree(value: object) -> object:
    """Copy with validation in one bounded traversal; never run caller hooks.

    Intra-process mutations during the traversal are not OS-atomic; imported
    node state must be independently captured by a future trusted custodian.
    """
    live: set[int] = set()
    budget = [0]

    def take(item: object, depth: int) -> object:
        budget[0] += 1
        _require(depth <= 20 and budget[0] <= 8192, "declaration nesting/node cap")
        if type(item) in (str, int, bool) or item is None:
            if type(item) is str:
                _require(len(item) <= 65536, "string length cap")
            elif type(item) is int:
                _require(item.bit_length() <= 64, "integer width cap")
            return item
        _require(type(item) in (list, tuple, dict), "non-builtin declaration container")
        _require(len(item) <= 256, "declaration container item cap")
        marker = id(item)
        _require(marker not in live, "cyclic input declaration")
        live.add(marker)
        try:
            if type(item) is dict:
                _require(all(type(k) is str for k in item), "non-string map key")
                copied = {key: take(value, depth + 1) for key, value in item.items()}
            elif type(item) is list:
                copied = [take(value, depth + 1) for value in item]
            else:
                copied = tuple(take(value, depth + 1) for value in item)
        finally:
            live.remove(marker)
        return copied

    return take(value, 0)


def _private_cycle(*, baseline: object, request: object, consent: object,
                   evidence: object, events: object) -> tuple:
    """Exact frozen dataclass copies, including rejection of hidden __dict__."""
    def clone(value: object, cls: type) -> object:
        _require(type(value) is cls, "cycle dataclass type")
        data = object.__getattribute__(value, "__dict__")
        _require(type(data) is dict, "cycle dataclass map type")
        names = {f.name for f in fields(cls)}
        private = {}
        for key, val in data.items():
            _require(type(key) is str and type(val) in (str, int, bool),
                     "cycle non-builtin field/value")
            if type(val) is str:
                _require(len(val) <= 65536, "cycle dataclass string cap")
            elif type(val) is int:
                _require(val.bit_length() <= 64, "cycle dataclass integer width cap")
            private[key] = val
        _require(set(private) == names, "cycle dataclass hidden/missing fields")
        return cls(**private)

    _require(type(events) is tuple and 4 <= len(events) <= 32,
             "cycle events tuple malformed")
    return (clone(baseline, FrozenBaseline), clone(request, CycleRequest),
            clone(consent, ConsentClaim), clone(evidence, CycleEvidence),
            tuple(clone(event, CycleEvent) for event in events))


def _canonical(value: object) -> bytes:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                     separators=(",", ":")).encode("ascii")
    _require(len(raw) <= 65536, "historical declaration byte cap")
    return raw


def _historical_graph(nodes: object) -> tuple[dict[str, dict], list[dict]]:
    _require(type(nodes) is list and 4 <= len(nodes) <= 64,
             "historical node count/shape")
    graph: dict[str, dict] = {}
    digest_owners: set[str] = set()
    for node in nodes:
        _require(type(node) is dict and len(node) <= 6,
                 "historical node type/size")
        kind = node.get("kind")
        nid = node.get("id")
        digest = node.get("sha256")
        parents = node.get("parents")
        _require(type(kind) is str and kind in _HIST_KINDS
                 and _id(nid) and _sha(digest), "historical node identity")
        _require(nid not in graph and digest not in digest_owners,
                 "duplicate historical node/digest")
        digest_owners.add(digest)
        _require(type(parents) is list and len(parents) <= 8
                 and all(_id(parent) for parent in parents),
                 "historical parent identity")
        _require(len(parents) == len(set(parents)),
                 "historical parent duplicated")
        if kind in _ROOTS:
            _require(set(node) == {"id", "kind", "sha256", "parents", "license_sha256", "rights"}
                     and not parents and _sha(node["license_sha256"])
                     and type(node["rights"]) is str and node["rights"] in _RIGHTS,
                     "historical root rights/license")
        else:
            _require(set(node) == {"id", "kind", "sha256", "parents"}
                     and parents, "historical derived node")
        graph[nid] = dict(node, parents=sorted(parents))

    _require(sum(n["kind"] == "BASE_MODEL" for n in graph.values()) == 1,
             "exactly one baseline model")
    _require(any(n["kind"] == "HISTORICAL_SHARD" for n in graph.values()),
             "missing historical shard")

    for node in graph.values():
        kinds = {graph[parent]["kind"] for parent in node["parents"]
                 if parent in graph}
        _require(all(parent in graph for parent in node["parents"]),
                 "historical parent missing")
        kind = node["kind"]
        if kind == "HISTORICAL_ADAPTER":
            _require(kinds <= {"BASE_MODEL", "HISTORICAL_SHARD",
                               "HISTORICAL_ADAPTER", "HISTORICAL_CHECKPOINT"},
                     "historical adapter parent kind")
        elif kind == "HISTORICAL_CHECKPOINT":
            _require(len(node["parents"]) == 1
                     and kinds == {"HISTORICAL_ADAPTER"},
                     "historical checkpoint must have exactly one adapter parent")
        elif kind == "HISTORICAL_EVIDENCE":
            _require(len(node["parents"]) == 1
                     and kinds == {"HISTORICAL_CHECKPOINT"},
                     "historical evidence must have exactly one checkpoint parent")

    visited: dict[str, set[str]] = {}
    visiting: set[str] = set()

    def ancestors(nid: str) -> set[str]:
        if nid in visited:
            return visited[nid]
        _require(nid not in visiting, "historical lineage cycle")
        visiting.add(nid)
        result = {nid}
        for parent in graph[nid]["parents"]:
            result.update(ancestors(parent))
        visiting.remove(nid)
        visited[nid] = result
        return result

    for nid, node in graph.items():
        source_kinds = {graph[a]["kind"] for a in ancestors(nid)}
        if node["kind"] not in _ROOTS:
            _require("BASE_MODEL" in source_kinds
                     and "HISTORICAL_SHARD" in source_kinds,
                     "derived historical artifact lacks model and source ancestry")

    normalized = [graph[nid] for nid in sorted(graph)]
    _canonical({"schema": FROZEN_SCHEMA, "nodes": normalized})
    return graph, normalized


def freeze_historical_graph(nodes: object) -> str:
    """Hash of frozen *caller claims*, not authenticated historical bytes."""
    private = _private_tree(nodes)
    _, canonical_nodes = _historical_graph(private)
    return sha256(_canonical({"schema": FROZEN_SCHEMA,
                             "nodes": canonical_nodes})).hexdigest()


def _historical_events(events: object, graph: dict[str, dict]) -> list[dict]:
    _require(type(events) is list and len(events) <= 32,
             "historical event count")
    out: list[dict] = []
    seen: set[str] = set()
    for event in events:
        _require(type(event) is dict and len(event) <= 7
                 and _id(event.get("id")) and _id(event.get("source_id"))
                 and _sha(event.get("source_sha256"))
                 and _sha(event.get("evidence_sha256")), "historical event fields")
        _require(event["id"] not in seen, "duplicate historical event id")
        seen.add(event["id"])
        action = event.get("action")
        _require(type(action) is str and action in _ACTIONS,
                 "historical event action")
        expected = {"id", "source_id", "source_sha256", "action", "evidence_sha256"}
        if action == "LICENSE_CHANGED":
            expected |= {"prior_license_sha256", "replacement_license_sha256"}
        _require(set(event) == expected, "historical event schema")
        source = graph.get(event["source_id"])
        _require(source is not None and source["kind"] in _ROOTS
                 and event["source_sha256"] == source["sha256"],
                 "historical event source mismatch")
        if action == "LICENSE_CHANGED":
            _require(_sha(event["prior_license_sha256"])
                     and _sha(event["replacement_license_sha256"])
                     and event["prior_license_sha256"] == source["license_sha256"]
                     and event["replacement_license_sha256"] != source["license_sha256"],
                     "historical license change malformed")
        out.append(dict(event))
    return sorted(out, key=lambda e: e["id"])


def _historical_taint(graph: dict[str, dict], events: list[dict]) -> dict[str, str]:
    ranks = {nid: {"DECLARED_ADMITTED": 0, "UNKNOWN": 1,
                   "REVIEW_REQUIRED": 1, "REVOKED": 4}[node["rights"]]
             for nid, node in graph.items() if node["kind"] in _ROOTS}
    for event in events:
        source_id = event["source_id"]
        ranks[source_id] = max(ranks[source_id], _ACTIONS[event["action"]])

    def rank(nid: str) -> int:
        if nid not in ranks:
            ranks[nid] = max(rank(parent) for parent in graph[nid]["parents"])
        return ranks[nid]

    return {nid: _STATUS[rank(nid)] for nid in sorted(graph)}


def assess_historical_bridge_v2(*, historical_nodes: object,
                                historical_events: object,
                                frozen_historical_sha256: object,
                                bridge_claim: object,
                                current_nodes: object,
                                current_rights_events: object,
                                baseline: FrozenBaseline,
                                request: CycleRequest,
                                consent: ConsentClaim,
                                evidence: CycleEvidence,
                                cycle_events: tuple[CycleEvent, ...],
                                now_unix: int) -> dict:
    """Public two-corpus source-consistency shadow; ALL real permissions false."""
    # No caller-owned list/dict/dataclass may be used after these copies.
    historic = _private_tree(historical_nodes)
    h_events = _private_tree(historical_events)
    claimed_bridge = _private_tree(bridge_claim)
    current = _private_tree(current_nodes)
    current_events = _private_tree(current_rights_events)
    b, r, c, e, ce = _private_cycle(baseline=baseline, request=request,
                                   consent=consent, evidence=evidence,
                                   events=cycle_events)
    _require(type(now_unix) is int and now_unix.bit_length() <= 64,
             "cycle wall-clock type/width")
    _require(_sha(frozen_historical_sha256), "frozen historical digest malformed")
    _require(type(claimed_bridge) is dict
             and set(claimed_bridge) == {"historical_graph_sha256",
                                          "previous_checkpoint_sha256",
                                          "current_shard_sha256",
                                          "candidate_adapter_sha256"}
             and all(_sha(v) for v in claimed_bridge.values()),
             "cross-graph bridge claim malformed")
    graph, canonical_nodes = _historical_graph(historic)
    checked_events = _historical_events(h_events, graph)
    frozen_digest = sha256(_canonical({"schema": FROZEN_SCHEMA,
                                      "nodes": canonical_nodes})).hexdigest()
    statuses = _historical_taint(graph, checked_events)
    current_report = assess_rights_bound_cycle(
        nodes=current, rights_events=current_events, baseline=b, request=r,
        consent=c, evidence=e, cycle_events=ce, now_unix=now_unix)

    blockers: list[str] = []

    def block(reason: str) -> None:
        if reason not in blockers:
            blockers.append(reason)

    if frozen_digest != frozen_historical_sha256:
        block("FROZEN_HISTORICAL_GRAPH_MISMATCH")
    if not current_report["synthetic_source_conditions_satisfied"]:
        block("CURRENT_CYCLE_SOURCE_BLOCKED")

    expected = {
        "historical_graph_sha256": frozen_historical_sha256,
        "previous_checkpoint_sha256": b.checkpoint_sha256,
        "current_shard_sha256": r.source_sha256,
        "candidate_adapter_sha256": r.proposed_adapter_sha256,
    }
    if claimed_bridge != expected:
        block("CROSS_GRAPH_CLAIM_MISMATCH")

    expected_baseline = {
        "BASE_MODEL": b.base_sha256,
        "HISTORICAL_ADAPTER": b.adapter_sha256,
        "HISTORICAL_CHECKPOINT": b.checkpoint_sha256,
        "HISTORICAL_EVIDENCE": b.evidence_sha256,
    }
    matching: dict[str, str] = {}
    for kind, digest in expected_baseline.items():
        matches = [nid for nid, node in graph.items()
                   if node["kind"] == kind and node["sha256"] == digest]
        if len(matches) != 1:
            block("HISTORICAL_BASELINE_IDENTITY_MISMATCH")
        else:
            matching[kind] = matches[0]

    if len(matching) == len(expected_baseline):
        def ancestors(nid: str) -> set[str]:
            found: set[str] = set()
            pending = [nid]
            while pending:
                key = pending.pop()
                if key in found:
                    continue
                found.add(key)
                pending.extend(graph[key]["parents"])
            return found

        previous_adapter_ancestors = ancestors(matching["HISTORICAL_ADAPTER"])
        previous_checkpoint_ancestors = ancestors(matching["HISTORICAL_CHECKPOINT"])
        prior_evidence_ancestors = ancestors(matching["HISTORICAL_EVIDENCE"])
        all_prior_sources = {nid for nid, node in graph.items()
                             if node["kind"] == "HISTORICAL_SHARD"}
        if (matching["BASE_MODEL"] not in previous_adapter_ancestors
                or not all_prior_sources <= previous_adapter_ancestors
                or matching["HISTORICAL_ADAPTER"] not in previous_checkpoint_ancestors
                or matching["HISTORICAL_CHECKPOINT"] not in prior_evidence_ancestors):
            block("HISTORICAL_SHARD_UNBOUND")
        if (graph[matching["HISTORICAL_CHECKPOINT"]]["parents"]
                != [matching["HISTORICAL_ADAPTER"]]
                or graph[matching["HISTORICAL_EVIDENCE"]]["parents"]
                != [matching["HISTORICAL_CHECKPOINT"]]):
            block("HISTORICAL_DIRECT_PARENT_MISMATCH")
    if any(status != "DECLARED_CLEAN_ONLY" for status in statuses.values()):
        block("HISTORICAL_RIGHTS_TAINTED")

    historical_shas = {node["sha256"] for node in graph.values()}
    if r.source_sha256 in historical_shas:
        block("CURRENT_HISTORICAL_ROLE_COLLISION")
    if (r.proposed_adapter_sha256 in historical_shas
            or r.proposed_checkpoint_sha256 in historical_shas):
        block("CANDIDATE_HISTORY_ARTIFACT_COLLISION")

    return {
        "schema": SCHEMA,
        "status": "HOLD",
        "blockers": blockers,
        "synthetic_cross_generation_lineage_consistent": not blockers,
        "frozen_historical_graph_sha256": frozen_digest,
        "historical_status_by_artifact": statuses,
        "current_cycle_source_conditions_satisfied":
            current_report["synthetic_source_conditions_satisfied"],
        "cross_graph_edge_independently_verified": False,
        "historical_corpus_rights_independently_attested": False,
        "current_corpus_rights_independently_attested": False,
        "previous_baseline_lineage_verified": False,
        "model_bytes_independently_verified": False,
        "consent_independently_authenticated": False,
        "protected_eval_unseal_authorized": False,
        "gpu_start_authorized": False,
        "continuous_weight_update_authorized": False,
        "model_promotion_authorized": False,
        "model_deployment_authorized": False,
    }
