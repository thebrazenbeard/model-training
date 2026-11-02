"""Source-only inspection of Labyrinth #95's adaptive Minotaur TRAIN manifest.

No raw prompts, arithmetic answers, feedback bits, or private evaluator records
leave this module. A self-hash and an operator-supplied producer revision are
consistency evidence, not signed provenance or model-training authorization.
The real producer additionally replays its deterministic Minotaur policy;
this deliberately isolated consumer cannot authenticate that policy or the
last feedback bit and must never be used as an optimizer launch gate.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any


SCHEMA = "vera.trainingday.minotaur-adaptive-source-intake.v1"
PRODUCER_SCHEMA = "labyrinth.hf-adaptive-source-manifest.v1"
PINNED_LABYRINTH_HEAD = "20c48b89ce359c43313e8a0fc3aec4d4023d2bd6"
PINNED_LABYRINTH_TREE = "c01920ae7a5fea9e3bc9cf180cdfead61b953da1"
UNVERIFIED_REJECTED = "UNVERIFIED_REJECTED"
_DOMAIN = b"labyrinth.hf-adaptive-source-manifest.v1\x00"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ROOM_ID = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_CASE_ID = re.compile(r"train-taskmaster-([0-9a-f]{32})-[0-9a-f]{12}\Z", re.ASCII)
_OBJECTIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)
_LABEL = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z", re.ASCII)
_PAYLOAD_FIELDS = frozenset({
    "schema", "scope", "cycle_index", "pilot_budget", "train_rooms",
    "ordered_optimizer_exposures", "initial_checkpoint_sha256",
    "adaptive_feedback_rows",
})
_SCOPE_FIELDS = frozenset({
    "schema", "objective_id", "domain", "skills", "specificity",
    "scale", "max_cycles", "seed",
})
_BUDGET_FIELDS = frozenset({"seed", "train_count", "eval_count", "tiers", "steps"})
_ROOM_FIELDS = frozenset({
    "room_id", "case_id", "skill", "tactic", "difficulty",
    "planner_source", "prompt_sha256",
})


def _reject_if(condition: bool) -> None:
    if condition:
        raise ValueError("adaptive TRAIN source is not structurally admissible")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _reject_if(key in value)
        value[key] = item
    return value


def _nonfinite(_value: str) -> Any:
    raise ValueError("nonfinite JSON is forbidden")


def _canonical_bytes(value: Any) -> bytes:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    _reject_if(len(encoded) > 131_072)
    return encoded


def _check_scope(scope: Any, train_count: int) -> None:
    _reject_if(type(scope) is not dict or set(scope) != _SCOPE_FIELDS)
    _reject_if(scope["schema"] != "labyrinth.training-scope.v1")
    _reject_if(type(scope["objective_id"]) is not str or
               _OBJECTIVE.fullmatch(scope["objective_id"]) is None)
    _reject_if(type(scope["domain"]) is not str or
               _LABEL.fullmatch(scope["domain"]) is None)
    _reject_if(type(scope["skills"]) is not list or scope["skills"] != ["arithmetic"])
    _reject_if(type(scope["specificity"]) is not int or
               not 1 <= scope["specificity"] <= 5)
    _reject_if(type(scope["scale"]) is not int or not 1 <= scope["scale"] <= 5)
    _reject_if(type(scope["max_cycles"]) is not int or
               not 1 <= scope["max_cycles"] <= 8)
    _reject_if(type(scope["seed"]) is not int or not 0 <= scope["seed"] < 2**31)
    _reject_if(scope["scale"] * (1 + scope["specificity"] // 2) != train_count)
    _reject_if(len(_canonical_bytes(scope)) > 16_384)


def _check_structure(manifest: dict[str, Any]) -> tuple[int, int]:
    _reject_if(set(manifest) != _PAYLOAD_FIELDS | {"source_sha256"})
    _reject_if(type(manifest["schema"]) is not str or
               manifest["schema"] != PRODUCER_SCHEMA)
    _reject_if(type(manifest["cycle_index"]) is not int or
               manifest["cycle_index"] != 0)
    digest = manifest["source_sha256"]
    _reject_if(type(digest) is not str or _HEX64.fullmatch(digest) is None)
    payload = {field: manifest[field] for field in _PAYLOAD_FIELDS}
    _reject_if(_sha256(_DOMAIN + _canonical_bytes(payload)) != digest)

    budget = manifest["pilot_budget"]
    _reject_if(type(budget) is not dict or set(budget) != _BUDGET_FIELDS)
    for field, lower, upper in (("seed", 0, 2**31 - 1), ("train_count", 2, 32),
                                ("eval_count", 2, 32), ("steps", 1, 128)):
        _reject_if(type(budget[field]) is not int or
                   not lower <= budget[field] <= upper)
    _reject_if(type(budget["tiers"]) is not list or budget["tiers"] != [1])
    _check_scope(manifest["scope"], budget["train_count"])

    rooms = manifest["train_rooms"]
    feedback = manifest["adaptive_feedback_rows"]
    _reject_if(type(rooms) is not list or len(rooms) != budget["train_count"])
    _reject_if(type(feedback) is not list or len(feedback) != len(rooms))
    initial = manifest["initial_checkpoint_sha256"]
    _reject_if(type(initial) is not str or _HEX64.fullmatch(initial) is None)
    room_ids: set[str] = set()
    case_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    checkpoint_before = initial
    for index, (room, row) in enumerate(zip(rooms, feedback)):
        _reject_if(type(room) is not dict or set(room) != _ROOM_FIELDS)
        room_id = room["room_id"]
        case_id = room["case_id"]
        prompt_sha = room["prompt_sha256"]
        _reject_if(type(room_id) is not str or _ROOM_ID.fullmatch(room_id) is None)
        _reject_if(type(case_id) is not str or _CASE_ID.fullmatch(case_id) is None or
                   _CASE_ID.fullmatch(case_id).group(1) != room_id)
        _reject_if(type(prompt_sha) is not str or _HEX64.fullmatch(prompt_sha) is None)
        _reject_if(room_id in room_ids or case_id in case_ids or
                   prompt_sha in prompt_hashes)
        room_ids.add(room_id)
        case_ids.add(case_id)
        prompt_hashes.add(prompt_sha)
        _reject_if(room["skill"] != "arithmetic" or
                   room["tactic"] not in ("practice", "contrast", "transfer"))
        _reject_if(type(room["difficulty"]) is not int or
                   not 1 <= room["difficulty"] <= 5)
        _reject_if(room["planner_source"] != (
            "initial_train_bootstrap" if index == 0 else "policy_plus_adaptive_train"
        ))
        _reject_if(type(row) is not list or len(row) != 6)
        _reject_if(any(type(value) is not str for pos, value in enumerate(row) if pos != 3))
        _reject_if(row[:3] != [room_id, case_id, prompt_sha])
        _reject_if(type(row[3]) is not int or row[3] not in (0, 1))
        _reject_if(row[4] != checkpoint_before or
                   _HEX64.fullmatch(row[5]) is None)
        checkpoint_before = row[5]

    exposures = manifest["ordered_optimizer_exposures"]
    _reject_if(type(exposures) is not list or len(exposures) != budget["steps"])
    ordered_indices: list[int] = []
    epoch = 0
    while len(ordered_indices) < budget["steps"]:
        shuffled = list(range(len(rooms)))
        random.Random(f'{budget["seed"]}:epoch:{epoch}:tier:1').shuffle(shuffled)
        ordered_indices.extend(shuffled)
        epoch += 1
    for index, pair in zip(ordered_indices, exposures):
        _reject_if(type(pair) is not list or len(pair) != 2 or
                   pair != [rooms[index]["case_id"], rooms[index]["prompt_sha256"]])
    return len(rooms), len(exposures)


def inspect_adaptive_minotaur_source(
    *, manifest_bytes: bytes, expected_manifest_sha256: str,
    producer_git_head: str, producer_git_tree: str,
    explicit_adaptive_opt_in: bool, staged_roster_present: bool,
    forbidden_case_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Inspect public source shape without emitting or trusting training secrets.

    Inputs including revision/hash/forbidden IDs are *caller assertions*.
    ``structural_source_consistent`` never confers independent custody.
    Malformed inputs fail closed in the same grant state with no echoed data.
    """
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": UNVERIFIED_REJECTED,
        "structural_source_consistent": False,
        "source_authenticated": False,
        "rights_holder_authorized": False,
        "data_steward_authorized": False,
        "private_evaluator_custody_authenticated": False,
        "q6_monotonic_witness_verified": False,
        "host_process_custody_verified": False,
        "gpu_lease_granted": False,
        "optimizer_steps_verified": False,
        "physical_training_authorized": False,
        "model_training_granted": False,
        "reason": "source_unverified",
    }
    try:
        _reject_if(type(manifest_bytes) is not bytes or
                   not 0 < len(manifest_bytes) <= 262_144)
        _reject_if(type(expected_manifest_sha256) is not str or
                   _HEX64.fullmatch(expected_manifest_sha256) is None or
                   _sha256(manifest_bytes) != expected_manifest_sha256)
        _reject_if(producer_git_head != PINNED_LABYRINTH_HEAD or
                   producer_git_tree != PINNED_LABYRINTH_TREE)
        _reject_if(type(explicit_adaptive_opt_in) is not bool or
                   explicit_adaptive_opt_in is not True)
        _reject_if(type(staged_roster_present) is not bool or staged_roster_present)
        _reject_if(type(forbidden_case_ids) is not tuple or
                   len(forbidden_case_ids) > 1_024 or
                   any(type(case_id) is not str or len(case_id) > 128 or
                       _CASE_ID.fullmatch(case_id) is None
                       for case_id in forbidden_case_ids) or
                   len(set(forbidden_case_ids)) != len(forbidden_case_ids))
        manifest = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=_unique_keys,
            parse_constant=_nonfinite,
        )
        _reject_if(type(manifest) is not dict)
        room_count, exposure_count = _check_structure(manifest)
        _reject_if(bool({row["case_id"] for row in manifest["train_rooms"]}
                        & set(forbidden_case_ids)))
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError,
            KeyError, RecursionError, OverflowError):
        return {**result, "reason": "source_shape_or_opt_in_rejected"}
    return {
        **result,
        "structural_source_consistent": True,
        "train_room_count": room_count,
        "optimizer_exposure_count": exposure_count,
        "reason": "source_shape_only_no_independent_provenance_or_custody",
    }
