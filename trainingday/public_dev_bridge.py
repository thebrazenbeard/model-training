"""Pinned, source-only bridge from original curriculum to the public DEV scorer.

This is a case-schema adapter, NOT permission to train or evidence of transfer.
Independent predictions for all five negative-control arms remain mandatory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Mapping

GENERATOR = "trainingday.original_curriculum:v1"
SOURCE_SHA256 = "2825b6d86a706ba614aab5da9ae67861c10328f4c1242b28f94c53db715c41a2"
TRAIN_SHA256 = "5a1f3064b840ac2f157bc12320d4c3a6e32415f3d180dfd3184933985b003acd"
DEV_SHA256 = "db52fd2fb9080d8f8b85a06c84e8104cc645e630dc4ac36c4edd0778338a7a31"
TRAIN_COUNT = 432
DEV_COUNT = 144
DOMAIN_COUNTS = {name: (72, 24) for name in (
    "logic", "math", "language", "tool_decisions", "uncertainty", "generalization"
)}
ARMS = ("trained", "untrained", "sham", "shuffled_labels", "corrupted_labels")
ROW_FIELDS = frozenset((
    "id", "case_id", "domain", "family_id", "template_id", "source_id",
    "archetype_id", "variant", "split", "prompt", "answer", "rights_status",
    "training_authorized", "provenance",
))


class BridgeError(ValueError):
    """Untrusted, incompatible or noncanonical candidate input."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_digest(value: object) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BridgeError("nonempty textual field required")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise BridgeError("duplicate JSON object key")
        obj[key] = value
    return obj


def _json(data: bytes) -> object:
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicate_keys,
                          parse_constant=lambda _: (_ for _ in ()).throw(BridgeError("nonfinite JSON")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeError("invalid JSON") from exc


def _rows(data: bytes, expected_sha: str, expected_count: int, split: str) -> list[dict]:
    if not isinstance(data, bytes) or len(data) > 2_000_000 or _sha(data) != expected_sha:
        raise BridgeError(f"{split} candidate bytes differ from pinned reviewed source")
    if not data.endswith(b"\n"):
        raise BridgeError("candidate JSONL must end with newline")
    parsed = [_json(line) for line in data.splitlines()]
    if len(parsed) != expected_count:
        raise BridgeError(f"{split} row count differs from reviewed source")
    return parsed


def _check_row(row: object, split: str) -> dict:
    if not isinstance(row, dict) or set(row) != ROW_FIELDS:
        raise BridgeError("noncanonical original-curriculum row fields")
    if row["split"] != split or row["rights_status"] != "NOT_ADMITTED" or row["training_authorized"] is not False:
        raise BridgeError("split or candidate training authority changed")
    if row["domain"] not in DOMAIN_COUNTS or type(row["variant"]) is not int or not 0 <= row["variant"] < 3:
        raise BridgeError("noncanonical domain or variant")
    provenance = row["provenance"]
    if provenance != {"kind": "source_authored", "generator": GENERATOR, "external_corpus": False}:
        raise BridgeError("noncanonical candidate provenance")
    for field in ("id", "case_id", "domain", "family_id", "template_id", "source_id",
                  "archetype_id", "prompt", "answer"):
        _text(row[field])
    if row["id"] != row["case_id"] or row["template_id"] != row["family_id"]:
        raise BridgeError("case/family/template identities disagree")
    if row["archetype_id"] != f'oc-v1-{row["domain"]}' or not row["source_id"].startswith(GENERATOR + "/"):
        raise BridgeError("synthetic archetype/source identity changed")
    return row


def _manifest(data: bytes, train: bytes, dev: bytes) -> dict:
    if not isinstance(data, bytes) or len(data) > 65_536:
        raise BridgeError("manifest must be bounded bytes")
    manifest = _json(data)
    if not isinstance(manifest, dict):
        raise BridgeError("manifest must be an object")
    for name, expected in (("schema_version", "original-curriculum-candidate-v1"),
                           ("generator", GENERATOR), ("generator_source_sha256", SOURCE_SHA256),
                           ("rights_status", "NOT_ADMITTED"), ("training_authorized", False),
                           ("archetype_overlap", True),
                           ("evaluation_scope", "same-archetype diagnostic only; NOT unseen-archetype transfer")):
        if manifest.get(name) != expected or (name in ("training_authorized", "archetype_overlap")
                                               and type(manifest[name]) is not bool):
            raise BridgeError(f"candidate manifest {name} not pinned")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {"train.jsonl", "dev.jsonl"}:
        raise BridgeError("manifest file inventory changed")
    for name, payload, count in (("train.jsonl", train, TRAIN_COUNT), ("dev.jsonl", dev, DEV_COUNT)):
        if files[name] != {"rows": count, "sha256": _sha(payload)}:
            raise BridgeError(f"manifest {name} byte witness invalid")
    return manifest


def build_w31_public_scaffold(train_jsonl: bytes, dev_jsonl: bytes, manifest_json: bytes) -> dict:
    """Adapt fixed, rights-pending source bytes; never invent predictions.

    #329 uses the literal `split=transfer` as an API discriminator. Here the
    generator shares archetypes across TRAIN/DEV, so the *actual* result scope
    must remain SAME_ARCHETYPE_DIAGNOSTIC_ONLY, regardless of that API token.
    """
    train = _rows(train_jsonl, TRAIN_SHA256, TRAIN_COUNT, "train")
    dev = _rows(dev_jsonl, DEV_SHA256, DEV_COUNT, "dev")
    _manifest(manifest_json, train_jsonl, dev_jsonl)
    seen_ids, seen_prompts = set(), set()
    groups = {"train": {field: set() for field in ("family_id", "source_id", "template_id")},
              "dev": {field: set() for field in ("family_id", "source_id", "template_id")}}
    counts = Counter()
    for split, rows in (("train", train), ("dev", dev)):
        for item in rows:
            row = _check_row(item, split)
            ident, prompt = _text(row["case_id"]), _text(row["prompt"])
            if ident in seen_ids or prompt in seen_prompts:
                raise BridgeError("duplicate normalized case or prompt")
            seen_ids.add(ident)
            seen_prompts.add(prompt)
            for field in groups[split]:
                groups[split][field].add(_text(row[field]))
            counts[(row["domain"], split)] += 1
    for domain, (train_count, dev_count) in DOMAIN_COUNTS.items():
        if counts[(domain, "train")] != train_count or counts[(domain, "dev")] != dev_count:
            raise BridgeError("per-domain source split counts changed")
    for field in groups["train"]:
        if groups["train"][field] & groups["dev"][field]:
            raise BridgeError(f"train/dev {field} leakage")
    cases = [{"case_id": row["case_id"], "prompt": row["prompt"],
              "expected_semantic": row["answer"], "stratum": row["domain"],
              "family_id": row["family_id"], "source_group_id": row["source_id"],
              "template_id": row["template_id"]} for row in dev]
    train_rows = [{"prompt": row["prompt"], "family_id": row["family_id"],
                   "source_group_id": row["source_id"], "template_id": row["template_id"]}
                  for row in train]
    scaffold = {
        "bridge_version": "public-dev-original-curriculum-v1",
        "fixture": {"visibility": "public", "purpose": "development", "split": "transfer",
                    "cases": cases},
        "train_manifest": {"manifest_id": f"oc-v1-unadmitted-{TRAIN_SHA256}",
                           "generator": GENERATOR, "dataset_sha256": TRAIN_SHA256,
                           "rights_status": "NOT_ADMITTED", "training_authorized": False},
        "train_rows": train_rows,
        "actual_evaluation_scope": "SAME_ARCHETYPE_DIAGNOSTIC_ONLY",
        "scorer_split_token_note": "transfer is #329's API token; no withheld archetypes or Q6 transfer proven",
        "prediction_status": "MISSING_INDEPENDENT_MODEL_PREDICTIONS",
        "ready_for_training": False,
        "source_byte_witnesses": {"generator_source_sha256": SOURCE_SHA256,
                                  "train_sha256": TRAIN_SHA256, "dev_sha256": DEV_SHA256},
    }
    scaffold["scaffold_sha256"] = _content_digest(scaffold)
    return scaffold


def attach_independent_predictions(
    scaffold: Mapping,
    predictions: Mapping,
    *,
    train_jsonl: bytes | None = None,
    dev_jsonl: bytes | None = None,
    manifest_json: bytes | None = None,
) -> dict:
    """Assemble #329 input only against a newly verified canonical source.

    A scaffold digest is unkeyed and therefore not proof of source origin:
    callers must supply the original pinned bytes again. Any forged gold,
    case/stratum mapping or train row is rejected even if its digest is updated.
    Independently generated model predictions themselves remain UNVERIFIED.
    """
    if not isinstance(scaffold, Mapping) or scaffold.get("bridge_version") != "public-dev-original-curriculum-v1":
        raise BridgeError("bridge scaffold required")
    if any(value is None for value in (train_jsonl, dev_jsonl, manifest_json)):
        raise BridgeError("pinned train/dev/manifest source bytes required at attachment")
    canonical_scaffold = build_w31_public_scaffold(train_jsonl, dev_jsonl, manifest_json)
    if scaffold != canonical_scaffold:
        raise BridgeError("scaffold differs from freshly revalidated pinned source")
    # All subsequent fields come from a fresh canonical reconstruction, not
    # from a caller's potentially mutated dictionary.
    scaffold = canonical_scaffold
    if scaffold.get("actual_evaluation_scope") != "SAME_ARCHETYPE_DIAGNOSTIC_ONLY" or scaffold.get("ready_for_training") is not False:
        raise BridgeError("candidate authority/scope changed")
    fixture = scaffold.get("fixture")
    if not isinstance(fixture, Mapping) or any(fixture.get(k) != v for k, v in
                                              (("visibility", "public"), ("purpose", "development"),
                                               ("split", "transfer"))):
        raise BridgeError("nonpublic or misclassified fixture")
    if any(fixture.get(k) for k in ("sealed", "protected", "frozen")):
        raise BridgeError("protected fixture claims forbidden")
    manifest = scaffold.get("train_manifest")
    if not isinstance(manifest, Mapping) or manifest.get("rights_status") != "NOT_ADMITTED" or manifest.get("training_authorized") is not False:
        raise BridgeError("training rights remain unadmitted")
    if scaffold.get("source_byte_witnesses") != {"generator_source_sha256": SOURCE_SHA256,
                                                 "train_sha256": TRAIN_SHA256,
                                                 "dev_sha256": DEV_SHA256}:
        raise BridgeError("source byte witness changed")
    try:
        if scaffold.get("scaffold_sha256") != _content_digest({k: v for k, v in scaffold.items()
                                                                if k != "scaffold_sha256"}):
            raise BridgeError("scaffold changed after generation")
    except (TypeError, ValueError) as exc:
        raise BridgeError("scaffold not canonical JSON") from exc
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BridgeError("scaffold cases missing")
    ids = [row.get("case_id") for row in cases if isinstance(row, Mapping)]
    if len(ids) != len(cases) or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise BridgeError("invalid scaffold IDs")
    if not isinstance(predictions, Mapping) or set(predictions) != set(ARMS):
        raise BridgeError("all five independent prediction arms required")
    clean = {}
    for arm in ARMS:
        rows = predictions[arm]
        if not isinstance(rows, list) or len(rows) != len(ids):
            raise BridgeError("prediction count mismatch")
        observed = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"case_id", "semantic_answer"}:
                raise BridgeError("prediction row must have only case_id and semantic_answer")
            if not isinstance(row["case_id"], str) or row["case_id"] not in ids or row["case_id"] in observed:
                raise BridgeError("prediction case coverage mismatch")
            observed.add(row["case_id"])
            _text(row["semantic_answer"])
        if observed != set(ids):
            raise BridgeError("prediction case coverage mismatch")
        clean[arm] = copy.deepcopy(rows)
    return {"fixture": copy.deepcopy(scaffold["fixture"]),
            "train_manifest": copy.deepcopy(scaffold["train_manifest"]),
            "train_rows": copy.deepcopy(scaffold["train_rows"]),
            "predictions": clean,
            "provenance": {"dataset_generator": GENERATOR, "dataset_sha256": TRAIN_SHA256,
                           "evaluation_bank_id": "UNKNOWN", "evaluation_custodian": "UNKNOWN"}}
