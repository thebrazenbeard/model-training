"""R28 bounded PUBLIC TOY admission adversary; not a production training gate.

The only filesystem work is disposable synthetic corpus staging inside a fresh
temporary directory. No supplied path is materialized until an exact, small
allowlist check passes. No source code, archive member, or model is executed.

Even successful local admission is a caller-declared source check, not proof
of independent rights, evaluation isolation, model improvement, or GPU lease.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from trainingday.admission import AdmissionError, admit
from trainingday.r27_supply_chain import assess_supply_chain


MAX_MANIFEST_BYTES = 16 * 1024
MAX_SHARD_BYTES = 32 * 1024
MAX_ROW_BYTES = 8 * 1024
MAX_TOTAL_SHARD_BYTES = 64 * 1024
MAX_ROWS_PER_SHARD = 32
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
_SPLITS = ("train", "dev", "test")
_SHARDS = frozenset(f"shards/{name}.jsonl" for name in _SPLITS)
_HEX40 = re.compile(r"[a-f0-9]{40}\Z")
_HEX64 = re.compile(r"[a-f0-9]{64}\Z")
_REQUIREMENTS = "numpy==2.5.1\n"


@dataclass(frozen=True)
class PublicAdmissionReceipt:
    reason: str
    status: str = "REJECTED"
    selected_count: int = 0
    source_admitted: bool = False
    rights_independently_authenticated: bool = False
    gpu_training_authorized: bool = False
    model_release_authorized: bool = False


@dataclass(frozen=True)
class AdversarialPublicCase:
    label: str
    manifest: bytes
    shards: dict[str, bytes]
    archive: bytes | None = None


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_wire(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _reject(reason: str) -> PublicAdmissionReceipt:
    return PublicAdmissionReceipt(reason=reason)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    obj: dict = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _nonfinite(value: str):
    raise ValueError("nonfinite JSON constant")


def _strict_json(raw: bytes) -> object:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                      parse_constant=_nonfinite)


def _clean_wire_values(value: object, *, depth: int = 0) -> bool:
    """Refuse surrogate-escape strings and deeply nested hostile JSON shapes."""
    if depth > 12:
        return False
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError:
            return False
        return True
    if type(value) is dict:
        return all(_clean_wire_values(k, depth=depth + 1)
                   and _clean_wire_values(v, depth=depth + 1)
                   for k, v in value.items())
    if type(value) is list:
        return all(_clean_wire_values(v, depth=depth + 1) for v in value)
    if type(value) is float:
        return math.isfinite(value)
    return type(value) in (bool, int, float, type(None))


def _archive_static_only(payload: bytes) -> bool:
    """Reuse R27 metadata-only archive guard, never unzip/extract a member."""
    requirements = _REQUIREMENTS
    declaration = {
        "schema": "TRAININGDAY_R27_STATIC_SUPPLY_CHAIN_V1",
        "requirements_sha256": _sha(requirements.encode("utf-8")),
        "components": [{
            "name": "toy-archive", "role": "archive",
            "locator": "public/toy-archive", "source_revision": "a" * 40,
            "artifact_sha256": _sha(payload), "declared_bytes": len(payload),
            "license_spdx": "MIT",
        }],
    }
    return assess_supply_chain(declaration, requirements_text=requirements,
                               observed_artifacts={"toy-archive": payload}).reason == "STATIC_BYTES_MATCH_ONLY"


def assess_public_admission_fixture(
    manifest: object, shards: object, *, archive: object = None,
) -> PublicAdmissionReceipt:
    """Probe actual admission through tiny, in-memory-bounded public fixtures.

    NEVER call this an authenticated training admission: it accepts invented
    public provenance and stages only a toy test. Direct production callers of
    ``trainingday.admission.admit`` are NOT changed or made bounded by R28.
    """
    if type(manifest) is not bytes:
        return _reject("MANIFEST_TYPE_INVALID")
    if len(manifest) > MAX_MANIFEST_BYTES:
        return _reject("MANIFEST_SIZE_LIMIT")
    if type(shards) is not dict or len(shards) != len(_SHARDS):
        return _reject("SHARD_SET_INVALID")
    if shards.keys() != _SHARDS or any(type(k) is not str or type(v) is not bytes
                                       for k, v in shards.items()):
        return _reject("SHARD_SET_INVALID")
    if any(len(content) > MAX_SHARD_BYTES for content in shards.values()):
        return _reject("SHARD_SIZE_LIMIT")
    if sum(map(len, shards.values())) > MAX_TOTAL_SHARD_BYTES:
        return _reject("SHARD_TOTAL_SIZE_LIMIT")
    try:
        document = _strict_json(manifest)
        if not _clean_wire_values(document):
            return _reject("MANIFEST_JSON_INVALID")
    except (ValueError, UnicodeError, RecursionError, json.JSONDecodeError):
        return _reject("MANIFEST_JSON_INVALID")
    if (type(document) is not dict
            or document.keys() != {"schema", "corpus_id", "source_commit", "shards"}
            or document["schema"] != "VERA_TRAININGDAY_ADMITTED_SHARDS_V1"
            or type(document["corpus_id"]) is not str or not document["corpus_id"].strip()
            or type(document["source_commit"]) is not str
            or _HEX40.fullmatch(document["source_commit"]) is None
            or type(document["shards"]) is not list or len(document["shards"]) != 3):
        return _reject("MANIFEST_SHAPE_INVALID")
    seen_specs: set[str] = set()
    for spec in document["shards"]:
        if type(spec) is not dict or spec.keys() != {"path", "split", "rows", "sha256"}:
            return _reject("SHARD_SPEC_INVALID")
        rel = spec["path"]
        split = spec["split"]
        if type(rel) is not str or rel not in _SHARDS or type(split) is not str or rel != f"shards/{split}.jsonl":
            return _reject("UNSAFE_SHARD_PATH")
        if rel in seen_specs:
            return _reject("DUPLICATE_SHARD_SPEC")
        seen_specs.add(rel)
        if (type(spec["rows"]) is not int or not (1 <= spec["rows"] <= MAX_ROWS_PER_SHARD)
                or type(spec["sha256"]) is not str
                or _HEX64.fullmatch(spec["sha256"]) is None):
            return _reject("SHARD_SPEC_INVALID")
    if seen_specs != _SHARDS:
        return _reject("SHARD_SET_INVALID")
    for rel, content in shards.items():
        if not content.endswith(b"\n") or b"\r" in content or b"\x00" in content:
            return _reject("JSONL_ROW_INVALID")
        lines = content.splitlines(keepends=True)
        if not 1 <= len(lines) <= MAX_ROWS_PER_SHARD:
            return _reject("SHARD_ROW_LIMIT")
        if any(not line.strip() or len(line) > MAX_ROW_BYTES for line in lines):
            return _reject("JSONL_ROW_SIZE_OR_BLANK")
        for line in lines:
            try:
                row = _strict_json(line)
                if type(row) is not dict or not _clean_wire_values(row):
                    return _reject("JSONL_ROW_INVALID")
            except (UnicodeError, ValueError, RecursionError, json.JSONDecodeError):
                return _reject("JSONL_ROW_INVALID")
    if archive is not None:
        if type(archive) is not bytes or len(archive) > MAX_ARCHIVE_BYTES:
            return _reject("ARCHIVE_STATIC_METADATA_REJECTED")
        if not _archive_static_only(archive):
            return _reject("ARCHIVE_STATIC_METADATA_REJECTED")

    # Staging paths are constant, not interpolated from caller-supplied paths.
    with tempfile.TemporaryDirectory(prefix="trainingday-r28-public-adversary-") as temp:
        root = Path(temp) / "corpus"
        root.mkdir()
        (root / "shards").mkdir()
        (root / "manifest.json").write_bytes(manifest)
        for split in _SPLITS:
            rel = f"shards/{split}.jsonl"
            (root / rel).write_bytes(shards[rel])
        work = Path(temp) / "scratch"
        work.mkdir()
        try:
            summary, selected = admit(root / "manifest.json", work, sample_size=2)
        except (AdmissionError, OSError, UnicodeError, ValueError):
            return _reject("UPSTREAM_ADMISSION_REJECTED")
    if summary.get("status") != "ADMITTED_SOURCE_ONLY_NOT_TRAINED" or len(selected) != 2:
        return _reject("UPSTREAM_ADMISSION_REJECTED")
    return PublicAdmissionReceipt(reason="PUBLIC_TOY_ADMISSION_ONLY",
                                  status="SOURCE_ADMITTED", selected_count=2,
                                  source_admitted=True)


def _row(case_id: str, split: str) -> dict:
    return {
        "id": case_id, "split": split, "family": "arithmetic",
        "prompt": f"What is independent task {case_id}?",
        "answer": f"Toy answer {case_id}",
        "training_use": "ADMITTED" if split == "train" else "EVALUATION_ONLY",
        "evaluation_holdout": False,
        "provenance": {
            "source_id": "public/procedural-toy", "source_revision": "a" * 40,
            "license": "MIT", "origin_class": "HUMAN_CROWDSOURCED",
            "origin_evidence": "public/example-synthetic-only",
            "rights_review": "PASS", "privacy_review": "PASS",
            "openai_output": False,
        },
    }


def public_fixture() -> tuple[bytes, dict[str, bytes]]:
    """Generate a deterministic all-public/no-PII toy fixture in memory."""
    shards: dict[str, bytes] = {}
    specs: list[dict] = []
    for split, rows in (
        ("train", [_row("train0", "train"), _row("train1", "train")]),
        ("dev", [_row("dev0", "dev")]),
        ("test", [_row("test0", "test")]),
    ):
        rel = f"shards/{split}.jsonl"
        payload = b"".join(_json_wire(row) + b"\n" for row in rows)
        shards[rel] = payload
        specs.append({"path": rel, "split": split, "rows": len(rows), "sha256": _sha(payload)})
    document = {
        "schema": "VERA_TRAININGDAY_ADMITTED_SHARDS_V1",
        "corpus_id": "r28-public-toy-input-1", "source_commit": "b" * 40,
        "shards": specs,
    }
    return _json_wire(document), shards


def _mutated_row_case(label: str, split: str, change) -> AdversarialPublicCase:
    raw, original = public_fixture()
    rows = [_strict_json(line) for line in original[f"shards/{split}.jsonl"].splitlines()]
    change(rows)
    shards = dict(original)
    rel = f"shards/{split}.jsonl"
    shards[rel] = b"".join(_json_wire(row) + b"\n" for row in rows)
    document = _strict_json(raw)
    for spec in document["shards"]:
        if spec["path"] == rel:
            spec["rows"] = len(rows)
            spec["sha256"] = _sha(shards[rel])
    return AdversarialPublicCase(label, _json_wire(document), shards)


def _mutated_manifest_case(label: str, change) -> AdversarialPublicCase:
    raw, shards = public_fixture()
    document = _strict_json(raw)
    change(document)
    return AdversarialPublicCase(label, _json_wire(document), shards)


def _archive(entries: list[tuple[str, bytes]]) -> bytes:
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in entries:
            z.writestr(name, content)
    return sink.getvalue()


def generate_adversarial_public_cases() -> tuple[AdversarialPublicCase, ...]:
    """Stable minimized abuse samples for deterministic CPU regression use.

    No fuzzed text is interpreted as Python or shell code. This list is bounded,
    not coverage-guided stochastic fuzzing or real-host sandbox verification.
    """
    raw, shards = public_fixture()
    damaged = dict(shards)
    damaged["shards/train.jsonl"] = b"\xff\n"
    duplicated = dict(shards)
    duplicated["shards/train.jsonl"] = b'{"id":"x","id":"y"}\n'
    cases = [
        _mutated_manifest_case("path-traversal", lambda d: d["shards"][0].update(path="shards/../../escape.jsonl")),
        _mutated_manifest_case("windows-backslash-path", lambda d: d["shards"][0].update(path="shards\\train.jsonl")),
        _mutated_manifest_case("absolute-path", lambda d: d["shards"][0].update(path="/tmp/escape.jsonl")),
        _mutated_manifest_case("split-path-forgery", lambda d: d["shards"][0].update(split="dev")),
        _mutated_manifest_case("duplicate-shard-spec", lambda d: d["shards"][1].update(d["shards"][0])),
        _mutated_manifest_case("rows-out-of-bounds", lambda d: d["shards"][0].update(rows=100000)),
        AdversarialPublicCase("manifest-bloat", b" " * (MAX_MANIFEST_BYTES + 1), shards),
        AdversarialPublicCase("nonfinite-manifest", b'{"schema":NaN}', shards),
        AdversarialPublicCase("duplicate-json-key", b'{"schema":1,"schema":2}', shards),
        AdversarialPublicCase("utf8-invalid-shard", raw, damaged),
        AdversarialPublicCase("row-duplicate-key", raw, duplicated),
        _mutated_row_case("rights-revoked", "train", lambda rows: rows[0]["provenance"].update(rights_review="UNKNOWN")),
        _mutated_row_case("privacy-revoked", "train", lambda rows: rows[0]["provenance"].update(privacy_review="FAIL")),
        _mutated_row_case("generated-origin", "train", lambda rows: rows[0]["provenance"].update(origin_class="MODEL_GENERATED")),
        _mutated_row_case("inadmissible-license", "train", lambda rows: rows[0]["provenance"].update(license="UNKNOWN")),
        _mutated_row_case("openai-output", "train", lambda rows: rows[0]["provenance"].update(openai_output=True)),
        _mutated_row_case("evaluation-split-forgery", "dev", lambda rows: rows[0].update(split="train")),
        _mutated_row_case("holdout-in-training", "train", lambda rows: rows[0].update(evaluation_holdout=True)),
        _mutated_row_case("train-use-forgery", "dev", lambda rows: rows[0].update(training_use="ADMITTED")),
        _mutated_row_case("duplicate-id-across-splits", "dev", lambda rows: rows[0].update(id="train0")),
        _mutated_row_case("normalized-prompt-leak", "test", lambda rows: rows[0].update(prompt=" WHAT   IS INDEPENDENT TASK TRAIN0? ")),
        AdversarialPublicCase("zip-path-escape", raw, shards, _archive([("../escape", b"x")])),
        AdversarialPublicCase("zip-compression-bomb", raw, shards, _archive([("data.bin", b"a" * 200000)])),
        AdversarialPublicCase("zip-nested-archive", raw, shards, _archive([("nested.zip", b"PK")])),
    ]
    return tuple(cases)
