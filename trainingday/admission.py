"""Streaming, fail-closed training-corpus admission.

This module validates a producer's public sharded JSONL before a model runner
may read *any* selected training example. It does not train neural weights.
The sqlite index is stored in a disposable caller-owned directory so intake
can scale beyond RAM. It does not persist a retrieval database.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath


SCHEMA = "VERA_TRAININGDAY_ADMITTED_SHARDS_V1"
LICENSE_ALLOWLIST = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
    "CC-BY-4.0", "CC0-1.0", "Unlicense", "USER_EXPLICIT_GRANT",
}
ORIGIN_ALLOWLIST = {"HUMAN_CROWDSOURCED", "INDEPENDENT_PROCEDURAL", "USER_AUTHORED"}
SPLITS = {"train", "dev", "test"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_JSONL_ROW_BYTES = 1024 * 1024
MAX_JSON_DEPTH = 64


class AdmissionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject exact and Unicode/case-equivalent JSON member ambiguity.

    This applies at every object depth, including provenance rights fields.
    Untrusted duplicate declarations must never be resolved by last-key-wins.
    """
    result: dict = {}
    normalized: set[str] = set()
    for key, value in pairs:
        alias = unicodedata.normalize("NFKC", key).casefold()
        _require(alias not in normalized, "duplicate/ambiguous JSON key")
        normalized.add(alias)
        result[key] = value
    return result


def _reject_nonfinite(_: str) -> None:
    raise AdmissionError("nonfinite JSON constant")


def _validate_json_wire(value: object, depth: int = 0) -> None:
    """Bound nesting and refuse UTF-8-invalid escaped surrogate codepoints."""
    _require(depth <= MAX_JSON_DEPTH, "JSON depth limit")
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise AdmissionError("invalid JSON surrogate") from exc
    elif isinstance(value, dict):
        for key, child in value.items():
            _validate_json_wire(key, depth + 1)
            _validate_json_wire(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _validate_json_wire(child, depth + 1)
    elif isinstance(value, float):
        _require(math.isfinite(value), "nonfinite JSON number")


def _strict_json(raw: bytes) -> object:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object,
                           parse_constant=_reject_nonfinite)
        _validate_json_wire(value)
        return value
    except AdmissionError:
        raise
    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        raise AdmissionError("invalid UTF-8 JSON") from exc


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for data in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _relative_file(root: Path, rel: str) -> Path:
    _require(isinstance(rel, str) and rel == rel.strip() and bool(rel), "invalid shard path")
    _require("\\" not in rel and ":" not in rel and "\x00" not in rel, "nonportable shard path")
    posix = PurePosixPath(rel)
    _require(rel == posix.as_posix() and not posix.is_absolute()
             and all(p not in {".", "..", ""} for p in posix.parts),
             "shard path traversal")
    _require(posix.parts[0] == "shards" and rel.endswith(".jsonl"),
             "shard must be a shards/*.jsonl path")
    path = root.joinpath(*posix.parts)
    _require(not root.is_symlink(), "symlinked corpus root")
    cursor = root
    for part in posix.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), "missing or symlinked shard")
    _require(path.is_file(), "missing or symlinked shard")
    _require(path.resolve().is_relative_to(root.resolve()), "shard escapes corpus root")
    return path


def _canonical_prompt(prompt: str) -> bytes:
    # A conservative exact-text normalized leakage check, not semantic matching.
    return " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold().encode("utf-8")


def _validate_row(row: object, split: str) -> tuple[str, str, str, str]:
    _require(isinstance(row, dict), "row is not an object")
    for key in ("id", "family", "prompt", "answer"):
        _require(isinstance(row.get(key), str) and bool(row[key].strip()), f"row {key} missing")
    _require(row.get("split") == split, "row split disagrees with shard")
    expected_use = "ADMITTED" if split == "train" else "EVALUATION_ONLY"
    # Explicit Boolean False is required; missing, integer 0 and other
    # falsy substitutes must not silently downgrade a protected row.
    _require(row.get("evaluation_holdout") is False, "evaluation_holdout not explicitly false")
    _require(row.get("training_use") == expected_use, "unadmitted or protected case")
    provenance = row.get("provenance")
    _require(isinstance(provenance, dict), "row provenance missing")
    for key in ("source_id", "source_revision", "license", "origin_class", "origin_evidence"):
        _require(isinstance(provenance.get(key), str) and bool(provenance[key].strip()),
                 f"row provenance {key} missing")
    _require(provenance["license"] in LICENSE_ALLOWLIST, "row license not admitted")
    _require(provenance["origin_class"] in ORIGIN_ALLOWLIST, "row origin not independently admitted")
    _require(provenance.get("rights_review") == "PASS"
             and provenance.get("privacy_review") == "PASS", "row review not PASS")
    _require(provenance.get("openai_output") is False, "OpenAI model output not admitted")
    if provenance["origin_class"] == "INDEPENDENT_PROCEDURAL":
        _require(isinstance(provenance.get("oracle_source_sha256"), str)
                 and HEX64.fullmatch(provenance["oracle_source_sha256"]) is not None,
                 "procedural answer lacks verified oracle binding")
    return (row["id"], row["family"], provenance["license"], provenance["origin_class"])


def admit(manifest_path: Path, work_root: Path, *, sample_size: int = 64) -> tuple[dict, list[dict]]:
    """Validate *every* shard and case, select train rows by stable min-hash.

    The selected rows are held in memory; every row is indexed only by short
    hashes in on-disk sqlite. The caller must use a private empty directory.
    No output is written if any source fails, except disposable sqlite scratch.
    """
    manifest_path, work_root = Path(manifest_path), Path(work_root)
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "manifest missing/symlinked")
    _require(work_root.is_dir() and not work_root.is_symlink(), "work root missing/symlinked")
    _require(isinstance(sample_size, int) and 1 <= sample_size <= 1024, "invalid sample bound")
    # Never materialize arbitrarily large attacker-controlled manifest bytes.
    with manifest_path.open("rb") as reader:
        manifest_raw = reader.read(MAX_MANIFEST_BYTES + 1)
    _require(len(manifest_raw) <= MAX_MANIFEST_BYTES, "manifest size limit")
    manifest = _strict_json(manifest_raw)
    _require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA, "manifest schema mismatch")
    _require(isinstance(manifest.get("corpus_id"), str) and manifest["corpus_id"].strip(), "corpus id missing")
    _require(isinstance(manifest.get("source_commit"), str)
             and HEX40.fullmatch(manifest["source_commit"]) is not None,
             "source Git commit not pinned")
    shards = manifest.get("shards")
    _require(isinstance(shards, list) and bool(shards), "no shards")
    _require(len(shards) <= 1000000, "too many shards")
    names = set()
    normalized_names = set()
    for shard in shards:
        _require(isinstance(shard, dict), "shard spec malformed")
        rel = shard.get("path")
        _require(isinstance(rel, str) and rel not in names, "duplicate/invalid shard path")
        names.add(rel)
        portable_name = unicodedata.normalize("NFKC", rel).casefold()
        _require(portable_name not in normalized_names, "ambiguous shard path")
        normalized_names.add(portable_name)
        _require(shard.get("split") in SPLITS, "invalid shard split")
        _require(type(shard.get("rows")) is int and shard["rows"] > 0, "invalid shard row count")
        _require(isinstance(shard.get("sha256"), str)
                 and HEX64.fullmatch(shard["sha256"]) is not None, "invalid shard hash")

    counts, licenses, origins = Counter(), Counter(), Counter()
    sampled = []
    root = manifest_path.parent
    with tempfile.TemporaryDirectory(prefix="trainingday-index-", dir=work_root) as temp:
        db = sqlite3.connect(str(Path(temp) / "intake.sqlite3"))
        try:
            db.execute("CREATE TABLE seen (kind TEXT NOT NULL, hash TEXT NOT NULL, "
                       "split TEXT NOT NULL, PRIMARY KEY(kind, hash)) WITHOUT ROWID")
            for shard in sorted(shards, key=lambda x: x["path"]):
                path = _relative_file(root, shard["path"])
                # Validate bytes before parsing; the second pass hashes while
                # reading to catch changes between the checks.
                _require(_hash_file(path) == shard["sha256"], "shard sha mismatch")
                n = 0
                second_digest = hashlib.sha256()
                with path.open("rb") as reader:
                    while raw := reader.readline(MAX_JSONL_ROW_BYTES + 1):
                        _require(len(raw) <= MAX_JSONL_ROW_BYTES, "JSONL row size limit")
                        second_digest.update(raw)
                        _require(raw.strip() != b"", "blank shard row")
                        row = _strict_json(raw)
                        case_id, family, lic, origin = _validate_row(row, shard["split"])
                        prompt_digest = hashlib.sha256(_canonical_prompt(row["prompt"])).hexdigest()
                        id_digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
                        try:
                            db.execute("INSERT INTO seen VALUES ('id', ?, ?)", (id_digest, shard["split"]))
                            db.execute("INSERT INTO seen VALUES ('prompt', ?, ?)", (prompt_digest, shard["split"]))
                        except sqlite3.IntegrityError as exc:
                            raise AdmissionError("duplicate case ID or normalized prompt across shards/splits") from exc
                        n += 1
                        counts[shard["split"]] += 1
                        licenses[lic] += 1
                        origins[origin] += 1
                        if shard["split"] == "train":
                            priority = hashlib.sha256((manifest["corpus_id"] + "\x00" + case_id).encode()).hexdigest()
                            # Keep the smallest K hashes independent of shard order.
                            entry = (-int(priority, 16), case_id, row)
                            if len(sampled) < sample_size:
                                heapq.heappush(sampled, entry)
                            elif entry[0] > sampled[0][0]:
                                heapq.heapreplace(sampled, entry)
                _require(n == shard["rows"], "shard row count mismatch")
                _require(second_digest.hexdigest() == shard["sha256"], "shard mutated during admission")
        finally:
            db.close()
    _require(counts["train"] >= sample_size, "insufficient admitted training rows")
    _require(counts["dev"] > 0 and counts["test"] > 0, "dev/test splits required")
    selected = [item[2] for item in sorted(sampled, key=lambda t: (-t[0], t[1]))]
    summary = {"status": "ADMITTED_SOURCE_ONLY_NOT_TRAINED", "schema": SCHEMA,
               "corpus_id": manifest["corpus_id"], "source_commit": manifest["source_commit"],
               # Bind this receipt to the bytes actually decoded, not a later
               # filesystem reread that might observe a different manifest.
               "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
               "rows_by_split": dict(counts), "licenses": dict(licenses),
               "origins": dict(origins), "selected_count": len(selected),
               "selected_id_sha256": hashlib.sha256(
                   "\n".join(row["id"] for row in selected).encode("utf-8")).hexdigest()}
    return summary, selected
