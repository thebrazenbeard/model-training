"""Bounded, read-only local overlap audit for TRAININGDAY JSONL datasets.

Only the bytes actually opened from two explicitly selected local files are
inspected. This module does not establish dataset rights, independent heldout
custody, absence of semantic paraphrases, or permission to train. Every result
retains a physical-launch and durable-state HARD HOLD.

Accepted JSONL records (one object per nonblank line):

    {"id": "example-1", "role": "user", "text": "How are you?"}
    {"id": "turn-1", "messages": [
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Well, thanks."}
    ]}

The closed schema intentionally avoids arbitrary metadata/answers, but cannot
detect malicious data covertly encoded inside otherwise allowed strings.
No source text or example identifiers are included in result objects or errors.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any
import unicodedata


MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_LINE_BYTES = 32 * 1024
MAX_RECORDS = 5000
MAX_MESSAGES = 16
MAX_TEXT_CHARS_PER_RECORD = 8192
MAX_TOTAL_TEXT_CHARS = 2_000_000
MAX_SHINGLE_POSTINGS = 500_000
MAX_CANDIDATE_PAIRS = 100_000
NEAR_JACCARD_THRESHOLD = 0.82
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z")
_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})


def _secure_descriptor_walk_supported() -> bool:
    return (os.open in getattr(os, "supports_dir_fd", set())
            and hasattr(os, "O_NOFOLLOW")
            and hasattr(os, "O_DIRECTORY"))


class DatasetAuditHold(ValueError):
    """A bounded inspection cannot safely establish the requested evidence."""


@dataclass(frozen=True, slots=True)
class DatasetAuditReport:
    train_file_sha256: str
    heldout_file_sha256: str
    train_record_count: int
    heldout_record_count: int
    matching_file_digest: bool
    exact_role_text_pair_count: int
    exact_text_any_role_pair_count: int
    shared_id_count: int
    shared_user_prompt_count: int
    shared_dialogue_prefix_count: int
    near_text_pair_count: int
    train_duplicate_text_count: int
    heldout_duplicate_text_count: int
    train_duplicate_id_count: int
    heldout_duplicate_id_count: int
    warnings: tuple[str, ...]

    @property
    def disposition(self) -> str:
        if any((
            self.matching_file_digest,
            self.exact_text_any_role_pair_count,
            self.shared_id_count,
            self.shared_user_prompt_count,
            self.shared_dialogue_prefix_count,
            self.near_text_pair_count,
            self.train_duplicate_id_count,
            self.heldout_duplicate_id_count,
        )):
            return "CONTAMINATION_CANDIDATE_HOLD"
        return "NO_MATCH_OBSERVED_UNVERIFIED_HOLD"

    @property
    def physical_launch_authorized(self) -> bool:
        return False

    @property
    def persistent_state_commit_authorized(self) -> bool:
        return False

    @property
    def independent_heldout_custody_verified(self) -> bool:
        return False

    @property
    def rights_verified(self) -> bool:
        return False

    @property
    def semantic_disjointness_verified(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable counts and digests only; no raw dataset examples."""
        return {
            "evidence_class": "LOCAL_CONTENT_INSPECTION",
            "train_file_sha256": self.train_file_sha256,
            "heldout_file_sha256": self.heldout_file_sha256,
            "train_record_count": self.train_record_count,
            "heldout_record_count": self.heldout_record_count,
            "matching_file_digest": self.matching_file_digest,
            "exact_role_text_pair_count": self.exact_role_text_pair_count,
            "exact_text_any_role_pair_count": self.exact_text_any_role_pair_count,
            "shared_id_count": self.shared_id_count,
            "shared_user_prompt_count": self.shared_user_prompt_count,
            "shared_dialogue_prefix_count": self.shared_dialogue_prefix_count,
            "near_text_pair_count": self.near_text_pair_count,
            "train_duplicate_text_count": self.train_duplicate_text_count,
            "heldout_duplicate_text_count": self.heldout_duplicate_text_count,
            "train_duplicate_id_count": self.train_duplicate_id_count,
            "heldout_duplicate_id_count": self.heldout_duplicate_id_count,
            "disposition": self.disposition,
            "warnings": list(self.warnings),
            "physical_launch_authorized": False,
            "persistent_state_commit_authorized": False,
            "independent_heldout_custody_verified": False,
            "rights_verified": False,
            "semantic_disjointness_verified": False,
        }


@dataclass(frozen=True, slots=True)
class _Record:
    record_id: str
    role_text: str
    text_only: str
    user_prompts: frozenset[str]
    prefix: str | None


@dataclass(frozen=True, slots=True)
class _Split:
    sha256: str
    records: tuple[_Record, ...]


def _no_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetAuditHold("duplicate JSON object key")
        result[key] = value
    return result


def _no_json_constants(_value: str) -> None:
    raise DatasetAuditHold("non-finite JSON constant")


def _normalize_text(raw: str) -> str:
    """Lexical NFKC/casefold normalization, never a semantic equivalence test."""
    transformed = unicodedata.normalize("NFKC", raw).casefold()
    out = []
    for char in transformed:
        category = unicodedata.category(char)
        if category == "Cf":
            # Format code points such as zero-width separators.
            continue
        if char.isspace() or category.startswith(("P", "S")):
            out.append(" ")
        else:
            out.append(char)
    return " ".join("".join(out).split())


def _parse_record(raw_line: bytes, line_number: int) -> tuple[_Record, int]:
    try:
        decoded = raw_line.decode("utf-8", errors="strict")
        row = json.loads(
            decoded, object_pairs_hook=_no_duplicate_json_keys,
            parse_constant=_no_json_constants,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise DatasetAuditHold(f"line {line_number}: invalid UTF-8/JSON") from None
    if type(row) is not dict or "id" not in row:
        raise DatasetAuditHold(f"line {line_number}: invalid JSONL record shape")
    rid = row["id"]
    if type(rid) is not str or not _IDENTIFIER.fullmatch(rid):
        raise DatasetAuditHold(f"line {line_number}: invalid record identifier")

    messages: list[tuple[str, str]] = []
    if set(row) == {"id", "role", "text"}:
        messages = [(row["role"], row["text"])]
    elif set(row) == {"id", "messages"}:
        declared = row["messages"]
        if type(declared) is not list or not 1 <= len(declared) <= MAX_MESSAGES:
            raise DatasetAuditHold(f"line {line_number}: invalid message count")
        for message in declared:
            if type(message) is not dict or set(message) != {"role", "content"}:
                raise DatasetAuditHold(f"line {line_number}: invalid message schema")
            messages.append((message["role"], message["content"]))
    else:
        raise DatasetAuditHold(f"line {line_number}: unknown or missing fields")

    canonical: list[tuple[str, str]] = []
    chars = 0
    for role, text in messages:
        if type(role) is not str or role not in _ROLES or type(text) is not str:
            raise DatasetAuditHold(f"line {line_number}: invalid role/text type")
        chars += len(text)
        if chars > MAX_TEXT_CHARS_PER_RECORD:
            raise DatasetAuditHold(f"line {line_number}: text budget exceeded")
        normalized = _normalize_text(text)
        if not normalized:
            raise DatasetAuditHold(f"line {line_number}: empty normalized content")
        canonical.append((role, normalized))
    role_text = "\x1e".join(role + "\x00" + value for role, value in canonical)
    text_only = "\x1e".join(value for _, value in canonical)
    prompts = frozenset(value for role, value in canonical if role == "user")
    prefix = (
        "\x1e".join(role + "\x00" + value for role, value in canonical[:-1])
        if len(canonical) > 1 else None
    )
    return _Record(rid, role_text, text_only, prompts, prefix), chars


def _validate_path(allowed_root: str | os.PathLike[str],
                   chosen: str | os.PathLike[str]) -> Path:
    """Lexically contain path and reject observed symbolic links.

    A true adversarial filesystem race still needs host-owned directory-fd
    custody: this source-only reader is not a secure remote execution broker.
    """
    try:
        supplied_root = Path(os.fspath(allowed_root))
        supplied_path = Path(os.fspath(chosen))
        if not supplied_root.is_absolute() or not supplied_path.is_absolute():
            raise ValueError("paths must be absolute")
        root = Path(os.path.abspath(supplied_root))
        path = Path(os.path.abspath(supplied_path))
    except (TypeError, ValueError, OSError):
        raise DatasetAuditHold("invalid or non-absolute dataset path") from None
    if len(root.parts) < (2 if os.name == "nt" else 3):
        raise DatasetAuditHold("allowed dataset root is too broad")
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise DatasetAuditHold("dataset path escapes allowed root") from None
    if not relative.parts:
        raise DatasetAuditHold("dataset path is directory root")
    # Check every absolute ancestor, not merely descendants of allowed_root.
    # On POSIX the descriptor walk below also makes this check race resistant.
    for component in (*reversed(path.parents), path):
        try:
            mode = component.lstat().st_mode
        except (OSError, ValueError):
            raise DatasetAuditHold("dataset path unavailable") from None
        if stat.S_ISLNK(mode):
            raise DatasetAuditHold("symbolic-link dataset path denied")
    return path


def _open_file_nofollow(path: Path) -> int:
    """Use a no-follow descriptor walk where openat-style APIs are available."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if not _secure_descriptor_walk_supported():
        # Fallback is only a bounded, owner-local read; it is not secure
        # against malicious concurrent directory replacement on Windows.
        return os.open(path, flags)
    parts = path.parts
    if not parts or not path.is_absolute():
        raise DatasetAuditHold("dataset must have absolute canonical path")
    dir_flags = flags | os.O_DIRECTORY
    descriptor = os.open(parts[0], dir_flags)
    try:
        for part in parts[1:-1]:
            nxt = os.open(part, dir_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = nxt
        return os.open(parts[-1], flags, dir_fd=descriptor)
    finally:
        os.close(descriptor)


def _read_split(path: Path) -> _Split:
    try:
        fd = _open_file_nofollow(path)
    except (OSError, ValueError):
        raise DatasetAuditHold("dataset file unavailable") from None
    records: list[_Record] = []
    hasher = hashlib.sha256()
    total_bytes = 0
    total_chars = 0
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise DatasetAuditHold("dataset must be regular file")
            if before.st_size < 1 or before.st_size > MAX_FILE_BYTES:
                raise DatasetAuditHold("dataset size limit exceeded")
            while True:
                line = handle.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                total_bytes += len(line)
                if total_bytes > MAX_FILE_BYTES or len(line) > MAX_LINE_BYTES:
                    raise DatasetAuditHold("dataset line/file byte budget exceeded")
                if len(records) >= MAX_RECORDS:
                    raise DatasetAuditHold("dataset record budget exceeded")
                if not line.strip():
                    raise DatasetAuditHold(f"line {len(records) + 1}: blank line")
                hasher.update(line)
                record, chars = _parse_record(line, len(records) + 1)
                total_chars += chars
                if total_chars > MAX_TOTAL_TEXT_CHARS:
                    raise DatasetAuditHold("dataset text budget exceeded")
                records.append(record)
            after = os.fstat(handle.fileno())
            if (
                total_bytes != before.st_size
                or (before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise DatasetAuditHold("dataset changed during read")
    except DatasetAuditHold:
        raise
    except (OSError, ValueError):
        raise DatasetAuditHold("dataset read failed") from None
    # The pathname might have been switched after opening; refuse a changed
    # object even if its content happens to be parseable.
    try:
        current = path.lstat()
    except (OSError, ValueError):
        raise DatasetAuditHold("dataset path disappeared") from None
    if (
        stat.S_ISLNK(current.st_mode)
        or (current.st_dev, current.st_ino, current.st_size,
            current.st_mtime_ns, current.st_ctime_ns)
        != (after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise DatasetAuditHold("dataset path changed during read")
    if not records:
        raise DatasetAuditHold("empty dataset")
    return _Split(hasher.hexdigest(), tuple(records))


def _shingles(value: str) -> frozenset[str]:
    # Delimiters become spaces. Short texts are exact-match-only to avoid
    # unreliable near-similarity on high-frequency fragments.
    compact = " ".join(value.replace("\x1e", " ").split())
    if len(compact) < 24:
        return frozenset()
    return frozenset(compact[index:index + 5]
                     for index in range(len(compact) - 4))


def _near_match_count(train: tuple[_Record, ...],
                      heldout: tuple[_Record, ...]) -> int:
    postings: dict[str, list[int]] = defaultdict(list)
    train_grams = []
    posting_count = 0
    for idx, row in enumerate(train):
        grams = _shingles(row.text_only)
        train_grams.append(grams)
        posting_count += len(grams)
        if posting_count > MAX_SHINGLE_POSTINGS:
            raise DatasetAuditHold("near-match index budget exceeded")
        for gram in grams:
            postings[gram].append(idx)

    compared = 0
    near_count = 0
    for evaluation in heldout:
        grams = _shingles(evaluation.text_only)
        if not grams:
            continue
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(postings.get(gram, ()))
            if len(candidates) + compared > MAX_CANDIDATE_PAIRS:
                raise DatasetAuditHold("near-match comparison budget exceeded")
        for idx in candidates:
            # Exact matches are reported independently, without double count.
            if evaluation.text_only == train[idx].text_only:
                continue
            compared += 1
            if compared > MAX_CANDIDATE_PAIRS:
                raise DatasetAuditHold("near-match comparison budget exceeded")
            donor = train_grams[idx]
            intersection = len(grams & donor)
            union = len(grams) + len(donor) - intersection
            if union and intersection / union >= NEAR_JACCARD_THRESHOLD:
                near_count += 1
    return near_count


def audit_jsonl_overlap(
    *, allowed_root: str | os.PathLike[str],
    train_path: str | os.PathLike[str],
    heldout_path: str | os.PathLike[str],
) -> DatasetAuditReport:
    """Read both local files and report bounded lexical contamination signals.

    Every return is a HOLD. If no matches are observed, semantic disjointness
    remains unknown. This cannot read private evaluator stores, grant rights,
    authenticate an independent custodian, or authorize GPU work.
    """
    train_file = _validate_path(allowed_root, train_path)
    heldout_file = _validate_path(allowed_root, heldout_path)
    if train_file == heldout_file:
        raise DatasetAuditHold("train and heldout path are identical")
    train = _read_split(train_file)
    heldout = _read_split(heldout_file)
    t_role = Counter(r.role_text for r in train.records)
    h_role = Counter(r.role_text for r in heldout.records)
    t_text = Counter(r.text_only for r in train.records)
    h_text = Counter(r.text_only for r in heldout.records)
    t_id_counts = Counter(r.record_id for r in train.records)
    h_id_counts = Counter(r.record_id for r in heldout.records)
    t_ids = set(t_id_counts)
    h_ids = set(h_id_counts)
    t_prompts = {prompt for r in train.records for prompt in r.user_prompts}
    h_prompts = {prompt for r in heldout.records for prompt in r.user_prompts}
    t_prefixes = {r.prefix for r in train.records if r.prefix is not None}
    h_prefixes = {r.prefix for r in heldout.records if r.prefix is not None}

    def cross_pairs(a: Counter[str], b: Counter[str]) -> int:
        return sum(count * b[key] for key, count in a.items())

    return DatasetAuditReport(
        train_file_sha256=train.sha256,
        heldout_file_sha256=heldout.sha256,
        train_record_count=len(train.records),
        heldout_record_count=len(heldout.records),
        matching_file_digest=train.sha256 == heldout.sha256,
        exact_role_text_pair_count=cross_pairs(t_role, h_role),
        exact_text_any_role_pair_count=cross_pairs(t_text, h_text),
        shared_id_count=len(t_ids & h_ids),
        shared_user_prompt_count=len(t_prompts & h_prompts),
        shared_dialogue_prefix_count=len(t_prefixes & h_prefixes),
        near_text_pair_count=_near_match_count(train.records, heldout.records),
        train_duplicate_text_count=sum(count - 1 for count in t_text.values()
                                       if count > 1),
        heldout_duplicate_text_count=sum(count - 1 for count in h_text.values()
                                         if count > 1),
        train_duplicate_id_count=sum(count - 1 for count in t_id_counts.values()
                                     if count > 1),
        heldout_duplicate_id_count=sum(count - 1 for count in h_id_counts.values()
                                       if count > 1),
        warnings=(
            "EXACT_MATCHES_ARE_NORMALIZED_CONTENT_EQUALITY_ONLY",
            "NEAR_MATCHES_ARE_HEURISTIC_NOT_SEMANTIC_PROOF",
            "SEMANTIC_PARAPHRASES_AND_CONFUSABLE_UNICODE_MAY_ESCAPE",
            "SPLIT_READS_NOT_ATOMIC_OR_INDEPENDENT_CUSTODY",
            "RIGHTS_AND_PHYSICAL_LAUNCH_HARD_HOLD",
        ) + (() if _secure_descriptor_walk_supported() else (
            "PLATFORM_PATH_CONTAINMENT_NOT_ATOMIC_NO_DIRFD_WALK",
        )),
    )


def main(argv: list[str] | None = None) -> int:
    """Operator-facing read-only diagnostic; never returns a launch PASS."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", required=True)
    args = parser.parse_args(argv)
    try:
        report = audit_jsonl_overlap(
            allowed_root=args.allowed_root,
            train_path=args.train,
            heldout_path=args.heldout,
        )
    except DatasetAuditHold as exc:
        print(json.dumps({
            "disposition": "INSPECTION_ERROR_HOLD",
            "evidence_class": "LOCAL_CONTENT_INSPECTION",
            "error": str(exc),
            "physical_launch_authorized": False,
            "rights_verified": False,
        }, sort_keys=True))
        return 2
    print(json.dumps(report.as_dict(), sort_keys=True))
    # Even no matches are an unverified HOLD, never an execution gate success.
    return 1


__all__ = ["DatasetAuditHold", "DatasetAuditReport", "audit_jsonl_overlap", "main"]


if __name__ == "__main__":
    sys.exit(main())
