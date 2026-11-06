"""Owner-local declared v-e-r-a browser transcript importer; no model attestation.

Maps a bounded `messages = [{role,content}, ...]` snapshot into the published
multi-turn format entirely in memory. It neither accesses the browser/vault nor
prints or writes conversation text. A displayed reply is not a quality PASS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from .multiturn_protocol import DialogueSessionError, screen_sessions, validate_sessions


IMPORT_SCHEMA = "VERA_DECLARED_LOCAL_CHAT_EXPORT_V1"
_EXPORT_FIELDS = frozenset({
    "schema", "arm", "case_id", "session_id", "source_claim",
    "model_selection_claim", "base_sha256", "adapter_sha256", "prompt_sha256",
    "generation", "used_saved_recall", "restored_from_vault", "messages",
    "terminal_failure",
})
_MESSAGE_FIELDS = frozenset({"role", "content"})
_FAILURE_FIELDS = frozenset({"user", "outcome"})
_TECHNICAL = frozenset({"truncated", "timeout", "upstream_error"})


class LocalChatImportError(ValueError):
    """Malformed, inaccessible or misleading owner-local transcript input."""


def _text(value: object) -> bool:
    return type(value) is str and 1 <= len(value) <= 8000


def adapt_export_records(rows: Sequence[Mapping[str, object]]) -> tuple[dict, ...]:
    """Map declared browser display history into existing source-only protocol.

    The producer is unverified; the caller must provide independent, explicit
    generation/provenance metadata rather than inventing a model hash or token
    budget. Saved-vault recall/restores are refused because their hidden context
    cannot be compared as a visible single-session conversation.
    """
    if type(rows) not in (list, tuple) or not 1 <= len(rows) <= 250:
        raise LocalChatImportError("bounded nonempty export rows required")
    normalized = []
    for row in rows:
        if type(row) is not dict or set(row) != _EXPORT_FIELDS:
            raise LocalChatImportError("local export schema mismatch")
        if row["schema"] != IMPORT_SCHEMA:
            raise LocalChatImportError("unsupported local export schema")
        if type(row["used_saved_recall"]) is not bool or type(row["restored_from_vault"]) is not bool:
            raise LocalChatImportError("explicit boolean context flags required")
        if row["used_saved_recall"] or row["restored_from_vault"]:
            raise LocalChatImportError("hidden saved context is outside visible-session comparison")
        if not _text(row["model_selection_claim"]) or len(row["model_selection_claim"]) > 200:
            raise LocalChatImportError("invalid unverified selected-model claim")
        messages = row["messages"]
        if type(messages) is not list or not 2 <= len(messages) <= 12 or len(messages) % 2:
            raise LocalChatImportError("browser message window needs complete pairs of at most twelve messages")
        pairs = []
        for position in range(0, len(messages), 2):
            user, assistant = messages[position:position + 2]
            if (type(user) is not dict or set(user) != _MESSAGE_FIELDS
                    or type(assistant) is not dict or set(assistant) != _MESSAGE_FIELDS
                    or user["role"] != "user" or assistant["role"] != "assistant"):
                raise LocalChatImportError("only alternating browser user/assistant messages are admissible")
            if not _text(user["content"]) or not _text(assistant["content"]):
                raise LocalChatImportError("browser message text malformed or too large")
            # `complete` means displayed as an assistant message in the UI
            # snapshot; it does not attest a real model finish_reason.
            pairs.append({"user": user["content"], "assistant": assistant["content"],
                          "outcome": "complete"})
        final = row["terminal_failure"]
        if final is not None:
            if (type(final) is not dict or set(final) != _FAILURE_FIELDS
                    or not _text(final["user"])
                    or type(final["outcome"]) is not str
                    or final["outcome"] not in _TECHNICAL):
                raise LocalChatImportError("terminal failure must be an explicit user-only technical event")
            pairs.append({"user": final["user"], "assistant": None,
                          "outcome": final["outcome"]})
        if len(pairs) < 2:
            raise LocalChatImportError("at least two declared attempts required")
        normalized.append({
            "arm": row["arm"],
            "case_id": row["case_id"],
            "session_id": row["session_id"],
            "source_claim": row["source_claim"],
            "origin_claim": row["model_selection_claim"],
            "base_sha256": row["base_sha256"],
            "adapter_sha256": row["adapter_sha256"],
            "prompt_sha256": row["prompt_sha256"],
            "generation": row["generation"],
            "turns": pairs,
        })
    try:
        return validate_sessions(normalized)
    except (DialogueSessionError, TypeError) as exc:
        # Caller can inspect the error class; don't ever print raw owner text.
        raise LocalChatImportError("declared local export does not satisfy multi-turn contract") from exc


def screen_local_exports(rows: Sequence[Mapping[str, object]]) -> dict:
    """Aggregate bounded declared import evidence, with no behavioral score."""
    report = screen_sessions(adapt_export_records(rows))
    report["import_schema"] = IMPORT_SCHEMA
    report["local_export_import_format_validated"] = True
    report["browser_snapshot_authenticated"] = False
    report["assistant_finish_reasons_attested"] = False
    report["saved_context_absence_attested"] = False
    return report


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise LocalChatImportError("duplicate JSON field")
        obj[key] = value
    return obj


def main(argv: Sequence[str] | None = None) -> int:
    """Read owner-local JSONL, print only aggregate output, never raw text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Private owner-prepared export JSONL; never commit")
    args = parser.parse_args(argv)
    try:
        if not args.input.is_file() or not 0 < args.input.stat().st_size <= 2_000_000:
            raise LocalChatImportError("bounded local file required")
        with args.input.open("r", encoding="utf-8") as stream:
            records = [json.loads(line, object_pairs_hook=_strict_json_object) for line in stream]
        result = screen_local_exports(records)
    except (ValueError, OSError, UnicodeError, TypeError) as exc:
        print(f"local chat import invalid input: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
