"""Read-only synthetic turn/result readback for the existing VeraOS Pre-Active store.

This module deliberately CANNOT verify real model consumption. The installed
VeraOS host transport verifier must provide separate authenticated evidence.
Even truthful SQLite COMPLETED is not neural weight identity proof, and a
mutable low-level Store update is not immutable terminal attestation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

_SHA=re.compile(r"^[0-9a-f]{64}$")
_ID=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_BLOB="85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
_MANIFEST="0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
_PROVIDER="dev:virtio-qwen4b"
_MODEL="qwen3:4b-instruct"
_ENDPOINT="http://127.0.0.1:18766/v1"


@dataclass(frozen=True)
class ExpectedTextTurn:
    request_id: str
    task_id: str
    semantic_sha256: str
    task_text: str
    provider_id: str
    model_id: str
    endpoint: str
    host_id: str
    host_generation: int
    host_session: str
    model_blob_sha256: str
    model_manifest_sha256: str

    def __post_init__(self):
        if type(self.request_id) is not str or _ID.fullmatch(self.request_id) is None:
            raise ValueError("invalid stable request ID")
        for value in (self.task_id,self.host_id,self.host_session):
            if type(value) is not str or not value.strip() or len(value)>128:
                raise ValueError("invalid expected text-turn subject")
        if type(self.task_text) is not str or not self.task_text.strip() or len(self.task_text)>2048:
            raise ValueError("task text must be bounded and nonempty")
        if type(self.semantic_sha256) is not str or _SHA.fullmatch(self.semantic_sha256) is None:
            raise ValueError("semantic digest required, but remains declared-only")
        if type(self.host_generation) is not int or self.host_generation<1:
            raise ValueError("invalid serving generation")
        if (self.provider_id!=_PROVIDER or self.model_id!=_MODEL or
            self.endpoint!=_ENDPOINT or self.model_blob_sha256!=_BLOB or
            self.model_manifest_sha256!=_MANIFEST):
            raise ValueError("not the reviewed Qwen development control")


@dataclass(frozen=True)
class ModelTransportEvidence:
    request_id: str
    provider_id: str
    model_id: str
    endpoint: str
    host_id: str
    host_generation: int
    host_session: str
    artifact_sha256: str
    manifest_sha256: str
    final_text_sha256: str
    verified_by_os: bool = False


@dataclass(frozen=True)
class TextTurnReadback:
    status: str
    holds: tuple[str,...]
    request_id: str
    task_id: str
    declared_semantic_sha256: str
    display_text: str | None = None
    attribution: str = "UNVERIFIED_MODEL_SOURCE"
    real_model_consumption_verified: bool = False
    canonical_task_completed: bool = False
    effect_authority: bool = False


def verify_resident_model_turn(
    store_or_connection: Any,
    expected: ExpectedTextTurn,
    claim: object,
    transport: ModelTransportEvidence,
) -> TextTurnReadback:
    """Verify actual durable event/run/message/journal links from the store.

    The *only* positive status is synthetic stored-link verification. A claimed
    transport verification Boolean is NOT a substitute for OS-owned trusted
    receipt provenance and is explicitly rejected. Use on an OS-owned
    read-only snapshot; this function neither executes nor writes to the store.
    """
    if type(expected) is not ExpectedTextTurn or type(transport) is not ModelTransportEvidence:
        raise ValueError("typed expectation and transport record required")
    holds=[]
    if type(claim) is not dict:
        claim={}
        holds.append("MALFORMED_COMPLETION_CLAIM")
    if transport.verified_by_os is not False:
        holds.append("UNTRUSTED_TRANSPORT_PROMOTION")
    transport_fields={
        "request_id":expected.request_id,"provider_id":expected.provider_id,
        "model_id":expected.model_id,"endpoint":expected.endpoint,
        "host_id":expected.host_id,"host_generation":expected.host_generation,
        "host_session":expected.host_session,
        "artifact_sha256":expected.model_blob_sha256,
        "manifest_sha256":expected.model_manifest_sha256,
    }
    if any(getattr(transport,k)!=v for k,v in transport_fields.items()):
        holds.append("TRANSPORT_SCOPE_SUBSTITUTED")
    if (type(claim.get("run_id")) is not str or
        type(claim.get("event_id")) is not str or
        claim.get("status")!="COMPLETED" or
        claim.get("provider_id")!=expected.provider_id or
        claim.get("route_id")!=expected.provider_id or
        type(claim.get("final_text")) is not str or
        not claim.get("final_text") or
        len(claim["final_text"])>4096):
        holds.append("COMPLETION_CLAIM_INVALID")
    conn=getattr(store_or_connection,"connection",store_or_connection)
    if not hasattr(conn,"execute"):
        raise ValueError("actual SQLite store/connection required")
    # A shared connection can expose another owner's uncommitted run result.
    # Never classify such a row as *durably* recorded. Consumer should use a
    # separate read-only connection or an independently committed snapshot.
    if conn.in_transaction:
        return TextTurnReadback(
            status="HOLD",
            holds=("UNCOMMITTED_STORE_READBACK",),
            request_id=expected.request_id,task_id=expected.task_id,
            declared_semantic_sha256=expected.semantic_sha256)
    try:
        # Existing PreActive Store holds task.requested event, source-linked
        # run, assistant message and route journal. Read within one snapshot.
        if not conn.in_transaction:
            conn.execute("BEGIN")
            owns=True
        else:
            owns=False
        try:
            event=conn.execute("SELECT id,kind,payload_json,dedup_key,status "
                "FROM events WHERE id=?",(claim.get("event_id"),)).fetchone()
            run=conn.execute("SELECT id,task,status,final_text,source_event_id,"
                "capabilities_json,last_error FROM runs WHERE id=?",
                (claim.get("run_id"),)).fetchone()
            journal=conn.execute("SELECT event_type,subject_id,payload_json "
                "FROM journal WHERE subject_id=? AND "
                "event_type='VERAOS_COGNITION_ROUTE_SELECTED'",
                (claim.get("event_id"),)).fetchall()
            msg=conn.execute("SELECT role,content FROM run_messages "
                "WHERE run_id=? ORDER BY id DESC LIMIT 1",
                (claim.get("run_id"),)).fetchone()
        finally:
            if owns: conn.rollback()  # read-only snapshot, never write
    except (Exception,) as exc:
        # This adapter is explicitly fail-closed on inaccessible persistence.
        holds.append("PERSISTED_STORE_READBACK_UNAVAILABLE")
        event=run=msg=None
        journal=[]
    if event is None or run is None:
        holds.append("EVENT_OR_RUN_MISSING")
    else:
        try:
            payload=json.loads(event["payload_json"])
            caps=json.loads(run["capabilities_json"])
        except (TypeError,ValueError):
            payload=caps=None
        if (event["kind"]!="task.requested" or
            event["dedup_key"]!=f"veraos-text:{expected.task_id}" or
            event["status"]!="DONE" or
            payload!={"task":expected.task_text,"capabilities":[]} or
            run["source_event_id"]!=event["id"] or
            run["task"]!=expected.task_text or
            run["status"]!="COMPLETED" or caps!=[] or
            run["last_error"] is not None):
            holds.append("PERSISTED_TASK_CONTINUITY_MISMATCH")
        text=run["final_text"]
        if (type(text) is not str or not text.strip() or len(text)>4096 or
            text!=claim.get("final_text") or
            hashlib.sha256(text.encode("utf8")).hexdigest()!=transport.final_text_sha256):
            holds.append("PERSISTED_RESPONSE_MISMATCH")
        if (msg is None or msg["role"]!="assistant" or
            msg["content"]!=text):
            holds.append("ASSISTANT_MESSAGE_MISMATCH")
        accepted_journal=False
        for item in journal:
            try:
                j=json.loads(item["payload_json"])
            except (TypeError,ValueError):
                continue
            if (j.get("provider_id")==expected.provider_id and
                j.get("route_id")==expected.provider_id and
                j.get("model")==expected.model_id and
                j.get("effect_class")=="COGNITION_ONLY"):
                accepted_journal=True
        if not accepted_journal or len(journal)!=1:
            holds.append("ROUTE_SELECTION_JOURNAL_MISMATCH")
    return TextTurnReadback(
        status="SYNTHETIC_STORE_LINK_VERIFIED" if not holds else "HOLD",
        holds=tuple(dict.fromkeys(holds)),
        request_id=expected.request_id,
        task_id=expected.task_id,
        declared_semantic_sha256=expected.semantic_sha256,
        display_text=claim.get("final_text") if not holds else None,
    )
