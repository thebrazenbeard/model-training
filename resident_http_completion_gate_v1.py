"""Portable PR37 source-HTTP boundary guard; no provider or runtime grant.

Installed VeraOS must independently read back PreActive durable event/run
and authenticated model response before displaying Qwen-generated output.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_MAX_OUTPUT=4096
_STATUSES={"PENDING","COMPLETED","FAILED","BLOCKED","CANCELLED"}


class HttpResultHold(ValueError):
    """Malformed source status or attempted trust promotion."""


def _token(value: object) -> bool:
    return type(value) is str and _TOKEN.fullmatch(value) is not None


def normalize_http_result(
    source: object,
    *, request_id: str,
    expected_provider: str,
    fixture_mode: object = False,
) -> dict:
    """Normalize coherent test-only PreActive envelope or fail closed.

    No real-host mode is implemented here. A source COMPLETED without
    durable linkage cannot be treated as HTTP completion. Even a coherent
    fixture remains a synthetic result and never grants display authority.
    """
    if fixture_mode is not True or not _token(request_id) or not _token(expected_provider):
        raise HttpResultHold("SOURCE_TEST_ONLY_NO_OS_AUTHORITY")
    if type(source) is not dict:
        raise HttpResultHold("MALFORMED_PREACTIVE_SOURCE")
    allowed={"status","run_id","event_id","final_text","route_id","provider_id"}
    if set(source)!=allowed or type(source.get("status")) is not str or source["status"] not in _STATUSES:
        raise HttpResultHold("MALFORMED_SOURCE_RESULT_OR_AUTHORITY_CLAIM")
    if source["route_id"]!=expected_provider or source["provider_id"]!=expected_provider:
        raise HttpResultHold("ROUTE_PROVIDER_SUBSTITUTED")
    if not _token(source["event_id"]):
        raise HttpResultHold("EVENT_SOURCE_MISSING")
    status=source["status"]
    if status=="COMPLETED":
        final=source["final_text"]
        if (not _token(source["run_id"]) or
            type(final) is not str or not final.strip() or
            len(final)>_MAX_OUTPUT):
            raise HttpResultHold("TERMINAL_RESULT_WITHOUT_DURABLE_CONTENT")
    else:
        if source["run_id"] is not None and not _token(source["run_id"]):
            raise HttpResultHold("MALFORMED_RUN_ID")
        if source["final_text"] is not None:
            raise HttpResultHold("NON_COMPLETED_RESPONSE_LEAK")
    return {
        "status":status,
        "request_id":request_id,
        "run_id":source["run_id"],
        "event_id":source["event_id"],
        "final_text":source["final_text"] if status=="COMPLETED" else None,
        "model_authenticated":False,
        "display_authorized":False,
        "canonical_task_completed":False,
        "effect_authority":False,
    }
