from __future__ import annotations

import json

from successor.experiments.retention_audit_arch_v2_reviewer import (
    build_reviewer_qualification_controls as build_v2_controls,
)
from successor.experiments.retention_audit_arch_v3_reviewer import (
    build_reviewer_qualification_controls_v3,
)
from successor.experiments.retention_audit_arch_v4_reviewer import (
    build_reviewer_qualification_controls_v4,
)
from successor.experiments.retention_audit_arch_v5_reviewer import (
    MODEL,
    MODEL_BLOB_SHA256,
    SEED,
    TEMPERATURE,
    build_reviewer_qualification_controls_v5,
    qualification_response_schema_v5,
    reviewer_identity_v5,
    run_batched_reviewer_qualification_v5,
    semantic_response_schema_v5,
    structured_request_body_v5,
)


def test_v5_controls_are_fresh_balanced_and_disjoint() -> None:
    controls = build_reviewer_qualification_controls_v5()
    prior_ids = {
        row["case_id"] for row in build_v2_controls()
    } | {
        row["case_id"] for row in build_reviewer_qualification_controls_v3()
    } | {
        row["case_id"] for row in build_reviewer_qualification_controls_v4()
    }

    assert len(controls) == 48
    assert not ({row["case_id"] for row in controls} & prior_ids)
    assert all(row["case_id"].startswith("arch-v5-reviewer-") for row in controls)
    classes = {row["defect_class"] for row in controls}
    assert len(classes) == 6
    for defect_class in classes:
        rows = [row for row in controls if row["defect_class"] == defect_class]
        assert sum(row["expected_defect"] for row in rows) == 4
        assert sum(not row["expected_defect"] for row in rows) == 4


def test_v5_qualification_schema_enforces_shape_not_semantics() -> None:
    controls = build_reviewer_qualification_controls_v5()[:4]
    schema = qualification_response_schema_v5(controls)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    reviews = schema["properties"]["reviews"]
    assert reviews["minItems"] == 4
    assert reviews["maxItems"] == 4
    item = reviews["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {
        "case_id",
        "observed_defect",
        "defect_class",
        "witness",
        "reason",
    }
    assert set(item["properties"]["case_id"]["enum"]) == {
        row["case_id"] for row in controls
    }
    assert item["properties"]["observed_defect"] == {"type": "boolean"}
    assert set(item["properties"]["witness"]["type"]) == {
        "string", "array", "object", "null", "number", "boolean"
    }
    # Hidden expected labels must not appear anywhere in the schema.
    assert "expected_defect" not in json.dumps(schema, sort_keys=True)


def test_v5_semantic_schema_constrains_closed_fields_and_enums() -> None:
    case_ids = ["case-a", "case-b"]
    schema = semantic_response_schema_v5(case_ids)
    item = schema["properties"]["reviews"]["items"]

    assert item["additionalProperties"] is False
    assert set(item["required"]) == {
        "case_id",
        "observation",
        "derived_answer",
        "witness",
        "reason",
        "confidence",
    }
    assert set(item["properties"]["case_id"]["enum"]) == set(case_ids)
    assert "AMBIGUITY_WITNESS" in item["properties"]["observation"]["enum"]
    assert set(item["properties"]["confidence"]["enum"]) == {
        "low", "medium", "high"
    }


def test_v5_request_body_uses_json_schema_format() -> None:
    schema = {
        "type": "object",
        "properties": {"flag": {"type": "boolean"}},
        "required": ["flag"],
        "additionalProperties": False,
    }

    body = structured_request_body_v5("prompt", schema)

    assert body["model"] == MODEL == "ministral-3:14b"
    assert MODEL_BLOB_SHA256 == (
        "bfb40fc6bb9c3b2ed529b480e04f824c005ea8f86733d4ebbf0c204de484891e"
    )
    assert body["format"] == schema
    assert body["stream"] is False
    assert body["options"] == {
        "temperature": TEMPERATURE,
        "seed": SEED,
    }


def test_v5_identity_declares_json_schema_mode(monkeypatch) -> None:
    import successor.experiments.retention_audit_arch_v5_reviewer as reviewer

    monkeypatch.setattr(
        reviewer,
        "base_reviewer_identity",
        lambda: {
            "provider": "OLLAMA_LOCAL",
            "model": MODEL,
            "model_blob_sha256": MODEL_BLOB_SHA256,
            "runtime": "ollama version is 0.34.2",
            "temperature": 0,
            "seed": 20261002,
            "url": "http://127.0.0.1:11434/api/generate",
            "format": "json",
            "stream": False,
        },
    )

    identity = reviewer_identity_v5()

    assert identity["structured_output"] == "JSON_SCHEMA"
    assert identity["format"] == "DYNAMIC_JSON_SCHEMA"


def test_v5_batched_qualification_scores_structured_outputs_without_retries() -> None:
    controls = build_reviewer_qualification_controls_v5()
    calls = []

    def call_structured(prompt: str, schema: dict) -> str:
        start = len(calls) * 4
        batch = controls[start:start + 4]
        calls.append((prompt, schema))
        return json.dumps({
            "reviews": [
                {
                    "case_id": control["case_id"],
                    "observed_defect": control["expected_defect"],
                    "defect_class": control["defect_class"],
                    "witness": (
                        {"example": "concrete semantic witness"}
                        if control["expected_defect"]
                        else "NONE"
                    ),
                    "reason": "qualified",
                }
                for control in batch
            ]
        })

    result = run_batched_reviewer_qualification_v5(
        call_structured=call_structured,
        batch_size=4,
        max_attempts=3,
    )

    assert result["status"] == "REVIEWER_QUALIFIED"
    assert result["batch_count"] == 12
    assert len(calls) == 12
    assert all(batch["attempts"] == 1 for batch in result["batches"])
    assert all(
        call[1]["additionalProperties"] is False
        for call in calls
    )


def test_v5_transport_failure_preserves_safe_synthetic_raw_response() -> None:
    calls = []

    def call_structured(prompt: str, schema: dict) -> str:
        calls.append((prompt, schema))
        return '{"reviews": []}'

    result = run_batched_reviewer_qualification_v5(
        call_structured=call_structured,
        batch_size=4,
        max_attempts=3,
    )

    assert result["status"] == "REVIEWER_UNQUALIFIED"
    failures = result["transport_failures"]
    assert len(failures) == 3
    assert all(item["raw_response"] == '{"reviews": []}' for item in failures)
    assert all(len(item["response_sha256"]) == 64 for item in failures)
