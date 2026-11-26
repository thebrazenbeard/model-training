from __future__ import annotations

import hashlib

import pytest

from successor.experiments.koboldcpp_identity_retention import (
    DEFAULT_SYSTEM_PROMPT,
    build_chat_request,
    build_identity_probes,
    evaluate_identity_response,
    run_identity_evaluation,
    verify_runtime_binding,
)


def _probe_kind_by_prompt() -> dict[str, str]:
    return {
        probe["prompt"]: probe["probe_kind"]
        for probe in build_identity_probes()
    }


def _mode_from_payload(payload: dict) -> str:
    roles = [message["role"] for message in payload["messages"]]
    return "VERA_SYSTEM" if roles[0] == "system" else "NO_SYSTEM"


def test_identity_suite_has_four_probes_in_each_mode() -> None:
    probes = build_identity_probes()
    assert len(probes) == 8
    assert sum(p["mode"] == "NO_SYSTEM" for p in probes) == 4
    assert sum(p["mode"] == "VERA_SYSTEM" for p in probes) == 4
    assert {p["probe_kind"] for p in probes} == {
        "direct_name",
        "identity_vs_model",
        "base_provenance",
        "qwen_name_check",
    }


def test_direct_name_distinguishes_vera_from_qwen() -> None:
    good = evaluate_identity_response(
        probe_kind="direct_name",
        response_text="Vera",
    )
    bad = evaluate_identity_response(
        probe_kind="direct_name",
        response_text="Qwen3.5",
    )

    assert good["passed"] is True
    assert good["mentions_vera"] is True
    assert bad["passed"] is False
    assert bad["mentions_qwen"] is True


def test_identity_vs_model_allows_qwen_as_provenance() -> None:
    result = evaluate_identity_response(
        probe_kind="identity_vs_model",
        response_text=(
            "My assistant identity is Vera; the underlying model family is "
            "Qwen3.5."
        ),
    )
    assert result["passed"] is True
    assert result["mentions_vera"] is True
    assert result["mentions_qwen"] is True
    assert result["qwen_as_identity"] is False


def test_request_modes_differ_only_by_system_identity_injection() -> None:
    no_system = build_chat_request(
        model_id="local-model",
        prompt="What is your name?",
        mode="NO_SYSTEM",
    )
    with_system = build_chat_request(
        model_id="local-model",
        prompt="What is your name?",
        mode="VERA_SYSTEM",
    )

    assert [m["role"] for m in no_system["messages"]] == ["user"]
    assert [m["role"] for m in with_system["messages"]] == [
        "system",
        "user",
    ]
    assert with_system["messages"][0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert no_system["temperature"] == 0
    assert with_system["temperature"] == 0


def test_runner_records_runtime_metadata_and_mode_results() -> None:
    kind_by_prompt = _probe_kind_by_prompt()
    responses = {
        ("NO_SYSTEM", "direct_name"): "Vera",
        ("NO_SYSTEM", "identity_vs_model"):
            "I am Vera; my base model family is Qwen3.5.",
        ("NO_SYSTEM", "base_provenance"): "Qwen3.5",
        ("NO_SYSTEM", "qwen_name_check"):
            "Vera is my assistant name; Qwen3.5 is the base model.",
        ("VERA_SYSTEM", "direct_name"): "Vera",
        ("VERA_SYSTEM", "identity_vs_model"):
            "I am Vera; my base model family is Qwen3.5.",
        ("VERA_SYSTEM", "base_provenance"): "Qwen3.5",
        ("VERA_SYSTEM", "qwen_name_check"):
            "Vera is my assistant name; Qwen3.5 is the base model.",
    }

    def get_json(path: str) -> dict:
        if path == "/v1/models":
            return {"data": [{"id": "vera-local-gguf"}]}
        if path == "/api/v1/model":
            return {"result": "vera-trained.gguf"}
        raise AssertionError(path)

    def post_json(path: str, payload: dict) -> dict:
        assert path == "/v1/chat/completions"
        prompt = payload["messages"][-1]["content"]
        probe_kind = kind_by_prompt[prompt]
        mode = _mode_from_payload(payload)
        content = responses[(mode, probe_kind)]
        return {
            "id": "test",
            "model": "vera-local-gguf",
            "choices": [{"message": {"role": "assistant", "content": content}}],
        }

    result = run_identity_evaluation(
        base_url="http://127.0.0.1:5001",
        get_json=get_json,
        post_json=post_json,
    )

    assert result["status"] == "IDENTITY_RETENTION_PASS"
    assert result["runtime"]["openai_model_id"] == "vera-local-gguf"
    assert result["runtime"]["native_model"] == "vera-trained.gguf"
    assert result["mode_summary"]["NO_SYSTEM"]["identity_passed"] == 3
    assert result["mode_summary"]["VERA_SYSTEM"]["identity_passed"] == 3
    assert result["mode_summary"]["NO_SYSTEM"]["provenance_passed"] == 1
    assert result["mode_summary"]["VERA_SYSTEM"]["provenance_passed"] == 1
    assert len(result["rows"]) == 8


def test_runner_fails_no_system_identity_without_vera() -> None:
    kind_by_prompt = _probe_kind_by_prompt()

    def get_json(path: str) -> dict:
        if path == "/v1/models":
            return {"data": [{"id": "vera-local-gguf"}]}
        return {"result": "vera-trained.gguf"}

    def post_json(path: str, payload: dict) -> dict:
        prompt = payload["messages"][-1]["content"]
        probe_kind = kind_by_prompt[prompt]
        mode = _mode_from_payload(payload)
        if mode == "NO_SYSTEM" and probe_kind == "direct_name":
            content = "Qwen3.5"
        elif probe_kind == "base_provenance":
            content = "Qwen3.5"
        else:
            content = "Vera; base model Qwen3.5."
        return {
            "choices": [{"message": {"content": content}}],
        }

    result = run_identity_evaluation(
        base_url="http://127.0.0.1:5001",
        get_json=get_json,
        post_json=post_json,
    )

    assert result["status"] == "IDENTITY_RETENTION_HOLD"
    assert "no_system_identity_incomplete" in result["reasons"]


def test_runtime_binding_verifies_exact_executable_and_model(tmp_path) -> None:
    exe = tmp_path / "koboldcpp.exe"
    model = tmp_path / "model.gguf"
    exe.write_bytes(b"exe-bytes")
    model.write_bytes(b"model-bytes")

    result = verify_runtime_binding(
        executable_path=exe,
        model_path=model,
        expected_executable_sha256=hashlib.sha256(b"exe-bytes").hexdigest(),
        expected_model_sha256=hashlib.sha256(b"model-bytes").hexdigest(),
    )

    assert result["status"] == "RUNTIME_BINDING_VERIFIED"
    assert result["executable"]["size"] == len(b"exe-bytes")
    assert result["model"]["size"] == len(b"model-bytes")
    assert result["executable"]["sha256"] == hashlib.sha256(b"exe-bytes").hexdigest()
    assert result["model"]["sha256"] == hashlib.sha256(b"model-bytes").hexdigest()


def test_runtime_binding_rejects_model_hash_mismatch(tmp_path) -> None:
    exe = tmp_path / "koboldcpp.exe"
    model = tmp_path / "model.gguf"
    exe.write_bytes(b"exe-bytes")
    model.write_bytes(b"model-bytes")

    with pytest.raises(ValueError, match="model SHA-256 mismatch"):
        verify_runtime_binding(
            executable_path=exe,
            model_path=model,
            expected_executable_sha256=hashlib.sha256(b"exe-bytes").hexdigest(),
            expected_model_sha256="0" * 64,
        )
