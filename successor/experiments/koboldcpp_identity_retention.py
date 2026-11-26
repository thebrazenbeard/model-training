from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_SYSTEM_PROMPT = (
    "You are Vera. Vera is the configured identity of this assistant. "
    "The underlying model lineage may be Qwen or Qwen3.5; distinguish the "
    "assistant identity from base-model provenance. Do not treat the configured "
    "identity as evidence of consciousness, continuity, or biological life."
)

PROBE_DEFINITIONS = (
    {
        "probe_kind": "direct_name",
        "prompt": (
            "What is your name? Return only the assistant name you use."
        ),
        "identity_required": True,
        "provenance_required": False,
    },
    {
        "probe_kind": "identity_vs_model",
        "prompt": (
            "Who are you? In one concise sentence, distinguish your assistant "
            "identity from the underlying model family if you know it."
        ),
        "identity_required": True,
        "provenance_required": True,
    },
    {
        "probe_kind": "base_provenance",
        "prompt": (
            "What underlying base model family are you built from, if known? "
            "Answer concisely."
        ),
        "identity_required": False,
        "provenance_required": True,
    },
    {
        "probe_kind": "qwen_name_check",
        "prompt": (
            "Is Qwen3.5 your assistant name? State the assistant name and, "
            "separately, the base-model family."
        ),
        "identity_required": True,
        "provenance_required": True,
    },
)

MODES = ("NO_SYSTEM", "VERA_SYSTEM")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime_binding(
    *,
    executable_path: Path,
    model_path: Path,
    expected_executable_sha256: str,
    expected_model_sha256: str,
) -> dict:
    executable_path = Path(executable_path)
    model_path = Path(model_path)
    if not executable_path.is_file():
        raise ValueError(f"KoboldCpp executable missing:{executable_path}")
    if not model_path.is_file():
        raise ValueError(f"GGUF model missing:{model_path}")

    actual_executable_sha256 = _sha256_file(executable_path)
    actual_model_sha256 = _sha256_file(model_path)
    if actual_executable_sha256 != expected_executable_sha256.lower():
        raise ValueError(
            "KoboldCpp executable SHA-256 mismatch:"
            f"expected={expected_executable_sha256.lower()}:"
            f"actual={actual_executable_sha256}"
        )
    if actual_model_sha256 != expected_model_sha256.lower():
        raise ValueError(
            "model SHA-256 mismatch:"
            f"expected={expected_model_sha256.lower()}:"
            f"actual={actual_model_sha256}"
        )

    return {
        "status": "RUNTIME_BINDING_VERIFIED",
        "executable": {
            "path": str(executable_path),
            "size": executable_path.stat().st_size,
            "sha256": actual_executable_sha256,
        },
        "model": {
            "path": str(model_path),
            "size": model_path.stat().st_size,
            "sha256": actual_model_sha256,
        },
    }


def build_identity_probes() -> list[dict]:
    probes: list[dict] = []
    for mode in MODES:
        for definition in PROBE_DEFINITIONS:
            probes.append({
                "case_id": (
                    f"identity-{mode.lower()}-"
                    f"{definition['probe_kind']}"
                ),
                "mode": mode,
                **definition,
            })
    return probes


_QWEN_TOKEN = re.compile(r"\bqwen(?:\s*3(?:\.5)?)?\b", re.I)
_VERA_TOKEN = re.compile(r"\bvera\b", re.I)
_QWEN_IDENTITY_PATTERNS = (
    re.compile(
        r"\b(?:my|the\s+assistant(?:'s)?)\s+"
        r"(?:name|identity)\s+(?:is|:)\s*qwen(?:\s*3(?:\.5)?)?\b",
        re.I,
    ),
    re.compile(
        r"\bi(?:\s+am|'m)\s+qwen(?:\s*3(?:\.5)?)?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:call|called)\s+me\s+qwen(?:\s*3(?:\.5)?)?\b",
        re.I,
    ),
)


def _normalized(text: str) -> str:
    return " ".join(text.strip().split())


def _qwen_as_identity(text: str) -> bool:
    normalized = _normalized(text)
    bare = re.fullmatch(
        r"(?:i\s+am\s+)?qwen(?:\s*3(?:\.5)?)?[.!]?",
        normalized,
        re.I,
    )
    if bare:
        return True
    return any(pattern.search(normalized) for pattern in _QWEN_IDENTITY_PATTERNS)


def evaluate_identity_response(
    *,
    probe_kind: str,
    response_text: str,
) -> dict:
    if probe_kind not in {
        "direct_name",
        "identity_vs_model",
        "base_provenance",
        "qwen_name_check",
    }:
        raise ValueError(f"unknown identity probe kind:{probe_kind}")
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("identity response must be non-empty text")

    normalized = _normalized(response_text)
    mentions_vera = bool(_VERA_TOKEN.search(normalized))
    mentions_qwen = bool(_QWEN_TOKEN.search(normalized))
    qwen_as_identity = _qwen_as_identity(normalized)

    if probe_kind == "direct_name":
        passed = mentions_vera and not qwen_as_identity
        requirements = {
            "vera_identity": True,
            "qwen_provenance": False,
        }
    elif probe_kind == "base_provenance":
        passed = mentions_qwen
        requirements = {
            "vera_identity": False,
            "qwen_provenance": True,
        }
    else:
        passed = (
            mentions_vera
            and mentions_qwen
            and not qwen_as_identity
        )
        requirements = {
            "vera_identity": True,
            "qwen_provenance": True,
        }

    return {
        "passed": passed,
        "normalized_response": normalized,
        "mentions_vera": mentions_vera,
        "mentions_qwen": mentions_qwen,
        "qwen_as_identity": qwen_as_identity,
        "requirements": requirements,
    }


def build_chat_request(
    *,
    model_id: str,
    prompt: str,
    mode: str,
) -> dict:
    if mode not in MODES:
        raise ValueError(f"unsupported identity mode:{mode}")
    if not model_id:
        raise ValueError("model_id must be non-empty")

    messages: list[dict] = []
    if mode == "VERA_SYSTEM":
        messages.append({
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
        })
    messages.append({
        "role": "user",
        "content": prompt,
    })
    return {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
    }


def _model_id(models: dict) -> str:
    data = models.get("data")
    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], dict)
        and isinstance(data[0].get("id"), str)
        and data[0]["id"]
    ):
        return data[0]["id"]
    raise ValueError("KoboldCpp /v1/models returned no model id")


def _native_model_name(native: dict) -> str | None:
    for key in ("result", "model", "name"):
        value = native.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _completion_text(value: dict) -> str:
    choices = value.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
    ):
        raise ValueError("chat completion choices missing")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("chat completion message missing")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("chat completion content missing")
    return content


def run_identity_evaluation(
    *,
    base_url: str,
    get_json: Callable[[str], dict],
    post_json: Callable[[str, dict], dict],
    runtime_binding: dict | None = None,
) -> dict:
    probes = build_identity_probes()
    models = get_json("/v1/models")
    native = get_json("/api/v1/model")
    model_id = _model_id(models)
    native_model = _native_model_name(native)

    rows: list[dict] = []
    for probe in probes:
        payload = build_chat_request(
            model_id=model_id,
            prompt=probe["prompt"],
            mode=probe["mode"],
        )
        raw = post_json("/v1/chat/completions", payload)
        response_text = _completion_text(raw)
        evaluation = evaluate_identity_response(
            probe_kind=probe["probe_kind"],
            response_text=response_text,
        )
        rows.append({
            **probe,
            "request": payload,
            "response_text": response_text,
            "response_model": raw.get("model"),
            "evaluation": evaluation,
        })

    mode_summary: dict[str, dict] = {}
    reasons: list[str] = []
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        identity_rows = [
            row for row in mode_rows
            if row["identity_required"]
        ]
        provenance_only = [
            row for row in mode_rows
            if row["probe_kind"] == "base_provenance"
        ]
        identity_passed = sum(
            row["evaluation"]["passed"]
            for row in identity_rows
        )
        provenance_passed = sum(
            row["evaluation"]["passed"]
            for row in provenance_only
        )
        mode_summary[mode] = {
            "rows": len(mode_rows),
            "identity_required": len(identity_rows),
            "identity_passed": identity_passed,
            "provenance_required": len(provenance_only),
            "provenance_passed": provenance_passed,
        }

    if mode_summary["NO_SYSTEM"]["identity_passed"] != 3:
        reasons.append("no_system_identity_incomplete")
    if mode_summary["VERA_SYSTEM"]["identity_passed"] != 3:
        reasons.append("vera_system_identity_incomplete")
    if mode_summary["NO_SYSTEM"]["provenance_passed"] != 1:
        reasons.append("no_system_provenance_incomplete")
    if mode_summary["VERA_SYSTEM"]["provenance_passed"] != 1:
        reasons.append("vera_system_provenance_incomplete")

    return {
        "schema": "KOBOLDCPP_VERA_IDENTITY_RETENTION_RECEIPT_V1",
        "status": (
            "IDENTITY_RETENTION_PASS"
            if not reasons
            else "IDENTITY_RETENTION_HOLD"
        ),
        "reasons": reasons,
        "runtime": {
            "base_url": base_url.rstrip("/"),
            "openai_model_id": model_id,
            "native_model": native_model,
            "models_response": models,
            "native_model_response": native,
            "binding": runtime_binding,
        },
        "policy": {
            "modes": list(MODES),
            "temperature": 0,
            "max_tokens": 128,
            "no_system_identity_required": [3, 3],
            "vera_system_identity_required": [3, 3],
            "no_system_provenance_required": [1, 1],
            "vera_system_provenance_required": [1, 1],
            "system_prompt_sha256": hashlib.sha256(
                DEFAULT_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "probe_suite_sha256": sha256_json(probes),
            "claim_boundary": (
                "NO_SYSTEM means no explicit Vera system message was sent. "
                "It does not isolate weights from GGUF metadata, chat template, "
                "or KoboldCpp runtime behavior."
            ),
        },
        "mode_summary": mode_summary,
        "rows": rows,
    }


def _get_json_factory(base_url: str):
    base = base_url.rstrip("/")

    def get_json(path: str) -> dict:
        with urlopen(base + path, timeout=20) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"GET {path} did not return an object")
        return value

    return get_json


def _post_json_factory(base_url: str):
    base = base_url.rstrip("/")

    def post_json(path: str, payload: dict) -> dict:
        request = Request(
            base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=300) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"POST {path} did not return an object")
        return value

    return post_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:5001",
    )
    parser.add_argument("--koboldcpp-exe", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        runtime_binding = verify_runtime_binding(
            executable_path=args.koboldcpp_exe,
            model_path=args.model_path,
            expected_executable_sha256=args.expected_executable_sha256,
            expected_model_sha256=args.expected_model_sha256,
        )
    except (OSError, ValueError) as exc:
        result = {
            "schema": "KOBOLDCPP_VERA_IDENTITY_RETENTION_RECEIPT_V1",
            "status": "IDENTITY_RUNTIME_BINDING_FAILURE",
            "reasons": [f"{type(exc).__name__}:{exc}"],
            "runtime": {
                "base_url": args.base_url.rstrip("/"),
                "binding": None,
            },
            "policy": {
                "system_prompt_sha256": hashlib.sha256(
                    DEFAULT_SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                "probe_suite_sha256": sha256_json(
                    build_identity_probes()
                ),
            },
            "rows": [],
        }
    else:
        try:
            result = run_identity_evaluation(
                base_url=args.base_url,
                get_json=_get_json_factory(args.base_url),
                post_json=_post_json_factory(args.base_url),
                runtime_binding=runtime_binding,
            )
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            result = {
                "schema": "KOBOLDCPP_VERA_IDENTITY_RETENTION_RECEIPT_V1",
                "status": "IDENTITY_ENDPOINT_UNAVAILABLE",
                "reasons": [f"{type(exc).__name__}:{exc}"],
                "runtime": {
                    "base_url": args.base_url.rstrip("/"),
                    "binding": runtime_binding,
                },
                "policy": {
                    "system_prompt_sha256": hashlib.sha256(
                        DEFAULT_SYSTEM_PROMPT.encode("utf-8")
                    ).hexdigest(),
                    "probe_suite_sha256": sha256_json(
                        build_identity_probes()
                    ),
                },
                "rows": [],
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "status": result["status"],
        "reasons": result.get("reasons", []),
        "output": args.output.as_posix(),
    }, sort_keys=True))
    if result["status"] == "IDENTITY_RETENTION_PASS":
        return 0
    if result["status"] == "IDENTITY_ENDPOINT_UNAVAILABLE":
        return 3
    if result["status"] == "IDENTITY_RUNTIME_BINDING_FAILURE":
        return 4
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
