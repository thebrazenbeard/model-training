from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
import uuid
from typing import Any, Protocol
from urllib import request as urllib_request

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from successor.vera_lab.smollm_adapter import SmolLMPEFTAdapter


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None


class ChatBackend(Protocol):
    candidate_digest: str

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> str:
        ...


def _openai_tools_to_xml_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    converted: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted.append(
            {
                "name": name,
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("messages must contain objects")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role!r}")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        clean: dict[str, Any] = {"role": role, "content": content}
        if role == "tool":
            if isinstance(message.get("name"), str):
                clean["name"] = message["name"]
            if isinstance(message.get("tool_call_id"), str):
                clean["tool_call_id"] = message["tool_call_id"]
        sanitized.append(clean)
    if not sanitized:
        raise ValueError("at least one message is required")
    return sanitized


def _extract_tool_calls(
    text: str,
    *,
    allow_bare_json: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for raw in _TOOL_CALL_RE.findall(text):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        calls.append(
            {
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )
    content = _TOOL_CALL_RE.sub("", text).strip()
    if not calls and allow_bare_json and content:
        candidate = content
        if candidate.startswith("```") and candidate.endswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()
                if candidate.lower().startswith("json\n"):
                    candidate = candidate[5:].strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if isinstance(name, str) and name and isinstance(arguments, dict):
                calls.append(
                    {
                        "id": "call_" + uuid.uuid4().hex[:24],
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
                content = ""
    return content, calls


@dataclass
class LlamaServerBackend:
    upstream_url: str
    candidate_digest: str
    upstream_model: str | None = None

    def _model_id(self) -> str:
        if self.upstream_model:
            return self.upstream_model
        with urllib_request.urlopen(
            self.upstream_url.rstrip("/") + "/v1/models",
            timeout=15,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("upstream returned no models")
        model_id = data[0].get("id")
        if not isinstance(model_id, str) or not model_id:
            raise RuntimeError("upstream model id is invalid")
        self.upstream_model = model_id
        return model_id

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> str:
        safe_messages = _ensure_smollm_thinking_mode(
            _messages_for_upstream(messages)
        )
        payload: dict[str, Any] = {
            "model": self._model_id(),
            "messages": safe_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        payload["max_tokens"] = max_tokens if max_tokens is not None else 512
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(
            self.upstream_url.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib_request.urlopen(req, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"upstream request failed: {exc}") from exc
        try:
            message = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("upstream response missing assistant message") from exc
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return _tool_calls_to_xml(tool_calls)
        raise RuntimeError("upstream returned an empty response")


def _tool_calls_to_xml(tool_calls: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"value": arguments}
        payload = {"name": name, "arguments": arguments}
        chunks.append(
            "<tool_call>\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n</tool_call>"
        )
    return "\n".join(chunks)


def _ensure_smollm_thinking_mode(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Default SmolLM3 to non-thinking chat unless the caller chose a mode explicitly."""
    result = [dict(message) for message in messages]
    explicit_mode = any(
        isinstance(message.get("content"), str)
        and ("/no_think" in message["content"] or "/think" in message["content"])
        for message in result
        if message.get("role") == "system"
    )
    if explicit_mode:
        return result
    for message in result:
        if message.get("role") == "system":
            message["content"] = "/no_think\n" + str(message.get("content", ""))
            return result
    result.insert(0, {"role": "system", "content": "/no_think"})
    return result


def _messages_for_upstream(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = _sanitize_messages(messages)
    forwarded: list[dict[str, Any]] = []
    for original, clean in zip(messages, safe):
        item = dict(clean)
        if clean["role"] == "assistant":
            tool_calls = original.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                xml = _tool_calls_to_xml(tool_calls)
                content = item.get("content", "")
                item["content"] = ((content + "\n") if content else "") + xml
        forwarded.append(item)
    return forwarded


@dataclass
class PEFTChatBackend:
    adapter: SmolLMPEFTAdapter

    @property
    def candidate_digest(self) -> str:
        return self.adapter.candidate_digest

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> str:
        tokenizer, model, torch = self.adapter._ensure_stack()
        safe_messages = _sanitize_messages(messages)
        xml_tools = _openai_tools_to_xml_tools(tools)

        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if xml_tools:
            template_kwargs["xml_tools"] = xml_tools
        prompt = tokenizer.apply_chat_template(safe_messages, **template_kwargs)
        batch = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")

        requested_max = max_tokens if max_tokens is not None else 512
        requested_max = max(1, min(int(requested_max), 4096))
        kwargs: dict[str, Any] = {
            "max_new_tokens": requested_max,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature is None or float(temperature) <= 0:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = max(0.01, float(temperature))
            if top_p is not None:
                kwargs["top_p"] = max(0.01, min(1.0, float(top_p)))

        with torch.inference_mode():
            generated = model.generate(**batch, **kwargs)
        new_tokens = generated[0, batch["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("model returned an empty response")
        return answer


def _completion_payload(
    *,
    model_id: str,
    candidate_digest: str,
    content: str,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "system_fingerprint": candidate_digest[:16],
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_chunks(payload: dict[str, Any]):
    choice = payload["choices"][0]
    message = choice["message"]
    first_delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content"):
        first_delta["content"] = message["content"]
    if message.get("tool_calls"):
        first_delta["tool_calls"] = [
            {**call, "index": index}
            for index, call in enumerate(message["tool_calls"])
        ]
    chunk = {
        "id": payload["id"],
        "object": "chat.completion.chunk",
        "created": payload["created"],
        "model": payload["model"],
        "choices": [{"index": 0, "delta": first_delta, "finish_reason": None}],
    }
    yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
    final = {
        "id": payload["id"],
        "object": "chat.completion.chunk",
        "created": payload["created"],
        "model": payload["model"],
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": choice["finish_reason"],
            }
        ],
    }
    yield "data: " + json.dumps(final, ensure_ascii=False) + "\n\n"
    yield "data: [DONE]\n\n"


def create_app(backend: ChatBackend, model_id: str) -> FastAPI:
    app = FastAPI(title="Vera Successor OpenAI-Compatible Server")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": model_id,
            "candidate_digest": backend.candidate_digest,
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        if request.model != model_id:
            raise HTTPException(status_code=404, detail="unknown model")
        max_tokens = (
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens
        )
        try:
            raw = backend.generate(
                request.messages,
                tools=request.tools,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=max_tokens,
            )
            content, tool_calls = _extract_tool_calls(
                raw,
                allow_bare_json=bool(request.tools),
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload = _completion_payload(
            model_id=model_id,
            candidate_digest=backend.candidate_digest,
            content=content,
            tool_calls=tool_calls,
        )
        if request.stream:
            return StreamingResponse(_stream_chunks(payload), media_type="text/event-stream")
        return JSONResponse(payload)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve or proxy a frozen Vera candidate over OpenAI-compatible HTTP")
    parser.add_argument("--base-manifest")
    parser.add_argument("--adapter")
    parser.add_argument("--upstream-url")
    parser.add_argument("--upstream-model")
    parser.add_argument("--candidate-digest")
    parser.add_argument("--model-id", default="vera-v3-dev")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = _parse_args()
    if args.upstream_url:
        if not args.candidate_digest:
            raise SystemExit("--candidate-digest is required with --upstream-url")
        backend: ChatBackend = LlamaServerBackend(
            upstream_url=args.upstream_url,
            upstream_model=args.upstream_model,
            candidate_digest=args.candidate_digest,
        )
    else:
        if not args.base_manifest or not args.adapter:
            raise SystemExit("--base-manifest and --adapter are required without --upstream-url")
        adapter = SmolLMPEFTAdapter(
            base_manifest_path=Path(args.base_manifest),
            adapter_path=Path(args.adapter),
            generation_config={"max_new_tokens": 512, "do_sample": False},
        )
        backend = PEFTChatBackend(adapter)
    app = create_app(backend, args.model_id)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
