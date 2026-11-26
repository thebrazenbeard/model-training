import json

from fastapi.testclient import TestClient

import successor.openwebui_server as server_module
from successor.openwebui_server import (
    LlamaServerBackend,
    _extract_tool_calls,
    _ensure_smollm_thinking_mode,
    _messages_for_upstream,
    _openai_tools_to_xml_tools,
    create_app,
)


class FakeBackend:
    candidate_digest = "a" * 64

    def __init__(self, response="hello"):
        self.response = response
        self.calls = []

    def generate(self, messages, *, tools, temperature, top_p, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        )
        return self.response


def test_openai_tools_are_reduced_for_smollm_xml_template():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    assert _openai_tools_to_xml_tools(tools) == [
        {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]


def test_extract_tool_call_maps_to_openai_shape():
    content, calls = _extract_tool_calls(
        'I will check. <tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>'
    )
    assert content == "I will check."
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt"}




class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload




def test_bare_json_tool_call_is_normalized_only_when_enabled():
    text = '{"name":"get_current_time","arguments":{"timezone":"UTC"}}'
    content, calls = _extract_tool_calls(text)
    assert content == text
    assert calls == []

    content, calls = _extract_tool_calls(text, allow_bare_json=True)
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_current_time"
    assert json.loads(calls[0]["function"]["arguments"]) == {"timezone": "UTC"}


def test_bare_json_is_not_misclassified_without_tool_shape():
    text = '{"answer":"UTC is Coordinated Universal Time"}'
    content, calls = _extract_tool_calls(text, allow_bare_json=True)
    assert content == text
    assert calls == []




def test_smollm_thinking_mode_defaults_system_to_no_think():
    messages = [
        {"role": "system", "content": "You are Computer."},
        {"role": "user", "content": "Hello"},
    ]
    result = _ensure_smollm_thinking_mode(messages)
    assert result[0]["content"].startswith("/no_think\n")
    assert result[0]["content"].endswith("You are Computer.")


def test_smollm_thinking_mode_preserves_explicit_mode():
    messages = [
        {"role": "system", "content": "/think\nYou are Computer."},
        {"role": "user", "content": "Hello"},
    ]
    assert _ensure_smollm_thinking_mode(messages) == messages


def test_smollm_thinking_mode_adds_system_when_missing():
    messages = [{"role": "user", "content": "Hello"}]
    result = _ensure_smollm_thinking_mode(messages)
    assert result[0] == {"role": "system", "content": "/no_think"}
    assert result[1] == messages[0]


def test_messages_for_upstream_rehydrates_assistant_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"x.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file contents",
        },
    ]
    forwarded = _messages_for_upstream(messages)
    assert "<tool_call>" in forwarded[0]["content"]
    assert '"name":"read_file"' in forwarded[0]["content"]
    assert forwarded[1]["role"] == "tool"
    assert forwarded[1]["tool_call_id"] == "call_1"


def test_llama_server_backend_forwards_tools_and_returns_smollm_tool_xml(monkeypatch):
    observed = []

    def fake_urlopen(request, timeout):
        if isinstance(request, str):
            observed.append(("models", request, None))
            return FakeHTTPResponse(
                {"data": [{"id": "C:/Vera/successor/deploy/base.gguf"}]}
            )
        body = json.loads(request.data.decode("utf-8"))
        observed.append(("chat", request.full_url, body))
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '<tool_call>{"name":"get_current_time",'
                                '"arguments":{"timezone":"UTC"}}</tool_call>'
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(server_module.urllib_request, "urlopen", fake_urlopen)
    backend = LlamaServerBackend(
        upstream_url="http://127.0.0.1:11435",
        candidate_digest="d" * 64,
    )
    result = backend.generate(
        [{"role": "user", "content": "What time is it?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                },
            }
        ],
        temperature=0,
        top_p=None,
        max_tokens=64,
    )
    assert "<tool_call>" in result
    assert observed[1][2]["tool_choice"] == "auto"
    assert observed[1][2]["tools"][0]["function"]["name"] == "get_current_time"




def test_llama_server_backend_defaults_max_tokens_when_computer_omits_cap(monkeypatch):
    observed = []

    def fake_urlopen(request, timeout):
        if isinstance(request, str):
            return FakeHTTPResponse({"data": [{"id": "base.gguf"}]})
        body = json.loads(request.data.decode("utf-8"))
        observed.append(body)
        return FakeHTTPResponse(
            {"choices": [{"message": {"content": "Hello."}}]}
        )

    monkeypatch.setattr(server_module.urllib_request, "urlopen", fake_urlopen)
    backend = LlamaServerBackend(
        upstream_url="http://127.0.0.1:11435",
        candidate_digest="d" * 64,
    )
    assert backend.generate(
        [{"role": "system", "content": "You are Computer."}, {"role": "user", "content": "Hello"}],
        tools=None,
        temperature=0,
        top_p=None,
        max_tokens=None,
    ) == "Hello."
    assert observed[0]["max_tokens"] == 512
    assert observed[0]["messages"][0]["content"].startswith("/no_think\n")


def test_proxy_app_exposes_true_openai_tool_calls(monkeypatch):
    def fake_urlopen(request, timeout):
        if isinstance(request, str):
            return FakeHTTPResponse({"data": [{"id": "base.gguf"}]})
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '<tool_call>{"name":"get_current_time",'
                                '"arguments":{"timezone":"UTC"}}</tool_call>'
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(server_module.urllib_request, "urlopen", fake_urlopen)
    backend = LlamaServerBackend(
        upstream_url="http://127.0.0.1:11435",
        candidate_digest="d" * 64,
    )
    client = TestClient(create_app(backend, "vera-v3-full-dev"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vera-v3-full-dev",
            "messages": [{"role": "user", "content": "time"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    call = body["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_current_time"
    assert json.loads(call["function"]["arguments"]) == {"timezone": "UTC"}


def test_models_endpoint_exposes_exact_dev_model():
    backend = FakeBackend()
    client = TestClient(create_app(backend, "vera-v3-full-dev"))
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "vera-v3-full-dev"


def test_chat_completion_returns_content_and_candidate_fingerprint():
    backend = FakeBackend("hello from Vera")
    client = TestClient(create_app(backend, "vera-v3-full-dev"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vera-v3-full-dev",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello from Vera"
    assert body["system_fingerprint"] == "a" * 16
    assert backend.calls[0]["temperature"] == 0


def test_chat_completion_maps_model_tool_call():
    backend = FakeBackend(
        '<tool_call>{"name":"read_file","arguments":{"path":"x.py"}}</tool_call>'
    )
    client = TestClient(create_app(backend, "vera-v3-full-dev"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vera-v3-full-dev",
            "messages": [{"role": "user", "content": "read x.py"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = body["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "read_file"




def test_streaming_tool_call_has_openai_index():
    backend = FakeBackend(
        '<tool_call>{"name":"get_current_time","arguments":{"timezone":"UTC"}}</tool_call>'
    )
    client = TestClient(create_app(backend, "vera-v3-full-dev"))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "vera-v3-full-dev",
            "messages": [{"role": "user", "content": "time"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "stream": True,
        },
    ) as response:
        text = "".join(response.iter_text())
    assert response.status_code == 200
    payloads = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payloads.append(json.loads(line[6:]))
    tool_delta = next(
        p["choices"][0]["delta"]["tool_calls"]
        for p in payloads
        if p["choices"][0]["delta"].get("tool_calls")
    )
    assert tool_delta[0]["index"] == 0
    assert tool_delta[0]["function"]["name"] == "get_current_time"


def test_streaming_completion_emits_openai_sse_done_marker():
    backend = FakeBackend("streamed")
    client = TestClient(create_app(backend, "vera-v3-full-dev"))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "vera-v3-full-dev",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        text = "".join(response.iter_text())
    assert response.status_code == 200
    assert "chat.completion.chunk" in text
    assert "data: [DONE]" in text


def test_unknown_model_is_rejected():
    backend = FakeBackend()
    client = TestClient(create_app(backend, "vera-v3-full-dev"))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "wrong",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 404
