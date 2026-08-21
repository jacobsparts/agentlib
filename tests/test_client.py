import json

import pytest

from agentlib.client import ValidationError, _extract_tool_calls_json, _preprocess_tool_call_response


def test_one_shot_user_shorthand_returns_truthy_text_blocks(monkeypatch):
    from agentlib import client as client_module
    from agentlib.client import one_shot

    calls = []

    class Client:
        def __init__(self, model):
            assert model == "test/model"

        def call(self, messages, tools=None):
            calls.append((messages, tools))
            return {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "text": "private"},
                    {"type": "text", "text": ""},
                    {"type": "commentary", "text": "progress"},
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": None},
                    {"type": "text", "text": "world"},
                ],
            }

    monkeypatch.setattr(client_module, "LLMClient", Client)

    result = one_shot(
        "test/model",
        system="Be concise.",
        user="Hello",
    )

    assert result == "hello\nworld"
    assert calls == [([
        {
            "role": "system",
            "content": [{"type": "text", "text": "Be concise."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        },
    ], None)]


def test_one_shot_messages_returns_canonical_response(monkeypatch):
    from agentlib import client as client_module
    from agentlib.client import one_shot

    response = {
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
    }
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}],
    }]

    class Client:
        def __init__(self, model):
            assert model == "test/model"

        def call(self, request, tools=None):
            assert request is messages
            assert tools is None
            return response

    monkeypatch.setattr(client_module, "LLMClient", Client)

    assert one_shot("test/model", messages=messages) is response


def test_one_shot_tools_returns_canonical_response(monkeypatch):
    from pydantic import BaseModel

    from agentlib import client as client_module
    from agentlib.client import one_shot

    class Classification(BaseModel):
        category: str

    response = {
        "role": "assistant",
        "content": [{
            "type": "tool_call",
            "id": "call_1",
            "name": "classify",
            "args": {"category": "question"},
        }],
    }

    class Client:
        def __init__(self, model):
            assert model == "test/model"

        def call(self, messages, tools=None):
            assert messages == [{
                "role": "user",
                "content": [{"type": "text", "text": "Classify this"}],
            }]
            assert tools == {"classify": Classification}
            return response

    monkeypatch.setattr(client_module, "LLMClient", Client)

    result = one_shot(
        "test/model",
        user="Classify this",
        tools={"classify": Classification},
    )

    assert result is response


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "one of messages or user is required"),
        (
            {"messages": [], "user": "hello"},
            "messages is mutually exclusive with system and user",
        ),
        (
            {"messages": [], "system": "system"},
            "messages is mutually exclusive with system and user",
        ),
    ],
)
def test_one_shot_rejects_invalid_input(kwargs, message):
    from agentlib.client import one_shot

    with pytest.raises(ValueError, match=message):
        one_shot("test/model", **kwargs)


def test_extract_tool_calls_json_single_document():
    content = 'prefix {"function_calls":[{"name":"analyze","arguments":{"x":1}}]} suffix'
    tool_calls, json_start_index, json_end_index = _extract_tool_calls_json(content)
    assert tool_calls == {"function_calls": [{"name": "analyze", "arguments": {"x": 1}}]}
    assert content[json_start_index] == "{"
    assert content[json_end_index] == "}"


def test_preprocess_tool_call_response_merges_multiple_documents():
    block1 = {"function_calls": [{"name": "analyze", "arguments": {"x": 1}}]}
    block2 = {"function_calls": [{"name": "decide", "arguments": {"y": 2}}]}
    content = (
        "```json\n"
        f"{json.dumps(block1)}\n"
        "```\n"
        "```json\n"
        f"{json.dumps(block2)}\n"
        "```"
    )
    normalized = _preprocess_tool_call_response(content)
    assert json.loads(normalized) == {
        "function_calls": [
            {"name": "analyze", "arguments": {"x": 1}},
            {"name": "decide", "arguments": {"y": 2}},
        ]
    }


def test_preprocess_tool_call_response_leaves_single_document_unchanged():
    content = '{"function_calls": [{"name": "analyze", "arguments": {"x": 1}}]}'
    assert _preprocess_tool_call_response(content) == content


def test_preprocess_tool_call_response_extracts_json_after_prose():
    content = (
        "Looking at the SKU first.\n\n"
        "**Baselines:**\n"
        "- Pack=10, standard: $6.04\n\n"
        '{"function_calls": [{"name": "decide", "arguments": {"x": 1}}]}'
    )
    normalized = _preprocess_tool_call_response(content)
    assert json.loads(normalized) == {
        "function_calls": [{"name": "decide", "arguments": {"x": 1}}]
    }


def test_preprocess_tool_call_response_closes_missing_outer_brace():
    content = '{"function_calls": [{"name": "decide", "arguments": {"x": 1}}]'
    normalized = _preprocess_tool_call_response(content)
    assert json.loads(normalized) == {
        "function_calls": [{"name": "decide", "arguments": {"x": 1}}]
    }


def test_preprocess_tool_call_response_closes_missing_bracket_and_brace():
    content = '{"function_calls": [{"name": "decide", "arguments": {"x": 1}}'
    normalized = _preprocess_tool_call_response(content)
    assert json.loads(normalized) == {
        "function_calls": [{"name": "decide", "arguments": {"x": 1}}]
    }


def test_extract_tool_calls_json_ignores_prose_braces_before_payload():
    content = (
        "Example shape: {not actually json}\n"
        'Then the real payload: {"function_calls": [{"name": "decide", "arguments": {"x": 1}}]}'
    )
    tool_calls, json_start_index, json_end_index = _extract_tool_calls_json(content)
    assert tool_calls == {"function_calls": [{"name": "decide", "arguments": {"x": 1}}]}
    assert content[json_start_index] == "{"
    assert content[json_end_index] == "}"


def test_extract_tool_calls_json_rejects_invalid_function_calls_type():
    content = '{"function_calls": {"name": "analyze", "arguments": {}}}'
    with pytest.raises(ValidationError, match="function_calls must be a list"):
        _extract_tool_calls_json(content)

def test_native_tool_validation_retry_uses_temporary_feedback(monkeypatch):
    from pydantic import ConfigDict, create_model

    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    ToolSpec = create_model(
        "Think",
        __config__=ConfigDict(extra="forbid"),
        notes=(str, ...),
    )
    ToolSpec.__doc__ = "Think tool."
    calls = []

    def fake_call(messages, tools):
        calls.append(messages)
        arguments = (
            {"notes": "ok", "reasoning": "extra"}
            if len(calls) == 1
            else {"notes": "ok"}
        )
        return {
            "role": "assistant",
            "content": [{
                "type": "tool_call",
                "id": "call_1",
                "name": "think",
                "args": arguments,
            }],
        }

    monkeypatch.setattr(client, "_call", fake_call)

    result = client.tool_call_native(
        [{"role": "user", "content": [{"type": "text", "text": "do it"}]}],
        {"think": ToolSpec},
        retry=1,
    )

    assert result["content"][0]["args"] == {"notes": "ok"}
    assert len(calls) == 2
    assert calls[0] == [{
        "role": "user",
        "content": [{"type": "text", "text": "do it"}],
    }]
    assert calls[1][0] == calls[0][0]
    assert calls[1][1]["role"] == "assistant"
    assert calls[1][1]["content"][0]["args"]["reasoning"] == "extra"
    assert calls[1][2]["role"] == "user"
    feedback = calls[1][2]["content"][0]["text"]
    assert "ERROR: Invalid arguments for tool 'think'" in feedback
    assert "valid tool call only" in feedback


def _container_tool():
    from typing import Optional

    from pydantic import BaseModel, create_model

    class Fact(BaseModel):
        sources: list[str]

    tool = create_model(
        "Submit",
        fact=(Fact, ...),
        tags=(list[str], ...),
        note=(str, ...),
        optional_fact=(Optional[Fact], None),
        optional_tags=(Optional[list[str]], None),
        optional_note=(Optional[str], None),
    )
    tool.__doc__ = "Submit structured facts."
    return tool


def _stringified_containers():
    return {
        "fact": json.dumps({"sources": ["source"]}),
        "tags": json.dumps(["tag"]),
        "note": "{}",
        "optional_fact": "null",
        "optional_tags": "null",
        "optional_note": "null",
    }


@pytest.mark.parametrize("native", [True, False])
def test_tool_call_normalizes_json_stringified_containers(monkeypatch, native):
    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    tool = _container_tool()
    arguments = _stringified_containers()
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": "submit"}],
    }]
    if native:
        monkeypatch.setattr(client, "_call", lambda messages, tools: {
            "role": "assistant",
            "content": [{
                "type": "tool_call",
                "id": "call_1",
                "name": "submit",
                "args": arguments,
            }],
        })
        result = client.tool_call_native(
            messages,
            {"submit": tool},
            retry=0,
        )
    else:
        monkeypatch.setattr(client, "_call", lambda messages: {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "function_calls": [{
                        "name": "submit",
                        "arguments": arguments,
                    }]
                }),
            }],
        })
        result = client.tool_call_shim(
            messages,
            {"submit": tool},
            retry=0,
        )

    normalized = result["content"][-1]["args"]
    tool.model_validate(normalized)
    assert isinstance(normalized["fact"], dict)
    assert isinstance(normalized["tags"], list)
    assert isinstance(normalized["note"], str)
    assert normalized["optional_fact"] is None
    assert normalized["optional_tags"] is None
    assert normalized["optional_note"] == "null"


def test_emulated_tool_calls_receive_unique_ids_and_ids_stay_off_wire(monkeypatch):
    from pydantic import create_model

    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    ToolSpec = create_model("Lookup", value=(int, ...))
    ToolSpec.__doc__ = "Look up a value."

    monkeypatch.setattr(client, "_call", lambda messages: {
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": json.dumps({
                "function_calls": [
                    {"name": "lookup", "arguments": {"value": 1}},
                    {"name": "lookup", "arguments": {"value": 2}},
                ]
            }),
        }],
    })

    result = client.tool_call_shim(
        [{
            "role": "user",
            "content": [{"type": "text", "text": "look these up"}],
        }],
        {"lookup": ToolSpec},
        retry=0,
    )

    calls = [
        block
        for block in result["content"]
        if block["type"] == "tool_call"
    ]
    ids = [call["id"] for call in calls]
    assert all(
        call_id.startswith("call_") and len(call_id) == 37
        for call_id in ids
    )
    assert len(set(ids)) == 2

    prepared_call = client.prepare_message(result)
    prepared_text = prepared_call["content"][0]["text"]
    assert all(call_id not in prepared_text for call_id in ids)
    assert json.loads(prepared_text) == {
        "function_calls": [
            {"name": "lookup", "arguments": {"value": 1}},
            {"name": "lookup", "arguments": {"value": 2}},
        ]
    }

    prepared_result = client.prepare_message({
        "role": "tool",
        "name": "lookup",
        "content": [{"type": "text", "text": "result"}],
        "tool_call_id": ids[0],
    })
    assert prepared_result == {
        "role": "user",
        "content": [{"type": "text", "text": "lookup: result"}],
    }




def test_context_budget_does_not_enforce_without_learned_token_ratio():
    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    client.model_config = {"max_input_tokens": 1}
    if hasattr(client.usage_tracker, "input_tokens_per_byte"):
        client.usage_tracker.input_tokens_per_byte.pop(client.model_name, None)

    client._validate_context_budget(10_000)


def test_context_budget_uses_current_input_bytes_with_learned_ratio():
    from agentlib.client import ContextOverflowError, LLMClient

    client = LLMClient("sonnet")
    client.model_config = {"max_input_tokens": 4_050}
    client.usage_tracker.input_tokens_per_byte = {client.model_name: 0.01}

    client._validate_context_budget(1_000)

    with pytest.raises(ContextOverflowError, match="estimated input"):
        client._validate_context_budget(10_000)


def test_input_token_ratio_updates_from_prompt_plus_cached_tokens():
    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    client.usage_tracker.input_tokens_per_byte = {}
    client._update_input_tokens_per_byte(
        1_000,
        {
            "prompt_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 50},
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    )

    assert client.usage_tracker.input_tokens_per_byte[client.model_name] == pytest.approx(0.2)


def test_input_token_ratio_handles_uncached_input_with_separate_cache_tokens():
    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    client.usage_tracker.input_tokens_per_byte = {}
    client._update_input_tokens_per_byte(
        1_000,
        {
            "input_tokens": 110,
            "cache_read_input_tokens": 3716,
            "output_tokens": 2935,
            "output_tokens_details": {"thinking_tokens": 384},
        },
    )

    assert client.usage_tracker.input_tokens_per_byte[client.model_name] == pytest.approx(3.826)


def test_usage_normalization_handles_uncached_input_with_separate_cache_tokens():
    from agentlib.utils import UsageTracker

    tracker = UsageTracker()
    usage = tracker._normalize(
        "sonnet",
        {
            "input_tokens": 110,
            "cache_read_input_tokens": 3716,
            "output_tokens": 2935,
            "output_tokens_details": {"thinking_tokens": 384},
        },
    )

    assert usage["prompt_tokens"] == 110
    assert usage["cached_tokens"] == 3716
    assert usage["completion_tokens"] == 2935
    assert usage["reasoning_tokens"] == 384


@pytest.mark.parametrize(
    "api_type,method_name",
    [
        ("completions", "_call_completions"),
        ("messages", "_call_messages"),
        ("gemini", "_call_gemini"),
    ],
)
def test_client_preserves_private_message_metadata_until_transport(monkeypatch, api_type, method_name):
    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    client.model_config = {
        "api_type": api_type,
        "model": "test-model",
        "context_window": 1_000_000,
    }
    captured = {}
    private_message = {
        "role": "user",
        "content": [{"type": "text", "text": "visible"}],
        "_stdout": "large hidden stdout",
        "_render_segments": [{"type": "stdout", "content": "large hidden stdout"}],
        "_final_result": "hidden final result",
        "_attachment_refs": ["file.py"],
        "_event_seq": 123,
    }

    def fake_validate(input_bytes):
        captured["input_bytes"] = input_bytes

    def fake_provider_call(messages, tools):
        captured["messages"] = messages
        captured["tools"] = tools
        return {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(client, "_validate_context_budget", fake_validate)
    monkeypatch.setattr(client, method_name, fake_provider_call)

    canonical_input = [{
        **private_message,
        "content": [{"type": "text", "text": "visible"}],
    }]
    client._call(canonical_input)

    # Dispatch preserves private metadata so transports can consume it.
    assert captured["messages"] == canonical_input
    assert captured["tools"] is None
    # Sizing ignores private metadata.
    assert captured["input_bytes"] == client._input_bytes(
        [{"role": "user", "content": [{"type": "text", "text": "visible"}]}],
        None,
    )
    assert client._public_messages(captured["messages"]) == [
        {"role": "user", "content": [{"type": "text", "text": "visible"}]}
    ]


def test_public_messages_helper_strips_underscore_keys():
    from agentlib.client import LLMClient

    assert LLMClient._public_messages([
        {
            "role": "user",
            "content": [{"type": "text", "text": "visible"}],
            "_stdout": "hidden",
            "_event_seq": 1,
        }
    ]) == [{
        "role": "user",
        "content": [{"type": "text", "text": "visible"}],
    }]





def test_unknown_transport_types_raise_not_implemented():
    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    client.model_config["api_type"] = "completions"
    with pytest.raises(NotImplementedError, match="Unknown transport content type"):
        client._completions_messages([{
            "role": "user",
            "content": [{"type": "unsupported_transport_type"}],
        }])

    with pytest.raises(NotImplementedError, match="Unknown Responses output type"):
        client._parse_responses_result({
            "output": [{"type": "future_responses_type"}],
        })


def test_provider_media_validation_and_filepath_projection(tmp_path):
    from agentlib.client import BadRequestError, LLMClient

    png_bytes = b"\x89PNG\r\n\x1a\nfake_png_data"

    # Provider media validation
    completions_client = LLMClient("sonnet")
    completions_client.model_config["api_type"] = "completions"
    with pytest.raises(BadRequestError, match="Audio input is not supported by OpenAI completions API"):
        completions_client.validate_media_type("audio/wav")

    anthropic_client = LLMClient("sonnet")
    anthropic_client.model_config["api_type"] = "messages"
    with pytest.raises(BadRequestError, match="Audio input is not supported by Anthropic Messages API"):
        anthropic_client.validate_media_type("audio/wav")

    responses_client = LLMClient("sonnet")
    responses_client.model_config["api_type"] = "responses"
    with pytest.raises(BadRequestError, match="Audio input is not supported by OpenAI Responses API"):
        responses_client.validate_media_type("audio/wav")

    gemini_client = LLMClient("google/gemini-3.6-flash")
    gemini_client.model_config["api_type"] = "gemini"
    # Gemini accepts audio
    gemini_client.validate_media_type("audio/wav")

    # Filepath data projection
    fpath = tmp_path / "test.png"
    fpath.write_bytes(png_bytes)
    fattachment = {"type": "attachment", "media_type": "image/png", "data_type": "filepath", "data": str(fpath)}
    projected = gemini_client._gemini_attachment(fattachment)
    assert projected["inlineData"]["mimeType"] == "image/png"


def test_gemini_preserves_thought_signatures_in_canonical_messages(
    monkeypatch,
):
    from agentlib.client import LLMClient

    response_payload = {
        "candidates": [{
            "content": {
                "parts": [
                    {
                        "thought": True,
                        "text": "new reasoning",
                        "thoughtSignature": "response-reasoning-signature",
                    },
                    {
                        "functionCall": {
                            "name": "calc",
                            "args": {"num": 42},
                        },
                        "thoughtSignature": "response-tool-signature",
                    },
                ],
            },
            "finishReason": "STOP",
        }],
    }
    requests = []

    class Response:
        status = 200

        def read(self):
            return json.dumps(response_payload).encode()

    class Connection:
        def __init__(self, *args, **kwargs):
            self.sock = self

        def connect(self):
            pass

        def setsockopt(self, *args):
            pass

        def request(self, method, path, body, headers):
            requests.append(json.loads(body))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(
        "agentlib.client.DeadlineHTTPSConnection",
        Connection,
    )
    client = LLMClient("google/gemini-3.6-flash")
    client.model_config["api_type"] = "gemini"
    client.model_config["api_key"] = "test-key"
    client._current_input_bytes = None

    response = client._call_gemini(
        [{
            "role": "assistant",
            "content": [
                {
                    "type": "reasoning",
                    "text": "prior reasoning",
                    "provider_metadata": {
                        "thought_signature": "request-reasoning-signature",
                    },
                },
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "calc",
                    "args": {"num": 1},
                    "provider_metadata": {
                        "thought_signature": "request-tool-signature",
                    },
                },
            ],
        }],
        None,
    )

    assert requests[0]["contents"][0]["parts"] == [
        {
            "text": "prior reasoning",
            "thought": True,
            "thoughtSignature": "request-reasoning-signature",
        },
        {
            "functionCall": {
                "name": "calc",
                "args": {"num": 1},
            },
            "thoughtSignature": "request-tool-signature",
        },
    ]
    assert response == {
        "role": "assistant",
        "content": [
            {
                "type": "reasoning",
                "text": "new reasoning",
                "provider_metadata": {
                    "thought_signature": "response-reasoning-signature",
                },
            },
            {
                "type": "tool_call",
                "id": "gemini_calc",
                "name": "calc",
                "args": {"num": 42},
                "provider_metadata": {
                    "thought_signature": "response-tool-signature",
                },
            },
        ],
        "provider_metadata": {"stop_reason": "STOP"},
    }


def test_gemini_history_without_thought_signature_uses_shim():
    from pydantic import BaseModel

    from agentlib.client import LLMClient

    class Calc(BaseModel):
        num: int

    client = LLMClient("google/gemini-3.6-flash", native=True)
    client.model_config["api_type"] = "gemini"
    historical_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "run"}],
        },
        {
            "role": "assistant",
            "content": [{
                "type": "tool_call",
                "id": "call_1",
                "name": "calc",
                "args": {"num": 1},
            }],
        },
    ]
    shim_calls = []

    def shim(messages, tools, **kwargs):
        shim_calls.append((messages, tools, kwargs))
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "shim"}],
        }

    client.tool_call_shim = shim

    response = client.call(historical_messages, {"calc": Calc})

    assert response["content"] == [{"type": "text", "text": "shim"}]
    assert shim_calls == [(historical_messages, {"calc": Calc}, {"retry": 3})]


def test_orphaned_tool_use_cleaned_up_before_transport():
    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling tool"},
                {
                    "type": "tool_call",
                    "id": "call_orphaned",
                    "name": "foo",
                    "args": {},
                },
                {
                    "type": "tool_call",
                    "id": "call_answered",
                    "name": "bar",
                    "args": {},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_answered",
            "name": "bar",
            "content": [{"type": "text", "text": "bar result"}],
        },
    ]

    cleaned = client._strip_orphaned_tool_use(messages)
    calls = [
        block
        for block in cleaned[0]["content"]
        if block["type"] == "tool_call"
    ]
    assert calls == [{
        "type": "tool_call",
        "id": "call_answered",
        "name": "bar",
        "args": {},
    }]


def test_responses_api_output_validation_and_reasoning():
    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    client.model_config["api_type"] = "responses"

    # Missing output field raises Exception
    with pytest.raises(Exception, match="output missing from response"):
        client._parse_responses_result({})

    # Non-list output raises Exception
    with pytest.raises(Exception, match="output missing from response"):
        client._parse_responses_result({"output": "not a list"})

    # Reasoning item with and without encrypted_content
    parsed = client._parse_responses_result({
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "text", "text": "unencrypted reasoning"}],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "text", "text": "encrypted reasoning"}],
                "encrypted_content": "enc-blob-123",
            },
            {
                "type": "output_text",
                "text": "final answer",
            },
        ],
        "status": "completed",
    })

    assert len(parsed["content"]) == 3
    assert parsed["content"][0] == {
        "type": "reasoning",
        "text": "unencrypted reasoning",
    }
    assert parsed["content"][1] == {
        "type": "reasoning",
        "text": "encrypted reasoning",
        "provider_metadata": {"encrypted_content": "enc-blob-123"},
    }
    assert parsed["content"][2] == {
        "type": "text",
        "text": "final answer",
    }

