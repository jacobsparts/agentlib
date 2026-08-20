import json

import pytest

from agentlib.client import ValidationError, _extract_tool_calls_json, _preprocess_tool_call_response


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
    client.concurrency_lock = type("NoopLock", (), {
        "__enter__": lambda self: None,
        "__exit__": lambda self, exc_type, exc, tb: False,
    })()

    ToolSpec = create_model("Think", __config__=ConfigDict(extra="forbid"), notes=(str, ...))
    ToolSpec.__doc__ = "Think tool."

    calls = []

    def fake_call(messages, tools):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "think",
                        "arguments": json.dumps({"notes": "ok", "reasoning": "extra"}),
                    }
                }],
            }
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "think",
                    "arguments": json.dumps({"notes": "ok"}),
                }
            }],
        }

    monkeypatch.setattr(client, "_call", fake_call)

    result = client.tool_call_native([{"role": "user", "content": "do it"}], {"think": ToolSpec}, retry=1)

    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"notes": "ok"}
    assert len(calls) == 2
    assert calls[0] == [{"role": "user", "content": "do it"}]
    assert calls[1][0] == {"role": "user", "content": "do it"}
    assert calls[1][1]["role"] == "assistant"
    assert '"reasoning": "extra"' in calls[1][1]["content"]
    assert calls[1][2]["role"] == "user"
    assert "ERROR: Invalid arguments for tool 'think'" in calls[1][2]["content"]
    assert "valid tool call only" in calls[1][2]["content"]


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
    if native:
        monkeypatch.setattr(client, "_call", lambda messages, tools: {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "submit",
                    "arguments": json.dumps(arguments),
                }
            }],
        })
        result = client.tool_call_native(
            [{"role": "user", "content": "submit"}],
            {"submit": tool},
            retry=0,
        )
    else:
        monkeypatch.setattr(client, "_call", lambda messages: {
            "role": "assistant",
            "content": json.dumps({
                "function_calls": [{
                    "name": "submit",
                    "arguments": arguments,
                }]
            }),
        })
        result = client.tool_call_shim(
            [{"role": "user", "content": "submit"}],
            {"submit": tool},
            retry=0,
        )

    normalized = json.loads(result["tool_calls"][0]["function"]["arguments"])
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
    client.concurrency_lock = type("NoopLock", (), {
        "__enter__": lambda self: None,
        "__exit__": lambda self, exc_type, exc, tb: False,
    })()

    ToolSpec = create_model("Lookup", value=(int, ...))
    ToolSpec.__doc__ = "Look up a value."

    monkeypatch.setattr(client, "_call", lambda messages: {
        "role": "assistant",
        "content": json.dumps({
            "function_calls": [
                {"name": "lookup", "arguments": {"value": 1}},
                {"name": "lookup", "arguments": {"value": 2}},
            ]
        }),
    })

    result = client.tool_call_shim(
        [{"role": "user", "content": "look these up"}],
        {"lookup": ToolSpec},
        retry=0,
    )

    ids = [call["id"] for call in result["tool_calls"]]
    assert all(call_id.startswith("call_") and len(call_id) == 37 for call_id in ids)
    assert len(set(ids)) == 2

    prepared_call = client.prepare_message(result)
    assert all(call_id not in prepared_call["content"] for call_id in ids)
    assert json.loads(prepared_call["content"]) == {
        "function_calls": [
            {"name": "lookup", "arguments": {"value": 1}},
            {"name": "lookup", "arguments": {"value": 2}},
        ]
    }

    prepared_result = client.prepare_message({
        "role": "tool",
        "name": "lookup",
        "content": "result",
        "tool_call_id": ids[0],
    })
    assert prepared_result == {"role": "user", "content": "lookup: result"}


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
    from agentlib.client import LLMClient, legacy_to_transport_messages

    client = LLMClient("sonnet")
    client.model_config = {
        "api_type": api_type,
        "model": "test-model",
        "context_window": 1_000_000,
    }
    captured = {}
    private_message = {
        "role": "user",
        "content": "visible",
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

    client._call([private_message])

    # Dispatch preserves private metadata so transports can consume it.
    assert captured["messages"] == legacy_to_transport_messages([private_message])
    assert captured["tools"] is None
    # Sizing ignores private metadata.
    assert captured["input_bytes"] == client._input_bytes(
        legacy_to_transport_messages([{"role": "user", "content": "visible"}]),
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
            "content": "visible",
            "_stdout": "hidden",
            "_event_seq": 1,
        }
    ]) == [{"role": "user", "content": "visible"}]



def test_canonical_transport_adapters_and_block_types():
    from agentlib.client import legacy_to_transport_messages, transport_to_legacy_message

    legacy_messages = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "input_file", "file_id": "file_abc", "media_type": "application/pdf"},
                {"type": "image_url", "image_url": "https://example.com/pic.png", "media_type": "image/png"},
            ],
            "_trace_id": "trace-123",
        },
        {
            "role": "assistant",
            "content": "working on it",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "run", "arguments": '{"cmd": "ls", "count": 2}'},
            }],
        },
        {
            "role": "tool",
            "name": "run",
            "tool_call_id": "call_1",
            "content": "file1\nfile2",
        },
    ]

    transport = legacy_to_transport_messages(legacy_messages)
    assert len(transport) == 4
    assert transport[0]["role"] == "system"
    assert transport[0]["content"] == [{"type": "text", "text": "system prompt"}]

    assert transport[1]["role"] == "user"
    assert transport[1]["_trace_id"] == "trace-123"
    assert transport[1]["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "attachment", "media_type": "application/pdf", "data_type": "provider_id", "data": "file_abc"},
        {"type": "attachment", "media_type": "image/png", "data_type": "url", "data": "https://example.com/pic.png"},
    ]

    assert transport[2]["role"] == "assistant"
    assert transport[2]["content"] == [
        {"type": "text", "text": "working on it"},
        {"type": "tool_call", "id": "call_1", "name": "run", "args": {"cmd": "ls", "count": 2}},
    ]
    # Check native decoded args
    assert transport[2]["content"][1]["args"] == {"cmd": "ls", "count": 2}

    # Test transport_to_legacy_message lossy projection
    canonical_response = {
        "role": "assistant",
        "content": [
            {"type": "commentary", "text": "thought step\nline 2"},
            {"type": "reasoning", "text": "hidden reasoning", "provider_metadata": {"encrypted": "abc"}},
            {"type": "tool_call", "id": "call_2", "name": "read", "args": {"file": "a.txt"}},
            {"type": "text", "text": "done"},
        ],
        "provider_metadata": {"stop_reason": "tool_calls"},
    }
    legacy_resp = transport_to_legacy_message(canonical_response)
    assert legacy_resp["role"] == "assistant"
    assert legacy_resp["_stop_reason"] == "tool_calls"
    assert legacy_resp["content"] == "# thought step\n# line 2\ndone"
    assert legacy_resp["tool_calls"] == [{
        "id": "call_2",
        "type": "function",
        "function": {"name": "read", "arguments": '{"file": "a.txt"}'},
    }]

    # Returned attachments must raise NotImplementedError
    with pytest.raises(NotImplementedError, match="Legacy Conversation cannot represent returned attachments"):
        transport_to_legacy_message({
            "role": "assistant",
            "content": [{"type": "attachment", "media_type": "image/png", "data_type": "bytes", "data": b"png"}],
        })


def test_unknown_types_raise_not_implemented():
    from agentlib.client import LLMClient, legacy_to_transport_messages, transport_to_legacy_message

    with pytest.raises(NotImplementedError, match="Unknown legacy content type"):
        legacy_to_transport_messages([{
            "role": "user",
            "content": [{"type": "unsupported_legacy_type"}],
        }])

    with pytest.raises(NotImplementedError, match="Unknown transport content type"):
        transport_to_legacy_message({
            "role": "assistant",
            "content": [{"type": "unsupported_transport_type"}],
        })

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


def test_attachment_ingress_from_images_and_audio(tmp_path):
    from agentlib.client import BadRequestError, LLMClient, legacy_to_transport_messages

    png_bytes = b"\x89PNG\r\n\x1a\nfake_png_data"
    jpeg_bytes = b"\xff\xd8\xfffake_jpeg_data"
    wav_bytes = b"RIFFfake_wav_data"

    messages = legacy_to_transport_messages([
        {
            "role": "user",
            "content": "analyze files",
            "images": [png_bytes, jpeg_bytes],
            "audio": [wav_bytes],
        }
    ])

    assert len(messages[0]["content"]) == 4
    assert messages[0]["content"][0] == {"type": "text", "text": "analyze files"}
    assert messages[0]["content"][1] == {"type": "attachment", "media_type": "image/png", "data_type": "bytes", "data": png_bytes}
    assert messages[0]["content"][2] == {"type": "attachment", "media_type": "image/jpeg", "data_type": "bytes", "data": jpeg_bytes}
    assert messages[0]["content"][3] == {"type": "attachment", "media_type": "audio/wav", "data_type": "bytes", "data": wav_bytes}

    # Invalid image format
    with pytest.raises(BadRequestError, match="Unsupported image format"):
        legacy_to_transport_messages([{"role": "user", "content": "x", "images": [b"bad_image_data"]}])

    # Invalid audio format
    with pytest.raises(BadRequestError, match="Unsupported audio format"):
        legacy_to_transport_messages([{"role": "user", "content": "x", "audio": [b"bad_audio_data"]}])

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

    gemini_client = LLMClient("gemini-3.6-flash")
    gemini_client.model_config["api_type"] = "gemini"
    # Gemini accepts audio
    gemini_client.validate_media_type("audio/wav")

    # Filepath data projection
    fpath = tmp_path / "test.png"
    fpath.write_bytes(png_bytes)
    fattachment = {"type": "attachment", "media_type": "image/png", "data_type": "filepath", "data": str(fpath)}
    projected = gemini_client._gemini_attachment(fattachment)
    assert projected["inlineData"]["mimeType"] == "image/png"


def test_gemini_thought_signatures_and_shim_fallback():
    from pydantic import BaseModel
    from agentlib.client import LLMClient

    class Calc(BaseModel):
        num: int

    client = LLMClient("gemini-3.6-flash", native=True)
    client.model_config["api_type"] = "gemini"

    # Response with thoughtSignature decodes to canonical provider_metadata and legacy tool_calls
    gemini_response = {
        "candidates": [{
            "content": {
                "parts": [
                    {"thought": True, "text": "thinking step", "thoughtSignature": "sig-123"},
                    {
                        "functionCall": {"name": "calc", "args": {"num": 42}},
                        "thoughtSignature": "sig-456",
                    },
                ]
            },
            "finishReason": "STOP",
        }]
    }

    # Simulate parse_gemini_response via _call_gemini response parser
    # We can inspect _call_gemini parts parsing
    parts = gemini_response["candidates"][0]["content"]["parts"]
    blocks = []
    for part in parts:
        if "text" in part and part.get("thought"):
            item = {"type": "reasoning", "text": part["text"]}
            if "thoughtSignature" in part:
                item["provider_metadata"] = {"thought_signature": part["thoughtSignature"]}
            blocks.append(item)
        elif "functionCall" in part:
            fc = part["functionCall"]
            call = {"type": "tool_call", "id": f"gemini_{fc['name']}", "name": fc["name"], "args": fc["args"]}
            if "thoughtSignature" in part:
                call["provider_metadata"] = {"thought_signature": part["thoughtSignature"]}
            blocks.append(call)

    from agentlib.client import transport_to_legacy_message
    legacy = transport_to_legacy_message({"role": "assistant", "content": blocks})
    assert legacy["tool_calls"][0]["thoughtSignature"] == "sig-456"

    # History lacking thoughtSignature triggers shim fallback in call()
    historical_messages = [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "calc", "arguments": '{"num": 1}'},
            }],
        },
    ]
    shim_called = []
    client.tool_call_shim = lambda msgs, tools, **kw: shim_called.append(True) or {"role": "assistant", "content": "shim"}
    client.call(historical_messages, {"calc": Calc})
    assert shim_called == [True]


def test_orphaned_tool_use_cleaned_up_before_transport():
    from agentlib.client import LLMClient

    client = LLMClient("sonnet")
    messages = [
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {"id": "call_orphaned", "function": {"name": "foo", "arguments": "{}"}},
                {"id": "call_answered", "function": {"name": "bar", "arguments": "{}"}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_answered",
            "name": "bar",
            "content": "bar result",
        },
    ]

    cleaned = client._strip_orphaned_tool_use(messages)
    assert len(cleaned[0]["tool_calls"]) == 1
    assert cleaned[0]["tool_calls"][0]["id"] == "call_answered"


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

