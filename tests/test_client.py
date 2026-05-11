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

    assert client.usage_tracker.input_tokens_per_byte[client.model_name] == pytest.approx(0.25)
