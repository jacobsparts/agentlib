import pytest

from agentlib import BaseAgent


class ToolSchemaAgent(BaseAgent):
    system = "test"

    @BaseAgent.tool
    def strict_tool(self, notes: str = "Notes"):
        """Strict tool."""
        return notes

    @BaseAgent.tool
    def flexible_tool(self, name: str = "Name", **kwargs):
        """Flexible tool."""
        return kwargs


class ToolLoopAgent(BaseAgent):
    model = "sonnet"
    system = "test"

    @BaseAgent.tool
    def finish(self, value: int = "Result value"):
        """Finish with a value."""
        self.respond(value)


def test_tool_free_agent_uses_plain_llm_call():
    class ToolFreeAgent(BaseAgent):
        model = "test"
        system = "test"

    class FakeConversation:
        def __init__(self):
            self.tools = "not called"

        def llm(self, tools=None):
            self.tools = tools
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": "plain response"}],
            }

    agent = ToolFreeAgent()
    conversation = FakeConversation()
    agent._conversation = conversation

    assert agent.text() == "plain response"
    assert conversation.tools is None


def test_native_tool_call_runs_end_to_end_with_canonical_client(monkeypatch):
    agent = ToolLoopAgent()
    agent.llm_client.native = True
    captured = {}

    def fake_call(messages, tools):
        captured["messages"] = messages
        captured["tools"] = tools
        return {
            "role": "assistant",
            "content": [{
                "type": "tool_call",
                "id": "call_native",
                "name": "finish",
                "args": {"value": 7},
            }],
            "provider_metadata": {"stop_reason": "tool_calls"},
        }

    monkeypatch.setattr(agent.llm_client, "_call", fake_call)

    assert agent.run("finish", max_turns=1) == 7
    assert captured["messages"][-1]["role"] == "user"
    assert captured["messages"][-1]["content"] == [
        {"type": "text", "text": "finish"},
    ]
    assert captured["tools"][0]["function"]["name"] == "finish"
    assistant = next(
        message
        for message in reversed(agent.conversation.stored_messages())
        if message["role"] == "assistant"
    )
    assert assistant["content"] == [{
        "type": "tool_call",
        "id": "call_native",
        "name": "finish",
        "args": {"value": 7},
    }]


def test_shim_tool_call_runs_end_to_end_with_canonical_client(monkeypatch):
    agent = ToolLoopAgent()
    agent.llm_client.native = False
    captured = {}

    def fake_call(messages, tools=None):
        captured["messages"] = messages
        captured["tools"] = tools
        return {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": (
                    '{"function_calls": [{"name": "finish", '
                    '"arguments": {"value": 9}}]}'
                ),
            }],
        }

    monkeypatch.setattr(agent.llm_client, "_call", fake_call)

    assert agent.run("finish", max_turns=1) == 9
    assert captured["tools"] is None
    request = captured["messages"][-1]
    assert request["role"] == "user"
    assert request["content"][0] == {"type": "text", "text": "finish"}
    assert "Available functions" in request["content"][1]["text"]
    assistant = next(
        message
        for message in reversed(agent.conversation.stored_messages())
        if message["role"] == "assistant"
    )
    stored_call = assistant["content"][0]
    assert stored_call["type"] == "tool_call"
    assert stored_call["name"] == "finish"
    assert stored_call["args"] == {"value": 9}


def test_signature_tool_without_varargs_forbids_extra_arguments():
    spec = ToolSchemaAgent().toolspecs["strict_tool"]

    schema = spec.model_json_schema()
    assert schema["additionalProperties"] is False

    with pytest.raises(Exception):
        spec.model_validate({"notes": "ok", "reasoning": "extra"})


def test_signature_tool_with_kwargs_allows_extra_arguments():
    spec = ToolSchemaAgent().toolspecs["flexible_tool"]

    schema = spec.model_json_schema()
    assert "additionalProperties" not in schema

    validated = spec.model_validate({"name": "ok", "extra": "allowed"})
    assert validated.model_dump() == {"name": "ok"}
