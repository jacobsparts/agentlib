import pytest

from agentlib import REPLAgent
from agentlib.conversation import Convo


class RecordingClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def call(self, messages, tools=None):
        self.calls.append(messages)
        return next(self.responses)


class CanonicalTestAgent(REPLAgent):
    system = "test system"

    def _ensure_setup(self):
        pass

    def _get_tool_repl(self):
        return object()


def test_repl_agent_uses_canonical_blocks_and_preserves_assistant_message():
    response = {
        "role": "assistant",
        "content": [
            {"type": "reasoning", "text": "private"},
            {"type": "commentary", "text": "status"},
            {"type": "text", "text": "emit('done', release=True)"},
        ],
    }
    client = RecordingClient([response])
    agent = CanonicalTestAgent()
    agent._llm_client = client
    agent._conversation = Convo(client, "test system")
    executed = []

    def execute(repl, code):
        executed.append(code)
        agent.complete = True
        agent._final_result = "done"
        return "", False, [], code

    agent._execute_with_tool_handling = execute

    assert agent.run("go", max_turns=1) == "done"
    assert executed == ["# status\nemit('done', release=True)"]
    assert client.calls[0][1]["content"] == [
        {"type": "text", "text": "go"},
    ]
    assert agent.conversation.stored_messages()[2] is response
    assert response["content"][0] == {"type": "reasoning", "text": "private"}


def test_repl_syntax_retry_uses_canonical_temporary_messages():
    responses = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "not valid python !"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "emit('done', release=True)"}],
        },
    ]
    client = RecordingClient(responses)
    agent = CanonicalTestAgent()
    agent._llm_client = client
    agent._conversation = Convo(client, "test system")
    attempts = []

    def execute(repl, code):
        attempts.append(code)
        if len(attempts) == 1:
            return "SyntaxError: invalid syntax\n", True, [], code
        agent.complete = True
        agent._final_result = "done"
        return "", False, [], code

    agent._execute_with_tool_handling = execute

    assert agent.run("go", max_turns=1) == "done"
    retry_messages = client.calls[1]
    assert retry_messages[-2] is responses[0]
    assert retry_messages[-1]["role"] == "user"
    assert retry_messages[-1]["content"][0]["type"] == "text"
    assert "Return only valid Python code" in retry_messages[-1]["content"][0]["text"]
    assert responses[0] not in agent.conversation.stored_messages()


def test_repl_projects_memory_attachments_as_canonical_text():
    agent = CanonicalTestAgent()
    client = RecordingClient([])
    agent._llm_client = client
    agent._conversation = Convo(client, "test system")
    agent.conversation.usermsg(
        "[Attachment: config]\n\ninspect this",
        _attachments={
            "config": (
                "-------- BEGIN config --------\nvalue=1\n"
                "-------- END config ----------"
            ),
        },
    )

    projected = agent._projected_messages()

    assert projected[-1]["content"] == [{
        "type": "text",
        "text": (
            "-------- BEGIN config --------\nvalue=1\n"
            "-------- END config ----------\n\ninspect this"
        ),
    }]
    assert agent.conversation.stored_messages()[-1]["content"] == [{
        "type": "text",
        "text": "[Attachment: config]\n\ninspect this",
    }]


def test_repl_usermsg_does_not_mutate_prior_repl_output():
    agent = CanonicalTestAgent()
    client = RecordingClient([])
    agent._llm_client = client
    agent._conversation = Convo(client, "test system")
    repl_output = agent.usermsg(
        ">>> print('done')\ndone\n",
        _stdout=">>> print('done')\ndone\n",
    )
    original = {
        **repl_output,
        "content": [dict(block) for block in repl_output["content"]],
        "_render_segments": [
            dict(segment) for segment in repl_output["_render_segments"]
        ],
    }

    human_input = agent.usermsg("continue", _user_content="continue")

    assert repl_output == original
    assert human_input is not repl_output
    assert agent.conversation.stored_messages()[-2:] == [
        repl_output,
        human_input,
    ]
    assert human_input["content"] == [{"type": "text", "text": "continue"}]


def test_repl_rejects_non_executable_assistant_blocks():
    with pytest.raises(NotImplementedError, match="tool_call"):
        REPLAgent._assistant_code({
            "role": "assistant",
            "content": [{
                "type": "tool_call",
                "id": "call-1",
                "name": "unexpected",
                "args": {"native": True},
            }],
        })

    with pytest.raises(NotImplementedError, match="Unknown assistant"):
        REPLAgent._assistant_code({
            "role": "assistant",
            "content": [{"type": "future_block"}],
        })
