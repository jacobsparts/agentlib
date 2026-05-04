from agentlib.conversation import Conversation


class DummyClient:
    def __init__(self):
        self.calls = []

    def call(self, messages, tools=None):
        self.calls.append((messages, tools))
        return {"role": "assistant", "content": "ok", "tool_calls": []}


def test_ephemeral_context_injected_at_top_of_last_user_message_and_cleared():
    client = DummyClient()
    conv = Conversation(client, "system")
    conv.usermsg("first")
    conv.toolmsg("tool output")
    conv.usermsg("last")
    conv.ephemeral("temporary context")

    messages = conv._messages()

    assert messages[-1]["content"] == "temporary context\n\nlast"
    assert conv.messages[-1]["content"] == "last"
    assert conv._ephemeral_context == ["temporary context"]

    conv.clear_ephemeral()
    assert conv._ephemeral_context == []


def test_ephemeral_context_applied_after_attachments():
    client = DummyClient()
    conv = Conversation(client, "system")
    conv.usermsg(
        "[Attachment: file.py]\n\nquestion",
        _attachments={"file.py": "file contents"},
    )
    conv.ephemeral("temporary context")

    messages = conv._messages()

    assert messages[-1]["content"] == "temporary context\n\nfile contents\n\nquestion"
    assert conv.messages[-1]["content"] == "[Attachment: file.py]\n\nquestion"


def test_ephemeral_context_not_added_to_history_during_llm_call():
    client = DummyClient()
    conv = Conversation(client, "system")
    conv.usermsg("question")
    conv.ephemeral("temporary context")

    conv.llm()

    assert client.calls[0][0][-1]["content"] == "temporary context\n\nquestion"
    assert conv.messages[1]["content"] == "question"
    assert conv._ephemeral_context == []


class FailingClient:
    def call(self, messages, tools=None):
        raise RuntimeError("provider failed")


def test_ephemeral_context_preserved_when_llm_call_raises():
    conv = Conversation(FailingClient(), "system")
    conv.usermsg("question")
    conv.ephemeral("temporary context")

    try:
        conv.llm()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    assert conv._ephemeral_context == ["temporary context"]


def test_base_agent_ephemeral_passthrough():
    from agentlib import BaseAgent

    class TestAgent(BaseAgent):
        model = "test-model"
        system = "system"

    agent = TestAgent()
    agent._conversation = Conversation(DummyClient(), "system")
    agent.usermsg("question")

    agent.ephemeral("temporary context")

    assert agent.conversation._messages()[-1]["content"] == "temporary context\n\nquestion"
