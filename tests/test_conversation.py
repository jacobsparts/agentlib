from agentlib.conversation import Conversation


class DummyClient:
    def __init__(self):
        self.calls = []

    def call(self, messages, tools=None):
        self.calls.append((messages, tools))
        return {"role": "assistant", "content": "ok", "tool_calls": []}


def test_preview_refs_expand_nested_refs_regardless_of_order():
    client = DummyClient()
    conv = Conversation(client, "system")
    conv.expanded_preview_refs = {
        "session://preview/inner": {"numbered": False},
        "session://preview/outer": {"numbered": False},
    }
    blobs = {
        "session://preview/outer": (
            "outer before\n"
            "[PreviewRef: session://preview/inner]\ninner summary\n[/PreviewRef]\n"
            "outer after"
        ),
        "session://preview/inner": "INNER FULL",
    }
    conv.preview_loader = blobs.get
    conv.usermsg("[PreviewRef: session://preview/outer]\nouter summary\n[/PreviewRef]")

    content = conv._messages()[-1]["content"]

    assert "[ExpandedPreviewRef: session://preview/outer]" in content
    assert "[ExpandedPreviewRef: session://preview/inner]" in content
    assert "INNER FULL" in content
    assert conv.rendered_preview_refs == ["session://preview/outer", "session://preview/inner"]


def test_preview_refs_do_not_expand_same_uri_twice_in_one_render():
    client = DummyClient()
    conv = Conversation(client, "system")
    conv.expanded_preview_refs = {"session://preview/self": {"numbered": False}}
    conv.preview_loader = lambda uri: (
        "self before\n"
        "[PreviewRef: session://preview/self]\nself summary\n[/PreviewRef]\n"
        "self after"
    )
    conv.usermsg("[PreviewRef: session://preview/self]\nself summary\n[/PreviewRef]")

    content = conv._messages()[-1]["content"]

    assert content.count("[ExpandedPreviewRef: session://preview/self]") == 1
    assert content.count("[PreviewRef: session://preview/self]") == 1
    assert conv.rendered_preview_refs == ["session://preview/self"]


def test_ephemeral_injected_at_top_of_last_user_message():
    client = DummyClient()
    conv = Conversation(client, "system")
    conv.usermsg("first")
    conv.toolmsg("tool output")
    conv.usermsg("last")
    conv.ephemeral = "temporary context"

    messages = conv._messages()

    assert messages[-1]["content"] == "temporary context\n\nlast"
    assert conv.messages[-1]["content"] == "last"
    assert conv.ephemeral == "temporary context"

    conv.ephemeral = ""
    assert conv.ephemeral == ""


def test_ephemeral_applied_after_attachments():
    client = DummyClient()
    conv = Conversation(client, "system")
    conv.usermsg(
        "[Attachment: file.py]\n\nquestion",
        _attachments={"file.py": "file contents"},
    )
    conv.ephemeral = "temporary context"

    messages = conv._messages()

    assert messages[-1]["content"] == "temporary context\n\nfile contents\n\nquestion"
    assert conv.messages[-1]["content"] == "[Attachment: file.py]\n\nquestion"


def test_ephemeral_not_added_to_history_or_cleared_by_llm_call():
    client = DummyClient()
    conv = Conversation(client, "system")
    conv.usermsg("question")
    conv.ephemeral = "temporary context"

    conv.llm()

    assert client.calls[0][0][-1]["content"] == "temporary context\n\nquestion"
    assert conv.messages[1]["content"] == "question"
    assert conv.ephemeral == "temporary context"


class FailingClient:
    def call(self, messages, tools=None):
        raise RuntimeError("provider failed")


def test_ephemeral_preserved_when_llm_call_raises():
    conv = Conversation(FailingClient(), "system")
    conv.usermsg("question")
    conv.ephemeral = "temporary context"

    try:
        conv.llm()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    assert conv.ephemeral == "temporary context"


def test_ephemeral_supports_string_append():
    conv = Conversation(DummyClient(), "system")
    conv.usermsg("question")

    conv.ephemeral = "first"
    conv.ephemeral += "\n\nsecond"

    assert conv.ephemeral == "first\n\nsecond"
    assert conv._messages()[-1]["content"] == "first\n\nsecond\n\nquestion"


def test_base_agent_ephemeral_property_passthrough():
    from agentlib import BaseAgent

    class TestAgent(BaseAgent):
        model = "test-model"
        system = "system"

    agent = TestAgent()
    agent._conversation = Conversation(DummyClient(), "system")
    agent.usermsg("question")

    agent.ephemeral = "first"
    agent.ephemeral += "\n\nsecond"

    assert agent.conversation.ephemeral == "first\n\nsecond"
    assert agent.conversation._messages()[-1]["content"] == "first\n\nsecond\n\nquestion"
