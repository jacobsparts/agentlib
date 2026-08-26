from agentlib.conversation import Convo


class DummyClient:
    def __init__(self):
        self.calls = []

    def call(self, messages, tools=None):
        self.calls.append((messages, tools))
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
        }




def test_ephemeral_injected_at_top_of_last_user_message():
    client = DummyClient()
    conv = Convo(client, "system")
    conv.usermsg("first")
    conv.append_message({
        "role": "tool",
        "content": [{"type": "text", "text": "tool output"}],
    })
    conv.usermsg("last")
    conv.ephemeral = "temporary context"

    messages = conv.projected_messages()

    assert messages[-1]["content"] == [
        {"type": "text", "text": "temporary context"},
        {"type": "text", "text": "last"},
    ]
    assert conv.stored_messages()[-1]["content"] == [
        {"type": "text", "text": "last"},
    ]
    assert conv.ephemeral == "temporary context"

    conv.ephemeral = ""
    assert conv.ephemeral == ""


def test_ephemeral_applied_after_attachments():
    client = DummyClient()
    conv = Convo(client, "system")
    conv.usermsg([
        {"type": "text", "text": "question"},
        {
            "type": "attachment",
            "media_type": "text/x-python",
            "data_type": "bytes",
            "data": b"file contents",
        },
    ])
    conv.ephemeral = "temporary context"

    messages = conv.projected_messages()

    assert messages[-1]["content"] == [
        {"type": "text", "text": "temporary context"},
        {"type": "text", "text": "question"},
        {
            "type": "attachment",
            "media_type": "text/x-python",
            "data_type": "bytes",
            "data": b"file contents",
        },
    ]
    assert conv.stored_messages()[-1]["content"][0] == {
        "type": "text",
        "text": "question",
    }


def test_ephemeral_not_added_to_history_or_cleared_by_call():
    client = DummyClient()
    conv = Convo(client, "system")
    conv.usermsg("question")
    conv.ephemeral = "temporary context"

    conv.call()

    assert client.calls[0][0][-1]["content"] == [
        {"type": "text", "text": "temporary context"},
        {"type": "text", "text": "question"},
    ]
    assert conv.stored_messages()[1]["content"] == [
        {"type": "text", "text": "question"},
    ]
    assert conv.ephemeral == "temporary context"


class FailingClient:
    def call(self, messages, tools=None):
        raise RuntimeError("provider failed")


def test_ephemeral_preserved_when_call_raises():
    conv = Convo(FailingClient(), "system")
    conv.usermsg("question")
    conv.ephemeral = "temporary context"

    try:
        conv.call()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    assert conv.ephemeral == "temporary context"


def test_ephemeral_supports_string_append():
    conv = Convo(DummyClient(), "system")
    conv.usermsg("question")

    conv.ephemeral = "first"
    conv.ephemeral += "\n\nsecond"

    assert conv.ephemeral == "first\n\nsecond"
    assert conv.projected_messages()[-1]["content"] == [
        {"type": "text", "text": "first\n\nsecond"},
        {"type": "text", "text": "question"},
    ]


def test_base_agent_ephemeral_property_passthrough():
    from agentlib import BaseAgent

    class TestAgent(BaseAgent):
        model = "test-model"
        system = "system"

    agent = TestAgent()
    agent._conversation = Convo(DummyClient(), "system")
    agent.usermsg("question")

    agent.ephemeral = "first"
    agent.ephemeral += "\n\nsecond"

    assert agent.conversation.ephemeral == "first\n\nsecond"
    assert agent.conversation.projected_messages()[-1]["content"] == [
        {"type": "text", "text": "first\n\nsecond"},
        {"type": "text", "text": "question"},
    ]


def test_base_agent_propagates_emulated_tool_call_id_to_result():
    from agentlib import BaseAgent

    class TestAgent(BaseAgent):
        system = "system"

        def __init__(self):
            self.recorded_tool_messages = []

        def llm(self):
            return {
                "role": "assistant",
                "content": [{
                    "type": "tool_call",
                    "id": "call_0123456789abcdef0123456789abcdef",
                    "name": "finish",
                    "args": {},
                }],
            }

        def toolcall(self, toolname, function_args):
            self.respond("done")

        def toolmsg(self, content, **kwargs):
            self.recorded_tool_messages.append((content, kwargs))

    agent = TestAgent()

    assert agent.run_loop(max_turns=1) == "done"
    assert agent.recorded_tool_messages == [(
        "done",
        {
            "name": "finish",
            "tool_call_id": "call_0123456789abcdef0123456789abcdef",
        },
    )]


def test_base_agent_factory_returns_convo():
    from agentlib import BaseAgent

    class TestAgent(BaseAgent):
        system = "system"

    client = DummyClient()
    client.conversation = lambda system: Convo(client, system)
    agent = TestAgent()
    agent._llm_client = client

    assert isinstance(agent.conversation, Convo)


def test_convo_projects_attachments_per_text_block_without_mutating_storage():
    conv = Convo(DummyClient(), "system")
    conv.usermsg(
        [
            {"type": "text", "text": "[Attachment: config]"},
            {"type": "text", "text": "question"},
        ],
        _attachments={"config": "rendered config"},
    )

    projected = conv.projected_messages()[-1]

    assert projected["role"] == "user"
    assert projected["content"] == [
        {"type": "text", "text": "rendered config"},
        {"type": "text", "text": "question"},
    ]
    assert conv.stored_messages()[-1]["content"][0]["text"] == "[Attachment: config]"


def test_convo_toolmsg_stores_canonical_blocks():
    conv = Convo(DummyClient(), "system")

    conv.toolmsg("ok", name="finish", tool_call_id="call_1")

    assert conv.stored_messages()[-1] == {
        "role": "tool",
        "content": [{"type": "text", "text": "ok"}],
        "name": "finish",
        "tool_call_id": "call_1",
    }

def test_convo_llm_appends_and_returns_response():
    client = DummyClient()
    conv = Convo(client, "system")
    conv.usermsg("hi")

    resp = conv.llm()

    assert resp == {
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
    }
    assert conv.stored_messages()[-1] == resp

def test_attachment_mixin_preserves_canonical_blocks():
    from agentlib import AttachmentMixin, BaseAgent

    class TestAgent(AttachmentMixin, BaseAgent):
        system = "system"

    agent = TestAgent()
    agent._conversation = Convo(DummyClient(), "system")
    attachment_block = {
        "type": "attachment",
        "media_type": "image/png",
        "data_type": "bytes",
        "data": b"png",
    }
    agent.attach("config", "rendered config")

    message = agent.usermsg([
        {"type": "text", "text": "question"},
        attachment_block,
    ])

    assert message["content"] == [
        {"type": "text", "text": "[Attachment: config]"},
        {"type": "text", "text": "question"},
        attachment_block,
    ]
    assert agent.conversation.projected_messages()[-1]["content"] == [
        {
            "type": "text",
            "text": agent._render_attachment("config", "rendered config"),
        },
        {"type": "text", "text": "question"},
        attachment_block,
    ]


def test_base_agent_switch_model_replaces_client_and_conversation_client(monkeypatch):
    from agentlib import BaseAgent

    class DummyLLMClient:
        def __init__(self, model_name, native=None):
            self.model_name = model_name
            self.native = native
            self.model_config = {"provider": "dummy", "model": model_name}

    monkeypatch.setattr("agentlib.agent.LLMClient", DummyLLMClient)

    class TestAgent(BaseAgent):
        model = "old-model"
        system = "system"

    agent = TestAgent()
    old_client = DummyLLMClient("old-model")
    agent._llm_client = old_client
    agent._conversation = Convo(old_client, "system")

    config = agent.switch_model("new-model")

    assert config == {"provider": "dummy", "model": "new-model"}
    assert agent.model == "new-model"
    assert agent.llm_client.model_name == "new-model"
    assert agent.conversation.llm_client is agent.llm_client
    assert agent.conversation.llm_client is not old_client
