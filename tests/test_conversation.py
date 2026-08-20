from agentlib.conversation import Conversation


class DummyClient:
    def __init__(self):
        self.calls = []

    def call(self, messages, tools=None):
        self.calls.append((messages, tools))
        return {"role": "assistant", "content": "ok", "tool_calls": []}




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

    assert client.calls[0][0][-1]["content"] == [{"type": "text", "text": "temporary context\n\nquestion"}]
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


def test_base_agent_propagates_emulated_tool_call_id_to_result():
    from agentlib import BaseAgent

    class TestAgent(BaseAgent):
        system = "system"

        def __init__(self):
            self.recorded_tool_messages = []

        def llm(self):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_0123456789abcdef0123456789abcdef",
                    "function": {"name": "finish", "arguments": "{}"},
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
    agent._conversation = Conversation(old_client, "system")

    config = agent.switch_model("new-model")

    assert config == {"provider": "dummy", "model": "new-model"}
    assert agent.model == "new-model"
    assert agent.llm_client.model_name == "new-model"
    assert agent.conversation.llm_client is agent.llm_client
    assert agent.conversation.llm_client is not old_client


def test_convo_native_and_conversation_wrapper():
    from agentlib.conversation import Convo

    client = DummyClient()
    convo = Convo(client, "canonical system")
    convo.usermsg("existing")
    conv = Conversation(client, "ignored", convo=convo)

    assert conv.stored_messages() == [
        {"role": "system", "content": "canonical system"},
        {"role": "user", "content": "existing"},
    ]

    user = conv.append_message({"role": "user", "content": "legacy user", "_synthetic": True})
    conv.update_message(user, _event_seq=9)

    assert convo.stored_messages()[-1] == {
        "role": "user",
        "content": [{"type": "text", "text": "legacy user"}],
        "_synthetic": True,
        "_event_seq": 9,
    }

    # Direct Convo modification is immediately visible in Conversation without sync
    canonical_user = convo.usermsg("direct to convo")
    assert conv.stored_messages()[-1] == {"role": "user", "content": "direct to convo"}

    convo.pop_message()
    assert conv.stored_messages()[-1] == {"role": "user", "content": "legacy user", "_synthetic": True, "_event_seq": 9}
