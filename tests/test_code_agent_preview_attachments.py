import re

from agentlib.agents.code_agent import CodeAgent
from agentlib.conversation import Conversation
from agentlib.session_store import SessionStore


class DummyClient:
    on_retry = None

    def call(self, messages, tools=None):
        return {"role": "assistant", "content": "ok"}


def make_agent():
    agent = CodeAgent()
    agent._conversation = Conversation(DummyClient(), "system")
    agent._session_id = "session"
    agent._session_store = None
    agent._next_event_seq = 1
    agent._suspend_persistence = True
    agent._explicit_attachment_refs = {}
    agent._pending_explicit_attachment_refs = {}
    agent._pending_session_events = []
    agent._display_capture = []
    agent._pending_unviewed_files = set()
    agent._auto_context_attachment_names = set()
    agent._pending_attachments = {}
    return agent


def test_pin_marks_previous_assistant_turn():
    agent = make_agent()
    assistant = {"role": "assistant", "content": "print('important')"}
    agent.conversation.messages.append({"role": "user", "content": "Task", "_user_content": "Task"})
    agent.conversation.messages.append(assistant)

    result = agent.pin()

    assert result == "Pinned previous turn for coalescing."
    assert assistant["_pinned_coalesce"] == {"label": "Pinned previous turn"}


def test_pin_no_previous_turn_is_noop():
    agent = make_agent()

    result = agent.pin()

    assert result == "No previous turn to pin."


def test_pin_can_target_previous_interaction_release_turn():
    agent = make_agent()
    old_assistant = {"role": "assistant", "content": "print('old')"}
    release_assistant = {"role": "assistant", "content": "emit('old done', release=True)"}
    agent.conversation.messages.extend([
        {"role": "user", "content": "Old task", "_user_content": "Old task"},
        old_assistant,
        {"role": "user", "content": ">>> print('old')\nold\n"},
        release_assistant,
        {"role": "user", "content": ">>> emit('old done', release=True)\nold done\nNew task", "_user_content": "New task"},
    ])

    result = agent.pin()

    assert result == "Pinned previous turn for coalescing."
    assert "_pinned_coalesce" not in old_assistant
    assert release_assistant["_pinned_coalesce"] == {"label": "Pinned previous turn"}


def test_pin_persists_metadata_event_for_existing_persisted_message(tmp_path):
    from agentlib.session_replay import replay_session_into_agent

    agent = make_persistent_agent(tmp_path)
    agent.conversation.messages.extend([
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('important')"},
    ])
    assistant = agent.conversation.messages[-1]
    agent._persist_message(agent.conversation.messages[-2])
    agent._persist_message(assistant)

    result = agent.pin()

    assert result == "Pinned previous turn for coalescing."
    events = agent._session_store.get_events(agent._session_id)
    assert events[-1]["event_type"] == "message_pinned"
    assert events[-1]["payload"]["message_event_seq"] == assistant["_event_seq"]

    class ReplayAgent:
        def __init__(self):
            self.conversation = Conversation(DummyClient(), "system")
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            pass

    replayed = ReplayAgent()
    replay_session_into_agent(replayed, agent._session_id, agent._session_store)
    replayed_assistant = next(
        msg for msg in replayed.conversation.messages
        if msg.get("role") == "assistant" and msg.get("content") == "print('important')"
    )
    assert replayed_assistant["_pinned_coalesce"] == {"label": "Pinned previous turn"}


def test_preview_uri_attachments_are_listed_by_default():
    agent = make_agent()
    agent.conversation.usermsg(
        "[Attachment: session://preview/abc]",
        _attachments={"session://preview/abc": "    1→preview content"},
        _attachment_refs={"session://preview/abc": "session://preview/abc"},
    )

    assert "session://preview/abc" in agent.list_attachments()
    assert "session://preview/abc" not in agent.list_attachments(include_session_blobs=False)


def test_preview_uri_attachments_appear_in_context_notice():
    agent = make_agent()
    notice = agent._file_context_ephemeral(["session://preview/abc"])

    assert "Context currently expanded:" in notice
    assert "session://preview/abc" in notice
    assert "unview(path_or_uri)" in notice


def test_context_pressure_notice_when_near_limit():
    agent = make_agent()
    agent.llm_client.model_config["context_window"] = 100
    agent.llm_client.usage_tracker.input_tokens_per_byte = {agent.llm_client.model_name: 1.0}
    agent.conversation.usermsg("x" * 200)

    notice = agent._file_context_ephemeral([])

    assert "Context window is near capacity." in notice
    assert "unview(path_or_uri)" in notice


def test_context_pressure_notice_combines_with_expanded_context():
    agent = make_agent()
    agent.llm_client.model_config["context_window"] = 100
    agent.llm_client.usage_tracker.input_tokens_per_byte = {agent.llm_client.model_name: 1.0}
    agent.conversation.usermsg("x" * 200)

    notice = agent._file_context_ephemeral(["session://preview/abc"])

    assert "Context currently expanded:" in notice
    assert "session://preview/abc" in notice
    assert "Context window is near capacity." in notice


def test_system_prompt_mentions_pin_and_context_pressure():
    agent = make_agent()
    prompt = agent._build_system_prompt()

    assert "REPL output may become a preview after three user interactions" in prompt
    assert "If the most\nrecent completed turn should remain in context long-term, call pin()" in prompt
    assert "pin() is\nonly for the previous turn" in prompt
    assert "cannot pin the current turn or other historical\nturns" in prompt
    assert "Context window is near capacity" in prompt



def test_current_context_names_include_preview_uris():
    agent = make_agent()
    agent.conversation.usermsg(
        "[Attachment: session://preview/abc]",
        _attachments={"session://preview/abc": "    1→preview content"},
        _attachment_refs={"session://preview/abc": "session://preview/abc"},
    )

    assert "session://preview/abc" in agent._current_file_context_names()



def test_current_context_names_only_include_preview_uris_that_can_render():
    agent = make_agent()
    agent._session_id = "session"
    agent._expanded_preview_refs = {
        "session://preview/outer": {"numbered": False},
        "session://preview/inner": {"numbered": False},
    }
    blobs = {
        "session://preview/outer": (
            "outer before\n"
            "[PreviewRef: session://preview/inner]\ninner summary\n[/PreviewRef]\n"
            "outer after"
        ),
        "session://preview/inner": "INNER FULL",
    }
    agent._preview_blob_content = blobs.get

    assert "session://preview/inner" not in agent._current_file_context_names()

    agent.conversation.usermsg("[PreviewRef: session://preview/outer]\nouter summary\n[/PreviewRef]")

    assert agent._current_file_context_names() == [
        "session://preview/outer",
        "session://preview/inner",
    ]


def test_expanded_preview_context_hides_nested_preview_when_parent_not_in_context():
    agent = make_agent()
    agent._session_id = "session"
    agent._expanded_preview_refs = {
        "session://preview/outer": {"numbered": False},
        "session://preview/inner": {"numbered": False},
    }
    blobs = {
        "session://preview/outer": (
            "outer before\n"
            "[PreviewRef: session://preview/inner]\ninner summary\n[/PreviewRef]\n"
            "outer after"
        ),
        "session://preview/inner": "INNER FULL",
    }
    agent._preview_blob_content = blobs.get

    assert agent._expanded_preview_context() == {}

    agent.conversation.usermsg("[PreviewRef: session://preview/outer]\nouter summary\n[/PreviewRef]")

    assert set(agent._expanded_preview_context()) == {
        "session://preview/outer",
        "session://preview/inner",
    }

def test_unview_collapses_preview_uri():
    agent = make_agent()
    agent._expanded_preview_refs = {"session://preview/abc": {"numbered": False}}

    result = agent.unview("session://preview/abc")

    assert result == "Collapsed preview: session://preview/abc"
    assert "session://preview/abc" not in agent._expanded_preview_refs
    assert "session://preview/abc" in agent._pending_unviewed_files


def make_persistent_agent(tmp_path):
    agent = CodeAgent()
    agent._ensure_setup()
    agent._session_store = SessionStore(str(tmp_path / "sessions.db"))
    agent._session_id = None
    agent._next_event_seq = 1
    return agent


def test_view_full_file_already_in_context_emits_notice(tmp_path):
    path = tmp_path / "already.py"
    path.write_text("print('already')\n")
    file_path = str(path)
    agent = make_persistent_agent(tmp_path)
    agent.complete = False
    agent.conversation.usermsg(
        f"[Attachment: {file_path}]",
        _attachments={file_path: "    1→print('already')\n"},
        _attachment_refs={file_path: file_path},
    )
    repl = agent._get_tool_repl()
    try:
        output, pure_syntax_error, output_chunks, _ = agent._execute_with_tool_handling(
            repl,
            f"view({file_path!r})",
        )
    finally:
        repl.close()

    assert pure_syntax_error is False
    assert "Notice: file was already in context." in output
    assert any(
        msg_type == "output" and "Calling view() on files that are already in context is wasteful." in chunk
        for msg_type, chunk in output_chunks
    )


def test_auto_preview_long_complete_turn_output(tmp_path):
    agent = make_persistent_agent(tmp_path)
    original = "x" * 6000

    result = agent.process_output_for_llm(original)

    assert "[PreviewRef: session://preview/" in result
    assert "x" * 6000 not in result
    match = re.search(r"session://preview/([0-9a-f]{16})", result)
    assert match is not None
    assert agent._session_store.get_preview_blob(agent._session_id, match.group(1)) == original


def test_auto_preview_boundary(tmp_path):
    agent = make_persistent_agent(tmp_path)
    agent.auto_preview_turn_chars = 5000

    assert agent.process_output_for_llm("x" * 5000) == "x" * 5000
    result = agent.process_output_for_llm("x" * 5001)
    assert "[PreviewRef: session://preview/" in result
    assert "x" * 5001 not in result


def test_auto_preview_after_attachment_conversion_does_not_expand_attachment_body(tmp_path):
    path = str(tmp_path / "large.txt")
    large_numbered_content = "\n".join(f"{i:>5}→{'x' * 100}" for i in range(100))
    agent = make_persistent_agent(tmp_path)

    output = agent.build_output_for_llm([
        ("read_attach", path + "\n"),
        ("read", large_numbered_content + "\n"),
    ])
    result = agent.process_output_for_llm(output)

    assert result == f"[Attachment: {path}]"
    assert "[PreviewRef:" not in result


def test_auto_preview_existing_preview_expansion(tmp_path):
    agent = make_persistent_agent(tmp_path)
    agent.complete = False
    original = "line\n" + ("x" * 6000)
    rendered = agent.process_output_for_llm(original)
    uri = re.search(r"session://preview/[0-9a-f]{16}", rendered).group(0)
    repl = agent._get_tool_repl()
    try:
        output, pure_syntax_error, output_chunks, _ = agent._execute_with_tool_handling(
            repl,
            f"view({uri!r})",
        )
        assert pure_syntax_error is False
        llm_output = agent.build_output_for_llm(output_chunks)
    finally:
        repl.close()

    assert f"Expanded preview: {uri}" in llm_output
    assert agent._expanded_preview_refs == {uri: {"numbered": False}}


def test_auto_preview_can_be_disabled(tmp_path):
    agent = make_persistent_agent(tmp_path)
    agent.auto_preview_turn_chars = 0
    original = "x" * 6000

    result = agent.process_output_for_llm(original)

    assert result == original
    assert "[PreviewRef:" not in result
