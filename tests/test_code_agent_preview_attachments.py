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



def test_current_context_names_include_preview_uris():
    agent = make_agent()
    agent.conversation.usermsg(
        "[Attachment: session://preview/abc]",
        _attachments={"session://preview/abc": "    1→preview content"},
        _attachment_refs={"session://preview/abc": "session://preview/abc"},
    )

    assert "session://preview/abc" in agent._current_file_context_names()


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

    assert result == f"[Attachment: {path}]\n"
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
