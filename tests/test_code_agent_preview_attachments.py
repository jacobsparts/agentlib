from agentlib.agents.code_agent import CodeAgent
from agentlib.conversation import Conversation


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

    assert "Attachments currently in context:" in notice
    assert "session://preview/abc" in notice
    assert "unview(path)" in notice


def test_current_context_names_include_preview_uris():
    agent = make_agent()
    agent.conversation.usermsg(
        "[Attachment: session://preview/abc]",
        _attachments={"session://preview/abc": "    1→preview content"},
        _attachment_refs={"session://preview/abc": "session://preview/abc"},
    )

    assert "session://preview/abc" in agent._current_file_context_names()


def test_unview_detaches_preview_uri():
    agent = make_agent()
    agent.conversation.usermsg(
        "[Attachment: session://preview/abc]",
        _attachments={"session://preview/abc": "    1→preview content"},
        _attachment_refs={"session://preview/abc": "session://preview/abc"},
    )

    result = agent.unview("session://preview/abc")

    assert result == "Removed from future context: session://preview/abc"
    assert "session://preview/abc" not in agent.list_attachments()
    assert "session://preview/abc" in agent._pending_unviewed_files
