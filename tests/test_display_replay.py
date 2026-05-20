from agentlib.session_replay import replay_display_text


class DummyStore:
    def __init__(self, events):
        self._events = events

    def get_events(self, session_id):
        return self._events


def test_replay_display_text_respects_rewind():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> first\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("one", release=True)',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "input", "text": "> second\n\n"}},
        {"seq": 4, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("two", release=True)',
        }}},
        {"seq": 5, "event_type": "rewind", "payload": {"target_seq": 2}},
        {"seq": 6, "event_type": "display", "payload": {"kind": "input", "text": "> third\n\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> first\n\n═ Output ═════════════════════════\none\n> third\n\n"


def test_replay_display_text_ignores_non_display_events():
    store = DummyStore([
        {"seq": 1, "event_type": "message_added", "payload": {"message": {"role": "user"}}},
        {"seq": 2, "event_type": "display", "payload": {"kind": "status", "text": "Attached x\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "Attached x\n"


def test_replay_display_text_keeps_prior_display_after_rewind():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> first\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("one", release=True)',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "input", "text": "> second\n\n"}},
        {"seq": 4, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("two", release=True)',
        }}},
        {"seq": 5, "event_type": "rewind", "payload": {"target_seq": 2}},
        {"seq": 6, "event_type": "display", "payload": {"kind": "status", "text": "Conversation rewound.\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> first\n\n═ Output ═════════════════════════\none\nConversation rewound.\n"


def test_replay_display_text_stops_after_release_until_next_input():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("answer", release=True)',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "python", "text": "debug work\n"}},
        {"seq": 4, "event_type": "display", "payload": {"kind": "status", "text": "Loading CLAUDE.md\n"}},
        {"seq": 5, "event_type": "display", "payload": {"kind": "input", "text": "> next\n\n"}},
        {"seq": 6, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("next answer", release=True)',
        }}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n═ Output ═════════════════════════\nanswer\n> next\n\n═ Output ═════════════════════════\nnext answer\n"


def test_replay_display_text_prefers_persisted_final_result():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": "emit(result, release=True)",
            "_final_result": "computed answer",
        }}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n═ Output ═════════════════════════\ncomputed answer\n"


def test_replay_display_text_extracts_multiline_concatenated_emit_literal():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit(\n    "Done.\\n\\n"\n    "Added entrypoint",\n    release=True,\n)',
        }}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n═ Output ═════════════════════════\nDone.\n\nAdded entrypoint\n"


def test_replay_display_text_ignores_emit_inside_string_literal():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'Path("x.py").write_text("""def f():\n    emit(\"55\", release=True)\n""")',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "input", "text": "> next\n\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n> next\n\n"


def test_replay_display_text_does_not_duplicate_emit_display_event():
    store = DummyStore([
        {"seq": 1, "event_type": "display", "payload": {"kind": "input", "text": "> question\n\n"}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("answer", release=True)',
        }}},
        {"seq": 3, "event_type": "display", "payload": {"kind": "emit", "text": "answer\n"}},
    ])

    out = replay_display_text("sid", store)
    assert out == "> question\n\n═ Output ═════════════════════════\nanswer\n"



def test_code_agent_file_diff_paths_extracts_unique_paths():
    from agentlib.agents.code_agent import CodeAgentBase

    diff = "\n".join([
        "--- a/src/old.py",
        "+++ b/src/old.py",
        "@@ -1 +1 @@",
        "-old",
        "+new",
        "--- /dev/null",
        "+++ b/src/new.py",
    ])

    assert CodeAgentBase._file_diff_paths(diff) == ["src/old.py", "src/new.py"]


def test_code_agent_records_file_diff_event_with_tool_and_paths():
    from agentlib.agents.code_agent import CodeAgentBase

    class Agent(CodeAgentBase):
        def __init__(self):
            self.events = []
            self._edit_echo_buffer = [">>> line_patch('src/app.py', 'replace 1:1\\nnew')"]

        def _append_session_event(self, event_type, payload, create_session=True):
            self.events.append((event_type, payload, create_session))
            return 1

    diff = "--- src/app.py\n+++ src/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    agent = Agent()
    agent._record_file_diff_event(diff)

    assert agent.events == [(
        "file_diff",
        {
            "kind": "unified_diff",
            "tool": "line_patch",
            "paths": ["src/app.py"],
            "diff": diff,
        },
        True,
    )]


def test_code_agent_formats_file_diff_history_filtered_and_limited():
    from agentlib.agents.code_agent import CodeAgentBase

    class Store:
        def get_events(self, session_id):
            return [
                {
                    "seq": 1,
                    "event_type": "file_diff",
                    "payload": {
                        "kind": "unified_diff",
                        "tool": "edit",
                        "paths": ["src/a.py"],
                        "diff": "--- src/a.py\n+++ src/a.py\n@@ -1 +1 @@\n-a\n+b\n",
                    },
                },
                {
                    "seq": 2,
                    "event_type": "file_diff",
                    "payload": {
                        "kind": "unified_diff",
                        "tool": "line_patch",
                        "paths": ["src/b.py"],
                        "diff": "--- src/b.py\n+++ src/b.py\n@@ -1 +1 @@\n-b\n+c\n",
                    },
                },
                {
                    "seq": 3,
                    "event_type": "file_diff",
                    "payload": {
                        "kind": "unified_diff",
                        "tool": None,
                        "paths": ["src/a.py"],
                        "diff": "--- src/a.py\n+++ src/a.py\n@@ -2 +2 @@\n-x\n+y\n",
                    },
                },
            ]

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._session_id = "sid"
    agent._session_store = Store()

    assert agent._format_file_diff_events("src/a.py", limit=1) == (
        "# file_diff seq=3 tool=unknown paths=src/a.py\n"
        "--- src/a.py\n+++ src/a.py\n@@ -2 +2 @@\n-x\n+y\n"
    )


def test_code_agent_formats_empty_file_diff_history():
    from agentlib.agents.code_agent import CodeAgentBase

    class Store:
        def get_events(self, session_id):
            return []

    agent = CodeAgentBase.__new__(CodeAgentBase)
    agent._session_id = "sid"
    agent._session_store = Store()

    assert agent._format_file_diff_events("src/missing.py") == "No file diffs recorded for src/missing.py."


def test_code_agent_diff_history_tool_bridge():
    from agentlib.agents.code_agent import CodeAgent

    class Store:
        def get_events(self, session_id):
            return [
                {
                    "seq": 10,
                    "event_type": "file_diff",
                    "payload": {
                        "kind": "unified_diff",
                        "tool": "edit",
                        "paths": ["src/demo.py"],
                        "diff": "--- src/demo.py\n+++ src/demo.py\n@@ -1 +1 @@\n-old\n+new\n",
                    },
                }
            ]

    class Repl:
        def __init__(self):
            self.replies = []
            self.acks = []

        def send_reply(self, request_id, result=None, error=None):
            self.replies.append((request_id, result, error))

        def send_ack(self, request_id):
            self.acks.append(request_id)

    agent = CodeAgent.__new__(CodeAgent)
    agent._session_id = "sid"
    agent._session_store = Store()
    repl = Repl()

    agent._handle_tool_request(repl, {
        "tool": "__file_diffs__",
        "request_id": 123,
        "args": {"file_path": "src/demo.py", "limit": None},
    })

    assert repl.replies == [(
        123,
        "# file_diff seq=10 tool=edit paths=src/demo.py\n"
        "--- src/demo.py\n+++ src/demo.py\n@@ -1 +1 @@\n-old\n+new\n",
        None,
    )]
    assert repl.acks == [123]