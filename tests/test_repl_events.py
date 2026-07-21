import warnings

from agentlib.repl_agent import REPLMixin, ToolREPL
from agentlib.repl_events import (
    ReplEvent,
    direct_call_name,
    events_output_text,
    normalize_worker_message,
)


def test_normalize_worker_message_maps_known_and_unknown_types():
    assert normalize_worker_message("output", "value\n") == ReplEvent(
        kind="output",
        text="value\n",
    )
    assert normalize_worker_message("emit", "done\n") == ReplEvent(
        kind="final_emit",
        text="done\n",
    )
    assert normalize_worker_message("custom", "text\n") == ReplEvent(
        kind="worker_output",
        text="text\n",
        data={"message_type": "custom"},
    )


def test_direct_call_name_is_structural_and_does_not_leak_warnings():
    source = 'edit("path", "bad' + chr(92) + 'q", "new")'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert direct_call_name(source) == "edit"

    assert caught == []
    assert direct_call_name("result = edit('a', 'b', 'c')") is None
    assert direct_call_name("print(edit('a', 'b', 'c'))") == "print"
    assert direct_call_name("obj.method()") is None


def test_events_output_text_uses_statement_echo_and_omits_metadata():
    events = [
        ReplEvent(
            kind="statement_started",
            data={"echo": ">>> value\n"},
        ),
        ReplEvent(kind="output", text="1\n"),
        ReplEvent(kind="tool_called", data={"name": "demo", "args": {}}),
        ReplEvent(kind="tool_returned", data={"name": "demo", "result": None}),
        ReplEvent(kind="statement_finished", data={"had_error": False}),
    ]

    assert events_output_text(events) == ">>> value\n1\n"


def test_execute_publishes_statement_boundaries_in_order():
    agent = REPLMixin()
    agent.complete = False
    streamed = []
    completed_statements = []
    agent.on_repl_event = streamed.append
    agent.on_statement_events = lambda events: completed_statements.append(list(events))
    repl = ToolREPL(echo=False)
    repl.inject_builtins()
    try:
        output, pure_syntax_error, events, corrected_code = agent._execute_with_tool_handling(
            repl,
            "value = 1\nprint(value)",
        )
    finally:
        repl.close()

    assert pure_syntax_error is False
    assert corrected_code == "value = 1\nprint(value)"
    assert output.endswith("1\n")
    assert streamed == events
    assert [event.kind for event in events] == [
        "statement_started",
        "statement_finished",
        "statement_started",
        "print",
        "statement_finished",
    ]
    assert len(completed_statements) == 2
    assert completed_statements[0][0].data["direct_call"] is None
    assert completed_statements[1][0].data["direct_call"] == "print"


class _ReplyRecorder:
    def __init__(self):
        self.replies = []
        self.acks = []

    def send_reply(self, request_id, result=None, error=None):
        self.replies.append((request_id, result, error))

    def send_ack(self, request_id):
        self.acks.append(request_id)


def test_relay_tool_lifecycle_events_wrap_authoritative_return_value():
    agent = REPLMixin()
    events = []
    agent._active_repl_event_publisher = events.append
    agent.toolcall = lambda name, args: args["value"] + 1
    repl = _ReplyRecorder()

    agent._handle_tool_request(
        repl,
        {"tool": "demo", "request_id": None, "args": {"value": 2}},
    )

    assert events == [
        ReplEvent(
            kind="tool_called",
            data={"name": "demo", "args": {"value": 2}},
        ),
        ReplEvent(
            kind="tool_returned",
            data={"name": "demo", "result": 3},
        ),
    ]
    assert repl.replies == [(None, 3, None)]
    assert repl.acks == []


def test_relay_tool_failure_publishes_error_metadata():
    agent = REPLMixin()
    events = []
    agent._active_repl_event_publisher = events.append

    def fail(name, args):
        raise ValueError("bad tool")

    agent.toolcall = fail
    repl = _ReplyRecorder()

    agent._handle_tool_request(
        repl,
        {"tool": "demo", "request_id": None, "args": {}},
    )

    assert events == [
        ReplEvent(
            kind="tool_called",
            data={"name": "demo", "args": {}},
        ),
        ReplEvent(
            kind="tool_failed",
            data={"name": "demo", "error": "bad tool"},
        ),
    ]
    assert repl.replies == [(None, None, "bad tool")]
    assert repl.acks == []
