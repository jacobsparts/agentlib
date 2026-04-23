from agentlib.cli.rewind import _find_assistant_after


def test_find_assistant_after_prefers_assistant_release_message():
    events = [
        {"seq": 1, "event_type": "message_added", "payload": {"message": {
            "role": "user",
            "_render_segments": [{"type": "input", "content": "question"}],
        }}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit("answer", release=True)',
        }}},
    ]

    assert _find_assistant_after(events, 1) == "answer"


def test_find_assistant_after_uses_final_result_for_nonliteral_emit():
    events = [
        {"seq": 1, "event_type": "message_added", "payload": {"message": {
            "role": "user",
            "_render_segments": [{"type": "input", "content": "question"}],
        }}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": "emit(result, release=True)",
            "_final_result": "computed answer",
        }}},
    ]

    assert _find_assistant_after(events, 1) == "computed answer"


def test_find_assistant_after_extracts_multiline_concatenated_emit_literal():
    events = [
        {"seq": 1, "event_type": "message_added", "payload": {"message": {
            "role": "user",
            "_render_segments": [{"type": "input", "content": "question"}],
        }}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'emit(\n    "Done.\\n\\n"\n    "Added entrypoint",\n    release=True,\n)',
        }}},
    ]

    assert _find_assistant_after(events, 1) == "Done.\n\nAdded entrypoint"


def test_find_assistant_after_ignores_emit_inside_string_literal():
    events = [
        {"seq": 1, "event_type": "message_added", "payload": {"message": {
            "role": "user",
            "_render_segments": [{"type": "input", "content": "question"}],
        }}},
        {"seq": 2, "event_type": "message_added", "payload": {"message": {
            "role": "assistant",
            "content": 'Path("x.py").write_text("""def f():\n    emit(\\"55\\", release=True)\n""")',
        }}},
        {"seq": 3, "event_type": "message_added", "payload": {"message": {
            "role": "user",
            "content": ">>> print(123)\n123\n",
            "_stdout": ">>> print(123)\n123\n",
        }}},
    ]

    assert _find_assistant_after(events, 1) == "123"
