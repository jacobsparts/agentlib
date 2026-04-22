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
