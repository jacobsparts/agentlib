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
    assert out == "> first\n\none\n> third\n\n"


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
    assert out == "> first\n\none\nConversation rewound.\n"


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
    assert out == "> question\n\nanswer\n> next\n\nnext answer\n"


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
    assert out == "> question\n\ncomputed answer\n"
