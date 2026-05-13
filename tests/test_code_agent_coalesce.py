import importlib.util
import re

from agentlib.code_agent_coalesce import coalesce_repl_messages


def interaction(i, work_size=2500):
    work = f"print({i})\n" + ("x" * work_size)
    output = f">>> print({i})\n{i}\n" + ("y" * work_size)
    return [
        {"role": "user", "content": f"Task {i}", "_user_content": f"Task {i}"},
        {"role": "assistant", "content": work},
        {"role": "user", "content": output},
        {"role": "assistant", "content": f"emit('Done {i}', release=True)"},
        {"role": "user", "content": f">>> emit('Done {i}', release=True)\nDone {i}\n"},
    ]


def five_interactions():
    messages = [{"role": "system", "content": "system"}]
    for i in range(5):
        messages.extend(interaction(i))
    return messages


def test_keeps_last_three_interactions_uncoalesced():
    messages = five_interactions()

    projected = coalesce_repl_messages(messages, min_chars=1, min_savings_chars=1)

    assert projected[0] == messages[0]
    assert sum(1 for m in projected if m.get("_coalesced")) == 2
    assert projected[-15:] == messages[-15:]


def test_preserves_real_user_inputs_and_release_assistant_messages():
    saved = {}
    messages = [{"role": "system", "content": "system"}]
    messages.extend(interaction(0))

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
    )

    assert any(m["role"] == "user" and m.get("_user_content") == "Task 0" for m in projected)
    assert any(m["role"] == "assistant" and m["content"] == "emit('Done 0', release=True)" for m in projected)
    visible = "\n".join(m.get("content") or "" for m in projected)
    assert "x" * 600 not in visible
    assert list(saved.values())[0].startswith("print(0)")


def test_saves_preview_blob_deterministically():
    messages = [{"role": "system", "content": "system"}] + interaction(0)
    saved1 = {}
    saved2 = {}

    projected1 = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1, save_preview_blob=saved1.setdefault)
    projected2 = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1, save_preview_blob=saved2.setdefault)

    assert saved1 == saved2
    assert projected1 == projected2
    key = next(iter(saved1))
    assert f"session://preview/{key}" in projected1[2]["content"]


def test_skips_small_interactions():
    messages = [{"role": "system", "content": "system"}] + interaction(0, work_size=1)

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=2000, min_savings_chars=1)

    assert projected == messages


def test_attachment_placeholders_and_payloads_survive():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "view('src/foo.py')\n" + ("x" * 2500)},
        {
            "role": "user",
            "content": ">>> view('src/foo.py')\n[Attachment: src/foo.py]\n" + ("y" * 2500),
            "_attachments": {"src/foo.py": "    1→print('hi')"},
            "_attachment_refs": {"src/foo.py": "src/foo.py"},
        },
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1)
    coalesced = next(m for m in projected if m.get("_coalesced"))

    assert "[Attachment: src/foo.py]" in coalesced["content"]
    assert coalesced["_attachments"] == {"src/foo.py": "    1→print('hi')"}
    assert coalesced["_attachment_refs"] == {"src/foo.py": "src/foo.py"}


def test_nested_preview_refs_remain_placeholders():
    saved = {}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "preview(value)\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> preview(value)\n[PreviewRef: session://preview/abc]\n(1 lines, 30000 chars)\n[/PreviewRef]\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1, save_preview_blob=saved.setdefault)
    content = next(iter(saved.values()))

    assert "[PreviewRef: session://preview/abc]" in content
    assert "[ExpandedPreviewRef:" not in content


def test_appended_user_content_is_preserved():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
        {
            "role": "user",
            "content": ">>> emit('Done', release=True)\nDone\nNext task",
            "_stdout": ">>> emit('Done', release=True)\nDone\nNext task",
            "_user_content": "Next task",
        },
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1)
    assert any(m["role"] == "user" and m.get("_user_content") == "Next task" for m in projected)
    coalesced = next(m for m in projected if m.get("_coalesced"))
    assert "Next task" not in coalesced["content"]


def test_release_output_extra_non_release_output_goes_to_preview():
    saved = {}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
        {"role": "user", "content": ">>> emit('Done', release=True)\nDone\nextra line\n"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1, save_preview_blob=saved.setdefault)

    assert any(m["role"] == "assistant" and m["content"] == "emit('Done', release=True)" for m in projected)
    assert "extra line" in next(iter(saved.values()))


def test_coalesced_messages_are_synthetic():
    messages = [{"role": "system", "content": "system"}] + interaction(0)

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1)
    coalesced = next(m for m in projected if m.get("_coalesced"))

    assert coalesced["_synthetic"] is True
    assert coalesced["_coalesced"] is True


def test_removed_legacy_manual_command_and_module():
    assert importlib.util.find_spec("agentlib.code_agent_condense") is None
    with open("src/agentlib/agents/code_agent.py") as f:
        assert "/condense" not in f.read()


def test_coalesce_is_idempotent_on_projected_messages():
    messages = [{"role": "system", "content": "system"}] + interaction(0)
    saved = {}

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
    )
    projected_again = coalesce_repl_messages(
        projected,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
    )

    assert projected_again == projected
    assert sum(1 for m in projected_again if m.get("_coalesced")) == 1


def test_release_detection_accepts_emit_value_metadata_when_final_result_is_none():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit(result, release=True)", "_final_result": None, "_emit_value": ""},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1)

    assert any(m.get("_coalesced") for m in projected)
    assert projected[-1]["role"] == "assistant"


def test_release_detection_finds_top_level_emit_after_non_emit_work():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "result = 'done'\nemit(result, release=True)"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1)

    assert any(m.get("_coalesced") for m in projected)


def test_render_segment_input_is_preserved_as_real_user_input():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "displayed", "_render_segments": [{"type": "input", "content": "Segment task"}]},
        {"role": "assistant", "content": "print('x')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('x')\nx\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1)

    assert projected[1]["_render_segments"] == [{"type": "input", "content": "Segment task"}]
    assert any(m.get("_coalesced") for m in projected)


def test_omitted_echo_is_reconstructed_in_preview_blob():
    saved = {}
    code = "for i in range(2):\n    print(i)"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": code + "\n" + ("x" * 2500)},
        {"role": "user", "content": "[content omitted from echo]\n0\n1\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1, save_preview_blob=saved.setdefault)
    content = next(iter(saved.values()))

    assert ">>> for i in range(2):" in content
    assert "...     print(i)" in content


def test_attachment_placeholder_order_is_first_seen_and_deduplicated():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "work\n" + ("x" * 2500)},
        {
            "role": "user",
            "content": ">>> work\n[Attachment: b.py]\n[Attachment: a.py]\n[Attachment: b.py]\n" + ("y" * 2500),
            "_attachments": {"a.py": "a", "c.py": "c"},
            "_attachment_refs": {"d.py": "d"},
        },
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    projected = coalesce_repl_messages(messages, keep_last_interactions=0, keep_last_execution_interactions=0, min_chars=1, min_savings_chars=1)
    coalesced = next(m for m in projected if m.get("_coalesced"))
    lines = coalesced["content"].splitlines()

    assert lines[1:5] == ["[Attachment: b.py]", "[Attachment: a.py]", "[Attachment: c.py]", "[Attachment: d.py]"]


def test_small_interaction_does_not_save_preview_blob():
    saved = {}
    messages = [{"role": "system", "content": "system"}] + interaction(0, work_size=1)

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=2000,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
    )

    assert projected == messages
    assert saved == {}


def test_replay_then_coalesce_keeps_raw_events_unmodified_and_reapplies_projection(tmp_path):
    from agentlib.session_replay import replay_session_into_agent
    from agentlib.session_store import SessionStore

    class Conversation:
        def __init__(self):
            self.messages = [{"role": "system", "content": "system"}]

    class Agent:
        def __init__(self):
            self.conversation = Conversation()
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.expanded_preview_refs = self._expanded_preview_refs

    raw_messages = interaction(0)
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    for seq, msg in enumerate(raw_messages, start=1):
        store.append_event(session_id, seq, "message_added", {"message": msg})

    agent = Agent()
    replay_session_into_agent(agent, session_id, store)
    before_projection_events = store.get_events(session_id)
    assert not any(m.get("_coalesced") for m in agent.conversation.messages)

    projected = coalesce_repl_messages(
        agent.conversation.messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=lambda key, content: store.save_preview_blob(session_id, key, content),
    )

    assert any(m.get("_synthetic") and m.get("_coalesced") for m in projected)
    assert store.get_events(session_id) == before_projection_events

    replayed_agent = Agent()
    replay_session_into_agent(replayed_agent, session_id, store)
    projected_again = coalesce_repl_messages(
        replayed_agent.conversation.messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=lambda key, content: store.save_preview_blob(session_id, key, content),
    )

    assert projected_again == projected


def test_preview_expansion_event_survives_resume_replay_for_coalesced_preview(tmp_path):
    from agentlib.session_replay import replay_session_into_agent
    from agentlib.session_store import SessionStore

    class Conversation:
        def __init__(self):
            self.messages = [{"role": "system", "content": "system"}]

    class Agent:
        def __init__(self):
            self.conversation = Conversation()
            self._expanded_preview_refs = {}

        def _configure_conversation(self, conversation):
            conversation.expanded_preview_refs = self._expanded_preview_refs

    raw_messages = interaction(0)
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")
    seq = 1
    for msg in raw_messages:
        store.append_event(session_id, seq, "message_added", {"message": msg})
        seq += 1

    agent = Agent()
    replay_session_into_agent(agent, session_id, store)
    projected = coalesce_repl_messages(
        agent.conversation.messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=lambda key, content: store.save_preview_blob(session_id, key, content),
    )
    coalesced = next(m for m in projected if m.get("_coalesced"))
    uri = coalesced["content"].split("[PreviewRef: ", 1)[1].split("]", 1)[0]
    key = uri.rsplit("/", 1)[1]
    assert store.get_preview_blob(session_id, key)

    store.append_event(session_id, seq, "preview_expanded", {"uri": uri, "numbered": True})

    resumed = Agent()
    replay_session_into_agent(resumed, session_id, store)
    assert resumed._expanded_preview_refs == {uri: {"numbered": True}}

    projected_after_resume = coalesce_repl_messages(
        resumed.conversation.messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=lambda key, content: store.save_preview_blob(session_id, key, content),
    )
    assert any(uri in (m.get("content") or "") for m in projected_after_resume if m.get("_coalesced"))
    assert resumed._expanded_preview_refs == {uri: {"numbered": True}}



def test_pinned_turn_creates_auto_expanded_preview_ref():
    saved = {}
    auto_expand = []
    messages = [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('unpinned')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('unpinned')\nunpinned\n" + ("y" * 2500)},
        {"role": "assistant", "content": "print('important')", "_pinned_coalesce": {"label": "Pinned previous turn"}},
        {"role": "user", "content": ">>> print('important')\nimportant\n"},
        {"role": "assistant", "content": "emit('Done', release=True)"},
        {"role": "user", "content": ">>> emit('Done', release=True)\nDone\n"},
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
        auto_expand_preview_refs=auto_expand,
    )
    coalesced = next(m for m in projected if m.get("_coalesced"))

    assert coalesced["content"].count("[PreviewRef: session://preview/") == 2
    assert len(auto_expand) == 1
    pinned_key = auto_expand[0].rsplit("/", 1)[1]
    assert "important" in saved[pinned_key]
    assert "unpinned" not in saved[pinned_key]


def test_pinned_turn_without_repl_output_is_preserved():
    saved = {}
    auto_expand = []
    messages = [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "think('important')", "_pinned_coalesce": {"label": "Pinned previous turn"}},
        {"role": "assistant", "content": "print('work')\n" + ("x" * 2500)},
        {"role": "user", "content": ">>> print('work')\nwork\n" + ("y" * 2500)},
        {"role": "assistant", "content": "emit('Done', release=True)"},
    ]

    coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
        auto_expand_preview_refs=auto_expand,
    )

    assert len(auto_expand) == 1
    assert saved[auto_expand[0].rsplit("/", 1)[1]] == "think('important')"


def test_multiple_pinned_turns_keep_ordered_preview_sections():
    saved = {}
    auto_expand = []
    messages = [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": "Task", "_user_content": "Task"},
        {"role": "assistant", "content": "print('before')\n" + ("b" * 2500)},
        {"role": "user", "content": ">>> print('before')\nbefore\n"},
        {"role": "assistant", "content": "print('pin1')", "_pinned_coalesce": {"label": "Pinned previous turn"}},
        {"role": "user", "content": ">>> print('pin1')\npin1\n"},
        {"role": "assistant", "content": "print('middle')\n" + ("m" * 2500)},
        {"role": "user", "content": ">>> print('middle')\nmiddle\n"},
        {"role": "assistant", "content": "print('pin2')", "_pinned_coalesce": {"label": "Pinned previous turn"}},
        {"role": "user", "content": ">>> print('pin2')\npin2\n"},
        {"role": "assistant", "content": "emit('Done', release=True)"},
        {"role": "user", "content": ">>> emit('Done', release=True)\nDone\n"},
    ]

    projected = coalesce_repl_messages(
        messages,
        keep_last_interactions=0,
        keep_last_execution_interactions=0,
        min_chars=1,
        min_savings_chars=1,
        save_preview_blob=saved.setdefault,
        auto_expand_preview_refs=auto_expand,
    )

    coalesced = next(m for m in projected if m.get("_coalesced"))
    uris = re.findall(r"session://preview/[0-9a-f]{16}", coalesced["content"])
    unique_uris = list(dict.fromkeys(uris))
    assert len(unique_uris) == 4
    assert len(auto_expand) == 2
    assert unique_uris[1] == auto_expand[0]
    assert unique_uris[3] == auto_expand[1]
    contents = [saved[uri.rsplit("/", 1)[1]] for uri in unique_uris]
    assert "before" in contents[0]
    assert contents[1].startswith("print('pin1')")
    assert "middle" in contents[2]
    assert contents[3].startswith("print('pin2')")


def chat_interaction(i):
    return [
        {"role": "user", "content": f"Question {i}", "_user_content": f"Question {i}"},
        {"role": "assistant", "content": f"emit('Answer {i}', release=True)"},
        {"role": "user", "content": f">>> emit('Answer {i}', release=True)\nAnswer {i}\n"},
    ]


def test_default_keeps_most_recent_execution_interaction_even_past_last_three():
    messages = [{"role": "system", "content": "system"}]
    for i in range(2):
        messages.extend(interaction(i))
    for i in range(3):
        messages.extend(chat_interaction(i))

    projected = coalesce_repl_messages(messages, min_chars=1, min_savings_chars=1)

    assert sum(1 for m in projected if m.get("_coalesced")) == 1
    visible = "\n".join(m.get("content") or "" for m in projected)
    assert "Task 1" in visible
    assert "print(1)" in visible
    assert "x" * 600 in visible
    assert "Task 0" in visible
    assert "print(0)\n" in visible
    assert "x" * 600 not in projected[2]["content"]


def test_release_output_is_not_treated_as_next_interaction_start_for_execution_policy():
    messages = [{"role": "system", "content": "system"}]
    messages.extend(interaction(0))
    messages.extend(interaction(1))
    for i in range(3):
        messages.extend(chat_interaction(i))

    projected = coalesce_repl_messages(messages, min_chars=1, min_savings_chars=1)

    visible = "\n".join(m.get("content") or "" for m in projected)
    assert sum(1 for m in projected if m.get("_coalesced")) == 1
    assert "Task 1" in visible
    assert "print(1)" in visible
    assert "x" * 600 in visible
