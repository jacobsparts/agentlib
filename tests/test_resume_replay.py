from agentlib.agents.code_agent import CodeAgentBase


class DummyReplayAgent:
    cli_prompt = "> "

    def __init__(self, messages):
        self.conversation = type("Conversation", (), {"messages": messages})()

    _replay_resumed_conversation = CodeAgentBase._replay_resumed_conversation
    _format_replayed_input = CodeAgentBase._format_replayed_input


def test_resume_replay_shows_input_segments_only(capsys):
    agent = DummyReplayAgent([
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": "ignored",
            "_render_segments": [
                {"type": "input", "content": "Go ahead and implement it."},
                {"type": "stdout", "content": "internal tool chatter that should not replay"},
                {"type": "input", "content": "actually that browser skill should not be built in."},
            ],
        },
    ])

    agent._replay_resumed_conversation()
    out = capsys.readouterr().out

    assert "> Go ahead and implement it." in out
    assert "> actually that browser skill should not be built in." in out
    assert "internal tool chatter that should not replay" not in out


def test_resume_replay_truncates_long_input_for_display(capsys):
    agent = DummyReplayAgent([
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": "ignored",
            "_render_segments": [
                {
                    "type": "input",
                    "content": "Header\n" + "\n".join(f"{i:>5}→line {i}" for i in range(1, 40)),
                },
            ],
        },
    ])

    agent.max_display_chars = 200
    agent._truncate_for_display = CodeAgentBase._truncate_for_display.__get__(agent, DummyReplayAgent)

    agent._replay_resumed_conversation()
    out = capsys.readouterr().out

    assert "> Header" in out
    assert "... (" in out
    assert "omitted for display" in out