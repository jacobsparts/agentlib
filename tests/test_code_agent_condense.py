from agentlib.code_agent_condense import condense_code_agent_messages, build_repl_transcript


def base_messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Do thing", "_user_content": "Do thing"},
        {"role": "assistant", "content": "print('hi')"},
        {"role": "user", "content": ">>> print('hi')\nhi\n"},
    ]


def test_simple_conversation_condenses_to_system_and_transcript():
    condensed = condense_code_agent_messages(base_messages())

    assert len(condensed) == 2
    assert condensed[0] == {"role": "system", "content": "system"}
    transcript = condensed[1]["content"]
    assert "[User]\nDo thing" in transcript
    assert ">>> print('hi')" in transcript
    assert "hi" in transcript
    assert condensed[1]["_render_segments"] == [{"type": "stdout", "content": transcript}]


def test_stdout_is_preferred_over_content():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "print('x')"},
        {"role": "user", "content": ">>> truncated", "_stdout": ">>> print('x')\nfull output\n"},
    ]

    transcript, _, _ = build_repl_transcript(messages)

    assert "full output" in transcript
    assert "truncated" not in transcript


def test_user_content_appended_to_repl_output_is_split_once():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "print('x')"},
        {
            "role": "user",
            "content": ">>> print('x')\nx\nNext task",
            "_stdout": ">>> print('x')\nx\nNext task",
            "_user_content": "Next task",
        },
    ]

    transcript, _, _ = build_repl_transcript(messages)

    assert transcript.count("Next task") == 1
    assert transcript.rstrip().endswith("[User]\nNext task")
    assert ">>> print('x')\nx\n\n[User]" in transcript


def test_attachments_are_merged():
    messages = base_messages()
    messages[1]["_attachments"] = {"a": "old", "b": "bee"}
    messages[3]["_attachments"] = {"a": "new"}
    messages[1]["_attachment_refs"] = {"a": "old-path"}
    messages[3]["_attachment_refs"] = {"a": "new-path", "c": "session://preview/key"}

    condensed = condense_code_agent_messages(messages)

    assert condensed[1]["_attachments"] == {"a": "new", "b": "bee"}
    assert condensed[1]["_attachment_refs"] == {"a": "new-path", "c": "session://preview/key"}


def test_synthetic_messages_are_skipped():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "keep"},
        {"role": "user", "content": "skip", "_synthetic": True},
    ]

    transcript, _, _ = build_repl_transcript(messages)

    assert "keep" in transcript
    assert "skip" not in transcript


def test_omitted_echo_is_reconstructed():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "x = 1\nprint(x)"},
        {"role": "user", "content": "[content omitted from echo]\n1\n"},
    ]

    transcript, _, _ = build_repl_transcript(messages)

    assert ">>> x = 1\n... print(x)\n1" in transcript
    assert "[content omitted from echo]" not in transcript


def test_no_truncation():
    long_output = ">>> print('x')\n" + ("x" * 30000) + "\n"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "print('x')"},
        {"role": "user", "content": long_output},
    ]

    transcript, _, _ = build_repl_transcript(messages)

    assert "x" * 30000 in transcript


def test_condense_keeps_attachment_placeholder_instead_of_stdout_blob():
    big = ">>> view('large.py')\n" + "\n".join(f"{i:>5}→line {i}" for i in range(1000)) + "\n"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "view('large.py')"},
        {
            "role": "user",
            "content": ">>> view('large.py')\n[Attachment: large.py]\n",
            "_stdout": big,
            "_attachments": {"large.py": "    1→line 1"},
            "_attachment_refs": {"large.py": "large.py"},
        },
    ]

    transcript, attachments, refs = build_repl_transcript(messages)

    assert "[Attachment: large.py]" in transcript
    assert "line 999" not in transcript
    assert attachments == {"large.py": "    1→line 1"}
    assert refs == {"large.py": "large.py"}


def test_condense_keeps_preview_ref_instead_of_stdout_blob():
    big = ">>> preview(value)\n" + ("x" * 30000) + "\n"
    content = ">>> preview(value)\n[PreviewRef: session://preview/key]\n(1 lines, 30000 chars)\n[/PreviewRef]\n"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "preview(value)"},
        {"role": "user", "content": content, "_stdout": big},
    ]

    transcript, _, _ = build_repl_transcript(messages)

    assert "[PreviewRef: session://preview/key]" in transcript
    assert "x" * 1000 not in transcript
