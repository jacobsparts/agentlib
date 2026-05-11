import copy


OMITTED_ECHO_MARKER = "[content omitted from echo]"


def _content_preserves_context_refs(content: str) -> bool:
    return (
        "[Attachment:" in content
        or "[PreviewRef:" in content
        or "[ExpandedPreviewRef:" in content
    )


def message_stdout(msg: dict) -> str:
    content = msg.get("content") or ""
    if _content_preserves_context_refs(content):
        return content
    return msg.get("_stdout") or content


def is_repl_output_message(msg: dict) -> bool:
    content = msg.get("content") or ""
    if content.lstrip().startswith(">>>") or OMITTED_ECHO_MARKER in content:
        return True
    for seg in msg.get("_render_segments") or []:
        if seg.get("type") == "stdout":
            seg_content = seg.get("content") or ""
            if seg_content.lstrip().startswith(">>>") or OMITTED_ECHO_MARKER in seg_content:
                return True
    return False


def merge_message_attachments(messages: list[dict]) -> tuple[dict, dict]:
    attachments = {}
    attachment_refs = {}
    for msg in messages:
        if msg.get("_synthetic"):
            continue
        attachments.update(msg.get("_attachments") or {})
        attachment_refs.update(msg.get("_attachment_refs") or {})
    return attachments, attachment_refs


def _append_block(parts: list[str], text: str):
    if text is None:
        return
    text = str(text).strip("\n")
    if not text:
        return
    if parts:
        parts.append("")
    parts.append(text)


def _human_inputs(msg: dict) -> list[str]:
    if msg.get("_user_content") is not None:
        return [str(msg.get("_user_content"))]
    inputs = [
        seg.get("content") or ""
        for seg in msg.get("_render_segments") or []
        if seg.get("type") == "input"
    ]
    inputs = [text for text in inputs if text]
    if inputs:
        return inputs
    content = msg.get("content") or ""
    return [content] if content else []


def _append_user_block(parts: list[str], text: str):
    text = str(text).strip("\n")
    if not text:
        return
    if parts:
        parts.append("")
    parts.append("[User]")
    parts.append(text)


def reconstruct_omitted_echo(assistant_code: str, repl_output: str) -> str:
    if OMITTED_ECHO_MARKER not in repl_output:
        return repl_output
    assistant_code = assistant_code or ""
    if not assistant_code.strip():
        return repl_output
    echo_lines = []
    for i, line in enumerate(assistant_code.rstrip("\n").splitlines()):
        prefix = ">>> " if i == 0 else "... "
        echo_lines.append(prefix + line)
    return repl_output.replace(OMITTED_ECHO_MARKER, "\n".join(echo_lines), 1)


def _strip_appended_user_content(text: str, user_content: str | None) -> tuple[str, str | None]:
    if user_content is None:
        return text, None
    suffix = str(user_content)
    if not suffix:
        return text, suffix
    candidates = [suffix, suffix.rstrip("\n")]
    for candidate in candidates:
        if candidate and text.endswith(candidate):
            stripped = text[: -len(candidate)]
            stripped = stripped.rstrip("\n")
            return stripped, suffix
    return text, suffix


def build_repl_transcript(messages: list[dict]) -> tuple[str, dict, dict]:
    attachments, attachment_refs = merge_message_attachments(messages)
    parts: list[str] = []
    prev_assistant = None

    for msg in messages[1:]:
        if msg.get("_synthetic"):
            continue
        role = msg.get("role")
        if role == "assistant":
            prev_assistant = msg
            continue
        if role != "user":
            prev_assistant = None
            continue

        if is_repl_output_message(msg):
            text = message_stdout(msg)
            if prev_assistant is not None:
                text = reconstruct_omitted_echo(prev_assistant.get("content") or "", text)
            text, appended_user = _strip_appended_user_content(text, msg.get("_user_content"))
            _append_block(parts, text)
            if appended_user is not None:
                _append_user_block(parts, appended_user)
        else:
            for text in _human_inputs(msg):
                _append_user_block(parts, text)

        prev_assistant = None

    return "\n".join(parts).rstrip("\n") + ("\n" if parts else ""), attachments, attachment_refs


def condense_code_agent_messages(messages: list[dict]) -> list[dict]:
    if not messages:
        raise ValueError("Cannot condense an empty conversation.")
    system_message = copy.deepcopy(messages[0])
    transcript, attachments, attachment_refs = build_repl_transcript(messages)
    condensed = {
        "role": "user",
        "content": transcript,
        "_render_segments": [
            {"type": "stdout", "content": transcript},
        ],
    }
    if attachments:
        condensed["_attachments"] = attachments
    if attachment_refs:
        condensed["_attachment_refs"] = attachment_refs
    return [system_message, condensed]
