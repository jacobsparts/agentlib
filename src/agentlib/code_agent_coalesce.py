import ast
import copy
import hashlib
import re


OMITTED_ECHO_MARKER = "[content omitted from echo]"


def _content_preserves_context_refs(content: str) -> bool:
    return (
        "[Attachment:" in content
        or "[PreviewRef:" in content
        or "[ExpandedPreviewRef:" in content
    )


def render_preview_ref(content: str) -> tuple[str, str]:
    key = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    uri = f"session://preview/{key}"
    lines = content.split("\n")
    nlines = len(lines)
    nchars = len(content)

    def render_preview_line(line):
        max_preview_line = 500
        if len(line) <= max_preview_line:
            return line
        return f"{line[:max_preview_line]}... [line truncated, {len(line)} chars total]"

    head = 8
    tail = 4
    head_indexes = list(range(min(head, nlines)))
    tail_start = max(len(head_indexes), nlines - tail)
    omitted = nlines - len(head_indexes) - (nlines - tail_start)

    parts = [f"({nlines} lines, {nchars} chars)"]
    parts.extend(render_preview_line(lines[i]) for i in head_indexes)
    if omitted:
        parts.append(f"  ... ({omitted} lines omitted)")
    parts.extend(render_preview_line(lines[i]) for i in range(tail_start, nlines))
    body = "\n".join(parts)
    return uri, f"[PreviewRef: {uri}]\n{body}\n[/PreviewRef]"


def is_release_assistant_message(msg: dict) -> bool:
    if msg.get("_final_result") is not None or msg.get("_emit_value") is not None:
        return True

    content = msg.get("content") or ""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            expr = node.value if isinstance(node, ast.Expr) else None
            if not isinstance(expr, ast.Call):
                continue
            if not isinstance(expr.func, ast.Name) or expr.func.id != "emit":
                continue
            for kw in expr.keywords:
                if kw.arg == "release":
                    try:
                        return bool(ast.literal_eval(kw.value))
                    except Exception:
                        return False
            return False

    stripped = content.lstrip()
    return stripped.startswith("emit(") and "release=True" in stripped.replace(" ", "")


def released_assistant_text(msg: dict) -> str:
    value = msg.get("_final_result")
    if value is None:
        value = msg.get("_emit_value")
    if value is not None:
        return str(value)

    content = msg.get("content") or ""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            expr = node.value if isinstance(node, ast.Expr) else None
            if not isinstance(expr, ast.Call):
                continue
            if not isinstance(expr.func, ast.Name) or expr.func.id != "emit":
                continue
            released = False
            for kw in expr.keywords:
                if kw.arg == "release":
                    try:
                        released = bool(ast.literal_eval(kw.value))
                    except Exception:
                        released = False
                    break
            if not released or not expr.args:
                continue
            try:
                return str(ast.literal_eval(expr.args[0]))
            except Exception:
                return ""

    return ""


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


def human_inputs(msg: dict) -> list[str]:
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
    if is_repl_output_message(msg):
        return []
    content = msg.get("content") or ""
    return [content] if content else []


def split_appended_user_content(msg: dict, text: str) -> tuple[str, str | None]:
    user_content = msg.get("_user_content")
    if user_content is None:
        return text, None
    suffix = str(user_content)
    if not suffix:
        return text, suffix
    candidates = [suffix, suffix.rstrip("\n")]
    for candidate in candidates:
        if candidate and text.endswith(candidate):
            stripped = text[: -len(candidate)].rstrip("\n")
            return stripped, suffix
    return text, suffix


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


def _append_block(parts: list[str], text: str):
    if text is None:
        return
    text = str(text).strip("\n")
    if not text:
        return
    if parts:
        parts.append("")
    parts.append(text)


def _real_user_message(msg: dict) -> bool:
    if msg.get("_synthetic"):
        return False
    return msg.get("role") == "user" and bool(human_inputs(msg))


def _split_repl_messages_with_appended_user(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        if msg.get("role") == "user" and is_repl_output_message(msg) and msg.get("_user_content") is not None:
            copied = copy.deepcopy(msg)
            text, appended = split_appended_user_content(copied, copied.get("content") or "")
            copied["content"] = text
            if copied.get("_stdout") is not None:
                stdout, _ = split_appended_user_content(copied, copied.get("_stdout") or "")
                copied["_stdout"] = stdout
            copied.pop("_user_content", None)
            out.append(copied)
            if appended is not None:
                out.append({"role": "user", "content": appended, "_user_content": appended})
        else:
            out.append(copy.deepcopy(msg))
    return out


def _interaction_has_execution(messages: list[dict], start: int, release: int) -> bool:
    range_messages = messages[start + 1:release]
    has_work = any(msg.get("role") == "assistant" for msg in range_messages)
    has_output = any(msg.get("role") == "user" and is_repl_output_message(msg) for msg in range_messages)
    return has_work and has_output


def _completed_interactions(messages: list[dict]) -> list[dict]:
    interactions = []
    start = None
    previous_release_output = None
    for i, msg in enumerate(messages[1:], start=1):
        if i == previous_release_output:
            continue
        if msg.get("_synthetic") and not msg.get("_coalesced"):
            continue
        if _real_user_message(msg):
            start = i
            continue
        if start is not None and msg.get("role") == "assistant" and is_release_assistant_message(msg):
            release_output = None
            if i + 1 < len(messages):
                nxt = messages[i + 1]
                if (
                    nxt.get("role") == "user"
                    and is_repl_output_message(nxt)
                    and not _real_user_message(nxt)
                    and not (nxt.get("_synthetic") and not nxt.get("_coalesced"))
                ):
                    release_output = i + 1
            interactions.append({
                "start": start,
                "release": i,
                "release_output": release_output,
                "has_execution": _interaction_has_execution(messages, start, i),
            })
            start = None
            previous_release_output = release_output
    return interactions


def _merge_message_attachments(messages: list[dict]) -> tuple[dict, dict]:
    attachments = {}
    attachment_refs = {}
    for msg in messages:
        attachments.update(msg.get("_attachments") or {})
        attachment_refs.update(msg.get("_attachment_refs") or {})
    return attachments, attachment_refs


def _attachment_placeholders(messages: list[dict]) -> list[str]:
    seen = set()
    placeholders = []
    for msg in messages:
        for match in re.finditer(r"\[Attachment: ([^\]\n]+)\]", msg.get("content") or ""):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                placeholders.append(f"[Attachment: {name}]")
        for name in (msg.get("_attachments") or {}):
            if name not in seen:
                seen.add(name)
                placeholders.append(f"[Attachment: {name}]")
        for name in (msg.get("_attachment_refs") or {}):
            if name not in seen:
                seen.add(name)
                placeholders.append(f"[Attachment: {name}]")
    return placeholders


def _strip_release_output_boilerplate(output: str, release_text: str) -> str:
    if not output:
        return ""
    lines = output.strip("\n").split("\n")
    kept = []
    release_text = (release_text or "").strip()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>> emit(") and "release=True" in stripped.replace(" ", ""):
            continue
        if release_text and stripped == release_text:
            continue
        kept.append(line)
    return "\n".join(kept).strip("\n")


def _preview_content(range_messages: list[dict], release_msg: dict | None, release_output_msg: dict | None) -> str:
    parts = []
    prev_assistant = None
    for msg in range_messages:
        role = msg.get("role")
        if role == "assistant":
            _append_block(parts, msg.get("content") or "")
            prev_assistant = msg
        elif role == "user" and is_repl_output_message(msg):
            text = message_stdout(msg)
            if prev_assistant is not None:
                text = reconstruct_omitted_echo(prev_assistant.get("content") or "", text)
            _append_block(parts, text)
            prev_assistant = None
        elif role == "user":
            for text in human_inputs(msg):
                _append_block(parts, text)
            prev_assistant = None
        else:
            prev_assistant = None

    if release_output_msg is not None:
        text = message_stdout(release_output_msg)
        text = _strip_release_output_boilerplate(text, released_assistant_text(release_msg or {}))
        _append_block(parts, text)

    return "\n".join(parts).rstrip("\n")


def _coalesced_message_from_refs(rendered_refs: list[str], range_messages: list[dict]) -> dict:
    placeholders = _attachment_placeholders(range_messages)
    visible_parts = ["[Assistant work and REPL output coalesced into preview]"]
    visible_parts.extend(placeholders)
    visible_parts.extend(["", *rendered_refs])
    visible = "\n".join(visible_parts).rstrip("\n")
    msg = {
        "role": "user",
        "content": visible,
        "_render_segments": [{"type": "stdout", "content": visible}],
        "_synthetic": True,
        "_coalesced": True,
    }
    attachments, attachment_refs = _merge_message_attachments(range_messages)
    if attachments:
        msg["_attachments"] = attachments
    if attachment_refs:
        msg["_attachment_refs"] = attachment_refs
    return msg


def _coalesced_message(content: str, range_messages: list[dict]) -> dict:
    _, rendered = render_preview_ref(content)
    return _coalesced_message_from_refs([rendered], range_messages)


def _coalesced_ordered_sections(
    range_messages: list[dict],
    release_msg: dict | None,
    release_output_msg: dict | None,
) -> list[tuple[bool, str]]:
    sections = []
    normal = []

    def flush_normal(include_release_output: bool = False):
        nonlocal normal
        content = _preview_content(
            normal,
            release_msg if include_release_output else None,
            release_output_msg if include_release_output else None,
        )
        if content:
            sections.append((False, content))
        normal = []

    i = 0
    while i < len(range_messages):
        msg = range_messages[i]
        if msg.get("role") == "assistant" and msg.get("_pinned_coalesce"):
            flush_normal()
            turn = [msg]
            if i + 1 < len(range_messages):
                nxt = range_messages[i + 1]
                if nxt.get("role") == "user" and is_repl_output_message(nxt):
                    turn.append(nxt)
                    i += 1
            content = _preview_content(turn, None, None)
            if content:
                sections.append((True, content))
        else:
            normal.append(msg)
        i += 1

    flush_normal(include_release_output=True)
    return sections


def coalesce_repl_messages(
    messages: list[dict],
    *,
    keep_last_interactions: int = 3,
    keep_last_execution_interactions: int = 1,
    min_chars: int = 2000,
    min_savings_chars: int = 1000,
    save_preview_blob=None,
    auto_expand_preview_refs: list[str] | None = None,
) -> list[dict]:
    if not messages:
        return []
    if keep_last_interactions < 0:
        raise ValueError("keep_last_interactions must be >= 0")
    if keep_last_execution_interactions < 0:
        raise ValueError("keep_last_execution_interactions must be >= 0")

    messages = _split_repl_messages_with_appended_user(messages)
    interactions = _completed_interactions(messages)
    if len(interactions) <= keep_last_interactions:
        return messages

    protected = set(range(max(0, len(interactions) - keep_last_interactions), len(interactions)))
    if keep_last_execution_interactions:
        execution_indexes = [
            i
            for i, item in enumerate(interactions)
            if item.get("has_execution")
        ]
        protected.update(execution_indexes[-keep_last_execution_interactions:])

    by_start = {item["start"]: (idx, item) for idx, item in enumerate(interactions)}
    skip_indexes = set()
    replacements = {}

    for idx, item in enumerate(interactions):
        if idx in protected:
            continue
        start = item["start"]
        release = item["release"]
        release_output = item.get("release_output")
        range_messages = messages[start + 1:release]
        replacement_range = list(range_messages)
        release_output_msg = None
        if release_output is not None:
            release_output_msg = messages[release_output]
            replacement_range.append(release_output_msg)

        if any(m.get("_coalesced") for m in replacement_range):
            continue

        section_contents = _coalesced_ordered_sections(range_messages, messages[release], release_output_msg)
        if not section_contents:
            continue

        rendered_refs = [render_preview_ref(section_content)[1] for _, section_content in section_contents]

        projected = _coalesced_message_from_refs(rendered_refs, replacement_range)
        original_chars = sum(len(m.get("content") or "") for m in replacement_range)
        replacement_chars = len(projected.get("content") or "")
        if original_chars < min_chars:
            continue
        if original_chars - replacement_chars < min_savings_chars:
            continue

        for pinned, section_content in section_contents:
            key = hashlib.sha256(section_content.encode("utf-8")).hexdigest()[:16]
            if save_preview_blob is not None:
                save_preview_blob(key, section_content)
            if pinned and auto_expand_preview_refs is not None:
                auto_expand_preview_refs.append(f"session://preview/{key}")

        replacements[start] = projected
        skip_indexes.update(range(start + 1, release))
        if release_output is not None:
            skip_indexes.add(release_output)

    projected = [copy.deepcopy(messages[0])]
    for i, msg in enumerate(messages[1:], start=1):
        if i in skip_indexes:
            continue
        projected.append(copy.deepcopy(msg))
        if i in by_start and i in replacements:
            projected.append(replacements[i])

    return projected
