import ast
import base64
import copy
import re
from pathlib import Path


def _preview_key(path: str) -> str | None:
    prefix = "session://preview/"
    if isinstance(path, str) and path.startswith(prefix):
        key = path[len(prefix):]
        return key or None
    return None


def render_attachment_content(path: str, store=None, session_id: str | None = None) -> str:
    if key := _preview_key(path):
        if store is None or session_id is None:
            raise FileNotFoundError(path)
        content = store.get_preview_blob(session_id, key)
        if content is None:
            raise FileNotFoundError(path)
    else:
        content = Path(path).expanduser().read_text()
    lines = content.split('\n')
    return '\n'.join(f"{i+1:>5}→{line}" for i, line in enumerate(lines))


def _load_attachment_map(refs: dict[str, str], missing: list[tuple[str, str]], store=None, session_id: str | None = None) -> dict[str, str]:
    loaded = {}
    for name, path in refs.items():
        try:
            loaded[name] = render_attachment_content(path, store, session_id)
        except Exception:
            missing.append((name, path))
    return loaded


def _coalesce_user_messages(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        if msg.get("role") == "user" and out and out[-1].get("role") == "user":
            prev = out[-1]
            prev.setdefault("_render_segments", []).extend(msg.get("_render_segments") or [])
            prev_content = prev.get("content", "")
            new_content = msg.get("content", "")
            sep = "" if not prev_content or prev_content.endswith("\n") else "\n"
            merged = prev_content + sep + new_content
            if not merged.endswith("\n"):
                merged += "\n"
            prev["content"] = merged
            if msg.get("_stdout"):
                prev_stdout = prev.get("_stdout", "")
                sep_s = "" if not prev_stdout or prev_stdout.endswith("\n") else "\n"
                stdout_merged = prev_stdout + sep_s + msg["_stdout"]
                if not stdout_merged.endswith("\n"):
                    stdout_merged += "\n"
                prev["_stdout"] = stdout_merged
            if msg.get("_user_content") is not None:
                prev["_user_content"] = msg["_user_content"]
            for k in ("images", "audio"):
                if msg.get(k):
                    prev[k] = (prev.get(k) or []) + msg[k]
            if msg.get("_attachment_refs"):
                refs = prev.get("_attachment_refs") or {}
                refs.update(msg["_attachment_refs"])
                prev["_attachment_refs"] = refs
            if msg.get("_attachments"):
                attachments = prev.get("_attachments") or {}
                attachments.update(msg["_attachments"])
                prev["_attachments"] = attachments
        else:
            out.append(msg)
    return out


def _decode_media(value):
    if isinstance(value, dict) and "__b64__" in value:
        return base64.b64decode(value["__b64__"])
    if isinstance(value, list):
        return [_decode_media(item) for item in value]
    return value


def replay_session_into_agent(agent, session_id: str, store):
    events = store.get_events(session_id)
    snapshots = {}
    messages = [copy.deepcopy(agent.conversation.messages[0])]
    agent._expanded_preview_refs = {}
    if hasattr(agent, "_configure_conversation"):
        agent._configure_conversation(agent.conversation)

    missing_seen = set()

    def snapshot(seq):
        snapshots[seq] = copy.deepcopy(messages)

    snapshot(0)
    for event in events:
        seq = event["seq"]
        payload = event["payload"]
        event_type = event["event_type"]
        if event_type == "message_added":
            msg = copy.deepcopy(payload["message"])
            for key in ("images", "audio"):
                if msg.get(key):
                    msg[key] = _decode_media(msg[key])
            refs = msg.pop("_attachment_refs", None) or {}
            local_missing = []
            if refs:
                loaded = _load_attachment_map(refs, local_missing, store, session_id)
                if loaded:
                    msg["_attachments"] = loaded
                msg["_attachment_refs"] = refs
                for item in local_missing:
                    missing_seen.add(item)
            msg["_event_seq"] = seq
            for seg in reversed(msg.get("_render_segments") or []):
                if "_event_seq" not in seg:
                    seg["_event_seq"] = seq
                    break
            messages.append(msg)
        elif event_type == "attachment_invalidated":
            name = payload["name"]
            for msg in messages:
                attachments = msg.get("_attachments")
                if attachments and name in attachments:
                    del attachments[name]
                    if not attachments:
                        del msg["_attachments"]
        elif event_type == "message_pinned":
            target_seq = payload.get("message_event_seq")
            for msg in reversed(messages):
                if msg.get("_event_seq") == target_seq:
                    msg["_pinned_coalesce"] = {
                        "label": payload.get("label") or "Pinned previous turn"
                    }
                    break
        elif event_type == "preview_expanded":
            uri = payload.get("uri")
            if uri:
                agent._expanded_preview_refs[uri] = {"numbered": bool(payload.get("numbered", False))}
        elif event_type == "preview_collapsed":
            uri = payload.get("uri")
            if uri:
                agent._expanded_preview_refs.pop(uri, None)
        elif event_type == "rewind":
            target_seq = payload["target_seq"]
            messages = copy.deepcopy(snapshots.get(target_seq, snapshots[0]))
        snapshot(seq)

    final_missing = []
    for msg in messages:
        refs = msg.get("_attachment_refs") or {}
        attachments = msg.get("_attachments", {})
        for name, path in refs.items():
            if _preview_key(path):
                if store.get_preview_blob(session_id, _preview_key(path)) is None:
                    attachments.pop(name, None)
                    final_missing.append((name, path))
            elif not Path(path).expanduser().exists():
                attachments.pop(name, None)
                final_missing.append((name, path))
        if "_attachments" in msg and not msg["_attachments"]:
            del msg["_attachments"]

    messages = _coalesce_user_messages(messages)
    agent.conversation.messages = messages
    deduped = []
    seen = set()
    for item in final_missing:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _extract_released_assistant_text(msg: dict) -> str:
    value = msg.get("_final_result", msg.get("_emit_value"))
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
                break

    stripped = content.lstrip()
    if not stripped.startswith("emit("):
        return ""

    match = re.search(
        r'emit\(\s*(?P<q>["\']{1,3})(?P<text>.*?)(?P=q)\s*,\s*release\s*=\s*True\s*\)',
        content,
        re.DOTALL,
    )
    if not match:
        return ""
    return bytes(match.group("text"), "utf-8").decode("unicode_escape")


def _is_released_assistant_message(msg: dict) -> bool:
    if msg.get("_final_result", msg.get("_emit_value")) is not None:
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


def replay_display_text(session_id: str, store, format_response=None) -> str:
    events = store.get_events(session_id)
    snapshots = {}
    chunks: list[str] = []
    released_to_user = False
    just_rewound = False

    def snapshot(seq):
        snapshots[seq] = (copy.deepcopy(chunks), released_to_user, just_rewound)

    snapshot(0)
    for event in events:
        seq = event["seq"]
        payload = event["payload"]
        event_type = event["event_type"]
        if event_type == "message_added":
            msg = payload.get("message", {})
            if msg.get("role") == "assistant" and _is_released_assistant_message(msg):
                released_to_user = True
                text = _extract_released_assistant_text(msg)
                if text:
                    if format_response is not None:
                        text = format_response(text)
                    if not text.endswith("\n"):
                        text += "\n"
                    chunks.append(text)
        elif event_type == "display":
            kind = payload.get("kind")
            if released_to_user and kind != "input" and not (just_rewound and kind == "status"):
                snapshot(seq)
                continue
            text = payload.get("text", "")
            if text:
                chunks.append(text)
            if kind == "input":
                released_to_user = False
            just_rewound = False
        elif event_type == "rewind":
            target_seq = payload["target_seq"]
            chunks, released_to_user, _ = copy.deepcopy(snapshots.get(target_seq, snapshots[0]))
            just_rewound = True
        snapshot(seq)

    return "".join(chunks)
