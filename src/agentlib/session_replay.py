import copy
import re
from pathlib import Path


def render_attachment_content(path: str) -> str:
    content = Path(path).expanduser().read_text()
    lines = content.split('\n')
    return '\n'.join(f"{i+1:>5}→{line}" for i, line in enumerate(lines))


def _load_attachment_map(refs: dict[str, str], missing: list[tuple[str, str]]) -> dict[str, str]:
    loaded = {}
    for name, path in refs.items():
        try:
            loaded[name] = render_attachment_content(path)
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


def replay_session_into_agent(agent, session_id: str, store):
    events = store.get_events(session_id)
    snapshots = {}
    messages = [copy.deepcopy(agent.conversation.messages[0])]
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
            refs = msg.pop("_attachment_refs", None) or {}
            local_missing = []
            if refs:
                loaded = _load_attachment_map(refs, local_missing)
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
        elif event_type == "rewind":
            target_seq = payload["target_seq"]
            messages = copy.deepcopy(snapshots.get(target_seq, snapshots[0]))
        snapshot(seq)

    final_missing = []
    for msg in messages:
        refs = msg.get("_attachment_refs") or {}
        attachments = msg.get("_attachments", {})
        for name, path in refs.items():
            if not Path(path).expanduser().exists():
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
    match = re.search(
        r'emit\(\s*(?P<q>["\']{1,3})(?P<text>.*?)(?P=q)\s*,\s*release\s*=\s*True\s*\)',
        content,
        re.DOTALL,
    )
    if not match:
        return ""
    return bytes(match.group("text"), "utf-8").decode("unicode_escape")


def replay_display_text(session_id: str, store) -> str:
    events = store.get_events(session_id)
    snapshots = {}
    chunks: list[str] = []
    released_to_user = False

    def snapshot(seq):
        snapshots[seq] = (copy.deepcopy(chunks), released_to_user)

    snapshot(0)
    for event in events:
        seq = event["seq"]
        payload = event["payload"]
        event_type = event["event_type"]
        if event_type == "message_added":
            msg = payload.get("message", {})
            if (
                msg.get("role") == "assistant"
                and "emit(" in (msg.get("content") or "")
                and "release=True" in (msg.get("content") or "")
            ):
                released_to_user = True
                text = _extract_released_assistant_text(msg)
                if text:
                    if not text.endswith("\n"):
                        text += "\n"
                    chunks.append(text)
        elif event_type == "display":
            kind = payload.get("kind")
            if released_to_user and kind != "input":
                snapshot(seq)
                continue
            text = payload.get("text", "")
            if text:
                chunks.append(text)
            if kind == "input":
                released_to_user = False
        elif event_type == "rewind":
            target_seq = payload["target_seq"]
            chunks, released_to_user = copy.deepcopy(snapshots.get(target_seq, snapshots[0]))
        snapshot(seq)

    return "".join(chunks)
