import copy
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
            messages.append(msg)
        elif event_type == "message_appended":
            target_seq = payload["target_seq"]
            target = next((m for m in messages if m.get("_event_seq") == target_seq), None)
            if target is not None:
                append_content = payload.get("append_content", "")
                if append_content:
                    prev = target.get("content", "")
                    sep = "" if not prev or prev.endswith("\n") else "\n"
                    target["content"] = prev + sep + append_content + "\n"
                if payload.get("user_content") is not None:
                    target["_user_content"] = payload["user_content"]
                if payload.get("stdout_append") is not None:
                    prev_stdout = target.get("_stdout", "")
                    sep_stdout = "" if not prev_stdout or prev_stdout.endswith("\n") else "\n"
                    target["_stdout"] = prev_stdout + sep_stdout + payload["stdout_append"] + "\n"
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

    agent.conversation.messages = messages
    deduped = []
    seen = set()
    for item in final_missing:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped