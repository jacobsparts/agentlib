"""Code-agent transcript viewer."""

from .pager import pager_ui


def _llm_visible_message(msg: dict) -> dict:
    out = {k: v for k, v in msg.items() if not k.startswith("_")}
    attachments = msg.get("_attachments") or {}
    if attachments:
        content = str(out.get("content") or "")
        for name, value in attachments.items():
            content = content.replace(f"[Attachment: {name}]", str(value))
        out["content"] = content
    return out


def _message_text(msg: dict) -> str:
    visible = _llm_visible_message(msg)
    content = visible.get("content")
    if isinstance(content, str):
        return content
    return str(content or "")


def format_agent_transcript(events: list[dict]) -> list[str]:
    lines: list[str] = []
    item_num = 0
    for ev in events:
        if ev.get("event_type") != "message_added":
            continue
        msg = ev.get("payload", {}).get("message", {})
        role = msg.get("role")
        if role not in {"assistant", "user", "tool", "system"}:
            continue
        text = _message_text(msg).strip("\n")
        if not text:
            continue
        if lines:
            lines.append("")
        item_num += 1
        seq = ev.get("seq", "?")
        lines.append(f"── {role.upper()} {item_num} [event #{seq}] " + "─" * 40)
        lines.extend(text.splitlines() or [""])
    return lines or ["No agent transcript yet."]


def transcript_viewer_ui(altmode, events: list[dict]):
    pager_ui(altmode, format_agent_transcript(events), title="Agent Transcript", start="end")
