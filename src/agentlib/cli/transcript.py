"""Code-agent transcript viewer."""

import ast
import re

from .pager import pager_ui
from .rewind import _strip_repl_echo


def _user_text(msg: dict) -> str:
    return str(msg.get("content") or "")


def _released_assistant_text(msg: dict) -> str:
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
            released = any(
                kw.arg == "release"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in expr.keywords
            )
            if released and expr.args:
                try:
                    return str(ast.literal_eval(expr.args[0]))
                except Exception:
                    return ""

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


def format_agent_transcript(events: list[dict]) -> list[str]:
    lines: list[str] = []
    for ev in events:
        if ev.get("event_type") != "message_added":
            continue
        msg = ev.get("payload", {}).get("message", {})
        role = msg.get("role")
        text = ""
        if role == "assistant":
            text = _released_assistant_text(msg)
            heading = "ASSISTANT"
        elif role == "user":
            text = _user_text(msg)
            heading = "USER"
        else:
            continue
        text = text.strip("\n")
        if not text:
            continue
        if lines:
            lines.append("")
        seq = ev.get("seq", "?")
        lines.append(f"── {heading} #{seq} " + "─" * 40)
        lines.extend(text.splitlines() or [""])
    return lines or ["No agent transcript yet."]


def transcript_viewer_ui(altmode, events: list[dict]):
    pager_ui(altmode, format_agent_transcript(events), title="Agent Transcript", start="end")