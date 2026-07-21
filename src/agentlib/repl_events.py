from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReplEvent:
    kind: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


_WORKER_EVENT_KINDS = {
    "output",
    "print",
    "error",
    "progress",
    "preview",
    "preview_expand",
    "read",
    "read_attach",
    "read_partial",
    "file_diff",
    "file_written",
    "file_unviewed",
}


def normalize_worker_message(message_type: str, text: str) -> ReplEvent:
    if message_type == "emit":
        return ReplEvent(kind="final_emit", text=text)
    if message_type in _WORKER_EVENT_KINDS:
        return ReplEvent(kind=message_type, text=text)
    return ReplEvent(
        kind="worker_output",
        text=text,
        data={"message_type": message_type},
    )


def direct_call_name(source: str) -> str | None:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        sanitized = tokenize.untokenize(
            token._replace(string="None") if token.type == tokenize.STRING else token
            for token in tokens
        )
        tree = ast.parse(sanitized)
    except (SyntaxError, tokenize.TokenError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    value = tree.body[0].value
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return None
    return value.func.id


def event_output_text(event: ReplEvent) -> str:
    if event.kind == "statement_started":
        return event.data.get("echo", event.text)
    if event.kind in {
        "statement_finished",
        "tool_called",
        "tool_returned",
        "tool_failed",
    }:
        return ""
    return event.text


def events_output_text(events: list[ReplEvent]) -> str:
    return "".join(event_output_text(event) for event in events)
