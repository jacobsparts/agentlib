"""
Conversation rewind UI.

Displays conversation exchanges in the alternate screen buffer and lets the
user select one to rewind to. Handles truncation and injects a notice message.
"""

import ast
import sys
import os
import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .altmode import AltMode
    from ..conversation import Conversation


@dataclass
class Exchange:
    user_preview: str
    assistant_preview: str
    input_text: str          # Raw user input to preload on selection
    event_seq: int           # Seq of the message_added event for this input
    target_seq: int          # event_seq - 1 (snapshot to restore on rewind)
    orphaned: bool = False   # True if not present in the current branch


def _preview(text: str, max_chars: int = 80) -> str:
    """First line of text, truncated."""
    first = text.strip().split('\n')[0]
    if len(first) > max_chars:
        return first[:max_chars - 3] + '...'
    return first


def build_exchanges_from_events(events: list[dict]) -> list[Exchange]:
    """Walk the event log; every input segment becomes a rewind candidate.

    Inputs orphaned by a later rewind get marked but kept (still chronological).
    """
    exchanges: list[tuple[int, Exchange]] = []
    live_msg_seqs: set[int] = set()
    last_input_display_seq: int | None = None
    for event in events:
        seq = event['seq']
        etype = event['event_type']
        payload = event['payload']
        if etype == 'display' and payload.get('kind') == 'input':
            last_input_display_seq = seq
        elif etype == 'message_added':
            msg = payload.get('message', {})
            live_msg_seqs.add(seq)
            if msg.get('role') == 'user':
                for seg in msg.get('_render_segments') or []:
                    if seg.get('type') != 'input':
                        continue
                    input_text = seg.get('content', '')
                    asst_text = _find_assistant_after(events, seq)
                    if last_input_display_seq is not None:
                        target = last_input_display_seq - 1
                    else:
                        target = seq - 1
                    exchanges.append((seq, Exchange(
                        user_preview=_preview(input_text),
                        assistant_preview=_preview(asst_text) if asst_text else '',
                        input_text=input_text,
                        event_seq=seq,
                        target_seq=target,
                    )))
                    last_input_display_seq = None
        elif etype == 'rewind':
            target_seq = payload.get('target_seq', 0)
            live_msg_seqs = {s for s in live_msg_seqs if s <= target_seq}
    for seq, ex in exchanges:
        ex.orphaned = seq not in live_msg_seqs
    return [ex for _, ex in exchanges]


def _strip_repl_echo(text: str) -> str:
    """Remove ``>>> ``/``... `` prefixed echo lines so the REPL output reads
    as the user-visible response."""
    lines = [ln for ln in text.splitlines() if not (ln.startswith('>>> ') or ln.startswith('... '))]
    return '\n'.join(lines)


def _extract_released_assistant_text(msg: dict) -> str:
    value = msg.get('_final_result', msg.get('_emit_value'))
    if value is not None:
        return str(value)

    content = msg.get('content') or ''
    try:
        tree = ast.parse(content)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            expr = node.value if isinstance(node, ast.Expr) else None
            if not isinstance(expr, ast.Call):
                continue
            if not isinstance(expr.func, ast.Name) or expr.func.id != 'emit':
                continue
            released = False
            for kw in expr.keywords:
                if kw.arg == 'release':
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
    if not stripped.startswith('emit('):
        return ''
    match = re.search(
        r'emit\(\s*(?P<q>["\']{1,3})(?P<text>.*?)(?P=q)\s*,\s*release\s*=\s*True\s*\)',
        content,
        re.DOTALL,
    )
    if not match:
        return ''
    return bytes(match.group('text'), 'utf-8').decode('unicode_escape')


def _find_assistant_after(events: list[dict], after_seq: int) -> str:
    """Return the stripped REPL output that follows the given input — i.e.
    what the user actually saw as the assistant's response."""
    for ev in events:
        if ev['seq'] <= after_seq:
            continue
        etype = ev['event_type']
        if etype == 'rewind':
            return ''
        if etype != 'message_added':
            continue
        msg = ev['payload'].get('message', {})
        if msg.get('role') == 'assistant':
            text = _extract_released_assistant_text(msg).strip()
            if text:
                return text
            continue
        if msg.get('role') != 'user':
            continue
        has_input = any(seg.get('type') == 'input' for seg in (msg.get('_render_segments') or []))
        if has_input:
            return ''
        stdout = msg.get('_stdout') or msg.get('content', '')
        stripped = _strip_repl_echo(stdout).strip()
        if stripped:
            return stripped
    return ''


def _find_last_assistant_text(messages: list[dict]) -> str:
    """Find the last assistant message with text content."""
    for msg in reversed(messages):
        if msg['role'] == 'assistant':
            text = msg.get('content', '')
            if text:
                return text
    return ''


def _get_term_size():
    try:
        sz = os.get_terminal_size()
        return sz.columns, sz.lines
    except OSError:
        return 80, 24


def _render(exchanges, selected, scroll_offset, term_width, term_height):
    """Render the exchange list for the alt buffer."""
    out = ['\x1b[?25l', '\x1b[2J', '\x1b[H']  # Hide cursor, clear, home

    title = ' /rewind - Select a user input to rewind to and edit '
    out.append(f'\x1b[7m{title:<{term_width}}\x1b[0m\n')

    out.append('\x1b[2m  Up/Down = navigate | Enter = load this input for editing | Esc = cancel\x1b[0m\n\n')

    header_lines = 3
    footer_lines = 2
    available = term_height - header_lines - footer_lines
    lines_per = 3
    items_visible = max(1, available // lines_per)

    visible = exchanges[scroll_offset:scroll_offset + items_visible]

    for i, ex in enumerate(visible):
        idx = scroll_offset + i
        is_sel = idx == selected
        marker = '>' if is_sel else ' '
        if is_sel:
            style = '\x1b[7m'
        elif ex.orphaned:
            style = '\x1b[2;33m'
        else:
            style = ''
        end = '\x1b[0m' if style else ''

        tag = '~' if ex.orphaned else ' '
        num = idx + 1
        user_line = f' {tag}{marker} [{num}] You: {ex.user_preview}'
        user_line = user_line[:term_width]
        out.append(f'{style}{user_line:<{term_width}}{end}\n')

        if ex.assistant_preview:
            asst_line = f'        Asst: {ex.assistant_preview}'
        else:
            asst_line = '        (no response)'
        asst_line = asst_line[:term_width]
        out.append(f'{style}{asst_line:<{term_width}}{end}\n')

        out.append('\n')

    num = selected + 1
    orphan_count = sum(1 for ex in exchanges if ex.orphaned)
    legend = f' | \x1b[2;33m~ = orphaned ({orphan_count})\x1b[0m\x1b[2m' if orphan_count else ''
    out.append(f'\x1b[2m  {len(exchanges)} inputs{legend} | Rewind to input {num} (preloads text for editing)\x1b[0m')

    return ''.join(out)


def rewind_ui(altmode: 'AltMode', events: list[dict]) -> Optional[dict]:
    """Interactive rewind UI. Returns the selected target metadata."""
    from .prompt import RawMode

    exchanges = build_exchanges_from_events(events)
    if not exchanges:
        return None

    selected = len(exchanges) - 1
    scroll_offset = 0

    session = altmode.session()
    session.enter()

    try:
        with RawMode():
            while True:
                term_width, term_height = _get_term_size()
                lines_per = 3
                items_visible = max(1, (term_height - 5) // lines_per)

                if selected < scroll_offset:
                    scroll_offset = selected
                if selected >= scroll_offset + items_visible:
                    scroll_offset = selected - items_visible + 1

                frame = _render(exchanges, selected, scroll_offset, term_width, term_height)
                sys.stdout.write(frame)
                sys.stdout.flush()

                try:
                    k = os.read(sys.stdin.fileno(), 4096)
                except KeyboardInterrupt:
                    return None
                if not k:
                    continue

                c = k[0]
                if c == 3:
                    return None
                if c == 27 and len(k) == 1:
                    return None
                if c == ord('q'):
                    return None
                if c in (10, 13):
                    chosen = exchanges[selected]
                    return {
                        "target_seq": chosen.target_seq,
                        "preload_input": chosen.input_text,
                    }
                if c == 27 and len(k) >= 3 and k[1] == 91:
                    if k[2] == 65:
                        selected = max(0, selected - 1)
                    elif k[2] == 66:
                        selected = min(len(exchanges) - 1, selected + 1)
    finally:
        session.exit()
