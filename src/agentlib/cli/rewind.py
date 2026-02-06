"""
Conversation rewind UI.

Displays conversation exchanges in the alternate screen buffer and lets the
user select one to rewind to. Handles truncation and injects a notice message.
"""

import sys
import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .altmode import AltMode
    from ..conversation import Conversation


@dataclass
class Exchange:
    user_preview: str
    assistant_preview: str
    truncate_at: int  # messages[:truncate_at] removes this exchange onward


def _preview(text: str, max_chars: int = 80) -> str:
    """First line of text, truncated."""
    first = text.strip().split('\n')[0]
    if len(first) > max_chars:
        return first[:max_chars - 3] + '...'
    return first


def build_exchanges(messages: list[dict]) -> list[Exchange]:
    """Group conversation messages into displayable exchanges."""
    exchanges = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg['role'] == 'user':
            content = msg.get('content', '')
            # Skip REPL output messages
            if content.lstrip().startswith('>>>'):
                i += 1
                continue

            user_preview = _preview(content)

            # Find last assistant message with text content before next user
            assistant_preview = ''
            j = i + 1
            while j < len(messages) and messages[j]['role'] != 'user':
                if messages[j]['role'] == 'assistant':
                    text = messages[j].get('content', '')
                    if text:
                        assistant_preview = _preview(text)
                j += 1

            exchanges.append(Exchange(
                user_preview=user_preview,
                assistant_preview=assistant_preview,
                truncate_at=i,
            ))
            i = j
        else:
            i += 1
    return exchanges


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

    # Title
    title = ' /rewind - Select a point to rewind to '
    out.append(f'\x1b[7m{title:<{term_width}}\x1b[0m\n')

    # Instructions
    out.append('\x1b[2m  Up/Down = navigate | Enter = rewind here | Esc = cancel\x1b[0m\n\n')

    # Available lines for exchanges (2 lines each + 1 blank separator)
    header_lines = 3
    footer_lines = 2
    available = term_height - header_lines - footer_lines
    lines_per = 3  # user line + assistant line + blank
    items_visible = max(1, available // lines_per)

    visible = exchanges[scroll_offset:scroll_offset + items_visible]

    for i, ex in enumerate(visible):
        idx = scroll_offset + i
        is_sel = idx == selected
        marker = '>' if is_sel else ' '
        rev = '\x1b[7m' if is_sel else ''
        end = '\x1b[0m' if is_sel else ''

        num = idx + 1
        user_line = f'  {marker} [{num}] You: {ex.user_preview}'
        user_line = user_line[:term_width]
        out.append(f'{rev}{user_line:<{term_width}}{end}\n')

        if ex.assistant_preview:
            asst_line = f'          Asst: {ex.assistant_preview}'
        else:
            asst_line = f'          \x1b[2m(no response)\x1b[0m'
        asst_line = asst_line[:term_width]
        out.append(f'{rev}{asst_line:<{term_width}}{end}\n')

        out.append('\n')

    # Footer
    sel_ex = exchanges[selected]
    num = selected + 1
    out.append(f'\x1b[2m  {len(exchanges)} exchanges | '
               f'Rewind removes exchange {num} onward\x1b[0m')

    return ''.join(out)


def rewind_ui(altmode: 'AltMode', conversation: 'Conversation') -> Optional[str]:
    """Interactive rewind UI. Truncates conversation on selection.

    Args:
        altmode: Active AltMode instance.
        conversation: Conversation object to rewind.

    Returns:
        Last assistant response text for caller to reprint, or None if cancelled.
    """
    from .prompt import RawMode

    exchanges = build_exchanges(conversation.messages)
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

                # Keep selected in view
                if selected < scroll_offset:
                    scroll_offset = selected
                if selected >= scroll_offset + items_visible:
                    scroll_offset = selected - items_visible + 1

                frame = _render(exchanges, selected, scroll_offset, term_width, term_height)
                sys.stdout.write(frame)
                sys.stdout.flush()

                k = os.read(sys.stdin.fileno(), 4096)
                if not k:
                    continue

                c = k[0]

                # Ctrl-C
                if c == 3:
                    return None

                # Bare Escape
                if c == 27 and len(k) == 1:
                    return None

                # q
                if c == ord('q'):
                    return None

                # Enter
                if c in (10, 13):
                    truncate_at = exchanges[selected].truncate_at
                    last_response = _find_last_assistant_text(
                        conversation.messages[:truncate_at]
                    )
                    conversation.messages = conversation.messages[:truncate_at]
                    conversation.usermsg(
                        '[Conversation rewound. REPL state may not match conversation context.]'
                    )
                    return last_response

                # Arrow keys
                if c == 27 and len(k) >= 3 and k[1] == 91:
                    if k[2] == 65:  # Up
                        selected = max(0, selected - 1)
                    elif k[2] == 66:  # Down
                        selected = min(len(exchanges) - 1, selected + 1)
    finally:
        session.exit()
