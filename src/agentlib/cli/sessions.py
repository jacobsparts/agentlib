import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SessionItem:
    session_id: str
    cwd: str
    updated_at: str
    created_at: str
    model: str | None
    last_user_text: str | None


def _preview(text: str | None, max_chars: int = 70) -> str:
    text = (text or "").strip().split("\n")[0]
    if len(text) > max_chars:
        return text[:max_chars - 3] + "..."
    return text or "(no prompt)"


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _format_time(ts: str) -> str:
    dt = _parse_iso(ts)
    if dt is None:
        return ts
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt.astimezone(timezone.utc)
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        mins = seconds // 60
        return f"{mins}m ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h ago"
    if seconds < 86400 * 7:
        days = seconds // 86400
        return f"{days}d ago"
    return dt.astimezone().strftime("%Y-%m-%d")


def _short_id(session_id: str) -> str:
    return session_id.split('-')[0]


def _display_cwd(cwd: str, current_cwd: str, max_chars: int = 44) -> str:
    cwd_path = Path(cwd)
    current_path = Path(current_cwd)
    home = Path.home()
    try:
        rel = cwd_path.relative_to(current_path)
        text = "." if str(rel) == "." else f"./{rel}"
    except ValueError:
        try:
            rel_home = cwd_path.relative_to(home)
            text = f"~/{rel_home}"
        except ValueError:
            text = str(cwd_path)
    if len(text) > max_chars:
        return "…" + text[-(max_chars - 1):]
    return text


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return "…"
    return text[:width - 1] + "…"


def _get_term_size():
    try:
        sz = os.get_terminal_size()
        return sz.columns, sz.lines
    except OSError:
        return 100, 30


def _render(items, selected, scroll_offset, term_width, term_height, mode, current_cwd):
    out = ['\x1b[?25l', '\x1b[2J', '\x1b[H']
    title = f' /resume [{mode}] '
    out.append(f'\x1b[7m{title:<{term_width}}\x1b[0m\n')
    subtitle = '  ↑/↓ navigate | Enter resume | Tab local/global | Esc cancel'
    out.append(f'\x1b[2m{_fit(subtitle, term_width):<{term_width}}\x1b[0m\n\n')
    header_lines = 3
    footer_lines = 2
    available = term_height - header_lines - footer_lines
    lines_per = 3
    items_visible = max(1, available // lines_per)
    visible = items[scroll_offset:scroll_offset + items_visible]
    for i, item in enumerate(visible):
        idx = scroll_offset + i
        is_sel = idx == selected
        marker = '>' if is_sel else ' '
        rev = '\x1b[7m' if is_sel else ''
        end = '\x1b[0m' if is_sel else ''
        sid = _short_id(item.session_id)
        model = item.model or "unknown"
        when = _format_time(item.updated_at)
        cwd_text = _display_cwd(item.cwd, current_cwd)
        number = f'[{idx+1:>2}]'
        line1 = f'  {marker} {number} {sid}  {cwd_text}  {model}  {when}'
        line2 = f'      {_preview(item.last_user_text, max(20, term_width - 8))}'
        out.append(f'{rev}{line1[:term_width]:<{term_width}}{end}\n')
        out.append(f'{rev}{line2[:term_width]:<{term_width}}{end}\n\n')
    selected_item = items[selected]
    footer = f'  {len(items)} sessions  •  selected: {selected_item.session_id}  •  updated {_format_time(selected_item.updated_at)}'
    out.append(f'\x1b[2m{footer[:term_width]:<{term_width}}\x1b[0m')
    return ''.join(out)


def _render_empty(term_width, term_height, mode, current_cwd):
    out = ['\x1b[?25l', '\x1b[2J', '\x1b[H']
    title = f' /resume [{mode}] '
    out.append(f'\x1b[7m{title:<{term_width}}\x1b[0m\n')
    subtitle = '  Tab local/global | Esc cancel'
    out.append(f'\x1b[2m{_fit(subtitle, term_width):<{term_width}}\x1b[0m\n\n')
    if mode == "local":
        msg = f'No sessions in {_display_cwd(current_cwd, current_cwd)}'
        hint = 'Press Tab to view sessions from all directories.'
    else:
        msg = 'No sessions found.'
        hint = 'Press Tab to view sessions in this directory, or Esc to cancel.'
    out.append(f'  {msg}\n')
    out.append(f'\x1b[2m  {hint}\x1b[0m\n')
    return ''.join(out)


def select_session_ui(altmode, store, cwd: str) -> str | None:
    from .prompt import RawMode
    mode = "local"

    def load_items():
        rows = store.list_sessions(cwd=cwd if mode == "local" else None, limit=200)
        return [SessionItem(**{
            "session_id": row["session_id"],
            "cwd": row["cwd"],
            "updated_at": row["updated_at"],
            "created_at": row["created_at"],
            "model": row.get("model"),
            "last_user_text": row.get("last_user_text"),
        }) for row in rows]

    items = load_items()
    selected = 0
    scroll_offset = 0
    session = altmode.session()
    session.enter()
    try:
        with RawMode():
            while True:
                term_width, term_height = _get_term_size()
                items_visible = max(1, (term_height - 5) // 3)
                if items:
                    if selected < scroll_offset:
                        scroll_offset = selected
                    if selected >= scroll_offset + items_visible:
                        scroll_offset = selected - items_visible + 1
                    sys.stdout.write(_render(items, selected, scroll_offset, term_width, term_height, mode, cwd))
                else:
                    sys.stdout.write(_render_empty(term_width, term_height, mode, cwd))
                sys.stdout.flush()
                k = os.read(sys.stdin.fileno(), 4096)
                if not k:
                    continue
                c = k[0]
                if c in (3, 27) and len(k) == 1:
                    return None
                if c == 9:
                    mode = "global" if mode == "local" else "local"
                    items = load_items()
                    selected = 0
                    scroll_offset = 0
                    continue
                if not items:
                    continue
                if c in (10, 13):
                    return items[selected].session_id
                if c == 27 and len(k) >= 3 and k[1] == 91:
                    if k[2] == 65:
                        selected = max(0, selected - 1)
                    elif k[2] == 66:
                        selected = min(len(items) - 1, selected + 1)
    finally:
        session.exit()