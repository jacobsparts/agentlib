import os
import sys


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


def _render(items, selected, scroll_offset, term_width, term_height):
    out = ['\x1b[?25l', '\x1b[2J', '\x1b[H']
    title = ' /skills '
    out.append(f'\x1b[7m{title:<{term_width}}\x1b[0m\n')
    subtitle = '  ↑/↓ navigate | Space toggle attach | Enter apply | Esc cancel'
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
        checked = 'x' if item['attached'] else ' '
        source = item['source']
        name = item['name']
        line1 = f'  {marker} [{checked}] {name}  [{source}]'
        line2 = f'      {item["description"] or "(no description)"}'
        out.append(f'{rev}{line1[:term_width]:<{term_width}}{end}\n')
        out.append(f'{rev}{line2[:term_width]:<{term_width}}{end}\n\n')
    attached_count = sum(1 for item in items if item['attached'])
    footer = f'  {len(items)} skills  •  attached: {attached_count}'
    out.append(f'\x1b[2m{footer[:term_width]:<{term_width}}\x1b[0m')
    return ''.join(out)


def _render_empty(term_width, term_height):
    out = ['\x1b[?25l', '\x1b[2J', '\x1b[H']
    title = ' /skills '
    out.append(f'\x1b[7m{title:<{term_width}}\x1b[0m\n')
    subtitle = '  Esc cancel'
    out.append(f'\x1b[2m{_fit(subtitle, term_width):<{term_width}}\x1b[0m\n\n')
    out.append('  No skills found.\n')
    return ''.join(out)


def select_skills_ui(altmode, skills: list[dict]) -> list[dict] | None:
    from .prompt import RawMode
    items = [dict(item) for item in skills]
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
                    sys.stdout.write(_render(items, selected, scroll_offset, term_width, term_height))
                else:
                    sys.stdout.write(_render_empty(term_width, term_height))
                sys.stdout.flush()
                k = os.read(sys.stdin.fileno(), 4096)
                if not k:
                    continue
                c = k[0]
                if c in (3, 27) and len(k) == 1:
                    return None
                if not items:
                    continue
                if c in (10, 13):
                    return items
                if c == 32:
                    items[selected]['attached'] = not items[selected]['attached']
                    continue
                if c == 27 and len(k) >= 3 and k[1] == 91:
                    if k[2] == 65:
                        selected = max(0, selected - 1)
                    elif k[2] == 66:
                        selected = min(len(items) - 1, selected + 1)
    finally:
        session.exit()