"""Reusable alternate-screen line pager."""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class ScrollView:
    lines: list[str]
    top: int = 0

    def clamp(self, height: int):
        max_top = max(0, len(self.lines) - max(1, height))
        self.top = min(max(self.top, 0), max_top)

    def scroll(self, delta: int, height: int):
        self.top += delta
        self.clamp(height)

    def page_up(self, height: int):
        self.scroll(-max(1, height - 1), height)

    def page_down(self, height: int):
        self.scroll(max(1, height - 1), height)

    def home(self):
        self.top = 0

    def end(self, height: int):
        self.top = max(0, len(self.lines) - max(1, height))

    def visible(self, height: int) -> list[str]:
        self.clamp(height)
        return self.lines[self.top:self.top + max(1, height)]


def _term_size():
    try:
        sz = os.get_terminal_size()
        return sz.columns, sz.lines
    except OSError:
        return 80, 24


def _read_key() -> str:
    data = os.read(sys.stdin.fileno(), 4096)
    if not data:
        return ""
    if data == b"\x03":
        return "ctrl_c"
    if data == b"\x1b":
        return "escape"
    if data in (b"q", b"Q"):
        return "quit"
    if data in (b"j", b"J"):
        return "down"
    if data in (b"k", b"K"):
        return "up"
    if data == b"G":
        return "end"
    if data in (b"g", b"gg"):
        return "home"
    if data in (b" ",):
        return "pagedown"
    if data in (b"b", b"B"):
        return "pageup"
    if data in (b"v", b"V"):
        return "vim"
    if data.startswith(b"\x1b["):
        seq = data[2:]
        if seq.startswith(b"A"):
            return "up"
        if seq.startswith(b"B"):
            return "down"
        if seq.startswith(b"H"):
            return "home"
        if seq.startswith(b"F"):
            return "end"
        if seq.startswith(b"5~"):
            return "pageup"
        if seq.startswith(b"6~"):
            return "pagedown"
        if seq.startswith(b"1~") or seq.startswith(b"7~"):
            return "home"
        if seq.startswith(b"4~") or seq.startswith(b"8~"):
            return "end"
    return ""


def _clip_line(line: str, width: int) -> str:
    if width <= 0:
        return ""
    line = line.replace("\t", "    ")
    return line[:width]


def _wrap_line(line: str, width: int) -> list[str]:
    if width <= 0:
        return [""]
    line = line.replace("\t", "    ")
    if not line:
        return [""]
    return [line[i:i + width] for i in range(0, len(line), width)]


def _wrap_lines(lines: list[str], width: int) -> list[str]:
    wrapped = []
    for line in lines:
        wrapped.extend(_wrap_line(line, width))
    return wrapped


def _render(lines: list[str], view: ScrollView, title: str, width: int, height: int, vim: bool = False) -> str:
    body_height = max(1, height - 2)
    visible = view.visible(body_height)
    total = len(lines)
    start = view.top + 1 if total else 0
    end = min(view.top + len(visible), total)

    out = ["\x1b[?25l", "\x1b[2J", "\x1b[H"]
    header = f" {title} "
    if total:
        header += f"{start}-{end}/{total} "
    out.append(f"\x1b[7m{header:<{width}}\x1b[0m")

    for i in range(body_height):
        out.append("\n")
        if i < len(visible):
            out.append(_clip_line(visible[i], width))
        out.append("\x1b[K")

    footer = " ↑/↓ j/k scroll | PgUp/PgDn b/Space page | gg/Home top | G/End bottom"
    if vim:
        footer += " | v vim"
    footer += " | q/Esc close "
    out.append(f"\x1b[{height};1H\x1b[2m{footer[:width]:<{width}}\x1b[0m")
    return "".join(out)


class _OpenVim(Exception):
    pass


def _open_in_vim(lines: list[str], title: str):
    suffix = os.path.splitext(title)[1] or ".txt"
    fd, path = tempfile.mkstemp(prefix="agentlib-viewer-", suffix=suffix, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        subprocess.run([os.environ.get("EDITOR", "vim"), path])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def pager_ui(altmode, lines: list[str], title: str = "Viewer", start: str = "top", vim: bool = False):
    from .prompt import RawMode

    raw_lines = lines
    view = ScrollView([])
    session = altmode.session()
    session.enter()
    try:
        with RawMode():
            width, height = _term_size()
            view.lines = _wrap_lines(raw_lines, width)
            if start == "end":
                view.end(max(1, height - 2))
            while True:
                width, height = _term_size()
                body_height = max(1, height - 2)
                view.lines = _wrap_lines(raw_lines, width)
                view.clamp(body_height)
                sys.stdout.write(_render(view.lines, view, title, width, height, vim=vim))
                sys.stdout.flush()

                key = _read_key()
                if key in ("quit", "escape", "ctrl_c"):
                    return
                if key == "vim" and vim:
                    raise _OpenVim
                if key == "down":
                    view.scroll(1, body_height)
                elif key == "up":
                    view.scroll(-1, body_height)
                elif key == "pagedown":
                    view.page_down(body_height)
                elif key == "pageup":
                    view.page_up(body_height)
                elif key == "home":
                    view.home()
                elif key == "end":
                    view.end(body_height)
    except _OpenVim:
        session.exit()
        _open_in_vim(raw_lines, title)
    finally:
        session.exit()