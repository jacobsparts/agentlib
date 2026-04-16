"""
Alternate screen buffer input mode.

Manages rendering of the prompt editor inside the terminal's alternate
screen buffer. Entered either because the user is navigating history (so
browsing tall items doesn't pollute scrollback) or because the current
input buffer exceeds what can be drawn on the main screen without scroll
pollution. On submit, the final buffer is echoed back into the main screen
as a single clean block.
"""

import sys
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .altmode import AltMode, Session


class AltInputMode:
    """Alt-screen rendering for the prompt editor, shared across triggers."""

    # Why we're in alt mode. Governs exit policy and layout.
    REASON_HISTORY = "history"
    REASON_OVERFLOW = "overflow"

    def __init__(
        self,
        altmode: 'AltMode',
        display_prompt: str,
        display_continuation: str,
        history_ratchet_cap: int = 5,
    ):
        self._altmode = altmode
        self._session: Optional['Session'] = None
        self.display_prompt = display_prompt
        self.display_continuation = display_continuation
        self.in_alt_mode = False
        self._reason: Optional[str] = None

        # History mode caps the input-area high-water-mark so a single
        # tall history item doesn't permanently push the captured-stdout
        # area off screen. Overflow mode disables the cap — the input can
        # use the whole alt screen.
        self._history_ratchet_cap = history_ratchet_cap

        self.main_cursor_row = 0
        self.main_screen_row = 0
        self.screen_content: Optional[str] = None
        self.term_height = 24
        self.max_input_rows = 0
        self.cursor_row_in_input = 0
        self.saved_prev_cursor_row = 0
        self.base_blank_lines = 0
        self.original_input_rows = 0

    @property
    def active(self) -> bool:
        return self.in_alt_mode

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def enter(
        self,
        buf: list,
        cursor: int,
        reason: str,
        main_cursor_row: Optional[int] = None,
        screen_content: Optional[str] = None,
    ):
        """Enter alternate screen buffer.

        main_cursor_row, if provided, is the caller's authoritative
        row-offset of the cursor from the top of input on the main screen.
        Use this when buf may not match what's on main (e.g. a bracketed
        paste has extended buf but no main-screen redraw has run). If
        omitted, we estimate from buf — only valid when buf hasn't changed
        since the last main-screen redraw.
        """
        if self.in_alt_mode:
            return

        try:
            term_width = os.get_terminal_size().columns
            term_height = os.get_terminal_size().lines
        except OSError:
            term_width = 80
            term_height = 24
        self.term_height = term_height

        self._reason = reason

        if main_cursor_row is not None:
            self.main_cursor_row = main_cursor_row
        else:
            self.main_cursor_row = self._cursor_row_in_input(buf, cursor, term_width)

        self._session = self._altmode.session()
        self._session.enter()

        row, _ = self._session.cursor_pos
        self.main_screen_row = row

        if screen_content is None:
            try:
                screen_content = self._altmode.get_recent_for_screen(term_height)
            except Exception:
                screen_content = None
        self.screen_content = screen_content.rstrip('\n') if screen_content else screen_content

        input_start_row = max(1, self.main_screen_row - self.main_cursor_row)
        content_height = (self.screen_content.count('\n') + 1) if self.screen_content else 0
        self.base_blank_lines = max(0, input_start_row - content_height - 1)
        self.original_input_rows = self.main_cursor_row + 1
        self.max_input_rows = self.original_input_rows

        self.in_alt_mode = True
        self._redraw(buf, cursor)

    def redraw(self, buf: list, cursor: int):
        if not self.in_alt_mode:
            return
        self._redraw(buf, cursor)

    def exit_silent(self):
        """Leave alt buffer without echoing anything to main. Use when the
        current buffer on exit already matches what was on main at enter."""
        if not self.in_alt_mode:
            return
        if self._session:
            self._session.exit()
            self._session = None
        self.in_alt_mode = False
        self._reason = None

    def exit(self, buf: list, cursor: int):
        """Leave alt buffer and echo the final buffer into main. Returns
        the cursor row-offset within the emitted input."""
        if not self.in_alt_mode:
            return None

        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80

        if self._session:
            self._session.exit()
            self._session = None
        self.in_alt_mode = False
        self._reason = None

        if self.main_cursor_row > 0:
            sys.stdout.write(f'\x1b[{self.main_cursor_row}A')
        sys.stdout.write('\r')
        sys.stdout.write('\x1b[J')

        content = ''.join(buf)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            prefix = self.display_prompt if i == 0 else self.display_continuation
            sys.stdout.write(f'{prefix}{line}')
            if i < len(lines) - 1:
                sys.stdout.write('\n')

        pos = 0
        cursor_line = cursor_col = 0
        for i, line in enumerate(lines):
            if pos + len(line) >= cursor or i == len(lines) - 1:
                cursor_line, cursor_col = i, cursor - pos
                break
            pos += len(line) + 1

        line_phys = self._physical_rows(lines, term_width)
        cprefix = len(self.display_prompt) if cursor_line == 0 else len(self.display_continuation)
        cursor_display_col = cprefix + cursor_col
        cursor_phys_col = cursor_display_col % term_width
        cursor_target_row = sum(line_phys[:cursor_line]) + cursor_display_col // term_width
        terminal_row = sum(line_phys) - 1

        rows_up = terminal_row - cursor_target_row
        if rows_up > 0:
            sys.stdout.write(f'\x1b[{rows_up}A')
        sys.stdout.write('\r')
        if cursor_phys_col > 0:
            sys.stdout.write(f'\x1b[{cursor_phys_col}C')
        sys.stdout.flush()

        return cursor_target_row

    def ensure_exit(self):
        if self.in_alt_mode:
            if self._session:
                self._session.exit()
                self._session = None
            self.in_alt_mode = False
            self._reason = None

    # --- internal helpers ---

    def _physical_rows(self, lines: list, term_width: int) -> list:
        counts = []
        for i, line in enumerate(lines):
            plen = len(self.display_prompt) if i == 0 else len(self.display_continuation)
            counts.append(max(1, (plen + len(line) + term_width - 1) // term_width))
        return counts

    def _cursor_row_in_input(self, buf: list, cursor: int, term_width: int) -> int:
        content = ''.join(buf)
        lines = content.split('\n')
        line_phys = self._physical_rows(lines, term_width)
        pos = 0
        cursor_line = cursor_col = 0
        for i, line in enumerate(lines):
            if pos + len(line) >= cursor or i == len(lines) - 1:
                cursor_line, cursor_col = i, cursor - pos
                break
            pos += len(line) + 1
        cprefix = len(self.display_prompt) if cursor_line == 0 else len(self.display_continuation)
        cursor_display_col = cprefix + cursor_col
        return sum(line_phys[:cursor_line]) + cursor_display_col // term_width

    def _redraw(self, buf: list, cursor: int):
        try:
            term_width = os.get_terminal_size().columns
            term_height = os.get_terminal_size().lines
        except OSError:
            term_width, term_height = 80, 24
        self.term_height = term_height

        content = ''.join(buf)
        lines = content.split('\n')
        line_phys = self._physical_rows(lines, term_width)

        pos = 0
        cursor_line = cursor_col = 0
        for i, line in enumerate(lines):
            if pos + len(line) >= cursor or i == len(lines) - 1:
                cursor_line, cursor_col = i, cursor - pos
                break
            pos += len(line) + 1

        prefix_len = len(self.display_prompt) if cursor_line == 0 else len(self.display_continuation)
        cursor_display_col = prefix_len + cursor_col
        cursor_phys_col = cursor_display_col % term_width
        cursor_row_in_input = sum(line_phys[:cursor_line]) + cursor_display_col // term_width
        self.cursor_row_in_input = cursor_row_in_input
        total_input_rows = max(sum(line_phys), cursor_row_in_input + 1)

        # Overflow mode takes the whole alt screen for the input; history
        # mode anchors the input at a ratcheted position with captured
        # stdout shown above it.
        if self._reason == self.REASON_OVERFLOW:
            input_start_row = 1
            input_area_rows = term_height
            show_captured = False
            blank_lines = 0
        else:
            if total_input_rows <= self._history_ratchet_cap:
                self.max_input_rows = max(self.max_input_rows, total_input_rows)
            scroll_up = max(0, self.max_input_rows - self.original_input_rows)
            blank_lines = max(0, self.base_blank_lines - scroll_up)
            input_start_row = max(1, self.main_screen_row - (self.max_input_rows - 1))
            input_area_rows = max(1, term_height - input_start_row + 1)
            show_captured = True

        # Tail viewport: when input exceeds available rows, show the slice
        # containing the cursor. The head drops off the top (alt has no
        # scrollback) and is effectively non-editable until the cursor
        # navigates back into it.
        if total_input_rows <= input_area_rows:
            view_start = 0
            view_rows = total_input_rows
        else:
            view_rows = input_area_rows
            view_start = max(0, cursor_row_in_input - (view_rows - 1))
            view_start = min(view_start, total_input_rows - view_rows)
        cursor_view_row = cursor_row_in_input - view_start

        out = ['\x1b[?25l', '\x1b[2J', '\x1b[H']

        if show_captured:
            if blank_lines > 0:
                out.append('\n' * blank_lines)
            if self.screen_content:
                out.append(self.screen_content + '\n')

        out.append(f'\x1b[{input_start_row};1H')

        emitted = 0
        row_idx = 0
        for i, line in enumerate(lines):
            n_phys = line_phys[i]
            line_end = row_idx + n_phys
            if line_end <= view_start:
                row_idx = line_end
                continue
            prefix = self.display_prompt if i == 0 else self.display_continuation
            full = prefix + line
            for sub in range(n_phys):
                abs_row = row_idx + sub
                if abs_row < view_start:
                    continue
                if emitted >= view_rows:
                    break
                chunk = full[sub * term_width:(sub + 1) * term_width]
                out.append(chunk)
                out.append('\x1b[K')
                if emitted < view_rows - 1:
                    out.append('\n')
                emitted += 1
            row_idx = line_end
            if emitted >= view_rows:
                break

        while emitted < view_rows and emitted <= cursor_view_row:
            out.append('\x1b[K')
            if emitted < view_rows - 1:
                out.append('\n')
            emitted += 1

        out.append('\x1b[J')

        cursor_screen_row = input_start_row + cursor_view_row
        out.append(f'\x1b[{cursor_screen_row};{cursor_phys_col + 1}H')
        out.append('\x1b[?25h')

        sys.stdout.write(''.join(out))
        sys.stdout.flush()
