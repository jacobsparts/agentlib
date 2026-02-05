"""
Alternate screen buffer history navigation.

Uses the terminal's alternate screen buffer for history navigation,
preventing scrollback buffer pollution when browsing multi-line history items.
"""

import sys
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .altmode import AltMode, Session


class AltHistoryMode:
    """Context manager for history navigation in alternate screen buffer."""

    def __init__(self, altmode: 'AltMode', display_prompt: str, display_continuation: str):
        self._altmode = altmode
        self._session: Optional['Session'] = None
        self.display_prompt = display_prompt
        self.display_continuation = display_continuation
        self.in_alt_mode = False
        self.main_cursor_row = 0  # Saved cursor row offset from start of input
        self.main_screen_row = 0  # Absolute screen row where cursor was (1-indexed)
        self.screen_content = None  # Cached screen content for redraws
        self.term_height = 24  # Cached terminal height
        self.max_input_rows = 0  # High water mark for input height (prevents bounce)
        self.cursor_row_in_input = 0  # Cursor row offset within input area
        self.saved_prev_cursor_row = 0  # Saved from main screen for restore
        self.blank_lines = 0  # Fixed blank lines above content (set on enter)

    def enter(self, buf: list, cursor: int, screen_content: str = None):
        """Enter alternate screen buffer, copying current display.

        Args:
            buf: Current input buffer
            cursor: Cursor position in buffer
            screen_content: Optional pre-rendered content to display above input
        """
        if self.in_alt_mode:
            return

        # Calculate cursor row offset from buf+cursor
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80
        
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
        self.main_cursor_row = sum(line_phys[:cursor_line]) + cursor_display_col // term_width

        try:
            self.term_height = os.get_terminal_size().lines
        except OSError:
            self.term_height = 24

        # Create and enter session (handles cursor query, pause, and alt buffer entry)
        self._session = self._altmode.session()
        self._session.enter()

        # Get cursor position from session
        row, _ = self._session.cursor_pos
        self.main_screen_row = row  # 1-indexed, or 0 if query failed

        # Get screen content from altmode if not provided directly
        if screen_content is None:
            try:
                screen_content = self._altmode.get_recent_for_screen(self.term_height)
            except Exception:
                screen_content = None

        # Cache the screen content for redraws (strip trailing newlines)
        self.screen_content = screen_content.rstrip('\n') if screen_content else screen_content

        # Base blank_lines from entry position; reduced by high water mark as items grow
        input_start_row = max(1, self.main_screen_row - self.main_cursor_row)
        content_height = (self.screen_content.count('\n') + 1) if self.screen_content else 0
        self.base_blank_lines = max(0, input_start_row - content_height - 1)
        self.original_input_rows = self.main_cursor_row + 1  # Input size at entry

        self.in_alt_mode = True
        self.max_input_rows = self.original_input_rows  # Start at entry size

        # Draw everything (content + input)
        self._redraw(buf, cursor)

    def exit_silent(self):
        """Exit alternate screen buffer, letting terminal restore main buffer."""
        if not self.in_alt_mode:
            return
        if self._session:
            self._session.exit()
            self._session = None
        self.in_alt_mode = False

    def exit(self, buf: list, cursor: int):
        """Exit alternate screen buffer, echoing content to main buffer.

        Use this when submitting content that differs from what was in main buffer.
        """
        if not self.in_alt_mode:
            return

        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80

        # Exit session (handles alt buffer exit and resume)
        if self._session:
            self._session.exit()
            self._session = None
        self.in_alt_mode = False

        # Move to start of input area
        if self.main_cursor_row > 0:
            sys.stdout.write(f'\x1b[{self.main_cursor_row}A')
        sys.stdout.write('\r')  # Column 0

        # Clear old content and echo new content
        content = ''.join(buf)
        lines = content.split('\n')

        sys.stdout.write('\x1b[J')  # Clear to end of screen
        for i, line in enumerate(lines):
            prefix = self.display_prompt if i == 0 else self.display_continuation
            sys.stdout.write(f'{prefix}{line}')
            if i < len(lines) - 1:
                sys.stdout.write('\n')

        # Position cursor
        pos = 0
        cursor_line = cursor_col = 0
        for i, line in enumerate(lines):
            if pos + len(line) >= cursor or i == len(lines) - 1:
                cursor_line, cursor_col = i, cursor - pos
                break
            pos += len(line) + 1

        # Calculate physical rows
        line_phys = self._physical_rows(lines, term_width)
        cprefix = len(self.display_prompt) if cursor_line == 0 else len(self.display_continuation)
        cursor_display_col = cprefix + cursor_col
        cursor_phys_col = cursor_display_col % term_width
        cursor_target_row = sum(line_phys[:cursor_line]) + cursor_display_col // term_width
        terminal_row = sum(line_phys) - 1

        # Position cursor
        rows_up = terminal_row - cursor_target_row
        if rows_up > 0:
            sys.stdout.write(f'\x1b[{rows_up}A')
        sys.stdout.write('\r')
        if cursor_phys_col > 0:
            sys.stdout.write(f'\x1b[{cursor_phys_col}C')
        sys.stdout.flush()

        # Return the row the cursor is on for the main prompt to track
        return cursor_target_row

    def redraw(self, buf: list, cursor: int):
        """Redraw in alt buffer."""
        if not self.in_alt_mode:
            return
        self._redraw(buf, cursor)

    def _physical_rows(self, lines: list, term_width: int) -> list:
        """Calculate physical rows for each logical line."""
        counts = []
        for i, line in enumerate(lines):
            plen = len(self.display_prompt) if i == 0 else len(self.display_continuation)
            counts.append(max(1, (plen + len(line) + term_width - 1) // term_width))
        return counts

    def _redraw(self, buf: list, cursor: int):
        """Full redraw of alternate buffer (content + input)."""
        try:
            term_width, term_height = os.get_terminal_size().columns, os.get_terminal_size().lines
        except OSError:
            term_width, term_height = 80, 24
        self.term_height = term_height

        content = ''.join(buf)
        lines = content.split('\n')
        line_phys = self._physical_rows(lines, term_width)

        # Find cursor position
        pos, cursor_line, cursor_col = 0, 0, 0
        for i, line in enumerate(lines):
            if pos + len(line) >= cursor or i == len(lines) - 1:
                cursor_line, cursor_col = i, cursor - pos
                break
            pos += len(line) + 1

        prefix_len = len(self.display_prompt) if cursor_line == 0 else len(self.display_continuation)
        cursor_display_col = prefix_len + cursor_col
        cursor_phys_col = cursor_display_col % term_width
        cursor_row_in_input = sum(line_phys[:cursor_line]) + cursor_display_col // term_width
        self.cursor_row_in_input = cursor_row_in_input  # Store for prompt.py to sync prev_cursor_row
        total_input_rows = max(sum(line_phys), cursor_row_in_input + 1)

        # Update high water mark (cap at 5 to avoid large items permanently shifting content)
        if total_input_rows <= 5:
            self.max_input_rows = max(self.max_input_rows, total_input_rows)

        # Layout: input area ratchets up via high water mark, never back down
        scroll_up = max(0, self.max_input_rows - self.original_input_rows)
        blank_lines = max(0, self.base_blank_lines - scroll_up)
        input_start_row = max(1, self.main_screen_row - (self.max_input_rows - 1))

        out = ['\x1b[?25l', '\x1b[2J', '\x1b[H']  # Hide cursor, clear screen, home

        if blank_lines > 0:
            out.append('\n' * blank_lines)
        if self.screen_content:
            out.append(self.screen_content + '\n')

        # Draw input at calculated position
        out.append(f'\x1b[{input_start_row};1H')
        for i, line in enumerate(lines):
            prefix = self.display_prompt if i == 0 else self.display_continuation
            out.append(f'{prefix}{line}\x1b[K')
            if i < len(lines) - 1:
                out.append('\n')
        out.append('\x1b[J')  # Clear to end of screen (removes remnants of larger items)

        # Position cursor within input area
        out.append(f'\x1b[{input_start_row + cursor_row_in_input};{cursor_phys_col + 1}H')
        out.append('\x1b[?25h')  # Show cursor
        sys.stdout.write(''.join(out))
        sys.stdout.flush()

    @property
    def active(self) -> bool:
        """Whether we're currently in alt mode."""
        return self.in_alt_mode

    def ensure_exit(self):
        """Ensure we exit alt mode (for cleanup)."""
        if self.in_alt_mode:
            if self._session:
                self._session.exit()
                self._session = None
            self.in_alt_mode = False
