"""
Pure Python readline replacement with bracketed paste support.

Provides terminal input handling without requiring GNU readline >= 8.0.
"""

import sys
import os
import termios
from typing import Optional, Callable, TYPE_CHECKING

from .alt_input import AltInputMode

if TYPE_CHECKING:
    from .altmode import AltMode

_PRINTABLE = set(range(32, 127))


class RawMode:
    """Context manager for raw terminal mode with bracketed paste."""
    
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.original_attrs = None
    
    def __enter__(self):
        self.original_attrs = termios.tcgetattr(self.fd)
        attrs = termios.tcgetattr(self.fd)  # Get fresh copy to modify
        attrs[3] &= ~termios.ECHO
        attrs[3] &= ~termios.ICANON
        termios.tcsetattr(self.fd, termios.TCSADRAIN, attrs)
        sys.stdout.write('\x1b[?2004h')  # Enable bracketed paste
        sys.stdout.flush()
        return self
    
    def __exit__(self, type, value, traceback):
        sys.stdout.write('\x1b[?2004l')  # Disable bracketed paste
        sys.stdout.flush()
        if self.original_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_attrs)


def prompt(
    prompt_str: str = '',
    continuation_str: str = '',
    history: Optional[list] = None,
    on_submit: Optional[Callable[[str], None]] = None,
    add_to_history: bool = True,
    altmode: Optional['AltMode'] = None,
    initial_text: str = '',
) -> str:
    """
    Read a line of input with editing support.
    
    Args:
        prompt_str: Prompt for first line
        continuation_str: Prompt for continuation lines (after Alt+Enter)
        history: List to use for history (modified in place)
        on_submit: Callback when input is submitted (for persistence)
        add_to_history: If False, don't add input to history
    
    Returns:
        The input string
    
    Raises:
        EOFError: On Ctrl+D
        KeyboardInterrupt: On Ctrl+C
    
    Supports:
        - Arrow keys for cursor movement
        - Home/End, Ctrl+A/E for line start/end
        - Backspace/Delete
        - Ctrl+K (kill to end), Ctrl+U (kill to start)
        - Ctrl+L (clear screen)
        - Up/Down for history navigation
        - Alt+Enter for newline insertion
        - Bracketed paste for multiline content
    """
    if history is None:
        history = []
    
    prev_lines = 1
    prev_cursor_row = 0  # Physical row where cursor was left (0-indexed from top)
    
    # Strip leading newlines from prompt - they're only for initial display
    display_prompt = prompt_str.lstrip('\n')
    display_continuation = continuation_str.lstrip('\n')
    # Alt mode for history navigation AND tall-input overflow (prevents
    # scrollback pollution in both cases).
    alt_input = AltInputMode(altmode, display_prompt, display_continuation) if altmode else None
    def _redraw(buf, cursor):
        nonlocal prev_lines, prev_cursor_row

        try:
            term_width = os.get_terminal_size().columns
            term_height = os.get_terminal_size().lines
        except OSError:
            term_width = 80
            term_height = 24

        # Scrollback is immutable: anything above the current viewport top
        # cannot be rewritten, and any '\n' we emit past the last visible
        # row causes the terminal to scroll, which pushes our freshly-drawn
        # content into scrollback on every keystroke. Stay strictly within
        # the viewport. Leave one row of clearance for submit's '\n'.
        view_budget = max(1, term_height - 1)

        content = ''.join(buf)
        lines = content.split('\n')

        line_physical_counts = []
        for i, line in enumerate(lines):
            prefix_len = len(display_prompt) if i == 0 else len(display_continuation)
            line_len = prefix_len + len(line)
            line_physical_counts.append(max(1, (line_len + term_width - 1) // term_width))

        content_rows = sum(line_physical_counts)

        pos = 0
        cursor_line = 0
        cursor_col = 0
        for i, line in enumerate(lines):
            if pos + len(line) >= cursor or i == len(lines) - 1:
                cursor_line = i
                cursor_col = cursor - pos
                break
            pos += len(line) + 1

        cprefix_len = len(display_prompt) if cursor_line == 0 else len(display_continuation)
        cursor_display_col = cprefix_len + cursor_col
        cursor_abs_row = sum(line_physical_counts[:cursor_line]) + cursor_display_col // term_width
        cursor_phys_col = cursor_display_col % term_width

        total_rows = max(content_rows, cursor_abs_row + 1)

        # Pick the viewport slice. When content fits, show from row 0. When
        # it overflows, show the tail containing the cursor. (The overflow
        # path here is a fallback that degrades legibility but preserves
        # scrollback; proper handling kicks input into the alt screen.)
        if total_rows <= view_budget:
            view_start = 0
            view_rows = total_rows
        else:
            view_rows = view_budget
            view_start = max(0, cursor_abs_row - (view_rows - 1))
            view_start = min(view_start, total_rows - view_rows)

        cursor_view_row = cursor_abs_row - view_start

        out = ['\x1b[?25l']  # Hide cursor
        # Defensive: never try to cursor-up past the top of the viewport.
        up = min(prev_cursor_row, view_budget - 1)
        if up > 0:
            out.append(f'\x1b[{up}A')
        out.append('\r')

        # Emit only the physical rows that fall inside the viewport.
        emitted = 0
        row_idx = 0
        for i, line in enumerate(lines):
            n_phys = line_physical_counts[i]
            line_end = row_idx + n_phys
            if line_end <= view_start:
                row_idx = line_end
                continue
            prefix = display_prompt if i == 0 else display_continuation
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

        # Handle cursor past content end (wrap boundary: typed to the Nth
        # column, cursor sits on a new row that content hasn't filled).
        while emitted < view_rows and emitted <= cursor_view_row:
            out.append('\x1b[K')
            if emitted < view_rows - 1:
                out.append('\n')
            emitted += 1

        # Clear anything left on screen below our viewport (remnants from a
        # previous taller draw).
        out.append('\x1b[J')

        last_row_idx = max(0, emitted - 1)
        rows_up = last_row_idx - cursor_view_row
        if rows_up > 0:
            out.append(f'\x1b[{rows_up}A')

        out.append('\r')
        if cursor_phys_col > 0:
            out.append(f'\x1b[{cursor_phys_col}C')

        out.append('\x1b[?25h')  # Show cursor
        sys.stdout.write(''.join(out))
        sys.stdout.flush()

        prev_cursor_row = cursor_view_row
        prev_lines = view_rows

    def _measure(buf, cursor):
        """Return (total_rows, cursor_abs_row) for the current buffer.
        Used by the dispatcher to decide main vs. alt-screen rendering."""
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80
        content = ''.join(buf)
        lines = content.split('\n')
        phys = []
        for i, line in enumerate(lines):
            prefix_len = len(display_prompt) if i == 0 else len(display_continuation)
            phys.append(max(1, (prefix_len + len(line) + term_width - 1) // term_width))
        pos = 0
        cursor_line = cursor_col = 0
        for i, line in enumerate(lines):
            if pos + len(line) >= cursor or i == len(lines) - 1:
                cursor_line, cursor_col = i, cursor - pos
                break
            pos += len(line) + 1
        cprefix_len = len(display_prompt) if cursor_line == 0 else len(display_continuation)
        cursor_abs_row = sum(phys[:cursor_line]) + (cprefix_len + cursor_col) // term_width
        return max(sum(phys), cursor_abs_row + 1), cursor_abs_row

    def redraw(buf, cursor):
        """Route rendering to main _redraw or alt-screen based on
        whether content fits in the main-screen viewport and whether
        alt mode is already active for some other reason."""
        nonlocal prev_cursor_row, prev_lines

        # Already in alt for some reason — stay in alt.
        if alt_input and alt_input.active:
            alt_input.redraw(buf, cursor)
            prev_cursor_row = alt_input.cursor_row_in_input
            return

        try:
            term_height = os.get_terminal_size().lines
        except OSError:
            term_height = 24
        view_budget = max(1, term_height - 1)

        total_rows, cursor_abs_row = _measure(buf, cursor)
        overflow = total_rows > view_budget or cursor_abs_row >= view_budget

        if overflow and alt_input:
            # main_cursor_row = what's actually on main (prev_cursor_row
            # reflects the last _redraw's layout, not buf, which may have
            # just been extended by a paste).
            alt_input.saved_prev_cursor_row = prev_cursor_row
            alt_input.enter(
                buf, cursor,
                reason=AltInputMode.REASON_OVERFLOW,
                main_cursor_row=prev_cursor_row,
            )
            prev_cursor_row = alt_input.cursor_row_in_input
            return

        _redraw(buf, cursor)

    with RawMode():
        sys.stdout.write(prompt_str + initial_text)
        sys.stdout.flush()
        buf = list(initial_text)
        cursor = len(buf)
        history_idx = len(history)
        saved_line = []
        
        try:
            while True:
                k = os.read(sys.stdin.fileno(), 4096)
                if not k:
                    raise EOFError()
            
                i = 0
                while i < len(k):
                    c = k[i]
                
                    # Ctrl+D - EOF
                    if c == 4:
                        if not buf:
                            sys.stdout.write('\n')
                            sys.stdout.flush()
                            raise EOFError()
                        i += 1
                        continue
                
                    # Ctrl+C - interrupt
                    if c == 3:
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                        raise KeyboardInterrupt()
                
                    # Alt+Enter - insert newline
                    if c == 27 and i + 1 < len(k) and k[i+1] in (10, 13):
                        buf.insert(cursor, '\n')
                        cursor += 1
                        redraw(buf, cursor)
                        i += 2
                        continue
                
                    # ESC [ sequences
                    if c == 27 and i + 2 < len(k) and k[i+1] == 91:
                        seq = k[i+2]
                        i += 3

                        # Ignore cursor position responses (ESC[<row>;<col>R)
                        if 48 <= seq <= 57:
                            j = i
                            saw_semicolon = False
                            found_dsr = False
                            while j < len(k):
                                b = k[j]
                                if b == 59:
                                    saw_semicolon = True
                                elif b == 82:
                                    if saw_semicolon:
                                        found_dsr = True
                                        j += 1
                                    break
                                elif 48 <= b <= 57:
                                    pass
                                else:
                                    break
                                j += 1
                            if found_dsr:
                                i = j
                                continue
                    
                        # Bracketed paste start: ESC[200~
                        if seq == 50 and i + 2 < len(k) and k[i] == 48 and k[i+1] == 48 and k[i+2] == 126:
                            i += 3
                            paste_content = []
                            paste_end = bytes([27, 91, 50, 48, 49, 126])  # ESC[201~
                            paste_buf = bytearray(k[i:])
                            while paste_end not in paste_buf:
                                paste_buf.extend(os.read(sys.stdin.fileno(), 4096))
                            end_pos = paste_buf.find(paste_end)
                            for b in paste_buf[:end_pos]:
                                if b == 10 or b in _PRINTABLE:
                                    paste_content.append(chr(b))
                            buf[cursor:cursor] = paste_content
                            cursor += len(paste_content)
                            redraw(buf, cursor)
                            i = len(k)
                        elif seq == 68 and cursor > 0:  # Left
                            cursor -= 1
                            redraw(buf, cursor)
                        elif seq == 67 and cursor < len(buf):  # Right
                            cursor += 1
                            redraw(buf, cursor)
                        elif seq == 65:  # Up - history previous
                            if history_idx > 0:
                                if history_idx == len(history):
                                    saved_line = buf[:]
                                # Enter alt for history if not already in alt.
                                if alt_input and not alt_input.active:
                                    alt_input.saved_prev_cursor_row = prev_cursor_row
                                    alt_input.enter(
                                        buf, cursor,
                                        reason=AltInputMode.REASON_HISTORY,
                                        main_cursor_row=prev_cursor_row,
                                    )
                                history_idx -= 1
                                buf = list(history[history_idx])
                                cursor = len(buf)
                                redraw(buf, cursor)
                        elif seq == 66:  # Down - history next
                            if history_idx < len(history):
                                history_idx += 1
                                if history_idx == len(history):
                                    buf = saved_line[:]
                                    cursor = len(buf)
                                    # Only exit alt if we entered it for
                                    # history AND the restored buffer now
                                    # fits on main — otherwise stay in alt.
                                    exited = False
                                    if (alt_input and alt_input.active
                                            and alt_input.reason
                                            == AltInputMode.REASON_HISTORY):
                                        total_rows, cabs = _measure(buf, cursor)
                                        try:
                                            th = os.get_terminal_size().lines
                                        except OSError:
                                            th = 24
                                        budget = max(1, th - 1)
                                        if total_rows <= budget and cabs < budget:
                                            alt_input.exit_silent()
                                            prev_cursor_row = alt_input.saved_prev_cursor_row
                                            _redraw(buf, cursor)
                                            exited = True
                                    if not exited:
                                        redraw(buf, cursor)
                                else:
                                    buf = list(history[history_idx])
                                    cursor = len(buf)
                                    redraw(buf, cursor)
                        elif seq == 72:  # Home - start of current line
                            content = ''.join(buf)
                            line_start = content.rfind('\n', 0, cursor) + 1
                            if cursor != line_start:
                                cursor = line_start
                                redraw(buf, cursor)
                        elif seq == 70:  # End - end of current line
                            content = ''.join(buf)
                            line_end = content.find('\n', cursor)
                            if line_end == -1:
                                line_end = len(buf)
                            if cursor != line_end:
                                cursor = line_end
                                redraw(buf, cursor)
                        elif seq == 51 and i < len(k) and k[i] == 126:  # Delete
                            i += 1
                            if cursor < len(buf):
                                del buf[cursor]
                                redraw(buf, cursor)
                        continue
                
                    # Ctrl+A - start of current line
                    if c == 1:
                        content = ''.join(buf)
                        line_start = content.rfind('\n', 0, cursor) + 1
                        if cursor != line_start:
                            cursor = line_start
                            redraw(buf, cursor)
                        i += 1
                        continue

                    # Ctrl+E - end of current line
                    if c == 5:
                        content = ''.join(buf)
                        line_end = content.find('\n', cursor)
                        if line_end == -1:
                            line_end = len(buf)
                        if cursor != line_end:
                            cursor = line_end
                            redraw(buf, cursor)
                        i += 1
                        continue

                    # Ctrl+K - kill to end of current line
                    if c == 11:
                        content = ''.join(buf)
                        line_end = content.find('\n', cursor)
                        if line_end == -1:
                            line_end = len(buf)
                        del buf[cursor:line_end]
                        redraw(buf, cursor)
                        i += 1
                        continue

                    # Ctrl+U - kill to start of current line
                    if c == 21:
                        content = ''.join(buf)
                        line_start = content.rfind('\n', 0, cursor) + 1
                        del buf[line_start:cursor]
                        cursor = line_start
                        redraw(buf, cursor)
                        i += 1
                        continue

                    # Ctrl+L - clear screen
                    if c == 12:
                        sys.stdout.write('\x1b[2J\x1b[H')
                        redraw(buf, cursor)
                        i += 1
                        continue

                    # Backspace
                    if c in (127, 8) and cursor > 0:
                        cursor -= 1
                        del buf[cursor]
                        redraw(buf, cursor)
                        i += 1
                        continue

                    # Enter - submit
                    if c in (10, 13):
                        line = ''.join(buf)
                        if alt_input and alt_input.active:
                            alt_input.exit(buf, len(buf))
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                        if add_to_history and line.strip() and (not history or history[-1] != line):
                            history.append(line)
                            if on_submit:
                                on_submit(line)
                        return line
                
                    # Printable character
                    if c in _PRINTABLE:
                        ch = chr(c)
                        buf.insert(cursor, ch)
                        cursor += 1
                        if (cursor == len(buf) and '\n' not in buf
                                and not (alt_input and alt_input.active)):
                            # Fast path: typing at end of single-line input
                            # on main screen. Skip full redraw.
                            try:
                                tw = os.get_terminal_size().columns
                            except OSError:
                                tw = 80
                            if tw > 0 and (len(display_prompt) + len(buf)) % tw == 0:
                                # Wrap boundary — full redraw needed.
                                redraw(buf, cursor)
                            else:
                                sys.stdout.write(ch)
                        else:
                            redraw(buf, cursor)
                        i += 1
                        continue
                
                    i += 1

                sys.stdout.flush()
        finally:
            # Ensure we exit alt mode on any exit
            if alt_input:
                alt_input.ensure_exit()
