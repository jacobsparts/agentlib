"""
Pure Python readline replacement with bracketed paste support.

Provides terminal input handling without requiring GNU readline >= 8.0.
"""

import sys
import os
import termios
from typing import Optional, Callable

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
    
    # Strip leading newlines from prompt - they're only for initial display
    display_prompt = prompt_str.lstrip('\n')
    display_continuation = continuation_str.lstrip('\n')
    
    def _redraw(buf, cursor):
        nonlocal prev_lines
        
        try:
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80
        
        out = ['\x1b[?25l']  # Hide cursor
        if prev_lines > 1:
            out.append(f'\x1b[{prev_lines - 1}A')
        out.append('\r')
        
        content = ''.join(buf)
        lines = content.split('\n')
        
        # Calculate physical lines each logical line takes (accounting for terminal wrap)
        line_physical_counts = []
        for i, line in enumerate(lines):
            prefix_len = len(display_prompt) if i == 0 else len(display_continuation)
            line_len = prefix_len + len(line)
            line_physical_counts.append(max(1, (line_len + term_width - 1) // term_width))
        
        prev_lines = sum(line_physical_counts)
        
        for i, line in enumerate(lines):
            prefix = display_prompt if i == 0 else display_continuation
            out.append(f'{prefix}{line}\x1b[K')
            if i < len(lines) - 1:
                out.append('\n')
        
        out.append('\x1b[J')  # Clear to end of screen
        
        # Find which logical line the cursor is on
        pos = 0
        cursor_line = 0
        cursor_col = 0
        for i, line in enumerate(lines):
            if pos + len(line) >= cursor or i == len(lines) - 1:
                cursor_line = i
                cursor_col = cursor - pos
                break
            pos += len(line) + 1
        
        # Calculate physical cursor position
        prefix_len = len(display_prompt) if cursor_line == 0 else len(display_continuation)
        cursor_display_col = prefix_len + cursor_col
        cursor_phys_row = cursor_display_col // term_width
        cursor_phys_col = cursor_display_col % term_width
        
        # Physical lines from cursor to end of content
        remaining_in_current = line_physical_counts[cursor_line] - cursor_phys_row - 1
        physical_lines_after = remaining_in_current + sum(line_physical_counts[cursor_line + 1:])
        
        if physical_lines_after > 0:
            out.append(f'\x1b[{physical_lines_after}A')
        
        out.append('\r')
        if cursor_phys_col > 0:
            out.append(f'\x1b[{cursor_phys_col}C')
        
        out.append('\x1b[?25h')  # Show cursor
        sys.stdout.write(''.join(out))
        sys.stdout.flush()

    with RawMode():
        sys.stdout.write(prompt_str)
        sys.stdout.flush()
        buf = []
        cursor = 0
        history_idx = len(history)
        saved_line = []
        
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
                    _redraw(buf, cursor)
                    i += 2
                    continue
                
                # ESC [ sequences
                if c == 27 and i + 2 < len(k) and k[i+1] == 91:
                    seq = k[i+2]
                    i += 3
                    
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
                        _redraw(buf, cursor)
                        i = len(k)
                    elif seq == 68 and cursor > 0:  # Left
                        cursor -= 1
                        _redraw(buf, cursor)
                    elif seq == 67 and cursor < len(buf):  # Right
                        cursor += 1
                        _redraw(buf, cursor)
                    elif seq == 65:  # Up - history previous
                        if history_idx > 0:
                            if history_idx == len(history):
                                saved_line = buf[:]
                            history_idx -= 1
                            buf = list(history[history_idx])
                            cursor = len(buf)
                            _redraw(buf, cursor)
                    elif seq == 66:  # Down - history next
                        if history_idx < len(history):
                            history_idx += 1
                            if history_idx == len(history):
                                buf = saved_line[:]
                            else:
                                buf = list(history[history_idx])
                            cursor = len(buf)
                            _redraw(buf, cursor)
                    elif seq == 72:  # Home
                        if cursor > 0:
                            cursor = 0
                            _redraw(buf, cursor)
                    elif seq == 70:  # End
                        if cursor < len(buf):
                            cursor = len(buf)
                            _redraw(buf, cursor)
                    elif seq == 51 and i < len(k) and k[i] == 126:  # Delete
                        i += 1
                        if cursor < len(buf):
                            del buf[cursor]
                            _redraw(buf, cursor)
                    continue
                
                # Ctrl+A - start of line
                if c == 1:
                    if cursor > 0:
                        cursor = 0
                        _redraw(buf, cursor)
                    i += 1
                    continue
                
                # Ctrl+E - end of line
                if c == 5:
                    if cursor < len(buf):
                        cursor = len(buf)
                        _redraw(buf, cursor)
                    i += 1
                    continue
                
                # Ctrl+K - kill to end
                if c == 11:
                    del buf[cursor:]
                    _redraw(buf, cursor)
                    i += 1
                    continue
                
                # Ctrl+U - kill to start
                if c == 21:
                    del buf[:cursor]
                    cursor = 0
                    _redraw(buf, cursor)
                    i += 1
                    continue
                
                # Ctrl+L - clear screen
                if c == 12:
                    sys.stdout.write('\x1b[2J\x1b[H')
                    _redraw(buf, cursor)
                    i += 1
                    continue
                
                # Backspace
                if c in (127, 8) and cursor > 0:
                    cursor -= 1
                    del buf[cursor]
                    _redraw(buf, cursor)
                    i += 1
                    continue
                
                # Enter - submit
                if c in (10, 13):
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    line = ''.join(buf)
                    if add_to_history and line and (not history or history[-1] != line):
                        history.append(line)
                        if on_submit:
                            on_submit(line)
                    return line
                
                # Printable character
                if c in _PRINTABLE:
                    ch = chr(c)
                    buf.insert(cursor, ch)
                    cursor += 1
                    if cursor == len(buf) and '\n' not in buf:
                        # Fast path: check if we wrapped to a new physical line
                        try:
                            tw = os.get_terminal_size().columns
                        except OSError:
                            tw = 80
                        if tw > 0 and (len(display_prompt) + len(buf)) % tw == 0:
                            prev_lines += 1
                        sys.stdout.write(ch)
                    else:
                        _redraw(buf, cursor)
                    i += 1
                    continue
                
                i += 1
            
            sys.stdout.flush()
