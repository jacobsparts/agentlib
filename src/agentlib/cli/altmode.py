"""
Alternate screen buffer mode with stdout capture.

Provides AltMode class for capturing stdout while passing through to the real
terminal, and Session class for entering/exiting the alternate screen buffer
with access to captured content.

Example:
    from agentlib.cli.altmode import AltMode
    
    with AltMode() as alt:
        print("This is captured and visible")
        with alt.session() as session:
            # Now in alternate screen buffer
            print(f"Cursor was at: {session.cursor_pos}")
            print(f"Content: {session.content}")
        # Back to main screen, capture resumed
"""

import sys
import os
import termios
import tty
from collections import deque
from typing import Optional, Tuple


class AltMode:
    """Captures stdout while passing through to the real stdout.

    Acts as a file-like object that wraps sys.stdout, capturing all output
    to an internal circular buffer while still displaying it to the terminal.
    Can be used as a context manager for automatic install/uninstall.

    Use session() to create a Session for alternate screen buffer operations.
    """

    def __init__(self, max_lines: int = 500):
        """Initialize capture with a maximum line buffer size."""
        self.max_lines = max_lines
        self._buffer = deque(maxlen=max_lines)
        self._current_line = []
        self._original_stdout = None
        self._installed = False
        self._paused = False

    def install(self) -> 'AltMode':
        """Install the capture, replacing sys.stdout.
        
        Returns:
            Self for method chaining.
        """
        if self._installed:
            return self
        self._original_stdout = sys.stdout
        sys.stdout = self
        self._installed = True
        return self

    def uninstall(self):
        """Restore the original stdout.
        
        Flushes any partial line to the buffer before restoring.
        """
        if not self._installed:
            return
        # Flush any partial line
        if self._current_line:
            self._buffer.append(''.join(self._current_line))
            self._current_line = []
        sys.stdout = self._original_stdout
        self._original_stdout = None
        self._installed = False

    def pause(self):
        """Pause capturing (still passes through to stdout)."""
        self._paused = True

    def resume(self):
        """Resume capturing."""
        self._paused = False

    def write(self, data: str) -> int:
        """Write data to stdout and capture it."""
        if self._original_stdout:
            self._original_stdout.write(data)

        # Capture the data (unless paused)
        if not self._paused:
            for char in data:
                if char == '\n':
                    self._buffer.append(''.join(self._current_line))
                    self._current_line = []
                else:
                    self._current_line.append(char)

        return len(data)

    def flush(self):
        """Flush the underlying stdout."""
        if self._original_stdout:
            self._original_stdout.flush()

    def fileno(self) -> int:
        """Return the file descriptor of the underlying stdout."""
        if self._original_stdout:
            return self._original_stdout.fileno()
        return 1  # stdout fd

    def isatty(self) -> bool:
        """Return whether the underlying stdout is a tty."""
        if self._original_stdout:
            return self._original_stdout.isatty()
        return True

    @property
    def encoding(self) -> str:
        """Return the encoding of the underlying stdout."""
        if self._original_stdout:
            return self._original_stdout.encoding
        return 'utf-8'

    def get_recent(self, num_lines: Optional[int] = None) -> str:
        """Get recent captured output.

        Args:
            num_lines: Number of lines to return. None for all captured lines.

        Returns:
            String containing the captured output with newlines.
        """
        lines = list(self._buffer)
        if num_lines is not None:
            lines = lines[-num_lines:]
        return '\n'.join(lines)

    def get_recent_for_screen(self, term_height: int, reserve_lines: int = 0) -> str:
        """Get captured output sized for screen display.

        Args:
            term_height: Terminal height in lines
            reserve_lines: Lines to reserve (default 0, caller handles input space)

        Returns:
            String containing output that fits on screen.
        """
        available_lines = term_height - reserve_lines
        return self.get_recent(available_lines)

    def clear(self):
        """Clear the capture buffer."""
        self._buffer.clear()
        self._current_line = []

    @property
    def installed(self) -> bool:
        """Whether capture is currently installed."""
        return self._installed

    def __enter__(self) -> 'AltMode':
        """Context manager entry."""
        return self.install()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.uninstall()
        return False

    def session(self) -> 'Session':
        """Create a Session for alternate screen buffer operations.

        The Session captures the current cursor position, enters the alternate
        screen buffer on context entry, and restores the main buffer on exit.

        Returns:
            Session instance bound to this capture.
        """
        return Session(self)



def get_cursor_position() -> Tuple[int, int]:
    """Query terminal for current cursor position.

    Returns:
        (row, col) tuple, 1-indexed. Returns (0, 0) on failure.
    """
    try:
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        try:
            tty.setraw(fd, termios.TCSANOW)
            sys.stdout.write('\x1b[6n')
            sys.stdout.flush()

            # Read response: \x1b[<row>;<col>R
            response = b''
            while True:
                ch = os.read(fd, 1)
                if not ch:
                    return (0, 0)
                response += ch
                if ch == b'R':
                    break
                if len(response) > 20:  # Safety limit
                    return (0, 0)

            # Parse response
            if response.startswith(b'\x1b[') and response.endswith(b'R'):
                parts = response[2:-1].split(b';')
                if len(parts) == 2:
                    return (int(parts[0]), int(parts[1]))
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old_attrs)
    except Exception:
        pass
    return (0, 0)


class Session:
    """Alternate screen buffer session tied to an AltMode instance.

    Captures cursor position on entry, switches to the alternate screen buffer,
    and provides access to the captured content from the main buffer. On exit,
    restores the main screen buffer and resumes capture.

    Always use as a context manager - enter() and exit() are called automatically.
    """

    def __init__(self, capture: AltMode):
        self._capture = capture
        self._cursor_pos: Tuple[int, int] = (0, 0)
        self._active = False

    @property
    def content(self) -> str:
        """Get recent captured stdout content."""
        return self._capture.get_recent()

    @property
    def cursor_pos(self) -> Tuple[int, int]:
        """Cursor position (row, col) from when enter() was called."""
        return self._cursor_pos

    def enter(self) -> bool:
        """Enter the alternate screen buffer.

        Queries cursor position, pauses capture of stdout, and switches to
        the alternate screen buffer. Called automatically when using as a
        context manager.

        Returns:
            True if entered successfully, False if already active.
        """
        if self._active:
            return False
        self._cursor_pos = get_cursor_position()
        self._capture.pause()
        sys.stdout.write('\x1b[?1049h')
        sys.stdout.flush()
        self._active = True
        return True

    def exit(self) -> None:
        """Exit the alternate screen buffer.

        Switches back to main buffer and resumes capture of stdout.
        Called automatically when exiting the context manager.
        """
        if not self._active:
            return
        sys.stdout.write('\x1b[?1049l')
        sys.stdout.flush()
        self._capture.resume()
        self._active = False

    def __enter__(self) -> 'Session':
        """Context manager entry - enters alt buffer and returns self."""
        self.enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - exits alt buffer."""
        self.exit()
        return False
