"""
SubREPL: Subprocess-based Python REPL with streaming output.

A production-ready REPL environment that runs Python code in an isolated
worker process with real-time output streaming and persistent state.

Example:
    repl = SubREPL()

    # Simple execution
    output = repl.execute("print('hello')")
    # "hello\n"

    # Long-running with polling
    output = repl.execute("import time; time.sleep(30)", timeout=5.0)
    # "partial output\n[still running]\n"
    if output.endswith("[still running]\n"):
        output = repl.interrupt()

    # Hard timeout with auto-interrupt
    output = repl.execute("while True: pass", timeout=5.0, hard_timeout=True)
    # Automatically interrupted

    # Clean up
    repl.close()
"""

from __future__ import annotations

import code
import os
import signal
import sys
import time
from codeop import compile_command
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, Optional


STILL_RUNNING = "[still running]\n"


def _with_still_running(output: str) -> str:
    """Append STILL_RUNNING marker, ensuring proper newline."""
    if output and not output.endswith('\n'):
        output += '\n'
    return output + STILL_RUNNING


def _split_into_statements(source: str) -> list[str]:
    """Split Python source into complete statements for REPL execution."""
    # Keywords that continue a previous compound statement
    CONTINUATION_KEYWORDS = ('else', 'elif', 'except', 'finally', 'case')

    lines = source.split('\n')
    statements = []
    current = []

    for line in lines:
        is_indented = line.startswith((' ', '\t'))
        stripped = line.strip()

        # Check if this line is a continuation keyword (else:, elif x:, except:, etc)
        is_continuation = any(
            stripped == kw + ':' or stripped.startswith(kw + ' ') or stripped.startswith(kw + ':')
            for kw in CONTINUATION_KEYWORDS
        )

        # When we see a non-indented, non-empty, non-continuation line
        # and have accumulated code, check if accumulated code is complete
        if not is_indented and stripped and not is_continuation and current:
            current_src = '\n'.join(current)
            try:
                # Double newline signals end of any indented block
                if compile_command(current_src + '\n\n') is not None:
                    statements.append(current_src)
                    current = []
            except (SyntaxError, OverflowError, ValueError):
                # Syntax error - save as-is, will error on exec
                statements.append(current_src)
                current = []

        # Add non-empty lines (keep indented structure)
        if stripped or (current and is_indented):
            current.append(line)

    # Handle remaining code
    if current:
        statements.append('\n'.join(current))

    return [s for s in statements if s.strip()]


def _format_echo(stmt: str) -> str:
    """Format a statement with REPL-style echo prefix."""
    lines = stmt.split('\n')
    result = [f">>> {lines[0]}"]
    for line in lines[1:]:
        result.append(f"... {line}")
    return '\n'.join(result) + '\n'


class _StreamingWriter:
    """
    Custom writer that sends output to a queue in real-time.
    Replaces sys.stdout/stderr in the worker process.
    """

    def __init__(self, queue: Queue, original: Any) -> None:
        self._queue = queue
        self._original = original

    def write(self, text: str) -> int:
        if text:
            self._queue.put(("output", text))
        return len(text)

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        return self._original.fileno()


def _worker_main(cmd_queue: Queue, output_queue: Queue) -> None:
    """
    Worker process entry point.
    """
    repl_locals: dict[str, Any] = {}

    def sigint_handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, sigint_handler)

    while True:
        try:
            cmd = cmd_queue.get()

            if cmd is None:
                break

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = _StreamingWriter(output_queue, old_stdout)
            sys.stderr = _StreamingWriter(output_queue, old_stderr)

            try:
                try:
                    compiled = code.compile_command(cmd, "<repl>", "exec")
                    if compiled is not None:
                        exec(compiled, repl_locals)
                    else:
                        exec(cmd, repl_locals)
                except SyntaxError as e:
                    sys.stderr.write(f"  File \"<repl>\", line {e.lineno}\n")
                    if e.text:
                        sys.stderr.write(f"    {e.text}")
                        if e.offset:
                            sys.stderr.write(" " * (e.offset + 3) + "^\n")
                    sys.stderr.write(f"SyntaxError: {e.msg}\n")

            except KeyboardInterrupt:
                sys.stderr.write("\nKeyboardInterrupt\n")
            except Exception as e:
                import traceback
                # Filter traceback to only show frames from <repl>, not our internals
                tb = e.__traceback__
                # Skip frames until we reach <repl>
                while tb is not None and tb.tb_frame.f_code.co_filename != "<repl>":
                    tb = tb.tb_next
                if tb is not None:
                    sys.stderr.write("Traceback (most recent call last):\n")
                    sys.stderr.write("".join(traceback.format_tb(tb)))
                sys.stderr.write(f"{type(e).__name__}: {e}\n")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            output_queue.put(("done", None))

        except KeyboardInterrupt:
            output_queue.put(("output", "\nKeyboardInterrupt\n"))
            output_queue.put(("done", None))


class SubREPL:
    """
    Subprocess-based Python REPL with streaming output.

    Executes Python code in an isolated worker process. State persists
    across executions (variables, imports, function definitions).
    Output streams in real-time rather than buffering until completion.

    Sessions are created lazily on first execute() call.

    Returns:
        All methods return a string. If execution is still running,
        the string ends with "[still running]\\n".
    """

    def __init__(self, echo: bool = True) -> None:
        """Initialize SubREPL. Worker is not started until first execute().

        Args:
            echo: If True, prefix output with ">>> statement" echo (default True)
        """
        self._cmd_queue: Optional[Queue] = None
        self._output_queue: Optional[Queue] = None
        self._worker: Optional[Process] = None
        self._running: bool = False
        self._echo: bool = echo

    def __del__(self) -> None:
        """Clean up worker process on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def _ensure_session(self) -> None:
        """Lazily create worker session if not exists."""
        if self._worker is None or not self._worker.is_alive():
            self._cmd_queue = Queue()
            self._output_queue = Queue()
            self._worker = Process(
                target=_worker_main,
                args=(self._cmd_queue, self._output_queue),
                daemon=True
            )
            self._worker.start()
            self._running = False

    def execute(
        self,
        code: str,
        timeout: float = 10.0,
        hard_timeout: bool = False
    ) -> str:
        """
        Execute code and return output.

        Args:
            code: Python code to execute
            timeout: Max seconds to wait for result (default 10.0)
            hard_timeout: If True and timeout reached, interrupt execution

        Returns:
            Output string with each statement echoed as ">>> statement" (and
            "... continuation" for multi-line statements).
            Ends with "[still running]\\n" if final statement not complete.
            If previous command still running, returns warning instead of executing.
        """
        # Split into complete statements for echo display
        statements = _split_into_statements(code)
        if not statements:
            return ""

        if self._echo:
            prefix = ''.join(_format_echo(stmt) for stmt in statements)
        else:
            prefix = ""

        if self._running:
            pending = self.read(timeout=0).removesuffix(STILL_RUNNING)
            return pending + prefix + "[Previous command still running. Use read(), interrupt(), or terminate() first.]\n"

        self._ensure_session()
        self._running = True
        # Send all statements as one batch
        self._cmd_queue.put('\n'.join(statements))

        return prefix + self.read(timeout=timeout, hard_timeout=hard_timeout)

    def read(
        self,
        timeout: float = 10.0,
        hard_timeout: bool = False
    ) -> str:
        """
        Read output from running execution.

        Args:
            timeout: Max seconds to wait (default 10.0)
            hard_timeout: If True and timeout reached, interrupt execution

        Returns:
            Output string. Ends with "[still running]\\n" if not complete.
        """
        if not self._running:
            return ""

        if self._output_queue is None:
            raise RuntimeError("No active session")

        output_chunks: list[str] = []
        deadline = time.time() + timeout

        # Always drain available output first (non-blocking)
        while True:
            try:
                msg_type, msg_data = self._output_queue.get_nowait()
                if msg_type == "output":
                    output_chunks.append(msg_data)
                elif msg_type == "done":
                    self._running = False
                    return "".join(output_chunks)
            except Empty:
                break

        # If timeout=0, return immediately with what we have
        if timeout <= 0:
            if hard_timeout:
                return self._escalating_interrupt(output_chunks)
            else:
                return _with_still_running("".join(output_chunks))

        # Wait for more output until deadline
        while True:
            remaining = deadline - time.time()

            if remaining <= 0:
                if hard_timeout:
                    return self._escalating_interrupt(output_chunks)
                else:
                    return _with_still_running("".join(output_chunks))

            try:
                msg_type, msg_data = self._output_queue.get(timeout=min(remaining, 0.1))

                if msg_type == "output":
                    output_chunks.append(msg_data)
                elif msg_type == "done":
                    self._running = False
                    return "".join(output_chunks)

            except Empty:
                continue

    def _escalating_interrupt(self, output_chunks: list[str]) -> str:
        """Interrupt with escalating signals: SIGINT (x3) -> SIGKILL."""
        if not self._running or self._worker is None:
            return ""

        # Try SIGINT up to 3 times
        for _ in range(3):
            try:
                os.kill(self._worker.pid, signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass

            result = self._drain_and_wait(output_chunks, timeout=1.0)
            if result is not None:
                return result

        # Nuclear option: SIGKILL
        try:
            os.kill(self._worker.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

        self._worker.join(timeout=1.0)
        self._running = False

        output = "".join(output_chunks) + "\n[Process killed]\n"

        # Session destroyed
        self._worker = None
        self._cmd_queue = None
        self._output_queue = None

        return output

    def _drain_and_wait(
        self,
        output_chunks: list[str],
        timeout: float
    ) -> Optional[str]:
        """Drain output queue and wait for completion."""
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                msg_type, msg_data = self._output_queue.get(timeout=0.05)

                if msg_type == "output":
                    output_chunks.append(msg_data)
                elif msg_type == "done":
                    self._running = False
                    return "".join(output_chunks)
            except Empty:
                continue

        return None

    def interrupt(self) -> str:
        """
        Interrupt running execution with SIGINT (like Ctrl+C).

        Returns:
            Output string (always complete after interrupt).
        """
        if not self._running or self._worker is None:
            return ""

        try:
            os.kill(self._worker.pid, signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass

        output_chunks: list[str] = []

        while True:
            try:
                msg_type, msg_data = self._output_queue.get(timeout=0.1)

                if msg_type == "output":
                    output_chunks.append(msg_data)
                elif msg_type == "done":
                    self._running = False
                    return "".join(output_chunks)
            except Empty:
                if not self._worker.is_alive():
                    self._running = False
                    return "".join(output_chunks) + "\n[Process died]\n"

    def terminate(self) -> Optional[str]:
        """
        Kill the session immediately with SIGKILL.

        Returns:
            Output string, or None if no session active.
        """
        if self._worker is None:
            return None

        output_chunks: list[str] = []

        if self._output_queue is not None:
            while True:
                try:
                    msg_type, msg_data = self._output_queue.get_nowait()
                    if msg_type == "output":
                        output_chunks.append(msg_data)
                except Empty:
                    break

        if self._worker.is_alive():
            try:
                os.kill(self._worker.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            self._worker.join(timeout=1.0)

        was_running = self._running
        result = "".join(output_chunks) if was_running else None

        self._worker = None
        self._cmd_queue = None
        self._output_queue = None
        self._running = False

        return result

    def close(self) -> None:
        """Clean up the session."""
        self.terminate()

    def __enter__(self) -> SubREPL:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
