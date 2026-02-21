"""
SubShell: Subprocess-based shell with streaming output.

Runs bash commands in an isolated subprocess with real-time output
streaming, persistent state, and timeout/interrupt support.

Example:
    shell = SubShell()
    output = shell.execute("echo hello")
    # "hello\n"

    # State persists
    shell.execute("export FOO=bar")
    shell.execute("cd /tmp")
    output = shell.execute("echo $FOO && pwd")
    # "bar\n/tmp\n"

    shell.close()
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Any, Optional


STILL_RUNNING = "[still running]\n"

_python_shim_done = False


def ensure_python_on_path() -> None:
    """If 'python' isn't on PATH but 'python3' is, create a shim symlink."""
    global _python_shim_done
    if _python_shim_done:
        return
    _python_shim_done = True
    python3 = shutil.which("python3")
    if shutil.which("python") or not python3:
        return
    shim_dir = Path.home() / ".agentlib" / "shims"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "python"
    # Recreate symlink in case python3 path changed
    shim.unlink(missing_ok=True)
    shim.symlink_to(python3)
    os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")


def _with_still_running(output: str) -> str:
    """Append STILL_RUNNING marker, ensuring proper newline."""
    if output and not output.endswith('\n'):
        output += '\n'
    return output + STILL_RUNNING


def _worker_main(cmd_queue: Queue, output_queue: Queue) -> None:
    """Worker process that executes commands in a persistent bash shell."""
    ensure_python_on_path()

    # Start bash process
    proc = subprocess.Popen(
        ["bash", "--norc", "--noprofile"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Unique marker for command completion
    marker = "__SUBSHELL_DONE_a7b3f9__"

    def read_output_thread(proc: subprocess.Popen, output_queue: Queue, stop_event: threading.Event):
        """Thread to read stdout and push to queue."""
        try:
            while not stop_event.is_set():
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        output_queue.put(("exit", proc.returncode))
                        break
                    continue
                output_queue.put(("line", line))
        except Exception as e:
            output_queue.put(("error", str(e)))

    stop_event = threading.Event()
    reader_thread = threading.Thread(
        target=read_output_thread,
        args=(proc, output_queue, stop_event),
        daemon=True
    )
    reader_thread.start()

    while True:
        try:
            cmd = cmd_queue.get(timeout=0.5)
        except Empty:
            if proc.poll() is not None:
                break
            continue
        except (KeyboardInterrupt, EOFError):
            break

        if cmd is None:
            break

        try:
            # Send command followed by marker echo
            full_cmd = f"{cmd}\necho '{marker}'\n"
            proc.stdin.write(full_cmd)
            proc.stdin.flush()

            # Signal that command was sent
            output_queue.put(("started", None))
        except Exception as e:
            output_queue.put(("error", str(e)))

    # Cleanup
    stop_event.set()
    proc.terminate()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()


class SubShell:
    """
    Subprocess-based shell with streaming output.

    Executes commands in an isolated bash process. Environment and
    working directory persist across executions.
    """

    def __init__(self, echo: bool = True) -> None:
        """Initialize SubShell.

        Args:
            echo: If True, prefix output with "$ command" echo (default True)
        """
        self._cmd_queue: Optional[Queue] = None
        self._output_queue: Optional[Queue] = None
        self._worker: Optional[Process] = None
        self._running: bool = False
        self._marker = "__SUBSHELL_DONE_a7b3f9__"
        self._echo: bool = echo

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_session(self) -> None:
        """Start worker if needed."""
        if self._worker is None or not self._worker.is_alive():
            self._cmd_queue = Queue(maxsize=1)
            self._output_queue = Queue(maxsize=1)
            self._worker = Process(
                target=_worker_main,
                args=(self._cmd_queue, self._output_queue),
                daemon=True
            )
            self._worker.start()
            self._running = False

    def execute(
        self,
        cmd: str,
        timeout: float = 10.0,
        hard_timeout: bool = False
    ) -> str:
        """
        Execute a shell command.

        Args:
            cmd: Command to execute
            timeout: Max seconds to wait
            hard_timeout: If True, interrupt on timeout

        Returns:
            Output string with each command echoed as "$ command".
            Ends with "[still running]\\n" if final command not complete.
            If previous command still running, returns warning instead of executing.
        """
        # Build echo prefix from individual lines
        lines = [l for l in cmd.split('\n') if l.strip()]
        if not lines:
            return ""

        if self._echo:
            prefix = ''.join(f"$ {line}\n" for line in lines)
        else:
            prefix = ""

        if self._running:
            pending = self.read(timeout=0).removesuffix(STILL_RUNNING)
            return pending + prefix + "[Previous command still running. Use read(), interrupt(), or terminate() first.]\n"

        self._ensure_session()
        self._running = True
        # Send all commands as one batch (shell executes them sequentially)
        self._cmd_queue.put('\n'.join(lines))

        # Wait for started signal
        try:
            msg_type, _ = self._output_queue.get(timeout=5.0)
            if msg_type != "started":
                pass  # Handle in read()
        except Empty:
            self._running = False
            return prefix + "[Failed to start command]\n"

        return prefix + self.read(timeout=timeout, hard_timeout=hard_timeout)

    def read(self, timeout: float = 10.0, hard_timeout: bool = False) -> str:
        """Read output from running command."""
        if not self._running:
            return ""

        output_chunks: list[str] = []
        deadline = time.time() + timeout

        while True:
            remaining = deadline - time.time()

            if remaining <= 0:
                if hard_timeout:
                    return self.interrupt()
                return _with_still_running("".join(output_chunks))

            try:
                msg_type, msg_data = self._output_queue.get(timeout=min(remaining, 0.1))

                if msg_type == "line":
                    # Check for completion marker
                    if self._marker in msg_data:
                        self._running = False
                        return "".join(output_chunks)
                    output_chunks.append(msg_data)

                elif msg_type == "exit":
                    self._running = False
                    return "".join(output_chunks) + f"\n[Shell exited: {msg_data}]\n"

                elif msg_type == "error":
                    self._running = False
                    return "".join(output_chunks) + f"\n[Error: {msg_data}]\n"

            except Empty:
                continue

    def interrupt(self) -> str:
        """Interrupt running command with SIGINT."""
        if not self._running or self._worker is None:
            return ""

        try:
            os.kill(self._worker.pid, signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass

        output_chunks: list[str] = []

        for _ in range(30):
            try:
                msg_type, msg_data = self._output_queue.get(timeout=0.1)
                if msg_type == "line":
                    if self._marker in msg_data:
                        self._running = False
                        return "".join(output_chunks)
                    output_chunks.append(msg_data)
                elif msg_type in ("exit", "error"):
                    self._running = False
                    return "".join(output_chunks)
            except Empty:
                if self._worker and not self._worker.is_alive():
                    break

        self._running = False
        return "".join(output_chunks)

    def terminate(self) -> Optional[str]:
        """Kill the session."""
        if self._worker is None:
            return None

        output_chunks: list[str] = []

        if self._output_queue:
            while True:
                try:
                    msg_type, msg_data = self._output_queue.get_nowait()
                    if msg_type == "line":
                        output_chunks.append(msg_data)
                except Empty:
                    break

        # Signal worker to exit gracefully
        if self._cmd_queue:
            try:
                self._cmd_queue.put(None)
            except Exception:
                pass

        if self._worker.is_alive():
            self._worker.join(timeout=0.5)
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=0.5)

        was_running = self._running
        self._worker = None
        self._cmd_queue = None
        self._output_queue = None
        self._running = False

        return "".join(output_chunks) if was_running else None

    def close(self) -> None:
        """Clean up."""
        self.terminate()

    def __enter__(self) -> SubShell:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
