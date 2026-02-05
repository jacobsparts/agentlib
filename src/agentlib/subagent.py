"""
Subagent spawning for Code Agent.

Spawn isolated Code Agent instances as subprocesses with socket communication.
Reuses the socket protocol pattern from SandboxMixin.

## Quick Start

    from agentlib.subagent import Subagent

    # Create and send a task
    agent = Subagent(cwd="/path/to/project")
    response = agent.send("Fix the bug in main.py line 42")
    print(response.result)

## Background Execution

    # Start task in background
    response = agent.send("Refactor the database module", bg=True)

    # Check progress
    for update in response.progress:
        print(update)

    # Wait for completion
    response.wait()
    print(response.result)

## Multiple Parallel Agents

    agents = [Subagent() for _ in range(3)]
    tasks = ["Fix bug in a.py", "Fix bug in b.py", "Fix bug in c.py"]

    # Start all in background
    responses = [a.send(t, bg=True) for a, t in zip(agents, tasks)]

    # Wait for all
    for r in responses:
        r.wait()
        print(r.result)

## Session Continuity

    agent = Subagent()
    agent.send("Read main.py and understand the structure")
    agent.send("Now add error handling to the parse function")  # Follows up

## Model Configuration

    # Subagents inherit the parent's model by default
    agent = Subagent()  # Uses parent's model

    # Or specify a different model
    agent = Subagent(model="anthropic/claude-haiku-3-5")

## Attributes

    Subagent:
        .id         - Unique identifier
        .cwd        - Working directory
        .model      - LLM model being used
        .done       - Whether last task is complete
        .result     - Result from last task
        .send()     - Send a task
        .wait()     - Wait for last task
        .kill()     - Kill the subprocess

    SubagentResponse:
        .done       - Whether task is complete
        .result     - Result text
        .progress   - List of progress updates (emit with release=False)
        .is_error   - Whether an error occurred
        .wait()     - Block until complete
"""

import fcntl
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid
from typing import Any, Optional

try:
    import cloudpickle as pickle
except ImportError:
    import pickle


# ---------------------------------------------------------------------------
# Socket protocol (shared with sandbox)
# ---------------------------------------------------------------------------

def _send_msg(sock: socket.socket, data: Any):
    """Send pickled message with length prefix."""
    payload = pickle.dumps(data)
    sock.sendall(struct.pack('!I', len(payload)) + payload)


def _recv_msg(sock: socket.socket, timeout: Optional[float] = None) -> Any:
    """Receive pickled message with length prefix."""
    if timeout is not None:
        sock.settimeout(timeout)
    else:
        sock.settimeout(None)

    raw_len = b''
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Connection closed")
        raw_len += chunk

    msg_len = struct.unpack('!I', raw_len)[0]

    chunks = []
    remaining = msg_len
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("Connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)

    return pickle.loads(b''.join(chunks))


# ---------------------------------------------------------------------------
# Worker code (runs in subprocess)
# ---------------------------------------------------------------------------

WORKER_CODE = '''
import os
import signal
import socket
import struct
import sys

try:
    import cloudpickle as pickle
except ImportError:
    import pickle


def _send_msg(sock, data):
    payload = pickle.dumps(data)
    sock.sendall(struct.pack('!I', len(payload)) + payload)


def _recv_msg(sock, timeout=None):
    if timeout is not None:
        sock.settimeout(timeout)
    else:
        sock.settimeout(None)
    raw_len = b''
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Connection closed")
        raw_len += chunk
    msg_len = struct.unpack('!I', raw_len)[0]
    chunks = []
    remaining = msg_len
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("Connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return pickle.loads(b''.join(chunks))


def worker_main(port, authkey, model, max_turns):
    # Request SIGTERM when parent dies (Linux-specific)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass

    # Connect to host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60)
    sock.connect(('127.0.0.1', port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Authenticate
    _send_msg(sock, authkey)
    ack = _recv_msg(sock)
    if ack != 'ok':
        raise RuntimeError("Authentication failed")

    # Import agent classes
    from agentlib.agents.code_agent import CodeAgentBase

    class SubagentWorker(CodeAgentBase):
        """Headless Code Agent for subprocess execution."""

        interactive = True
        welcome_message = ""

        def __init__(self, host_sock, model_name, default_max_turns):
            self._host_sock = host_sock
            self.model = model_name
            self.max_turns = default_max_turns
            super().__init__()

        # Disable CLI display hooks
        def on_repl_execute(self, code):
            pass

        def on_repl_chunk(self, chunk, msg_type="echo"):
            pass

        def on_repl_output(self, output_chunks):
            pass

        def on_statement_output(self, statement_chunks):
            pass

        def _handle_tool_request(self, repl, req):
            """Override to send progress/result via socket."""
            tool_name = req.get('tool')
            request_id = req.get('request_id')
            args = req.get('args', {})

            # Deserialize special types (bytes encoded as base64)
            def _deserialize(x):
                if isinstance(x, dict) and "__b64__" in x:
                    import base64
                    return base64.b64decode(x["__b64__"])
                if isinstance(x, list):
                    return [_deserialize(i) for i in x]
                if isinstance(x, dict):
                    return {k: _deserialize(v) for k, v in x.items()}
                return x

            args = {k: _deserialize(v) for k, v in args.items()}

            try:
                if tool_name == '__emit__':
                    value = args.get('value')
                    release = args.get('release', False)
                    self._final_result = value

                    if release:
                        self.complete = True
                        _send_msg(self._host_sock, ("result", str(value) if value is not None else ""))
                    else:
                        _send_msg(self._host_sock, ("progress", str(value) if value is not None else ""))
                    # No reply needed for emit, just ACK

                else:
                    # Normal tool call - send reply with result
                    from agentlib.agent import _CompleteException
                    try:
                        result = self.toolcall(tool_name, args)
                        repl.send_reply(request_id, result=result)
                    except _CompleteException:
                        raise
                    except Exception as e:
                        repl.send_reply(request_id, error=str(e))
            finally:
                # Always send ACK to unblock the sender
                repl.send_ack(request_id)

    # Create agent
    agent = SubagentWorker(sock, model, max_turns)

    # Main loop - receive tasks
    while True:
        try:
            msg = _recv_msg(sock, timeout=300)

            if msg is None:
                break

            cmd_type, cmd_data = msg

            if cmd_type == "task":
                prompt = cmd_data.get("prompt", "")
                task_max_turns = cmd_data.get("max_turns", max_turns)

                try:
                    agent.usermsg(prompt)
                    result = agent.run_loop(max_turns=task_max_turns)

                    # If loop exited without emit(release=True), send result
                    if not agent.complete:
                        result_str = str(result) if result is not None else ""
                        _send_msg(sock, ("result", result_str))

                    # Reset for next task
                    agent.complete = False
                    agent._final_result = None

                except KeyboardInterrupt:
                    _send_msg(sock, ("error", "Task interrupted"))
                except Exception as e:
                    import traceback
                    _send_msg(sock, ("error", f"{type(e).__name__}: {e}\\n{traceback.format_exc()}"))

            elif cmd_type == "shutdown":
                break

        except socket.timeout:
            continue
        except ConnectionError:
            break

    sock.close()
'''


# ---------------------------------------------------------------------------
# SubagentError
# ---------------------------------------------------------------------------

class SubagentError(Exception):
    """Raised when a subagent returns an error."""

    def __init__(self, message: str, response: 'SubagentResponse'):
        super().__init__(message)
        self.response = response


# ---------------------------------------------------------------------------
# SubagentResponse
# ---------------------------------------------------------------------------

class SubagentResponse:
    """Result of a Subagent.send() call.

    Attributes:
        done: Whether the task has completed
        result: The result text (empty until done)
        progress: List of progress updates from emit(release=False)
        is_error: Whether the task resulted in an error
    """

    def __init__(self, agent: 'Subagent'):
        self._agent = agent
        self._result: Optional[str] = None
        self._error: Optional[str] = None
        self._done = False
        self._progress: list[str] = []

    @property
    def done(self) -> bool:
        """Check if the task has completed."""
        if self._done:
            return True
        self._agent._poll()
        return self._done

    @property
    def result(self) -> str:
        """The result text. Empty string if not yet complete."""
        if not self.done:
            return ""
        return self._result or ""

    @property
    def progress(self) -> list[str]:
        """Progress updates received so far."""
        self._agent._poll()
        return list(self._progress)

    @property
    def is_error(self) -> bool:
        """Whether the task resulted in an error."""
        return self._error is not None

    def wait(self, timeout: Optional[float] = None) -> 'SubagentResponse':
        """Wait for the task to complete.

        Args:
            timeout: Maximum seconds to wait. None = wait indefinitely.

        Returns:
            self, for chaining.

        Raises:
            SubagentError: If the task resulted in an error.
        """
        start = time.time()
        while not self.done:
            if timeout is not None and (time.time() - start) > timeout:
                break
            time.sleep(0.1)

        if self.done and self.is_error:
            raise SubagentError(self._error, self)

        return self

    def __repr__(self) -> str:
        if not self._done:
            self._agent._poll()
        if not self._done:
            progress_info = f", {len(self._progress)} updates" if self._progress else ""
            return f"[SubagentResponse: running{progress_info}]"
        if self.is_error:
            return f"[SubagentResponse: error] {self._error}"
        # Truncate long results for repr
        r = self._result or ""
        if len(r) > 100:
            r = r[:100] + "..."
        return r if r else "[SubagentResponse: empty]"


# ---------------------------------------------------------------------------
# Subagent
# ---------------------------------------------------------------------------

# Global registry of subagents
_subagents: dict[str, 'Subagent'] = {}


class Subagent:
    """A Code Agent running in an isolated subprocess.

    Each Subagent maintains its own session with persistent state.
    Follow-up tasks share conversation context.

    Args:
        cwd: Working directory for the agent. Defaults to current directory.
        model: LLM model to use. Defaults to anthropic/claude-sonnet-4-5.
        max_turns: Maximum turns per task. Default 50.

    Example:
        agent = Subagent(cwd="/path/to/project")
        response = agent.send("Fix the bug in main.py")
        print(response.result)

        # Follow-up in same session
        response = agent.send("Now add tests")
    """

    # Default model, can be set by parent agent via /subagents command
    default_model: Optional[str] = None

    def __init__(
        self,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 50
    ):
        self.id = str(uuid.uuid4())[:8]
        self.cwd = cwd or os.getcwd()
        # Use explicit model, or class default (set by parent via /subagents)
        self.model = model or Subagent.default_model
        self.max_turns = max_turns

        self._proc: Optional[subprocess.Popen] = None
        self._conn: Optional[socket.socket] = None
        self._server: Optional[socket.socket] = None
        self._current_response: Optional[SubagentResponse] = None
        self._started = False

        # Register globally
        _subagents[self.id] = self

    def _ensure_started(self) -> None:
        """Start the subprocess if not already running."""
        if self._started and self._proc and self._proc.poll() is None:
            return

        if self._proc:
            self._cleanup()

        authkey = os.urandom(16)

        # Create server socket
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('127.0.0.1', 0))
        self._server.listen(1)
        port = self._server.getsockname()[1]
        self._server.settimeout(30)

        # Bootstrap code - include parent's sys.path so subprocess can find agentlib
        worker_bootstrap = f'''
import sys
sys.path = {repr(sys.path)}
exec({repr(WORKER_CODE)})
worker_main({port}, bytes.fromhex({repr(authkey.hex())}), {repr(self.model)}, {self.max_turns})
'''

        # Start subprocess
        self._proc = subprocess.Popen(
            [sys.executable, '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            start_new_session=True,
        )

        self._proc.stdin.write(worker_bootstrap.encode())
        self._proc.stdin.close()

        # Accept connection
        try:
            self._conn, _ = self._server.accept()
            self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except socket.timeout:
            self._proc.kill()
            stdout, stderr = self._proc.communicate()
            raise TimeoutError(f"Subagent failed to connect. stderr: {stderr.decode()}")
        finally:
            self._server.close()
            self._server = None

        # Authenticate
        client_key = _recv_msg(self._conn)
        if client_key != authkey:
            self._conn.close()
            self._proc.kill()
            raise RuntimeError("Subagent authentication failed")
        _send_msg(self._conn, 'ok')

        # Set socket to non-blocking for polling
        fd = self._conn.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        self._started = True

    def _poll(self) -> None:
        """Poll for messages from subprocess (non-blocking)."""
        if not self._conn or not self._current_response:
            return

        if self._current_response._done:
            return

        # Check if process died
        if self._proc and self._proc.poll() is not None:
            self._current_response._error = "Subagent process died unexpectedly"
            self._current_response._done = True
            return

        while True:
            try:
                msg = _recv_msg(self._conn, timeout=0.001)

                msg_type, msg_data = msg

                if msg_type == "progress":
                    self._current_response._progress.append(msg_data)
                elif msg_type == "result":
                    self._current_response._result = msg_data
                    self._current_response._done = True
                    break
                elif msg_type == "error":
                    self._current_response._error = msg_data
                    self._current_response._done = True
                    break
            except socket.timeout:
                break
            except BlockingIOError:
                break
            except ConnectionError:
                self._current_response._error = "Connection lost"
                self._current_response._done = True
                break

    def send(
        self,
        prompt: str,
        *,
        bg: bool = False,
        max_turns: Optional[int] = None,
        timeout: float = 1800
    ) -> SubagentResponse:
        """Send a task to the subagent.

        Args:
            prompt: The task or message to send.
            bg: If True, return immediately without waiting.
            max_turns: Override max turns for this task.
            timeout: Seconds to wait before returning (ignored if bg=True).

        Returns:
            SubagentResponse object with the result.

        Raises:
            SubagentError: If the task results in an error (only when blocking).
        """
        self._ensure_started()

        response = SubagentResponse(self)
        self._current_response = response

        _send_msg(self._conn, ("task", {
            "prompt": prompt,
            "max_turns": max_turns or self.max_turns
        }))

        if bg:
            return response

        # Wait with timeout
        start = time.time()
        while not response.done:
            if (time.time() - start) > timeout:
                break
            time.sleep(0.1)

        if response.done and response.is_error:
            raise SubagentError(response._error, response)

        return response

    @property
    def last(self) -> Optional[SubagentResponse]:
        """Last response object."""
        return self._current_response

    @property
    def done(self) -> bool:
        """Check if the last task has completed. True if no task sent."""
        return self._current_response.done if self._current_response else True

    @property
    def result(self) -> str:
        """Result text from the last task. Empty if none or not done."""
        return self._current_response.result if self._current_response else ""

    def wait(self, timeout: Optional[float] = None) -> Optional[SubagentResponse]:
        """Wait for the last task to complete.

        Args:
            timeout: Maximum seconds to wait. None = wait indefinitely.

        Returns:
            The SubagentResponse, or None if no task sent.

        Raises:
            SubagentError: If the task resulted in an error.
        """
        if self._current_response:
            return self._current_response.wait(timeout)
        return None

    def kill(self) -> str:
        """Kill the subagent process."""
        if self._proc:
            pid = self._proc.pid
            try:
                os.killpg(pid, signal.SIGKILL)
                return f"Killed subagent {self.id} (pid={pid})"
            except ProcessLookupError:
                return "Process already terminated"
        return "No process running"

    def _cleanup(self) -> None:
        """Clean up subprocess and socket."""
        if self._conn:
            try:
                _send_msg(self._conn, ("shutdown", None))
            except:
                pass
            try:
                self._conn.close()
            except:
                pass
            self._conn = None

        if self._proc:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except:
                    self._proc.kill()
            self._proc = None

        self._started = False

    def close(self) -> None:
        """Gracefully close the subagent."""
        self._cleanup()
        if self.id in _subagents:
            del _subagents[self.id]

    def __enter__(self) -> 'Subagent':
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self):
        try:
            self._cleanup()
        except:
            pass

    def __repr__(self) -> str:
        if self._proc and self._proc.poll() is None:
            status = "running" if (self._current_response and not self._current_response._done) else "idle"
            return f"[Subagent id={self.id} pid={self._proc.pid} status={status} cwd={self.cwd}]"
        return f"[Subagent id={self.id} status=stopped cwd={self.cwd}]"
