"""
SandboxMixin: Run REPLAgent code in an isolated overlay filesystem.

All filesystem modifications are captured in a temporary overlay layer.
When the session ends, changes are available as a tarball that can be
reviewed, applied, or discarded.

Usage:
    from agentlib.sandbox import SandboxMixin

    class SandboxedAgent(SandboxMixin, CodeAgent):
        pass

    with SandboxedAgent() as agent:
        result = agent.run("Create a new file called test.txt")

    # Changes are isolated - real filesystem unchanged
    # Apply if desired:
    agent.apply_changes()
"""

import base64
import json
import logging
import os
import signal
import socket
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import io
from pathlib import Path
from typing import Any, Optional

try:
    import cloudpickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger('agentlib.sandbox')


class SandboxCrashError(Exception):
    """Raised when the sandbox worker dies unexpectedly.

    This error contains diagnostic information to help debug the crash.
    The tarball from the crashed session (if any) is preserved.
    """

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics

    def __str__(self):
        lines = [super().__str__(), "", "=== Sandbox Crash Diagnostics ==="]
        for key, value in self.diagnostics.items():
            if key == 'stderr' and value:
                lines.append(f"{key}:")
                for line in value.strip().split('\n'):
                    lines.append(f"  {line}")
            elif key == 'stdout' and value:
                lines.append(f"{key}:")
                for line in value.strip().split('\n'):
                    lines.append(f"  {line}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Socket-based message passing
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
# Sandboxed worker process code (runs inside the sandbox)
# ---------------------------------------------------------------------------

WORKER_CODE = '''
import ast
import json
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


class StreamingWriter:
    def __init__(self, sock, original):
        self._sock = sock
        self._original = original

    def write(self, text):
        if text:
            _send_msg(self._sock, ("output", text))
        return len(text)

    def flush(self):
        pass

    def fileno(self):
        return self._original.fileno()


def worker_main(port, authkey):
    import os
    # Re-chdir to refresh CWD file descriptor after overlay mount.
    # Without this, relative paths would write to the real filesystem
    # because the CWD fd was opened before the overlay was mounted.
    os.chdir(os.getcwd())

    # Request SIGTERM when parent (sandbox_helper) dies (Linux-specific)
    # This handles the edge case where sandbox_helper is killed but we continue
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        SIGTERM = 15
        libc.prctl(PR_SET_PDEATHSIG, SIGTERM)
    except Exception:
        pass  # Not on Linux or ctypes unavailable

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60)  # Periodic timeout to check if we should exit
    sock.connect(('127.0.0.1', port))

    # Authenticate
    _send_msg(sock, authkey)
    ack = _recv_msg(sock)
    if ack != 'ok':
        raise RuntimeError("Authentication failed")

    repl_locals = {
        '_tool_request_sock': sock,
    }

    def sigint_handler(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, sigint_handler)

    while True:
        try:
            msg = _recv_msg(sock)

            if msg is None:
                break

            cmd_type, cmd_data = msg

            if cmd_type == "exec":
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StreamingWriter(sock, old_stdout)
                sys.stderr = StreamingWriter(sock, old_stderr)

                had_error = False
                try:
                    try:
                        tree = ast.parse(cmd_data, "<repl>", "exec")
                        for node in tree.body:
                            if isinstance(node, ast.Expr):
                                code_obj = compile(ast.Expression(node.value), "<repl>", "eval")
                                result = eval(code_obj, repl_locals)
                                if result is not None:
                                    print(repr(result))
                            else:
                                mod = ast.Module(body=[node], type_ignores=[])
                                code_obj = compile(mod, "<repl>", "exec")
                                exec(code_obj, repl_locals)
                    except SyntaxError as e:
                        had_error = True
                        sys.stderr.write(f"  File \\"<repl>\\", line {e.lineno}\\n")
                        if e.text:
                            sys.stderr.write(f"    {e.text}")
                            if e.offset:
                                sys.stderr.write(" " * (e.offset + 3) + "^\\n")
                        sys.stderr.write(f"SyntaxError: {e.msg}\\n")
                except KeyboardInterrupt:
                    had_error = True
                    sys.stderr.write("\\nKeyboardInterrupt\\n")
                except Exception as e:
                    had_error = True
                    import traceback
                    tb = e.__traceback__
                    while tb is not None and tb.tb_frame.f_code.co_filename != "<repl>":
                        tb = tb.tb_next
                    if tb is not None:
                        sys.stderr.write("Traceback (most recent call last):\\n")
                        sys.stderr.write("".join(traceback.format_tb(tb)))
                    sys.stderr.write(f"{type(e).__name__}: {e}\\n")
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

                _send_msg(sock, ("done", had_error))

            elif cmd_type == "inject":
                # Silent code injection (for tool stubs)
                try:
                    exec(cmd_data, repl_locals)
                    _send_msg(sock, ("inject_ok", None))
                except Exception as e:
                    _send_msg(sock, ("inject_error", str(e)))

        except KeyboardInterrupt:
            _send_msg(sock, ("output", "\\nKeyboardInterrupt\\n"))
            _send_msg(sock, ("done", True))
        except socket.timeout:
            # Periodic timeout - check if parent CLI is still alive
            # If ppid is 1 (init/systemd), parent died and we should exit
            if os.getppid() == 1:
                break
            # Otherwise continue waiting
        except ConnectionError:
            break

    sock.close()
'''


# ---------------------------------------------------------------------------
# SandboxedToolREPL: ToolREPL that runs inside overlay filesystem
# ---------------------------------------------------------------------------

class SandboxedToolREPL:
    """
    REPL that runs inside an overlay filesystem sandbox.

    All filesystem changes are isolated to a temporary layer.
    Changes are captured as a tarball when the session closes.
    """

    def __init__(self, target_dir: str = None, echo: bool = True):
        """
        Initialize sandboxed REPL.

        Args:
            target_dir: Directory to overlay (default: home directory)
            echo: Echo statements with >>> prefix
        """
        self.target_dir = target_dir or str(Path.home())
        self._echo = echo
        self._proc: Optional[subprocess.Popen] = None
        self._conn: Optional[socket.socket] = None
        self._server: Optional[socket.socket] = None
        self._tar_path: Optional[str] = None
        self._running: bool = False
        self._tools_injected: bool = False
        self._closed: bool = False
        self.tarball: Optional[bytes] = None

        # Diagnostics tracking
        self._session_start_time: Optional[float] = None
        self._session_pid: Optional[int] = None
        self._last_command: Optional[str] = None
        self._command_count: int = 0
        self._had_session: bool = False  # True once any session has started

    def _ensure_session(self) -> None:
        """Start the sandboxed worker if not running."""
        if self._proc is not None:
            exit_code = self._proc.poll()
            if exit_code is None:
                return  # Still running

            # Process exited - check if it was user-initiated or a crash
            if self._had_session:
                # SIGINT (-2) and SIGTERM (-15) are typically user-initiated
                # Exit code 0 is clean shutdown
                # These should allow restart without error
                if exit_code in (0, -2, -15):
                    # Check if there were potential unsaved changes
                    if self._command_count > 0:
                        logger.warning(
                            f"Sandbox session ended (exit code {exit_code}) after "
                            f"{self._command_count} command(s). Starting fresh session - "
                            f"previous changes may not be saved."
                        )
                    else:
                        logger.debug(
                            f"Sandbox session ended (exit code {exit_code}). "
                            f"Starting new session."
                        )
                    # Clean up old process reference
                    self._proc = None
                else:
                    # Actual unexpected crash
                    self._handle_unexpected_crash(exit_code)

        self._start_new_session()

    def _handle_unexpected_crash(self, exit_code: int) -> None:
        """Handle unexpected sandbox worker crash with full diagnostics."""
        # Capture all available diagnostic info
        diagnostics = {
            'exit_code': exit_code,
            'pid': self._session_pid,
            'target_dir': self.target_dir,
            'session_duration': None,
            'commands_executed': self._command_count,
            'last_command': None,
            'tarball_path': self._tar_path,
            'tarball_exists': False,
            'tarball_size': 0,
            'stdout': '',
            'stderr': '',
        }

        # Calculate session duration
        if self._session_start_time:
            diagnostics['session_duration'] = f"{time.time() - self._session_start_time:.1f}s"

        # Truncate last command if too long
        if self._last_command:
            cmd = self._last_command
            if len(cmd) > 500:
                cmd = cmd[:500] + f"... ({len(self._last_command)} chars total)"
            diagnostics['last_command'] = cmd

        # Check tarball status
        if self._tar_path and os.path.exists(self._tar_path):
            diagnostics['tarball_exists'] = True
            diagnostics['tarball_size'] = os.path.getsize(self._tar_path)

        # Get stdout/stderr from crashed process (non-blocking)
        if self._proc:
            try:
                stdout, stderr = self._proc.communicate(timeout=1)
                diagnostics['stdout'] = stdout.decode('utf-8', errors='replace')
                diagnostics['stderr'] = stderr.decode('utf-8', errors='replace')
            except subprocess.TimeoutExpired:
                diagnostics['stderr'] = "(process hung during output capture)"
            except Exception as e:
                diagnostics['stderr'] = f"(failed to capture: {e})"

        # Log the crash
        logger.error(
            f"Sandbox worker crashed unexpectedly (exit code {exit_code}). "
            f"PID={diagnostics['pid']}, duration={diagnostics['session_duration']}, "
            f"commands={diagnostics['commands_executed']}"
        )

        # Preserve tarball path in error message
        tarball_msg = ""
        if diagnostics['tarball_exists']:
            tarball_msg = f" Partial tarball preserved at: {self._tar_path}"

        raise SandboxCrashError(
            f"Sandbox worker died unexpectedly (exit code {exit_code}). "
            f"Previous edits may have been lost!{tarball_msg}",
            diagnostics
        )

    def _start_new_session(self) -> None:
        """Start a fresh sandbox session."""
        # Import here to trigger lazy compilation
        from . import get_sandbox_helper
        sandbox_helper = get_sandbox_helper()

        authkey = os.urandom(16)

        # Create server socket
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('127.0.0.1', 0))
        self._server.listen(1)
        port = self._server.getsockname()[1]
        self._server.settimeout(30)

        # Create temp file for tarball
        fd, self._tar_path = tempfile.mkstemp(suffix='.tar', prefix='sandbox_repl_')
        os.close(fd)
        os.unlink(self._tar_path)  # sandbox_helper will create it

        # Pass worker code as -c argument (inline)
        worker_bootstrap = f'''
import base64, sys
exec(base64.b64decode({repr(base64.b64encode(WORKER_CODE.encode()).decode())}).decode())
worker_main({port}, bytes.fromhex({repr(authkey.hex())}))
'''

        # Build command: sandbox_helper wraps Python running worker
        cmd = [
            sandbox_helper,
            '--tar', self._tar_path,
            self.target_dir,
            '--',
            sys.executable, '-c', worker_bootstrap
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Create new session to isolate from terminal signals (SIGINT from Ctrl+C)
            # Without this, Ctrl+C at terminal kills sandbox_helper, losing all edits
            start_new_session=True,
        )

        # Track session info for diagnostics
        self._session_start_time = time.time()
        self._session_pid = self._proc.pid
        self._command_count = 0
        self._last_command = None
        self._had_session = True

        logger.debug(f"Started sandbox session: PID={self._session_pid}, tar={self._tar_path}")

        # Accept connection from worker
        try:
            self._conn, addr = self._server.accept()
        except socket.timeout:
            self._proc.kill()
            stdout, stderr = self._proc.communicate()
            raise TimeoutError(f"Sandboxed worker failed to connect. stderr: {stderr.decode()}")
        finally:
            self._server.close()
            self._server = None

        # Authenticate
        client_key = _recv_msg(self._conn)
        if client_key != authkey:
            self._conn.close()
            self._proc.kill()
            raise RuntimeError("Worker authentication failed")
        _send_msg(self._conn, 'ok')

        self._tools_injected = False

    def inject_tools(self, tools: dict) -> None:
        """Inject tool implementations that run inside the sandbox.

        Extracts source code from actual tool methods and transforms them
        into standalone functions for the sandbox REPL.
        """
        self._ensure_session()

        if self._tools_injected:
            return

        # Pre-inject common imports that tools may need
        self._inject_code('''
import os
import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
''')

        import ast
        import inspect
        import textwrap

        for name, (impl, spec) in tools.items():
            if impl is None:
                continue  # Skip tools without implementations (e.g., MCP tools)

            try:
                source = inspect.getsource(impl)
            except (OSError, TypeError):
                continue  # Can't get source, skip

            # Dedent before parsing (source may be indented if from a class)
            source = textwrap.dedent(source)

            # Parse and transform the AST
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            # Find the function definition (skip decorators in AST)
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    func_def = node
                    break

            if func_def is None:
                continue

            # Remove 'self' parameter
            if func_def.args.args and func_def.args.args[0].arg == 'self':
                func_def.args.args.pop(0)

            # Strip type annotations (they may reference unavailable types)
            for arg in func_def.args.args:
                arg.annotation = None
            for arg in func_def.args.kwonlyargs:
                arg.annotation = None
            if func_def.args.vararg:
                func_def.args.vararg.annotation = None
            if func_def.args.kwarg:
                func_def.args.kwarg.annotation = None
            func_def.returns = None

            # Fix defaults: in agentlib, string defaults are descriptions, not values
            # Convert them to None (the actual intended default)
            new_defaults = []
            for default in func_def.args.defaults:
                if isinstance(default, ast.Constant) and isinstance(default.value, str):
                    new_defaults.append(ast.Constant(value=None))
                else:
                    new_defaults.append(default)
            func_def.args.defaults = new_defaults

            # Remove decorators
            func_def.decorator_list = []

            # Replace getattr(self, 'attr', default) with just default
            # This handles patterns like: getattr(self, 'jina_timeout', 60.0)
            class SelfGetattrReplacer(ast.NodeTransformer):
                def visit_Call(self, node):
                    self.generic_visit(node)
                    # Match getattr(self, 'attr', default)
                    if (isinstance(node.func, ast.Name) and
                        node.func.id == 'getattr' and
                        len(node.args) >= 3 and
                        isinstance(node.args[0], ast.Name) and
                        node.args[0].id == 'self'):
                        # Return the default value (3rd argument)
                        return node.args[2]
                    return node

            func_def = SelfGetattrReplacer().visit(func_def)
            ast.fix_missing_locations(func_def)

            # Unparse back to source
            func_source = ast.unparse(func_def)

            self._inject_code(func_source)

        self._tools_injected = True

    def inject_builtins(self) -> None:
        """Inject built-in functions like submit()."""
        self._ensure_session()

        # Inject socket-based tool call helpers
        self._inject_code('''
import json as _json
import struct as _struct

def _send_tool_request(data):
    payload = data.encode() if isinstance(data, str) else data
    import pickle as _pickle
    msg = _pickle.dumps(("tool_request", payload.decode() if isinstance(payload, bytes) else payload))
    _tool_request_sock.sendall(_struct.pack('!I', len(msg)) + msg)

def _recv_tool_response():
    import pickle as _pickle
    raw_len = b''
    while len(raw_len) < 4:
        chunk = _tool_request_sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Connection closed")
        raw_len += chunk
    msg_len = _struct.unpack('!I', raw_len)[0]
    chunks = []
    remaining = msg_len
    while remaining > 0:
        chunk = _tool_request_sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("Connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return _pickle.loads(b''.join(chunks))
''')

        # submit() function
        self._inject_code('''
def submit(result):
    """Submit your final result and end the task."""
    _send_tool_request(_json.dumps({
        "tool": "__submit__",
        "args": {"result": result}
    }))
    _recv_tool_response()  # Wait for ack
''')

    def _inject_code(self, code_str: str) -> None:
        """Execute code silently (no echo)."""
        self._ensure_session()
        _send_msg(self._conn, ("inject", code_str))

        msg_type, msg_data = _recv_msg(self._conn, timeout=10.0)
        if msg_type == "inject_error":
            raise RuntimeError(f"Failed to inject code: {msg_data}")

    def execute(self, code: str) -> None:
        """Send code to execute (non-blocking start)."""
        self._ensure_session()
        self._running = True
        self._last_command = code
        self._command_count += 1
        _send_msg(self._conn, ("exec", code))

    def poll_output(self, timeout: float = 0.05) -> Optional[tuple]:
        """
        Poll for output from worker.

        Returns:
            ("output", text) - Output chunk
            ("done", had_error) - Execution complete
            ("tool_request", json_str) - Tool call request
            None - No message available
        """
        if self._conn is None:
            return None

        try:
            return _recv_msg(self._conn, timeout=timeout)
        except socket.timeout:
            return None
        except ConnectionError:
            self._running = False
            return ("done", True)

    def send_tool_response(self, result: Any = None, error: str = None) -> None:
        """Send tool execution result back to REPL."""
        if self._conn is None:
            raise RuntimeError("No active session")

        if error:
            _send_msg(self._conn, json.dumps({"error": error}))
        else:
            try:
                _send_msg(self._conn, json.dumps({"result": result}))
            except TypeError:
                _send_msg(self._conn, json.dumps({"result": repr(result)}))

    def send_ack(self) -> None:
        """Send simple acknowledgment (for submit())."""
        if self._conn is None:
            raise RuntimeError("No active session")
        _send_msg(self._conn, "ack")

    def interrupt(self) -> None:
        """Send SIGINT to worker process group."""
        if self._proc and self._proc.poll() is None:
            try:
                # Send to process group (sandbox runs in its own session)
                os.killpg(self._proc.pid, signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass

    def close(self) -> bytes:
        """
        Close the session and get the tarball of filesystem changes.

        Returns:
            Tarball bytes (empty if no changes)
        """
        if self._closed:
            return self.tarball or b""

        self._closed = True

        # Send shutdown
        if self._conn:
            try:
                _send_msg(self._conn, None)
            except:
                pass
            try:
                self._conn.close()
            except:
                pass
            self._conn = None

        # Wait for process
        if self._proc:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Kill entire process group (sandbox runs in its own session)
                try:
                    os.killpg(self._proc.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    self._proc.kill()
                self._proc.wait()
            self._proc = None

        # Read tarball
        if self._tar_path:
            try:
                with open(self._tar_path, 'rb') as f:
                    self.tarball = f.read()
            except FileNotFoundError:
                self.tarball = b""
            finally:
                try:
                    os.unlink(self._tar_path)
                except:
                    pass

        return self.tarball or b""

    def __del__(self):
        try:
            self.close()
        except:
            pass


# ---------------------------------------------------------------------------
# SandboxMixin: Drop-in mixin for REPLAgent subclasses
# ---------------------------------------------------------------------------

class SandboxMixin:
    """
    Mixin that runs REPLAgent code in a sandboxed overlay filesystem.

    All filesystem modifications are isolated. Changes are captured as
    a tarball when the agent session ends.

    Usage:
        class SandboxedAgent(SandboxMixin, CodeAgent):
            sandbox_target = "/home/user"  # Optional, defaults to home

        with SandboxedAgent() as agent:
            result = agent.run("Create test.txt with hello")

        # Real filesystem unchanged
        # Review changes:
        for path, content in agent.get_changed_files().items():
            print(f"{path}: {len(content)} bytes")

        # Apply if desired:
        agent.apply_changes()
    """

    sandbox_target: Optional[str] = None  # Directory to overlay

    def _get_tool_repl(self):
        """Override to return sandboxed REPL."""
        self._ensure_setup()

        if not hasattr(self, '_sandbox_repl'):
            target = getattr(self, 'sandbox_target', None) or str(Path.home())
            self._sandbox_repl = SandboxedToolREPL(target_dir=target, echo=True)

        repl = self._sandbox_repl

        # Inject tools - check both _toolimpl and instance methods
        tools = {}
        for name, spec in self.toolspecs.items():
            impl = self._toolimpl.get(name)
            if impl is None:
                # Try to find as instance method (e.g., JinaMixin tools)
                impl = getattr(self, name, None)
            tools[name] = (impl, spec)

        repl.inject_tools(tools)
        repl.inject_builtins()

        return repl

    def _execute_with_tool_handling(self, repl: SandboxedToolREPL, code: str) -> tuple[str, bool]:
        """
        Execute code in sandboxed REPL, handling tool calls.

        Overrides parent to use socket-based communication.
        """
        from agentlib.tools.subrepl import _split_into_statements, _format_echo
        from agentlib.agent import _CompleteException

        statements = _split_into_statements(code)
        if not statements:
            return "", False

        repl._ensure_session()
        output_chunks = []
        any_executed = False

        def stream(chunk):
            output_chunks.append(chunk)
            if hasattr(self, 'on_repl_chunk'):
                self.on_repl_chunk(chunk)

        for stmt in statements:
            # Pre-validate syntax
            try:
                compile(stmt, '<repl>', 'exec')
            except SyntaxError as e:
                stream(_format_echo(stmt))
                stream(self._format_syntax_error(e))
                break

            # Valid syntax - echo and execute
            any_executed = True
            stream(_format_echo(stmt))
            repl.execute(stmt)

            # Poll loop for this statement
            statement_had_error = False
            try:
                while True:
                    msg = repl.poll_output(timeout=0.05)
                    if msg is None:
                        continue

                    msg_type, msg_data = msg

                    if msg_type == "output":
                        stream(msg_data)

                    elif msg_type == "tool_request":
                        # Tool call from REPL
                        req = json.loads(msg_data)
                        self._handle_tool_request(repl, req)
                        if self.complete:
                            # submit() was called - drain and return
                            while True:
                                drain_msg = repl.poll_output(timeout=0.1)
                                if drain_msg is None:
                                    break
                                if drain_msg[0] == "output":
                                    stream(drain_msg[1])
                                elif drain_msg[0] == "done":
                                    break
                            repl._running = False
                            return "".join(output_chunks), False

                    elif msg_type == "done":
                        repl._running = False
                        statement_had_error = msg_data
                        break

            except KeyboardInterrupt:
                repl.interrupt()
                # Drain remaining output
                while True:
                    drain_msg = repl.poll_output(timeout=0.5)
                    if drain_msg is None:
                        break
                    if drain_msg[0] == "output":
                        stream(drain_msg[1])
                    elif drain_msg[0] == "done":
                        break
                repl._running = False
                from agentlib.repl_agent import _InterruptedError
                raise _InterruptedError("".join(output_chunks))

            if statement_had_error:
                break

        output = "".join(output_chunks)
        is_pure_syntax_error = not any_executed and bool(output_chunks)
        return output, is_pure_syntax_error

    def _handle_tool_request(self, repl: SandboxedToolREPL, req: dict) -> None:
        """Handle a tool request from the sandboxed REPL."""
        from agentlib.agent import _CompleteException

        tool_name = req.get('tool')
        args = req.get('args', {})

        if tool_name == '__submit__':
            self._final_result = args.get('result')
            self.complete = True
            repl.send_ack()

        elif tool_name:
            try:
                result = self.toolcall(tool_name, args)
                repl.send_tool_response(result=result)
            except _CompleteException:
                raise
            except Exception as e:
                repl.send_tool_response(error=str(e))

    def _cleanup(self) -> None:
        """Close sandbox and capture tarball."""
        if hasattr(self, '_sandbox_repl'):
            self._sandbox_repl.close()
        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    def get_tarball(self) -> bytes:
        """Get the tarball of all filesystem changes."""
        if hasattr(self, '_sandbox_repl'):
            return self._sandbox_repl.tarball or b""
        return b""

    def get_changed_files(self) -> dict[str, bytes]:
        """
        Get dict of changed files: {relative_path: content}.

        Deleted files are not included (check get_deleted_files()).
        """
        tarball = self.get_tarball()
        if not tarball:
            return {}

        files = {}
        with tarfile.open(fileobj=io.BytesIO(tarball)) as tf:
            for member in tf.getmembers():
                if member.isfile():
                    f = tf.extractfile(member)
                    if f:
                        files[member.name] = f.read()
        return files

    def get_deleted_files(self) -> list[str]:
        """Get list of files that were deleted (overlay whiteouts)."""
        tarball = self.get_tarball()
        if not tarball:
            return []

        deleted = []
        with tarfile.open(fileobj=io.BytesIO(tarball)) as tf:
            for member in tf.getmembers():
                # Overlay whiteouts are character devices with major/minor 0,0
                if member.ischr() and member.devmajor == 0 and member.devminor == 0:
                    deleted.append(member.name)
        return deleted

    def apply_changes(self, target_dir: str = None) -> dict:
        """
        Apply sandbox changes to the real filesystem.

        Args:
            target_dir: Where to apply (default: sandbox_target or home)

        Returns:
            Dict with 'created', 'modified', 'deleted' lists
        """
        target = target_dir or getattr(self, 'sandbox_target', None) or str(Path.home())
        tarball = self.get_tarball()

        if not tarball:
            return {'created': [], 'modified': [], 'deleted': []}

        result = {'created': [], 'modified': [], 'deleted': []}

        with tarfile.open(fileobj=io.BytesIO(tarball)) as tf:
            for member in tf.getmembers():
                real_path = Path(target) / member.name

                if member.ischr() and member.devmajor == 0 and member.devminor == 0:
                    # Whiteout = delete
                    if real_path.exists():
                        real_path.unlink()
                        result['deleted'].append(member.name)
                elif member.isfile():
                    existed = real_path.exists()
                    real_path.parent.mkdir(parents=True, exist_ok=True)
                    tf.extract(member, target)
                    if existed:
                        result['modified'].append(member.name)
                    else:
                        result['created'].append(member.name)
                elif member.isdir():
                    real_path.mkdir(parents=True, exist_ok=True)

        return result

    def discard_changes(self) -> None:
        """Discard all sandbox changes (they're already isolated, this is a no-op)."""
        pass  # Changes were never applied to real filesystem

    def _ensure_setup(self):
        """Initialize sandbox and register exit hook if CLIMixin is present."""
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        # Register pre-exit hook if CLIMixin is in the MRO
        if not hasattr(self, '_sandbox_hook_registered'):
            self._sandbox_hook_registered = True
            if hasattr(self, 'register_pre_exit_hook'):
                self.register_pre_exit_hook(self.finalize_sandbox)

    def finalize_sandbox(self) -> None:
        """
        Interactive prompt to review and handle sandbox changes.

        Called automatically on CLI exit if CLIMixin is present.
        Shows summary of changes and prompts for action.
        """
        # Make sure sandbox is closed and tarball is captured
        if hasattr(self, '_sandbox_repl'):
            self._sandbox_repl.close()

        changed = self.get_changed_files()
        deleted = self.get_deleted_files()

        if not changed and not deleted:
            print("\n[Sandbox] No filesystem changes made.")
            return

        # Categorize changes
        target = getattr(self, 'sandbox_target', None) or str(Path.home())
        created = []
        modified = []
        for path, content in changed.items():
            real_path = Path(target) / path
            if real_path.exists():
                modified.append((path, len(content)))
            else:
                created.append((path, len(content)))

        # Show file list
        print("\n" + "=" * 60)
        print("SANDBOX CHANGES")
        print("=" * 60)

        if created:
            print(f"\nCreated ({len(created)}):")
            for path, size in sorted(created):
                print(f"  + {path} ({size} bytes)")

        if modified:
            print(f"\nModified ({len(modified)}):")
            for path, size in sorted(modified):
                print(f"  M {path} ({size} bytes)")

        if deleted:
            print(f"\nDeleted ({len(deleted)}):")
            for path in sorted(deleted):
                print(f"  - {path}")

        print()

        # Prompt for action
        while True:
            print("What would you like to do?")
            print("  [r] Review diff (default)")
            print("  [a] Apply changes to filesystem")
            print("  [s] Save as patch file")
            print("  [d] Discard")
            print()

            try:
                choice = input("> ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nDiscarding changes.")
                return

            if choice == '' or choice == 'r':
                self._show_diff()
                # Loop back to prompt

            elif choice == 'd':
                print("Changes discarded.")
                return

            elif choice == 'a':
                confirm = input("Apply all changes to real filesystem? [y/N] ").strip().lower()
                if confirm == 'y':
                    result = self.apply_changes()
                    total = len(result['created']) + len(result['modified']) + len(result['deleted'])
                    print(f"Applied {total} change(s).")
                    return
                # Loop back if not confirmed

            elif choice == 's':
                default_path = "sandbox_changes.tar"
                path = input(f"Save path [{default_path}]: ").strip() or default_path
                try:
                    with open(path, 'wb') as f:
                        f.write(self.get_tarball())
                    print(f"Saved to {path}")
                    return
                except Exception as e:
                    print(f"Error saving: {e}")
                    # Loop back

            else:
                print("Invalid choice. Please enter r, a, s, or d.")

    def _show_diff(self) -> None:
        """Show unified diff of all changes."""
        import difflib

        changed = self.get_changed_files()
        deleted = self.get_deleted_files()
        target = getattr(self, 'sandbox_target', None) or str(Path.home())

        print("\n" + "-" * 60)

        # Show deleted files
        for path in sorted(deleted):
            print(f"\n--- {path}")
            print(f"+++ /dev/null")
            real_path = Path(target) / path
            if real_path.exists():
                try:
                    original = real_path.read_text().splitlines(keepends=True)
                    for line in difflib.unified_diff(original, [], fromfile=path, tofile='/dev/null'):
                        print(line, end='')
                except:
                    print(f"(binary or unreadable file)")

        # Show modified/created files
        for path in sorted(changed.keys()):
            real_path = Path(target) / path
            new_content = changed[path]

            # Try to decode as text
            try:
                new_lines = new_content.decode('utf-8').splitlines(keepends=True)
            except UnicodeDecodeError:
                print(f"\n{path}: (binary file, {len(new_content)} bytes)")
                continue

            if real_path.exists():
                try:
                    original = real_path.read_text().splitlines(keepends=True)
                except:
                    original = []
                label = "modified"
            else:
                original = []
                label = "new file"

            diff = list(difflib.unified_diff(
                original, new_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}"
            ))

            if diff:
                print(f"\n{path} ({label}):")
                for line in diff:
                    print(line, end='')
                if not diff[-1].endswith('\n'):
                    print()

        print("\n" + "-" * 60)
