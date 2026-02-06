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
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm

    # Authenticate
    _send_msg(sock, authkey)
    ack = _recv_msg(sock)
    if ack != 'ok':
        raise RuntimeError("Authentication failed")

    repl_locals = {
        '_sock': sock,  # For _send_output() in builtins
        '_tool_request_sock': sock,
        '_send_msg': _send_msg,  # For SOCKET_SEND_OUTPUT_CODE
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
        self._accumulated_tarball: Optional[bytes] = None  # Accumulated changes across restarts

    def _ensure_session(self) -> None:
        """Start the sandboxed worker if not running."""
        if self._proc is not None:
            exit_code = self._proc.poll()
            if exit_code is None:
                return  # Still running

            # Process exited - capture tarball before cleanup
            self._capture_tarball_from_ended_session()
            
            if self._had_session and self._command_count > 0:
                # Session ended with potential changes - warn user
                logger.warning(
                    f"Sandbox session ended (exit code {exit_code}) after "
                    f"{self._command_count} command(s). Recovering changes and restarting..."
                )
            else:
                logger.debug(
                    f"Sandbox session ended (exit code {exit_code}). "
                    f"Starting new session."
                )
            
            # Clean up old process reference
            self._proc = None

        self._start_new_session()

    def _capture_tarball_from_ended_session(self) -> None:
        """Capture tarball from a session that ended, accumulating changes."""
        if not self._tar_path:
            return
            
        try:
            if os.path.exists(self._tar_path):
                with open(self._tar_path, 'rb') as f:
                    new_tarball = f.read()
                
                if new_tarball:
                    # Merge with any existing accumulated tarball
                    if self._accumulated_tarball:
                        self._accumulated_tarball = self._merge_tarballs(
                            self._accumulated_tarball, new_tarball
                        )
                    else:
                        self._accumulated_tarball = new_tarball
                    
                    logger.info(
                        f"Captured {len(new_tarball)} bytes from ended session "
                        f"(accumulated: {len(self._accumulated_tarball)} bytes)"
                    )
        except Exception as e:
            logger.error(f"Failed to capture tarball from ended session: {e}")
        finally:
            # Clean up the old tarball file
            try:
                if self._tar_path and os.path.exists(self._tar_path):
                    os.unlink(self._tar_path)
            except:
                pass
            self._tar_path = None
    
    def _merge_tarballs(self, base: bytes, overlay: bytes) -> bytes:
        """Merge two tarballs, with overlay taking precedence."""
        if not base:
            return overlay
        if not overlay:
            return base
        
        # Extract both to memory, overlay wins for conflicts
        files = {}
        
        for tarball_bytes in [base, overlay]:
            try:
                with tarfile.open(fileobj=io.BytesIO(tarball_bytes)) as tf:
                    for member in tf.getmembers():
                        if member.isfile():
                            f = tf.extractfile(member)
                            if f:
                                files[member.name] = (member, f.read())
                        else:
                            # Preserve non-file entries (dirs, whiteouts)
                            files[member.name] = (member, None)
            except Exception as e:
                logger.warning(f"Error reading tarball during merge: {e}")
        
        # Create merged tarball
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode='w') as tf:
            for name, (member, content) in files.items():
                if content is not None:
                    member.size = len(content)
                    tf.addfile(member, io.BytesIO(content))
                else:
                    tf.addfile(member)
        
        return out.getvalue()

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

        # If we have accumulated changes from a previous session, write them to restore file
        restore_path = None
        if self._accumulated_tarball:
            fd, restore_path = tempfile.mkstemp(suffix='.tar', prefix='sandbox_restore_')
            try:
                os.write(fd, self._accumulated_tarball)
            finally:
                os.close(fd)
            logger.info(f"Restoring {len(self._accumulated_tarball)} bytes of accumulated changes")

        # Pass worker bootstrap over stdin to avoid huge argv (ps spam)
        worker_bootstrap = f'''\
import sys
exec({repr(WORKER_CODE)})
worker_main({port}, bytes.fromhex({repr(authkey.hex())}))
'''

        # Build command: sandbox_helper wraps Python running worker
        cmd = [
            sandbox_helper,
            '--tar', self._tar_path,
        ]
        if restore_path:
            cmd.extend(['--restore', restore_path])
        cmd.extend([
            self.target_dir,
            '--',
            sys.executable, '-'
        ])
        
        # Track restore path for cleanup
        self._restore_path = restore_path

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Create new session to isolate from terminal signals (SIGINT from Ctrl+C)
            # Without this, Ctrl+C at terminal kills sandbox_helper, losing all edits
            start_new_session=True,
        )

        assert self._proc.stdin is not None
        self._proc.stdin.write(worker_bootstrap.encode())
        self._proc.stdin.close()

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
            self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle
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

        # Clean up restore file now that sandbox has loaded it
        if hasattr(self, '_restore_path') and self._restore_path:
            try:
                os.unlink(self._restore_path)
            except:
                pass
            self._restore_path = None

        self._tools_injected = False
        self._builtins_injected = False

    def inject_tools(self, tools: dict) -> None:
        """Inject tool functions into the sandbox.

        Tools marked with inject=True have their source extracted and run
        directly in the sandbox. Other tools get relay stubs that call
        back to the host process via socket.

        Args:
            tools: Dict of tool_name -> (implementation, pydantic_spec)
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

        from agentlib.repl_agent import _extract_tool_source, _generate_socket_relay_stub

        for name, (impl, spec) in tools.items():
            should_inject = impl is not None and getattr(impl, '_tool_inject', False)
            
            if should_inject:
                code = _extract_tool_source(impl, name)
            elif impl is not None or spec is not None:
                # Socket relay stub - calls back to host process
                code = _generate_socket_relay_stub(name, impl, spec)
            else:
                continue
            
            self._inject_code(code)

        self._tools_injected = True

    def inject_builtins(self) -> None:
        """Inject built-in functions like emit() and patched print()."""
        from agentlib.repl_agent import BUILTINS_CODE, SOCKET_SEND_OUTPUT_CODE

        self._ensure_session()
        if getattr(self, '_builtins_injected', False):
            return

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

        # Inject shared builtins (emit, patched print)
        self._inject_code(SOCKET_SEND_OUTPUT_CODE)
        self._inject_code(BUILTINS_CODE)
        self._builtins_injected = True

    def _inject_code(self, code: str, timeout: float = 10.0) -> None:
        """Override to use socket transport."""
        self._ensure_session()
        _send_msg(self._conn, ("inject", code))

        msg_type, msg_data = _recv_msg(self._conn, timeout=timeout)
        if msg_type == "inject_error":
            raise RuntimeError(f"Failed to inject code: {msg_data}")

    def inject_startup(self, code_list: list[str], timeout: float = 10.0) -> None:
        """
        Inject startup code silently.
        
        Args:
            code_list: List of Python code strings to execute
            timeout: Max seconds per code block (default 10.0)
        """
        for code in code_list:
            if code and code.strip():
                self._inject_code(code, timeout=timeout)

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

    def _check_worker_alive(self) -> None:
        """Check if worker is alive, close connection and raise if dead."""
        if self._proc is None or self._proc.poll() is not None:
            # Worker dead - close connection
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
            raise RuntimeError("Worker process died")

    def send_reply(self, request_id: int, result: Any = None, error: str = None) -> None:
        """Send application-level reply with result or error.

        Args:
            request_id: The request ID this reply is for.
            result: The result value (if successful).
            error: The error message (if failed).
        """
        if self._conn is None:
            raise RuntimeError("No active session")
        self._check_worker_alive()

        msg = {"type": "reply", "request_id": request_id}
        if error is not None:
            msg["error"] = error
        else:
            try:
                json.dumps(result)  # Test serializability
                msg["result"] = result
            except TypeError:
                msg["result"] = repr(result)
        _send_msg(self._conn, json.dumps(msg))

    def send_ack(self, request_id: int) -> None:
        """Send transport-level ACK to unblock the sender.

        Args:
            request_id: The request ID this ACK is for.
        """
        if self._conn is None:
            raise RuntimeError("No active session")
        self._check_worker_alive()
        _send_msg(self._conn, json.dumps({"type": "ack", "request_id": request_id}))

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

        # Wait for process to finish (tarball generation can take time for large changesets)
        if self._proc:
            self._proc.wait()
            self._proc = None

        # Read tarball
        final_tarball = b""
        if self._tar_path:
            try:
                with open(self._tar_path, 'rb') as f:
                    final_tarball = f.read()
            except FileNotFoundError:
                pass
            finally:
                try:
                    os.unlink(self._tar_path)
                except:
                    pass
        
        # Merge with accumulated tarball from any previous session restarts
        if self._accumulated_tarball:
            if final_tarball:
                self.tarball = self._merge_tarballs(self._accumulated_tarball, final_tarball)
            else:
                self.tarball = self._accumulated_tarball
        else:
            self.tarball = final_tarball

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

        # Inject builtins first so tools can use _send_output, _original_print, etc.
        repl.inject_builtins()

        tools = {
            name: (self._toolimpl.get(name), spec)
            for name, spec in self.toolspecs.items()
        }

        repl.inject_tools(tools)
        
        # Inject startup code if defined
        if not getattr(self, '_repl_startup_injected', False):
            startup = getattr(self, 'repl_startup', None)
            if startup:
                if callable(startup):
                    startup = startup()
                repl.inject_startup(startup)
            self._repl_startup_injected = True

        return repl

    def _execute_with_tool_handling(self, repl: SandboxedToolREPL, code: str) -> tuple[str, bool, list, str]:
        """
        Execute code in sandboxed REPL, handling tool calls.

        Overrides parent to use socket-based communication.
        """
        from agentlib.tools.subrepl import _split_into_statements, _format_echo
        from agentlib.agent import _CompleteException

        # Allow subclasses to preprocess code before splitting
        if hasattr(self, 'preprocess_code'):
            code = self.preprocess_code(code)

        # Split into statements, then transform each individually
        # (Can't transform whole code first because AST strips comments,
        # which would cause misalignment between original and transformed statements)
        original_statements = _split_into_statements(code)
        if not original_statements:
            return "", False, [], code

        repl._ensure_session()
        output_chunks = []  # List of (msg_type, chunk) tuples for entire turn
        any_executed = False

        # Per-statement tracking for on_statement_output hook
        statement_chunks = []

        def stream(chunk, msg_type="echo"):
            output_chunks.append((msg_type, chunk))
            statement_chunks.append((msg_type, chunk))
            if hasattr(self, 'on_repl_chunk'):
                self.on_repl_chunk(chunk, msg_type)

        for original_stmt in original_statements:
            # Transform this statement (print -> _print for output tagging)
            exec_stmt = self._transform_toplevel_print(original_stmt)

            # Pre-validate syntax (use transformed for validation)
            try:
                compile(exec_stmt, '<repl>', 'exec')
            except SyntaxError as e:
                stream(_format_echo(original_stmt), "echo")
                stream(self._format_syntax_error(e), "error")
                if hasattr(self, 'on_statement_output'):
                    self.on_statement_output(statement_chunks)
                break

            # Valid syntax - echo original, execute transformed
            any_executed = True
            stream(_format_echo(original_stmt), "echo")
            repl.execute(exec_stmt)

            # Poll loop for this statement
            statement_had_error = False
            try:
                while True:
                    msg = repl.poll_output(timeout=0.05)
                    if msg is None:
                        continue

                    msg_type, msg_data = msg

                    if msg_type in ("output", "print", "emit", "read", "progress"):
                        stream(msg_data, msg_type)

                    elif msg_type == "tool_request":
                        # Tool call from REPL
                        req = json.loads(msg_data)
                        self._handle_tool_request(repl, req)
                        if self.complete:
                            # emit(release=True) was called - drain and return
                            while True:
                                drain_msg = repl.poll_output(timeout=0.1)
                                if drain_msg is None:
                                    break
                                if drain_msg[0] in ("output", "print", "emit", "read", "progress"):
                                    stream(drain_msg[1], drain_msg[0])
                                elif drain_msg[0] == "done":
                                    break
                            repl._running = False
                            output = "".join(chunk for _, chunk in output_chunks)
                            return output, False, output_chunks, code

                    elif msg_type == "done":
                        repl._running = False
                        statement_had_error = msg_data
                        # Drain any remaining output that arrived with "done"
                        while True:
                            drain = repl.poll_output(timeout=0.01)
                            if drain is None:
                                break
                            if drain[0] in ("output", "print", "emit", "read", "progress"):
                                stream(drain[1], drain[0])
                        break

            except KeyboardInterrupt:
                repl.interrupt()
                # Drain remaining output
                while True:
                    drain_msg = repl.poll_output(timeout=0.5)
                    if drain_msg is None:
                        break
                    if drain_msg[0] in ("output", "print", "emit", "read", "progress"):
                        stream(drain_msg[1], drain_msg[0])
                    elif drain_msg[0] == "done":
                        break
                repl._running = False
                from agentlib.repl_agent import _InterruptedError
                raise _InterruptedError("".join(chunk for _, chunk in output_chunks))

            # Statement complete - call hook and reset per-statement tracking
            if hasattr(self, 'on_statement_output'):
                self.on_statement_output(statement_chunks)
            statement_chunks.clear()  # Must use clear() to preserve closure reference

            if statement_had_error:
                break

        output = "".join(chunk for _, chunk in output_chunks)
        is_pure_syntax_error = not any_executed and bool(output_chunks)
        return output, is_pure_syntax_error, output_chunks, code

    def _handle_tool_request(self, repl: SandboxedToolREPL, req: dict) -> None:
        """Handle a tool request from the sandboxed REPL."""
        from agentlib.agent import _CompleteException

        tool_name = req.get('tool')
        request_id = req.get('request_id')
        args = req.get('args', {})

        # Deserialize special types (like bytes) - copied from REPLAgent
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
                # No reply needed for emit, just ACK

            elif tool_name:
                # Regular tool call - send reply with result
                try:
                    result = self.toolcall(tool_name, args)
                    repl.send_reply(request_id, result=result)
                except _CompleteException:
                    # Tool called self.respond() - still need ACK
                    raise
                except Exception as e:
                    repl.send_reply(request_id, error=str(e))
        finally:
            # Always send ACK to unblock the sender
            repl.send_ack(request_id)

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
