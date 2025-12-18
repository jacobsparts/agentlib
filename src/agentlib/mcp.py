#!/usr/bin/env python3
from __future__ import annotations
"""
MCP (Model Context Protocol) Client

A single-file implementation using only the Python standard library.
Supports both Stdio and SSE transports.

No external dependencies required. Compatible with gevent when monkey-patched.

Requires: Python 3.9+ (uses parameterized generics in type annotations).

Platform: Linux only. This library uses fcntl for non-blocking I/O which is not
available on Windows.

Thread Safety: This library has been developed with thread safety in mind. Public
methods use internal locking and can be called from multiple threads. Note that
calling close() while operations are in-flight will cause those operations to
raise TransportError.

Security Model: This library assumes a trusted execution environment. The subprocess
command, environment variables, and HTTP headers are passed through without validation.
This is appropriate for controlled environments where the caller constructs these values
from trusted sources. Do not use with untrusted input without additional validation.

Malicious Server Warning: This library does not protect against malicious MCP servers.
A malicious server could send unbounded endpoint URLs, oversized HTTP headers, oversized
JSON messages, or excessive pending messages to cause memory exhaustion. Only connect to
trusted servers.

Limitations:
- No HTTP keep-alive for POST requests (new TCP connection per message)
- No automatic SSE reconnection (caller should catch TransportError and reconnect)
- No SSL certificate configuration (uses system defaults)
- No SSE connection heartbeat/keep-alive detection. This library is designed for
  RPC-style request/response patterns where the client actively calls server methods.
  Silent connection drops are detected on the next operation attempt.
- Pending message queue drops oldest messages when full (MAX_PENDING_MESSAGES=100)
"""

import errno
import json
import os
import select
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urljoin

# Platform check: fcntl is required for non-blocking I/O
try:
    import fcntl
except ImportError:
    raise ImportError(
        "This module requires fcntl, which is only available on Unix/Linux. "
        "Windows is not supported."
    )

# JSON-RPC 2.0 version string
JSONRPC_VERSION = "2.0"

# Maximum buffer size to prevent memory exhaustion (100 MB)
MAX_BUFFER_SIZE = 100 * 1024 * 1024

# Maximum HTTP header size (64 KB - standard safe limit)
MAX_HEADER_SIZE = 64 * 1024

# Chunk sizes for reading from sockets/pipes
READ_CHUNK_SIZE = 65536
HEADER_CHUNK_SIZE = 4096

# Grace period for subprocess termination before sending SIGKILL
PROCESS_TERMINATE_TIMEOUT = 3

# Sentinel for unspecified timeout (distinguishes "not passed" from "explicitly None")
_TIMEOUT_NOT_SPECIFIED = object()

# Public API
__all__ = [
    # Exceptions
    "MCPError",
    "TransportError",
    "ProtocolError",
    "RPCError",
    "MCPTimeoutError",
    # Transport classes
    "Transport",
    "StdioTransport",
    "SSETransport",
    # Client
    "MCPClient",
    # Factory functions
    "create_stdio_client",
    "create_sse_client",
]


# ============================================================================
# Internal Helpers
# ============================================================================

def _log_stderr(message: str) -> None:
    """Write a message to stderr and flush immediately."""
    sys.stderr.write(message)
    sys.stderr.flush()


# ============================================================================
# Exceptions
# ============================================================================

class MCPError(Exception):
    """Base exception for MCP errors."""
    pass


class TransportError(MCPError):
    """Transport-level error (connection, I/O)."""
    pass


class ProtocolError(MCPError):
    """Protocol-level error (invalid messages, handshake failures)."""
    pass


class RPCError(MCPError):
    """JSON-RPC error returned by the server."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"RPC Error {code}: {message}")


class MCPTimeoutError(MCPError, TimeoutError):
    """Timeout waiting for server response."""
    pass


# ============================================================================
# Transport Layer - Abstract Base
# ============================================================================

class Transport(ABC):
    """Abstract base class for MCP transports."""

    @abstractmethod
    def connect(self) -> None:
        """Establish the transport connection."""
        pass

    @abstractmethod
    def send(self, message: dict) -> None:
        """Send a JSON-RPC message."""
        pass

    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> Optional[dict]:
        """
        Receive a JSON-RPC message.

        Args:
            timeout: Maximum time to wait in seconds. None means block forever.

        Returns:
            Parsed JSON message dict, or None on timeout.

        Raises:
            TransportError: On connection/I/O errors.
            ProtocolError: On invalid message format.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the transport connection."""
        pass


# ============================================================================
# Stdio Transport
# ============================================================================

class StdioTransport(Transport):
    """
    Transport using subprocess stdin/stdout pipes.

    Messages are newline-delimited JSON on stdout.
    Server stderr is forwarded to client stderr.
    Uses select.select() for non-blocking multiplexed I/O (gevent-compatible).
    """

    def __init__(
        self,
        command: list[str],
        env: Optional[dict[str, str]] = None,
        forward_stderr: bool = True
    ) -> None:
        """
        Args:
            command: Command and arguments to spawn the MCP server.
            env: Additional environment variables for the subprocess.
            forward_stderr: If True (default), forward server stderr to client stderr.
                           Set to False to silence server stderr output.
        """
        self.command = command
        self.env = env
        self.forward_stderr = forward_stderr
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.stdout_buffer = b""
        self._closed = False
        self._lock = threading.Lock()  # Protects _closed and process
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        self._process_group_created = False  # Track if we created a process group

    def connect(self) -> None:
        """Spawn the subprocess and set up non-blocking I/O."""
        # Reset closed flag to allow reconnection after close()
        self._closed = False
        # Reset buffer from any previous connection attempt
        self.stdout_buffer = b""

        # Merge environment
        process_env = os.environ.copy()
        if self.env:
            process_env.update(self.env)

        try:
            # Create a new session to ensure subprocess and its children form a process group
            # This allows us to kill the entire process group on cleanup
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=process_env,
                start_new_session=True  # Create new process group (Unix only)
            )
            self._process_group_created = True
        except OSError as e:
            raise TransportError(f"Failed to start process: {e}") from e

        # Set stdout, stderr, and stdin to non-blocking mode
        # stdin is set non-blocking to prevent deadlock if server stalls
        try:
            self._set_nonblocking(self.process.stdout.fileno())
            self._set_nonblocking(self.process.stderr.fileno())
            self._set_nonblocking(self.process.stdin.fileno())
        except OSError as e:
            # Clean up process if non-blocking setup fails
            try:
                if self._process_group_created and self.process.pid:
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=1)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    if self._process_group_created and self.process.pid:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait()
                except (OSError, ProcessLookupError):
                    pass
            self.process = None
            self._process_group_created = False
            raise TransportError(f"Failed to set non-blocking I/O: {e}") from e

    def _set_nonblocking(self, fd: int) -> None:
        """Set a file descriptor to non-blocking mode."""
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def send(self, message: dict, timeout: float = 30.0) -> None:
        """
        Send a JSON-RPC message as newline-delimited JSON.

        Uses non-blocking I/O with select to prevent deadlock if the server
        stalls and stops reading from stdin.

        Args:
            message: The JSON-RPC message to send.
            timeout: Maximum time to wait for the write to complete (default 30s).

        Raises:
            TransportError: On write errors or timeout.
        """
        with self._send_lock:
            # Capture process reference under _lock to avoid race with close()
            with self._lock:
                if self._closed:
                    raise TransportError("Transport not connected")
                process = self.process

            if process is None:
                raise TransportError("Transport not connected")

            if process.poll() is not None:
                raise TransportError(f"Process exited with code {process.returncode}")

            # Check stdin is still available (may be closed by close() in another thread)
            if not process.stdin:
                raise TransportError("Process stdin not available")

            try:
                data = (json.dumps(message, separators=(',', ':')) + "\n").encode('utf-8')
            except (TypeError, ValueError) as e:
                raise TransportError(f"Failed to serialize message: {e}") from e

            # Non-blocking write with select-based timeout
            deadline = time.monotonic() + timeout
            bytes_written = 0
            while bytes_written < len(data):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportError(
                        f"Timeout writing to server stdin after {bytes_written}/{len(data)} bytes"
                    )

                # Wait for stdin to be writable
                try:
                    _, writable, _ = select.select([], [process.stdin], [], min(remaining, 1.0))
                except (OSError, ValueError) as e:
                    raise TransportError(f"Select error on stdin: {e}") from e

                if not writable:
                    # Check if process died while we were waiting
                    if process.poll() is not None:
                        raise TransportError(f"Process exited with code {process.returncode}")
                    continue

                try:
                    n = process.stdin.write(data[bytes_written:])
                    if n is None:
                        # Non-blocking write returned None (would block)
                        continue
                    if n == 0:
                        raise TransportError("Write returned 0 bytes")
                    bytes_written += n
                except BlockingIOError:
                    # Would block, loop back and select again
                    continue
                except (BrokenPipeError, OSError) as e:
                    raise TransportError(f"Failed to send message: {e}") from e

            # Flush is also non-blocking now, but we've written all data
            try:
                process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise TransportError(f"Failed to flush message: {e}") from e

    def receive(self, timeout: Optional[float] = None) -> Optional[dict]:
        """
        Receive a JSON-RPC message using select-based I/O multiplexing.

        Handles interleaved stderr output by forwarding it to sys.stderr.
        Uses select.select() for gevent compatibility (gevent monkey-patches it).

        Thread safety: This method holds _recv_lock throughout, serializing
        concurrent receive() calls. send() uses a separate lock and can run
        concurrently.
        """
        with self._recv_lock:
            # Capture process reference under _lock to avoid race with close()
            with self._lock:
                if self._closed:
                    raise TransportError("Transport not connected")
                process = self.process

            if not process:
                raise TransportError("Transport not connected")

            deadline = None
            if timeout is not None:
                deadline = time.monotonic() + timeout

            while True:
                # Check for complete message in buffer
                if b"\n" in self.stdout_buffer:
                    line, self.stdout_buffer = self.stdout_buffer.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            decoded = line.decode('utf-8')
                        except UnicodeDecodeError as e:
                            raise ProtocolError(f"Invalid UTF-8 from server: {e}") from e
                        try:
                            return json.loads(decoded)
                        except json.JSONDecodeError as e:
                            raise ProtocolError(f"Invalid JSON from server: {e}") from e

                # Check process status
                if process.poll() is not None:
                    # Process exited - drain any remaining output
                    self._drain_remaining(process)
                    if b"\n" in self.stdout_buffer:
                        continue  # Try to parse remaining messages
                    # Clear any incomplete data to avoid confusion on future calls
                    if self.stdout_buffer:
                        self.stdout_buffer = b""
                    raise TransportError(
                        f"Server process exited with code {process.returncode}"
                    )

                # Calculate remaining timeout
                select_timeout = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None  # Timeout
                    select_timeout = remaining

                # Wait for data on stdout or stderr using select.select()
                # This is compatible with gevent monkey-patching
                try:
                    rlist = [process.stdout, process.stderr]
                    readable, _, _ = select.select(rlist, [], [], select_timeout)
                except OSError as e:
                    if e.errno == errno.EINTR:
                        continue  # Interrupted by signal, retry
                    raise TransportError(f"Select error: {e}") from e
                except ValueError as e:
                    raise TransportError(f"Select error: {e}") from e

                if not readable:
                    return None  # Timeout

                for fd in readable:
                    try:
                        if fd is process.stdout:
                            chunk = process.stdout.read(READ_CHUNK_SIZE)
                            if chunk:
                                if len(self.stdout_buffer) + len(chunk) > MAX_BUFFER_SIZE:
                                    raise TransportError(
                                        f"Buffer size exceeded {MAX_BUFFER_SIZE} bytes"
                                    )
                                self.stdout_buffer += chunk
                            elif chunk == b"":
                                # EOF on stdout - pipe closed
                                raise TransportError("Server stdout closed unexpectedly")
                        elif fd is process.stderr:
                            chunk = process.stderr.read(READ_CHUNK_SIZE)
                            if chunk and self.forward_stderr:
                                # Forward server stderr to our stderr
                                _log_stderr(chunk.decode('utf-8', errors='replace'))
                            # EOF on stderr is normal, don't raise
                    except BlockingIOError:
                        # Non-blocking read with no data available (race condition)
                        pass
                    except OSError as e:
                        if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                            raise TransportError(f"Read error: {e}") from e
                        # Non-blocking read with no data available

    def _drain_remaining(self, process: subprocess.Popen[bytes]) -> None:
        """
        Drain any remaining data from stdout/stderr after process exit.

        Note: Caller must hold _recv_lock.

        Args:
            process: The process to drain.
        """

        # Drain stdout (caller holds _recv_lock, enforce size limit)
        if process.stdout:
            try:
                # Limit read to prevent memory exhaustion
                remaining = process.stdout.read(MAX_BUFFER_SIZE)
                if remaining:
                    # Enforce buffer size limit even during drain
                    available = MAX_BUFFER_SIZE - len(self.stdout_buffer)
                    if available > 0:
                        self.stdout_buffer += remaining[:available]
            except OSError:
                pass

        # Drain stderr and optionally forward to sys.stderr (with size limit)
        if process.stderr:
            try:
                # Limit stderr read to prevent memory exhaustion
                remaining = process.stderr.read(MAX_BUFFER_SIZE)
                if remaining and self.forward_stderr:
                    _log_stderr(remaining.decode('utf-8', errors='replace'))
            except OSError:
                pass

    def close(self) -> None:
        """Close the transport and terminate the subprocess."""
        # Set closed flag first to signal other threads to stop.
        # This avoids deadlock - we don't hold locks while closing resources.
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self.process
            process_group_created = self._process_group_created
            self.process = None

        if process is not None:
            # Close all pipe handles - this will interrupt blocked select() calls
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe:
                    try:
                        pipe.close()
                    except OSError:
                        pass

            # Terminate the process and its children using process group if available
            try:
                if process_group_created and process.pid:
                    # Kill the entire process group to ensure child processes are terminated
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        # Process group may already be gone
                        pass
                else:
                    # Fall back to terminating just the parent process
                    process.terminate()

                # Wait for the process to exit
                process.wait(timeout=PROCESS_TERMINATE_TIMEOUT)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination times out
                try:
                    if process_group_created and process.pid:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except (OSError, ProcessLookupError):
                    # Process may already be dead
                    pass
                # Final wait to reap the zombie
                try:
                    process.wait(timeout=1)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            except OSError:
                pass

    def __del__(self) -> None:
        """Ensure subprocess cleanup even if close() is not explicitly called."""
        try:
            # Suppress any exceptions during cleanup to avoid issues during interpreter shutdown
            self.close()
        except Exception:
            pass


# ============================================================================
# SSE Transport
# ============================================================================

class SSETransport(Transport):
    """
    Transport using Server-Sent Events (SSE) over HTTP.

    Per MCP specification:
    - Client opens a GET request to receive SSE stream
    - Client sends POST requests to a message endpoint for outgoing messages
    - The message endpoint URL is provided by the server via an 'endpoint' event

    Uses raw sockets with select.select() for the SSE stream to enable non-blocking reads
    (gevent-compatible). POST messages use separate socket connections without keep-alive.
    """

    # Maximum pending messages to prevent unbounded memory growth
    MAX_PENDING_MESSAGES = 100

    def __init__(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0
    ) -> None:
        """
        Args:
            url: The SSE endpoint URL (e.g., "http://localhost:8080/sse")
            headers: Optional HTTP headers to include in requests (e.g., for authentication)
            timeout: Connection timeout in seconds for socket operations.
        """
        self.url = url
        self.parsed = urlparse(url)

        # Validate URL scheme and hostname
        if self.parsed.scheme not in ('http', 'https'):
            raise ValueError(
                f"Invalid URL scheme '{self.parsed.scheme or '(empty)'}': "
                f"expected 'http' or 'https'. Did you forget the scheme? "
                f"Example: http://localhost:3000/sse"
            )
        if not self.parsed.hostname:
            raise ValueError(
                f"Invalid URL '{url}': missing hostname. "
                f"Example: http://localhost:3000/sse"
            )

        self.host = self.parsed.hostname
        self.port = self.parsed.port or (443 if self.parsed.scheme == 'https' else 80)
        self.use_ssl = self.parsed.scheme == 'https'
        self.path = self.parsed.path or '/'
        if self.parsed.query:
            self.path += '?' + self.parsed.query

        self.headers = headers or {}
        self.timeout = timeout
        self.sse_socket: Optional[socket.socket] = None
        self.buffer = b""
        self.message_endpoint: Optional[str] = None
        self.pending_messages: deque[dict] = deque()
        self._closed = False
        self._lock = threading.Lock()  # Protects _closed and sse_socket
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        self._is_chunked = False
        self._chunk_buffer = b""  # Buffer for incomplete chunk data

    def _cleanup_sse_socket(self) -> None:
        """Close and clear the SSE socket, ignoring errors."""
        if self.sse_socket is not None:
            try:
                self.sse_socket.close()
            except OSError:
                pass
            self.sse_socket = None

    def connect(self) -> None:
        """Open the SSE connection."""
        # Reset closed flag to allow reconnection after close()
        self._closed = False
        # Reset state from any previous connection attempt
        self.buffer = b""
        self.pending_messages = deque()
        self.message_endpoint = None
        self._chunk_buffer = b""

        self._open_sse_stream()

        # Wait for the endpoint event from server (required before we can send)
        try:
            deadline = time.monotonic() + self.timeout
            while self.message_endpoint is None:
                # Check if close() was called during connect
                if self._closed:
                    raise TransportError("Transport closed during connect")
                if time.monotonic() > deadline:
                    raise TransportError("Server did not provide message endpoint")

                # Read and parse SSE events
                self._read_sse_data(timeout=1.0)
                self._parse_sse_events()
        except Exception:
            # Clean up socket on any failure during endpoint wait
            self._cleanup_sse_socket()
            raise

    def _create_socket(self, host: str, port: int, use_ssl: bool, timeout: float = 30.0) -> socket.socket:
        """Create a connected socket with optional SSL wrapping."""
        sock = socket.create_connection((host, port), timeout=timeout)

        if use_ssl:
            try:
                context = ssl.create_default_context()
                sock = context.wrap_socket(sock, server_hostname=host)
            except OSError:
                sock.close()
                raise

        return sock

    def _build_host_header(self, host: str, port: int, use_ssl: bool) -> str:
        """Build the Host header value, including port if non-standard."""
        default_port = 443 if use_ssl else 80
        # IPv6 addresses must be bracketed in Host header
        if ':' in host and not host.startswith('['):
            host = f"[{host}]"
        return host if port == default_port else f"{host}:{port}"

    def _open_sse_stream(self) -> None:
        """Open the SSE GET connection."""
        try:
            self.sse_socket = self._create_socket(self.host, self.port, self.use_ssl, self.timeout)
        except socket.error as e:
            raise TransportError(f"Failed to connect to {self.host}:{self.port}: {e}") from e

        try:
            host_header = self._build_host_header(self.host, self.port, self.use_ssl)

            # Build and send GET request
            request_lines = [
                f"GET {self.path} HTTP/1.1",
                f"Host: {host_header}",
                "Accept: text/event-stream",
                "Cache-Control: no-cache",
                "Connection: keep-alive",
                "User-Agent: mcp-python-client/1.0.0",
            ]
            # Add custom headers
            for name, value in self.headers.items():
                request_lines.append(f"{name}: {value}")
            request_lines.extend(["", ""])
            request = "\r\n".join(request_lines)

            try:
                self.sse_socket.sendall(request.encode('utf-8'))
            except socket.error as e:
                raise TransportError(f"Failed to send SSE request: {e}") from e

            # Read and validate HTTP response headers
            self._read_http_response_headers()

            # Set socket to non-blocking for select-based reading
            self.sse_socket.setblocking(False)
        except Exception:
            # Clean up socket on any failure after connection
            self._cleanup_sse_socket()
            raise

    def _read_http_response_headers(self) -> None:
        """Read HTTP response headers and validate status."""
        header_data = b""

        while b"\r\n\r\n" not in header_data:
            if len(header_data) > MAX_HEADER_SIZE:
                raise TransportError("HTTP headers too large")
            try:
                chunk = self.sse_socket.recv(HEADER_CHUNK_SIZE)
                if not chunk:
                    raise TransportError("Connection closed while reading headers")
                header_data += chunk
            except socket.timeout as e:
                raise TransportError("Timeout reading HTTP headers") from e

        # Split headers from any body data already received
        header_part, self.buffer = header_data.split(b"\r\n\r\n", 1)

        # Parse status line
        header_text = header_part.decode('utf-8', errors='replace')
        lines = header_text.split('\r\n')

        if not lines:
            raise TransportError("Empty HTTP response")

        status_line = lines[0]
        parts = status_line.split(' ', 2)

        if len(parts) < 2:
            raise TransportError(f"Invalid HTTP status line: {status_line}")

        try:
            status_code = int(parts[1])
        except ValueError as e:
            raise TransportError(f"Invalid HTTP status code: {parts[1]}") from e

        if status_code != 200:
            reason = parts[2] if len(parts) > 2 else "Unknown"
            # Include any body data in the error for debugging
            body = self.buffer.decode('utf-8', errors='replace').strip() if self.buffer else ""
            if body:
                raise TransportError(f"HTTP error {status_code}: {reason}\n{body}")
            else:
                raise TransportError(f"HTTP error {status_code}: {reason}")

        # Parse relevant headers
        self._is_chunked = False
        content_type = None
        for line in lines[1:]:
            if ':' in line:
                name, value = line.split(':', 1)
                header_name = name.strip().lower()
                header_value = value.strip()
                if header_name == 'transfer-encoding':
                    if 'chunked' in header_value.lower():
                        self._is_chunked = True
                elif header_name == 'content-type':
                    content_type = header_value.lower()

        # Validate Content-Type for SSE
        if content_type is None or not content_type.startswith('text/event-stream'):
            raise ProtocolError(
                f"Expected Content-Type 'text/event-stream', got '{content_type}'"
            )

        # If chunked, decode any initial chunk data already in buffer
        if self._is_chunked:
            self.buffer = self._decode_chunked_data(self.buffer)

        # Normalize line endings in any initial body data
        self.buffer = self.buffer.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

    def _decode_chunked_data(self, data: bytes) -> bytes:
        """
        Decode HTTP chunked transfer encoding.

        Chunked format:
            <chunk-size-hex>\r\n
            <chunk-data>\r\n
            ...
            0\r\n
            \r\n

        Incomplete chunks are stored in _chunk_buffer for next call.
        Returns decoded data.
        """
        # Prepend any leftover data from previous call
        data = self._chunk_buffer + data
        self._chunk_buffer = b""

        # Check combined size doesn't exceed limit
        if len(data) > MAX_BUFFER_SIZE:
            raise ProtocolError(
                f"Chunked data size {len(data)} exceeds maximum buffer size"
            )

        # Use list accumulation to avoid O(n²) concatenation
        chunks: list[bytes] = []

        while data:
            # Find the chunk size line
            idx = data.find(b"\r\n")
            if idx < 0:
                # Incomplete chunk header, save for later
                self._chunk_buffer = data
                break

            # Parse chunk size (hex)
            size_line = data[:idx]
            try:
                # Chunk size may have extensions after semicolon, ignore them
                size_str = size_line.split(b";")[0].strip()
                chunk_size = int(size_str, 16)
            except ValueError as e:
                raise ProtocolError(
                    f"Invalid chunk size in chunked transfer encoding: {size_line!r}"
                ) from e

            # Validate chunk size is reasonable
            if chunk_size > MAX_BUFFER_SIZE:
                raise ProtocolError(
                    f"Chunk size {chunk_size} exceeds maximum buffer size"
                )

            # Check if this is the final chunk (size 0)
            if chunk_size == 0:
                # Consume the terminating \r\n after "0\r\n" if present
                trailer_start = idx + 2
                if len(data) >= trailer_start + 2 and data[trailer_start:trailer_start + 2] == b"\r\n":
                    # Full terminator present, discard it
                    pass
                else:
                    # Incomplete terminator, save for later in case more data arrives
                    # (though for SSE streams this is typically the end)
                    self._chunk_buffer = data
                break

            # Check if we have the full chunk data + trailing \r\n
            chunk_start = idx + 2  # After the \r\n
            chunk_end = chunk_start + chunk_size
            if len(data) < chunk_end + 2:  # +2 for trailing \r\n
                # Incomplete chunk, save for later
                self._chunk_buffer = data
                break

            # Extract chunk data
            chunks.append(data[chunk_start:chunk_end])

            # Move past this chunk (data + \r\n)
            data = data[chunk_end + 2:]

        return b"".join(chunks)

    def _read_sse_data(
        self,
        timeout: Optional[float] = None,
        sock: Optional[socket.socket] = None
    ) -> bool:
        """
        Read available data from SSE socket into buffer.

        Returns True if data was read, False on timeout.
        Uses select.select() for gevent compatibility (gevent monkey-patches it).

        Thread safety: Caller must hold _recv_lock when called after connect()
        completes. During connect() (before the object is shared), no lock is needed.

        Args:
            timeout: Maximum time to wait in seconds.
            sock: Socket to read from. If None, uses self.sse_socket.
        """
        if sock is None:
            sock = self.sse_socket
        if sock is None:
            return False

        try:
            readable, _, _ = select.select([sock], [], [], timeout)
        except ValueError:
            # Invalid file descriptor - socket was closed
            raise TransportError("SSE socket closed or invalid")
        except OSError as e:
            # Distinguish between transient errors and actual socket issues
            if e.errno == errno.EINTR:
                # Interrupted by signal, treat as timeout
                return False
            raise TransportError(f"SSE socket error: {e}")

        if not readable:
            return False

        try:
            chunk = sock.recv(READ_CHUNK_SIZE)
            if chunk:
                if self._is_chunked:
                    chunk = self._decode_chunked_data(chunk)
                # Normalize line endings as data arrives (more efficient than
                # normalizing the entire buffer repeatedly in _parse_sse_events)
                chunk = chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                if len(self.buffer) + len(chunk) > MAX_BUFFER_SIZE:
                    raise TransportError(
                        f"Buffer size exceeded {MAX_BUFFER_SIZE} bytes"
                    )
                self.buffer += chunk
            else:
                raise TransportError("SSE connection closed by server")
        except BlockingIOError:
            return False
        except socket.error as e:
            raise TransportError(f"SSE socket error: {e}") from e

        return True

    def _parse_sse_events(self) -> None:
        """
        Parse SSE events from buffer.

        Handles:
        - 'endpoint' events: Sets the message POST endpoint
        - 'message' events: Adds parsed JSON to pending_messages queue

        Thread safety: Caller must hold _recv_lock when called after connect()
        completes. During connect() (before the object is shared), no lock is needed.

        Note: Line endings are normalized in _read_sse_data as data arrives.
        """
        # SSE events are separated by blank lines (\n\n)
        while b"\n\n" in self.buffer:
            event_data, self.buffer = self.buffer.split(b"\n\n", 1)

            # Parse the event fields
            event_type = None
            data_lines = []

            for line in event_data.decode('utf-8', errors='replace').split('\n'):
                if line.startswith('event:'):
                    event_type = line[6:].strip()
                elif line.startswith('data:'):
                    # Per SSE spec, remove only a single leading space if present
                    value = line[5:]
                    if value.startswith(' '):
                        value = value[1:]
                    data_lines.append(value)
                elif line.startswith('id:'):
                    # Could track event ID for reconnection
                    pass
                elif line.startswith(':'):
                    # Comment/heartbeat, ignore
                    pass

            if not data_lines:
                continue

            data = '\n'.join(data_lines)

            if event_type == 'endpoint':
                # Server provides the endpoint for POST requests
                # It may be a relative or absolute URL
                endpoint = data.strip()
                if endpoint.startswith('http://') or endpoint.startswith('https://'):
                    self.message_endpoint = endpoint
                else:
                    # Relative URL - combine with base
                    self.message_endpoint = urljoin(self.url, endpoint)

            elif event_type == 'message' or event_type is None:
                # JSON-RPC message from server
                try:
                    msg = json.loads(data)
                    # Limit queue size to prevent unbounded memory growth
                    if len(self.pending_messages) >= self.MAX_PENDING_MESSAGES:
                        dropped = self.pending_messages.popleft()
                        _log_stderr(
                            f"[warning] Dropping oldest pending message due to queue overflow: "
                            f"id={dropped.get('id')}\n"
                        )
                    self.pending_messages.append(msg)
                except json.JSONDecodeError:
                    # Skip malformed messages
                    _log_stderr(f"[warning] Invalid JSON in SSE message: {data[:100]}\n")

    class _IncrementalChunkedDecoder:
        """
        Incremental decoder for HTTP chunked transfer encoding.

        Unlike the streaming SSE decoder, this is designed for POST responses
        where we need to accumulate a complete body. It maintains parsing state
        to avoid re-parsing from the beginning on each new data chunk (O(n) vs O(n²)).
        """

        def __init__(self) -> None:
            self.chunks: list[bytes] = []
            self.buffer = b""
            self.is_complete = False
            self._state = 'size'  # 'size', 'data', 'trailer'
            self._chunk_size = 0

        def feed(self, data: bytes) -> None:
            """Feed more data to the decoder."""
            if len(self.buffer) + len(data) > MAX_BUFFER_SIZE:
                raise ProtocolError(
                    f"Chunked data exceeds maximum buffer size of {MAX_BUFFER_SIZE}"
                )
            self.buffer += data
            self._parse()

        def _parse(self) -> None:
            """Parse as much data as possible from the buffer."""
            while self.buffer and not self.is_complete:
                if self._state == 'size':
                    # Looking for chunk size line
                    idx = self.buffer.find(b"\r\n")
                    if idx < 0:
                        break  # Need more data

                    size_line = self.buffer[:idx]
                    try:
                        # Chunk size may have extensions after semicolon
                        size_str = size_line.split(b";")[0].strip()
                        self._chunk_size = int(size_str, 16)
                    except ValueError:
                        raise ProtocolError(f"Invalid chunk size: {size_line!r}")

                    if self._chunk_size > MAX_BUFFER_SIZE:
                        raise ProtocolError(
                            f"Chunk size {self._chunk_size} exceeds maximum buffer size"
                        )

                    self.buffer = self.buffer[idx + 2:]

                    if self._chunk_size == 0:
                        self._state = 'trailer'
                    else:
                        self._state = 'data'

                elif self._state == 'data':
                    # Reading chunk data + trailing \r\n
                    needed = self._chunk_size + 2
                    if len(self.buffer) < needed:
                        break  # Need more data

                    self.chunks.append(self.buffer[:self._chunk_size])
                    self.buffer = self.buffer[needed:]
                    self._state = 'size'

                elif self._state == 'trailer':
                    # Looking for end of trailers (blank line)
                    idx = self.buffer.find(b"\r\n")
                    if idx < 0:
                        break  # Need more data

                    if idx == 0:
                        # Blank line - chunked body is complete
                        self.buffer = self.buffer[2:]
                        self.is_complete = True
                    else:
                        # Skip trailer line
                        self.buffer = self.buffer[idx + 2:]

        def get_body(self) -> bytes:
            """Return the accumulated decoded body."""
            return b"".join(self.chunks)

    def _post_message(self, message: dict) -> None:
        """Send a JSON-RPC message via HTTP POST."""
        if not self.message_endpoint:
            raise TransportError("No message endpoint available")

        # Parse the endpoint URL
        parsed = urlparse(self.message_endpoint)
        host = parsed.hostname or self.host
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        use_ssl = parsed.scheme == 'https'
        path = parsed.path or '/'
        if parsed.query:
            path += '?' + parsed.query

        # Serialize the message
        try:
            body = json.dumps(message, separators=(',', ':')).encode('utf-8')
        except (TypeError, ValueError) as e:
            raise TransportError(f"Failed to serialize message: {e}") from e

        # Use deadline-based timeout for entire POST operation
        deadline = time.monotonic() + self.timeout

        sock = None
        try:
            sock = self._create_socket(host, port, use_ssl, self.timeout)
            host_header = self._build_host_header(host, port, use_ssl)

            # Build POST request (HTTP/1.0 for simplicity - connection closes after response)
            request_lines = [
                f"POST {path} HTTP/1.0",
                f"Host: {host_header}",
                "Content-Type: application/json",
                f"Content-Length: {len(body)}",
                "User-Agent: mcp-python-client/1.0.0",
            ]
            # Add custom headers
            for name, value in self.headers.items():
                request_lines.append(f"{name}: {value}")
            request_lines.extend(["", ""])
            request = "\r\n".join(request_lines).encode('utf-8') + body

            sock.sendall(request)

            # Read HTTP response headers first
            header_data = b""
            while b"\r\n\r\n" not in header_data:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportError("POST request timeout reading headers")
                if len(header_data) > MAX_HEADER_SIZE:
                    raise TransportError("POST response headers too large")
                sock.settimeout(min(remaining, 5.0))
                try:
                    chunk = sock.recv(HEADER_CHUNK_SIZE)
                    if not chunk:
                        raise TransportError("Connection closed while reading headers")
                    header_data += chunk
                except socket.timeout:
                    continue

            # Split headers from any body data already received
            header_part, body_data = header_data.split(b"\r\n\r\n", 1)
            header_text = header_part.decode('utf-8', errors='replace')
            lines = header_text.split('\r\n')

            if not lines:
                raise TransportError("Empty HTTP response")

            # Parse status line
            first_line = lines[0]
            status_parts = first_line.split(' ', 2)
            if len(status_parts) < 2:
                raise TransportError(f"Malformed HTTP response: {first_line}")
            try:
                status_code = int(status_parts[1])
            except ValueError as e:
                raise TransportError(f"Invalid HTTP status code: {first_line}") from e

            # Parse headers for Content-Length and Transfer-Encoding
            content_length = None
            is_chunked = False
            for line in lines[1:]:
                if ':' in line:
                    name, value = line.split(':', 1)
                    header_name = name.strip().lower()
                    header_value = value.strip()
                    if header_name == 'content-length':
                        try:
                            content_length = int(header_value)
                        except ValueError:
                            pass
                    elif header_name == 'transfer-encoding':
                        if 'chunked' in header_value.lower():
                            is_chunked = True

            # Read the response body based on framing method
            if is_chunked:
                # Read chunked body using incremental decoder (O(n) vs O(n²))
                decoder = self._IncrementalChunkedDecoder()
                decoder.feed(body_data)
                while not decoder.is_complete:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TransportError("POST request timeout reading body")
                    sock.settimeout(min(remaining, 5.0))
                    try:
                        chunk = sock.recv(READ_CHUNK_SIZE)
                        if not chunk:
                            break  # EOF - server closed connection
                        decoder.feed(chunk)
                    except socket.timeout:
                        continue
                if not decoder.is_complete:
                    raise TransportError(
                        "Connection closed before chunked response body complete"
                    )
                response_body = decoder.get_body()
            elif content_length is not None:
                # Read exactly Content-Length bytes
                if content_length > MAX_BUFFER_SIZE:
                    raise TransportError(
                        f"Content-Length {content_length} exceeds maximum buffer size"
                    )
                body_chunks = [body_data]
                bytes_read = len(body_data)
                while bytes_read < content_length:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TransportError("POST request timeout reading body")
                    sock.settimeout(min(remaining, 5.0))
                    try:
                        to_read = min(READ_CHUNK_SIZE, content_length - bytes_read)
                        chunk = sock.recv(to_read)
                        if not chunk:
                            raise TransportError(
                                f"Connection closed after {bytes_read}/{content_length} bytes"
                            )
                        body_chunks.append(chunk)
                        bytes_read += len(chunk)
                    except socket.timeout:
                        continue
                response_body = b"".join(body_chunks)
            else:
                # No Content-Length or chunked - read until EOF (HTTP/1.0 style)
                body_chunks = [body_data]
                body_size = len(body_data)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TransportError("POST request timeout reading body")
                    if body_size > MAX_BUFFER_SIZE:
                        raise TransportError(
                            f"POST response body exceeded {MAX_BUFFER_SIZE} bytes"
                        )
                    sock.settimeout(min(remaining, 5.0))
                    try:
                        chunk = sock.recv(READ_CHUNK_SIZE)
                        if not chunk:
                            break  # EOF
                        body_chunks.append(chunk)
                        body_size += len(chunk)
                    except socket.timeout:
                        continue
                response_body = b"".join(body_chunks)

            # Check for success status
            if status_code < 200 or status_code >= 300:
                body_text = response_body.decode('utf-8', errors='replace').strip()
                if body_text:
                    raise TransportError(f"POST request failed: {first_line}\n{body_text}")
                else:
                    raise TransportError(f"POST request failed: {first_line}")

        except socket.error as e:
            raise TransportError(f"Failed to POST message: {e}") from e
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def send(self, message: dict) -> None:
        """Send a JSON-RPC message to the server."""
        with self._send_lock:
            # Check closed state under _lock to avoid race with close()
            with self._lock:
                if self._closed:
                    raise TransportError("Transport not connected")

            self._post_message(message)

    def receive(self, timeout: Optional[float] = None) -> Optional[dict]:
        """
        Receive a JSON-RPC message from the SSE stream.

        Uses select.select() for non-blocking I/O to read from the stream.

        Thread safety: This method holds _recv_lock throughout, serializing
        concurrent receive() calls. send() uses a separate lock and can run
        concurrently.
        """
        with self._recv_lock:
            # Capture socket reference under _lock to avoid race with close()
            with self._lock:
                if self._closed:
                    raise TransportError("Transport not connected")
                sock = self.sse_socket

            if sock is None:
                raise TransportError("SSE socket not connected")

            # Return any pending messages first
            if self.pending_messages:
                return self.pending_messages.popleft()

            deadline = None
            if timeout is not None:
                deadline = time.monotonic() + timeout

            while True:
                # Parse any buffered SSE data
                self._parse_sse_events()

                if self.pending_messages:
                    return self.pending_messages.popleft()

                # Calculate remaining timeout
                recv_timeout = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    recv_timeout = remaining

                # Read more data from socket
                # _read_sse_data returns False on timeout or transient conditions
                # (EINTR, BlockingIOError). Don't treat transient conditions as timeout -
                # loop back and let the deadline check handle actual timeouts.
                self._read_sse_data(timeout=recv_timeout, sock=sock)

    def close(self) -> None:
        """Close the SSE connection."""
        # Set closed flag first to signal other threads to stop.
        # This avoids deadlock - we don't hold locks while closing resources.
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sock = self.sse_socket
            self.sse_socket = None

        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


# ============================================================================
# MCP Client
# ============================================================================

class MCPClient:
    """
    Model Context Protocol (MCP) Client.

    Provides a high-level interface to communicate with MCP servers
    using JSON-RPC 2.0 over configurable transports (Stdio or SSE).

    Connection and handshake happen automatically on instantiation.

    Usage:
        client = create_stdio_client(["python", "server.py"])
        tools = client.list_tools()
        result = client.call_tool("my_tool", {"arg": "value"})
        client.close()

    Or use as context manager:
        with create_stdio_client(["python", "server.py"]) as client:
            tools = client.list_tools()
            result = client.call_tool("my_tool", {"arg": "value"})
    """

    # MCP Protocol version we implement
    PROTOCOL_VERSION = "2024-11-05"

    # Maximum cached responses to prevent unbounded memory growth
    MAX_PENDING_RESPONSES = 100

    def __init__(
        self,
        transport: Transport,
        client_name: str = "mcp-python-client",
        client_version: str = "1.0.0",
        timeout: float = 300.0
    ) -> None:
        """
        Initialize and connect to an MCP server.

        Args:
            transport: The transport to use for communication.
            client_name: Client name reported during handshake.
            client_version: Client version reported during handshake.
            timeout: Default timeout in seconds for all operations. Individual
                    method calls can override this. Use a larger value for
                    servers with long-running operations.

        Raises:
            MCPError: On protocol or transport errors.
        """
        self.transport = transport
        self.client_name = client_name
        self.client_version = client_version
        self.default_timeout = timeout

        self._request_id = 0
        self._connected = False
        self._init_result: Optional[dict[str, Any]] = None
        # JSON-RPC IDs can be int, str, or null per spec
        self._pending_responses: dict[Any, dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Coordination for multi-threaded response waiting
        self._response_cv = threading.Condition(self._lock)
        self._receiving = False  # True if a thread is currently in transport.receive()

        # Optional callback for handling notifications
        self.on_notification: Optional[Callable[[str, dict[str, Any]], None]] = None

        # Connect immediately
        try:
            self._connect(timeout)
        except Exception:
            # Ensure transport is closed if handshake fails
            try:
                self.transport.close()
            except Exception:
                pass
            raise

    def _next_id(self) -> int:
        """Generate the next unique request ID."""
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _check_connected(self) -> None:
        """Raise MCPError if not connected. Thread-safe."""
        with self._lock:
            if not self._connected:
                raise MCPError("Client not connected")

    def _is_connected(self) -> bool:
        """Return connection status. Thread-safe."""
        with self._lock:
            return self._connected

    @staticmethod
    def _normalize_id(id_value: Any) -> Any:
        """
        Normalize a JSON-RPC ID for consistent dictionary key usage.

        Numeric IDs are converted to strings so that 1 (int) and 1.0 (float)
        both become "1" and match in dictionary lookups.
        """
        if isinstance(id_value, float):
            return f"{id_value:.15g}"
        if isinstance(id_value, int):
            return str(id_value)
        return id_value

    def _send_request(self, method: str, params: Optional[dict] = None) -> int:
        """
        Send a JSON-RPC request.

        Args:
            method: The RPC method name.
            params: Optional parameters.

        Returns:
            The request ID for correlating the response.
        """
        request_id = self._next_id()
        message: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": method
        }
        if params is not None:
            message["params"] = params

        self.transport.send(message)
        return request_id

    def _send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """
        Send a JSON-RPC notification (no response expected).

        Args:
            method: The notification method name.
            params: Optional parameters.
        """
        message: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method
        }
        if params is not None:
            message["params"] = params

        self.transport.send(message)

    def _send_error_response(
        self,
        request_id: Any,
        code: int,
        message: str,
        data: Any = None
    ) -> None:
        """
        Send a JSON-RPC error response.

        Used to respond to server requests we don't support.

        Args:
            request_id: The request ID to respond to.
            code: JSON-RPC error code.
            message: Error message.
            data: Optional additional error data.
        """
        error: dict[str, Any] = {
            "code": code,
            "message": message
        }
        if data is not None:
            error["data"] = data

        response: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": error
        }
        self.transport.send(response)

    def _validate_message(self, message: Any) -> None:
        """
        Validate basic JSON-RPC 2.0 message structure.

        Args:
            message: The parsed message to validate.

        Raises:
            ProtocolError: If the message structure is invalid.
        """
        if not isinstance(message, dict):
            raise ProtocolError(f"Expected JSON object, got {type(message).__name__}")

        if message.get("jsonrpc") != JSONRPC_VERSION:
            raise ProtocolError(
                f"Invalid or missing jsonrpc version: {message.get('jsonrpc')!r}"
            )

        # A valid JSON-RPC 2.0 message must have either:
        # - 'method' field (request or notification)
        # - 'result' or 'error' field (response)
        has_method = "method" in message
        has_result = "result" in message
        has_error = "error" in message

        if not (has_method or has_result or has_error):
            raise ProtocolError(
                "Invalid JSON-RPC message: must have 'method', 'result', or 'error'"
            )

        # Responses must have an 'id' field (can be null)
        if (has_result or has_error) and "id" not in message:
            raise ProtocolError("JSON-RPC response missing 'id' field")

        # Per JSON-RPC 2.0: response must have exactly one of result or error
        if has_result and has_error:
            raise ProtocolError(
                "Invalid JSON-RPC response: cannot have both 'result' and 'error'"
            )

        # Validate ID type if present (per JSON-RPC 2.0: string, number, or null)
        if "id" in message:
            id_value = message["id"]
            if not (id_value is None or isinstance(id_value, (str, int, float))):
                raise ProtocolError(
                    f"Invalid JSON-RPC id type: {type(id_value).__name__}"
                )

    def _handle_notification(self, message: dict) -> None:
        """
        Handle an incoming notification from the server.

        Default behavior logs to stderr. Override on_notification for custom handling.
        """
        method = message.get("method", "unknown")
        params = message.get("params", {})
        if not isinstance(params, dict):
            _log_stderr(
                f"[warning] Notification '{method}' has non-object params: {type(params).__name__}\n"
            )
            params = {}

        # Default handling for common notifications
        if method in ("notifications/message", "notifications/log", "$/logMessage"):
            level = params.get("level", "info")
            data = params.get("data") if "data" in params else params.get("message", "")
            logger = params.get("logger", "server")
            _log_stderr(f"[{level}] [{logger}] {data}\n")
        elif method == "notifications/progress":
            progress = params.get("progress", 0)
            total = params.get("total", 100)
            _log_stderr(f"[progress] {progress}/{total}\n")
        else:
            # Log all other notifications, truncating long params for readability
            params_str = json.dumps(params)
            if len(params_str) > 200:  # Truncate to avoid flooding logs
                params_str = params_str[:200] + "..."
            _log_stderr(f"[notification] {method}: {params_str}\n")

        # Call user callback if registered
        if self.on_notification:
            try:
                self.on_notification(method, params)
            except Exception as e:
                _log_stderr(f"[error] Exception in notification callback: {e}\n")

    def _wait_for_response(
        self,
        request_id: Any,
        timeout: Optional[float] = _TIMEOUT_NOT_SPECIFIED
    ) -> dict:
        """
        Wait for a response with the specified request ID.

        Handles notifications that arrive while waiting without blocking.
        Uses a coordination pattern to avoid head-of-line blocking when multiple
        threads are waiting for different responses.

        Args:
            request_id: The request ID to wait for.
            timeout: Maximum time to wait in seconds. If not specified, uses
                    self.default_timeout. Pass None explicitly to wait forever.

        Returns:
            The JSON-RPC response message.

        Raises:
            MCPTimeoutError: If timeout expires before response arrives.
            TransportError: On transport-level errors.
        """
        # Normalize request_id for consistent dictionary lookups
        normalized_request_id = self._normalize_id(request_id)

        # Calculate deadline:
        # - _TIMEOUT_NOT_SPECIFIED: use default_timeout
        # - None: wait forever (no deadline)
        # - float: use specified timeout
        if timeout is _TIMEOUT_NOT_SPECIFIED:
            deadline = time.monotonic() + self.default_timeout
        elif timeout is None:
            deadline = None
        else:
            deadline = time.monotonic() + timeout

        while True:
            # Calculate remaining time (None means no limit)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPTimeoutError(
                        f"Timeout waiting for response to request {request_id}"
                    )
            else:
                remaining = None

            # Check cache and coordinate receiver role under lock
            with self._response_cv:
                # Check if our response is already cached
                if normalized_request_id in self._pending_responses:
                    return self._pending_responses.pop(normalized_request_id)

                # If another thread is receiving, wait for it to finish
                if self._receiving:
                    # Wait with a short timeout to allow periodic deadline checks
                    wait_timeout = 1.0 if remaining is None else min(remaining, 1.0)
                    self._response_cv.wait(timeout=wait_timeout)
                    continue  # Loop back to check cache

                # Become the receiver
                self._receiving = True

            # We are the receiver - call transport.receive() outside the lock
            # to allow other threads to check the cache
            message = None
            try:
                recv_timeout = 1.0 if remaining is None else min(remaining, 1.0)
                message = self.transport.receive(timeout=recv_timeout)
            finally:
                # Release receiver role and notify waiters
                with self._response_cv:
                    self._receiving = False
                    self._response_cv.notify_all()

            if message is None:
                # Timeout on this receive iteration - loop back to check deadline
                continue

            # Validate message structure
            self._validate_message(message)

            # Determine message type and handle accordingly
            has_id = "id" in message
            has_method = "method" in message

            if has_method and has_id:
                # Server request - has both method and id, expects a response
                # We don't implement any server-to-client requests, so respond
                # with "method not found" error (-32601)
                self._send_error_response(
                    message["id"],
                    -32601,
                    f"Method not found: {message['method']}"
                )
            elif has_method:
                # Notification - has method but no id
                self._handle_notification(message)
            elif has_id:
                # Response - has id but no method
                msg_id = message["id"]
                normalized_msg_id = self._normalize_id(msg_id)
                if normalized_msg_id == normalized_request_id:
                    return message
                else:
                    # Cache response for a different request and notify waiters
                    with self._response_cv:
                        # Limit cache size to prevent unbounded memory growth
                        if len(self._pending_responses) >= self.MAX_PENDING_RESPONSES:
                            # Remove oldest entry (first inserted)
                            oldest_key = next(iter(self._pending_responses))
                            _log_stderr(
                                f"[warning] Dropping cached response due to overflow: "
                                f"id={oldest_key}\n"
                            )
                            del self._pending_responses[oldest_key]
                        self._pending_responses[normalized_msg_id] = message
                        self._response_cv.notify_all()  # Wake waiters to check cache

    def _call(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = _TIMEOUT_NOT_SPECIFIED
    ) -> Any:
        """
        Make a JSON-RPC call and return the result.

        Args:
            method: The RPC method to call.
            params: Optional method parameters.
            timeout: Request timeout in seconds. If not specified, uses
                    self.default_timeout. Pass None explicitly to wait forever.

        Returns:
            The 'result' field from the response.

        Raises:
            RPCError: If the server returns an error response.
            MCPTimeoutError: If timeout expires.
            TransportError: On transport-level errors.
        """
        request_id = self._send_request(method, params)
        response = self._wait_for_response(request_id, timeout=timeout)

        if "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                raise RPCError(
                    code=error.get("code", -1),
                    message=error.get("message", "Unknown error"),
                    data=error.get("data")
                )
            else:
                raise RPCError(code=-1, message=str(error))

        return response.get("result")

    def _connect(self, timeout: float) -> dict[str, Any]:
        """Establish connection and perform MCP protocol handshake."""
        # Connect the underlying transport
        self.transport.connect()

        # Build initialize request parameters
        # Note: We only advertise capabilities we actually implement.
        # Currently no client capabilities are implemented.
        init_params = {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": self.client_name,
                "version": self.client_version
            }
        }

        # Send initialize request and wait for response
        result = self._call("initialize", init_params, timeout=timeout)

        # Validate response structure
        if not isinstance(result, dict):
            raise ProtocolError(
                f"Initialize result must be a dict, got {type(result).__name__}"
            )

        # Validate expected fields
        if "serverInfo" in result and not isinstance(result["serverInfo"], dict):
            raise ProtocolError("serverInfo must be a dict")
        if "capabilities" in result and not isinstance(result["capabilities"], dict):
            raise ProtocolError("capabilities must be a dict")

        # Store full initialize result for access to all fields (e.g., instructions)
        self._init_result = result

        # Validate protocol version compatibility
        server_version = result.get("protocolVersion")
        if server_version is None:
            raise ProtocolError("Server did not return protocolVersion in initialize response")
        if server_version != self.PROTOCOL_VERSION:
            raise ProtocolError(
                f"Protocol version mismatch: client supports {self.PROTOCOL_VERSION}, "
                f"server returned {server_version}"
            )

        # Send initialized notification to complete handshake
        # Include empty params for compatibility with strict MCP servers
        self._send_notification("notifications/initialized", {})

        with self._lock:
            self._connected = True
        return result

    @property
    def init_result(self) -> Optional[dict[str, Any]]:
        """Full initialize response from the server (includes all fields)."""
        return self._init_result

    @property
    def server_info(self) -> Optional[dict[str, Any]]:
        """Server information from the handshake (name, version)."""
        if self._init_result is None:
            return None
        return self._init_result.get("serverInfo", {})

    @property
    def server_capabilities(self) -> Optional[dict[str, Any]]:
        """Server capabilities from the handshake."""
        if self._init_result is None:
            return None
        return self._init_result.get("capabilities", {})

    @property
    def instructions(self) -> Optional[str]:
        """Server instructions from the handshake."""
        if self._init_result is None:
            return None
        return self._init_result.get("instructions")

    def list_prompts(self, timeout: Optional[float] = _TIMEOUT_NOT_SPECIFIED) -> list[dict[str, Any]]:
        """
        List available prompts from the server.

        Args:
            timeout: Request timeout in seconds. If not specified, uses
                    self.default_timeout. Pass None explicitly to wait forever.

        Returns:
            List of prompt definitions, each containing:
            - name: Prompt identifier
            - description: Optional human-readable description
            - arguments: Optional list of argument definitions

        Raises:
            MCPError: If not connected or on protocol errors.
        """
        self._check_connected()

        # Check if server supports prompts (only call if capability advertised)
        if not self.server_capabilities or not self.server_capabilities.get("prompts"):
            return []

        result = self._call("prompts/list", timeout=timeout)
        if not isinstance(result, dict):
            raise ProtocolError(f"prompts/list result must be a dict, got {type(result).__name__}")
        return result.get("prompts", [])

    def list_tools(self, timeout: Optional[float] = _TIMEOUT_NOT_SPECIFIED) -> list[dict[str, Any]]:
        """
        List available tools from the server.

        Args:
            timeout: Request timeout in seconds. If not specified, uses
                    self.default_timeout. Pass None explicitly to wait forever.

        Returns:
            List of tool definitions, each containing:
            - name: Tool identifier
            - description: Human-readable description
            - inputSchema: JSON Schema for tool arguments

        Raises:
            MCPError: If not connected or on protocol errors.
        """
        self._check_connected()

        # Check if server supports tools (only call if capability advertised)
        if not self.server_capabilities or not self.server_capabilities.get("tools"):
            return []

        result = self._call("tools/list", timeout=timeout)
        if not isinstance(result, dict):
            raise ProtocolError(f"tools/list result must be a dict, got {type(result).__name__}")
        return result.get("tools", [])

    def call_tool(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = _TIMEOUT_NOT_SPECIFIED
    ) -> dict[str, Any]:
        """
        Call a tool on the server.

        Args:
            name: The tool name to invoke.
            arguments: Tool arguments as a dictionary.
            timeout: Request timeout in seconds. If not specified, uses
                    self.default_timeout. Pass None explicitly to wait forever
                    for long-running tools.

        Returns:
            Tool result containing:
            - content: List of content items (text, images, etc.)
            - isError: Boolean indicating if tool execution failed

        Raises:
            MCPError: If not connected or on protocol errors.
            RPCError: If the tool returns an error.
            ValueError: If tool name is empty.
        """
        self._check_connected()

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string")

        params = {
            "name": name,
            "arguments": arguments or {}
        }

        return self._call("tools/call", params, timeout=timeout)

    def list_resources(self, timeout: Optional[float] = _TIMEOUT_NOT_SPECIFIED) -> list[dict[str, Any]]:
        """
        List available resources from the server.

        Args:
            timeout: Request timeout in seconds. If not specified, uses
                    self.default_timeout. Pass None explicitly to wait forever.

        Returns:
            List of resource definitions, each containing:
            - uri: Resource identifier
            - name: Human-readable name
            - description: Optional description
            - mimeType: Optional MIME type

        Raises:
            MCPError: If not connected or on protocol errors.
        """
        self._check_connected()

        # Check if server supports resources (only call if capability advertised)
        if not self.server_capabilities or not self.server_capabilities.get("resources"):
            return []

        result = self._call("resources/list", timeout=timeout)
        if not isinstance(result, dict):
            raise ProtocolError(f"resources/list result must be a dict, got {type(result).__name__}")
        return result.get("resources", [])

    def read_resource(
        self,
        uri: str,
        timeout: Optional[float] = _TIMEOUT_NOT_SPECIFIED
    ) -> dict[str, Any]:
        """
        Read a resource from the server.

        Args:
            uri: The resource URI to read.
            timeout: Request timeout in seconds. If not specified, uses
                    self.default_timeout. Pass None explicitly to wait forever.

        Returns:
            Resource contents with 'contents' list containing items with:
            - uri: The resource URI
            - mimeType: Content MIME type
            - text or blob: The actual content

        Raises:
            MCPError: If not connected or on protocol errors.
            ValueError: If URI is empty or not a string.
        """
        self._check_connected()

        if not isinstance(uri, str) or not uri.strip():
            raise ValueError("Resource URI must be a non-empty string")

        return self._call("resources/read", {"uri": uri}, timeout=timeout)

    def get_prompt(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = _TIMEOUT_NOT_SPECIFIED
    ) -> dict[str, Any]:
        """
        Get a specific prompt from the server.

        Args:
            name: The prompt name to retrieve.
            arguments: Optional arguments to fill prompt template.
            timeout: Request timeout in seconds. If not specified, uses
                    self.default_timeout. Pass None explicitly to wait forever.

        Returns:
            Prompt content with:
            - description: Optional description
            - messages: List of message objects

        Raises:
            MCPError: If not connected or on protocol errors.
            ValueError: If prompt name is empty.
        """
        self._check_connected()

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Prompt name must be a non-empty string")

        params = {
            "name": name,
            "arguments": arguments or {}
        }

        return self._call("prompts/get", params, timeout=timeout)

    def ping(self, timeout: Optional[float] = _TIMEOUT_NOT_SPECIFIED) -> bool:
        """
        Send a ping to check if the server is responsive.

        Args:
            timeout: Request timeout in seconds. If not specified, uses
                    self.default_timeout. For quick health checks, pass a
                    shorter value like 10.0.

        Returns:
            True if server responded, False otherwise.
        """
        if not self._is_connected():
            return False

        try:
            self._call("ping", timeout=timeout)
            return True
        except MCPError:
            return False

    def close(self) -> None:
        """
        Close the client connection.

        Safe to call multiple times.
        """
        with self._lock:
            self._connected = False
        try:
            self.transport.close()
        except OSError:
            # Expected error during close - socket/pipe already closed
            pass

    def __enter__(self) -> "MCPClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.close()
        return False


# ============================================================================
# Factory Functions
# ============================================================================

def create_stdio_client(
    command: list[str],
    env: Optional[dict[str, str]] = None,
    client_name: str = "mcp-python-client",
    client_version: str = "1.0.0",
    timeout: float = 300.0,
    forward_stderr: bool = True
) -> MCPClient:
    """
    Create and connect to an MCP server using stdio transport.

    The server is spawned as a subprocess and communication occurs
    via stdin/stdout pipes. Connection and handshake happen immediately.

    Args:
        command: Command to run the MCP server (e.g., ["python", "server.py"]).
        env: Additional environment variables for the subprocess.
        client_name: Client name for protocol handshake.
        client_version: Client version for protocol handshake.
        timeout: Default timeout in seconds for all operations (default 300).
                Individual method calls can override this.
        forward_stderr: If True (default), forward server stderr to client stderr.
                       Set to False to silence server stderr output.

    Returns:
        Connected MCPClient instance.

    Example:
        client = create_stdio_client(["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
        tools = client.list_tools()
    """
    transport = StdioTransport(command, env, forward_stderr)
    return MCPClient(transport, client_name, client_version, timeout)


def create_sse_client(
    url: str,
    headers: Optional[dict[str, str]] = None,
    client_name: str = "mcp-python-client",
    client_version: str = "1.0.0",
    timeout: float = 300.0
) -> MCPClient:
    """
    Create and connect to an MCP server using SSE transport.

    Connects to a remote MCP server over HTTP/HTTPS using the SSE protocol.
    Connection and handshake happen immediately.

    Args:
        url: The SSE endpoint URL (e.g., "http://localhost:8080/sse").
        headers: Optional HTTP headers for authentication or other purposes.
        client_name: Client name for protocol handshake.
        client_version: Client version for protocol handshake.
        timeout: Default timeout in seconds for all operations (default 300).
                Individual method calls can override this.

    Returns:
        Connected MCPClient instance.

    Example:
        client = create_sse_client("http://localhost:3000/sse")
        tools = client.list_tools()

        # With authentication:
        client = create_sse_client(
            "http://localhost:3000/sse",
            headers={"Authorization": "Bearer token123"}
        )
    """
    transport = SSETransport(url, headers, timeout)
    return MCPClient(transport, client_name, client_version, timeout)


# ============================================================================
# Main / Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MCP Client - Model Context Protocol Client Library")
    print("=" * 60)
    print()
    print("This module provides a complete MCP client implementation")
    print("using only the Python standard library.")
    print()
    print("Supported transports:")
    print("  - Stdio: Subprocess with stdin/stdout pipes")
    print("  - SSE:   Server-Sent Events over HTTP/HTTPS")
    print()
    print("Quick start:")
    print("  client = create_stdio_client(['python', 'your_server.py'])")
    print("  tools = client.list_tools()")
    print("  result = client.call_tool('tool_name', {'arg': 'value'})")
    print("  client.close()")
