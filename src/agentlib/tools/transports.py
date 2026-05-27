"""
Transport backends for SubREPL and ToolREPL workers.

Transports own worker lifecycle and expose queue-like endpoints used by the
REPL application protocol.
"""

from __future__ import annotations

import os
import signal
import sys
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, Callable, Optional, Tuple


# Transport compatibility contract:
# - Create and own the worker lifecycle for a REPL session.
# - Expose `queues` as an ordered tuple of queue-like endpoints in the order
#   expected by the worker target. For SubREPL this is
#   (cmd_queue, output_queue); for ToolREPL this is
#   (cmd_queue, output_queue, tool_request_queue, tool_response_queue).
# - Each queue-like endpoint must implement put(item), get(timeout=None), and
#   get_nowait().
# - get(timeout=...) and get_nowait() must raise queue.Empty when no item is
#   available, matching multiprocessing.Queue behavior used throughout the REPL
#   code.
# - Queue payloads are opaque application protocol objects. The transport moves
#   them unchanged and must preserve FIFO ordering per queue.
# - start(), is_alive(), interrupt(), terminate(), kill(), join(), and close()
#   provide lifecycle controls without callers needing to know whether the
#   worker is a local Process, a fresh interpreter, a remote sandbox, etc.
# - close() should attempt graceful shutdown by sending None on the command
#   queue before escalating to terminate().
# REPLAgent subclasses can opt into an alternate transport by setting
# `repl_transport` before the REPL session is created:
#
#     from agentlib import REPLAgent
#
#     class MyAgent(REPLAgent):
#         repl_transport = StdioSubprocessTransport
#
# Default transport selection for direct SubREPL use currently happens at the
# import site. For local SubREPL experiments, monkey patch the importing module
# before creating the repl:
#
#     import agentlib.tools.subrepl as subrepl
#     from agentlib.tools.transports import StdioSubprocessTransport
#
#     subrepl.MultiprocessingTransport = StdioSubprocessTransport
#
# This keeps MultiprocessingTransport as the normal default while allowing
# selected agents/processes to opt into a sandbox/remote-capable transport.


class MultiprocessingTransport:
    """Worker lifecycle plus queue endpoints backed by multiprocessing."""

    def __init__(
        self,
        target: Callable[..., None],
        args: Tuple[Any, ...] = (),
        queue_count: int = 2,
        maxsize: int = 1,
    ) -> None:
        self._target = target
        self._args = args
        self._queue_count = queue_count
        self._maxsize = maxsize
        self.queues: Tuple[Queue, ...] = ()
        self.worker: Optional[Process] = None

    def start(self) -> None:
        self.queues = tuple(Queue(maxsize=self._maxsize) for _ in range(self._queue_count))
        self.worker = Process(
            target=self._target,
            args=(*self.queues, *self._args),
            daemon=True,
        )
        self.worker.start()

    def is_alive(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def interrupt(self) -> None:
        if self.worker is None or self.worker.pid is None:
            return
        try:
            os.kill(self.worker.pid, signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass

    def kill(self) -> None:
        if self.worker is None or self.worker.pid is None:
            return
        try:
            os.kill(self.worker.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    def join(self, timeout: Optional[float] = None) -> None:
        if self.worker is not None:
            self.worker.join(timeout=timeout)

    def terminate(self) -> None:
        self.kill()
        self.join(timeout=1.0)

    def close(self) -> None:
        if self.queues:
            try:
                self.queues[0].put(None)
                self.join(timeout=1.0)
            except Exception:
                pass
        if self.is_alive():
            self.terminate()



class StdioSubprocessTransport:
    """Worker lifecycle plus queue endpoints backed by subprocess stdio."""

    LOADER = (
        'import sys,struct;'
        'exec(sys.stdin.buffer.read(struct.unpack("!I",sys.stdin.buffer.read(4))[0]))'
    )

    _CHANNEL_NAMES = ("cmd", "output", "tool_request", "tool_response")

    def __init__(
        self,
        target: Callable[..., None],
        args: Tuple[Any, ...] = (),
        queue_count: int = 2,
        maxsize: int = 1,
        executable: Optional[str] = None,
    ) -> None:
        self._target = target
        self._args = args
        self._queue_count = queue_count
        self._maxsize = maxsize
        self._executable = executable or sys.executable
        self.queues: Tuple[Any, ...] = ()
        self.worker: Any = None
        self._mux: Any = None
        self._stderr_thread: Any = None
        self._stderr_chunks: list[bytes] = []

    @staticmethod
    def _bootstrap_source(
        target_name: str,
        runtime_blob: str,
        args_blob: str,
        queue_count: int,
        maxsize: int,
    ) -> bytes:
        source = """
import base64
import os
import pickle
import queue
import struct
import sys
import threading

_READY = b"AGENTLIB_REPL_WORKER" + b"_READY\\n"
_CHANNEL_NAMES = ("cmd", "output", "tool_request", "tool_response")


class _StdioMux:
    def __init__(self, input_stream, output_stream, channel_names, maxsize):
        self._input = input_stream
        self._output = output_stream
        self._write_lock = threading.Lock()
        self._queues = {name: queue.Queue(maxsize=maxsize) for name in channel_names}
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def queue(self, name):
        return _StdioQueue(self, name)

    def send(self, channel, item):
        payload = pickle.dumps((channel, item), protocol=pickle.HIGHEST_PROTOCOL)
        with self._write_lock:
            self._output.write(struct.pack("!I", len(payload)))
            self._output.write(payload)
            self._output.flush()

    def recv(self, channel, timeout=None):
        return self._queues[channel].get(timeout=timeout)

    def recv_nowait(self, channel):
        return self._queues[channel].get_nowait()

    def _read_exact(self, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = self._input.read(remaining)
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_loop(self):
        try:
            while True:
                header = self._read_exact(4)
                size = struct.unpack("!I", header)[0]
                channel, item = pickle.loads(self._read_exact(size))
                self._queues[channel].put(item)
        except BaseException as exc:
            self._closed = True
            for q in self._queues.values():
                try:
                    q.put_nowait(exc)
                except queue.Full:
                    pass
            os._exit(1)


class _StdioQueue:
    def __init__(self, mux, channel):
        self._mux = mux
        self._channel = channel

    def put(self, item):
        self._mux.send(self._channel, item)

    def get(self, timeout=None):
        item = self._mux.recv(self._channel, timeout=timeout)
        if isinstance(item, BaseException):
            raise item
        return item

    def get_nowait(self):
        item = self._mux.recv_nowait(self._channel)
        if isinstance(item, BaseException):
            raise item
        return item


class _CapturedWriter:
    def __init__(self, output_queue, original):
        self._output_queue = output_queue
        self._original = original

    def write(self, text):
        if text:
            self._output_queue.put(("output", text))
        return len(text)

    def flush(self):
        pass

    def fileno(self):
        return self._original.fileno()


_protocol_stdin = sys.stdin.buffer
_protocol_stdout = sys.stdout.buffer
_channel_names = _CHANNEL_NAMES[:__QUEUE_COUNT__]
_mux = _StdioMux(_protocol_stdin, _protocol_stdout, _channel_names, __MAXSIZE__)

_queues = tuple(_mux.queue(name) for name in _channel_names)

sys.stdout = _CapturedWriter(_queues[1], sys.stdout)
sys.stderr = _CapturedWriter(_queues[1], sys.stderr)

_protocol_stdout.write(_READY)
_protocol_stdout.flush()

_runtime_globals = {"__name__": "__agentlib_repl_worker__", "__file__": "<agentlib-repl-worker>"}
exec(base64.b64decode(__RUNTIME_BLOB__).decode(), _runtime_globals)
_args = pickle.loads(base64.b64decode(__ARGS_BLOB__))

_runtime_globals[__TARGET_NAME__](*_queues, *_args)
"""
        source = source.replace("__QUEUE_COUNT__", repr(queue_count))
        source = source.replace("__MAXSIZE__", repr(maxsize))
        source = source.replace("__TARGET_NAME__", repr(target_name))
        source = source.replace("__RUNTIME_BLOB__", repr(runtime_blob))
        source = source.replace("__ARGS_BLOB__", repr(args_blob))
        return source.encode()

    @staticmethod
    def _build_worker_payload(target: Callable[..., None]) -> tuple[str, bytes]:
        import inspect
        import textwrap
        import types

        modules: dict[str, types.ModuleType] = {}
        constants: dict[str, Any] = {}
        sources: dict[int, str] = {}
        visiting: set[int] = set()
        emitted: list[str] = []

        def is_simple(value: Any) -> bool:
            if value is None or isinstance(value, (str, bytes, int, float, bool)):
                return True
            if isinstance(value, tuple):
                return all(is_simple(item) for item in value)
            if isinstance(value, list):
                return all(is_simple(item) for item in value)
            if isinstance(value, dict):
                return all(isinstance(key, str) and is_simple(item) for key, item in value.items())
            return False

        def should_bundle(value: Any) -> bool:
            module = getattr(value, "__module__", "")
            return module == target.__module__ or module.startswith("agentlib.")

        def include_object(value: Any) -> None:
            ident = id(value)
            if ident in sources:
                return
            if ident in visiting:
                return
            if not (inspect.isfunction(value) or inspect.isclass(value)):
                return
            if not should_bundle(value):
                return

            visiting.add(ident)
            try:
                names = set()
                if inspect.isfunction(value):
                    names.update(value.__code__.co_names)
                    namespace = value.__globals__
                else:
                    namespace = vars(__import__(value.__module__, fromlist=["*"]))
                    for member in vars(value).values():
                        if inspect.isfunction(member):
                            names.update(member.__code__.co_names)

                for name in sorted(names):
                    if name not in namespace or name == value.__name__:
                        continue
                    dep = namespace[name]
                    if inspect.ismodule(dep):
                        modules[name] = dep
                    elif (inspect.isfunction(dep) or inspect.isclass(dep)) and should_bundle(dep):
                        include_object(dep)
                    elif is_simple(dep) and not name.startswith("__"):
                        constants[name] = dep

                source = textwrap.dedent(inspect.getsource(value)).strip()
            except (OSError, TypeError) as exc:
                raise RuntimeError(f"Cannot build portable stdio worker payload for {value!r}: {exc}") from exc
            finally:
                visiting.discard(ident)

            sources[ident] = source
            emitted.append(source)

        include_object(target)

        import_lines = ["from __future__ import annotations"]
        for name, module in sorted(modules.items()):
            module_name = module.__name__
            if module_name == name:
                import_lines.append(f"import {module_name}")
            else:
                import_lines.append(f"import {module_name} as {name}")

        constant_lines = [f"{name} = {value!r}" for name, value in sorted(constants.items())]
        payload = "\n".join(import_lines + [""] + constant_lines + [""] + emitted + [""])
        return target.__name__, payload.encode()

    class _Queue:
        def __init__(self, mux: Any, channel: str) -> None:
            self._mux = mux
            self._channel = channel

        def put(self, item: Any) -> None:
            self._mux.send(self._channel, item)

        def get(self, timeout: Optional[float] = None) -> Any:
            return self._mux.recv(self._channel, timeout=timeout)

        def get_nowait(self) -> Any:
            return self.get(timeout=0)

    class _Mux:
        def __init__(self, input_stream: Any, output_stream: Any, channel_names: Tuple[str, ...], maxsize: int) -> None:
            import queue
            import threading

            self._input = input_stream
            self._output = output_stream
            self._write_lock = threading.Lock()
            self._queues = {name: queue.Queue(maxsize=maxsize) for name in channel_names}
            self._closed = False
            self._error: Optional[BaseException] = None
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()

        def queue(self, name: str) -> "StdioSubprocessTransport._Queue":
            return StdioSubprocessTransport._Queue(self, name)

        def send(self, channel: str, item: Any) -> None:
            import pickle
            import struct

            if self._closed:
                raise BrokenPipeError("stdio transport is closed")
            payload = pickle.dumps((channel, item), protocol=pickle.HIGHEST_PROTOCOL)
            with self._write_lock:
                self._output.write(struct.pack("!I", len(payload)))
                self._output.write(payload)
                self._output.flush()

        def recv(self, channel: str, timeout: Optional[float] = None) -> Any:
            item = self._queues[channel].get(timeout=timeout)
            if isinstance(item, BaseException):
                raise item
            return item

        def _read_exact(self, size: int) -> bytes:
            chunks = []
            remaining = size
            while remaining:
                chunk = self._input.read(remaining)
                if not chunk:
                    raise EOFError
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        def _read_loop(self) -> None:
            import pickle
            import queue
            import struct

            try:
                while True:
                    header = self._read_exact(4)
                    size = struct.unpack("!I", header)[0]
                    channel, item = pickle.loads(self._read_exact(size))
                    self._queues[channel].put(item)
            except BaseException as exc:
                self._closed = True
                self._error = exc
                for q in self._queues.values():
                    try:
                        q.put_nowait(exc)
                    except queue.Full:
                        pass

    def start(self) -> None:
        import base64
        import pickle
        import subprocess
        import threading

        target_name, runtime_source = self._build_worker_payload(self._target)
        runtime_blob = base64.b64encode(runtime_source).decode()
        args_blob = base64.b64encode(pickle.dumps(self._args, protocol=pickle.HIGHEST_PROTOCOL)).decode()
        bootstrap = self._bootstrap_source(
            target_name,
            runtime_blob,
            args_blob,
            self._queue_count,
            self._maxsize,
        )

        self.worker = subprocess.Popen(
            [self._executable, "-u", "-c", self.LOADER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self.worker.stdin is None or self.worker.stdout is None:
            raise RuntimeError("Failed to open subprocess stdio pipes")

        self.worker.stdin.write(len(bootstrap).to_bytes(4, "big"))
        self.worker.stdin.write(bootstrap)
        self.worker.stdin.flush()

        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

        self._wait_for_ready(self.worker.stdout)

        channel_names = self._CHANNEL_NAMES[:self._queue_count]
        self._mux = self._Mux(self.worker.stdout, self.worker.stdin, channel_names, self._maxsize)
        self.queues = tuple(self._mux.queue(name) for name in channel_names)

    def _read_stderr(self) -> None:
        if self.worker is None or self.worker.stderr is None:
            return
        while True:
            chunk = self.worker.stderr.read(4096)
            if not chunk:
                return
            self._stderr_chunks.append(chunk)

    def _wait_for_ready(self, stdout: Any, timeout: float = 10.0) -> None:
        import queue
        import threading

        marker = b"AGENTLIB_REPL_WORKER" + b"_READY\n"
        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)

        def read_until_marker() -> None:
            buf = bytearray()
            try:
                while marker not in buf:
                    chunk = stdout.read(1)
                    if not chunk:
                        raise EOFError("stdio worker stdout closed before readiness marker")
                    buf.extend(chunk)
                result_queue.put(None)
            except BaseException as exc:
                result_queue.put(exc)

        thread = threading.Thread(target=read_until_marker, daemon=True)
        thread.start()

        try:
            result = result_queue.get(timeout=timeout)
        except Empty:
            err = b"".join(self._stderr_chunks).decode(errors="replace")
            raise RuntimeError(f"Timeout waiting for stdio worker readiness marker: {err}")

        if isinstance(result, BaseException):
            err = b"".join(self._stderr_chunks).decode(errors="replace")
            raise RuntimeError(f"stdio worker exited before ready: {err}") from result

    def is_alive(self) -> bool:
        return self.worker is not None and self.worker.poll() is None

    def interrupt(self) -> None:
        if self.worker is None:
            return
        try:
            self.worker.send_signal(signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass

    def kill(self) -> None:
        if self.worker is None:
            return
        try:
            self.worker.kill()
        except (ProcessLookupError, OSError):
            pass

    def join(self, timeout: Optional[float] = None) -> None:
        if self.worker is not None:
            try:
                self.worker.wait(timeout=timeout)
            except TimeoutError:
                pass
            except Exception:
                pass

    def terminate(self) -> None:
        self.kill()
        self.join(timeout=1.0)

    def close(self) -> None:
        if self.queues:
            try:
                self.queues[0].put(None)
                self.join(timeout=1.0)
            except Exception:
                pass
        if self.is_alive():
            self.terminate()
