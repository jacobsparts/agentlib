"""
REPLAgent: Agent paradigm where LLM writes Python code instead of tool calls.

Instead of JSON tool schemas and structured tool calls, the agent operates
inside a Python REPL environment. Tools defined with @BaseAgent.tool become
callable functions in the REPL that bridge back to the main process.

Basic Usage
-----------

    from agentlib import REPLAgent

    class MyAgent(REPLAgent):
        system = "You are a helpful assistant."

        @REPLAgent.tool
        def read_file(self, path: str = "Path to read"):
            '''Read a file from disk.'''
            return Path(path).read_text()

    result = MyAgent().run("Read config.json")

The LLM sees a Python REPL and writes code directly. No markdown blocks -
the response IS Python code. Tools become callable functions in the REPL.

Built-in Functions
------------------

emit(value, release=False) - Emit output. release=True yields control.

How It Works
------------

1. LLM writes Python code as its response
2. Code executes statement-by-statement in isolated subprocess
3. Tool calls bridge back to main process via queues
4. Output streams back to LLM as REPL feedback
5. Loop continues until emit(..., release=True) is called
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import signal
import sys
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, Callable, Optional, Union

logger = logging.getLogger('agentlib')

from agentlib.agent import BaseAgent, _CompleteException
from agentlib.client import BadRequestError


class _InterruptedError(KeyboardInterrupt):
    """Raised when user interrupts execution with Ctrl+C."""
    def __init__(self, output: str = ""):
        self.output = output
        super().__init__("Interrupted by user")


from agentlib.tools.subrepl import (
    SubREPL,
    _format_echo,
    _split_into_statements,
)
from agentlib.tools.source_extract import extract_method_source as _extract_tool_source


# =============================================================================
# Syntax Correction Helpers
# =============================================================================

def fix_triple_quote_conflict(code: str) -> str:
    '''
    Fix triple-quote conflicts where outer """ contains inner """ docstrings.

    When an LLM writes code like:
        print("""
            def foo():
                """docstring"""
        """)

    The inner """ prematurely closes the outer string. This function detects
    and fixes such cases by converting outer """ to single quotes.

    Returns the fixed code, or original if no fix needed/possible.
    '''
    # Check if code already compiles
    try:
        compile(code, '<repl>', 'exec')
        return code
    except SyntaxError:
        pass

    # Find all """ positions
    positions = []
    i = 0
    while True:
        pos = code.find('"""', i)
        if pos == -1:
            break
        positions.append(pos)
        i = pos + 3

    if len(positions) < 4:
        return code  # Need at least 4 (outer open/close + inner open/close)

    # Try swapping first and last """ to '''
    first_pos = positions[0]
    last_pos = positions[-1]

    fixed = (
        code[:first_pos] + "'''" +
        code[first_pos + 3:last_pos] +
        "'''" + code[last_pos + 3:]
    )

    try:
        compile(fixed, '<repl>', 'exec')
        return fixed
    except SyntaxError:
        pass

    return code


# =============================================================================
# Shared REPL builtins code (used by both ToolREPL and SandboxedToolREPL)
# =============================================================================
# These functions are injected into the subprocess. They rely on transport-specific
# implementations of _send_output() being injected first.

BUILTINS_CODE = '''
# Request ID counter for matching requests with responses
_request_id = 0

# Tagged print for top-level REPL output (normal print stays available for functions)
def _print(*args, **kwargs):
    """Print with output tagging for display control."""
    import io
    # If printing to a file, use original print
    if kwargs.get('file') is not None:
        return print(*args, **kwargs)
    # Capture output and send tagged
    buf = io.StringIO()
    print(*args, **{**kwargs, 'file': buf})
    _send_output("print", buf.getvalue())

def emit(value, release=False):
    """Emit a value to the output.

    Args:
        value: The value to emit.
        release: If True, release control to the user (for questions or completion).
                 If False (default), the value is emitted but execution continues.
    """
    global _request_id
    _request_id += 1
    req_id = _request_id

    # Use different message types: "emit" for release=True (shown at turn end),
    # "progress" for release=False (shown inline immediately)
    msg_type = "emit" if release else "progress"
    _send_output(msg_type, str(value) + "\\n")
    import json as _json
    _send_tool_request(_json.dumps({
        "tool": "__emit__",
        "args": {"value": value, "release": release},
        "request_id": req_id
    }))
    _wait_for_ack(req_id)

def _wait_for_ack(expected_id):
    """Wait for ACK with matching request_id, capturing reply data."""
    import json as _json
    _result = None
    _error = None
    while True:
        raw = _recv_tool_response()
        # Parse JSON if string
        if isinstance(raw, str):
            try:
                msg = _json.loads(raw)
            except (ValueError, _json.JSONDecodeError):
                continue  # Not JSON, discard
        elif isinstance(raw, dict):
            msg = raw
        else:
            continue  # Unknown format, discard

        # Check for matching request_id
        if msg.get("request_id") != expected_id:
            continue  # Wrong ID, discard stale message

        msg_type = msg.get("type")
        if msg_type == "reply":
            # Capture result/error from reply
            _result = msg.get("result")
            _error = msg.get("error")
        elif msg_type == "ack":
            # ACK received - return result or raise error
            if _error is not None:
                raise Exception(_error)
            return _result
'''

# Queue-based transport for ToolREPL
QUEUE_TRANSPORT_CODE = '''
def _send_output(msg_type, data):
    _output_queue.put((msg_type, data))
def _send_tool_request(msg):
    _tool_request_queue.put(msg)
def _recv_tool_response():
    return _tool_response_queue.get()
'''

# Socket-based transport for SandboxedToolREPL (adds _send_output only;
# _send_tool_request and _recv_tool_response are defined separately due to complexity)
SOCKET_SEND_OUTPUT_CODE = '''
def _send_output(msg_type, data):
    _send_msg(_sock, (msg_type, data))
'''


class _StreamingWriter:
    """Sends output to queue in real-time. Replaces sys.stdout/stderr."""

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


# ---------------------------------------------------------------------------
# Tool-aware worker process
# ---------------------------------------------------------------------------

def _tool_worker_main(
    cmd_queue: Queue,
    output_queue: Queue,
    tool_request_queue: Queue,
    tool_response_queue: Queue,
) -> None:
    """
    Worker process that can make tool calls back to main process.

    Tool stubs in repl_locals use the request/response queues to
    execute tools in the main process where the actual implementations live.
    """
    repl_locals: dict[str, Any] = {
        '_output_queue': output_queue,
        '_tool_request_queue': tool_request_queue,
        '_tool_response_queue': tool_response_queue,
    }

    def sigint_handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, sigint_handler)

    while True:
        try:
            msg = cmd_queue.get()

            if msg is None:
                break

            # Commands can be (seq_id, cmd) tuples or plain strings (legacy/injection)
            if isinstance(msg, tuple):
                seq_id, cmd = msg
            else:
                seq_id, cmd = None, msg

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = _StreamingWriter(output_queue, old_stdout)
            sys.stderr = _StreamingWriter(output_queue, old_stderr)

            had_error = False
            try:
                try:
                    # Parse and execute each node, displaying expression results
                    tree = ast.parse(cmd, "<repl>", "exec")
                    for node in tree.body:
                        if isinstance(node, ast.Expr):
                            # Expression statement - eval and display result
                            code_obj = compile(ast.Expression(node.value), "<repl>", "eval")
                            result = eval(code_obj, repl_locals)
                            if result is not None:
                                print(repr(result))
                        else:
                            # Other statement - just exec
                            mod = ast.Module(body=[node], type_ignores=[])
                            code_obj = compile(mod, "<repl>", "exec")
                            exec(code_obj, repl_locals)
                except SyntaxError as e:
                    had_error = True
                    sys.stderr.write(f"  File \"<repl>\", line {e.lineno}\n")
                    if e.text:
                        sys.stderr.write(f"    {e.text}")
                        if e.offset:
                            sys.stderr.write(" " * (e.offset + 3) + "^\n")
                    sys.stderr.write(f"SyntaxError: {e.msg}\n")

            except KeyboardInterrupt:
                had_error = True
                sys.stderr.write("\nKeyboardInterrupt\n")
            except Exception as e:
                had_error = True
                import traceback
                tb = e.__traceback__
                while tb is not None and tb.tb_frame.f_code.co_filename != "<repl>":
                    tb = tb.tb_next
                if tb is not None:
                    sys.stderr.write("Traceback (most recent call last):\n")
                    sys.stderr.write("".join(traceback.format_tb(tb)))
                sys.stderr.write(f"{type(e).__name__}: {e}\n")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            output_queue.put(("done", (seq_id, had_error)))

        except KeyboardInterrupt:
            output_queue.put(("output", "\nKeyboardInterrupt\n"))
            output_queue.put(("done", (seq_id, True)))


# ---------------------------------------------------------------------------
# ToolREPL: SubREPL with tool call bridging
# ---------------------------------------------------------------------------

class ToolREPL(SubREPL):
    """
    SubREPL extended with bidirectional tool call support.

    Tools are injected as Python functions that call back to the main
    process for execution. This allows the LLM to use tools by writing
    normal Python code instead of structured tool calls.
    """

    def __init__(self, echo: bool = True) -> None:
        super().__init__(echo=echo)
        self._tool_request_queue: Optional[Queue] = None
        self._tool_response_queue: Optional[Queue] = None
        self._tools_injected: bool = False
        self._cmd_seq: int = 0  # Command sequence number for detecting stale messages

    def _ensure_session(self) -> None:
        """Override to use tool-aware worker and add tool queues."""
        if self._worker is None or not self._worker.is_alive():
            self._cmd_queue = Queue(maxsize=1)
            self._output_queue = Queue(maxsize=1)
            self._tool_request_queue = Queue(maxsize=1)
            self._tool_response_queue = Queue(maxsize=1)

            self._worker = Process(
                target=_tool_worker_main,
                args=(
                    self._cmd_queue,
                    self._output_queue,
                    self._tool_request_queue,
                    self._tool_response_queue,
                ),
                daemon=True,
            )
            self._worker.start()
            self._running = False
            self._tools_injected = False
            self._builtins_injected = False
            self._cmd_seq = 0  # Reset sequence on new session

    def inject_tools(self, tools: dict[str, tuple[Callable, Any]]) -> None:
        """
        Inject tool functions into the REPL.

        Tools marked with inject=True have their source extracted and run
        directly in the subprocess. Other tools get relay stubs that call
        back to the host process.

        Args:
            tools: Dict of tool_name -> (implementation, pydantic_spec)
        """
        self._ensure_session()

        if self._tools_injected:
            return

        # Inject common imports needed by tools
        self._inject_code('from pathlib import Path')

        for name, (impl, spec) in tools.items():
            should_inject = impl is not None and getattr(impl, '_tool_inject', False)
            
            if should_inject:
                code = _extract_tool_source(impl, name)
            else:
                # Relay stub - calls back to host process
                code = _generate_tool_stub(name, impl, spec)
            
            self._inject_code(code)

        self._tools_injected = True

    def inject_builtins(self) -> None:
        """Inject built-in functions like emit() and patched print()."""
        self._ensure_session()
        if getattr(self, '_builtins_injected', False):
            return
        self._inject_code(QUEUE_TRANSPORT_CODE)
        self._inject_code(BUILTINS_CODE)
        self._builtins_injected = True

    def _inject_code(self, code: str, timeout: float = 10.0) -> None:
        """Override to use queue directly and handle tool worker."""
        # Note: timeout parameter accepted for interface compatibility but
        # we use a fixed 5.0s poll timeout internally
        self._ensure_session()
        self._running = True
        self._cmd_queue.put(code)  # Plain string, worker uses seq_id=None

        while True:
            try:
                msg_type, msg_data = self._output_queue.get(timeout=5.0)
                if msg_type == "done":
                    # msg_data is (seq_id, had_error) - injection uses seq_id=None
                    self._running = False
                    return
                if msg_type == "output" and msg_data.strip():
                    # Unexpected output during injection - likely an error
                    print(f"[ToolREPL inject] {msg_data}", file=sys.stderr)
            except Empty:
                raise RuntimeError("Timeout injecting code into REPL")

    def poll_tool_request(self, timeout: float = 0.0) -> Optional[dict]:
        """
        Check for pending tool request from REPL.

        Returns:
            Tool request dict {"tool": name, "args": {...}} or None
        """
        if self._tool_request_queue is None:
            return None
        try:
            request_json = self._tool_request_queue.get(timeout=timeout)
            return json.loads(request_json)
        except Empty:
            return None

    def _check_worker_alive(self) -> None:
        """Check if worker is alive, drain queues and raise if dead."""
        if self._worker is None or not self._worker.is_alive():
            # Drain all queues to prevent blocking
            for q in (self._tool_response_queue, self._tool_request_queue, self._output_queue):
                if q is not None:
                    try:
                        while True:
                            q.get_nowait()
                    except Empty:
                        pass
            raise RuntimeError("Worker process died")

    def send_reply(self, request_id: int, result: Any = None, error: str = None) -> None:
        """Send application-level reply with result or error.

        Args:
            request_id: The request ID this reply is for.
            result: The result value (if successful).
            error: The error message (if failed).
        """
        if self._tool_response_queue is None:
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
        self._tool_response_queue.put(json.dumps(msg))

    def send_ack(self, request_id: int) -> None:
        """Send transport-level ACK to unblock the sender.

        Args:
            request_id: The request ID this ACK is for.
        """
        if self._tool_response_queue is None:
            raise RuntimeError("No active session")
        self._check_worker_alive()
        self._tool_response_queue.put(json.dumps({"type": "ack", "request_id": request_id}))


# ---------------------------------------------------------------------------
# Relay stub generation (for relay tools and MCP)
# ---------------------------------------------------------------------------

def _extract_stub_signature(name: str, impl: Optional[Callable], spec: Any) -> tuple[str, str, str]:
    """
    Extract signature info for generating relay stubs.
    
    Returns:
        (signature_str, docstring, args_dict) - components for stub template
    """
    param_names: list[str] = []
    doc_parts: list[str] = []
    required_sig_parts: list[str] = []
    optional_sig_parts: list[str] = []

    if impl is not None:
        # Extract signature from implementation
        sig = inspect.signature(impl)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_names.append(param_name)

            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, str):
                    # In agentlib, string defaults are descriptions, not actual defaults
                    doc_parts.append(f"        {param_name}: {param.default}")
                    # Check if type annotation is Optional (Union with None)
                    ann = param.annotation
                    is_optional = (
                        getattr(ann, '__origin__', None) is Union
                        and type(None) in getattr(ann, '__args__', ())
                    )
                    if is_optional:
                        optional_sig_parts.append(f"{param_name}=None")
                    else:
                        required_sig_parts.append(param_name)
                else:
                    # Actual default value
                    doc_parts.append(f"        {param_name}: (default: {repr(param.default)})")
                    optional_sig_parts.append(f"{param_name}={repr(param.default)}")
            else:
                doc_parts.append(f"        {param_name}")
                required_sig_parts.append(param_name)

        signature_parts = required_sig_parts + optional_sig_parts
        docstring = (impl.__doc__ or f"Call the {name} tool.").strip()

    else:
        # Dynamic tool - extract from Pydantic spec
        signature_parts = []
        if hasattr(spec, 'model_fields'):
            required_params = []
            optional_params = []

            for field_name, field_info in spec.model_fields.items():
                sanitized = field_name.lstrip('-')
                if sanitized and sanitized[0].isdigit():
                    sanitized = 'p_' + sanitized

                desc = field_info.description or ''
                doc_parts.append(f"        {sanitized}: {desc}")

                has_default = (
                    field_info.default is not None and
                    type(field_info.default).__name__ != 'PydanticUndefinedType'
                )

                if has_default:
                    optional_params.append((field_name, sanitized, repr(field_info.default)))
                elif field_info.is_required():
                    required_params.append((field_name, sanitized))
                else:
                    optional_params.append((field_name, sanitized, 'None'))

            for orig, sanitized in required_params:
                param_names.append((orig, sanitized))
                signature_parts.append(sanitized)
            for orig, sanitized, default in optional_params:
                param_names.append((orig, sanitized))
                signature_parts.append(f"{sanitized}={default}")

        docstring = f"Call the {name} tool."

    signature_str = ", ".join(signature_parts)

    if doc_parts:
        docstring += "\n\n    Args:\n" + "\n".join(doc_parts)

    # Build args dict - handle both tuple and string param_names
    if param_names and isinstance(param_names[0], tuple):
        args_items = ", ".join(f'"{orig}": {sanitized}' for orig, sanitized in param_names)
    else:
        args_items = ", ".join(f'"{n}": {n}' for n in param_names)
    args_dict = f"{{{args_items}}}"

    return signature_str, docstring, args_dict


def _generate_tool_stub(name: str, impl: Optional[Callable], spec: Any) -> str:
    """Generate relay stub using queue transport (for ToolREPL)."""
    sig, doc, args = _extract_stub_signature(name, impl, spec)

    # Check for files_param marker - adds path-to-bytes conversion
    files_param = getattr(impl, '_tool_files_param', None) if impl else None
    preprocess = ""
    if files_param:
        preprocess = f'''
    # Convert file paths to bytes
    from pathlib import Path as _Path
    def _read_file(f):
        if isinstance(f, bytes):
            return f
        p = _Path(f).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {{f}}")
        return p.read_bytes()
    if isinstance({files_param}, list):
        {files_param} = [_read_file(f) for f in {files_param}]
    elif isinstance({files_param}, str):
        {files_param} = [_read_file({files_param})]
'''

    return f'''
def {name}({sig}):
    """{doc}"""
    import json as _json
    {preprocess}
    def _serialize(x):
        if isinstance(x, bytes):
            import base64
            return {{"__b64__": base64.b64encode(x).decode()}}
        if isinstance(x, (list, tuple)):
            return [_serialize(i) for i in x]
        if isinstance(x, dict):
            return {{k: _serialize(v) for k, v in x.items()}}
        return x

    _args = {args}
    _safe_args = {{k: _serialize(v) for k, v in _args.items()}}

    _tool_request_queue.put(_json.dumps({{"tool": "{name}", "args": _safe_args}}))
    _response = _json.loads(_tool_response_queue.get())
    if "error" in _response:
        raise Exception(_response["error"])
    return _response["result"]
'''


def _generate_socket_relay_stub(name: str, impl: Optional[Callable], spec: Any) -> str:
    """Generate relay stub using socket transport (for SandboxedToolREPL)."""
    sig, doc, args = _extract_stub_signature(name, impl, spec)

    # Check for files_param marker - adds path-to-bytes conversion
    files_param = getattr(impl, '_tool_files_param', None) if impl else None
    preprocess = ""
    if files_param:
        preprocess = f'''
    # Convert file paths to bytes
    def _read_file(f):
        if isinstance(f, bytes):
            return f
        p = Path(f).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {{f}}")
        return p.read_bytes()
    if isinstance({files_param}, list):
        {files_param} = [_read_file(f) for f in {files_param}]
    elif isinstance({files_param}, str):
        {files_param} = [_read_file({files_param})]
'''

    return f'''
def {name}({sig}):
    """{doc}"""
    {preprocess}
    global _request_id
    _request_id += 1
    _req_id = _request_id

    def _serialize(x):
        if isinstance(x, bytes):
            import base64
            return {{"__b64__": base64.b64encode(x).decode()}}
        if isinstance(x, (list, tuple)):
            return [_serialize(i) for i in x]
        if isinstance(x, dict):
            return {{k: _serialize(v) for k, v in x.items()}}
        return x

    _args = {args}
    _safe_args = {{k: _serialize(v) for k, v in _args.items()}}

    _send_tool_request(_json.dumps({{"tool": "{name}", "args": _safe_args, "request_id": _req_id}}))
    return _wait_for_ack(_req_id)
'''


def _type_to_str(type_hint: Any) -> str:
    """Convert type annotation to string for stub."""
    if type_hint is str:
        return "str"
    elif type_hint is int:
        return "int"
    elif type_hint is float:
        return "float"
    elif type_hint is bool:
        return "bool"
    elif type_hint is list:
        return "list"
    elif type_hint is dict:
        return "dict"
    elif hasattr(type_hint, "__origin__"):
        return str(type_hint)
    else:
        return ""


# ---------------------------------------------------------------------------
# REPLMixin: The main agent mixin
# ---------------------------------------------------------------------------

class REPLMixin:
    """
    Mixin that replaces tool-calling with direct REPL code execution.

    The LLM writes Python code which is executed in a persistent REPL.
    Tools defined with @BaseAgent.tool become callable functions in
    the REPL that bridge back to the main process for execution.

    The model's response is passed directly to the REPL - no markdown
    code blocks, no extraction. The response IS the code.

    Use emit(value, release=False) to output values. Only emit(..., release=True)
    releases control to the user.
    """

    def build_output_for_llm(self, output_chunks):
        """Build the output string sent to the LLM from typed output chunks.

        Override to filter specific chunk types (e.g., exclude "emit" output).
        Each chunk is a (msg_type, text) tuple where msg_type is one of:
        "echo", "output", "print", "emit", "read", "read_attach", "file_written", "progress", "error".
        """
        return "".join(chunk for _, chunk in output_chunks)

    interactive: bool = False  # Legacy flag, kept for compatibility

    def usermsg(self, content, **kwargs):
        """
        Add a user message, appending to REPL output if continuing a session.

        When the last message was REPL output (from a previous turn), new user
        messages are appended to maintain the illusion of a continuous REPL
        session. The message appears as if emitted via emit().
        """
        # Check if we should append to last REPL output
        if getattr(self, '_last_was_repl_output', False) and self.conversation.messages:
            last_msg = self.conversation.messages[-1]
            if last_msg.get("role") == "user":
                # Append as output of the previous emit() call
                # Add trailing \n since this simulates print() output from emit()
                prev = last_msg["content"]
                sep = "" if prev.endswith("\n") else "\n"
                last_msg["content"] = prev + sep + content + "\n"
                last_msg["_user_content"] = content

                # Also update _stdout with the appended content
                if '_stdout' in last_msg:
                    prev_stdout = last_msg['_stdout']
                    sep_stdout = "" if prev_stdout.endswith("\n") else "\n"
                    last_msg['_stdout'] = prev_stdout + sep_stdout + content + "\n"

                # If new content has images, append them too
                if 'images' in kwargs:
                    last_msg['images'] = last_msg.get('images', []) + kwargs['images']

                self._last_was_repl_output = False
                return

        # Fall back to normal message append
        self._last_was_repl_output = False
        return super().usermsg(content, **kwargs)

    def _ensure_setup(self) -> None:
        """Initialize the ToolREPL."""
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        if not hasattr(self, '_tool_repl'):
            self._tool_repl = ToolREPL(echo=True)

    def _get_tool_repl(self) -> ToolREPL:
        """Get or create the ToolREPL instance, with tools and startup code injected."""
        self._ensure_setup()
        # Inject builtins first so tools can use _send_output, _original_print, etc.
        self._tool_repl.inject_builtins()
        tools = {
            name: (self._toolimpl.get(name), spec)
            for name, spec in self.toolspecs.items()
        }
        self._tool_repl.inject_tools(tools)
        
        # Inject startup code if defined
        if not getattr(self, '_repl_startup_injected', False):
            startup = getattr(self, 'repl_startup', None)
            if startup:
                if callable(startup):
                    startup = startup()
                self._tool_repl.inject_startup(startup)
            self._repl_startup_injected = True
        
        return self._tool_repl

    def _cleanup(self) -> None:
        """Clean up REPL resources."""
        if hasattr(self, '_tool_repl'):
            self._tool_repl.close()
        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    def _build_system_prompt(self) -> str:
        """Build REPL-style system prompt."""
        # Ensure mixins are set up (e.g., MCPMixin connects to servers)
        self._ensure_setup()

        # Get base prompt from parent (includes mixin additions)
        if hasattr(super(), '_build_system_prompt'):
            base_prompt = super()._build_system_prompt()
        else:
            base_prompt = getattr(self, 'system', '') or ''

        # List available tools (static + dynamic)
        tool_list = []
        for name, spec in self.toolspecs.items():
            impl = self._toolimpl.get(name)
            param_strs = []
            if impl:
                # Static tool - get params from signature
                sig = inspect.signature(impl)
                for pname, param in sig.parameters.items():
                    if pname == 'self':
                        continue
                    type_str = _type_to_str(param.annotation) if param.annotation != inspect.Parameter.empty else ''
                    if type_str:
                        param_strs.append(f"{pname}: {type_str}")
                    else:
                        param_strs.append(pname)
                doc = (impl.__doc__ or '').strip()
                first_line = doc.split('\n')[0] if doc else ''
            else:
                # Dynamic tool - get params from spec
                if hasattr(spec, 'model_fields'):
                    for field_name, field_info in spec.model_fields.items():
                        # Sanitize for valid Python identifier
                        sanitized = field_name.lstrip('-')
                        if sanitized and sanitized[0].isdigit():
                            sanitized = 'p_' + sanitized
                        type_str = _type_to_str(field_info.annotation) if field_info.annotation else ''
                        if type_str:
                            param_strs.append(f"{sanitized}: {type_str}")
                        else:
                            param_strs.append(sanitized)
                    doc = (getattr(spec, '__doc__', '') or '').strip()
                    first_line = doc.split('\n')[0] if doc else ''
                else:
                    first_line = ''

            # Format: name(param: type, ...) - description
            params_str = ", ".join(param_strs)
            tool_entry = f"{name}({params_str})"
            if first_line:
                # Strip leading non-alphanumeric (bullets, dashes, etc.)
                desc = first_line.lstrip()
                while desc and not desc[0].isalnum():
                    desc = desc[1:]
                if desc:
                    tool_entry += f" - {desc}"
            tool_list.append(tool_entry)

        tools_str = "\n".join(tool_list) if tool_list else "(no tools defined)"

        return f"""You are in a Python REPL. Respond with unescaped Python.

{base_prompt}

You have full access to Python. These additional functions are available:
{tools_str}
emit(value, release=False) - Emit output. release=True yields control.

Call help(function_name) for parameter descriptions.
"""

    def run_loop(self, max_turns: int = 50, max_syntax_retries: int = 3) -> Any:
        """
        Main agent loop using REPL-first paradigm.

        The LLM writes Python code, we execute it, feed output back.
        Loop continues until emit(release=True) is called or max_turns reached.

        Pure syntax errors (where no statements executed) are retried without
        polluting the conversation history - only successful attempts are committed.
        """
        self._ensure_setup()

        self.complete = False

        for turn in range(max_turns):
            # Get REPL each turn to handle session restart (re-injects tools if needed)
            repl = self._get_tool_repl()
            messages = self.conversation._messages()

            # Fire execute hook once at start of turn (before any attempts)
            if hasattr(self, 'on_repl_execute'):
                self.on_repl_execute(None)

            # Retry loop for pure syntax errors (nothing executed)
            output = ""
            try:
                for syntax_retry in range(max_syntax_retries):
                    try:
                        if hasattr(self.llm_client, 'text_call'):
                            resp = self.llm_client.text_call(messages)
                        else:
                            resp = self.llm_client.call(messages, tools=None)
                    except KeyboardInterrupt:
                        # User interrupted LLM call - subprocess may also have received SIGINT
                        # Wait briefly for subprocess to handle signal, then drain
                        # Use poll_output if available (SandboxedToolREPL), else _output_queue (ToolREPL)
                        if hasattr(repl, 'poll_output'):
                            while True:
                                msg = repl.poll_output(timeout=0.1)
                                if msg is None or msg[0] == "done":
                                    break
                        elif hasattr(repl, '_output_queue'):
                            while True:
                                try:
                                    msg_type, _ = repl._output_queue.get(timeout=0.1)
                                    if msg_type == "done":
                                        break
                                except Empty:
                                    break
                        raise _InterruptedError("")
                    except BadRequestError:
                        raise

                    content = (resp.get('content') or '').strip()
                    if not content:
                        break

                    # Strip markdown fences if present
                    if content.startswith("```"):
                        first_newline = content.find('\n')
                        if first_newline != -1:
                            content = content[first_newline + 1:]
                        if content.endswith("```"):
                            content = content[:-3].rstrip('\n')

                    output, pure_syntax_error, output_chunks, corrected_code = self._execute_with_tool_handling(repl, content)

                    # Apply silent corrections to conversation (both sides see corrected code)
                    if corrected_code != content:
                        resp['content'] = corrected_code
                        content = corrected_code

                    if not pure_syntax_error:
                        break

                    # Pure syntax error - retry with temporary error context
                    logger.debug(f"SyntaxError, retry #{syntax_retry + 1}")
                    hint = (
                        "Your response was not valid Python and was rejected. "
                        "Try again using only Python code. "
                        "Use an appropriate function to communicate text."
                    )
                    messages = self.conversation._messages() + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": f"{output}\n{hint}"}
                    ]
                else:
                    # All retries exhausted
                    print(f"\n*** CRASH: Model failed to produce valid Python after {max_syntax_retries} retries ***", file=sys.stderr)
                    raise SyntaxError(
                        f"Your response must be valid Python code without preamble or markdown.\n\n{output}"
                    )
            except _InterruptedError:
                # Interrupted output is discarded - don't pollute conversation
                raise

            # Fire output hook after successful execution
            if hasattr(self, 'on_repl_output'):
                self.on_repl_output(output_chunks)

            # Commit successful response to conversation
            self.conversation.messages.append(resp)

            if not content:
                continue

            # Feed output back to LLM as the REPL response
            # build_output_for_llm lets subclasses filter chunk types (e.g., exclude emit)
            output_for_llm = self.build_output_for_llm(output_chunks)
            if hasattr(self, 'process_repl_output'):
                output_for_llm = self.process_repl_output(output_for_llm)

            # Store full output as _stdout when it differs from filtered content
            kwargs = {}
            if output_for_llm != output:
                kwargs['_stdout'] = output

            if output_for_llm.strip():
                self._last_was_repl_output = False  # Clear before usermsg check
                self.usermsg(output_for_llm, **kwargs)
            else:
                self._last_was_repl_output = False
                self.usermsg("# [no output]", **kwargs)

            if self.complete:
                # Mark that last message is REPL output - next user message appends
                self._last_was_repl_output = True
                return self._final_result

        raise Exception(f"Agent did not complete within {max_turns} turns")

    def _format_syntax_error(self, e: SyntaxError) -> str:
        """Format a SyntaxError like Python's REPL does."""
        lines = []
        lines.append(f'  File "<repl>", line {e.lineno}')
        if e.text:
            lines.append(f'    {e.text.rstrip()}')
            if e.offset:
                # Python 3.10+ has end_offset for better caret positioning
                end_offset = getattr(e, 'end_offset', None)
                if end_offset and end_offset > e.offset:
                    lines.append('    ' + ' ' * (e.offset - 1) + '^' * (end_offset - e.offset))
                else:
                    lines.append('    ' + ' ' * (e.offset - 1) + '^')
        lines.append(f'SyntaxError: {e.msg}')
        return '\n'.join(lines) + '\n'

    def _transform_toplevel_print(self, code: str) -> str:
        """Transform top-level print() calls to _print() for output tagging.

        Only transforms print calls at module level, not inside function/class definitions.
        This allows tools and user-defined functions to use normal print().
        """
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code  # Let the REPL handle syntax errors

        class PrintTransformer(ast.NodeTransformer):
            def __init__(self):
                self.in_function = False

            def visit_FunctionDef(self, node):
                # Don't transform inside function bodies
                old = self.in_function
                self.in_function = True
                self.generic_visit(node)
                self.in_function = old
                return node

            def visit_AsyncFunctionDef(self, node):
                return self.visit_FunctionDef(node)

            def visit_ClassDef(self, node):
                # Don't transform inside class bodies
                old = self.in_function
                self.in_function = True
                self.generic_visit(node)
                self.in_function = old
                return node

            def visit_Call(self, node):
                self.generic_visit(node)
                # Transform print() to _print() at top level only
                if not self.in_function:
                    if isinstance(node.func, ast.Name) and node.func.id == 'print':
                        node.func.id = '_print'
                return node

        transformer = PrintTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)

    def _execute_with_tool_handling(self, repl: ToolREPL, code: str) -> tuple[str, bool, list, str]:
        """Execute code statement-by-statement, handling tool calls as they occur.

        Returns:
            Tuple of (output, is_pure_syntax_error, output_chunks, corrected_code) where:
            - is_pure_syntax_error is True when no statements executed successfully
            - output_chunks is list of (msg_type, chunk) tuples
            - corrected_code is the code after any preprocessing corrections
        """
        # Allow subclasses to preprocess code before splitting
        # (e.g., fix triple-quote conflicts that would break statement parsing)
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

            # Pre-validate syntax before echoing (use transformed for validation)
            try:
                compile(exec_stmt, '<repl>', 'exec')
            except SyntaxError as e:
                # Echo the original statement, show error, stop processing
                stream(_format_echo(original_stmt), "echo")
                stream(self._format_syntax_error(e), "error")
                if hasattr(self, 'on_statement_output'):
                    self.on_statement_output(statement_chunks)
                break

            # Valid syntax - echo original, execute transformed
            any_executed = True
            stream(_format_echo(original_stmt), "echo")
            repl._running = True
            repl._cmd_seq += 1
            current_seq = repl._cmd_seq
            repl._cmd_queue.put((current_seq, exec_stmt))

            # Poll loop for this statement
            statement_had_error = False
            try:
                while True:
                    # Check for tool requests (non-blocking)
                    tool_req = repl.poll_tool_request(timeout=0)
                    if tool_req:
                        self._handle_tool_request(repl, tool_req)
                        if self.complete:
                            # emit(release=True) was called - drain output and return
                            while True:
                                try:
                                    msg_type, msg_data = repl._output_queue.get(timeout=0.1)
                                    if msg_type in ("output", "print", "emit", "read", "progress"):
                                        stream(msg_data, msg_type)
                                    elif msg_type == "done":
                                        seq_id, _ = msg_data
                                        if seq_id == current_seq:
                                            break
                                        # Stale done from previous command, ignore
                                except Empty:
                                    break
                            repl._running = False
                            output = "".join(chunk for _, chunk in output_chunks)
                            return output, False, output_chunks, code

                    # Check for output
                    try:
                        msg_type, msg_data = repl._output_queue.get(timeout=0.05)
                        if msg_type in ("output", "print", "emit", "read", "progress"):
                            stream(msg_data, msg_type)
                        elif msg_type == "done":
                            seq_id, had_error = msg_data
                            if seq_id == current_seq:
                                repl._running = False
                                statement_had_error = had_error
                                break  # Statement complete, move to next
                            # Stale done from previous command, ignore and keep polling
                    except Empty:
                        pass
            except KeyboardInterrupt:
                # User pressed Ctrl+C - interrupt the subprocess
                # interrupt() drains and returns all output, so stream it immediately
                interrupted_output = repl.interrupt()
                if interrupted_output:
                    stream(interrupted_output, "output")
                repl._running = False
                raise _InterruptedError()

            # Statement complete - call hook and reset per-statement tracking
            if hasattr(self, 'on_statement_output'):
                self.on_statement_output(statement_chunks)
            statement_chunks.clear()  # Must use clear() to preserve closure reference

            # Stop processing if this statement had a runtime error
            if statement_had_error:
                break

        output = "".join(chunk for _, chunk in output_chunks)
        is_pure_syntax_error = not any_executed and bool(output_chunks)
        return output, is_pure_syntax_error, output_chunks, code

    def _handle_tool_request(self, repl: ToolREPL, req: dict) -> None:
        """Handle a tool request from the REPL."""
        tool_name = req.get('tool')
        request_id = req.get('request_id')
        args = req.get('args', {})

        # Deserialize special types (like bytes)
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
            # Send ACK only for builtin protocol calls (emit, etc.) that use
            # _wait_for_ack().  Relay stubs don't send request_id and only do
            # a single _tool_response_queue.get() for the reply — a stale ACK
            # left in the queue would poison the next relay call and deadlock.
            if request_id is not None:
                repl.send_ack(request_id)

# ---------------------------------------------------------------------------
# REPLAgent: First-class agent type
# ---------------------------------------------------------------------------

class REPLAgent(REPLMixin, BaseAgent):
    """
    Agent that operates in a Python REPL instead of using tool calls.

    Example:
        class MyAgent(REPLAgent):
            system = "You are helpful."

            @REPLAgent.tool
            def search(self, query: str = "Search query"):
                '''Search for something.'''
                return do_search(query)
    """
    pass
