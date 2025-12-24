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

The REPL always has `submit(result)` available for returning the final answer.

How It Works
------------

1. LLM writes Python code as its response
2. Code executes statement-by-statement in isolated subprocess
3. Tool calls bridge back to main process via queues
4. Output streams back to LLM as REPL feedback
5. Loop continues until submit() is called
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
from typing import Any, Callable, Optional

logger = logging.getLogger('agentlib')

from agentlib.agent import BaseAgent, _CompleteException


class _InterruptedError(KeyboardInterrupt):
    """Raised when user interrupts execution with Ctrl+C."""
    def __init__(self, output: str = ""):
        self.output = output
        super().__init__("Interrupted by user")


from agentlib.subrepl import (
    SubREPL,
    _format_echo,
    _split_into_statements,
)


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
        '_tool_request_queue': tool_request_queue,
        '_tool_response_queue': tool_response_queue,
    }

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

            output_queue.put(("done", had_error))

        except KeyboardInterrupt:
            output_queue.put(("output", "\nKeyboardInterrupt\n"))
            output_queue.put(("done", None))


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

    def _ensure_session(self) -> None:
        """Override to use tool-aware worker and add tool queues."""
        if self._worker is None or not self._worker.is_alive():
            self._cmd_queue = Queue()
            self._output_queue = Queue()
            self._tool_request_queue = Queue()
            self._tool_response_queue = Queue()

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

    def inject_tools(self, tools: dict[str, tuple[Callable, Any]]) -> None:
        """
        Inject tool stub functions into the REPL.

        Args:
            tools: Dict of tool_name -> (implementation, pydantic_spec)
        """
        self._ensure_session()

        if self._tools_injected:
            return

        for name, (impl, spec) in tools.items():
            stub_code = _generate_tool_stub(name, impl, spec)
            self._inject_code(stub_code)

        self._tools_injected = True

    def inject_builtins(self) -> None:
        """Inject built-in functions like submit()."""
        self._ensure_session()

        # submit() - signals task completion
        self._inject_code('''
def submit(result):
    """Submit your final result and end the task."""
    import json as _json
    _tool_request_queue.put(_json.dumps({
        "tool": "__submit__",
        "args": {"result": result}
    }))
    _tool_response_queue.get()  # Wait for ack
''')

    def _inject_code(self, code_str: str) -> None:
        """Execute code without echo, waiting for completion."""
        self._ensure_session()
        self._running = True
        self._cmd_queue.put(code_str)

        while True:
            try:
                msg_type, msg_data = self._output_queue.get(timeout=5.0)
                if msg_type == "done":
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

    def send_tool_response(self, result: Any = None, error: str = None) -> None:
        """Send tool execution result back to REPL."""
        if self._tool_response_queue is None:
            raise RuntimeError("No active session")

        if error:
            self._tool_response_queue.put(json.dumps({"error": error}))
        else:
            try:
                self._tool_response_queue.put(json.dumps({"result": result}))
            except TypeError:
                # Fall back to repr for non-JSON-serializable
                self._tool_response_queue.put(json.dumps({"result": repr(result)}))

    def send_ack(self) -> None:
        """Send simple acknowledgment (for submit())."""
        if self._tool_response_queue is None:
            raise RuntimeError("No active session")
        self._tool_response_queue.put("ack")


# ---------------------------------------------------------------------------
# Stub generation
# ---------------------------------------------------------------------------

def _generate_tool_stub(name: str, impl: Optional[Callable], spec: Any) -> str:
    """
    Generate Python stub function that bridges to main process.

    The stub:
    1. Serializes the call to the tool request queue
    2. Blocks waiting for response
    3. Returns result or raises exception

    Args:
        name: Tool name
        impl: Tool implementation (may be None for dynamic tools)
        spec: Pydantic model spec for the tool
    """
    param_names: list[str] = []
    signature_parts: list[str] = []
    doc_parts: list[str] = []

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
                    signature_parts.append(param_name)
                else:
                    # Actual default value
                    doc_parts.append(f"        {param_name}: (default: {repr(param.default)})")
                    signature_parts.append(f"{param_name}={repr(param.default)}")
            else:
                doc_parts.append(f"        {param_name}")
                signature_parts.append(param_name)

        docstring = (impl.__doc__ or f"Call the {name} tool.").strip()

    else:
        # Dynamic tool - extract from Pydantic spec
        if hasattr(spec, 'model_fields'):
            # Separate required and optional params (required must come first)
            required_params = []
            optional_params = []

            for field_name, field_info in spec.model_fields.items():
                # Sanitize param name for valid Python identifier
                sanitized = field_name.lstrip('-')
                if sanitized and sanitized[0].isdigit():
                    sanitized = 'p_' + sanitized

                desc = field_info.description or ''
                doc_parts.append(f"        {sanitized}: {desc}")

                # Check if field has a real default (not PydanticUndefined)
                has_default = (
                    field_info.default is not None and
                    type(field_info.default).__name__ != 'PydanticUndefinedType'
                )

                if has_default:
                    optional_params.append((field_name, sanitized, repr(field_info.default)))
                elif field_info.is_required():
                    required_params.append((field_name, sanitized))
                else:
                    # Optional without default - use None
                    optional_params.append((field_name, sanitized, 'None'))

            # Build signature: required params first, then optional
            # param_names stores (original, sanitized) tuples
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

    # Build args dict construction
    # For dynamic tools, param_names contains (original, sanitized) tuples
    # For static tools, param_names contains just the name strings
    # Filter out None values per MCP spec (optional params should be omitted, not null)
    if param_names and isinstance(param_names[0], tuple):
        args_items = ", ".join(f'"{orig}": {sanitized}' for orig, sanitized in param_names)
    else:
        args_items = ", ".join(f'"{n}": {n}' for n in param_names)
    args_dict = f"{{{args_items}}}"

    return f'''
def {name}({signature_str}):
    """{docstring}"""
    import json as _json
    _args = {{k: v for k, v in {args_dict}.items() if v is not None}}
    _tool_request_queue.put(_json.dumps({{"tool": "{name}", "args": _args}}))
    _response = _json.loads(_tool_response_queue.get())
    if "error" in _response:
        raise Exception(_response["error"])
    return _response["result"]
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

    Set `interactive = True` for CLI/chat agents to enable `respond(text)`
    as a more natural alternative to `submit(result)`.
    """

    interactive: bool = False  # Set True to enable respond() function

    def _ensure_setup(self) -> None:
        """Initialize the ToolREPL."""
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        if not hasattr(self, '_tool_repl'):
            self._tool_repl = ToolREPL(echo=True)

    def _get_tool_repl(self) -> ToolREPL:
        """Get or create the ToolREPL instance."""
        self._ensure_setup()
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
submit(result) - Submit your final answer.

Call help(function_name) for parameter descriptions.
"""

    def run_loop(self, max_turns: int = 50, max_syntax_retries: int = 3) -> Any:
        """
        Main agent loop using REPL-first paradigm.

        The LLM writes Python code, we execute it, feed output back.
        Loop continues until submit() is called or max_turns reached.

        Pure syntax errors (where no statements executed) are retried without
        polluting the conversation history - only successful attempts are committed.
        """
        self._ensure_setup()
        repl = self._get_tool_repl()

        # Inject tool stubs (includes dynamic tools from mixins)
        tools = {
            name: (self._toolimpl.get(name), spec)
            for name, spec in self.toolspecs.items()
        }
        repl.inject_tools(tools)
        repl.inject_builtins()

        # In interactive mode, add respond() as a friendlier alias for submit()
        if getattr(self, 'interactive', False):
            repl._inject_code('''
def respond(text):
    """Respond to the user."""
    submit(text)
''')

        self.complete = False

        for turn in range(max_turns):
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
                        # User interrupted LLM call - close delimiter and return to prompt
                        raise _InterruptedError("")

                    content = resp.get('content', '').strip()
                    if not content:
                        break

                    output, pure_syntax_error = self._execute_with_tool_handling(repl, content)

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
                    # All retries exhausted - remind LLM of expected format
                    raise SyntaxError(
                        f"Your response must be valid Python code without preamble or markdown.\n\n{output}"
                    )
            except _InterruptedError as e:
                # Fire output hook to close delimiter
                if hasattr(self, 'on_repl_output'):
                    self.on_repl_output(e.output)
                # Add interrupt output to conversation so agent sees it next turn
                if e.output:
                    self.usermsg(e.output)
                raise

            # Fire output hook after successful execution
            if hasattr(self, 'on_repl_output'):
                self.on_repl_output(output)

            # Commit successful response to conversation
            self.conversation.messages.append(resp)

            if not content:
                continue

            if self.complete:
                return self._final_result

            # Feed output back to LLM as the REPL response
            # Allow subclasses to process/truncate large outputs
            output_for_llm = output
            if hasattr(self, 'process_repl_output'):
                output_for_llm = self.process_repl_output(output)

            if output_for_llm.strip():
                self.usermsg(output_for_llm)
            else:
                self.usermsg("(no output)")

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

    def _execute_with_tool_handling(self, repl: ToolREPL, code: str) -> tuple[str, bool]:
        """Execute code statement-by-statement, handling tool calls as they occur.

        Returns:
            Tuple of (output, is_pure_syntax_error) where is_pure_syntax_error is True
            when no statements executed successfully (first statement had syntax error).
        """
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
            # Pre-validate syntax before echoing
            try:
                compile(stmt, '<repl>', 'exec')
            except SyntaxError as e:
                # Echo the bad statement, show error, stop processing
                stream(_format_echo(stmt))
                stream(self._format_syntax_error(e))
                break

            # Valid syntax - echo and execute
            any_executed = True
            stream(_format_echo(stmt))
            repl._running = True
            repl._cmd_queue.put(stmt)

            # Poll loop for this statement
            statement_had_error = False
            try:
                while True:
                    # Check for tool requests (non-blocking)
                    tool_req = repl.poll_tool_request(timeout=0)
                    if tool_req:
                        self._handle_tool_request(repl, tool_req)
                        if self.complete:
                            # submit() was called - drain output and return
                            while True:
                                try:
                                    msg_type, msg_data = repl._output_queue.get(timeout=0.1)
                                    if msg_type == "output":
                                        stream(msg_data)
                                    elif msg_type == "done":
                                        break
                                except Empty:
                                    break
                            repl._running = False
                            return "".join(output_chunks), False

                    # Check for output
                    try:
                        msg_type, msg_data = repl._output_queue.get(timeout=0.05)
                        if msg_type == "output":
                            stream(msg_data)
                        elif msg_type == "done":
                            repl._running = False
                            statement_had_error = msg_data  # msg_data is had_error boolean
                            break  # Statement complete, move to next
                    except Empty:
                        pass
            except KeyboardInterrupt:
                # User pressed Ctrl+C - interrupt the subprocess
                repl.interrupt()
                # Drain remaining output (will contain KeyboardInterrupt message)
                while True:
                    try:
                        msg_type, msg_data = repl._output_queue.get(timeout=0.5)
                        if msg_type == "output":
                            stream(msg_data)
                        elif msg_type == "done":
                            break
                    except Empty:
                        break
                repl._running = False
                raise _InterruptedError("".join(output_chunks))

            # Stop processing if this statement had a runtime error
            if statement_had_error:
                break

        output = "".join(output_chunks)
        is_pure_syntax_error = not any_executed and bool(output_chunks)
        return output, is_pure_syntax_error

    def _handle_tool_request(self, repl: ToolREPL, req: dict) -> None:
        """Handle a tool request from the REPL."""
        tool_name = req.get('tool')
        args = req.get('args', {})

        if tool_name == '__submit__':
            self._final_result = args.get('result')
            self.complete = True
            repl.send_ack()

        elif tool_name:
            # Regular tool call
            try:
                result = self.toolcall(tool_name, args)
                repl.send_tool_response(result=result)
            except _CompleteException:
                # Tool called self.respond()
                raise
            except Exception as e:
                repl.send_tool_response(error=str(e))

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
