"""
SubREPLMixin - Mixin that adds Python REPL execution to any agent.

Example:
    from agentlib import BaseAgent, SubREPLMixin

    class MyAgent(SubREPLMixin, BaseAgent):
        model = 'google/gemini-2.5-flash'
        system = "You are a helpful assistant."

        @BaseAgent.tool
        def done(self, response: str = "Your response"):
            self.respond(response)

    with MyAgent() as agent:
        result = agent.run("Calculate the sum of squares from 1 to 100")

Notes:
    - Python state (variables, imports, functions) persists across calls.
    - For long-running code, use python_read() to continue reading output.
    - Use python_interrupt() to stop running code.
    - Call close() when done, or use as context manager.
"""

import threading
from pydantic import create_model, Field
from typing import Optional, Any

from .subrepl import SubREPL, STILL_RUNNING
from .tool_mixin import ToolMixin


class SubREPLMixin(ToolMixin):
    """Mixin that adds Python REPL execution. Use with BaseAgent."""

    # Configuration
    repl_echo: bool = False  # Echo statements in output (default False for agent use)
    repl_timeout: float = 30.0  # Default timeout for REPL operations

    # === HOOK IMPLEMENTATIONS ===

    def _ensure_setup(self):
        # Chain to next in MRO (might be another mixin or BaseAgent)
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        # Fast path - already initialized
        if getattr(self, '_repl_initialized', False):
            return

        # Thread-safe initialization
        if not hasattr(self, '_repl_lock'):
            self._repl_lock = threading.Lock()

        with self._repl_lock:
            if getattr(self, '_repl_initialized', False):
                return

            self._repl: Optional[SubREPL] = None
            self._repl_initialized = True

    def _get_repl(self) -> SubREPL:
        """Lazily create REPL on first use."""
        if self._repl is None:
            self._repl = SubREPL(echo=getattr(self, 'repl_echo', False))
        return self._repl

    def _build_system_prompt(self):
        # Get base system prompt from chain
        if hasattr(super(), '_build_system_prompt'):
            system = super()._build_system_prompt()
        else:
            system = getattr(self, 'system', '')

        # Add REPL instructions
        repl_instructions = """

PYTHON EXECUTION:
You have access to a persistent Python REPL via these tools:
- python_execute: Run Python code. State (variables, imports, functions) persists.
- python_read: Continue reading output if code is still running.
- python_interrupt: Stop running code (sends SIGINT/Ctrl+C).

When output ends with "[still running]", the code hasn't finished.
Use python_read() to get more output or python_interrupt() to stop it.

You can define functions and classes that persist across calls.
Import statements persist - no need to re-import modules."""

        return system + repl_instructions

    def _get_dynamic_toolspecs(self):
        # Get specs from chain first
        if hasattr(super(), '_get_dynamic_toolspecs'):
            specs = super()._get_dynamic_toolspecs()
        else:
            specs = {}

        # Create Pydantic models for tool specs
        specs['python_execute'] = create_model(
            'PythonExecute',
            code=(str, Field(..., description="Python code to execute")),
            timeout=(float, Field(30.0, description="Max seconds to wait for output")),
            __doc__="Execute Python code in a persistent REPL. Variables, imports, and function definitions persist across calls."
        )

        specs['python_read'] = create_model(
            'PythonRead',
            timeout=(float, Field(30.0, description="Max seconds to wait for more output")),
            __doc__="Continue reading output from long-running Python code. Use when previous output ended with '[still running]'."
        )

        specs['python_interrupt'] = create_model(
            'PythonInterrupt',
            __doc__="Interrupt currently running Python code with SIGINT (like Ctrl+C)."
        )

        return specs

    def _cleanup(self):
        # Clean up REPL
        if hasattr(self, '_repl') and self._repl is not None:
            try:
                self._repl.close()
            except Exception:
                pass
            self._repl = None

        if hasattr(self, '_repl_initialized'):
            self._repl_initialized = False

        # Chain to next
        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    # === PUBLIC METHODS ===

    def python_execute(self, code: str, timeout: Optional[float] = None) -> str:
        """
        Execute Python code directly (without going through the agent).

        Args:
            code: Python code to execute
            timeout: Max seconds to wait (default: repl_timeout)

        Returns:
            Code output. Ends with "[still running]\\n" if not complete.
        """
        self._ensure_setup()
        t = timeout if timeout is not None else getattr(self, 'repl_timeout', 30.0)
        return self._get_repl().execute(code, timeout=t)

    def python_read(self, timeout: Optional[float] = None) -> str:
        """
        Continue reading output from long-running code.

        Args:
            timeout: Max seconds to wait (default: repl_timeout)

        Returns:
            More output. Ends with "[still running]\\n" if still running.
        """
        self._ensure_setup()
        t = timeout if timeout is not None else getattr(self, 'repl_timeout', 30.0)
        result = self._get_repl().read(timeout=t)
        return result if result else "[No output available]"

    def python_interrupt(self) -> str:
        """
        Interrupt currently running Python code.

        Returns:
            Any remaining output after interrupt.
        """
        self._ensure_setup()
        result = self._get_repl().interrupt()
        return result if result else "[Code interrupted]"


class SubREPLResponseMixin(SubREPLMixin):
    """
    Extended REPL mixin that adds python_execute_response tool.

    Use this when you want agents to be able to execute code and return
    the output directly as their response.

    Example:
        class CalcAgent(SubREPLResponseMixin, BaseAgent):
            model = 'google/gemini-2.5-flash'
            system = "You are a calculator. Use python_execute_response to compute and return results."
    """

    def _build_system_prompt(self):
        system = super()._build_system_prompt()
        # Add info about the response tool
        system += "\n- python_execute_response: Execute Python code and return ALL printed output directly to the user as your final response. The entire stdout (everything you print) becomes the user's response - format it cleanly for them using PLAINTEXT.\n  **Tip: MCP tool results can be parsed and formatted in the same script before returning.**"
        return system

    def _get_dynamic_toolspecs(self):
        specs = super()._get_dynamic_toolspecs()

        specs['python_execute_response'] = create_model(
            'PythonExecuteResponse',
            code=(str, Field(..., description="Python code to execute")),
            preamble=(str, Field("", description="Optional additional text to display before the code output.")),
            postamble=(str, Field("", description="Optional additional text to display after the code output.")),
            __doc__="Execute Python code and return the output as your final response to the user."
        )

        return specs

    def python_execute_response(self, code: str, preamble: str = "", postamble: str = ""):
        """
        Execute Python code and return output as the agent's response.

        Args:
            code: Python code to execute
            preamble: Optional text to display before the code output
            postamble: Optional text to display after the code output

        Returns:
            None on success (response sent via self.respond()),
            or error message for agent to handle.
        """
        self._ensure_setup()
        timeout = getattr(self, 'repl_timeout', 30.0)
        output = self._get_repl().execute(code, timeout=timeout)

        if output.endswith(STILL_RUNNING):
            # Timed out - return to agent so it can handle
            clean = output[:-len(STILL_RUNNING)].strip()
            return f"{clean}\n\n[still running - code timed out]"

        # Check for exceptions - return to agent so it can fix
        if 'Traceback (most recent call last):' in output:
            return output

        # Success - format and respond to user
        parts = []
        if preamble:
            parts.append(preamble)
        parts.append(f"```\n{output.strip()}\n```")
        if postamble:
            parts.append(postamble)
        self.respond("\n\n".join(parts))

        return None  # respond() handles the response


class REPLMCPMixin(SubREPLMixin):
    """
    Mixin that sets up MCP clients in the REPL for lightweight MCP access.

    Instead of exposing each MCP tool as an agent tool (token-heavy), this mixin
    pre-instantiates MCP clients in the REPL. The agent uses python_execute to
    interact with them directly.

    Example:
        from agentlib import BaseAgent, REPLMCPMixin

        class MyAgent(REPLMCPMixin, BaseAgent):
            model = 'google/gemini-2.5-flash'
            system = "You are a helpful assistant."
            repl_mcp_servers = [
                ('fs', '/usr/bin/mcp-server-filesystem /tmp'),
                ('api', 'http://localhost:3000/sse'),
            ]

            @BaseAgent.tool
            def done(self, response: str = "Your response"):
                self.respond(response)

        with MyAgent() as agent:
            # Agent can use python_execute to call:
            #   fs.list_tools()
            #   fs.call_tool('read_file', {'path': '/tmp/test.txt'})
            result = agent.run("List files in /tmp using the fs MCP server")

    For python_execute_response support, combine with SubREPLResponseMixin:
        class MyAgent(REPLMCPMixin, SubREPLResponseMixin, BaseAgent):
            repl_mcp_servers = [...]

    Notes:
        - MCP clients are available as variables in the REPL (e.g., `fs`, `api`)
        - Use client.list_tools() to see available tools
        - Use client.call_tool('name', {'arg': 'value'}) to invoke tools
        - Server instructions and tool definitions are added to the system prompt
        - Set repl_mcp_enumerate_tools=False to skip tool enumeration globally
        - Per-server: ('name', 'server', {'enumerate_tools': False})
    """

    repl_mcp_servers: list = []  # List of (name, server) or (name, server, options) tuples
    repl_mcp_enumerate_tools: bool = True  # Enumerate tools at setup (can be overridden per-server)

    def _ensure_setup(self):
        # Chain to parent (sets up REPL)
        super()._ensure_setup()

        # Fast path - already initialized MCP
        if getattr(self, '_repl_mcp_initialized', False):
            return

        # Thread-safe initialization
        with self._repl_lock:
            if getattr(self, '_repl_mcp_initialized', False):
                return

            self._repl_mcp_servers_info: dict[str, dict] = {}
            self._repl_mcp_instructions: dict[str, str] = {}
            self._repl_mcp_tools: dict[str, list] = {}

            try:
                self._setup_mcp_in_repl()
                self._repl_mcp_initialized = True
            except Exception:
                # Cleanup on failure
                self._repl_mcp_servers_info = {}
                self._repl_mcp_instructions = {}
                self._repl_mcp_tools = {}
                raise

    def _setup_mcp_in_repl(self):
        """Initialize MCP clients in the REPL."""
        servers = getattr(self, 'repl_mcp_servers', [])
        if not servers:
            return

        repl = self._get_repl()

        # Import MCP client factory functions
        repl.execute(
            "from agentlib.mcp import create_stdio_client, create_sse_client",
            timeout=10.0
        )

        for server_def in servers:
            if len(server_def) == 2:
                name, server = server_def
                options = {}
            else:
                name, server, options = server_def

            self._repl_mcp_servers_info[name] = {'server': server, 'options': options}

            # Build connection code
            if server.startswith('http://') or server.startswith('https://'):
                # SSE transport
                headers = options.get('headers', {})
                timeout = options.get('timeout', 300.0)
                code = f"{name} = create_sse_client({server!r}, headers={headers!r}, timeout={timeout!r})"
            else:
                # Stdio transport
                forward_stderr = options.get('forward_stderr', False)
                timeout = options.get('timeout', 300.0)
                code = f"{name} = create_stdio_client({server.split()!r}, forward_stderr={forward_stderr!r}, timeout={timeout!r})"

            repl.execute(code, timeout=30.0)

            # Get instructions if available
            output = repl.execute(f"print(repr({name}.instructions))", timeout=10.0)
            instructions = self._parse_repl_repr(output)
            if instructions and instructions != 'None':
                self._repl_mcp_instructions[name] = instructions

            # Get tools list (if enabled)
            enumerate_tools = options.get('enumerate_tools', getattr(self, 'repl_mcp_enumerate_tools', True))
            if enumerate_tools:
                output = repl.execute(f"print(repr({name}.list_tools()))", timeout=10.0)
                tools = self._parse_repl_repr(output)
                if tools:
                    import ast
                    try:
                        self._repl_mcp_tools[name] = ast.literal_eval(tools)
                    except (ValueError, SyntaxError):
                        pass

    def _parse_repl_repr(self, output: str) -> Optional[str]:
        """Parse repr() output from REPL, handling the output format."""
        lines = output.strip().split('\n')
        # Skip any prompt/echo lines, get the actual output
        for line in lines:
            line = line.strip()
            if line and not line.startswith('>>>'):
                return line
        return None

    def _build_system_prompt(self):
        # Get base system prompt from chain (includes REPL instructions)
        system = super()._build_system_prompt()

        servers_info = getattr(self, '_repl_mcp_servers_info', {})
        if not servers_info:
            return system

        # Add MCP context
        mcp_parts = [
            "",
            "",
            "MCP CLIENTS IN REPL:",
            "=" * 40,
            "The following MCP client instances are available in the Python REPL:",
        ]

        for name, info in servers_info.items():
            mcp_parts.append(f"  - `{name}`: connected to {info['server']}")

        mcp_parts.extend([
            "",
            "Usage:",
            "  tools = client_name.list_tools()  # List available tools",
            "  result = client_name.call_tool('tool_name', {'arg': 'value'})  # Call a tool",
            "  # Result contains 'content' (list of items) and 'isError' (bool)",
        ])

        # Add tool documentation
        tools_info = getattr(self, '_repl_mcp_tools', {})
        if tools_info:
            mcp_parts.append("")
            mcp_parts.append("AVAILABLE MCP TOOLS:")
            mcp_parts.append("=" * 40)
            for name, tools in tools_info.items():
                if tools:
                    mcp_parts.append(f"\n--- {name} ---")
                    for tool in tools:
                        tool_name = tool.get('name', 'unknown')
                        tool_desc = tool.get('description', 'No description')
                        mcp_parts.append(f"  {tool_name}: {tool_desc}")
                        if 'inputSchema' in tool and 'properties' in tool['inputSchema']:
                            props = tool['inputSchema']['properties']
                            if props:
                                params = ', '.join(props.keys())
                                mcp_parts.append(f"      Parameters: {params}")

        # Add server instructions
        instructions = getattr(self, '_repl_mcp_instructions', {})
        if instructions:
            mcp_parts.append("")
            mcp_parts.append("MCP SERVER INSTRUCTIONS:")
            mcp_parts.append("=" * 40)
            for name, instr in instructions.items():
                mcp_parts.append(f"\n--- {name} ---")
                mcp_parts.append(instr)

        return system + '\n'.join(mcp_parts)

    def _cleanup(self):
        # Close MCP clients in REPL
        servers_info = getattr(self, '_repl_mcp_servers_info', {})
        if servers_info and hasattr(self, '_repl') and self._repl is not None:
            for name in servers_info:
                try:
                    self._repl.execute(f"{name}.close()", timeout=5.0)
                except Exception:
                    pass

        # Reset state
        if hasattr(self, '_repl_mcp_servers_info'):
            self._repl_mcp_servers_info = {}
        if hasattr(self, '_repl_mcp_instructions'):
            self._repl_mcp_instructions = {}
        if hasattr(self, '_repl_mcp_tools'):
            self._repl_mcp_tools = {}
        if hasattr(self, '_repl_mcp_initialized'):
            self._repl_mcp_initialized = False

        # Chain to parent (closes REPL)
        super()._cleanup()
