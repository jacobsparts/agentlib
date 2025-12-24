"""
MCPMixin - Mixin that adds MCP server integration to any agent.

Example:
    from agentlib import BaseAgent, MCPMixin

    class MyAgent(MCPMixin, BaseAgent):
        model = 'google/gemini-2.5-flash'
        system = "You are a helpful assistant."
        mcp_servers = [
            ('fs', 'npx -y @mcp/server-filesystem /tmp'),
            ('api', 'http://localhost:3000/sse'),
            ('db', 'python db_server.py', {'timeout': 60.0}),
            # Optional: expose only specific tools
            ('browser', '/path/to/browser-server', {'include': ['navigate', 'screenshot']}),
            # Optional: expose all except certain tools
            ('other', '/path/to/server', {'exclude': ['dangerous_tool']}),
        ]

        @BaseAgent.tool
        def done(self, response: str = "Your response"):
            self.respond(response)

    with MyAgent() as agent:
        # Tools available: fs_read_file, fs_list_directory, api_*, db_*, etc.
        result = agent.run("List files in /tmp")

Notes:
    - Tools are cached on connect. If an MCP server adds/removes tools dynamically,
      call disconnect_mcp() and reconnect to refresh.
    - Optional 'include'/'exclude' filters let you expose a subset of tools.
    - Call close() when done, or use as context manager.
"""

import threading
from pydantic import create_model, Field
from typing import Optional, Any

from .mcp import create_stdio_client, create_sse_client, MCPError
from .tool_mixin import ToolMixin


class MCPMixin(ToolMixin):
    """Mixin that adds MCP server integration. Use with BaseAgent."""

    mcp_servers: list = []  # List of (name, server) or (name, server, options) tuples

    # === HOOK IMPLEMENTATIONS ===

    def _ensure_setup(self):
        # Chain to next in MRO (might be another mixin or BaseAgent)
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        # Fast path - already initialized
        if getattr(self, '_mcp_initialized', False):
            return

        # Thread-safe double-checked locking (gevent compatible)
        if not hasattr(self, '_mcp_lock'):
            self._mcp_lock = threading.Lock()

        with self._mcp_lock:
            # Check again inside lock
            if getattr(self, '_mcp_initialized', False):
                return

            self._mcp_clients: dict[str, Any] = {}
            self._mcp_tools: dict[str, tuple[Any, str, dict]] = {}
            self._mcp_instructions: dict[str, str] = {}

            try:
                for server_def in getattr(self, 'mcp_servers', []):
                    self.connect_mcp(*server_def)
                self._mcp_initialized = True
            except:
                # Rollback on failure
                for client in self._mcp_clients.values():
                    try:
                        client.close()
                    except Exception:
                        pass
                self._mcp_clients = {}
                self._mcp_tools = {}
                self._mcp_instructions = {}
                raise

    def _build_system_prompt(self):
        # Get base system prompt from chain
        if hasattr(super(), '_build_system_prompt'):
            system = super()._build_system_prompt()
        else:
            system = getattr(self, 'system', '')

        instructions = getattr(self, '_mcp_instructions', {})
        if instructions:
            parts = [f"=== {n} ===\n{i}" for n, i in instructions.items()]
            system += "\n\nMCP SERVER INSTRUCTIONS:\n" + "\n\n".join(parts)

        return system

    def _get_dynamic_toolspecs(self):
        # Get specs from chain first
        if hasattr(super(), '_get_dynamic_toolspecs'):
            specs = super()._get_dynamic_toolspecs()
        else:
            specs = {}

        for tool_name, (client, orig_name, tool_def) in getattr(self, '_mcp_tools', {}).items():
            specs[tool_name] = self._make_mcp_spec(tool_name, tool_def)

        return specs

    def _dispatch_tool(self, toolname, function_args):
        """Handle MCP tool calls dynamically."""
        mcp_tools = getattr(self, '_mcp_tools', {})
        if toolname in mcp_tools:
            try:
                result = self._call_mcp_raw(toolname, **function_args)
                return True, self._format_mcp_result(result)
            except MCPError as e:
                return True, f"[MCP Error] {e}"
        return None

    def _call_mcp_raw(self, toolname, **function_args):
        """
        Call an MCP tool and return the raw result dict.

        Use this in subclass method overrides to get unformatted results:

            def claude_Bash(self, command, **kw):
                r = self._call_mcp_raw('claude_Bash', command=command, **kw)
                return r.get('stdout', '') + r.get('stderr', '')
        """
        client, orig_name, _ = self._mcp_tools[toolname]
        return client.call_tool(orig_name, function_args)

    def _cleanup(self):
        # Clean up MCP connections
        for client in getattr(self, '_mcp_clients', {}).values():
            try:
                client.close()
            except Exception:
                pass

        if hasattr(self, '_mcp_clients'):
            self._mcp_clients = {}
        if hasattr(self, '_mcp_tools'):
            self._mcp_tools = {}
        if hasattr(self, '_mcp_instructions'):
            self._mcp_instructions = {}
        if hasattr(self, '_mcp_initialized'):
            self._mcp_initialized = False

        # Chain to next
        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    # === MCP-SPECIFIC METHODS ===

    def connect_mcp(self, name: str, server: str, options: Optional[dict] = None):
        """
        Connect to an MCP server (auto-detects transport type).

        Args:
            name: Identifier for this connection (used as tool name prefix)
            server: Server address - URL for SSE (http://...) or command string for stdio
            options: Optional dict with:
                - timeout: Request timeout in seconds
                - forward_stderr: Forward server stderr (stdio only, default: False)
                - env: Environment variables (stdio only)
                - headers: HTTP headers (SSE only)
                - include: Optional list of tool names to expose (whitelist)
                - exclude: Optional list of tool names to hide (blacklist)

        Examples:
            agent.connect_mcp('fs', '/usr/bin/mcp-server-filesystem /tmp')
            agent.connect_mcp('api', 'http://localhost:3000/sse', {'headers': {'Authorization': 'Bearer xxx'}})
            agent.connect_mcp('browser', '/path/to/server', {'include': ['navigate', 'click']})
        """
        opts = options.copy() if options else {}
        include = opts.pop('include', None)
        exclude = opts.pop('exclude', None)

        if server.startswith('http://') or server.startswith('https://'):
            client = create_sse_client(server, **opts)
        else:
            opts.setdefault('forward_stderr', False)
            client = create_stdio_client(server.split(), **opts)

        self._register_mcp_client(name, client, include=include, exclude=exclude)

    def connect_mcp_stdio(
        self,
        name: str,
        command: list[str],
        env: Optional[dict[str, str]] = None,
        timeout: float = 300.0,
        forward_stderr: bool = True
    ):
        """
        Connect to an MCP server via subprocess.

        Args:
            name: Identifier for this connection (used as tool name prefix)
            command: Command to spawn the server (e.g., ['python', 'server.py'])
            env: Additional environment variables for the subprocess
            timeout: Default timeout for MCP operations
            forward_stderr: If True, forward server stderr to client stderr
        """
        client = create_stdio_client(
            command,
            env=env,
            timeout=timeout,
            forward_stderr=forward_stderr
        )
        self._register_mcp_client(name, client)

    def connect_mcp_sse(
        self,
        name: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 300.0
    ):
        """
        Connect to an MCP server via SSE/HTTP.

        Args:
            name: Identifier for this connection (used as tool name prefix)
            url: SSE endpoint URL (e.g., 'http://localhost:3000/sse')
            headers: HTTP headers (e.g., for authentication)
            timeout: Default timeout for MCP operations
        """
        client = create_sse_client(
            url,
            headers=headers,
            timeout=timeout
        )
        self._register_mcp_client(name, client)

    def _register_mcp_client(self, name: str, client, include: list = None, exclude: list = None):
        """Register an MCP client and cache its tools and instructions.

        Args:
            name: Server name prefix
            client: MCP client instance
            include: Optional list - only register these tool names (whitelist)
            exclude: Optional list - skip these tool names (blacklist)
        """
        self._mcp_clients[name] = client

        # Cache instructions
        if client.instructions:
            self._mcp_instructions[name] = client.instructions

        # Cache tools (with optional filtering)
        for tool_def in client.list_tools():
            orig_name = tool_def['name']

            # Apply optional filters
            if include is not None and orig_name not in include:
                continue
            if exclude is not None and orig_name in exclude:
                continue

            tool_name = f"{name}_{orig_name}"
            self._mcp_tools[tool_name] = (client, orig_name, tool_def)

    def disconnect_mcp(self, name: str):
        """
        Disconnect an MCP server and unregister its tools.

        Args:
            name: The server identifier used in connect_mcp_*
        """
        client = self._mcp_clients.get(name)
        if not client:
            return

        # Remove tools from this client
        self._mcp_tools = {
            k: v for k, v in self._mcp_tools.items()
            if v[0] is not client
        }

        # Remove instructions
        self._mcp_instructions.pop(name, None)

        # Close and remove client
        del self._mcp_clients[name]
        try:
            client.close()
        except Exception:
            pass

    def _make_mcp_spec(self, tool_name: str, tool_def: dict) -> Any:
        """Convert MCP tool JSON Schema to Pydantic model."""
        schema = tool_def.get('inputSchema', {})
        props = schema.get('properties', {})
        required = set(schema.get('required', []))

        type_map = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }

        fields = {}
        for pname, pschema in props.items():
            ptype = type_map.get(pschema.get('type', 'string'), str)
            desc = pschema.get('description', '')
            if pname in required:
                fields[pname] = (ptype, Field(..., description=desc))
            else:
                fields[pname] = (Optional[ptype], Field(None, description=desc))

        model_name = ''.join(w.title() for w in tool_name.split('_'))
        model = create_model(model_name, **fields)
        model.__doc__ = tool_def.get('description', tool_name)
        return model

    def _format_mcp_result(self, result: dict) -> str:
        """Format MCP tool result as string for conversation."""
        parts = []
        for item in result.get('content', []):
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'image':
                parts.append(f"[Image: {item.get('mimeType', 'unknown')}]")
            else:
                parts.append(str(item))
        text = '\n'.join(parts) if parts else str(result)
        return f"[MCP Error] {text}" if result.get('isError') else text
