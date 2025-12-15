"""
MCPAgent - BaseAgent with MCP server integration.

Example:
    class MyAgent(MCPAgent):
        model = 'google/gemini-2.5-flash'
        system = "You are a helpful assistant."
        mcp_servers = [
            ('fs', 'npx -y @mcp/server-filesystem /tmp'),
            ('api', 'http://localhost:3000/sse'),
            ('db', 'python db_server.py', {'timeout': 60.0}),
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
    - Call close() when done, or use as context manager.
"""

from pydantic import create_model, Field
from typing import Optional, Any

from .agent import BaseAgent
from .mcp import MCPClient, create_stdio_client, create_sse_client, MCPError


class MCPAgent(BaseAgent):
    """BaseAgent with MCP server integration."""

    mcp_servers: list = []  # List of (name, server) or (name, server, options) tuples

    def __init__(self):
        self._mcp_clients: dict[str, MCPClient] = {}
        self._mcp_tools: dict[str, tuple[MCPClient, str, dict]] = {}  # tool_name -> (client, orig_name, tool_def)
        self._mcp_instructions: dict[str, str] = {}  # server_name -> instructions

        for server_def in self.mcp_servers:
            self.connect_mcp(*server_def)

    def connect_mcp(self, name: str, server: str, options: Optional[dict] = None):
        """
        Connect to an MCP server (auto-detects transport type).

        Args:
            name: Identifier for this connection (used as tool name prefix)
            server: Server address - URL for SSE (http://...) or command string for stdio
            options: Optional dict of kwargs passed to underlying connect method

        Examples:
            agent.connect_mcp('fs', '/usr/bin/mcp-server-filesystem /tmp')
            agent.connect_mcp('api', 'http://localhost:3000/sse', {'headers': {'Authorization': 'Bearer xxx'}})
        """
        opts = options or {}
        if server.startswith('http://') or server.startswith('https://'):
            self.connect_mcp_sse(name, server, **opts)
        else:
            opts.setdefault('forward_stderr', False)
            self.connect_mcp_stdio(name, server.split(), **opts)

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
        self._register_mcp(name, client)

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
        self._register_mcp(name, client)

    def _register_mcp(self, name: str, client: MCPClient):
        """Register an MCP client and cache its tools and instructions."""
        self._mcp_clients[name] = client

        # Cache instructions
        if client.instructions:
            self._mcp_instructions[name] = client.instructions

        # Cache tools
        for tool_def in client.list_tools():
            orig_name = tool_def['name']
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

    def close(self):
        """Close all MCP connections."""
        for client in self._mcp_clients.values():
            try:
                client.close()
            except Exception:
                pass
        self._mcp_clients.clear()
        self._mcp_tools.clear()
        self._mcp_instructions.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def conversation(self):
        try:
            return self._conversation
        except AttributeError:
            assert hasattr(self, 'system'), "system must be defined"
            system = self.system
            if self._mcp_instructions:
                parts = [f"=== {n} ===\n{i}" for n, i in self._mcp_instructions.items()]
                system += "\n\nMCP SERVER INSTRUCTIONS:\n" + "\n\n".join(parts)
            self._conversation = self.llm_client.conversation(system)
            return self._conversation

    @property
    def toolspecs(self):
        specs = super().toolspecs.copy()
        for tool_name, (client, orig_name, tool_def) in self._mcp_tools.items():
            specs[tool_name] = self._make_spec(tool_name, tool_def)
        return specs

    def toolcall(self, toolname, function_args):
        if toolname in self._mcp_tools:
            client, orig_name, _ = self._mcp_tools[toolname]
            try:
                result = client.call_tool(orig_name, function_args)
                return self._format_result(result)
            except MCPError as e:
                return f"[MCP Error] {e}"
        return super().toolcall(toolname, function_args)

    def _make_spec(self, tool_name: str, tool_def: dict) -> Any:
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

    def _format_result(self, result: dict) -> str:
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
