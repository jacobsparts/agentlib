# AgentLib Mixins Reference

Mixins add specialized capabilities to agents. For core agent concepts, see [LLM-GUIDE.md](LLM-GUIDE.md).

## MCPMixin (MCP Integration)

`MCPMixin` adds MCP (Model Context Protocol) server support to any agent via mixin composition.

### Basic Structure

```python
from agentlib import BaseAgent, MCPMixin

class MyAgent(MCPMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."
    mcp_servers = [
        ('fs', 'npx -y @mcp/server-filesystem /tmp'),
        ('api', 'http://localhost:3000/sse'),
        ('db', 'python db_server.py', {'timeout': 60.0}),
    ]

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        """Send response to user."""
        self.respond(response)
```

### mcp_servers Format

```python
mcp_servers = [
    (name, server),              # Basic
    (name, server, options),     # With options dict
]
```

**Transport auto-detection:**
- `http://` or `https://` → SSE transport
- Anything else → Stdio transport (command split on spaces)

**Options:**
- Stdio: `timeout`, `forward_stderr` (default: False), `env`
- SSE: `timeout`, `headers`

### Tool Naming

MCP tools are prefixed with server name:
```python
# mcp_servers = [('browser', '/path/to/server')]
# Server exposes: navigate, click, screenshot
# Agent gets: browser_navigate, browser_click, browser_screenshot
```

### Server Instructions

MCP server instructions are auto-appended to system prompt:
```
{your system prompt}

MCP SERVER INSTRUCTIONS:
=== server_name ===
{instructions from server}
```

### Runtime Methods

```python
agent = MyAgent()

# Connect dynamically
agent.connect_mcp('name', 'server_command_or_url')
agent.connect_mcp('name', 'server', {'timeout': 30.0})

# Disconnect
agent.disconnect_mcp('name')

# Clean up all
agent.close()
```

### Lifecycle

```python
# Context manager (recommended)
with MyAgent() as agent:
    result = agent.run("Do something")

# Manual
agent = MyAgent()
try:
    result = agent.run("Do something")
finally:
    agent.close()
```

### Notes

- Tools cached on connect; reconnect to refresh if server tools change
- MCP errors returned as `"[MCP Error] ..."` strings
- Server stderr suppressed by default

## SubShellMixin (Bash Shell)

Adds persistent bash shell. Environment and cwd persist across calls.

```python
from agentlib import BaseAgent, SubShellMixin

class MyAgent(SubShellMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant with shell access."
    shell_timeout = 30.0  # Default timeout (optional)

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)
```

**Tools provided:**
- `shell_execute(command, timeout=30)` - Run bash command
- `shell_read(timeout=30)` - Continue reading if output ended with `[still running]\n`
- `shell_interrupt()` - Send SIGINT to stop command

**Direct access:** `agent.shell_execute("ls -la")`

## SubREPLMixin (Python REPL)

Adds persistent Python REPL. Variables, imports, functions persist across calls.

```python
from agentlib import BaseAgent, SubREPLMixin

class MyAgent(SubREPLMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant with Python execution."
    repl_timeout = 30.0  # Default timeout (optional)

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)
```

**Tools provided:**
- `python_execute(code, timeout=30)` - Run Python code
- `python_read(timeout=30)` - Continue reading if output ended with `[still running]\n`
- `python_interrupt()` - Send SIGINT to stop code

**Direct access:** `agent.python_execute("print(1+1)")`

**Variant:** `SubREPLResponseMixin` adds `python_execute_response(code, preamble, postamble)` - executes code and returns output as the agent's final response.

## Combining Mixins

```python
from agentlib import BaseAgent, MCPMixin, SubShellMixin, SubREPLMixin

class PowerAgent(SubREPLMixin, SubShellMixin, MCPMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You have shell, Python, and MCP server access."
    mcp_servers = [('fs', 'npx -y @mcp/server-filesystem /tmp')]

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)
```

**Mixin order:** List mixins before `BaseAgent`.
