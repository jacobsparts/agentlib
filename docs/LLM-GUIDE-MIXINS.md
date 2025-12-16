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

**Variant:** `SubREPLResponseMixin` adds `python_execute_response(code, preamble, postamble)` - executes code and returns output as the agent's final response. If code throws an exception or times out, error returns to agent for retry.

## REPLMCPMixin (Lightweight MCP)

Alternative to `MCPMixin` that uses fewer tokens. MCP clients are pre-instantiated in the REPL; agent calls tools via code.

```python
from agentlib import BaseAgent, REPLMCPMixin

class MyAgent(REPLMCPMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."
    repl_mcp_servers = [
        ('fs', '/path/to/mcp-server-filesystem /tmp'),
        ('api', 'http://localhost:3000/sse'),
        ('db', 'python db_server.py', {'timeout': 60.0}),
    ]

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        """Send response to user."""
        self.respond(response)
```

### repl_mcp_servers Format

Same as `mcp_servers` in MCPMixin:
```python
repl_mcp_servers = [
    (name, server),              # Basic
    (name, server, options),     # With options dict
]
```

### Configuration

```python
class MyAgent(REPLMCPMixin, BaseAgent):
    repl_mcp_servers = [...]
    repl_mcp_enumerate_tools = True  # Enumerate tools at setup (default: True)
```

Set `repl_mcp_enumerate_tools = False` to skip tool enumeration—useful when there are many MCP servers. Tools can still be discovered at runtime via `client.list_tools()`.

Per-server override:
```python
repl_mcp_servers = [
    ('fs', '/path/to/server'),                          # Uses global default
    ('api', 'http://localhost/sse', {'enumerate_tools': False}),  # Skip for this server
]
```

### How Agent Uses MCP

MCP clients available as REPL variables. Agent writes code:
```python
# In python_execute:
result = fs.call_tool('read_file', {'path': '/tmp/test.txt'})
for item in result['content']:
    if item['type'] == 'text':
        print(item['text'])
```

### MCP Client Methods

```python
client.list_tools()                           # List available tools
client.call_tool('name', {'arg': 'value'})    # Call a tool
# Returns: {'content': [...], 'isError': bool}
```

### With python_execute_response

Combine with `SubREPLResponseMixin` for direct output:
```python
class MyAgent(REPLMCPMixin, SubREPLResponseMixin, BaseAgent):
    repl_mcp_servers = [...]
```

MRO handles diamond inheritance correctly—`SubREPLMixin` appears once.

### MCPMixin vs REPLMCPMixin

| Aspect | MCPMixin | REPLMCPMixin |
|--------|----------|--------------|
| Token usage | Higher (tool per MCP function) | Lower (tools via code) |
| Agent complexity | Simpler (native function calls) | Requires writing code |
| Best for | Simple MCP usage | Code-oriented agents |

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

**Lightweight MCP + direct response:**
```python
from agentlib import BaseAgent, REPLMCPMixin, SubREPLResponseMixin

class DataAgent(REPLMCPMixin, SubREPLResponseMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "Query data and return formatted results."
    repl_mcp_servers = [('db', 'python db_server.py')]
```

**Mixin order:** List mixins before `BaseAgent`.

## CLIMixin (Interactive Terminal)

Adds interactive CLI REPL functionality with terminal rendering, readline history, and customizable hooks.

### Quick Start with CLIAgent

`CLIAgent` pre-composes `CLIMixin + BaseAgent`:

```python
from agentlib.cli import CLIAgent

class MyAssistant(CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."
    welcome_message = "[bold]My Bot[/bold]\nReady to help!"

if __name__ == "__main__":
    MyAssistant.main()
```

Add mixins for additional capabilities:
```python
from agentlib import SubREPLResponseMixin
from agentlib.cli import CLIAgent

class CodeAssistant(SubREPLResponseMixin, CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a Python assistant."
```

### Configuration

```python
class MyAssistant(CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are helpful."

    # CLI options
    welcome_message = "[bold]Welcome![/bold]"  # Supports markup
    cli_prompt = ">>> "                        # Input prompt (default: "> ")
    history_db = "~/.myapp_history.db"         # SQLite history path
    max_turns = 20                             # Max iterations per message
    thinking_message = "Processing..."         # Status while working
```

### Markup Tags

For `welcome_message` and `console.print()`:
- Style: `[bold]`, `[dim]`, `[italic]`, `[underline]`, `[strike]`
- Color: `[red]`, `[green]`, `[yellow]`, `[blue]`, `[magenta]`, `[cyan]`, `[white]`, `[gray]`
- Close: `[/bold]` or `[/]`

### Customization Hooks

Override to customize tool display:

```python
class MyAssistant(CLIAgent):
    def on_tool_call(self, name, args):
        """Called before tool executes."""
        self.console.clear_line()
        if name == 'python_execute':
            self.console.panel(args.get('code', ''), title="Code", border_style="blue")

    def on_tool_result(self, name, result):
        """Called after tool returns."""
        if result:
            self.console.panel(str(result)[:1000], title="Output", border_style="green")

    def format_response(self, response):
        """Format final response. Default: render_markdown(response)"""
        return response  # Return plain text
```

### Using CLIMixin Directly

```python
from agentlib import BaseAgent, SubREPLResponseMixin
from agentlib.cli import CLIMixin

class CustomAgent(CLIMixin, SubREPLResponseMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are helpful."
    welcome_message = "Hello!"

with CustomAgent() as agent:
    agent.cli_run()
```

### With MCP Servers

```python
from agentlib import REPLMCPMixin
from agentlib.cli import CLIAgent

class MCPAssistant(REPLMCPMixin, CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are helpful."
    repl_mcp_servers = [
        ('fs', '/path/to/mcp-server /tmp'),
    ]

MCPAssistant.main()
```

### Console API

```python
# Available via self.console in hooks
self.console.print("[bold]text[/bold]")           # Print with markup
self.console.panel("content", title="T")          # Print bordered panel
self.console.markdown("# Heading")                # Render markdown
self.console.status("Loading...")                 # Status (no newline)
self.console.clear_line()                         # Clear current line
```

### Panel Border Styles

`cyan`, `blue`, `green`, `red`, `magenta`, `yellow`, `gray`, `white`

### Terminal Utilities

```python
from agentlib.cli import (
    render_markdown,      # Markdown → ANSI
    highlight_python,     # Python syntax highlighting
    parse_markup,         # [bold] tags → ANSI
    strip_ansi,          # Remove ANSI codes
    Panel,               # Bordered box
    Console,             # Output with markup
)
```
