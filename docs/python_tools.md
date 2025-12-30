# Python Tool Mixins

These mixins add Python execution to **tool-calling agents** (BaseAgent). The model calls `python_execute` as a tool.

For the **code-first paradigm** where the LLM writes Python directly as its response, see [replagent.md](replagent.md).

## PythonToolMixin

Adds persistent Python REPL as a tool. Variables, imports, functions persist across calls.

```python
from agentlib import BaseAgent, PythonToolMixin

class MyAgent(PythonToolMixin, BaseAgent):
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

### @repl Decorator

Mark methods for injection into the REPL as callable functions:

```python
from agentlib import BaseAgent, PythonToolMixin

class MyAgent(PythonToolMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You have a multiply() function available."

    @PythonToolMixin.repl
    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)
```

The agent can then use `multiply(6, 7)` directly in `python_execute`. The method is transformed:
- `self` parameter removed
- Type annotations stripped
- String defaults converted to `None`
- `getattr(self, 'attr', default)` replaced with `default`

### repl_startup

Inject arbitrary code strings at REPL initialization:

```python
class MyAgent(PythonToolMixin, BaseAgent):
    repl_startup = [
        "import numpy as np",
        "from pathlib import Path",
        "DATA_DIR = '/tmp/data'",
    ]
    # Or as a method:
    def repl_startup(self):
        return [f"BASE_URL = '{self.base_url}'"]
```

## PythonToolResponseMixin

Extends `PythonToolMixin` - includes all its tools plus `python_execute_response(code, preamble, postamble)`. Use this instead of `PythonToolMixin` when you want the agent's code output to become its response.

If code throws an exception or times out, error returns to agent for retry.

```python
from agentlib import BaseAgent, PythonToolResponseMixin

class CalcAgent(PythonToolResponseMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a calculator. Use python_execute_response to compute and return results."
```

## PythonMCPMixin

Alternative to `MCPMixin` that uses fewer tokens. MCP clients are pre-instantiated in the REPL; agent calls tools via code.

```python
from agentlib import BaseAgent, PythonMCPMixin

class MyAgent(PythonMCPMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."
    repl_mcp_servers = [
        ('fs', '/path/to/mcp-server-filesystem /tmp'),
        ('api', 'http://localhost:3000/sse'),
        ('db', 'python db_server.py', {'timeout': 60.0}),
    ]

    @BaseAgent.tool
    def done(self, response: str = "Response"):
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
class MyAgent(PythonMCPMixin, BaseAgent):
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

Combine with `PythonToolResponseMixin` for direct output:
```python
class MyAgent(PythonMCPMixin, PythonToolResponseMixin, BaseAgent):
    repl_mcp_servers = [...]
```

MRO handles diamond inheritance correctly—`PythonToolMixin` appears once.

### MCPMixin vs PythonMCPMixin

| Aspect | MCPMixin | PythonMCPMixin |
|--------|----------|--------------|
| Token usage | Higher (tool per MCP function) | Lower (tools via code) |
| Agent complexity | Simpler (native function calls) | Requires writing code |
| Best for | Simple MCP usage | Code-oriented agents |
