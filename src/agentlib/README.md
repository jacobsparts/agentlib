# agentlib source structure

## Core

| Module | Description |
|--------|-------------|
| `agent.py` | `BaseAgent`, `@tool` decorator, `AgentMeta` metaclass |
| `client.py` | `LLMClient` - unified API for LLM providers |
| `conversation.py` | Conversation state management |
| `llm_registry.py` | Model/provider registration |
| `config.py` | User configuration loading |
| `core.py` | Core re-exports |

## Agent Paradigms

| Module | Description |
|--------|-------------|
| `tool_mixin.py` | `ToolMixin` - base mixin for method-based tool dispatch |
| `repl_agent.py` | `REPLAgent` - LLM writes Python code, tool injection/relay |
| `python_tool_mixin.py` | `PythonToolMixin` - adds `python_execute` tool to any agent |

## Capability Mixins

| Module | Description |
|--------|-------------|
| `mcp_mixin.py` | `MCPMixin` - MCP server integration |
| `shell_mixin.py` | `SubShellMixin` - persistent bash shell |
| `jina_mixin.py` | `JinaMixin` - `web_fetch`, `web_search` via Jina AI |
| `patch_mixin.py` | `FilePatchMixin` - unified diff patching |
| `attachment_mixin.py` | `AttachmentMixin` - persistent context files |
| `repl_attachment_mixin.py` | `REPLAttachmentMixin` - attachments as REPL output |

## Isolation

| Module | Description |
|--------|-------------|
| `sandbox/__init__.py` | Sandbox utilities, helper compilation |
| `sandbox/mixin.py` | `SandboxMixin`, `SandboxedToolREPL` - overlay filesystem isolation |

## CLI

| Module | Description |
|--------|-------------|
| `cli/__init__.py` | `CLIAgent` convenience class |
| `cli/mixin.py` | `CLIMixin` - interactive terminal UI |
| `cli/prompt.py` | Readline replacement with bracketed paste |
| `cli/terminal.py` | ANSI codes, markdown rendering, `Panel`, `Console` |

## Tools (subprocess implementations)

| Module | Description |
|--------|-------------|
| `tools/subrepl.py` | `SubREPL` - streaming Python subprocess |
| `tools/subshell.py` | `SubShell` - streaming bash subprocess |
| `tools/mcp.py` | `MCPClient` - stdio/SSE MCP protocol client |
| `tools/apply_patch.py` | Unified diff parser/applier |

## Ready-to-use Agents

| Module | Description |
|--------|-------------|
| `agents/code_agent.py` | `CodeAgent` - full-featured coding assistant with file ops |

---

## Tool Bridging (repl_agent.py)

REPLAgent tools can be **injected** or **relayed**:

```python
@REPLAgent.tool(inject=True)   # Source extracted, runs in subprocess
def read(self, path): ...

@REPLAgent.tool                 # Default: relay via IPC
def mcp_tool(self, arg): ...
```

| Function | Purpose |
|----------|---------|
| `_extract_tool_source()` | AST transform for `inject=True` tools |
| `_extract_stub_signature()` | Shared signature extraction |
| `_generate_tool_stub()` | Queue-based relay stub (ToolREPL) |
| `_generate_socket_relay_stub()` | Socket-based relay stub (SandboxedToolREPL) |

**Injection primitives:**

| Class | `_inject_code()` | Transport |
|-------|------------------|-----------|
| `SubREPL` | Base implementation | execute() with echo disabled |
| `ToolREPL` | Override | Queue (direct to worker) |
| `SandboxedToolREPL` | Override | Socket |

`inject_startup(code_list)` calls `_inject_code()` for each item.

---

## REPL Startup Code

All REPL-based agents support `repl_startup` for injecting code at session start:

```python
class MyAgent(REPLAgent):
    repl_startup = [
        'import sqlite3',
        '''
def sql_query(query):
    conn = sqlite3.connect("data.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
''',
    ]

class DynamicAgent(REPLAgent):
    db_path = "data.db"
    
    @property
    def repl_startup(self):
        return [f'DB_PATH = {self.db_path!r}']
```

Supported by:
- `REPLAgent` / `REPLMixin` (via ToolREPL)
- `PythonToolMixin` (via SubREPL)
- `SandboxMixin` (via SandboxedToolREPL)

The startup code is injected silently (no output shown to agent).
