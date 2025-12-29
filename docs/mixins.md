# AgentLib Mixins Reference

Mixins add specialized capabilities to agents. They work with both `BaseAgent` and `REPLAgent`.

For core agent concepts, see [guide.md](guide.md). For REPLAgent, see [replagent.md](replagent.md).

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

**Optional tool filtering:**
- `include`: Whitelist - only expose these tools
- `exclude`: Blacklist - expose all except these

```python
mcp_servers = [
    ('browser', '/path/to/server', {'include': ['navigate', 'click']}),
    ('fs', '/path/to/fs-server', {'exclude': ['delete_file']}),
]
```

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

## Python Tool Mixins

See [python_tools.md](python_tools.md) for `PythonToolMixin`, `PythonToolResponseMixin`, and `PythonMCPMixin`.

## AttachmentMixin (Persistent Context)

Adds named attachments that persist in conversation context. Attachments are rendered into messages for the LLM and can be updated or removed.

```python
from agentlib import BaseAgent, AttachmentMixin

class MyAgent(AttachmentMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)

with MyAgent() as agent:
    agent.attach("config", {"debug": True, "timeout": 30})
    agent.attach("schema", "CREATE TABLE users (id INT, name TEXT)")
    result = agent.run("Update the timeout to 60")
```

**Methods:**
- `attach(name, content)` - Add or update attachment (str, dict, or list)
- `detach(name)` - Remove attachment from context

**Rendering:** Attachments appear as delimited blocks prepended to user/tool messages:
```
-------- BEGIN config --------
{"debug": true, "timeout": 30}
-------- END config ----------

Update the timeout to 60
```

**Updates:** When an attachment changes, the old version is marked as removed:
```
[Attachment removed: config]

-------- BEGIN config --------
{"debug": false, "timeout": 60}
-------- END config ----------
```

**Variant:** `REPLAttachmentMixin` renders attachments as synthetic REPL file-read exchanges for use with `REPLAgent`.

## Combining Mixins

```python
from agentlib import BaseAgent, MCPMixin, SubShellMixin, PythonToolMixin

class PowerAgent(PythonToolMixin, SubShellMixin, MCPMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You have shell, Python, and MCP server access."
    mcp_servers = [('fs', 'npx -y @mcp/server-filesystem /tmp')]

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)
```

**With REPLAgent:**
```python
from agentlib import REPLAgent, MCPMixin
from agentlib.cli import CLIMixin

class MyAgent(CLIMixin, MCPMixin, REPLAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a coding assistant."
    mcp_servers = [('browser', 'npx -y @anthropic/mcp-server-puppeteer')]
    interactive = True
```

> **Tip:** For a production-ready coding assistant, see `agentlib.agents.CodeAgent` which combines REPLAgent, CLIMixin, and built-in file/search/web tools. Run `code-agent` from the command line or extend it programmatically.

**Lightweight MCP + direct response:**
```python
from agentlib import BaseAgent, PythonMCPMixin, PythonToolResponseMixin

class DataAgent(PythonMCPMixin, PythonToolResponseMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "Query data and return formatted results."
    repl_mcp_servers = [('db', 'python db_server.py')]
```

**Mixin order:** List mixins before `BaseAgent` or `REPLAgent`.

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
from agentlib import PythonToolResponseMixin
from agentlib.cli import CLIAgent

class CodeAssistant(PythonToolResponseMixin, CLIAgent):
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
from agentlib import BaseAgent, PythonToolResponseMixin
from agentlib.cli import CLIMixin

class CustomAgent(CLIMixin, PythonToolResponseMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are helpful."
    welcome_message = "Hello!"

with CustomAgent() as agent:
    agent.cli_run()
```

### With MCP Servers

```python
from agentlib import PythonMCPMixin
from agentlib.cli import CLIAgent

class MCPAssistant(PythonMCPMixin, CLIAgent):
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

## FilePatchMixin (File Editing)

Adds efficient file patching with context-based matching. Supports adding, updating, and deleting files.

```python
from agentlib import BaseAgent, FilePatchMixin

class MyAgent(FilePatchMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a coding assistant."
    patch_preview = None  # True=always, False=never, None=agent decides

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)
```

### patch_preview Configuration

| Value | Behavior |
|-------|----------|
| `True` | Always require user approval |
| `False` | Auto-apply without approval |
| `None` | Agent decides per-call (preview field in tool) |

### Tool: apply_patch

**Parameters:**
- `patch` (str): Patch text in format below
- `preview` (bool): Request approval (only when `patch_preview=None`)
- `preamble` (str): Optional text before preview
- `postamble` (str): Optional text after preview

### Patch Format

```
*** Begin Patch
*** Add File: path/to/new.py
+line 1
+line 2
*** Update File: path/to/existing.py
@@ def function_name():
 context line
-old line
+new line
 context line
*** Delete File: path/to/remove.py
*** End Patch
```

**Operations:**
- `*** Add File:` - Create file (all lines start with `+`)
- `*** Update File:` - Modify file with hunks
- `*** Delete File:` - Remove file

**Hunk lines:**
- ` ` (space) - Context (unchanged)
- `-` - Remove line
- `+` - Add line
- `@@` - Locate change by context (e.g., function name)
- `*** End of File` - Anchor at file end

### Approval Hook

```python
def _prompt_patch_approval(self, preview_text, preamble, postamble):
    """Override for custom approval UI.
    Returns: (approved: bool, comments: str, disable_future_preview: bool)
    """
    return True, "", False  # Default: auto-approve
```

### With CLIMixin

```python
from agentlib import FilePatchMixin
from agentlib.cli import CLIAgent

class Editor(FilePatchMixin, CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a code editor."
```

CLIMixin automatically provides interactive approval: `[Y]es / [N]o / [A]lways`

### Path Resolution

Paths are relative to base path (in order):
1. `PWD` environment variable
2. Directory of main script
3. Current working directory

### Notes

- Context-based matching is resilient to line number changes
- Unicode punctuation normalized during matching
- Whitespace-insensitive matching as fallback
- Multi-file patches in single operation

## JinaMixin (Web Tools)

Adds `web_fetch` and `web_search` tools powered by Jina AI.

```python
from agentlib import BaseAgent, JinaMixin

class MyAgent(JinaMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a research assistant."

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)
```

**Tools provided:**
- `web_fetch(url, ...)` - Fetch URL as LLM-friendly markdown
- `web_search(query, ...)` - Search web, return results with content extracted

**Configuration:**
- `JINA_API_KEY` env var for higher rate limits (optional, free at jina.ai)
- `jina_timeout = 60.0` - Default request timeout

**Common options:**
- `target_selector` / `remove_selector` - CSS selectors to include/exclude
- `return_format` - 'markdown', 'html', 'text'
- `engine` - 'browser' (quality), 'direct' (speed)
- `no_cache` - Bypass cache for fresh content
- `site` - Limit search to domain (web_search only)
- `num` - Max results (web_search only)

## SandboxMixin (Filesystem Isolation)

Runs agent code in an isolated overlay filesystem. All writes go to a temporary layer; real filesystem unchanged until explicitly applied.

**Linux only.** Requires user namespaces and gcc. See [sandbox.md](sandbox.md) for full documentation.

```python
from agentlib import SandboxMixin, CodeAgent

class SandboxedAgent(SandboxMixin, CodeAgent):
    sandbox_target = "/home/user"  # Optional, defaults to $HOME

with SandboxedAgent() as agent:
    agent.run("Create ~/test.txt")

# Review changes
print(agent.get_changed_files())

# Apply to real filesystem
agent.apply_changes()
```

**Methods:**
- `get_tarball()` - Raw tarball bytes of all changes
- `get_changed_files()` - Dict of `{path: content}`
- `get_deleted_files()` - List of deleted paths
- `apply_changes(target_dir=None)` - Apply to real filesystem
- `discard_changes()` - No-op (changes already isolated)
