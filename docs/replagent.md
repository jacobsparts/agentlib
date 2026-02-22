# REPLAgent Quick Reference

REPLAgent is an alternative to BaseAgent where the LLM writes Python code directly instead of making JSON tool calls.

For tool-calling agents (BaseAgent), see [guide.md](guide.md).

For tool-calling agents that need Python execution as a tool (not the paradigm), see [python_tools.md](python_tools.md).

## Basic Structure

```python
from agentlib import REPLAgent

class MyAgent(REPLAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."

    @REPLAgent.tool
    def search(self, query: str = "Search query"):
        """Search for information."""
        return do_search(query)

result = MyAgent().run("Find info about Python")
```

**Key difference:** The LLM's response IS Python code. No markdown, no preamble. Code runs directly in a persistent REPL.

## How It Works

1. LLM receives system prompt describing REPL environment
2. LLM writes Python code as response
3. Code executes statement-by-statement in isolated subprocess
4. Output streams back to LLM as REPL feedback
5. Loop continues until `emit(..., release=True)` is called

## Built-in Functions

These are injected directly as Python source into the subprocess — they are not tools.

| Function | Description |
|----------|-------------|
| `emit(value, release=False)` | Output a value. `release=True` ends the run and returns the value. |
| `help(func)` | Get parameter descriptions for any tool |

- `print()`: For your own inspection/debugging — output appears in your next turn
- `emit()`: Deliberate output intended for the caller — use `release=True` to complete

`emit(value, release=True)` is the primary REPLAgent completion mechanism — the emitted value becomes the return value of `agent.run()`. Tools can also use `self.respond(value)` to complete, same as BaseAgent.

## Defining Tools

Same decorator syntax as BaseAgent. Tools become callable Python functions in the REPL.

### Proxy tools (default)

By default, tools are **proxied** — a relay stub runs in the subprocess, serializes args, and calls back to the host process where the real method executes. The tool has full access to `self`.

```python
@REPLAgent.tool
def search(self, query: str = "Search query"):
    """Search the database."""
    results = self.db.query(query)  # Access to self
    return results
```

### Injected tools (`inject=True`)

With `inject=True`, the tool's source code is extracted and injected directly into the subprocess. Faster (no IPC), but no access to `self`.

```python
@REPLAgent.tool(inject=True)
def read_file(self, path: str = "File path"):
    """Read a file from disk."""
    return Path(path).read_text()  # Runs in subprocess, no self access
```

LLM uses both kinds the same way:

```python
data = read_file("config.json")
results = search("SELECT * FROM users")
emit(f"Found {len(results)} users", release=True)
```

## Interactive Mode

For CLI/chat agents, enable multi-turn autonomous work:

```python
class ChatAgent(REPLAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."
    interactive = True  # Enables multi-turn autonomous workflow
```

With `interactive=True`:
- `emit("progress")` - output emitted, agent keeps working
- `emit("result", release=True)` - release control to user
- `emit("question?", release=True)` - ask user a question
- `print()` - output visible to agent (and user) in next turn

The agent controls the conversation until it explicitly releases with `release=True`.

## Configuration

```python
class MyAgent(REPLAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are helpful."
    interactive = False  # Default: single run, emit(release=True) returns to caller
    repl_startup = "import json"  # Optional: code to run silently at REPL startup
```

`repl_startup` can also be a callable that returns a string:

```python
class MyAgent(REPLAgent):
    def repl_startup(self):
        return f"API_KEY = {self.api_key!r}"
```

## Combining with Mixins

REPLAgent works with the same mixins as BaseAgent:

```python
from agentlib import REPLAgent, MCPMixin
from agentlib.cli import CLIMixin

class PowerAgent(CLIMixin, MCPMixin, REPLAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a powerful assistant."
    mcp_servers = [
        ('browser', 'npx -y @anthropic/mcp-server-puppeteer'),
    ]
    welcome_message = "[bold]Power Agent[/bold]"
    interactive = True

if __name__ == "__main__":
    PowerAgent.main()
```

MCP tools become callable functions alongside your defined tools.

**Mixin order:** List mixins before `REPLAgent`.

## Hooks

Override for custom output handling:

```python
class CustomAgent(REPLAgent):
    def on_repl_execute(self, code):
        """Called at start of each turn."""
        pass

    def on_repl_chunk(self, chunk: str):
        """Called for each output chunk (streaming)."""
        pass

    def on_repl_output(self, output: str):
        """Called after execution with full output."""
        pass

    def process_repl_output(self, output: str) -> str:
        """Process/truncate output before sending to LLM."""
        if len(output) > 10000:
            return output[:5000] + "\n...(truncated)..."
        return output
```

## When to Use REPLAgent

| Use REPLAgent | Use BaseAgent |
|---------------|---------------|
| Code-heavy tasks | Structured operations |
| Data analysis | API calls |
| Exploratory workflow | Predictable multi-step |
| Computing results | Well-defined tool contracts |

## Complete Example

```python
from agentlib import REPLAgent

class AnalysisAgent(REPLAgent):
    model = 'google/gemini-2.5-flash'
    system = """You are a data analysis assistant.
    Write Python to analyze data. Use emit(..., release=True) for final answers."""

    @REPLAgent.tool
    def load_csv(self, path: str = "Path to CSV"):
        """Load CSV file as pandas DataFrame."""
        import pandas as pd
        return pd.read_csv(path)

agent = AnalysisAgent()
result = agent.run("Load sales.csv and find the top product by revenue")
print(result)
```

LLM might produce:

```python
# Turn 1
df = load_csv("sales.csv")
df.head()

# Turn 2
top = df.groupby('product')['revenue'].sum().idxmax()
revenue = df.groupby('product')['revenue'].sum().max()
emit(f"Top product: {top} with ${revenue:,.2f} revenue", release=True)
```

## Key Differences from BaseAgent

| Aspect | BaseAgent | REPLAgent |
|--------|-----------|-----------|
| LLM output | JSON tool calls | Python code |
| Tool interface | Schema-validated | Python functions |
| Completion | `self.respond()` | `emit(..., release=True)` or `self.respond()` in tools |
| State | Instance variables | REPL + instance vars |
| Syntax errors | Returned to LLM | Auto-retried |

## Tips

- LLM response must be pure Python (no markdown blocks)
- Variables persist across turns in the REPL
- Tools return values directly to the REPL
- Use `emit(..., release=True)` to complete from REPL code, or `self.respond(value)` from tools
- Default tools are proxied (run in host process with `self` access); use `inject=True` for tools that should run directly in the subprocess
- Syntax errors are retried automatically (up to 3 times)
- Use context manager for cleanup: `with MyAgent() as agent:`
