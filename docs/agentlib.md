# Building AI Agents with agentlib: A Tutorial

## 1. Introduction to agentlib

agentlib is a framework for building AI agents. It provides a structured way to create agents that can maintain context, use tools, and manage interactions with LLM services.

### Key Features
- Built-in conversation management
- Tool registration system with simple decorators
- LLM integration with straightforward model selection
- Control flow management for multi-turn interactions

## 2. Getting Started

### Creating Your First Agent

Let's build a simple pricing agent that sets prices for grocery items:

```python
from agentlib import BaseAgent

class PricingAgent(BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are pricing agent. Set a reasonable price for the indicated item."

    @BaseAgent.tool
    def set_price(self,
        reason: str = "Explanation for the decision",
        price: float = "The price to set"
    ):
        """Use this tool to set a price."""
        print(f"Reason: {reason}")
        self.respond(price)

if __name__ == "__main__":
    import random
    items = ['Whole Wheat Bread', '1% Milk', 'Free Range Eggs (Dozen)']
    agent = PricingAgent()
    item = random.choice(items)
    print(f"Item: {item}")
    print(f"Price: {agent.run(item)}")
```

## 3. Core Components

### The Agent Class
Every agent is a subclass of `BaseAgent` with at least:
- A `model` attribute to specify which LLM to use (e.g., 'google/gemini-2.5-flash')
- A `system` prompt that defines the agent's role and behavior

### Tools
Tools are methods that your agent can call to accomplish tasks. Tools are defined using the `@BaseAgent.tool` decorator:

```python
@BaseAgent.tool
def get_data(self, query: str = "The search query"):
    """Retrieve information from a database."""
    # Implementation
    return result
```

The decorator automatically:
- Registers the method as an available tool
- Creates a schema based on type hints and default values
- Makes the tool available to the LLM during conversations

#### Tool Parameter Types
The `@BaseAgent.tool` decorator supports three methods for defining parameter types:

1. **Function Signature-based Schema Generation (Recommended)**
   
   This is the default method when you don't provide an explicit model to the decorator. The schema is inferred from the method's signature:

   ```python
   @BaseAgent.tool
   def submit_decision(self,
       decision: ['approve', 'reject'] = "The final decision.",
       reason: str = "Explanation for the decision"
   ):
       """Submit the final decision."""
       # Implementation
       return f"Decision submitted: {decision}"
   ```

   For each parameter, the type hint can be:
   - A standard Python type (`str`, `int`, `bool`, etc.)
   - A typing construct (`List[str]`, `Optional[float]`, etc.)
   - A list/tuple of options that will be converted to a `Literal` type
   - A function that returns a type (for dynamic type generation)

2. **Pydantic Model**
   For complex schemas, you can pass a Pydantic model to the decorator.

3. **Generator Function**
   For runtime flexibility, you can pass a function that returns a Pydantic model.   

### Conversation Flow

An agent using the default `run()` method follows this flow:

1. **User message**: The agent receives a user message via `agent.run(message)`
2. **LLM processing**: The message is processed by the LLM with available tools
3. **Tool execution**: The LLM is required to call one or more tools (no direct text responses)
4. **Tool result**: The tool's return value is added to the conversation
5. **Loop continuation**: Steps 2-4 repeat until a tool calls `self.respond(value)`
6. **Direct return**: When `self.respond(value)` is called, the value is passed directly back to the user without further LLM processing. This response is also appended to the conversation for future context.

This design gives developers precise control over the final response structure and format.

### Key Behaviors

- **Chat mode**: `agent.chat(message)` provides direct LLM responses without tool-calling loops
- **Context retention**: Agent instances maintain conversation state across multiple `run()` calls automatically
- **Custom control flow**: Override `run()` for preprocessing, custom loops, or entirely different interaction patterns
- **Inheritance support**: Create base classes that inherit from BaseAgent with custom `run()` methods for reuse across agent types
- **Typical pattern**: Use completion tools like `respond_to_user(message)` that call `self.respond(message)` for formatted text responses

## 4. Building Agents Step-by-Step

### Step 1: Define Your Agent Class

```python
class CustomerSupportAgent(BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful customer support agent. Help users with their questions."
```

### Step 2: Add Tools

```python
@BaseAgent.tool
def lookup_order(self, order_id: str = "The order number to look up"):
    """Look up information about a customer order."""
    # Implementation
    return {"status": "shipped", "estimated_delivery": "2023-06-15"}

@BaseAgent.tool
def create_return(self, 
                 order_id: str = "The order number",
                 reason: str = "Reason for return"):
    """Create a return request for an order."""
    # Implementation
    self.respond("Return created successfully")
```

### Step 3: Implement Run Logic (Optional)

The default `run` method works for most cases, but you can customize it:

```python
def run(self, item_name, max_turns=5):
    # Preprocess the input or set up initial state
    self.current_item_name = item_name 
    processed_msg = f"Please process item: {item_name}"
    return super().run(processed_msg, max_turns=max_turns)
```

## 5. Control Flow and State Management

### Managing Completion
Call `self.respond(value)` when your agent has finished its task. This signals the run loop to return the provided value:

```python
@BaseAgent.tool
def resolve_issue(self, solution: str = "The solution provided"):
    """Mark an issue as resolved."""
    # Implementation
    self.respond("Issue resolved: " + solution)
```

### Attachments in Tools
Tools can attach named content to their responses using `self.attach()`:

```python
@BaseAgent.tool
def read_file(self, path: str = "File path to read"):
    """Read a file and return its contents."""
    content = open(path).read()
    self.attach(path, content)
    return f"Read {len(content)} bytes from {path}"
```

Attachments are automatically formatted and included in the conversation context. Re-attach with the same name to update content, or use `self.detach(name)` to remove an attachment entirely. Note: attachments are for text content only, not binary data.

### Maintaining State
Store information between turns using instance variables:

```python
def __init__(self):
    self.user_info = {}
    
@BaseAgent.tool
def save_user_preference(self, preference: str = "The user preference"):
    """Save a user preference."""
    self.user_info["preference"] = preference
    return f"Saved preference: {preference}"
```

## 6. Error Handling

Create specialized tools for handling errors:

```python
@BaseAgent.tool
def panic(self, reason: str = "A brief explanation of the critical error"):
    """
    Call this function ONLY if you encounter a critical, unrecoverable error
    or a situation that makes fulfilling the request impossible.
    """
    raise Exception(f"🚨 Agent called panic: {reason}")
```

## 7. Complete Example: Grocery Pricing Agent

```python
from agentlib import BaseAgent

class PricingAgent(BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = """You are a pricing agent for a grocery store.
    Set reasonable prices for grocery items based on market rates.
    Consider factors like item quality (e.g., standard, premium, organic), organic status, and market trends."""

    def __init__(self):
        self.pricing_history = {}

    @BaseAgent.tool
    def research_market_price(self, item: str = "The grocery item"):
        """Research the current market price range for an item."""
        # Simulated market research
        import random
        min_price = random.uniform(1.0, 10.0)
        max_price = min_price * 1.3
        return f"Market price range for {item}: ${min_price:.2f} - ${max_price:.2f}"

    @BaseAgent.tool
    def set_price(self,
                 reason: str = "Explanation for the decision",
                 price: float = "The price to set",
                 quality: ['standard', 'premium', 'organic'] = "The quality of the item"):
        """Use this tool to set a final price for the item."""
        print(f"Pricing rationale for {quality} quality: {reason}")
        self.pricing_history[self.current_item] = {'price': price, 'quality': quality}
        self.respond(f"Price for {quality} {self.current_item} set to ${price:.2f}")

    def run(self, item, max_turns=5):
        self.current_item = item # Set instance variable for tools to use
        # Construct the initial message for the LLM
        initial_message = f"Set a price for: {item}"
        return super().run(initial_message, max_turns=max_turns)

if __name__ == "__main__":
    items = ['Organic Apples (1 lb)', 'Sourdough Bread', 'Free Range Eggs (Dozen)']
    agent = PricingAgent()
    
    for item in items:
        print(f"\nPricing: {item}")
        result = agent.run(item)
        print(f"Result: {result}")
```

## 8. Code-First Agents with REPLAgent

REPLAgent is a fundamentally different paradigm from tool-calling agents. Instead of the LLM selecting tools from JSON schemas, the LLM writes Python code directly and executes it in a persistent REPL environment.

### Why REPLAgent?

Tool-calling agents work well for structured operations, but some tasks are naturally code-oriented:
- Data analysis and transformation
- File manipulation and scripting
- Complex computations
- Tasks where the LLM needs to see output before deciding next steps

REPLAgent excels here because:
- **Natural for code tasks**: The LLM writes Python instead of describing operations through tool schemas
- **Persistent state**: Variables, imports, and definitions survive across turns
- **Exploratory workflow**: The LLM can run code, see output, and iterate
- **Syntax error recovery**: Pure syntax errors are retried without polluting conversation history

### Basic Usage

```python
from agentlib import REPLAgent

class DataAgent(REPLAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a data analysis assistant."

    @REPLAgent.tool
    def read_csv(self, path: str = "Path to CSV file"):
        """Read a CSV file and return its contents."""
        import pandas as pd
        return pd.read_csv(path).to_dict()

result = DataAgent().run("Analyze sales.csv and tell me the top 3 products by revenue")
```

The key difference: the LLM's response IS Python code, not markdown or tool calls. The code runs directly in the REPL.

### How It Works

1. The LLM receives a system prompt describing the REPL environment
2. The LLM writes Python code as its response (no markdown, no preamble)
3. Code executes statement-by-statement in an isolated subprocess
4. Output streams back to the LLM as REPL feedback
5. Loop continues until `submit(result)` is called or max turns reached

### Built-in Functions

Every REPLAgent has these functions available in the REPL:

| Function | Description |
|----------|-------------|
| `submit(result)` | Submit the final answer and end the task |
| `respond(text)` | Same as submit, enabled when `interactive = True` |
| `help(func)` | Get parameter descriptions for any tool |

### Defining Tools

Tools are defined the same way as BaseAgent, but they become callable Python functions in the REPL:

```python
class MyAgent(REPLAgent):
    system = "You are a helpful assistant."

    @REPLAgent.tool
    def search(self, query: str = "Search query"):
        """Search for information."""
        return do_search(query)

    @REPLAgent.tool
    def save_file(self, path: str = "File path", content: str = "Content to write"):
        """Save content to a file."""
        Path(path).write_text(content)
        return f"Saved {len(content)} bytes to {path}"
```

The LLM can then use these naturally in code:

```python
# LLM writes:
results = search("python tutorials")
for r in results[:3]:
    print(r['title'])
summary = "\n".join(r['title'] for r in results[:3])
save_file("top_results.txt", summary)
submit(f"Found {len(results)} results, saved top 3 to top_results.txt")
```

### Interactive Mode

For CLI or chat-style agents, set `interactive = True` to enable `respond()` as a more natural alternative to `submit()`:

```python
class ChatAgent(REPLAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."
    interactive = True  # Enables respond() function

# Now the LLM can use respond() instead of submit():
# respond("Here's what I found...")
```

### Combining with Mixins

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

MCP tools from connected servers become callable functions in the REPL alongside your defined tools.

### Hooks for Customization

REPLAgent provides hooks for customizing output handling:

```python
class CustomAgent(REPLAgent):
    def on_repl_execute(self, code):
        """Called at the start of each turn."""
        print("Starting execution...")

    def on_repl_chunk(self, chunk: str):
        """Called for each chunk of output."""
        pass  # Real-time streaming hook

    def on_repl_output(self, output: str):
        """Called after execution completes with full output."""
        print(f"Output: {output[:100]}...")

    def process_repl_output(self, output: str) -> str:
        """Process/truncate output before sending to LLM."""
        if len(output) > 10000:
            return output[:5000] + "\n... (truncated) ..."
        return output
```

### When to Use REPLAgent vs BaseAgent

| Use REPLAgent when... | Use BaseAgent when... |
|-----------------------|-----------------------|
| Task is code-heavy | Task is structured operations |
| LLM needs to iterate on output | Operations are well-defined |
| Exploratory workflow | Predictable multi-step workflows |
| Computing results | Calling APIs or services |

### Complete Example: Data Analysis Agent

```python
from agentlib import REPLAgent

class AnalysisAgent(REPLAgent):
    model = 'google/gemini-2.5-flash'
    system = """You are a data analysis assistant.
    Write Python code to analyze data and answer questions.
    Use pandas for data manipulation. Submit your final answer with submit()."""

    @REPLAgent.tool
    def load_data(self, path: str = "Path to data file"):
        """Load data from a file (CSV, JSON, or Excel)."""
        import pandas as pd
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.json'):
            return pd.read_json(path)
        elif path.endswith('.xlsx'):
            return pd.read_excel(path)
        raise ValueError(f"Unsupported format: {path}")

if __name__ == "__main__":
    agent = AnalysisAgent()
    result = agent.run("""
        Load sales_2024.csv and answer:
        1. What's the total revenue?
        2. Which product category has the highest sales?
        3. What's the month-over-month growth trend?
    """)
    print(result)
```

The LLM might produce multiple turns of Python code:

```python
# Turn 1: Load and explore
df = load_data("sales_2024.csv")
df.head()
df.info()

# Turn 2: Answer questions
total_revenue = df['revenue'].sum()
top_category = df.groupby('category')['sales'].sum().idxmax()
monthly = df.groupby('month')['revenue'].sum()
growth = monthly.pct_change().mean() * 100
submit(f"""Analysis Results:
1. Total Revenue: ${total_revenue:,.2f}
2. Top Category: {top_category}
3. Average MoM Growth: {growth:.1f}%""")
```

## 9. MCP Integration with MCPMixin

`MCPMixin` adds Model Context Protocol (MCP) support to any agent via mixin composition. MCP servers expose tools that your agent can call, allowing integration with external systems, APIs, and services.

### Basic Usage

Define MCP servers as a class attribute:

```python
from agentlib import BaseAgent, MCPMixin

class BrowserAgent(MCPMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a browser automation assistant."
    mcp_servers = [
        ('browser', '/path/to/browser-mcp-server'),
        ('api', 'http://localhost:3000/sse'),
    ]

    @BaseAgent.tool
    def done(self, response: str = "Your response"):
        """Send final response to user."""
        self.respond(response)

with BrowserAgent() as agent:
    result = agent.run("Navigate to example.com and get the page title")
```

### Server Configuration

Each entry in `mcp_servers` is a tuple of `(name, server)` or `(name, server, options)`:

```python
mcp_servers = [
    # Stdio transport (command string, split on spaces)
    ('fs', 'npx -y @mcp/server-filesystem /tmp'),

    # SSE transport (detected by http:// or https://)
    ('api', 'http://localhost:3000/sse'),

    # With options
    ('db', 'python db_server.py', {'timeout': 60.0}),
    ('auth_api', 'https://api.example.com/mcp', {'headers': {'Authorization': 'Bearer xxx'}}),
]
```

**Transport detection:**
- URLs starting with `http://` or `https://` use SSE transport
- Everything else uses stdio transport (command is split on spaces)

**Common options:**
- Stdio: `timeout`, `forward_stderr` (default: False), `env`
- SSE: `timeout`, `headers`

**Optional tool filtering:**
- `include`: List of tool names to expose (whitelist)
- `exclude`: List of tool names to hide (blacklist)

```python
mcp_servers = [
    # Optional: expose only specific tools
    ('browser', '/path/to/server', {'include': ['navigate', 'screenshot', 'get_title']}),

    # Optional: expose all except certain tools
    ('fs', 'npx -y @mcp/server-filesystem /tmp', {'exclude': ['write_file', 'delete_file']}),
]
```

Use `include` for a whitelist (only these tools) or `exclude` for a blacklist (all except these). They are mutually exclusive.

### How Tools Are Registered

When an MCP server is connected, all its tools are automatically registered with a prefix:

```python
# If mcp_servers = [('browser', '/path/to/server')]
# And the server exposes tools: navigate, screenshot, click

# Your agent gets tools:
#   browser_navigate
#   browser_screenshot
#   browser_click
```

The prefix prevents name collisions when using multiple MCP servers.

### MCP Server Instructions

MCP servers can provide instructions that guide how they should be used. These are automatically appended to your system prompt:

```python
class MyAgent(MCPMixin, BaseAgent):
    system = "You are a helpful assistant."
    mcp_servers = [('browser', '/path/to/server')]

# The actual system prompt becomes:
# "You are a helpful assistant.
#
# MCP SERVER INSTRUCTIONS:
# === browser ===
# [Instructions from the browser MCP server]"
```

### Dynamic Connection

You can also connect MCP servers at runtime:

```python
class MyAgent(MCPMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)

agent = MyAgent()
agent.connect_mcp('browser', '/path/to/browser-server')
agent.connect_mcp('api', 'http://localhost:3000/sse', {'headers': {'Auth': 'Bearer xxx'}})

result = agent.run("Do something with the browser")

agent.disconnect_mcp('browser')  # Remove a specific server
agent.close()  # Clean up all connections
```

### Lifecycle Management

Always close MCP connections when done:

```python
# Option 1: Context manager (recommended)
with MyAgent() as agent:
    result = agent.run("Do something")

# Option 2: Explicit close
agent = MyAgent()
try:
    result = agent.run("Do something")
finally:
    agent.close()
```

### Notes

- Tools are cached when servers connect. If an MCP server dynamically adds/removes tools, disconnect and reconnect to refresh.
- MCP errors are returned as strings (e.g., `"[MCP Error] Connection refused"`) so the LLM can handle them gracefully.
- Server stderr is suppressed by default. Pass `{'forward_stderr': True}` in options to see server logs.

## 10. Shell Execution with SubShellMixin

`SubShellMixin` gives your agent access to a persistent bash shell. Commands run in an isolated subprocess with environment variables and working directory preserved across calls.

### Basic Usage

```python
from agentlib import BaseAgent, SubShellMixin

class DevAgent(SubShellMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a development assistant with shell access."

    @BaseAgent.tool
    def done(self, response: str = "Your response"):
        self.respond(response)

with DevAgent() as agent:
    result = agent.run("Find all Python files and count total lines of code")
```

### Tools Provided to the Agent

The mixin automatically registers three tools:

| Tool | Parameters | Description |
|------|------------|-------------|
| `shell_execute` | `command`, `timeout=30` | Execute a bash command |
| `shell_read` | `timeout=30` | Continue reading output from a running command |
| `shell_interrupt` | (none) | Send SIGINT to stop a running command |

### Configuration

```python
class MyAgent(SubShellMixin, BaseAgent):
    shell_echo = False     # Prefix output with "$ command" (default: False)
    shell_timeout = 30.0   # Default timeout in seconds (default: 30.0)
```

### How It Works

1. **Persistent State**: Environment variables set with `export` and directory changes with `cd` persist across tool calls.
2. **Streaming Output**: Output streams in real-time. If a command takes longer than the timeout, partial output is returned with `[still running]\n` appended.
3. **Interrupt Support**: Long-running commands can be stopped with `shell_interrupt()`.

### Handling Long-Running Commands

When output ends with `[still running]\n`, the command hasn't finished:

```python
# The agent might receive:
# "Processing file 1...\nProcessing file 2...\n[still running]\n"

# The agent should then call shell_read() to get more output,
# or shell_interrupt() to stop the command.
```

### Direct API Access

You can also use the shell directly without going through the LLM:

```python
agent = DevAgent()
output = agent.shell_execute("ls -la")
print(output)
agent.close()
```

Or use the underlying `SubShell` class directly:

```python
from agentlib import SubShell

with SubShell(echo=True) as shell:
    print(shell.execute("echo hello"))
    shell.execute("export FOO=bar")
    print(shell.execute("echo $FOO"))  # Prints: bar
```

## 11. Python Execution with SubREPLMixin

`SubREPLMixin` gives your agent access to a persistent Python REPL. Variables, imports, and function definitions persist across calls.

### Basic Usage

```python
from agentlib import BaseAgent, SubREPLMixin

class DataAgent(SubREPLMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a data analysis assistant with Python execution."

    @BaseAgent.tool
    def done(self, response: str = "Your response"):
        self.respond(response)

with DataAgent() as agent:
    result = agent.run("Calculate the mean and standard deviation of [1, 2, 3, 4, 5]")
```

### Tools Provided to the Agent

| Tool | Parameters | Description |
|------|------------|-------------|
| `python_execute` | `code`, `timeout=30` | Execute Python code |
| `python_read` | `timeout=30` | Continue reading output from running code |
| `python_interrupt` | (none) | Send SIGINT to stop running code |

### Configuration

```python
class MyAgent(SubREPLMixin, BaseAgent):
    repl_echo = False      # Prefix output with ">>> code" (default: False)
    repl_timeout = 30.0    # Default timeout in seconds (default: 30.0)
```

### How It Works

1. **Persistent State**: Variables, imports, and function definitions persist across calls.
2. **Isolated Process**: Code runs in a separate process, protecting the main application.
3. **Streaming Output**: Output streams in real-time with `[still running]\n` for incomplete execution.
4. **Clean Tracebacks**: Error tracebacks are filtered to show only relevant frames.

### Example Session

The agent can build up state across multiple calls:

```python
# Call 1: Import libraries
python_execute("import pandas as pd\nimport numpy as np")

# Call 2: Create data
python_execute("df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})")

# Call 3: Analyze (variables persist)
python_execute("print(df.describe())")
```

### Direct API Access

```python
agent = DataAgent()
output = agent.python_execute("print(sum(range(100)))")
print(output)  # 4950
agent.close()
```

Or use the underlying `SubREPL` class:

```python
from agentlib import SubREPL

with SubREPL(echo=True) as repl:
    repl.execute("x = 42")
    repl.execute("def double(n): return n * 2")
    print(repl.execute("print(double(x))"))  # 84
```

### SubREPLResponseMixin Variant

For agents that compute results and return them directly, use `SubREPLResponseMixin` instead. It adds a fourth tool:

| Tool | Parameters | Description |
|------|------------|-------------|
| `python_execute_response` | `code`, `preamble=""`, `postamble=""` | Execute code and return output as final response |

```python
from agentlib import BaseAgent, SubREPLResponseMixin

class CalcAgent(SubREPLResponseMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a calculator. Compute results and return them directly."

with CalcAgent() as agent:
    # Agent can use python_execute_response to compute and respond in one step
    result = agent.run("What is the sum of squares from 1 to 100?")
```

The `preamble` and `postamble` parameters let the agent wrap the code output with explanatory text.

**Error handling:** If the script throws an exception or times out, the error is returned to the agent (not the user) so it can fix the code and retry.

## 12. Lightweight MCP with REPLMCPMixin

`REPLMCPMixin` provides an alternative to `MCPMixin` that uses significantly fewer tokens. Instead of exposing each MCP tool as an agent function (which adds to the system prompt), it pre-instantiates MCP clients in the Python REPL. The agent calls MCP tools by writing Python code.

### When to Use

- **MCPMixin**: Best when you want the LLM to see MCP tools as native functions. Simpler for the agent but uses more tokens.
- **REPLMCPMixin**: Best for token efficiency. The agent writes code to call MCP tools, which works well for agents that are already code-oriented.

### Basic Usage

```python
from agentlib import BaseAgent, REPLMCPMixin

class MyAgent(REPLMCPMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."
    repl_mcp_servers = [
        ('fs', '/usr/bin/mcp-server-filesystem /tmp'),
        ('api', 'http://localhost:3000/sse'),
    ]

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        """Send response to user."""
        self.respond(response)

with MyAgent() as agent:
    result = agent.run("List the files in /tmp")
```

### How It Works

1. MCP clients are created as variables in the REPL (e.g., `fs`, `api`)
2. Tool documentation is added to the system prompt (but not as individual functions)
3. The agent uses `python_execute` to call MCP tools:

```python
# Agent writes code like:
result = fs.call_tool('list_directory', {'path': '/tmp'})
print(result)
```

### Server Configuration

Same format as `MCPMixin`:

```python
repl_mcp_servers = [
    ('name', 'command or url'),
    ('name', 'command or url', {'timeout': 60.0}),
]
```

### Tool Enumeration

By default, tools are enumerated at setup and added to the system prompt. Disable this for faster startup when you have many MCP servers:

```python
class MyAgent(REPLMCPMixin, BaseAgent):
    repl_mcp_servers = [...]
    repl_mcp_enumerate_tools = False  # Skip tool enumeration globally
```

Per-server override:
```python
repl_mcp_servers = [
    ('fs', '/path/to/server'),                          # Uses global default
    ('api', 'http://localhost/sse', {'enumerate_tools': False}),  # Skip for this server
]
```

When enumeration is disabled, the agent can still discover tools at runtime via `client.list_tools()` in the REPL.

### With python_execute_response

For agents that compute results and return them directly, combine with `SubREPLResponseMixin`:

```python
from agentlib import BaseAgent, REPLMCPMixin, SubREPLResponseMixin

class DataAgent(REPLMCPMixin, SubREPLResponseMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a data assistant. Use MCP servers via Python and return results."
    repl_mcp_servers = [
        ('db', 'python db_server.py'),
    ]

with DataAgent() as agent:
    # Agent can call MCP tools and format results in one python_execute_response call
    result = agent.run("Get all users from the database and format as a table")
```

Python's MRO handles the diamond inheritance correctly—`SubREPLMixin` appears only once.

### MCP Client API in REPL

The agent has access to these methods on each MCP client:

```python
# List available tools
tools = client.list_tools()

# Call a tool
result = client.call_tool('tool_name', {'arg': 'value'})
# result = {'content': [...], 'isError': False}

# Access content
for item in result['content']:
    if item['type'] == 'text':
        print(item['text'])
```

### System Prompt Additions

`REPLMCPMixin` adds to the system prompt:
- List of available MCP clients and their servers
- Usage documentation for calling tools
- Tool names and descriptions (compact format)
- Server instructions (if provided by the server)

## 13. Combining Multiple Mixins

Mixins can be combined to create agents with multiple capabilities:

```python
from agentlib import BaseAgent, MCPMixin, SubShellMixin, SubREPLMixin

class PowerAgent(SubREPLMixin, SubShellMixin, MCPMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You have shell, Python, and MCP server access."
    mcp_servers = [
        ('fs', 'npx -y @mcp/server-filesystem /tmp'),
    ]

    @BaseAgent.tool
    def done(self, response: str = "Response"):
        self.respond(response)

with PowerAgent() as agent:
    result = agent.run("List files in /tmp, then analyze them with Python")
```

### Mixin Order

**Important:** List mixins before `BaseAgent` in the class definition. The order determines method resolution:

```python
# Correct: mixins before BaseAgent
class MyAgent(SubREPLMixin, SubShellMixin, MCPMixin, BaseAgent):
    ...

# Wrong: BaseAgent should be last
class MyAgent(BaseAgent, SubShellMixin):  # Don't do this
    ...
```

### Available Tools

When combining mixins, your agent gets all tools from each:

| Mixin | Tools |
|-------|-------|
| `SubShellMixin` | `shell_execute`, `shell_read`, `shell_interrupt` |
| `SubREPLMixin` | `python_execute`, `python_read`, `python_interrupt` |
| `SubREPLResponseMixin` | Above + `python_execute_response` |
| `MCPMixin` | `{server}_{tool}` for each MCP server tool |
| `REPLMCPMixin` | REPL tools + MCP clients in REPL (no per-tool functions) |
| `CLIMixin` | CLI REPL loop, console output, history |

## 14. Building Interactive CLI Assistants

The `agentlib.cli` module provides composable components for building terminal-based interactive assistants with minimal code. It includes terminal rendering utilities (markdown, syntax highlighting, panels) and a `CLIMixin` that adds a full CLI REPL loop to any agent.

### Quick Start with CLIAgent

`CLIAgent` is a pre-composed class that combines `CLIMixin` and `BaseAgent`:

```python
from agentlib.cli import CLIAgent

class MyAssistant(CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant."
    welcome_message = "[bold]My Assistant[/bold]\nHow can I help?"

if __name__ == "__main__":
    MyAssistant.main()
```

This gives you:
- Markdown rendering with syntax highlighting
- Panels for code and output display
- SQLite-backed readline history
- Multiline input support (Alt+Enter for newlines)
- Graceful Ctrl+C/D handling

Add mixins for additional capabilities:

```python
from agentlib import SubREPLResponseMixin
from agentlib.cli import CLIAgent

class CodeAssistant(SubREPLResponseMixin, CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a Python assistant."
    welcome_message = "[bold]Code Helper[/bold]\nI can run Python code."
```

### Configuration Options

```python
class MyAssistant(CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are helpful."

    # CLI-specific options
    welcome_message = "[bold]Welcome![/bold]\nHow can I help?"  # Supports markup
    cli_prompt = ">>> "                    # Input prompt (default: "> ")
    history_db = "~/.myapp_history.db"     # History file (default: ~/.agentlib_cli_history.db)
    max_turns = 20                         # Max agent iterations per message
    thinking_message = "Processing..."     # Status while agent works
```

### Welcome Message Markup

The welcome message supports rich-style markup tags:

```python
welcome_message = """[bold]My Assistant[/bold]
[dim]Version 1.0[/dim]

[cyan]Features:[/cyan]
- Answer questions
- Help with tasks
"""
```

Available tags: `[bold]`, `[dim]`, `[italic]`, `[underline]`, `[strike]`, `[red]`, `[green]`, `[yellow]`, `[blue]`, `[magenta]`, `[cyan]`, `[white]`, `[gray]`. Close with `[/tag]` or `[/]`.

### Customizing Tool Display with Hooks

Override these methods to customize how tool calls and results are displayed:

```python
class VerboseAssistant(CLIAgent):
    def on_tool_call(self, name, args):
        """Called before each tool executes."""
        self.console.clear_line()
        self.console.print(f"[yellow]Calling: {name}[/yellow]")

        if name == 'python_execute':
            self.console.panel(args.get('code', ''), title="Code", border_style="blue")

    def on_tool_result(self, name, result):
        """Called after each tool returns."""
        if result:
            self.console.panel(str(result)[:1000], title="Output", border_style="green")

    def format_response(self, response):
        """Format the final response before display."""
        # Default renders markdown; override for custom formatting
        return response  # Return plain text instead
```

### Adding MCP Servers

Combine with `REPLMCPMixin` to add MCP server access:

```python
from agentlib import REPLMCPMixin
from agentlib.cli import CLIAgent

class MCPAssistant(REPLMCPMixin, CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are helpful."
    repl_mcp_servers = [
        ('fs', '/path/to/mcp-filesystem-server /tmp'),
        ('api', 'http://localhost:3000/sse'),
    ]

MCPAssistant.main()
```

### Using CLIMixin Directly

For more control, use `CLIMixin` with your own agent composition:

```python
from agentlib import BaseAgent, SubREPLResponseMixin
from agentlib.cli import CLIMixin

class CustomAgent(CLIMixin, SubREPLResponseMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are helpful."
    welcome_message = "Hello!"

    @BaseAgent.tool
    def custom_tool(self, query: str = "User query"):
        """A custom tool."""
        return f"Processed: {query}"

with CustomAgent() as agent:
    agent.cli_run()
```

### Console Utilities

The `Console` class is available for custom output:

```python
from agentlib.cli import Console, Panel, Markdown

console = Console()

# Print with markup
console.print("[bold]Hello[/bold] [cyan]world[/cyan]")

# Print a panel
console.panel("Content here", title="My Panel", border_style="green")

# Print markdown
console.markdown("# Heading\nSome **bold** text")

# Status message (no newline)
console.status("Loading...")
console.clear_line()  # Clear status
```

### Panel Border Styles

Panels support these border colors: `cyan`, `blue`, `green`, `red`, `magenta`, `yellow`, `gray`, `white`.

```python
from agentlib.cli import Panel

# Full-width panel
p = Panel("Content", title="Title", border_style="blue")
print(p.render())

# Fit to content width
p = Panel.fit("Short content", border_style="green")
print(p.render())
```

### Terminal Utilities

Additional utilities available from `agentlib.cli`:

```python
from agentlib.cli import (
    render_markdown,      # Convert markdown to ANSI
    highlight_python,     # Syntax highlight Python code
    parse_markup,         # Convert [bold] tags to ANSI
    strip_ansi,          # Remove ANSI codes from text
    get_terminal_width,  # Get terminal width
)

# Render markdown to terminal
print(render_markdown("# Hello\n**Bold** and *italic*"))

# Highlight Python code
print(highlight_python("def hello():\n    print('world')"))
```

## 15. File Patching with FilePatchMixin

`FilePatchMixin` gives your agent the ability to add, update, and delete files using a context-based patch format. This is more efficient than rewriting entire files and provides clear visibility into changes.

### Basic Usage

```python
from agentlib import BaseAgent, FilePatchMixin

class CodeAssistant(FilePatchMixin, BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a coding assistant. Use apply_patch to modify files."

    @BaseAgent.tool
    def done(self, response: str = "Your response"):
        """Send final response to user."""
        self.respond(response)

with CodeAssistant() as agent:
    result = agent.run("Add a hello() function to main.py")
```

### Configuration

```python
class MyAgent(FilePatchMixin, BaseAgent):
    patch_preview = None  # True, False, or None (default)
```

| Value | Behavior |
|-------|----------|
| `True` | Always require user approval before applying patches |
| `False` | Auto-apply patches without preview/approval |
| `None` | Agent decides per-call (default: preview=True) |

### The Patch Format

Patches are wrapped in begin/end markers with file operations inside:

```
*** Begin Patch
*** Add File: path/to/new_file.py
+def hello():
+    print("Hello, world!")
*** Update File: path/to/existing.py
@@ def main():
 def main():
-    print("old")
+    print("new")
*** Delete File: path/to/remove.py
*** End Patch
```

**File Operations:**
- `*** Add File: <path>` - Create a new file. All content lines must start with `+`.
- `*** Update File: <path>` - Modify an existing file using hunks.
- `*** Delete File: <path>` - Remove an existing file.

**Hunk Format (for updates):**
- Lines starting with ` ` (space) are context (unchanged)
- Lines starting with `-` are removed
- Lines starting with `+` are added
- Use `@@` with optional context (e.g., function name) to locate the change
- Use `*** End of File` to anchor changes at the end of a file

### Tool Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `patch` | str | Patch text in the format above |
| `preview` | bool | Request preview/approval (only when `patch_preview=None`) |
| `preamble` | str | Optional text to display before the preview |
| `postamble` | str | Optional text to display after the preview |

### Preview and Approval Flow

When preview is enabled:
1. The patch is validated and a unified diff preview is generated
2. `_prompt_patch_approval()` is called to get user approval
3. If approved, the patch is applied; otherwise, rejection message is returned

**Default behavior:** Auto-approve (for non-interactive use)

**With CLIMixin:** Interactive terminal prompt with options:
- `[Y]es` - Apply the patch
- `[N]o` - Reject (with optional comments)
- `[A]lways` - Apply and disable future previews

### Combining with CLIMixin

```python
from agentlib import FilePatchMixin
from agentlib.cli import CLIAgent

class CodeEditor(FilePatchMixin, CLIAgent):
    model = 'google/gemini-2.5-flash'
    system = "You are a code editor. Use apply_patch to modify files."
    welcome_message = "[bold]Code Editor[/bold]\nI can edit your files."

if __name__ == "__main__":
    CodeEditor.main()
```

When combined with `CLIMixin`, the approval prompt is automatically enhanced with an interactive UI.

### Custom Approval Logic

Override `_prompt_patch_approval()` for custom approval workflows:

```python
class MyAgent(FilePatchMixin, BaseAgent):
    def _prompt_patch_approval(self, preview_text, preamble, postamble):
        """
        Custom approval logic.

        Returns:
            Tuple of (approved: bool, comments: str, disable_future_preview: bool)
        """
        # Example: auto-approve small patches, require approval for large ones
        if preview_text.count('\n') < 20:
            return True, "", False
        # ... custom approval UI ...
        return True, "", False
```

### Path Resolution

Paths in patches are relative to the base path, which is determined by:
1. `PWD` environment variable (if set)
2. Directory containing the main script
3. Current working directory (fallback)

### Notes

- The patch format is designed for context-based matching, making it resilient to line number changes
- Unicode punctuation (smart quotes, em-dashes) is normalized during matching
- Whitespace-insensitive matching is attempted if exact matching fails
- Patches can span multiple files in a single operation

## Summary

agentlib provides a flexible framework for building LLM-powered agents:

1. **Define your agent** by subclassing `BaseAgent` or `REPLAgent`
2. **Add tools** using the `@BaseAgent.tool` or `@REPLAgent.tool` decorator
3. **Manage state** with instance variables
4. **Control flow** with `self.respond()` or `submit()`
5. **Handle errors** with specialized tools
6. **Choose your paradigm**:
   - `BaseAgent` for tool-calling workflows (JSON schemas, structured operations)
   - `REPLAgent` for code-first workflows (LLM writes Python directly)
7. **Add capabilities** with mixins:
   - `MCPMixin` for MCP server integration (tools as functions)
   - `REPLMCPMixin` for lightweight MCP via code (more token-efficient)
   - `SubShellMixin` for bash shell execution
   - `SubREPLMixin` / `SubREPLResponseMixin` for Python REPL execution
   - `FilePatchMixin` for efficient file editing with preview/approval
   - `CLIMixin` / `CLIAgent` for interactive terminal assistants

This foundation allows you to create sophisticated agents with minimal code while handling the complexity of LLM interactions for you. For advanced use cases, agentlib also supports agent composition, where one agent can use another agent as a tool.
