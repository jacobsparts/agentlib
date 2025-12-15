# AgentLib Quick Reference

## Basic Agent Structure

```python
from agentlib import BaseAgent

class MyAgent(BaseAgent):
    model = 'google/gemini-2.5-flash'  # or 'openai/gpt-5', etc.
    system = "You are a helpful assistant. Use tools to accomplish tasks."

    @BaseAgent.tool
    def my_tool(self, param: str = "Description"):
        """Tool description for the LLM."""
        result = do_something(param)
        self.respond(result)  # Ends run loop, returns result to caller
```

## Key Concepts

### Model Selection
- **Google**: `google/gemini-2.5-flash`, `google/gemini-2.5-pro`
- **OpenAI**: `openai/gpt-5.1`, `openai/gpt-5-mini`
- **Anthropic**: `anthropic/claude-sonnet-4-5`
- Set via env vars: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

### Tool Decorator
```python
@BaseAgent.tool
def tool_name(self,
    param1: str = "Description",
    param2: int = "Number to process",
    choice: ['option1', 'option2'] = "Pick one"
):
    """This docstring becomes the tool description."""
    return result
```

**Parameter types:**
- Standard types: `str`, `int`, `float`, `bool`
- Lists for enums: `['opt1', 'opt2']` → Literal type
- Optional: `Optional[str]`
- Dynamic: `lambda self: [self.available_options]`

### Conversation Flow

**Default flow** (`agent.run(message)`):
1. User message sent to LLM
2. LLM **must** call a tool (no direct responses)
3. Tool result added to conversation
4. Loop continues until a tool calls `self.respond(value)`
5. Value passed to `respond()` is returned to the caller

**Alternative** (`agent.chat(message)`):
- Direct LLM text response
- No tool-calling loop

### State Management

```python
class MyAgent(BaseAgent):
    def __init__(self, sku):
        self.sku = sku           # Instance variables persist
        self.data = {}

    @BaseAgent.tool
    def store_data(self, key: str = "Key", value: str = "Value"):
        """Store data in agent state."""
        self.data[key] = value
        return f"Stored {key}"
```

### Attachments

Inject named text content into conversations with automatic formatting and smart invalidation:

```python
# In custom run() method
def run(self, filename, max_turns=5):
    with open(filename) as f:
        self.attach(filename, f.read())
    self.usermsg(f"Analyze: {filename}")
    return self.run_loop(max_turns=max_turns)

# Or with structured data
self.attach("config.json", {"key": "value"})
self.usermsg("Process this config")

# In tools
@BaseAgent.tool
def read_file(self, path: str = "File path"):
    """Read a file."""
    content = open(path).read()
    self.attach(path, content)
    return f"Read {len(content)} bytes"

@BaseAgent.tool
def clear_file(self, path: str = "File path"):
    """Remove a file from context."""
    self.detach(path)
    return f"Cleared {path}"
```

**Features:**
- `usermsg()`/`toolmsg()` return `None` (side effect only)
- `self.attach(name, content)` to attach data to a response
- `self.detach(name)` to remove an attachment from context
- Auto-formatted as `-------- BEGIN name --------` / `-------- END name ----------`
- Dict/list values auto-serialized to JSON
- Re-attach with the same name to update content

### Completion Control

```python
@BaseAgent.tool
def respond_to_user(self, message: str = "Response to user"):
    """Send final response."""
    self.respond(message)  # Ends the run loop and returns value
```

### Custom Run Method

```python
def run(self, item_name, max_turns=5):
    # Preprocessing
    self.current_item = item_name
    context = self.load_context(item_name)

    # Build initial message
    msg = f"Process: {item_name}\nContext: {context}"

    # Call parent run
    return super().run(msg, max_turns=max_turns)
```

### Advanced Tool Patterns

**Pydantic model:**
```python
from pydantic import BaseModel, Field

class Decision(BaseModel):
    action: str = Field(..., description="Action to take")
    price: float = Field(..., description="Price value")

@BaseAgent.tool(model=Decision)
def submit(self, **payload):
    """Submit decision with structured data."""
    self.respond(payload)
```

**Dynamic model generator:**
```python
def get_submit_model(self):
    return create_model('SubmitModel',
        choice=(int, Field(..., description=f"Pick from: {self.choices}")))

@BaseAgent.tool(model=get_submit_model)
def submit(self, **payload):
    self.respond(payload)
```

## Multi-Agent Pattern

```python
class AnalystAgent(BaseAgent):
    model = 'google/gemini-2.5-flash'
    system = "Analyze data and provide insights."

    @BaseAgent.tool
    def respond_with_analysis(self, analysis: str = "Analysis results"):
        self.respond(analysis)

class MainAgent(BaseAgent):
    model = 'google/gemini-2.5-pro'
    system = "Make decisions based on analyst input."

    def __init__(self, sku):
        self.analyst = AnalystAgent()

    @BaseAgent.tool
    def consult_analyst(self, question: str = "Question for analyst"):
        """Get analysis from the analyst agent."""
        return self.analyst.run(question)

    @BaseAgent.tool
    def decide(self, decision: str = "Final decision"):
        self.respond(decision)
```

## Common Patterns

### Error handling:
```python
@BaseAgent.tool
def panic(self, reason: str = "Error description"):
    """Call only for unrecoverable errors."""
    raise Exception(f"Agent panic: {reason}")
```

### Message building:
```python
def run(self):
    self.attach("context.txt", self.load_context())
    self.usermsg("Process the context")
    return self.run_loop(max_turns=10)
```

### Manual conversation:
```python
agent = MyAgent()
agent.usermsg("User message")     # Add user message
response = agent.text()            # Get LLM response (text only)
result = agent.run_loop(max_turns=5)  # Run tool loop
```

## Tips

- `self.respond(value)` is required to exit the run loop and return a value
- Tools are **required** in `run()` mode (LLM cannot respond without calling a tool)
- Use `chat()` for simple Q&A without tools
- Instance variables persist across multiple `run()` calls
- Override `run()` for custom preprocessing or control flow
- Tools can call other agents for delegation patterns
