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
        self.complete = True
        return price

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
5. **Loop continuation**: Steps 2-4 repeat until a tool sets `self.complete = True`
6. **Direct return**: When `self.complete = True`, the final tool's return value is passed directly back to the user without further LLM processing. This response is also appended to the conversation for future context.

This design gives developers precise control over the final response structure and format.

### Key Behaviors

- **Chat mode**: `agent.chat(message)` provides direct LLM responses without tool-calling loops
- **Context retention**: Agent instances maintain conversation state across multiple `run()` calls automatically
- **Custom control flow**: Override `run()` for preprocessing, custom loops, or entirely different interaction patterns
- **Inheritance support**: Create base classes that inherit from BaseAgent with custom `run()` methods for reuse across agent types
- **Typical pattern**: Use completion tools like `respond_to_user(message)` that set `self.complete = True` for formatted text responses

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
    self.complete = True
    return "Return created successfully"
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
Set `self.complete = True` when your agent has finished its task. This signals the run loop to return:

```python
@BaseAgent.tool
def resolve_issue(self, solution: str = "The solution provided"):
    """Mark an issue as resolved."""
    # Implementation
    self.complete = True
    return "Issue resolved: " + solution
```

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
        self.complete = True
        return f"Price for {quality} {self.current_item} set to ${price:.2f}"

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

## Summary

agentlib provides a flexible framework for building LLM-powered agents:

1. **Define your agent** by subclassing BaseAgent
2. **Add tools** using the @BaseAgent.tool decorator
3. **Manage state** with instance variables
4. **Control flow** by setting self.complete
5. **Handle errors** with specialized tools

This foundation allows you to create sophisticated agents with minimal code while handling the complexity of LLM interactions for you. For advanced use cases, agentlib also supports agent composition, where one agent can use another agent as a tool.
