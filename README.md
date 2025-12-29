# AgentLib

*A lightweight library for crafting and shipping LLM agents quickly, powered by Python signatures and Pydantic under the hood.*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
&nbsp;
![Python 3.9‒3.12](https://img.shields.io/badge/python-3.9‒3.12-blue)
&nbsp;

> **💡 Tip:** AgentLib works well with AI coding assistants like Claude Code. Add [`docs/guide.md`](docs/guide.md) to your context and start building. Include [`docs/mixins.md`](docs/mixins.md) for shell, REPL, MCP, or CLI features. For the code-first REPLAgent paradigm, see [`docs/replagent.md`](docs/replagent.md).

```python
from agentlib import BaseAgent

class FactorialAgent(BaseAgent):
    model = "google/gemini-2.5-flash"
    system = "You are a factorial calculation assistant. Use the tool to fulfill user requests."
    
    @BaseAgent.tool
    def factorial(self, number: int = "Number to calculate factorial for"):
        """Calculates a factorial."""
        def fact(n):
            return 1 if n == 0 else n * fact(n - 1)
        self.respond(fact(number))  # Marks conversation done and returns result

agent = FactorialAgent()
print(agent.run("What is the factorial of 20?"))
# Output: 2432902008176640000
```

<!--ts-->
## Table of Contents
- [Why AgentLib?](#why-agentlib)
- [Features](#features)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Supported LLM Providers](#supported-llm-providers)
- [Installation](#installation)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
<!--te-->

---

## Why AgentLib?

AgentLib was born from necessity during a business crisis—sudden tariff changes required an adaptive pricing system immediately. We needed to iterate at unprecedented speed, so we built from first principles: a minimal foundation that went straight into production.

The result:

* **Production-proven.** Powers our live dynamic-pricing, product-classification, and customer-support automations.  
* **Fast iteration.** New tools or model swaps are often a one-line change.  
* **Minimal deps.** Only `pydantic` (v1 & v2) and `python-dotenv`.  

Use AgentLib as a lightweight workhorse, a prototyping playground, or a study in minimalist agent design.

---

## Features

• **Python-native agent classes** – subclass `BaseAgent`, add methods, you're done.  
• **Decorator-based tool registry** – function signature & docstring ⇒ tool schema; Pydantic validation happens behind the scenes.  
• **Runtime tool mutation** – Dynamically adjust tool parameters, enums, or availability at any step, improving agent focus and performance by presenting only relevant options.  
• **Clean separation** – LLM orchestration lives in the core; your business logic lives in agents and tools.  
• **Conversation management** – tracks multi-turn context and system prompts for you.  
• **Provider-agnostic** – OpenAI, Anthropic, Google, X.AI, OpenRouter, or roll your own.  
• **Tool call emulation** – Enables both native and emulated tool calls with built-in validation and retry, bypassing inconsistent or poor constrained output performance.  
• **Attachment system** – Inject files and data into conversations as dynamic context.  
• **Multi-tool calls in a single LLM turn** – Execute multiple tools efficiently in one response.  
• **Automatic retry with exponential back-off** – Built-in resilience for API failures and rate limits.  
• **MCP integration** – Connect to Model Context Protocol servers for external tools and APIs.  
• **Shell & Python execution** – Give agents their own persistent bash shell or Python environment.  
• **Code-first agent paradigm** – REPLAgent lets the LLM write Python directly instead of JSON tool calls—ideal for code-heavy tasks.  
• **CLI builder** – Build interactive terminal assistants with markdown rendering and persistent history.  
• **Efficient file patching** – Context-based file editing with preview, approval workflow, and multi-file operations.  

---

## Quick Start

```bash
# 1. Install
pip install git+https://github.com/jacobsparts/agentlib.git

# 2. Set an API key (example: Anthropic Claude)
export ANTHROPIC_API_KEY=sk-...

# 3. Try the built-in code agent (Python REPL-based coding assistant)
code-agent

# Or run an example agent
python examples/todo_agent.py
```

Or copy–paste the snippet below into a new file:

```python
from agentlib import BaseAgent
import hashlib

class HashAgent(BaseAgent):
    model = "google/gemini-2.5-flash"
    system = "You are a hashing assistant. Use the tool to fulfill user requests."

    @BaseAgent.tool
    def sha256(self, text: str = "Text to hash"):
        """Return the SHA-256 hex digest of the input text."""
        self.respond(hashlib.sha256(text.encode()).hexdigest())

agent = HashAgent()
print(agent.run("What is the SHA-256 of hello world?"))
```

Expected output:

```
b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
google/gemini-2.5-flash: In=342, Out=54, Rsn=61, Cost=$0.000
```

### Build an Interactive CLI Assistant

```python
from agentlib import PythonMCPMixin, PythonToolResponseMixin
from agentlib.cli import CLIAgent

class DataExtractor(
    PythonMCPMixin,             # Python REPL with lightweight MCP client
    PythonToolResponseMixin,    # Direct code execution response
    CLIAgent,               # Interactive terminal interface
):
    model = "google/gemini-2.5-flash"
    system = """You are a data extraction specialist. You scrape websites, pull tables
from PDFs, and transform messy data into clean formats. You have browser automation
via puppeteer and Python with pandas, pdfplumber, beautifulsoup4, and openpyxl."""
    welcome_message = "[bold]Data Extractor[/bold]\nGive me a URL or file. I'll get you the data."
    repl_mcp_servers = [
        ('browser', 'npx -y @anthropic/mcp-server-puppeteer'),
    ]

if __name__ == "__main__":
    DataExtractor.main()
```

### Built-in Code Agent

The code agent inverts the usual agent paradigm: instead of an LLM calling tools via JSON, the model *lives inside a Python REPL*. Model output becomes REPL input; REPL output becomes the model's next prompt. The model writes Python directly—no tool schemas, no JSON marshalling.

```bash
code-agent
code-agent --model anthropic/claude-opus-4-5
```

This means:
- **Unrestricted Python** — stdlib, installed packages, subprocesses, network, filesystem
- **Tools on the fly** — model can define helper functions mid-conversation and reuse them
- **True collaboration** — drop into `/repl` and work alongside the agent in the same environment
- **Inspect everything** — model can read its own tool source code, introspect objects, experiment

Production-ready tools are included (glob, grep, read, edit, web_fetch, bash) for feature parity with leading coding agents. And since CodeAgent is just a Python class, it's easy to extend with custom tools, swap the model, modify the system prompt, or use it as a base for specialized agents in your own workflows.

```python
from agentlib.agents import CodeAgent

with CodeAgent() as agent:
    result = agent.run("Find all TODO comments in the codebase")
```

---

## How It Works

1. **Define tools** with ordinary Python functions.  
2. A metaclass decorator captures each function's signature & docstring, generating a JSON schema with Pydantic.  
3. At runtime the agent builds a prompt that exposes available tools to the LLM.  
4. The LLM selects a tool; AgentLib routes calls, validates inputs/outputs, and appends results to the conversation.  
5. The cycle repeats until a tool calls `self.respond()` or max turns are reached.  
6. An agent is typically *required* to make at least one tool call, until a tool calls `self.respond(value)`--that value is then sent directly to the caller, bypassing a final agent response. This differs from the usual user-agent-tool-agent-user flow. The loop is simple and customizable via the run method. Agents can be called directly without tools using the chat method.  The run method can be invoked multiple times, retaining context.  

---

## Supported LLM Providers

| Provider | Env var key        |
|----------|--------------------|
| OpenAI   | `OPENAI_API_KEY`   |
| Anthropic | `ANTHROPIC_API_KEY`   |
| Google   | `GOOGLE_API_KEY`   |
| X.AI     | `XAI_API_KEY`      |
| OpenRouter | `OPENROUTER_API_KEY` |

Add more chat completions compatible endpoints with `register_provider` and `register_model`.  See `examples/config.py` and `llm_registry.py` for details.

---

## Installation

```bash
pip install git+https://github.com/jacobsparts/agentlib.git
```

AgentLib supports Python 3.9+ on Linux.  Untested on macOS and Windows.

---

## FAQ

**Can I compose agents?**  
Yes—agents are normal Python classes, so you can instantiate or subclass them inside each other.

**Is Pydantic mandatory?**  
You don't need to import it directly; AgentLib uses it internally for validation generated from your function signatures.  However, you can use Pydantic models directly by passing them to the tool decorator, or you can pass a model generator function.

**What about concurrency?**  
AgentLib doesn't implement concurrency internally, but it's thread-safe, allowing you to safely use it in multi-threaded applications—which is exactly what we do in production. It also works well with gevent. We'll consider adding async/await support based on our own production needs or community interest.


---

## Contributing

Issues, feature requests, and pull requests are welcome.  

---

## License

AgentLib is released under the MIT License.  
See [LICENSE](LICENSE) for the full text.
