# AgentLib

*A lightweight library I use to craft and ship LLM agents quickly—Python signatures and Pydantic do the heavy lifting.*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
&nbsp;
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
&nbsp;

> **💡 Tip:** I pair AgentLib with AI coding assistants like Claude Code. Drop `docs/guide.md` into your context and start building. Mixins add shell, Python execution, MCP, and CLI features. 

```python
import sqlite3
from agentlib import BaseAgent

class DatabaseAgent(BaseAgent):
    model = "google/gemini-3.7-flash"
    system = """You answer questions about a SQLite database.

Schema:
  customers(id, name, region)
  orders(id, customer_id, total, status, created_at)

Use execute_query to gather the information needed to answer the user.
When you have enough information, answer clearly and directly. If the
requested information is unavailable, the request is ambiguous, or it
cannot be fulfilled from this database, explain that instead of guessing."""

    def __init__(self):
        self.db = sqlite3.connect("file:sales.db?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row

    @BaseAgent.tool
    def execute_query(self, sql: str = "The read-only SQLite query to execute"):
        """Executes SQL and returns the resulting rows."""
        if sql.lstrip().split(None, 1)[0].upper() not in {"SELECT", "WITH"}:
            raise ValueError("Only read-only queries are allowed")
        return [dict(row) for row in self.db.execute(sql)]

    @BaseAgent.tool
    def answer(self, response: str = "The answer to the user's request"):
        """Returns the final answer to the user."""
        self.respond(response)

agent = DatabaseAgent()
print(agent.run("Which five customers spent the most on paid orders this year?"))
```

<!--ts-->
## Table of Contents
- [Why AgentLib?](#why-agentlib)  
- [Features](#features)  
- [Quick Start](#quick-start)  
- [How It Works](#how-it-works)  
- [Supported LLM Providers](#supported-llm-providers)  
- [Installation](#installation)  
- [FAQ](#faq)  
- [Contributing](#contributing)  
- [Related Projects](#related-projects)  
- [License](#license)  
<!--te-->

---

## Why AgentLib?

I built AgentLib during a business crisis—sudden tariff changes, and I needed an adaptive pricing system *immediately*. There wasn't time for a framework tour. I started from first principles: a tiny Python-native core that went straight into production.

Years later, I still reach for it first:

* **Production-proven.** It powers my live dynamic-pricing, product-classification, and customer-support automations.  
* **Fast iteration.** New tools or model swaps are often a one-line change.  
* **Minimal deps.** Only `pydantic` (v1 & v2).  

I treat AgentLib as a lightweight workhorse, a prototyping playground, and a study in minimalist agent design. You're welcome to do the same.

---

## Features

• **Python-native agent classes** – subclass `BaseAgent`, add methods, you're done.  
• **Decorator-based tool registry** – function signature & docstring ⇒ tool schema; Pydantic validation happens behind the scenes.  
• **Runtime tool mutation** – Dynamically adjust tool parameters, enums, or availability at any step, so the model only sees the options that matter.  
• **Clean separation** – LLM orchestration lives in the core; your business logic lives in agents and tools.  
• **Conversation management** – tracks multi-turn context and system prompts for you.  
• **Provider-agnostic** – OpenAI, Anthropic, Google, X.AI, OpenRouter, or roll your own.  
• **Tool call emulation** – Native or emulated tool calls with built-in validation and retry, bypassing inconsistent or poor constrained output performance.  
• **Attachment system** – Inject files and data into conversations as dynamic context.  
• **Multi-tool calls in a single LLM turn** – Fire several tools in one response.  
• **Automatic retry with exponential back-off** – Built-in resilience for API failures and rate limits.  
• **MCP integration** – Optional mixin if you want Model Context Protocol servers as extra tools.  
• **Shell & Python execution** – Give agents their own persistent bash shell or Python environment.  
• **Code-first agent paradigm** – REPLAgent lets the LLM write Python directly instead of JSON tool calls—ideal for code-heavy tasks.  
• **CLI builder** – Build interactive terminal assistants with markdown rendering and persistent history.  
• **Efficient file patching** – Context-based file editing with preview, approval workflow, and multi-file operations.  

---

## Quick Start

Three commands and you're in:

```bash
# 1. Install
pip install git+https://github.com/jacobsparts/agentlib.git

# 2. Set an API key (example: Anthropic Claude)
export ANTHROPIC_API_KEY=sk-...

# 3. Run an example agent
python examples/todo_agent.py
```

Or drop this into a new file:

```python
from agentlib import BaseAgent
import hashlib

class HashAgent(BaseAgent):
    model = "google/gemini-3.7-flash"
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
google/gemini-3.7-flash: In=342, Out=54, Rsn=61, Cost=$0.000
```

### Build an Interactive CLI Assistant

```python
from agentlib import PythonToolResponseMixin
from agentlib.cli import CLIAgent

class DataExtractor(
    PythonToolResponseMixin,    # Direct code execution response
    CLIAgent,               # Interactive terminal interface
):
    model = "google/gemini-3.7-flash"
    system = """You are a data extraction specialist. You scrape websites, pull tables
from PDFs, and transform messy data into clean formats. You have Python with pandas,
pdfplumber, beautifulsoup4, and openpyxl."""
    welcome_message = "[bold]Data Extractor[/bold]\nGive me a URL or file. I'll get you the data."

if __name__ == "__main__":
    DataExtractor.main()
```

---

## How It Works

I kept the loop simple:

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

AgentLib supports Python 3.10+ on Linux.  Untested on macOS and Windows.

---

## FAQ

**Can I compose agents?**  
Yes—agents are normal Python classes, so you can instantiate or subclass them inside each other.

**Is Pydantic mandatory?**  
You don't need to import it directly; I use it internally for validation generated from your function signatures.  However, you can use Pydantic models directly by passing them to the tool decorator, or you can pass a model generator function.

**What about concurrency?**  
I use traditional concurrency internally—spawned processes for isolated execution environments (shell, REPL) and threading with select-based I/O for the MCP client. Public APIs are thread-safe, so you can safely call agents from multiple threads—which is exactly what I do in production. The select-based I/O is gevent-compatible when monkey-patched.

I never use the ambient `multiprocessing` start method for workers AgentLib owns; I explicitly use `spawn`. Do the same for application-owned workers that will create agents or call providers:

```python
import multiprocessing as mp

ctx = mp.get_context("spawn")
process = ctx.Process(target=run_agent)
process.start()
```

Do not use `fork` for agent workers. If AgentLib was imported before an external `fork`, creating provider-admission, shell, or REPL resources in that child fails fast with an actionable error rather than risking inherited locks, threads, queues, HTTP state, or SQLite locking failures.

---

## Contributing

Issues, feature requests, and pull requests are welcome.  

---

## Related Projects

Part of a family of developer tools I maintain for agentic coding and model gateways. [Code Agent](https://github.com/jacobsparts/code-agent) started life inside AgentLib—I split it into its own project as it grew, and it stands as a large-scale example of what you can build on this library:

- **[Code Agent](https://github.com/jacobsparts/code-agent)** — A Python REPL-native coding agent designed around lean context, persistent execution state, and infinite context via lossless turn coalescing.  
- **[codex-gateway](https://github.com/jacobsparts/codex-gateway)** — Pure-Python OpenAI Responses API-compatible gateway for Codex/ChatGPT OAuth accounts with quota management, account rotation, and automated resets.  
- **[cursor-gateway](https://github.com/jacobsparts/cursor-gateway)** — Pure-Python OpenAI-compatible Chat Completions gateway that wraps the Cursor Agent API with synthetic checkpoints to provide real native tool calling and cache-friendly session routing.  

---

## License

AgentLib is released under the MIT License.  
See [LICENSE](LICENSE) for the full text.
