# AgentLib

*A lightweight library for crafting and shipping LLM agents quickly, powered by Python signatures and Pydantic under the hood.*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
&nbsp;
![Python 3.9‒3.12](https://img.shields.io/badge/python-3.9‒3.12-blue)
&nbsp;

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
- [License](#license)
<!--te-->

---

## Why AgentLib?

AgentLib was forged in a real business fire-drill: sudden tariff changes demanded an adaptive pricing system immediately.
To iterate at the speed required, and to ensure the power, flexibility, and simplicity we needed, we built from first principles. This clean-slate approach resulted in a disciplined, minimal codebase that went straight into production.
The result:

* **Production-proven.** Powers our live dynamic-pricing, product-classification, and customer-support automations.  
* **Fast iteration.** New tools or model swaps are often a one-line change.  
* **Minimal deps.** Only `pydantic` (v1 & v2) and `python-dotenv`.  

Use AgentLib as a lightweight workhorse, a prototyping playground, or a study in minimalist agent design.

---

## Features

• **Python-native agent classes** – subclass `BaseAgent`, add methods, you’re done.  
• **Decorator-based tool registry** – function signature & docstring ⇒ tool schema; Pydantic validation happens behind the scenes.  
• **Runtime tool mutation** – Dynamically adjust tool parameters, enums, or availability at any step, improving agent focus and performance by presenting only relevant options.  
• **Clean separation** – LLM orchestration lives in the core; your business logic lives in agents and tools.  
• **Conversation management** – tracks multi-turn context and system prompts for you.  
• **Provider-agnostic** – OpenAI, Google, X.AI, OpenRouter, or roll your own.  
• **Tool call emulation** – Enables both native and emulated tool calls with built-in validation and retry, bypassing inconsistent or poor constrained output performance.

---

## Quick Start

```bash
# 1. Install
pip install git+https://github.com/jacobsparts/agentlib.git

# 2. Set an API key (example: Google Gemini)
export GOOGLE_API_KEY=sk-...

# 3. Run the minimal agent
python examples/todo_agent.py # First run builds a todo list and writes to sqlite
python examples/todo_agent.py # Second run retrieves todo items
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
        self.complete = True                 # marks conversation done
        return hashlib.sha256(text.encode()).hexdigest()

agent = HashAgent()
print(agent.run("What is the SHA-256 of hello world?"))
```

Expected output:

```
b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
google/gemini-2.5-flash: In=342, Out=54, Rsn=61, Cost=$0.000
```

---

## How It Works

1. **Define tools** with ordinary Python functions.  
2. A metaclass decorator captures each function’s signature & docstring, generating a JSON schema with Pydantic.  
3. At runtime the agent builds a prompt that exposes available tools to the LLM.  
4. The LLM selects a tool (or answers directly); AgentLib routes calls, validates inputs/outputs, and appends results to the conversation.  
5. The cycle repeats until `agent.complete` is `True` or max turns are reached.

---

## Supported LLM Providers

| Provider | Env var key        |
|----------|--------------------|
| OpenAI   | `OPENAI_API_KEY`   |
| Google   | `GOOGLE_API_KEY`   |
| X.AI     | `XAI_API_KEY`      |
| OpenRouter | `OPENROUTER_API_KEY` |

Add more chat completions compatible endpoints with `register_provider` and `register_model`.  See `llm_registry.py` for details.

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


---

## Contributing

Issues, feature requests, and pull requests are welcome.  

---

## License

AgentLib is released under the MIT License.  
See [LICENSE](LICENSE) for the full text.
