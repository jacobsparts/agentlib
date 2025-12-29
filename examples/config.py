# Example user configuration for agentlib
# Place this file at ~/.agentlib/config.py
import os
os.environ['LOCAL_API_KEY'] = 'sk-YOUR-API-KEY'

# LLM Registry: Register custom models and providers
register_provider("local",
    host="localhost",
    port=8080,
    path="/v1/chat/completions",
    tpm=100,
    concurrency=10,
    timeout=300,
    tools=False,
    api_type="completions",
)

register_model("local", "llama-3.3-70b",
    aliases="llama",
    input_cost=0.0,
    output_cost=0.0,
)

# Code Agent: Configure default behavior
code_agent_model = "anthropic/claude-opus-4-5"   # Default model
code_agent_max_turns = 50                        # Max conversation turns
code_agent_max_output_kb = 100                   # Max REPL output size (KB)
code_agent_max_display_chars = 300               # Max chars per line in display
code_agent_sandbox = False                       # Run in sandbox by default (--no-sandbox to override)
