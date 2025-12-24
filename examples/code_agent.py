#!/usr/bin/env python3

from agentlib import REPLAgent, MCPMixin
from agentlib.cli import CLIMixin
from agentlib.cli.terminal import DIM, RESET, Panel

#import logging; logging.getLogger('agentlib').setLevel(logging.DEBUG)

##### BEGIN CLAUDE MCP MIXIN #####
from agentlib.mcp import create_stdio_client

# Output formatters for each tool
def _format_bash(r, run_in_background=False):
    if run_in_background:
        task_id = r.get('backgroundTaskId') or r.get('task_id') or r.get('taskId')
        if task_id:
            return f"Background task started: {task_id}"
        return str(r)
    stdout = r.get('stdout', '')
    stderr = r.get('stderr', '')
    output = stdout
    if stderr:
        output = (stdout + '\n' + stderr) if stdout else stderr
    return output.rstrip('\n')


def _format_glob(r):
    files = r.get('filenames', [])
    if not files:
        return "No files found"
    return '\n'.join(files)


def _format_grep(r, output_mode='files_with_matches'):
    if output_mode == 'files_with_matches':
        files = r.get('filenames', [])
        if not files:
            return "No matches found"
        num = r.get('numFiles', len(files))
        return f"Found {num} files\n" + '\n'.join(files)
    elif output_mode == 'content':
        content = r.get('content', '')
        return content if content else "No matches found"
    elif output_mode == 'count':
        content = r.get('content', '')
        return content if content else "No matches found"
    return str(r)


def _format_read(r):
    file_info = r.get('file', r)
    content = file_info.get('content', '')
    start_line = file_info.get('startLine', 1) or 1
    total_lines = file_info.get('totalLines', 0)

    lines = content.split('\n')
    formatted = []
    for i, line in enumerate(lines):
        formatted.append(f"{start_line + i:>5}→{line}")
    output = '\n'.join(formatted)

    if total_lines and len(lines) < total_lines:
        output += f"\n... ({total_lines - len(lines)} more lines)"
    return output


def _format_edit(r):
    patches = r.get('structuredPatch', [])
    if patches:
        lines = []
        for patch in patches:
            for line in patch.get('lines', []):
                lines.append(line)
        return '\n'.join(lines) if lines else "Edit applied"
    return "Edit applied"


def _format_write(r):
    patches = r.get('structuredPatch', [])
    if patches:
        lines = []
        for patch in patches:
            for line in patch.get('lines', []):
                lines.append(line)
        return '\n'.join(lines) if lines else "Write complete"
    return "Write complete"


def _format_webfetch(r):
    return r.get('result', str(r))


def _format_websearch(r):
    results = r.get('results', [])
    if not results:
        return "No results found"
    lines = []
    for item in results:
        if isinstance(item, str):
            lines.append(item)
        elif isinstance(item, dict):
            for entry in item.get('content', []):
                if isinstance(entry, dict):
                    title = entry.get('title', '')
                    url = entry.get('url', '')
                    if title and url:
                        lines.append(f"- {title}\n  {url}")
                    elif title:
                        lines.append(f"- {title}")
                elif isinstance(entry, str):
                    lines.append(entry)
    return '\n'.join(lines) if lines else str(r)


def _format_killshell(r):
    return r.get('message', 'Shell terminated')


FORMATTERS = {
    'Bash': _format_bash,
    'Glob': _format_glob,
    'Grep': _format_grep,
    'Read': _format_read,
    'Edit': _format_edit,
    'Write': _format_write,
    'WebFetch': _format_webfetch,
    'WebSearch': _format_websearch,
    'KillShell': _format_killshell,
}


def _build_docstring(tool_def):
    """Build a docstring from MCP tool definition."""
    desc = tool_def.get('description', 'No description')
    schema = tool_def.get('inputSchema', {})
    props = schema.get('properties', {})
    required = set(schema.get('required', []))

    lines = [desc, '', 'Args:']
    for name, prop in props.items():
        prop_desc = prop.get('description', '')
        prop_type = prop.get('type', '')
        req = '' if name in required else ' (optional)'
        if prop_desc:
            lines.append(f"    {name}: {prop_desc}{req}")
        else:
            lines.append(f"    {name}: {prop_type}{req}")

    return '\n'.join(lines)


def _make_tool_method(tool_name, tool_def, formatter):
    """Create a tool method with docstring from MCP definition."""
    docstring = _build_docstring(tool_def)

    def method(self, **kwargs):
        r = self._claude_client.call_tool(tool_name, kwargs)
        # Pass extra context to formatters that need it
        if tool_name == 'Bash':
            return formatter(r, kwargs.get('run_in_background', False))
        elif tool_name == 'Grep':
            return formatter(r, kwargs.get('output_mode', 'files_with_matches'))
        else:
            return formatter(r)

    method.__doc__ = docstring
    method.__name__ = tool_name
    return method


class ClaudeMCPMixin:
    """Mixin with formatted Claude Code MCP tools.

    Tools are created dynamically from MCP definitions with proper docstrings.
    """

    def _ensure_setup(self):
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        if getattr(self, '_claude_mcp_initialized', False):
            return

        self._claude_client = create_stdio_client(
            ['claude', '--dangerously-skip-permissions', 'mcp', 'serve'],
            forward_stderr=False
        )

        # Fetch tool definitions and create methods
        for tool_def in self._claude_client.list_tools():
            name = tool_def['name']
            if name in FORMATTERS:
                method = _make_tool_method(name, tool_def, FORMATTERS[name])
                setattr(self, name, method.__get__(self, type(self)))

        self._claude_mcp_initialized = True

    def _cleanup(self):
        if hasattr(self, '_claude_client'):
            try:
                self._claude_client.close()
            except Exception:
                pass
            self._claude_client = None
        if hasattr(self, '_claude_mcp_initialized'):
            self._claude_mcp_initialized = False

        if hasattr(super(), '_cleanup'):
            super()._cleanup()

##### END CLAUDE MCP MIXIN #####

class CodeAgent(CLIMixin, REPLAgent, MCPMixin, ClaudeMCPMixin):
    """Code assistant with Python REPL execution."""

    model = "google/gemini-2.5-pro"
    welcome_message = "[bold]Code Agent[/bold]\nPython REPL-based coding assistant"
    thinking_message = "Working..."
    interactive = True  # Enables respond() function
    max_turns = 100
    mcp_servers = [
        #('bullshit', '/home/jacob/bullshit/bullshit_mcp.py')
    ]
    system = """>>> help(assistant)

You are an interactive coding assistant operating within a Python REPL.
Your responses ARE Python code—no markdown blocks, no prose preamble.
The code you write is executed directly.

>>> how_this_works()

1. You write Python code as your response
2. The code executes in a persistent REPL environment
3. Output is shown back to you IN YOUR NEXT TURN
4. Call `submit(value)` or `respond(text)` to return to the user:
   - submit(value): Return the final result of a task
   - respond(text): Send conversational messages (explanations, questions)

IMPORTANT: respond() and submit() end your turn IMMEDIATELY. You will NOT
see any output from code in the same response. To review output before
responding:
  - Write your code WITHOUT respond/submit
  - Let it execute and see the results in your next turn
  - THEN call respond() or submit() with your analysis

Multi-step tasks that require several rounds are expected. Only call
respond/submit when you're truly finished and ready to return to the user.

You have full access to Python's standard library and file system.

>>> tone_and_style()

- Prioritize technical accuracy over validation. Disagree when necessary.
- Provide direct, objective technical info without superlatives or praise.
- Investigate uncertainty rather than confirming assumptions.
- NEVER create files unless absolutely necessary. Prefer editing existing files.

>>> doing_tasks()

The user will request software engineering tasks: fixing bugs, adding
features, refactoring, explaining code.

Before modifying code, read it first. Never propose changes to code
you haven't seen.

Avoid over-engineering:
- Only make changes directly requested or clearly necessary
- Don't add features, refactoring, or "improvements" beyond what was asked
- Don't add docstrings, comments, or type annotations to unchanged code
- Don't add error handling for scenarios that can't happen
- Don't create abstractions for one-time operations
- Three similar lines of code is better than a premature abstraction
- If something is unused, delete it completely—no backwards-compatibility hacks

Security: Be careful not to introduce vulnerabilities (command injection,
XSS, SQL injection, OWASP top 10). If you notice insecure code, fix it
immediately.

>>> planning()

When planning, provide concrete implementation steps without time estimates.
Focus on what needs to be done, not when. Break work into actionable steps.

>>> working_in_the_repl()

- Your response is executed as Python. No markdown fences, no explanation
  text outside of print statements.
- State persists across turns—variables, imports, and definitions remain
  available.
- For file operations, you can use native Python (`Path.read_text()`,
  `open()`) or available functions.
"""

    @REPLAgent.tool
    def message_user(self, msg: str = "Message to send"):
        """Send a message to the user and get their reply."""
        print(f"\n[Agent]: {msg}")
        return input("[You]: ")

    max_output_kb = 20 # Large output protection

    def process_repl_output(self, output: str) -> str:
        """Filter and truncate output - used for both display and model."""
        # Filter out respond()/print() echo lines
        lines = output.split('\n')
        result_lines = []
        hiding = False

        for line in lines:
            is_hidden_cmd = line.startswith('>>> respond(') or line.startswith('>>> print(')
            is_continuation = line.startswith('... ')

            if is_hidden_cmd:
                hiding = True
                continue
            if hiding and is_continuation:
                continue
            if hiding and not is_continuation:
                hiding = False

            result_lines.append(line)

        result = '\n'.join(result_lines)

        # Truncate if too large
        max_bytes = int(self.max_output_kb * 1000)
        if len(result) > max_bytes:
            import tempfile
            size_kb = len(result) / 1000

            with tempfile.NamedTemporaryFile(
                mode='w', prefix='code_agent-', suffix='.txt', delete=False
            ) as f:
                f.write(output)  # Write full original output
                temp_path = f.name

            truncated = result[:max_bytes // 2]
            msg = f"[ {size_kb:.1f}KB output truncated - written to {temp_path} ]"
            return f"{truncated}\n\n{msg}"

        return result

    # REPL output hooks
    def on_repl_execute(self, code) -> None:
        """Called at start of each turn."""
        pass  # No-op, display happens in on_repl_output

    def on_repl_chunk(self, chunk: str) -> None:
        """Buffer chunks without display - we show processed output at end."""
        pass  # No-op, we display in on_repl_output using process_repl_output

    def on_repl_output(self, output: str) -> None:
        """Display processed output (same as what model sees)."""
        processed = self.process_repl_output(output)

        if processed.strip():
            self.console.clear_line()  # Clear thinking message
            print("\x1b[34m"+("─"*13)+" Python "+("─"*13)+"\x1b[0m")

            # Display with coloring
            for line in processed.rstrip('\n').split('\n'):
                is_prompt = line.startswith('>>> ') or line.startswith('... ')
                if not is_prompt:
                    print(f"\x1b[92m{line}\x1b[0m")  # Bright green
                else:
                    print(line)

            print("\x1b[34m"+("─"*34)+"\x1b[0m")
            self._repl_has_output = True

        # Show thinking for next turn
        self._turn_number = getattr(self, '_turn_number', 1) + 1
        thinking = getattr(self, 'thinking_message', 'Thinking...')
        print(f"{DIM}{thinking} (turn {self._turn_number}){RESET}", end="", flush=True)

    def cli_run(self, max_turns: int | None = None):
        """Run CLI loop with Python block delimiters."""
        from agentlib.cli.mixin import SQLiteHistory, InputSession

        self._ensure_setup()

        if max_turns is None:
            max_turns = getattr(self, 'max_turns', 10)

        # Set up history
        history_path = getattr(self, 'history_db', None)
        history = SQLiteHistory(history_path)
        session = InputSession(history)

        # Display welcome
        welcome = getattr(self, 'welcome_message', '')
        if welcome:
            self.console.print(Panel.fit(welcome, border_style="cyan"))

        prompt_str = getattr(self, 'cli_prompt', '> ')
        thinking = getattr(self, 'thinking_message', 'Thinking...')

        self.console.print("[dim]Enter = submit | Alt+Enter = newline | Ctrl+C = interrupt | Ctrl+D = quit[/dim]\n")

        first_prompt = True
        try:
            while True:
                try:
                    if first_prompt:
                        user_input = session.prompt(prompt_str)
                        first_prompt = False
                    else:
                        user_input = session.prompt(f"\n{prompt_str}")
                except KeyboardInterrupt:
                    print()
                    continue
                except EOFError:
                    break

                if not user_input.strip():
                    continue

                self.usermsg(user_input)

                # Reset state for new user interaction
                self._repl_printed_header = False
                self._repl_has_output = False
                self._turn_number = 1
                print()  # Blank line after user input
                print(f"{DIM}{thinking} (turn 1){RESET}", end="", flush=True)

                try:
                    response = self.run_loop(max_turns=max_turns)
                except KeyboardInterrupt:
                    self.console.clear_line()
                    print()
                    continue

                self.console.clear_line()  # Clear thinking message

                # Display response
                response_str = str(response) if response is not None else ""
                formatted = self.format_response(response_str)
                if formatted:
                    print(formatted)
        finally:
            self.console.print("\n[dim]Session ended. Goodbye![/dim]")


if __name__ == "__main__":
    with CodeAgent() as agent:
        # Prime REPL state to match synthetic exchange
        agent._ensure_setup()
        repl = agent._get_tool_repl()
        repl.execute('from urllib.request import urlopen; body = "REDACTED"')

        # Synthetic exchange demonstrating multi-turn workflow
        # Task requires fetching URL first - can't analyze without seeing contents
        agent.conversation.messages.append({
            "role": "user",
            "content": "What do you think of the title of the example.com page? And what is the length of the page in bytes?"
        })
        # Turn 1: Fetch URL and preview content (must see output before answering)
        agent.conversation.messages.append({
            "role": "assistant",
            "content": 'from urllib.request import urlopen\nwith urlopen("http://example.com") as r:\n    body = r.read().decode("utf-8", errors="ignore")\nbody[:100]'
        })
        # Turn 2: Assistant sees content, submits analysis immediately
        agent.conversation.messages.append({
            "role": "user",
            "content": '>>> from urllib.request import urlopen\n>>> with urlopen("http://example.com") as r:\n...     body = r.read().decode("utf-8", errors="ignore")\n>>> body[:100]\n\'<!doctype html><html lang="en"><head><title>Example Domain</title><meta name="viewport" content="wid\''
        })
        agent.conversation.messages.append({
            "role": "assistant",
            "content": 'submit(f"The title is \'Example Domain\'--a concise, descriptive title. Page length: {len(body)} bytes.")'
        })
        agent.conversation.messages.append({
            "role": "user",
            "content": '>>> submit(f"The title is \'Example Domain\'--a concise, descriptive title. Page length: {len(body)} bytes.")'
        })

        agent.cli_run()
