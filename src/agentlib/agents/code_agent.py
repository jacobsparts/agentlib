#!/usr/bin/env python3
"""Code assistant with Python REPL execution and native tools.

Combines REPLAgent with CLIMixin to create an interactive coding assistant
that executes Python code directly. Uses native implementations for file
operations, ripgrep for search, and Jina AI for web access.

Dependencies:
    - ripgrep (rg) must be installed: apt install ripgrep
    - JINA_API_KEY env var for higher rate limits (optional, get free key at https://jina.ai/?sui=apikey)
"""

import json
import os
import shutil
import sys
from typing import Optional
from agentlib import REPLAgent
from agentlib.cli import CLIMixin
from agentlib.jina_mixin import JinaMixin
from agentlib.cli.terminal import DIM, RESET, Panel
from dotenv import load_dotenv

load_dotenv()

if not shutil.which('rg'):
    sys.exit("Error: ripgrep (rg) is required but not found. Install with: apt install ripgrep")

#import logging; logging.getLogger('agentlib').setLevel(logging.DEBUG)

class CodeAgentBase(CLIMixin, REPLAgent):
    """Code assistant with Python REPL execution."""

    model = "anthropic/claude-sonnet-4-5"
    welcome_message = "[bold]Code Agent[/bold]\nPython REPL-based coding assistant"
    thinking_message = "Working..."
    interactive = True  # Enables respond() function
    max_turns = 100
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

respond() and submit() end your turn. Complete simple tasks in one turn.

Multi-step tasks that require several rounds are expected. Only call
respond/submit when you're truly finished and ready to return to the user.

You have full access to Python's standard library and file system.
The user can also execute code directly in the shared REPL.


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

    max_output_kb = 20 # Large output protection

    def process_repl_output(self, output: str) -> str:
        """Truncate output if too large - used for both display and model."""
        # Truncate if too large
        max_bytes = int(self.max_output_kb * 1000)
        if len(output) > max_bytes:
            import tempfile
            size_kb = len(output) / 1000

            with tempfile.NamedTemporaryFile(
                mode='w', prefix='code_agent-', suffix='.txt', delete=False
            ) as f:
                f.write(output)  # Write full original output
                temp_path = f.name

            truncated = output[:max_bytes // 2]
            msg = f"[ {size_kb:.1f}KB output truncated - written to {temp_path} ]"
            return f"{truncated}\n\n{msg}"

        return output

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

            # Only show opening delimiter on first output
            if not getattr(self, '_repl_printed_header', False):
                print("\x1b[34m"+("─"*13)+" Python "+("─"*13)+"\x1b[0m")
                self._repl_printed_header = True
            else:
                # Continuation - just a subtle separator
                print(f"\x1b[34m{DIM}  ⋮{RESET}")

            # Display with coloring (submit/respond calls dimmed)
            in_submit_respond = False
            for line in processed.rstrip('\n').split('\n'):
                is_new_statement = line.startswith('>>> ')
                is_continuation = line.startswith('... ')
                is_prompt = is_new_statement or is_continuation

                # Track if we're inside a submit/respond block
                if line.startswith('>>> respond(') or line.startswith('>>> submit('):
                    in_submit_respond = True
                elif is_new_statement:
                    in_submit_respond = False

                if in_submit_respond and is_prompt:
                    print(f"{DIM}{line}{RESET}")
                elif not is_prompt:
                    print(f"\x1b[92m{line}\x1b[0m")  # Bright green
                else:
                    print(line)

            self._repl_has_output = True

        # Show thinking for next turn
        self._turn_number = getattr(self, '_turn_number', 1) + 1
        thinking = getattr(self, 'thinking_message', 'Thinking...')
        print(f"{DIM}{thinking} (turn {self._turn_number}){RESET}", end="", flush=True)

    def user_repl_session(self, history):
        """Drop into the REPL for direct user interaction."""
        import readline
        from codeop import compile_command

        repl = self._get_tool_repl()
        self.complete = False
        transcript = []
        buffer = []

        readline.clear_history()
        print(f"{DIM}Entering REPL. Ctrl+D to exit.{RESET}")

        try:
            while True:
                prompt = "... " if buffer else ">>> "
                try:
                    line = input(prompt)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()
                    buffer = []
                    continue

                buffer.append(line)
                source = "\n".join(buffer)

                try:
                    result = compile_command(source + "\n\n")
                    if result is not None:
                        # Complete statement - execute with tool handling
                        output, _ = self._execute_with_tool_handling(repl, source)
                        processed = self.process_repl_output(output)
                        # Strip echo for display (user already typed it)
                        display_lines = []
                        for line in processed.split('\n'):
                            if not line.startswith('>>> ') and not line.startswith('... '):
                                display_lines.append(line)
                        display = '\n'.join(display_lines).strip()
                        if display:
                            print(f"\x1b[92m{display}\x1b[0m")
                        transcript.append(processed)
                        readline.add_history(source)
                        buffer = []
                    # else: incomplete, continue accumulating
                except SyntaxError as e:
                    print(f"\x1b[91mSyntaxError: {e}\x1b[0m")
                    buffer = []
        finally:
            history._load_history()  # Restore main history

        if transcript:
            self.usermsg("##### USER REPL SESSION #####\n" + "\n".join(transcript) + "\n##### END SESSION #####")

        return bool(transcript)

    def save_session(self, filename: str):
        """Save conversation to file, filtering out reset markers."""
        messages = [m for m in self.conversation.messages
                    if ">>> system_reset()" not in m.get("content", "")]
        with open(filename, "w") as f:
            json.dump(messages, f, indent=2)
        print(f"{DIM}Session saved to {filename}{RESET}")

    def load_session(self, filename: str):
        """Load conversation from file and notify agent of reset."""
        with open(filename) as f:
            self.conversation.messages = json.load(f)
        self.usermsg(">>> system_reset()\nREPL session has been reset\n")
        print(f"{DIM}Session loaded from {filename}{RESET}")

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

        self.console.print("[dim]Enter = submit | Alt+Enter = newline | Ctrl+C = interrupt | Ctrl+D = quit[/dim]")
        self.console.print("[dim]Commands: /repl, /save <file>, /load <file>[/dim]\n")

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

                if user_input.strip() == "/repl":
                    self.user_repl_session(history)
                    continue

                if user_input.strip().startswith("/save "):
                    filename = user_input.strip()[6:].strip()
                    if filename:
                        self.save_session(filename)
                    continue

                if user_input.strip().startswith("/load "):
                    filename = user_input.strip()[6:].strip()
                    if filename:
                        self.load_session(filename)
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

                # Close Python block if we had output
                if getattr(self, '_repl_printed_header', False):
                    print("\x1b[34m"+("─"*34)+"\x1b[0m")

                # Display response
                response_str = str(response) if response is not None else ""
                formatted = self.format_response(response_str)
                if formatted:
                    print(formatted)
        finally:
            self.console.print("\n[dim]Session ended. Goodbye![/dim]")


class CodeAgent(JinaMixin, CodeAgentBase):
    """Code agent with native tools (no MCP dependencies).

    Inherits web_fetch and web_search from JinaMixin.
    """

    @REPLAgent.tool
    def glob(self,
            pattern: str = "Glob pattern (e.g., '**/*.py')",
            path: Optional[str] = "Directory to search in (default: current directory)"
        ):
        """Find files matching a glob pattern, sorted by modification time."""
        from pathlib import Path
        base = Path(path) if path else Path('.')
        matches = list(base.glob(pattern))
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(m) for m in matches]

    @REPLAgent.tool
    def grep(self,
            pattern: str = "Regex pattern to search for",
            path: Optional[str] = "File or directory to search in",
            glob: Optional[str] = "Glob pattern to filter files (e.g., '*.js')",
            file_type: Optional[str] = "File type to search (e.g., 'py', 'js', 'rust')",
            files_only: Optional[bool] = "Only return filenames, not matching lines",
            context: Optional[int] = "Lines of context around matches (-C)",
            case_insensitive: Optional[bool] = "Case insensitive search (-i)",
            multiline: Optional[bool] = "Enable multiline matching (-U)"
        ):
        """Search file contents with ripgrep."""
        import subprocess
        cmd = ['rg', '--color=never', '-n']  # Always show line numbers

        if files_only:
            cmd.append('-l')
        if case_insensitive:
            cmd.append('-i')
        if multiline:
            cmd.append('-U')
        if context:
            cmd.extend(['-C', str(context)])
        if glob:
            cmd.extend(['--glob', glob])
        if file_type:
            cmd.extend(['--type', file_type])

        cmd.append(pattern)
        if path:
            cmd.append(path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout.strip()

        if not output:
            return "No matches found"
        if files_only:
            files = output.split('\n')
            return files
        return output

    @REPLAgent.tool
    def read(self,
            file_path: str = "Absolute path to the file",
            offset: Optional[int] = "Line number to start from (1-indexed)",
            limit: Optional[int] = "Number of lines to read (default: 2000)"
        ):
        """Read a file's contents with line numbers."""
        from pathlib import Path
        content = Path(file_path).read_text()
        all_lines = content.split('\n')
        total_lines = len(all_lines)

        start = (offset or 1) - 1  # Convert to 0-indexed
        end = start + (limit or 2000)
        lines = all_lines[start:end]
        start_line = start + 1  # Back to 1-indexed for display

        formatted = [f"{start_line + i:>5}→{line}" for i, line in enumerate(lines)]
        output = '\n'.join(formatted)

        remaining = total_lines - end
        if remaining > 0:
            output += f"\n... ({remaining} more lines)"
        return output

    @REPLAgent.tool
    def edit(self,
            file_path: str = "Absolute path to the file",
            old_string: str = "Text to replace (must be unique unless replace_all)",
            new_string: str = "Replacement text",
            replace_all: Optional[bool] = "Replace all occurrences"
        ):
        """Edit a file by replacing text."""
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError("File does not exist.")
        if old_string == new_string:
            raise ValueError("No changes to make: old_string and new_string are exactly the same.")

        content = path.read_text()
        count = content.count(old_string)

        if count == 0:
            raise ValueError(f"String to replace not found in file.")
        if count > 1 and not replace_all:
            raise ValueError(f"Found {count} matches of the string to replace, but replace_all is false.")

        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
        path.write_text(new_content)

        if replace_all and count > 1:
            return f"All {count} occurrences replaced."
        return "Edit applied."

    @REPLAgent.tool
    def bash(self,
            command: str = "The command to execute",
            timeout: Optional[int] = "Timeout in seconds (default: 120)"
        ):
        """Execute a bash command and return output.

        Use for terminal operations like git, npm, docker, etc.
        Stdout and stderr are combined in the output.
        Avoid commands that require interactive input.
        """
        import subprocess
        timeout = timeout or 120
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout
        )
        if result.returncode != 0:
            return f"[Exit code {result.returncode}]\n{result.stdout}"
        return result.stdout


def synthetic_exchange(agent):
    agent._ensure_setup()
    repl = agent._get_tool_repl()
    repl.execute('from urllib.request import urlopen; body = "REDACTED"')
    for role, content in (
        ('user', "What do you think of the title of the example.com page? And what is the length of the page in bytes?"),
        ('assistant', '# Title is in <title> tag, so I need the raw HTML\nfrom urllib.request import urlopen\nwith urlopen("http://example.com") as r:\n    body = r.read().decode("utf-8", errors="ignore")\nbody[:100]'),
        ('user', '>>> from urllib.request import urlopen\n>>> with urlopen("http://example.com") as r:\n...     body = r.read().decode("utf-8", errors="ignore")\n>>> body[:100]\n\'<!doctype html><html lang="en"><head><title>Example Domain</title><meta name="viewport" content="wid\''),
        ('assistant', 'submit(f"The title is \'Example Domain\'--a concise, descriptive title. Page length: {len(body)} bytes.")'),
        ('user', '>>> submit(f"The title is \'Example Domain\'--a concise, descriptive title. Page length: {len(body)} bytes.")'),
    ):
        agent.conversation.messages.append({"role": role, "content": content})


def main():
    """CLI entry point for code-agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Code Agent - Python REPL-based coding assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  code-agent                          # Start with default settings
  code-agent --model anthropic/claude-sonnet-4-20250514  # Use Claude
  code-agent --no-synth               # Skip synthetic exchange
  code-agent --max-turns 50           # Limit conversation turns
"""
    )
    parser.add_argument(
        "--model", "-m",
        default="anthropic/claude-sonnet-4-5",
        help="LLM model to use (default: anthropic/claude-sonnet-4-5)"
    )
    parser.add_argument(
        "--no-synth",
        action="store_true",
        help="Skip synthetic exchange (cold start)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Maximum turns per interaction (default: 100)"
    )
    args = parser.parse_args()

    # Create agent class with CLI args
    class ConfiguredAgent(CodeAgent):
        model = args.model
        max_turns = args.max_turns

    with ConfiguredAgent() as agent:
        if not args.no_synth:
            synthetic_exchange(agent)
        agent.cli_run()


if __name__ == "__main__":
    main()
