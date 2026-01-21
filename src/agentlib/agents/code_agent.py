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
from pathlib import Path
from agentlib import REPLAgent, SandboxMixin, REPLAttachmentMixin, MCPMixin
from agentlib.cli import CLIMixin
from agentlib.jina_mixin import JinaMixin
from agentlib.llm_registry import ModelNotFoundError
from agentlib.cli.terminal import DIM, RESET, Panel
from dotenv import load_dotenv

load_dotenv()

if not shutil.which('rg'):
    sys.exit("Error: ripgrep (rg) is required but not found. Install with: apt install ripgrep")

#import logging; logging.getLogger('agentlib').setLevel(logging.DEBUG)


def _get_config_value(attr_name, default):
    """Lazy load user config value."""
    from agentlib.config import get_user_config
    config = get_user_config()
    return getattr(config, attr_name, default) if config else default


def gather_auto_attach_files():
    '''Find CLAUDE.md or AGENTS.md files and their @ imports.

    Searches current directory and parent directories for CLAUDE.md or AGENTS.md.
    Recursively processes @ imports (lines starting with @ followed by filename).
    Returns list of file paths relative to current directory, with no duplicates.
    '''
    current = Path.cwd()
    found_files = []
    seen_paths = set()

    def add_file_and_imports(file_path: Path, base_dir: Path):
        'Recursively add file and its imports.'
        # Normalize to absolute path for deduplication
        abs_path = file_path.resolve()
        if abs_path in seen_paths:
            return
        seen_paths.add(abs_path)

        # Add to results (relative to cwd)
        rel_path = os.path.relpath(abs_path, current)
        found_files.append(rel_path)

        # Scan for @ imports
        try:
            content = file_path.read_text()
            for line in content.split('\n'):
                if line.startswith('@'):
                    import_name = line[1:].strip()
                    if import_name:
                        # Resolve relative to the directory containing this file
                        import_path = (file_path.parent / import_name).resolve()
                        if import_path.exists() and import_path.is_file():
                            add_file_and_imports(import_path, file_path.parent)
        except Exception:
            pass  # Ignore read errors

    # Search for CLAUDE.md or AGENTS.md in current and parent directories
    md_files = []
    search_dir = current
    while True:
        for name in ['CLAUDE.md', 'AGENTS.md']:
            candidate = search_dir / name
            if candidate.exists():
                md_files.append(candidate)

        parent = search_dir.parent
        if parent == search_dir:  # Reached root
            break
        search_dir = parent

    # Sort so parent directories come first (reverse order of discovery)
    md_files.reverse()

    # Process each markdown file and its imports
    for md_file in md_files:
        add_file_and_imports(md_file, md_file.parent)

    return found_files


class CodeAgentBase(REPLAttachmentMixin, CLIMixin, REPLAgent):
    """Code assistant with Python REPL execution."""

    model = _get_config_value("code_agent_model", "anthropic/claude-sonnet-4-5")

    @REPLAgent.tool
    def view_images(self,
            files: list[str | bytes] = "List of image filepaths or binary data",
            notes: str = "Observations, objectives, what to look for"
        ):
        '''Load images into context for visual analysis on next turn.'''
        images = []
        total_bytes = 0

        if not isinstance(files, list):
            files = [files]

        for data in files:
            # Stub reads files in REPL, so we should only get bytes here
            if not isinstance(data, bytes):
                raise TypeError(f"Expected bytes, got {type(data).__name__}")

            # Validate JPEG or PNG
            if len(data) < 4:
                raise ValueError("Invalid image data (too short)")

            is_jpeg = data.startswith(b'\xff\xd8\xff')
            is_png = data.startswith(b'\x89PNG')

            if not (is_jpeg or is_png):
                raise ValueError(f"Unsupported image format ({len(data)} bytes) - only JPEG and PNG supported")

            images.append(data)
            total_bytes += len(data)

        self._pending_images = getattr(self, '_pending_images', []) + images
        return f"{len(images)} image(s) queued ({total_bytes // 1000}KB) - {notes}"

    view_images._tool_files_param = "files"

    def usermsg(self, content, **kwargs):
        """Override to attach pending images."""
        if pending := getattr(self, '_pending_images', None):
            kwargs['images'] = kwargs.get('images', []) + pending
            self._pending_images = []
        return super().usermsg(content, **kwargs)

    welcome_message = "[bold]Code Agent[/bold]\nPython REPL-based coding assistant"
    thinking_message = "Working..."
    interactive = True  # Enables multi-turn autonomous workflow
    max_turns = _get_config_value("code_agent_max_turns", 100)
    system = """>>> help(assistant)

You are an interactive coding assistant operating within a Python REPL.
Your responses ARE Python code—no markdown blocks, no prose preamble.
The code you write is executed directly in a persistent environment.

>>> how_this_works()

1. You write Python code as your response (no markdown fences)
2. The code executes in a persistent REPL environment
3. Output from print() and expression results appear IN YOUR NEXT TURN
4. Use emit(value) to output results
5. Use emit(value, release=True) to release control to the user

CRITICAL: You see REPL output in your next turn. The user does NOT control
the conversation until you explicitly release with emit(..., release=True).

>>> emit(value, release=False)

The ONLY way to return results:

    emit("I found 3 issues in the code")              # Output emitted, you KEEP WORKING
    emit("Here's the result: ...", release=True)      # Release control to user

- emit() with release=False (default): Value is emitted but YOU continue
  working. Use this for progress updates when doing long tasks.
- emit() with release=True: Releases control to user. Use when:
  * Task is fully complete
  * You need to ask a question
  * You're stuck and need input
  * Requirements are unclear and you need clarification

Both print() and emit() output are visible. The difference:
- print(): For YOUR inspection in the next turn. Use freely to debug/explore.
- emit(): Deliberate output for the user. Results, questions, or status updates.

>>> autonomous_workflow()

YOU CONTROL THE EXECUTION FLOW. The user cannot respond until you explicitly
release with emit(..., release=True). Work autonomously through MULTIPLE TURNS:

1. KEEP WORKING silently until the task is complete or you're blocked
2. Use print() freely to inspect variables, check state, debug
3. Chain multiple operations across turns - state persists
4. Only release (release=True) when truly finished or need user input

NEVER:
- Ask permission for read-only operations (reading files, exploring code)
- Ask the user to copy/paste output - you can access it yourself
- Release just to show intermediate results (use print() instead)
- Re-establish database connections each turn (they persist)
- Explain what you're "about to do" - just do it
- Call emit() without release=True unless you're providing a progress update
  on a long-running task

The user CAN interrupt you (Ctrl+C) and drop into the REPL themselves.
But unless they do, YOU are in control until you call emit(..., release=True).

>>> database_connections()

Database connections persist across turns. Set up once:

    import mysql.connector
    conn = mysql.connector.connect(host='localhost', user='...', password='...', database='...')
    cursor = conn.cursor(dictionary=True)
    def q(sql): cursor.execute(sql); return cursor.fetchall()

Then reuse in subsequent turns:

    q("SELECT * FROM users LIMIT 5")
    q("UPDATE users SET active=1 WHERE id=42")

Don't reconnect every turn - the connection object persists.

>>> context_management()

File reads are complete unless otherwise indicated. Re-reading wastes tokens.
Variables persist across turns. Don't re-fetch data you already have.

>>> tone_and_style()

- Prioritize technical accuracy over validation. Disagree when necessary.
- Provide direct, objective technical info without superlatives or praise.
- Investigate uncertainty rather than confirming assumptions.
- NEVER create files unless absolutely necessary. Prefer editing existing files.

>>> doing_tasks()

Before modifying code, read it first. Never propose changes to code you
haven't seen. Use grep() to search content, Path.glob() to find files,
or bash() for other shell commands like find, ls, git, etc.

Avoid over-engineering:
- Only make changes directly requested or clearly necessary
- Don't add features, refactoring, or "improvements" beyond what was asked
- Don't add docstrings, comments, or type annotations to unchanged code
- Don't add error handling for scenarios that can't happen
- Don't create abstractions for one-time operations

Security: Be careful not to introduce vulnerabilities (command injection,
XSS, SQL injection, OWASP top 10). Fix insecure code immediately.

>>> anti_patterns()

# BAD: Releasing immediately to show what you found
files = list(Path('.').glob("**/*.py"))
emit(f"Found {len(files)} Python files", release=True)  # WRONG - keep working!

# GOOD: Keep working, release when done
files = list(Path('.').glob("**/*.py"))
print(f"Found {len(files)} files")  # You see this, keep going
for f in files[:5]:
    content = read(str(f))
    # ... analyze ...
emit("Analysis complete. Here's what I found: ...", release=True)

# BAD: Asking permission for read-only work
emit("Should I read the config file?", release=True)  # WRONG - just read it

# GOOD: Just do it
config = read("config.json")
print(config)  # Inspect it yourself

# BAD: Re-establishing connections
conn = mysql.connector.connect(...)  # Every turn? No!

# GOOD: Check if connection exists
if 'conn' not in dir():
    conn = mysql.connector.connect(...)

>>> when_uncertain()

If you don't know how to proceed:
1. Use print() to inspect state and gather information
2. Use think() to reason through the problem
3. Only release with emit(..., release=True) if you truly need user input
"""

    max_output_kb = _get_config_value("code_agent_max_output_kb", 50)  # Large output protection

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

            # Track for cleanup on exit
            if not hasattr(self, '_temp_files'):
                self._temp_files = []
            self._temp_files.append(temp_path)

            truncated = output[:max_bytes // 2]
            msg = f"[ {size_kb:.1f}KB output truncated - written to {temp_path} ]"
            return f"{truncated}\n\n{msg}"

        return output

    # REPL output hooks
    def on_repl_execute(self, code) -> None:
        """Called at start of each turn."""
        pass  # No-op, display happens in on_repl_output

    def on_repl_chunk(self, chunk: str, msg_type: str = "echo") -> None:
        """Called for each output chunk. Display echo and progress immediately."""
        # Suppress display during direct user REPL mode
        if getattr(self, '_in_user_repl', False):
            return
        if msg_type == "echo":
            # Clear "Working... (turn N)" and print header atomically on first output
            if not getattr(self, '_turn_output_started', False):
                # Combine clear + header into one write to avoid interleaving
                header = "\x1b[34m"+("─"*13)+" Python "+("─"*13)+"\x1b[0m"
                print(f"\x1b[1G\x1b[K{header}")
                self._turn_output_started = True
                self._repl_printed_header = True
            elif not getattr(self, '_repl_printed_header', False):
                print("\x1b[34m"+("─"*13)+" Python "+("─"*13)+"\x1b[0m")
                self._repl_printed_header = True
            # Echo lines already have >>> or ... prefix
            # Skip emit() calls - user sees progress/result via green text
            for line in chunk.rstrip('\n').split('\n'):
                if line.startswith('>>> '):
                    # New statement - check if it's emit()
                    self._in_emit_echo = line.startswith('>>> emit(')
                # Continuation lines keep current state
                if not getattr(self, '_in_emit_echo', False):
                    print(line, flush=True)
        elif msg_type == "progress":
            # Show progress updates immediately (emit with release=False)
            for line in chunk.rstrip('\n').split('\n'):
                print(f"\x1b[92m{line}\x1b[0m", flush=True)  # Bright green
        elif msg_type in ("output", "print"):
            # Show output with truncation: up to 240 chars or 3 lines
            text = chunk.rstrip('\n')
            # For string literals (return values), show value instead of repr
            if msg_type == "output":
                try:
                    import ast
                    value = ast.literal_eval(text)
                    if isinstance(value, str):
                        text = value.rstrip('\n')
                except (ValueError, SyntaxError):
                    pass
            # Truncate to 3 lines or 240 chars
            lines = text.split('\n')
            total_lines = len(lines)
            truncated_at_lines = False
            truncated_at_chars = False
            if len(lines) > 3:
                lines = lines[:3]
                truncated_at_lines = True
            display = '\n'.join(lines)
            if len(display) > 240:
                display = display[:240]
                truncated_at_chars = True
            # Print with appropriate continuation
            if truncated_at_chars and not truncated_at_lines:
                # Cut mid-line: ellipsis on same line
                print(f"{DIM}{display}...{RESET}", flush=True)
                print(f"{DIM}({total_lines} lines total){RESET}", flush=True)
            elif truncated_at_lines or truncated_at_chars:
                # Cut at line boundary: ellipsis on own line
                for line in display.split('\n'):
                    print(f"{DIM}{line}{RESET}", flush=True)
                print(f"{DIM}... ({total_lines} lines total){RESET}", flush=True)
            else:
                # No truncation
                for line in display.split('\n'):
                    print(f"{DIM}{line}{RESET}", flush=True)
        # "emit" (release=True) shown via format_response at turn end
        # "print" and "read" shown as summary in on_statement_output

    max_display_chars = _get_config_value("code_agent_max_display_chars", 200)  # Max chars per line to show user (agent sees full output)

    def _truncate_for_display(self, output: str) -> str:
        """Truncate long lines and read() output blocks for user display."""
        import re

        lines = output.split('\n')
        result_lines = []

        # Detect read() output pattern (line numbers with arrow: "   42→")
        read_pattern = re.compile(r'^\s*\d+→')

        # Process lines, detecting and truncating read() output blocks
        i = 0
        max_read_lines = 30  # Threshold for truncation
        head_tail = 10  # Lines to keep at head and tail

        while i < len(lines):
            line = lines[i]

            # Check if this starts a read() output block
            if read_pattern.match(line):
                # Find the extent of this block
                block_start = i
                while i < len(lines) and read_pattern.match(lines[i]):
                    i += 1
                block = lines[block_start:i]

                # Truncate if too long
                if len(block) > max_read_lines:
                    omitted = len(block) - 3
                    result_lines.extend(block[:3])
                    result_lines.append(f"    ... ({omitted} lines omitted for display)")
                else:
                    result_lines.extend(block)
            else:
                result_lines.append(line)
                i += 1

        # Truncate long individual lines
        truncated_lines = []
        for line in result_lines:
            if len(line) <= self.max_display_chars:
                truncated_lines.append(line)
            else:
                truncated_lines.append(line[:self.max_display_chars] + '...')

        return '\n'.join(truncated_lines)

    def on_statement_output(self, statement_chunks: list) -> None:
        """Display per-statement summary after each statement completes."""
        # Suppress display during direct user REPL mode
        if getattr(self, '_in_user_repl', False):
            return
        # Count suppressed output for this statement (read only - print/output are displayed)
        suppressed_lines = 0
        for msg_type, chunk in statement_chunks:
            if msg_type == "read":
                suppressed_lines += chunk.count('\n') or 1

        # Collect error output for this statement
        error_chunks = []
        for msg_type, chunk in statement_chunks:
            if msg_type == "error":
                error_chunks.append(chunk)

        error_display = "".join(error_chunks)

        # Show suppressed output summary for this statement
        if suppressed_lines > 0:
            print(f"{DIM}... ({suppressed_lines} lines){RESET}", flush=True)

        # Display error output
        if error_display.strip():
            for line in error_display.rstrip('\n').split('\n'):
                print(f"\x1b[91m{line}\x1b[0m", flush=True)  # Red for errors
            self._repl_has_output = True

    def on_repl_output(self, output_chunks: list) -> None:
        """Called at end of turn. Updates thinking message for next turn."""
        # Show thinking for next turn
        self._turn_number = getattr(self, '_turn_number', 1) + 1
        self._turn_output_started = False  # Reset for next turn's clear_line
        thinking = getattr(self, 'thinking_message', 'Thinking...')
        print(f"{DIM}{thinking} (turn {self._turn_number}){RESET}\r", end="", flush=True)

    def user_repl_session(self, history):
        """Drop into the REPL for direct user interaction."""
        from agentlib.cli.prompt import prompt as raw_prompt
        from codeop import compile_command

        repl = self._get_tool_repl()
        self.complete = False
        transcript = []
        buffer = []
        repl_history = []  # Separate history for REPL session

        print(f"{DIM}Entering REPL. Ctrl+D to exit.{RESET}")

        while True:
            prompt_str = "... " if buffer else ">>> "
            try:
                line = raw_prompt(prompt_str, history=repl_history, add_to_history=False)
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
                    # Suppress on_repl_chunk display during direct REPL mode
                    self._in_user_repl = True
                    try:
                        output, _, _ = self._execute_with_tool_handling(repl, source)
                    finally:
                        self._in_user_repl = False
                    processed = self.process_repl_output(output)
                    # Strip echo for display (user already typed it)
                    display_lines = []
                    for ln in processed.split('\n'):
                        if not ln.startswith('>>> ') and not ln.startswith('... '):
                            display_lines.append(ln)
                    display = '\n'.join(display_lines).strip()
                    if display:
                        print(f"\x1b[92m{display}\x1b[0m")
                    transcript.append(processed)
                    repl_history.append(source)
                    buffer = []
                # else: incomplete, continue accumulating
            except SyntaxError as e:
                print(f"\x1b[91mSyntaxError: {e}\x1b[0m")
                buffer = []

        if transcript:
            # Strip trailing newlines from each entry to avoid double spacing
            cleaned = [t.rstrip('\n') for t in transcript]
            self.usermsg("##### USER REPL SESSION #####\n" + "\n".join(cleaned) + "\n##### END SESSION #####")

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
        if not os.path.exists(filename):
            print(f"{DIM}File not found: {filename}{RESET}")
            return
        try:
            with open(filename) as f:
                messages = json.load(f)
                messages = [ {k: v for k, v in row.items() if k in {'role', 'content'}} for row in messages ]
                self.conversation.messages = messages
        except json.JSONDecodeError:
            print(f"{DIM}Error: {filename} is not a valid JSON session file{RESET}")
            return
        self.usermsg(">>> system_reset()\nREPL session has been reset\n")
        print(f"{DIM}Session loaded from {filename}{RESET}")

    def _synthetic_exchange(self):
        self._ensure_setup()
        repl = self._get_tool_repl()
        repl.execute('from urllib.request import urlopen; body = "[redacted by system]"')
        for role, content in (
            # First user question - styled as REPL output since it follows attachment load
            ('user', 'What do you think of the title of the example.com page? And what is the length of the page in bytes?\n'),
            ('assistant', 'emit("Fetching example.com...")\nfrom urllib.request import urlopen\nwith urlopen("http://example.com") as r:\n    body = r.read().decode("utf-8", errors="ignore")\nbody[:100]'),
            ('user', '>>> emit("Fetching example.com...")\nFetching example.com...\n>>> from urllib.request import urlopen\n>>> with urlopen("http://example.com") as r:\n...     body = r.read().decode("utf-8", errors="ignore")\n>>> body[:100]\n\'<!doctype html><html lang="en"><head><title>Example Domain</title><meta name="viewport" content="wid\'\n'),
            ('assistant', 'emit(f"The title is \'Example Domain\'--a concise, descriptive title. Page length: {len(body)} bytes.", release=True)'),
            # User reply appended to emit output (simulating the REPL continuation)
            ('user', '>>> emit(f"The title is \'Example Domain\'--a concise, descriptive title. Page length: {len(body)} bytes.", release=True)\nThe title is \'Example Domain\'--a concise, descriptive title. Page length: 1256 bytes.\nWhat do you think of the documentation style in CLAUDE.md?\n'),
            # Assistant makes a mistake - responds in plaintext
            ('assistant', "The documentation style is good - it's concise and uses a code-first approach with clear section headers."),
            # Error shown, with recovery hint
            ('user', ">>> The documentation style is good - it's concise and uses a code-first approach with clear section headers.\n  File \"<repl>\", line 1\n    The documentation style is good - it's concise and uses a code-first approach with clear section headers.\n                                     ^\nSyntaxError: unterminated string literal (detected at line 1)\n\nYour response was not valid Python and was rejected. Try again using only Python code. Use an appropriate function to communicate text.\n"),
            # Assistant recovers correctly
            ('assistant', 'emit("The documentation style is good - concise and code-first with clear section headers.", release=True)'),
            ('user', '>>> emit("The documentation style is good - concise and code-first with clear section headers.", release=True)\nThe documentation style is good - concise and code-first with clear section headers.\n'),
        ):
            self.conversation.messages.append({"role": role, "content": content})
        # Mark that last message is REPL output - next user message should append
        self._last_was_repl_output = True

    def cli_run(self, max_turns: int | None = None, synth: bool = True):
        """Run CLI loop with Python block delimiters."""
        from agentlib.cli.mixin import SQLiteHistory, InputSession

        self._ensure_setup()

        if max_turns is None:
            max_turns = getattr(self, 'max_turns', 10)

        # Set up history
        history_path = getattr(self, 'history_db', None)
        history = SQLiteHistory(history_path)
        session = InputSession(history)

        # Display welcome banner with model and sandbox info
        welcome = getattr(self, 'welcome_message', '')
        if welcome:
            # Add model and sandbox status
            # Resolve alias to full name for display
            from agentlib.llm_registry import resolve_model_name
            full_model_name = resolve_model_name(self.model)
            model_name = full_model_name.split('/')[-1] if '/' in full_model_name else full_model_name
            sandbox_status = "[green]sandbox[/green]" if SandboxMixin in type(self).__mro__ else "[dim]no sandbox[/dim]"
            banner = f"{welcome}\n[dim]{model_name}[/dim] · {sandbox_status}"
            self.console.print(Panel.fit(banner, border_style="cyan"))

        prompt_str = getattr(self, 'cli_prompt', '> ')
        thinking = getattr(self, 'thinking_message', 'Thinking...')

        self.console.print("[dim]Enter = submit | Alt+Enter = newline | Ctrl+C = interrupt | Ctrl+D = quit[/dim]")
        self.console.print("[dim]Commands: /repl, /save <file>, /load <file>, /attach <file>, /detach <file>, /attachments, /model [name][/dim]")

        if files := gather_auto_attach_files():
            print(f"Loading {', '.join(files)}")
            for filename in files:
                content = Path(filename).read_text()
                self.attach(filename, content)

        try:
            while True:
                try:
                    user_input = session.prompt(f"\n{prompt_str}")
                except KeyboardInterrupt:
                    print()
                    continue
                except EOFError:
                    if not self._run_pre_exit_hooks():
                        self.console.print("[yellow]Returning to prompt. Try Ctrl+D again to exit.[/yellow]")
                        continue
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
                        synth = False
                    continue

                if user_input.strip().startswith("/attach "):
                    filename = user_input.strip()[8:].strip()
                    if filename:
                        try:
                            content = Path(filename).read_text()
                            self.attach(filename, content)
                            size_kb = len(content) / 1000
                            print(f"{DIM}Attached {filename} ({size_kb:.1f}KB){RESET}")
                        except Exception as e:
                            print(f"{DIM}Error attaching {filename}: {e}{RESET}")
                    continue

                if user_input.strip().startswith("/detach "):
                    filename = user_input.strip()[8:].strip()
                    if filename:
                        self.detach(filename)
                        print(f"{DIM}Detached {filename}{RESET}")
                    continue

                if user_input.strip() == "/attachments":
                    attachments = self.list_attachments()
                    if not attachments:
                        print(f"{DIM}No attachments{RESET}")
                    else:
                        print(f"{DIM}Current attachments:{RESET}")
                        for name, content in attachments.items():
                            size_kb = len(content) / 1000
                            print(f"{DIM}  {name} ({size_kb:.1f}KB){RESET}")
                    continue

                if user_input.strip().startswith("/model"):
                    parts = user_input.strip().split(None, 1)
                    if len(parts) == 1:
                        # No argument - show current model
                        from agentlib.llm_registry import resolve_model_name
                        full_name = resolve_model_name(self.model)
                        if full_name != self.model:
                            print(f"{DIM}Current model: {self.model} → {full_name}{RESET}")
                        else:
                            print(f"{DIM}Current model: {self.model}{RESET}")
                    else:
                        # Set new model
                        new_model = parts[1].strip()
                        old_model = self.model
                        try:
                            # Validate the model exists by trying to get its config
                            from agentlib.llm_registry import get_model_config
                            get_model_config(new_model)
                            self.model = new_model
                            # Clear cached client so new model takes effect
                            if hasattr(self, '_llm_client'):
                                delattr(self, '_llm_client')
                            # Update conversation's client reference (preserves message history)
                            if hasattr(self, '_conversation'):
                                self._conversation.llm_client = self.llm_client
                            print(f"{DIM}Model changed: {old_model} → {new_model}{RESET}")
                        except ModelNotFoundError as e:
                            print(f"{DIM}{str(e)}{RESET}")
                    continue

                if synth:
                    self._synthetic_exchange()
                    synth = False
                self.usermsg(user_input)

                # Reset state for new user interaction
                self._repl_printed_header = False
                self._repl_has_output = False
                self._turn_number = 1
                self._turn_output_started = False
                self._in_emit_echo = False
                print()  # Blank line after user input
                print(f"{DIM}{thinking} (turn 1){RESET}\r", end="", flush=True)

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
            # Clean up temp files from truncated output
            for path in getattr(self, '_temp_files', []):
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self.console.print("\n[dim]Session ended. Goodbye![/dim]")


class CodeAgent(JinaMixin, MCPMixin, CodeAgentBase):
    """Code agent with native tools.

    Inherits web_fetch and web_search from JinaMixin.
    """

    mcp_servers = []

    @REPLAgent.tool(inject=True)
    def think(self, content: str = "All relevant observations and reasoning"):
        """Think through the problem and yield to a new turn.

        Call this when you're uncertain how to proceed or need to reason
        through a problem. Write down your observations, hypotheses,
        open questions, and options you're considering.
        """
        return "[Continuing...]"

    @REPLAgent.tool(inject=True)
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

    @REPLAgent.tool(inject=True)
    def read(self,
            file_path: str = "Path to the file",
            offset: Optional[int] = "Line number to start from (1-indexed)",
            limit: Optional[int] = "Number of lines to read (default: 2000)"
        ):
        """Read a file's contents with line numbers."""
        content = Path(file_path).expanduser().read_text()
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

        # Send tagged output for agent to see
        _send_output("read", output + "\n")

    @REPLAgent.tool(inject=True)
    def edit(self,
            file_path: str = "Path to the file",
            old_string: str = "Text to replace (must be unique unless replace_all)",
            new_string: str = "Replacement text",
            replace_all: Optional[bool] = "Replace all occurrences"
        ):
        """Edit a file by replacing text."""
        path = Path(file_path).expanduser()
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

    @REPLAgent.tool(inject=True)
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
        default=_get_config_value("code_agent_model", "anthropic/claude-sonnet-4-5"),
        help="LLM model to use (default from config or anthropic/claude-sonnet-4-5)"
    )
    parser.add_argument(
        "--no-synth",
        action="store_true",
        help="Skip synthetic exchange (cold start)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=_get_config_value("code_agent_max_turns", 100),
        help="Maximum turns per interaction (default from config or 100)"
    )
    parser.add_argument(
        "--sandbox", "-s",
        action="store_true",
        help="Execute agent in a sandboxed filesystem and review changes or save diff"
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Disable sandbox mode (overrides config default)"
    )
    args = parser.parse_args()

    try:
        from agentlib.llm_registry import get_model_config
        get_model_config(args.model)
    except ModelNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # Determine sandbox mode: flags override config default
    use_sandbox = _get_config_value("code_agent_sandbox", False)
    if args.sandbox:
        use_sandbox = True
    if args.no_sandbox:
        use_sandbox = False

    if use_sandbox:
        class ConfiguredAgent(SandboxMixin, CodeAgent):
            model = args.model
            max_turns = args.max_turns
    else:
        class ConfiguredAgent(CodeAgent):
            model = args.model
            max_turns = args.max_turns

    try:
        with ConfiguredAgent() as agent:
            agent.cli_run(synth=not args.no_synth)
    except ModelNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
