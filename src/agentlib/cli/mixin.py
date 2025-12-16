"""
CLIMixin - Mixin that adds CLI REPL functionality to any agent.

Example:
    from agentlib import BaseAgent, SubREPLResponseMixin
    from agentlib.cli import CLIMixin

    class MyAssistant(CLIMixin, SubREPLResponseMixin, BaseAgent):
        model = 'anthropic/claude-sonnet-4-5'
        system = "You are a helpful assistant."
        welcome_message = "Welcome! I can help you with Python tasks."

    if __name__ == "__main__":
        with MyAssistant() as agent:
            agent.cli_run()

Or use the pre-composed CLIAgent:

    from agentlib.cli import CLIAgent

    class MyAssistant(CLIAgent):
        model = 'anthropic/claude-sonnet-4-5'
        system = "You are a helpful assistant."

    MyAssistant.main()
"""

import sqlite3
import readline
import sys
from pathlib import Path
from typing import Optional, Any

from .terminal import (
    Console, Panel, Markdown, render_markdown, parse_markup,
    DIM, RESET, strip_ansi
)


# =============================================================================
# SQLite History for readline
# =============================================================================

class SQLiteHistory:
    """SQLite-backed history that integrates with readline."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".agentlib_cli_history.db")
        else:
            db_path = str(Path(db_path).expanduser())
        self.db_path = db_path
        self._init_db()
        self._load_history()

    def _init_db(self):
        """Create the history table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _load_history(self):
        """Load history from SQLite into readline."""
        readline.clear_history()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT command FROM history ORDER BY id ASC"
            )
            for row in cursor:
                readline.add_history(row[0])

    def add(self, command: str):
        """Add a command to history."""
        if command.strip():
            readline.add_history(command)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO history (command) VALUES (?)",
                    (command,)
                )
                conn.commit()


# =============================================================================
# Input Session
# =============================================================================

class InputSession:
    """Input session with readline history support.

    By default, Enter submits and Alt+Enter inserts a newline.
    """

    def __init__(self, history: Optional[SQLiteHistory] = None):
        self.history = history or SQLiteHistory()
        self._setup_bindings()

    def _setup_bindings(self):
        """Configure readline for input."""
        # Alt+Enter inserts a literal newline
        readline.parse_and_bind(r'"\e\C-m": "\C-v\n"')

    def prompt(self, prompt_str: str = "> ") -> str:
        """Get input from user."""
        try:
            user_input = input(prompt_str)
            if user_input.strip():
                self.history.add(user_input)
            return user_input
        except EOFError:
            raise
        except KeyboardInterrupt:
            raise


# =============================================================================
# CLIMixin
# =============================================================================

class CLIMixin:
    """
    Mixin that adds CLI REPL functionality to any agent.

    Class Attributes:
        welcome_message: Message to display when CLI starts (supports markup)
        cli_prompt: Input prompt string (default: "> ")
        history_db: Path to SQLite history database (default: ~/.agentlib_cli_history.db)
        max_turns: Maximum agent turns per user message (default: 20)
        thinking_message: Message to show while agent is thinking (default: "Thinking...")

    Override these methods for customization:
        on_tool_call(name, args): Called before each tool execution
        on_tool_result(name, result): Called after each tool returns
        format_response(response): Format the final response before display

    Example:
        class MyAssistant(CLIMixin, SubREPLResponseMixin, BaseAgent):
            model = 'anthropic/claude-sonnet-4-5'
            system = "You are helpful."
            welcome_message = "[bold]My Assistant[/bold]\\nReady to help!"

            def on_tool_call(self, name, args):
                if name == 'python_execute':
                    self.console.panel(args.get('code', ''), title="Python", border_style="blue")

        with MyAssistant() as agent:
            agent.cli_run()
    """

    # Configuration
    welcome_message: str = "[bold]Assistant[/bold]\nReady to help."
    cli_prompt: str = "> "
    history_db: Optional[str] = None
    max_turns: int = 20
    thinking_message: str = "Thinking..."

    def _ensure_setup(self):
        """Initialize CLI components."""
        # Chain to next in MRO
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        # Initialize console if not already done
        if not hasattr(self, '_cli_console'):
            self._cli_console = Console()

    @property
    def console(self) -> Console:
        """Get the console instance."""
        self._ensure_setup()
        return self._cli_console

    # === CUSTOMIZATION HOOKS ===

    def on_tool_call(self, name: str, args: dict) -> None:
        """
        Called before each tool is executed.

        Override to customize tool call display. Default shows Python code
        in a panel and dim messages for other tools.

        Args:
            name: Tool name
            args: Tool arguments
        """
        self.console.clear_line()

        if name == 'python_execute':
            self.console.panel(args.get('code', ''), title="Python", border_style="blue")
            print()
        elif name == 'python_execute_response':
            self.console.panel(args.get('code', ''), title="Python (response)", border_style="blue")
            print()
        elif name == 'python_read':
            self.console.print(f"[dim]Reading output (timeout={args.get('timeout', 30)})...[/dim]")
        elif name == 'python_interrupt':
            self.console.print("[red]Interrupting execution...[/red]")
        elif name == 'shell_execute':
            self.console.panel(args.get('command', ''), title="Shell", border_style="yellow")
            print()
        elif name.startswith('shell_'):
            self.console.print(f"[dim]{name}...[/dim]")

    def on_tool_result(self, name: str, result: Any) -> None:
        """
        Called after each tool returns.

        Override to customize tool result display. Default shows output
        in a panel for execution tools.

        Args:
            name: Tool name
            result: Tool result
        """
        if result is None:
            return

        result_str = str(result).strip()
        if not result_str:
            return

        # Truncate long output
        max_len = 2000
        if len(result_str) > max_len:
            result_str = result_str[:max_len] + '...'

        if name in ('python_execute', 'python_read', 'shell_execute', 'shell_read'):
            self.console.panel(result_str, title="Output", border_style="green")
        elif name.startswith('python_') or name.startswith('shell_'):
            if result_str:
                self.console.panel(result_str, title="Result", border_style="green")

    def format_response(self, response: str) -> str:
        """
        Format the final response before display.

        Override to customize response formatting. Default renders markdown.

        Args:
            response: The agent's response string

        Returns:
            Formatted response string
        """
        return render_markdown(response)

    # === INTERNAL HOOK ===

    def _handle_toolcall(self, toolname: str, function_args: dict):
        """Intercept tool calls to invoke hooks."""
        # Call the on_tool_call hook
        self.on_tool_call(toolname, function_args)

        # Call the actual tool handler (parent)
        if hasattr(super(), '_handle_toolcall'):
            handled, result = super()._handle_toolcall(toolname, function_args)
        else:
            handled, result = False, None

        # Call the on_tool_result hook
        if handled:
            self.on_tool_result(toolname, result)

        return handled, result

    # === CLI ENTRY POINT ===

    def cli_run(self) -> None:
        """
        Run the interactive CLI loop.

        This is the main entry point for CLI interaction. It displays
        the welcome message, then enters a loop where it:
        1. Prompts for user input
        2. Sends input to the agent
        3. Displays the response

        The loop continues until Ctrl+C or Ctrl+D.
        """
        self._ensure_setup()

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
        max_turns = getattr(self, 'max_turns', 20)

        self.console.print("[dim]Enter = submit | Alt+Enter = newline | Ctrl+C/D = quit[/dim]\n")

        try:
            while True:
                try:
                    user_input = session.prompt(f"\n{prompt_str}")
                except (KeyboardInterrupt, EOFError):
                    break

                if not user_input.strip():
                    continue

                # Send to agent
                self.usermsg(user_input)

                # Show thinking indicator
                print(f"\n{DIM}{thinking}{RESET}", end="", flush=True)

                # Run agent loop
                response = self.run_loop(max_turns=max_turns)

                # Clear thinking and display response
                self.console.clear_line()
                formatted = self.format_response(response)
                print(formatted)

        finally:
            self.console.print("\n[dim]Session ended. Goodbye![/dim]")

    @classmethod
    def main(cls, **init_kwargs) -> None:
        """
        Convenience entry point that creates an instance and runs the CLI.

        Usage:
            if __name__ == "__main__":
                MyAssistant.main()

        Args:
            **init_kwargs: Arguments to pass to the constructor
        """
        with cls(**init_kwargs) as agent:
            agent.cli_run()
