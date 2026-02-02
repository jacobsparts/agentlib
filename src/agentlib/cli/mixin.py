"""
CLIMixin - Mixin that adds interactive CLI functionality to any agent.

Example:
    from agentlib import BaseAgent
    from agentlib.cli import CLIMixin

    class MyAssistant(CLIMixin, BaseAgent):
        model = 'anthropic/claude-sonnet-4-5'
        system = "You are a helpful assistant."
        welcome_message = "Welcome!"

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
import sys
from pathlib import Path
from typing import Optional, Any

from .prompt import prompt as raw_prompt
from .stdout_capture import StdoutCapture

from .terminal import (
    Console, Panel, Markdown, render_markdown, parse_markup,
    DIM, RESET, strip_ansi
)


# =============================================================================
# SQLite History
# =============================================================================

class SQLiteHistory:
    """SQLite-backed history for CLI input."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".agentlib_cli_history.db")
        else:
            db_path = str(Path(db_path).expanduser())
        self.db_path = db_path
        self._init_db()

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

    def load_history(self) -> list[str]:
        """Load history from SQLite as a list."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT command FROM history ORDER BY id ASC"
            )
            return [row[0] for row in cursor]

    def add(self, command: str):
        """Add a command to history."""
        if command.strip():
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
    """Input session with history support and bracketed paste.

    By default, Enter submits and Alt+Enter inserts a newline.
    Pasted multiline content is buffered as a single input.
    """

    def __init__(self, history: Optional[SQLiteHistory] = None,
                 stdout_capture: Optional[StdoutCapture] = None):
        self.history = history or SQLiteHistory()
        self._history_list = self.history.load_history()
        self.stdout_capture = stdout_capture

    def prompt(self, prompt_str: str = "> ") -> str:
        """Get input from user."""
        # Create callbacks for captured output and alt buffer pause/resume
        get_screen_content = None
        on_alt_enter = None
        on_alt_exit = None
        if self.stdout_capture:
            get_screen_content = self.stdout_capture.get_recent_for_screen
            on_alt_enter = self.stdout_capture.pause
            on_alt_exit = self.stdout_capture.resume

        user_input = raw_prompt(
            prompt_str=prompt_str,
            history=self._history_list,
            on_submit=self.history.add,
            get_screen_content=get_screen_content,
            on_alt_enter=on_alt_enter,
            on_alt_exit=on_alt_exit,
        )
        return user_input


# =============================================================================
# CLIMixin
# =============================================================================

class CLIMixin:
    """
    Mixin that adds interactive CLI functionality to any agent.

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
        class MyAssistant(CLIMixin, BaseAgent):
            model = 'anthropic/claude-sonnet-4-5'
            system = "You are helpful."
            welcome_message = "[bold]My Assistant[/bold]\\nReady to help!"

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

        Override to customize tool call display.

        Args:
            name: Tool name
            args: Tool arguments
        """
        pass

    def on_tool_result(self, name: str, result: Any) -> None:
        """
        Called after each tool returns.

        Override to customize tool result display.

        Args:
            name: Tool name
            result: Tool result
        """
        pass

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

    def toolcall(self, toolname: str, function_args: dict):
        """Intercept tool calls to invoke hooks."""
        self.on_tool_call(toolname, function_args)
        result = super().toolcall(toolname, function_args)
        self.on_tool_result(toolname, result)
        return result

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

        # Set up stdout capture for alt-buffer replay
        stdout_capture = StdoutCapture()
        stdout_capture.install()

        # Set up history
        history_path = getattr(self, 'history_db', None)
        history = SQLiteHistory(history_path)
        session = InputSession(history, stdout_capture=stdout_capture)

        # Display welcome
        welcome = getattr(self, 'welcome_message', '')
        if welcome:
            self.console.print(Panel.fit(welcome, border_style="cyan"))

        prompt_str = getattr(self, 'cli_prompt', '> ')
        thinking = getattr(self, 'thinking_message', 'Thinking...')
        max_turns = getattr(self, 'max_turns', 20)

        self.console.print("[dim]Enter = submit | Alt+Enter = newline | Ctrl+C = interrupt | Ctrl+D = quit[/dim]\n")

        try:
            while True:
                try:
                    user_input = session.prompt(f"\n{prompt_str}")
                except KeyboardInterrupt:
                    print()  # Just print newline, stay at prompt
                    continue
                except EOFError:
                    if not self._run_pre_exit_hooks():
                        self.console.print("[yellow]Returning to prompt. Try Ctrl+D again to exit.[/yellow]")
                        continue
                    break

                if not user_input.strip():
                    continue

                # Send to agent
                self.usermsg(user_input)

                # Show thinking indicator (cursor left at start so output overwrites)
                print(f"{DIM}{thinking}{RESET}\r", end="", flush=True)

                # Run agent loop (may be interrupted by Ctrl+C)
                try:
                    response = self.run_loop(max_turns=max_turns)
                except KeyboardInterrupt:
                    # User interrupted - return to prompt
                    print()  # Newline after ^C
                    continue

                # Display response
                response_str = str(response) if response is not None else ""
                formatted = self.format_response(response_str)
                if formatted:
                    print(formatted)

        finally:
            stdout_capture.uninstall()
            self.console.print("\n[dim]Session ended. Goodbye![/dim]")

    def _run_pre_exit_hooks(self) -> bool:
        """Run registered pre-exit hooks before CLI exits.
        
        Returns:
            True if all hooks succeeded, False if any failed.
        """
        success = True
        if hasattr(self, '_pre_exit_hooks'):
            for hook in self._pre_exit_hooks:
                try:
                    hook()
                except Exception as e:
                    self.console.print(f"[red]Pre-exit hook error: {e}[/red]")
                    success = False
        return success

    def register_pre_exit_hook(self, hook) -> None:
        """
        Register a hook to run before CLI exits.

        Hooks are called in registration order. Exceptions are caught
        and printed but don't prevent other hooks from running.

        Args:
            hook: Callable with no arguments
        """
        if not hasattr(self, '_pre_exit_hooks'):
            self._pre_exit_hooks = []
        self._pre_exit_hooks.append(hook)

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

    # === PATCH APPROVAL UI ===

    def _cli_prompt_patch_approval(
        self,
        preview_text: str,
        preamble: str = "",
        postamble: str = ""
    ) -> tuple:
        """
        Interactive patch approval prompt for CLI.

        Displays the preview and prompts user with options:
        - [Y]es: Apply the patch
        - [N]o: Reject the patch (prompts for comments)
        - [A]lways: Apply and disable future previews

        Returns:
            Tuple of (approved, comments, disable_future_preview)
        """
        print()  # Blank line before preview

        # Show preamble if provided
        if preamble:
            self.console.print(parse_markup(preamble))
            print()

        # Show preview in a panel
        self.console.panel(preview_text, title="Patch Preview", border_style="yellow")

        # Show postamble if provided
        if postamble:
            print()
            self.console.print(parse_markup(postamble))

        # Prompt for approval
        print()
        self.console.print("[bold]Apply this patch?[/bold]")
        self.console.print("[dim][Y]es / [N]o / [A]lways (yes, don't ask again)[/dim]")

        while True:
            try:
                response = input("> ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return False, "User cancelled", False

            if response in ('y', 'yes', ''):
                return True, "", False
            elif response in ('n', 'no'):
                # Ask for optional comments
                self.console.print("[dim]Comments (optional, press Enter to skip):[/dim]")
                try:
                    comments = input("> ").strip()
                except (KeyboardInterrupt, EOFError):
                    comments = ""
                return False, comments, False
            elif response in ('a', 'always'):
                self.console.print("[dim]Future patches will be auto-applied without preview.[/dim]")
                return True, "", True
            else:
                self.console.print("[red]Please enter Y, N, or A[/red]")
