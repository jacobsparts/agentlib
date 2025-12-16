"""
CLI module for agentlib - composable CLI functionality for agents.

This module provides terminal rendering utilities and a CLIMixin that adds
interactive CLI capabilities to any agent.

Quick Start:
    from agentlib.cli import CLIAgent

    class MyAssistant(CLIAgent):
        model = 'anthropic/claude-sonnet-4-5'
        system = "You are a helpful assistant."
        welcome_message = "Hello! How can I help?"

    if __name__ == "__main__":
        MyAssistant.main()

With Python execution:
    from agentlib import SubREPLResponseMixin
    from agentlib.cli import CLIAgent

    class CodeAssistant(SubREPLResponseMixin, CLIAgent):
        model = 'anthropic/claude-sonnet-4-5'
        system = "You are a Python assistant."

    CodeAssistant.main()
"""

from .terminal import (
    # ANSI codes
    RESET, BOLD, DIM, ITALIC, UNDERLINE, STRIKE,
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, GRAY,
    # Functions
    parse_markup, strip_ansi, get_terminal_width, render_markdown,
    highlight_python,
    # Classes
    Panel, Markdown, Console,
    DEFAULT_THEME,
)

from .mixin import (
    CLIMixin,
    SQLiteHistory,
    InputSession,
)

# Import base classes for CLIAgent
from ..core import BaseAgent


class CLIAgent(CLIMixin, BaseAgent):
    """
    Base CLI agent with interactive terminal capabilities.

    This is a convenience class that combines CLIMixin and BaseAgent. For most
    CLI applications, subclass this and add mixins for the capabilities you need.

    Class Attributes:
        model: LLM model to use (e.g., 'anthropic/claude-sonnet-4-5')
        system: System prompt for the agent
        welcome_message: Message displayed when CLI starts (supports [bold], [dim], etc.)
        cli_prompt: Input prompt string (default: "> ")
        history_db: Path to SQLite history file (default: ~/.agentlib_cli_history.db)
        max_turns: Maximum agent turns per user message (default: 20)

    Example:
        from agentlib.cli import CLIAgent

        class MyAssistant(CLIAgent):
            model = 'anthropic/claude-sonnet-4-5'
            system = "You are a helpful assistant."
            welcome_message = "[bold]My Assistant[/bold]"

        if __name__ == "__main__":
            MyAssistant.main()

    With Python REPL:
        from agentlib import SubREPLResponseMixin
        from agentlib.cli import CLIAgent

        class CodeAssistant(SubREPLResponseMixin, CLIAgent):
            model = 'anthropic/claude-sonnet-4-5'
            system = "You are a Python assistant."

    With MCP servers:
        from agentlib import REPLMCPMixin, SubREPLResponseMixin
        from agentlib.cli import CLIAgent

        class MCPAssistant(REPLMCPMixin, SubREPLResponseMixin, CLIAgent):
            model = 'anthropic/claude-sonnet-4-5'
            system = "You are helpful."
            repl_mcp_servers = [('fs', '/path/to/mcp-server')]
    """

    # Default configuration - override in subclasses
    model: str = 'anthropic/claude-sonnet-4-5'
    system: str = "You are a helpful assistant."
    welcome_message: str = "[bold]CLI Assistant[/bold]"

    @BaseAgent.tool
    def submit_response(self, response: str = "Your response to the user in markdown format"):
        """Send your final response to the user. Use markdown formatting."""
        self.respond(response)


__all__ = [
    # Terminal utilities
    'RESET', 'BOLD', 'DIM', 'ITALIC', 'UNDERLINE', 'STRIKE',
    'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE', 'GRAY',
    'parse_markup', 'strip_ansi', 'get_terminal_width', 'render_markdown',
    'highlight_python',
    'Panel', 'Markdown', 'Console',
    'DEFAULT_THEME',
    # Mixin and history
    'CLIMixin',
    'SQLiteHistory',
    'InputSession',
    # Pre-composed agent
    'CLIAgent',
]
