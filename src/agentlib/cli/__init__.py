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

For more control, use CLIMixin directly:
    from agentlib import BaseAgent, SubREPLResponseMixin
    from agentlib.cli import CLIMixin

    class MyAssistant(CLIMixin, SubREPLResponseMixin, BaseAgent):
        model = 'anthropic/claude-sonnet-4-5'
        system = "You are helpful."

    with MyAssistant() as agent:
        agent.cli_run()
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
from ..repl_agent import SubREPLResponseMixin


class CLIAgent(CLIMixin, SubREPLResponseMixin, BaseAgent):
    """
    Pre-composed CLI agent with Python REPL and response capabilities.

    This is a convenience class that combines CLIMixin, SubREPLResponseMixin,
    and BaseAgent. For most CLI applications, you just need to subclass this
    and set a few attributes.

    Class Attributes:
        model: LLM model to use (e.g., 'anthropic/claude-sonnet-4-5')
        system: System prompt for the agent
        welcome_message: Message displayed when CLI starts (supports [bold], [dim], etc.)
        cli_prompt: Input prompt string (default: "> ")
        history_db: Path to SQLite history file (default: ~/.agentlib_cli_history.db)
        max_turns: Maximum agent turns per user message (default: 20)
        repl_timeout: Timeout for Python execution (default: 30.0)

    Example:
        from agentlib.cli import CLIAgent

        class MyAssistant(CLIAgent):
            model = 'anthropic/claude-sonnet-4-5'
            system = "You are a helpful Python assistant."
            welcome_message = "[bold]Python Helper[/bold]\\nI can run Python code for you."
            history_db = "~/.myapp_history.db"

        if __name__ == "__main__":
            MyAssistant.main()

    For MCP integration, add REPLMCPMixin:
        from agentlib import REPLMCPMixin
        from agentlib.cli import CLIAgent

        class MCPAssistant(REPLMCPMixin, CLIAgent):
            model = 'anthropic/claude-sonnet-4-5'
            system = "You are helpful."
            repl_mcp_servers = [
                ('fs', '/path/to/mcp-server'),
            ]

        MCPAssistant.main()
    """

    # Default configuration - override in subclasses
    model: str = 'anthropic/claude-sonnet-4-5'
    system: str = "You are a helpful assistant with Python execution capabilities."
    welcome_message: str = (
        "[bold]CLI Assistant[/bold]\n"
        "I can execute Python code and help with various tasks."
    )

    @BaseAgent.tool
    def submit_response(self, response: str = "Your response to the user in markdown format"):
        """
        Send your final response to the user. Use markdown formatting.
        Note: python_execute_response may be more efficient for computed results.
        """
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
