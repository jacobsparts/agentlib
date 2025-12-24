from .core import BaseAgent
from .client import ValidationError, BadRequestError
from .llm_registry import register_provider, register_model
from .tool_mixin import ToolMixin
from .mcp_mixin import MCPMixin
from .shell_mixin import SubShellMixin
from .repl_mixin import SubREPLMixin, SubREPLResponseMixin, REPLMCPMixin
from .patch_agent import FilePatchMixin
from .subshell import SubShell, STILL_RUNNING
from .subrepl import SubREPL
from .cli import CLIMixin, CLIAgent
from .repl_agent import REPLAgent

__all__ = [
    "BaseAgent",
    "ToolMixin",
    "MCPMixin",
    "REPLMCPMixin",
    "SubShellMixin",
    "SubREPLMixin",
    "SubREPLResponseMixin",
    "FilePatchMixin",
    "SubShell",
    "SubREPL",
    "STILL_RUNNING",
    "ValidationError",
    "BadRequestError",
    "register_provider",
    "register_model",
    # CLI
    "CLIMixin",
    "CLIAgent",
    # REPL-based agent
    "REPLAgent",
]

__version__ = "0.3.0"
