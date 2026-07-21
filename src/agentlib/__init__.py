import sys
from pathlib import Path

_user_mixins = Path.home() / ".agentlib" / "mixins"
if _user_mixins.is_dir() and str(_user_mixins) not in sys.path:
    sys.path.append(str(_user_mixins))

from .core import BaseAgent
from .client import ValidationError, BadRequestError
from .llm_registry import register_provider, register_model, ModelNotFoundError
from .tool_mixin import ToolMixin
from .mcp_mixin import MCPMixin
from .shell_mixin import SubShellMixin
from .python_tool_mixin import PythonToolMixin, PythonToolResponseMixin, PythonMCPMixin
from .patch_mixin import FilePatchMixin
from .attachment_mixin import AttachmentMixin
from .tools.subshell import SubShell, STILL_RUNNING
from .tools.subrepl import SubREPL
from .cli import CLIMixin, CLIAgent
from .repl_agent import REPLAgent
from .repl_events import ReplEvent
__all__ = [
    "BaseAgent",
    "ToolMixin",
    "MCPMixin",
    "PythonMCPMixin",
    "SubShellMixin",
    "PythonToolMixin",
    "PythonToolResponseMixin",
    "FilePatchMixin",
    "AttachmentMixin",
    "SubShell",
    "SubREPL",
    "STILL_RUNNING",
    "ValidationError",
    "BadRequestError",
    "ModelNotFoundError",
    "register_provider",
    "register_model",
    # CLI
    "CLIMixin",
    "CLIAgent",
    "REPLAgent",
    "ReplEvent",
]

__version__ = "0.3.0"
