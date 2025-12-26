"""
Tools subpackage - Heavy implementations for mixins.

This package contains the core tool implementations:
- SubShell: Persistent bash shell subprocess
- SubREPL: Persistent Python REPL subprocess
- MCP protocol: Model Context Protocol client implementation
- apply_patch: File patching utilities
"""

from .subshell import SubShell, STILL_RUNNING
from .subrepl import SubREPL
from .mcp import (
    create_stdio_client,
    create_sse_client,
    MCPError,
    MCPClient,
)
from .apply_patch import (
    text_to_patch,
    patch_to_commit,
    apply_commit,
    preview_patch,
    identify_files_needed,
    identify_files_added,
    generate_unified_diff,
    print_summary,
    DiffError,
    ParseError,
    ContextNotFoundError,
)

__all__ = [
    # SubShell
    "SubShell",
    "STILL_RUNNING",
    # SubREPL
    "SubREPL",
    # MCP
    "create_stdio_client",
    "create_sse_client",
    "MCPError",
    "MCPClient",
    # apply_patch
    "text_to_patch",
    "patch_to_commit",
    "apply_commit",
    "preview_patch",
    "identify_files_needed",
    "identify_files_added",
    "generate_unified_diff",
    "print_summary",
    "DiffError",
    "ParseError",
    "ContextNotFoundError",
]
