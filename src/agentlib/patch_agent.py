"""
FilePatchMixin - Mixin that adds file patching capabilities to any agent.

Example:
    from agentlib import BaseAgent, FilePatchMixin

    class MyAgent(FilePatchMixin, BaseAgent):
        model = 'google/gemini-2.5-flash'
        system = "You are a helpful coding assistant."
        patch_preview = None  # Agent decides (default)

        @BaseAgent.tool
        def done(self, response: str = "Your response"):
            self.respond(response)

    with MyAgent() as agent:
        result = agent.run("Add a hello() function to main.py")

Configuration:
    - patch_preview=True: Always require user approval
    - patch_preview=False: Auto-apply without approval
    - patch_preview=None: Agent decides per-call (default preview=True)

Customization:
    - Override _prompt_patch_approval() for custom approval UI
    - CLIMixin automatically provides interactive approval when combined
"""

import os
import sys
import threading
from typing import Optional, Tuple, Dict, Any

from pydantic import create_model, Field

# Import apply_patch utilities from local module
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
    APPLY_PATCH_INSTRUCTIONS,
)


# Condensed patch format instructions for system prompt
PATCH_FORMAT_INSTRUCTIONS = """
FILE PATCHING:
You can add, update, and delete files using the apply_patch tool.

PATCH FORMAT:
```
*** Begin Patch
*** Add File: path/to/new_file.py
+line 1
+line 2
*** Update File: path/to/existing.py
@@ optional context marker (e.g., function/class name)
 context line (unchanged)
-line to remove
+line to add
 context line
*** Delete File: path/to/remove.py
*** End Patch
```

RULES:
- Wrap patches in *** Begin Patch / *** End Patch
- Add File: prefix all content lines with +
- Update File: use @@ for hunks, space for context, - for removals, + for additions
- Delete File: just the header, no content
- Show ~3 lines of context around changes
- Use @@ with function/class names if context isn't unique
- Use *** End of File to anchor changes at file end
"""

AGENT_PREVIEW_INSTRUCTIONS = """
PREVIEW BEHAVIOR:
- Set preview=True by default to let users review changes before applying
- Users can request auto-apply mode if they prefer not to review
"""


def _absolutize_patch(text: str, base_path: str) -> str:
    """Convert relative paths in patch text to absolute paths."""
    lines = text.splitlines()
    result = []
    for line in lines:
        for prefix in ('*** Update File: ', '*** Delete File: ', '*** Add File: ', '*** Move to: '):
            if line.startswith(prefix):
                path = line[len(prefix):]
                if not path.startswith('/'):
                    path = os.path.join(base_path, path)
                line = prefix + path
                break
        result.append(line)
    return '\n'.join(result)


def _relativize_paths(paths: list, base_path: str) -> list:
    """Convert absolute paths back to relative for display."""
    result = []
    for path in paths:
        if path.startswith(base_path + '/'):
            path = path[len(base_path) + 1:]
        result.append(path)
    return result


class FilePatchMixin:
    """Mixin that adds file patching capabilities. Use with BaseAgent."""

    # Configuration
    patch_preview: Optional[bool] = None  # True=always, False=never, None=agent decides

    # === HOOK IMPLEMENTATIONS ===

    def _ensure_setup(self):
        # Chain to next in MRO
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        # Fast path - already initialized
        if getattr(self, '_patch_initialized', False):
            return

        # Thread-safe initialization
        if not hasattr(self, '_patch_lock'):
            self._patch_lock = threading.Lock()

        with self._patch_lock:
            if getattr(self, '_patch_initialized', False):
                return

            self._patch_initialized = True

    def _get_patch_base_path(self) -> str:
        """Get the base path for resolving relative paths in patches."""
        # Try PWD environment variable first
        base_path = os.environ.get('PWD')
        if base_path and os.path.isdir(base_path):
            return base_path

        # Fall back to directory of main module
        try:
            main_module = sys.modules.get('__main__')
            if main_module and hasattr(main_module, '__file__') and main_module.__file__:
                return os.path.dirname(os.path.abspath(main_module.__file__))
        except Exception:
            pass

        # Last resort: current working directory
        return os.getcwd()

    def _prompt_patch_approval(
        self,
        preview_text: str,
        preamble: str = "",
        postamble: str = ""
    ) -> Tuple[bool, str, bool]:
        """
        Prompt user for patch approval.

        Override this method to provide custom approval UI.
        CLIMixin automatically overrides this when combined.

        Args:
            preview_text: Unified diff preview of changes
            preamble: Optional text to display before preview
            postamble: Optional text to display after preview

        Returns:
            Tuple of (approved: bool, comments: str, disable_future_preview: bool)
            - approved: Whether to apply the patch
            - comments: User comments (especially if rejected)
            - disable_future_preview: If True, set patch_preview=False
        """
        # Default: auto-approve
        return True, "", False

    def _build_system_prompt(self):
        # Get base system prompt from chain
        if hasattr(super(), '_build_system_prompt'):
            system = super()._build_system_prompt()
        else:
            system = getattr(self, 'system', '')

        # Add patch instructions
        system += PATCH_FORMAT_INSTRUCTIONS

        # Add preview instructions if agent decides
        if getattr(self, 'patch_preview', None) is None:
            system += AGENT_PREVIEW_INSTRUCTIONS

        return system

    def _get_dynamic_toolspecs(self):
        # Get specs from chain first
        if hasattr(super(), '_get_dynamic_toolspecs'):
            specs = super()._get_dynamic_toolspecs()
        else:
            specs = {}

        # Build tool spec based on preview mode
        preview_mode = getattr(self, 'patch_preview', None)

        if preview_mode is None:
            # Agent decides - include preview field
            specs['apply_patch'] = create_model(
                'ApplyPatch',
                patch=(str, Field(..., description="Patch text in apply_patch format")),
                preview=(bool, Field(True, description="If True, show preview and ask for approval before applying")),
                preamble=(str, Field("", description="Optional text to display before the preview")),
                postamble=(str, Field("", description="Optional text to display after the preview")),
                __doc__="Apply a patch to add, update, or delete files."
            )
        else:
            # Fixed mode - omit preview field
            specs['apply_patch'] = create_model(
                'ApplyPatch',
                patch=(str, Field(..., description="Patch text in apply_patch format")),
                preamble=(str, Field("", description="Optional text to display before the preview")),
                postamble=(str, Field("", description="Optional text to display after the preview")),
                __doc__="Apply a patch to add, update, or delete files."
            )

        return specs

    def _handle_toolcall(self, toolname: str, function_args: dict):
        if toolname == 'apply_patch':
            result = self._execute_patch(
                patch=function_args.get('patch', ''),
                preview=function_args.get('preview', True),
                preamble=function_args.get('preamble', ''),
                postamble=function_args.get('postamble', ''),
            )
            return True, result

        # Pass to next in chain
        if hasattr(super(), '_handle_toolcall'):
            return super()._handle_toolcall(toolname, function_args)
        return False, None

    def _cleanup(self):
        # Reset state
        if hasattr(self, '_patch_initialized'):
            self._patch_initialized = False

        # Chain to next
        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    # === CORE IMPLEMENTATION ===

    def _execute_patch(
        self,
        patch: str,
        preview: bool = True,
        preamble: str = "",
        postamble: str = "",
    ) -> str:
        """Execute a patch with optional preview/approval."""
        self._ensure_setup()

        # Determine effective preview mode
        preview_mode = getattr(self, 'patch_preview', None)
        if preview_mode is True:
            effective_preview = True
        elif preview_mode is False:
            effective_preview = False
        else:
            effective_preview = preview

        # Get base path and absolutize paths in patch
        base_path = self._get_patch_base_path()
        absolute_patch = _absolutize_patch(patch, base_path)

        try:
            # Validate and generate preview
            previews = preview_patch(absolute_patch)
        except ParseError as e:
            return f"[Patch Error] Parse error: {e}"
        except DiffError as e:
            return f"[Patch Error] {e}"
        except FileNotFoundError as e:
            return f"[Patch Error] File not found: {e}"
        except Exception as e:
            return f"[Patch Error] Unexpected error: {e}"

        # Build preview text
        preview_parts = []
        for path, change in previews.items():
            # Convert to relative path for display
            display_path = path
            if path.startswith(base_path + '/'):
                display_path = path[len(base_path) + 1:]

            if change.type.value == 'add':
                preview_parts.append(f"=== ADD: {display_path} ===")
                if change.new_content:
                    # Show first 50 lines of new file
                    lines = change.new_content.split('\n')
                    if len(lines) > 50:
                        preview_parts.append('\n'.join(lines[:50]))
                        preview_parts.append(f"... ({len(lines) - 50} more lines)")
                    else:
                        preview_parts.append(change.new_content)
            elif change.type.value == 'delete':
                preview_parts.append(f"=== DELETE: {display_path} ===")
            elif change.type.value == 'update':
                preview_parts.append(f"=== UPDATE: {display_path} ===")
                if change.move_path:
                    move_display = change.move_path
                    if move_display.startswith(base_path + '/'):
                        move_display = move_display[len(base_path) + 1:]
                    preview_parts.append(f"(moving to: {move_display})")
                if change.unified_diff:
                    preview_parts.append(change.unified_diff)
            preview_parts.append("")

        preview_text = '\n'.join(preview_parts)

        # Prompt for approval if preview is enabled
        if effective_preview:
            approved, comments, disable_preview = self._prompt_patch_approval(
                preview_text, preamble, postamble
            )

            # Handle "don't ask again" option
            if disable_preview:
                self.patch_preview = False

            if not approved:
                if comments:
                    return f"[Patch Rejected] User declined the patch. Comments: {comments}"
                return "[Patch Rejected] User declined the patch."

        # Apply the patch
        try:
            paths = identify_files_needed(absolute_patch)
            orig = {p: open(p).read() for p in paths}
            parsed_patch = text_to_patch(absolute_patch, orig)
            commit = patch_to_commit(parsed_patch, orig)
            affected = apply_commit(commit)

            # Build success message with relative paths
            lines = ["Patch applied successfully:"]
            for path in affected.added:
                display = path[len(base_path) + 1:] if path.startswith(base_path + '/') else path
                lines.append(f"  A {display}")
            for path in affected.modified:
                display = path[len(base_path) + 1:] if path.startswith(base_path + '/') else path
                lines.append(f"  M {display}")
            for path in affected.deleted:
                display = path[len(base_path) + 1:] if path.startswith(base_path + '/') else path
                lines.append(f"  D {display}")
            return '\n'.join(lines)

        except DiffError as e:
            return f"[Patch Error] Failed to apply: {e}"
        except Exception as e:
            return f"[Patch Error] Unexpected error during apply: {e}"

    # === PUBLIC METHODS ===

    def apply_patch(
        self,
        patch: str,
        preview: bool = True,
        preamble: str = "",
        postamble: str = "",
    ) -> str:
        """
        Apply a patch directly (without going through the agent).

        Args:
            patch: Patch text in apply_patch format
            preview: Whether to prompt for approval
            preamble: Optional text before preview
            postamble: Optional text after preview

        Returns:
            Result message (success or error)
        """
        return self._execute_patch(patch, preview, preamble, postamble)
