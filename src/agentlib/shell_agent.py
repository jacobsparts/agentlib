"""
SubShellMixin - Mixin that adds bash shell execution to any agent.

Example:
    from agentlib import BaseAgent, SubShellMixin

    class MyAgent(SubShellMixin, BaseAgent):
        model = 'google/gemini-2.5-flash'
        system = "You are a helpful assistant."

        @BaseAgent.tool
        def done(self, response: str = "Your response"):
            self.respond(response)

    with MyAgent() as agent:
        result = agent.run("List files in /tmp")

Notes:
    - Shell state (environment, working directory) persists across calls.
    - For long-running commands, use shell_read() to continue reading output.
    - Use shell_interrupt() to stop a running command.
    - Call close() when done, or use as context manager.
"""

import threading
from pydantic import create_model, Field
from typing import Optional, Any

from .subshell import SubShell, STILL_RUNNING


# Tool schema models
class ShellExecuteParams:
    """Execute a bash command in a persistent shell session."""
    command: str = Field(..., description="The bash command to execute")
    timeout: float = Field(30.0, description="Max seconds to wait for output (default 30)")


class ShellReadParams:
    """Continue reading output from a long-running command."""
    timeout: float = Field(30.0, description="Max seconds to wait for more output (default 30)")


class ShellInterruptParams:
    """Interrupt the currently running command with SIGINT (Ctrl+C)."""
    pass


class SubShellMixin:
    """Mixin that adds bash shell execution. Use with BaseAgent."""

    # Configuration
    shell_echo: bool = False  # Echo commands in output (default False for agent use)
    shell_timeout: float = 30.0  # Default timeout for shell operations

    # === HOOK IMPLEMENTATIONS ===

    def _ensure_setup(self):
        # Chain to next in MRO (might be another mixin or BaseAgent)
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        # Fast path - already initialized
        if getattr(self, '_shell_initialized', False):
            return

        # Thread-safe initialization
        if not hasattr(self, '_shell_lock'):
            self._shell_lock = threading.Lock()

        with self._shell_lock:
            if getattr(self, '_shell_initialized', False):
                return

            self._shell: Optional[SubShell] = None
            self._shell_initialized = True

    def _get_shell(self) -> SubShell:
        """Lazily create shell on first use."""
        if self._shell is None:
            self._shell = SubShell(echo=getattr(self, 'shell_echo', False))
        return self._shell

    def _build_system_prompt(self):
        # Get base system prompt from chain
        if hasattr(super(), '_build_system_prompt'):
            system = super()._build_system_prompt()
        else:
            system = getattr(self, 'system', '')

        # Add shell instructions
        shell_instructions = """
SHELL EXECUTION:
You have access to a persistent bash shell via these tools:
- shell_execute: Run bash commands. State (env vars, cwd) persists between calls.
- shell_read: Continue reading output if a command is still running.
- shell_interrupt: Stop a running command (sends SIGINT/Ctrl+C).

When output ends with "[still running]", the command hasn't finished.
Use shell_read() to get more output or shell_interrupt() to stop it."""

        return system + shell_instructions

    def _get_dynamic_toolspecs(self):
        # Get specs from chain first
        if hasattr(super(), '_get_dynamic_toolspecs'):
            specs = super()._get_dynamic_toolspecs()
        else:
            specs = {}

        # Create Pydantic models for tool specs
        specs['shell_execute'] = create_model(
            'ShellExecute',
            command=(str, Field(..., description="The bash command to execute")),
            timeout=(float, Field(30.0, description="Max seconds to wait for output")),
            __doc__="Execute a bash command in a persistent shell. Environment and working directory persist across calls."
        )

        specs['shell_read'] = create_model(
            'ShellRead',
            timeout=(float, Field(30.0, description="Max seconds to wait for more output")),
            __doc__="Continue reading output from a long-running command. Use when previous output ended with '[still running]'."
        )

        specs['shell_interrupt'] = create_model(
            'ShellInterrupt',
            __doc__="Interrupt the currently running command with SIGINT (like Ctrl+C)."
        )

        return specs

    def _handle_toolcall(self, toolname, function_args):
        if toolname == 'shell_execute':
            command = function_args.get('command', '')
            timeout = function_args.get('timeout', getattr(self, 'shell_timeout', 30.0))
            result = self._get_shell().execute(command, timeout=timeout)
            return True, result

        elif toolname == 'shell_read':
            timeout = function_args.get('timeout', getattr(self, 'shell_timeout', 30.0))
            result = self._get_shell().read(timeout=timeout)
            return True, result if result else "[No output available]"

        elif toolname == 'shell_interrupt':
            result = self._get_shell().interrupt()
            return True, result if result else "[Command interrupted]"

        # Pass to next in chain
        if hasattr(super(), '_handle_toolcall'):
            return super()._handle_toolcall(toolname, function_args)
        return False, None

    def _cleanup(self):
        # Clean up shell
        if hasattr(self, '_shell') and self._shell is not None:
            try:
                self._shell.close()
            except Exception:
                pass
            self._shell = None

        if hasattr(self, '_shell_initialized'):
            self._shell_initialized = False

        # Chain to next
        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    # === PUBLIC METHODS ===

    def shell_execute(self, command: str, timeout: Optional[float] = None) -> str:
        """
        Execute a bash command directly (without going through the agent).

        Args:
            command: The bash command to execute
            timeout: Max seconds to wait (default: shell_timeout)

        Returns:
            Command output. Ends with "[still running]\\n" if not complete.
        """
        self._ensure_setup()
        t = timeout if timeout is not None else getattr(self, 'shell_timeout', 30.0)
        return self._get_shell().execute(command, timeout=t)

    def shell_read(self, timeout: Optional[float] = None) -> str:
        """
        Continue reading output from a long-running command.

        Args:
            timeout: Max seconds to wait (default: shell_timeout)

        Returns:
            More output. Ends with "[still running]\\n" if still running.
        """
        self._ensure_setup()
        t = timeout if timeout is not None else getattr(self, 'shell_timeout', 30.0)
        return self._get_shell().read(timeout=t)

    def shell_interrupt(self) -> str:
        """
        Interrupt the currently running command.

        Returns:
            Any remaining output after interrupt.
        """
        self._ensure_setup()
        return self._get_shell().interrupt()
