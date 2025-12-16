"""
SubREPLMixin - Mixin that adds Python REPL execution to any agent.

Example:
    from agentlib import BaseAgent, SubREPLMixin

    class MyAgent(SubREPLMixin, BaseAgent):
        model = 'google/gemini-2.5-flash'
        system = "You are a helpful assistant."

        @BaseAgent.tool
        def done(self, response: str = "Your response"):
            self.respond(response)

    with MyAgent() as agent:
        result = agent.run("Calculate the sum of squares from 1 to 100")

Notes:
    - Python state (variables, imports, functions) persists across calls.
    - For long-running code, use python_read() to continue reading output.
    - Use python_interrupt() to stop running code.
    - Call close() when done, or use as context manager.
"""

import threading
from pydantic import create_model, Field
from typing import Optional, Any

from .subrepl import SubREPL, STILL_RUNNING


class SubREPLMixin:
    """Mixin that adds Python REPL execution. Use with BaseAgent."""

    # Configuration
    repl_echo: bool = False  # Echo statements in output (default False for agent use)
    repl_timeout: float = 30.0  # Default timeout for REPL operations

    # === HOOK IMPLEMENTATIONS ===

    def _ensure_setup(self):
        # Chain to next in MRO (might be another mixin or BaseAgent)
        if hasattr(super(), '_ensure_setup'):
            super()._ensure_setup()

        # Fast path - already initialized
        if getattr(self, '_repl_initialized', False):
            return

        # Thread-safe initialization
        if not hasattr(self, '_repl_lock'):
            self._repl_lock = threading.Lock()

        with self._repl_lock:
            if getattr(self, '_repl_initialized', False):
                return

            self._repl: Optional[SubREPL] = None
            self._repl_initialized = True

    def _get_repl(self) -> SubREPL:
        """Lazily create REPL on first use."""
        if self._repl is None:
            self._repl = SubREPL(echo=getattr(self, 'repl_echo', False))
        return self._repl

    def _build_system_prompt(self):
        # Get base system prompt from chain
        if hasattr(super(), '_build_system_prompt'):
            system = super()._build_system_prompt()
        else:
            system = getattr(self, 'system', '')

        # Add REPL instructions
        repl_instructions = """
PYTHON EXECUTION:
You have access to a persistent Python REPL via these tools:
- python_execute: Run Python code. State (variables, imports, functions) persists.
- python_read: Continue reading output if code is still running.
- python_interrupt: Stop running code (sends SIGINT/Ctrl+C).

When output ends with "[still running]", the code hasn't finished.
Use python_read() to get more output or python_interrupt() to stop it.

You can define functions and classes that persist across calls.
Import statements persist - no need to re-import modules."""

        return system + repl_instructions

    def _get_dynamic_toolspecs(self):
        # Get specs from chain first
        if hasattr(super(), '_get_dynamic_toolspecs'):
            specs = super()._get_dynamic_toolspecs()
        else:
            specs = {}

        # Create Pydantic models for tool specs
        specs['python_execute'] = create_model(
            'PythonExecute',
            code=(str, Field(..., description="Python code to execute")),
            timeout=(float, Field(30.0, description="Max seconds to wait for output")),
            __doc__="Execute Python code in a persistent REPL. Variables, imports, and function definitions persist across calls."
        )

        specs['python_read'] = create_model(
            'PythonRead',
            timeout=(float, Field(30.0, description="Max seconds to wait for more output")),
            __doc__="Continue reading output from long-running Python code. Use when previous output ended with '[still running]'."
        )

        specs['python_interrupt'] = create_model(
            'PythonInterrupt',
            __doc__="Interrupt currently running Python code with SIGINT (like Ctrl+C)."
        )

        return specs

    def _handle_toolcall(self, toolname, function_args):
        if toolname == 'python_execute':
            code = function_args.get('code', '')
            timeout = function_args.get('timeout', getattr(self, 'repl_timeout', 30.0))
            result = self._get_repl().execute(code, timeout=timeout)
            return True, result

        elif toolname == 'python_read':
            timeout = function_args.get('timeout', getattr(self, 'repl_timeout', 30.0))
            result = self._get_repl().read(timeout=timeout)
            return True, result if result else "[No output available]"

        elif toolname == 'python_interrupt':
            result = self._get_repl().interrupt()
            return True, result if result else "[Code interrupted]"

        # Pass to next in chain
        if hasattr(super(), '_handle_toolcall'):
            return super()._handle_toolcall(toolname, function_args)
        return False, None

    def _cleanup(self):
        # Clean up REPL
        if hasattr(self, '_repl') and self._repl is not None:
            try:
                self._repl.close()
            except Exception:
                pass
            self._repl = None

        if hasattr(self, '_repl_initialized'):
            self._repl_initialized = False

        # Chain to next
        if hasattr(super(), '_cleanup'):
            super()._cleanup()

    # === PUBLIC METHODS ===

    def python_execute(self, code: str, timeout: Optional[float] = None) -> str:
        """
        Execute Python code directly (without going through the agent).

        Args:
            code: Python code to execute
            timeout: Max seconds to wait (default: repl_timeout)

        Returns:
            Code output. Ends with "[still running]\\n" if not complete.
        """
        self._ensure_setup()
        t = timeout if timeout is not None else getattr(self, 'repl_timeout', 30.0)
        return self._get_repl().execute(code, timeout=t)

    def python_read(self, timeout: Optional[float] = None) -> str:
        """
        Continue reading output from long-running code.

        Args:
            timeout: Max seconds to wait (default: repl_timeout)

        Returns:
            More output. Ends with "[still running]\\n" if still running.
        """
        self._ensure_setup()
        t = timeout if timeout is not None else getattr(self, 'repl_timeout', 30.0)
        return self._get_repl().read(timeout=t)

    def python_interrupt(self) -> str:
        """
        Interrupt currently running Python code.

        Returns:
            Any remaining output after interrupt.
        """
        self._ensure_setup()
        return self._get_repl().interrupt()


class SubREPLResponseMixin(SubREPLMixin):
    """
    Extended REPL mixin that adds python_execute_response tool.

    Use this when you want agents to be able to execute code and return
    the output directly as their response.

    Example:
        class CalcAgent(SubREPLResponseMixin, BaseAgent):
            model = 'google/gemini-2.5-flash'
            system = "You are a calculator. Use python_execute_response to compute and return results."
    """

    def _build_system_prompt(self):
        system = super()._build_system_prompt()
        # Add info about the response tool
        system += "\n- python_execute_response: Execute code and return output as your final response."
        return system

    def _get_dynamic_toolspecs(self):
        specs = super()._get_dynamic_toolspecs()

        specs['python_execute_response'] = create_model(
            'PythonExecuteResponse',
            code=(str, Field(..., description="Python code to execute")),
            preamble=(str, Field("", description="Optional text before the output")),
            postamble=(str, Field("", description="Optional text after the output")),
            __doc__="Execute Python code and return the output as your final response to the user."
        )

        return specs

    def _handle_toolcall(self, toolname, function_args):
        if toolname == 'python_execute_response':
            code = function_args.get('code', '')
            preamble = function_args.get('preamble', '')
            postamble = function_args.get('postamble', '')
            timeout = getattr(self, 'repl_timeout', 30.0)

            output = self._get_repl().execute(code, timeout=timeout)

            if output.endswith(STILL_RUNNING):
                # Timed out - return partial with note
                clean = output[:-len(STILL_RUNNING)].strip()
                self.respond(f"```\n{clean}\n```\n\n*(execution timed out)*")
            else:
                # Format response with optional preamble/postamble
                parts = []
                if preamble:
                    parts.append(preamble)
                parts.append(f"```\n{output.strip()}\n```")
                if postamble:
                    parts.append(postamble)
                self.respond("\n\n".join(parts))

            return True, None  # respond() handles the response

        return super()._handle_toolcall(toolname, function_args)
