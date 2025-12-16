#!/usr/bin/env python3
"""
CLI Assistant Demo - Shows how to build interactive CLI agents with minimal code.

This example demonstrates:
1. Using CLIAgent for a batteries-included CLI experience
2. Customizing the welcome message and prompt
3. Using the Python REPL capabilities
4. Custom tool call display hooks

Run with:
    python examples/cli_assistant.py

Or create your own minimal assistant:
    from agentlib.cli import CLIAgent

    class MyBot(CLIAgent):
        model = 'google/gemini-2.5-flash'
        system = "You are helpful."

    MyBot.main()
"""

from agentlib.cli import CLIAgent


class PythonHelper(CLIAgent):
    """A Python-focused assistant with a custom welcome message."""

    model = 'google/gemini-2.5-flash'
    system = """You are a Python programming assistant. You can:
- Execute Python code to answer questions
- Help debug code
- Explain programming concepts
- Perform calculations

Use python_execute to run code and show results.
Use python_execute_response when you want to compute something and return it directly.
Use submit_response for text-only answers that don't need code execution."""

    welcome_message = (
        "[bold]Python Helper[/bold]\n"
        "I can run Python code, do calculations, and help with programming.\n"
        "[dim]Try: 'Calculate the first 10 Fibonacci numbers'[/dim]"
    )

    cli_prompt = ">>> "
    history_db = "~/.python_helper_history.db"


class VerboseAssistant(CLIAgent):
    """An assistant that shows detailed tool execution."""

    model = 'google/gemini-2.5-flash'
    system = "You are a helpful assistant. Use Python to solve problems when appropriate."

    welcome_message = (
        "[bold]Verbose Assistant[/bold]\n"
        "This example shows custom tool call hooks."
    )

    def on_tool_call(self, name, args):
        """Custom hook that shows more detail about tool calls."""
        self.console.clear_line()
        self.console.print(f"[yellow]Tool: {name}[/yellow]")

        if name in ('python_execute', 'python_execute_response'):
            code = args.get('code', '')
            # Show line count
            lines = code.strip().split('\n')
            self.console.print(f"[dim]Executing {len(lines)} line(s) of Python...[/dim]")
            self.console.panel(code, title="Code", border_style="blue")
        elif name == 'submit_response':
            self.console.print("[dim]Preparing response...[/dim]")
        else:
            # Show all args for other tools
            for k, v in args.items():
                self.console.print(f"[dim]  {k}: {v}[/dim]")

    def on_tool_result(self, name, result):
        """Custom hook that shows result summaries."""
        if result is None:
            return

        result_str = str(result).strip()
        if not result_str:
            return

        # Show character count for long results
        if len(result_str) > 500:
            self.console.print(f"[dim]Result: {len(result_str)} characters[/dim]")
            self.console.panel(result_str[:500] + "...", title="Output (truncated)", border_style="green")
        else:
            self.console.panel(result_str, title="Output", border_style="green")


def main():
    """Run the demo - defaults to PythonHelper."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--verbose':
        print("Running VerboseAssistant demo...")
        VerboseAssistant.main()
    else:
        print("Running PythonHelper demo...")
        print("(Use --verbose for the VerboseAssistant demo)\n")
        PythonHelper.main()


if __name__ == "__main__":
    main()
