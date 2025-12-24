"""
ToolMixin - Base mixin for tool handling with method-based dispatch.

Define methods matching tool names to handle them directly:

    class MyMixin(ToolMixin):
        def my_tool(self, arg1, arg2=None):
            return f"Called with {arg1}"

For dynamic tools (unknown at class definition time), implement _dispatch_tool:

    class DynamicMixin(ToolMixin):
        def _dispatch_tool(self, toolname, function_args):
            if toolname in self._my_tools:
                return True, self._call_tool(toolname, function_args)
            return None  # Not handled

Lookup order (like Python attribute access):
1. Method with exact tool name
2. _dispatch_tool fallback (if defined)
3. Chain to super()._handle_toolcall
"""


class ToolMixin:
    """Base mixin for tool handling. Define methods matching tool names."""

    def _handle_toolcall(self, toolname, function_args):
        """
        Handle a tool call by name.

        Checks for a method matching the tool name, then falls back to
        _dispatch_tool if defined, then chains to super().

        Returns:
            (True, result) if handled, (False, None) otherwise.
        """
        # 1. Method override (like attribute lookup)
        method = getattr(self, toolname, None)
        if method is not None and callable(method) and not toolname.startswith('_'):
            return True, method(**function_args)

        # 2. Dynamic dispatch (like __getattr__ - optional)
        dispatch = getattr(self, '_dispatch_tool', None)
        if dispatch is not None:
            result = dispatch(toolname, function_args)
            if result is not None:
                return result

        # 3. Chain to next in MRO
        if hasattr(super(), '_handle_toolcall'):
            return super()._handle_toolcall(toolname, function_args)
        return False, None
