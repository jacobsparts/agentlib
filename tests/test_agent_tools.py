import pytest

from agentlib import BaseAgent


class ToolSchemaAgent(BaseAgent):
    system = "test"

    @BaseAgent.tool
    def strict_tool(self, notes: str = "Notes"):
        """Strict tool."""
        return notes

    @BaseAgent.tool
    def flexible_tool(self, name: str = "Name", **kwargs):
        """Flexible tool."""
        return kwargs


def test_signature_tool_without_varargs_forbids_extra_arguments():
    spec = ToolSchemaAgent().toolspecs["strict_tool"]

    schema = spec.model_json_schema()
    assert schema["additionalProperties"] is False

    with pytest.raises(Exception):
        spec.model_validate({"notes": "ok", "reasoning": "extra"})


def test_signature_tool_with_kwargs_allows_extra_arguments():
    spec = ToolSchemaAgent().toolspecs["flexible_tool"]

    schema = spec.model_json_schema()
    assert "additionalProperties" not in schema

    validated = spec.model_validate({"name": "ok", "extra": "allowed"})
    assert validated.model_dump() == {"name": "ok"}
