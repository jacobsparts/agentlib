"""
Ready-to-use agents built on agentlib.

This package contains production-ready agents that can be used directly
or extended for custom use cases.
"""

__all__ = [
    "CodeAgent",
    "CodeAgentBase",
]


def __getattr__(name):
    if name in {"CodeAgent", "CodeAgentBase"}:
        from .code_agent import CodeAgent, CodeAgentBase
        return {"CodeAgent": CodeAgent, "CodeAgentBase": CodeAgentBase}[name]
    raise AttributeError(name)
