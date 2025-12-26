"""
Ready-to-use agents built on agentlib.

This package contains production-ready agents that can be used directly
or extended for custom use cases.
"""

from .code_agent import CodeAgent, CodeAgentBase

__all__ = [
    "CodeAgent",
    "CodeAgentBase",
]
