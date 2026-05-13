"""LLM infrastructure — provider-agnostic IR, agent loop, and native adapters."""

from .errors import OperatorActionRequired, UpstreamUnavailable
from .ir import ContentPart, Message, Provider, ToolCallMsg, ToolDef, ToolResponse
from .loop import tool_loop
from .tool import Tool, tool

__all__ = [
    "ContentPart", "Message", "Provider", "ToolCallMsg", "ToolDef", "ToolResponse",
    "tool_loop",
    "Tool", "tool",
    "OperatorActionRequired", "UpstreamUnavailable",
]
