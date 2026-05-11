"""LLM infrastructure — provider-agnostic IR, agent loop, and native adapters."""

from .conversation import ConversationBuffer
from .errors import TransientCredentialError, UpstreamUnavailable
from .ir import ContentPart, Message, Provider, ToolCallMsg, ToolDef, ToolResponse
from .loop import tool_loop
from .tool import Tool, tool

__all__ = [
    "ConversationBuffer",
    "ContentPart", "Message", "Provider", "ToolCallMsg", "ToolDef", "ToolResponse",
    "tool_loop",
    "Tool", "tool",
    "TransientCredentialError", "UpstreamUnavailable",
]
