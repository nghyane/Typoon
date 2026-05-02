"""ConversationBuffer — manages message history for multi-turn LLM calls."""

from __future__ import annotations

from .ir import Message


class ConversationBuffer:
    """Accumulates messages for a multi-turn conversation.

    Ensures assistant responses and user follow-ups are appended correctly
    so the model sees its own prior output on each retry.
    """

    __slots__ = ("_messages",)

    def __init__(self, system: str, first_user: str) -> None:
        self._messages: list[Message] = [
            Message.system(system),
            Message.user_text(first_user),
        ]

    def append_assistant(self, text: str) -> None:
        self._messages.append(Message.assistant(text=text or "…"))

    def append_user(self, text: str) -> None:
        self._messages.append(Message.user_text(text))

    def messages(self) -> list[Message]:
        return list(self._messages)
