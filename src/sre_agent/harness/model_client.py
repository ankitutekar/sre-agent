"""The interface between the loop and a model, and the errors a model call can raise."""

from collections.abc import Sequence
from typing import Protocol

from sre_agent.harness.messages import Message, ModelTurn, ToolSpec


class ModelError(Exception):
    """A model call failed. The run stops; it is never retried silently."""


class MissingAPIKeyError(ModelError):
    """The API key is not set in the environment."""


class ModelHTTPError(ModelError):
    """The model API answered with an HTTP error (401 auth, 402 key limit, 429 rate limit...)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code


class ModelResponseError(ModelError):
    """The response could not be used: wrong shape, an error inside it, or missing usage."""


class PinViolationError(ModelError):
    """The response came from a model or provider other than the one pinned."""


class ModelClient(Protocol):
    """Anything that can send a conversation to a model and return its reply.

    A Protocol is structural: a class counts as a ModelClient if it has this
    method, without inheriting from anything. The OpenRouter backend is one; a
    second backend (ADR 0008 fallback) or a test fake would be others.
    """

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        max_tokens: int,
    ) -> ModelTurn:
        """Return the model's next turn, or raise ModelError."""
        ...
