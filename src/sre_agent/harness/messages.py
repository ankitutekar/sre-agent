"""Typed records for one agent run: the conversation, and what the model returned.

Every record is frozen (read-only after creation), so the loop cannot quietly edit
history. Claude 5.5 rejects edited history, and an audit trail is only useful if
nobody rewrites it (ADR 0008).
"""

from pydantic import BaseModel, ConfigDict, JsonValue


class _Record(BaseModel):
    """Base for the records below: read-only, and unknown fields are a bug."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SystemMessage(_Record):
    """The agent's standing instructions."""

    text: str


class UserMessage(_Record):
    """The task given to the agent."""

    text: str


class ToolCall(_Record):
    """One tool call the model asked for."""

    id: str
    name: str
    arguments: str  # JSON text, exactly as the model wrote it; validated by the tool


class ToolResult(_Record):
    """What goes back to the model for one tool call.

    `content` is exactly the text the model will see, errors included (Chat
    Completions has no separate error flag, so error text says so itself).
    """

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class ToolSpec(_Record):
    """A tool as the model sees it: name, description and JSON Schema for its arguments."""

    name: str
    description: str
    parameters: dict[str, JsonValue]


class Usage(_Record):
    """Token counts and cost for one model call, or summed over a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float | None = None  # in OpenRouter credits (US dollars); None if not reported

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        if self.cost is None and other.cost is None:
            cost = None
        else:
            cost = (self.cost or 0.0) + (other.cost or 0.0)
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost=cost,
        )


class ModelTurn(_Record):
    """One reply from the model, in two forms.

    The typed fields are what the loop reasons over. `raw_message` is the assistant
    message exactly as the provider sent it (including reasoning details); the
    backend sends it back unchanged on later turns. Nothing may modify it.
    """

    text: str | None
    refusal: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None
    native_finish_reason: str | None
    usage: Usage
    model: str
    provider: str | None
    raw_message: dict[str, JsonValue]


# Anything that can appear in a conversation. `type X = ...` is Python 3.12's
# syntax for a named type alias.
type Message = SystemMessage | UserMessage | ModelTurn | ToolResult
