"""Test doubles for the harness: a scripted model and two toy tools.

The toy tools are the ones the step 1.3 smoke test uses: get_service_status and
get_owner. Data is made up.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, JsonValue

from sre_agent.harness import (
    LocalToolClient,
    Message,
    ModelError,
    ModelTurn,
    Tool,
    ToolArgs,
    ToolCall,
    ToolError,
    ToolSpec,
    Usage,
)

# --- Toy tools ---

_STATUS = {"payments-api": "degraded", "orders-api": "healthy"}
_OWNERS = {"payments-api": "team-payments", "orders-api": "team-orders"}


class ServiceArgs(ToolArgs):
    service: str


def _status(args: ServiceArgs) -> str:
    if args.service not in _STATUS:
        raise ToolError(f"no such service {args.service!r}")
    return _STATUS[args.service]


def _owner(args: ServiceArgs) -> str:
    if args.service not in _OWNERS:
        raise ToolError(f"no such service {args.service!r}")
    return _OWNERS[args.service]


get_service_status = Tool(
    name="get_service_status",
    description="Current health of a service: healthy, degraded or down.",
    args_model=ServiceArgs,
    fn=_status,
)
get_owner = Tool(
    name="get_owner",
    description="The team that owns a service.",
    args_model=ServiceArgs,
    fn=_owner,
)


def toy_tools() -> LocalToolClient:
    return LocalToolClient([get_service_status, get_owner])


class Triage(BaseModel):
    """The toy agent's final answer."""

    service: str
    status: str
    owner: str


# --- Scripted model ---


def turn(
    *,
    calls: Sequence[tuple[str, str]] = (),
    text: str | None = None,
    finish: str | None = None,
    native: str | None = None,
    refusal: str | None = None,
    tokens: int = 100,
) -> ModelTurn:
    """A model reply. `calls` are (name, JSON arguments); ids are made up."""
    tool_calls = tuple(
        ToolCall(id=f"call_{i}_{name}", name=name, arguments=args)
        for i, (name, args) in enumerate(calls)
    )
    if finish is None:
        finish = "tool_calls" if tool_calls else "stop"
    raw: dict[str, JsonValue] = {"role": "assistant", "content": text}
    return ModelTurn(
        text=text,
        refusal=refusal,
        tool_calls=tool_calls,
        finish_reason=finish,
        native_finish_reason=native,
        usage=Usage(prompt_tokens=tokens // 2, completion_tokens=tokens - tokens // 2, cost=0.01),
        model="test/model",
        provider="Test",
        raw_message=raw,
    )


def final(service: str = "payments-api", status: str = "degraded", owner: str = "team-payments"):
    args = Triage(service=service, status=status, owner=owner).model_dump_json()
    return turn(calls=[("final_answer", args)])


@dataclass
class ScriptedModel:
    """Returns the scripted replies in order, and records what it was sent."""

    replies: list[ModelTurn | ModelError]
    seen: list[tuple[tuple[Message, ...], tuple[ToolSpec, ...], int]] = field(
        default_factory=lambda: []
    )

    def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, max_tokens: int
    ) -> ModelTurn:
        self.seen.append((tuple(messages), tuple(tools), max_tokens))
        if not self.replies:
            raise AssertionError("the loop called the model more times than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, ModelError):
            raise reply
        return reply
