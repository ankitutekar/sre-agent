"""Test doubles for the harness: a scripted model, plus the toy agent re-exported.

The toy agent (tools, task, answer model) lives in scripts/toy_agent.py so the live
smoke test and the offline tests share one definition.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from sre_agent.harness import Message, ModelError, ModelTurn, ToolCall, ToolSpec, Usage
from toy_agent import ServiceArgs, Triage, get_owner, get_service_status, toy_tools

__all__ = [
    "ScriptedModel",
    "ServiceArgs",
    "Triage",
    "final",
    "get_owner",
    "get_service_status",
    "toy_tools",
    "turn",
]


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
        wire="chat_completions",
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
