"""Trace hooks: the loop reports every model call and tool call here.

By default nothing happens. Tests record the events; step 1.5 turns them into
OpenTelemetry spans. Events never contain the API key (the loop never sees it).
"""

from dataclasses import dataclass, field
from typing import Protocol

from sre_agent.harness.messages import ModelTurn, ToolCall, ToolResult


@dataclass(frozen=True)
class ModelCallEvent:
    turn: int
    duration_s: float
    reply: ModelTurn | None  # None if the call failed
    error: str | None = None


@dataclass(frozen=True)
class ToolCallEvent:
    turn: int
    duration_s: float
    call: ToolCall
    result: ToolResult


class TraceHook(Protocol):
    def on_model_call(self, event: ModelCallEvent) -> None: ...

    def on_tool_call(self, event: ToolCallEvent) -> None: ...


class NoOpTrace:
    def on_model_call(self, event: ModelCallEvent) -> None:
        pass

    def on_tool_call(self, event: ToolCallEvent) -> None:
        pass


@dataclass
class RecordingTrace:
    """Keeps every event in order, for tests."""

    events: list[ModelCallEvent | ToolCallEvent] = field(default_factory=lambda: [])

    def on_model_call(self, event: ModelCallEvent) -> None:
        self.events.append(event)

    def on_tool_call(self, event: ToolCallEvent) -> None:
        self.events.append(event)
