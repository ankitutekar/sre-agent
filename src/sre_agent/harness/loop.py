"""The agent loop: call the model, run the tools it asks for, repeat until it answers.

Every way out of the loop is explicit and has a StopReason. Anything unexpected
stops the run (fail closed) rather than being guessed at.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ValidationError

from sre_agent.harness.messages import (
    Message,
    ModelTurn,
    SystemMessage,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    UserMessage,
)
from sre_agent.harness.model_client import ModelClient, ModelError
from sre_agent.harness.tools import ToolClient, error_result, validation_summary
from sre_agent.harness.trace import ModelCallEvent, NoOpTrace, ToolCallEvent, TraceHook

FINAL_ANSWER_TOOL = "final_answer"
_FINAL_ANSWER_DESCRIPTION = (
    "Submit your final answer. Call this once, on its own, when you are done. "
    "The run ends when a valid answer is accepted."
)


@dataclass(frozen=True)
class Limits:
    """Hard limits on one run. Each run may override them."""

    max_turns: int = 10  # model calls
    max_total_tokens: int = 50_000  # prompt + completion, summed over the run
    max_tokens_per_request: int = 4_000  # output cap sent with each request
    max_tool_result_chars: int = 16_000  # longer tool output is cut

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    TOKEN_BUDGET = "token_budget"  # noqa: S105 - a stop reason, not a password
    LENGTH = "length"
    REFUSAL = "refusal"
    NO_ANSWER = "no_answer"
    MODEL_ERROR = "model_error"


@dataclass(frozen=True)
class RunResult[T: BaseModel]:
    stop_reason: StopReason
    answer: T | None
    error: str | None
    turns: int
    usage: Usage
    conversation: tuple[Message, ...]

    @property
    def ok(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED


def run_agent[T: BaseModel](
    *,
    model: ModelClient,
    tools: ToolClient,
    system: str,
    task: str,
    answer_model: type[T],
    limits: Limits | None = None,
    trace: TraceHook | None = None,
) -> RunResult[T]:
    """Run one task to a validated `answer_model`, or to a stated reason for stopping."""
    limits = limits or Limits()
    trace = trace or NoOpTrace()
    tool_specs = list(tools.list_tools())
    if any(spec.name == FINAL_ANSWER_TOOL for spec in tool_specs):
        raise ValueError(f"{FINAL_ANSWER_TOOL!r} is reserved for the harness")
    specs = (
        *tool_specs,
        ToolSpec(
            name=FINAL_ANSWER_TOOL,
            description=_FINAL_ANSWER_DESCRIPTION,
            parameters=answer_model.model_json_schema(),
        ),
    )

    # Append-only: items are added, never changed or removed.
    conversation: list[Message] = [SystemMessage(text=system), UserMessage(text=task)]
    usage = Usage()
    turns = 0

    def stop(
        reason: StopReason, *, answer: T | None = None, error: str | None = None
    ) -> RunResult[T]:
        return RunResult(reason, answer, error, turns, usage, tuple(conversation))

    while True:
        if turns >= limits.max_turns:
            return stop(StopReason.MAX_TURNS, error=f"no final answer after {turns} turns")
        remaining = limits.max_total_tokens - usage.total_tokens
        if remaining <= 0:
            return stop(StopReason.TOKEN_BUDGET, error=f"used {usage.total_tokens} tokens")
        turns += 1

        started = time.monotonic()
        try:
            reply = model.complete(
                conversation, specs, max_tokens=min(limits.max_tokens_per_request, remaining)
            )
        except ModelError as exc:
            trace.on_model_call(
                ModelCallEvent(turns, time.monotonic() - started, None, error=str(exc))
            )
            return stop(StopReason.MODEL_ERROR, error=str(exc))
        trace.on_model_call(ModelCallEvent(turns, time.monotonic() - started, reply))
        conversation.append(reply)
        usage = usage + reply.usage

        early = _early_stop(reply)
        if early is not None:
            reason, error = early
            return stop(reason, error=error)

        results, answer = _handle_tool_calls(
            reply.tool_calls, tools, answer_model, specs, limits, trace, turns
        )
        conversation.extend(results)  # all results of a turn go back together
        if answer is not None:
            return stop(StopReason.COMPLETED, answer=answer)


def _early_stop(reply: ModelTurn) -> tuple[StopReason, str] | None:
    """Why this reply ends the run, or None if it asks for tools and the loop goes on."""
    native = reply.native_finish_reason
    if reply.refusal is not None or native == "refusal" or reply.finish_reason == "content_filter":
        detail = reply.refusal or f"finish_reason={reply.finish_reason!r}, native={native!r}"
        return StopReason.REFUSAL, f"the model refused: {detail}"
    if reply.finish_reason == "length":
        return StopReason.LENGTH, "the model hit max_tokens before finishing"
    if reply.finish_reason not in ("tool_calls", "stop"):
        return StopReason.MODEL_ERROR, (
            f"unhandled finish_reason {reply.finish_reason!r} (native {native!r})"
        )
    if not reply.tool_calls:
        visible = "with text" if (reply.text or "").strip() else "with no visible answer"
        return StopReason.NO_ANSWER, f"the model stopped {visible} and did not call final_answer"
    return None


def _handle_tool_calls[T: BaseModel](
    calls: Sequence[ToolCall],
    tools: ToolClient,
    answer_model: type[T],
    specs: Sequence[ToolSpec],
    limits: Limits,
    trace: TraceHook,
    turn: int,
) -> tuple[list[ToolResult], T | None]:
    """Run every call of one turn. Every call gets exactly one result, in order."""
    results: list[ToolResult] = []
    answer: T | None = None
    for call in calls:
        started = time.monotonic()
        if call.name == FINAL_ANSWER_TOOL:
            result, answer = _check_final_answer(call, answer_model, alone=len(calls) == 1)
        else:
            result = _run_tool(call, tools, limits)
        trace.on_tool_call(ToolCallEvent(turn, time.monotonic() - started, call, result))
        results.append(result)
    return results, answer


def _check_final_answer[T: BaseModel](
    call: ToolCall, answer_model: type[T], *, alone: bool
) -> tuple[ToolResult, T | None]:
    if not alone:
        reason = "final_answer must be the only tool call in its turn; nothing was submitted"
        return error_result(call, reason), None
    try:
        answer = answer_model.model_validate_json(call.arguments)
    except ValidationError as exc:
        return error_result(call, f"invalid final answer: {validation_summary(exc)}"), None
    return ToolResult(tool_call_id=call.id, name=call.name, content="accepted"), answer


def _run_tool(call: ToolCall, tools: ToolClient, limits: Limits) -> ToolResult:
    try:
        result = tools.call(call)
    except Exception as exc:  # a ToolClient should not raise; contain it if one does
        return error_result(call, f"tool call failed ({type(exc).__name__})")
    # The loop owns the call id and name, so a tool client cannot answer for another call.
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=_cap(result.content, limits.max_tool_result_chars),
        is_error=result.is_error,
    )


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[truncated: {len(text) - limit} characters omitted]"
